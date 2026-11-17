"""Score a quantile model on the held-out water years, all days and AR days, against persistence and climatology.

AR days: a qualifying landfalling AR (CW3E catalog, AR scale >= 3) touched the coast within 1.5 degrees of the
dam's latitude (caudal-fleet's definition, `caudal.data.cw3e`). Those are the days the ledger exists for.

Each point is scored at several leads. At lead L the model sees no inflow newer than L + 1 days before the
target day (what an issue L days ahead has in hand), and persistence is that same newest observation. Lead 0
is the day-ahead simulation v1 was first scored on. The weather is the observed one (ERA5) at every lead, so
the scores at leads above 1 leave out the weather forecast's own error.

    python -m caudal.evaluate quantile_inflow_v1_s0      # re-score a saved checkpoint, writes data/results/eval_<run>.csv
"""

from __future__ import annotations

import sys
from functools import partial

import numpy as np
import pandas as pd
import torch

from caudal import frames as F
from caudal.config import MODELS_DIR, RESULTS_DIR
from caudal.data.cw3e import basin_ar_frame
from caudal.models.quantile_lstm import QuantileLSTM, predict_quantiles
from caudal.points import POINTS
from caudal.scores import crps_ensemble
from caudal.training import MaskRecent

LEADS = (0, 1, 7, 14)


def crps_climatology(obs: pd.Series, fit: pd.Series) -> np.ndarray:
    """Per-day CRPS of the per-day-of-year fit-window ensemble."""
    fit = fit.dropna()
    by_doy = fit.groupby(fit.index.day_of_year)
    return np.array([crps_ensemble(by_doy.get_group(d.day_of_year).to_numpy()[None, :], np.array([y]))[0]
                     for d, y in obs.items()])


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    return float(1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2))


def persistence(target: pd.Series, idx: pd.DatetimeIndex, lead: int) -> np.ndarray:
    """The newest inflow a forecast at this lead could see: the observation lead + 1 days before each target day."""
    return target.reindex(idx - pd.Timedelta(days=lead + 1)).to_numpy()


def evaluate(model, frames: dict, lid: str, quantiles, features=F.FEATURES, lead: int = 0) -> dict | None:
    q = {x: i for i, x in enumerate(quantiles)}
    raw, s = frames["raw"][lid], frames["s_target"][lid]
    hist = raw.loc[: F.EVAL_START].iloc[-(F.LOOKBACK + 1) : -1]
    ev = F.prep(pd.concat([hist, raw.loc[F.EVAL_START : F.EVAL_END]]), frames["f_mean"], frames["f_std"], s)
    hide = None  # a model without the inflow inputs (v1) has nothing to hide
    if lead and F.OBS[0] in features:
        hide = partial(MaskRecent, value=features.index(F.OBS[0]), flag=features.index(F.OBS[1]), days=lead)
    preds, obs_s, idx = predict_quantiles(model, ev, features, F.TARGET, F.LOOKBACK, wrap=hide)
    if len(idx) == 0:
        return None
    preds, obs = np.clip(preds * s, 0, None), obs_s * s
    crps = {"model": crps_ensemble(preds, obs), "persistence": np.abs(persistence(raw[F.TARGET], idx, lead) - obs),
            "climatology": crps_climatology(pd.Series(obs, index=idx), raw.loc[: F.FIT_END, F.TARGET])}
    ar = basin_ar_frame(POINTS[lid][2], F.EVAL_START, F.EVAL_END)["is_ar_day"].reindex(idx).fillna(False).to_numpy()
    row = {"lead": lead, "lid": lid, "n_days": len(idx), "n_ar_days": int(ar.sum())}
    for name, v in crps.items():
        row[f"crps_{name}"] = float(np.nanmean(v))
        row[f"crps_{name}_ar"] = float(np.nanmean(v[ar])) if ar.any() else np.nan
    row |= {"cov90": float(np.mean((preds[:, q[0.05]] <= obs) & (obs <= preds[:, q[0.95]]))),
            "nse_median": nse(preds[:, q[0.50]], obs),
            "rel_spread": float((preds[:, -1] - preds[:, 0]).mean() / max(preds[:, q[0.50]].mean(), 1e-6))}
    return row


def evaluate_all(model, frames: dict, quantiles, features=F.FEATURES, leads=LEADS) -> pd.DataFrame:
    """One row per lead and point, indexed (lead, lid)."""
    rows = [r for lead in leads for lid in frames["train"] if (r := evaluate(model, frames, lid, quantiles, features, lead))]
    return pd.DataFrame(rows).set_index(["lead", "lid"])


def load_model(run: str) -> tuple[QuantileLSTM, dict]:
    ckpt = torch.load(MODELS_DIR / f"{run}.pt", map_location="cpu", weights_only=False)
    model = QuantileLSTM(n_features=len(ckpt["features"]), hidden_size=ckpt["hidden"], quantiles=ckpt["quantiles"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def show(table: pd.DataFrame) -> pd.DataFrame:
    """Print the table and the mean over points at each lead; return that mean, indexed by lead."""
    pooled = table.drop(columns=["n_days", "n_ar_days"]).groupby("lead").mean()
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(), "\npooled over points, by lead:", pooled.round(3).to_string(), sep="\n", flush=True)
    return pooled


if __name__ == "__main__":
    run = sys.argv[1]
    model, ckpt = load_model(run)
    frames = F.build_regional(ckpt["points"], ckpt["fit_end"])
    table = evaluate_all(model, frames, ckpt["quantiles"], ckpt["features"])
    show(table)
    table.to_csv(RESULTS_DIR / f"eval_{run}.csv")
    print(f"wrote {RESULTS_DIR / f'eval_{run}.csv'}")
