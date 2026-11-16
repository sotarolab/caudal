"""Score a quantile model on the held-out water years, all days and AR days, against persistence and climatology.

AR days: a qualifying landfalling AR (CW3E catalog, AR scale >= 3) touched the coast within 1.5 degrees of the
dam's latitude (caudal-fleet's definition, `caudal.data.cw3e`). Those are the days the ledger exists for.

    python -m caudal.evaluate quantile_inflow_v1_s0      # re-score a saved checkpoint, writes data/results/eval_<run>.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from caudal import frames as F
from caudal.config import MODELS_DIR, RESULTS_DIR
from caudal.data.cw3e import basin_ar_frame
from caudal.models.quantile_lstm import QuantileLSTM, predict_quantiles
from caudal.points import POINTS
from caudal.scores import crps_ensemble


def crps_climatology(obs: pd.Series, fit: pd.Series) -> np.ndarray:
    """Per-day CRPS of the per-day-of-year fit-window ensemble."""
    fit = fit.dropna()
    by_doy = fit.groupby(fit.index.day_of_year)
    return np.array([crps_ensemble(by_doy.get_group(d.day_of_year).to_numpy()[None, :], np.array([y]))[0]
                     for d, y in obs.items()])


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    return float(1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2))


def evaluate(model, frames: dict, lid: str, quantiles) -> dict | None:
    q = {x: i for i, x in enumerate(quantiles)}
    raw, s = frames["raw"][lid], frames["s_target"][lid]
    hist = raw.loc[: F.EVAL_START].iloc[-(F.LOOKBACK + 1) : -1]
    ev = F.prep(pd.concat([hist, raw.loc[F.EVAL_START : F.EVAL_END]]), frames["f_mean"], frames["f_std"], s)
    preds, obs_s, idx = predict_quantiles(model, ev, F.FEATURES, F.TARGET, F.LOOKBACK)
    if len(idx) == 0:
        return None
    preds, obs = np.clip(preds * s, 0, None), obs_s * s
    yesterday = raw[F.TARGET].reindex(idx - pd.Timedelta(days=1)).to_numpy()
    crps = {"model": crps_ensemble(preds, obs), "persistence": np.abs(yesterday - obs),
            "climatology": crps_climatology(pd.Series(obs, index=idx), raw.loc[: F.FIT_END, F.TARGET])}
    ar = basin_ar_frame(POINTS[lid][2], F.EVAL_START, F.EVAL_END)["is_ar_day"].reindex(idx).fillna(False).to_numpy()
    row = {"lid": lid, "n_days": len(idx), "n_ar_days": int(ar.sum())}
    for name, v in crps.items():
        row[f"crps_{name}"] = float(np.nanmean(v))
        row[f"crps_{name}_ar"] = float(np.nanmean(v[ar])) if ar.any() else np.nan
    row |= {"cov90": float(np.mean((preds[:, q[0.05]] <= obs) & (obs <= preds[:, q[0.95]]))),
            "nse_median": nse(preds[:, q[0.50]], obs),
            "rel_spread": float((preds[:, -1] - preds[:, 0]).mean() / max(preds[:, q[0.50]].mean(), 1e-6))}
    return row


def evaluate_all(model, frames: dict, quantiles) -> pd.DataFrame:
    rows = [r for lid in frames["train"] if (r := evaluate(model, frames, lid, quantiles))]
    return pd.DataFrame(rows).set_index("lid")


def load_model(run: str) -> tuple[QuantileLSTM, dict]:
    ckpt = torch.load(MODELS_DIR / f"{run}.pt", map_location="cpu", weights_only=False)
    model = QuantileLSTM(n_features=len(ckpt["features"]), hidden_size=ckpt["hidden"], quantiles=ckpt["quantiles"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def show(table: pd.DataFrame) -> pd.DataFrame:
    pooled = table.drop(columns=["n_days", "n_ar_days"]).mean().rename("pooled")
    pd.set_option("display.width", 200)
    print(pd.concat([table, pooled.to_frame().T]).round(3).to_string(), flush=True)
    return pooled


if __name__ == "__main__":
    run = sys.argv[1]
    model, ckpt = load_model(run)
    frames = F.build_regional(ckpt["points"], ckpt["fit_end"])
    table = evaluate_all(model, frames, ckpt["quantiles"])
    show(table)
    table.to_csv(RESULTS_DIR / f"eval_{run}.csv")
    print(f"wrote {RESULTS_DIR / f'eval_{run}.csv'}")
