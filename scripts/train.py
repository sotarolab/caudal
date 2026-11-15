"""Train one regional quantile LSTM on the reservoir inflow points and score it on the held-out water years.

Protocol (caudal's regional arm, applied to inflow): fit through FIT_END with nothing held out, evaluate
EVAL_START..EVAL_END, pinball loss over DEFAULT_QUANTILES, lr 1e-3, 30 epochs, lookback 365, hidden 128.
References on the same days: persistence (yesterday's inflow) and per-day-of-year climatology from the fit window.

Usage
-----
    python scripts/train.py --smoke                 # 2 points, 1 epoch, nothing overwritten
    python scripts/train.py --seed 0 --epochs 30    # ~10 min on MPS

Writes data/results/runs/<run>/ (config, metrics.csv, manifest), data/models/<run>.pt and data/results/eval_<run>.csv.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from embalse import frames as F
from embalse.config import MODELS_DIR, RESULTS_DIR
from embalse.models.quantile_lstm import DEFAULT_QUANTILES, QuantileLSTM, predict_quantiles, train_quantile_epochs
from embalse.models.sequence import SequenceDataset, get_device
from embalse.points import TRAINED
from embalse.scores import crps_ensemble
from embalse.training import MetricsLogger, ProgressLogger, RunArtifacts

VERSION = "v1"
LR = 1e-3
Q = {q: i for i, q in enumerate(DEFAULT_QUANTILES)}


def crps_climatology(obs: pd.Series, fit: pd.Series) -> float:
    """Mean CRPS of the per-day-of-year fit-window ensemble over the observed days."""
    by_doy = fit.dropna().groupby(fit.dropna().index.day_of_year)
    return float(np.mean([crps_ensemble(by_doy.get_group(d.day_of_year).to_numpy()[None, :], np.array([y]))[0]
                          for d, y in obs.items()]))


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    return float(1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2))


def evaluate(model, frames: dict, lid: str) -> dict | None:
    raw, s = frames["raw"][lid], frames["s_target"][lid]
    hist = raw.loc[: F.EVAL_START].iloc[-(F.LOOKBACK + 1) : -1]
    ev = F.prep(pd.concat([hist, raw.loc[F.EVAL_START : F.EVAL_END]]), frames["f_mean"], frames["f_std"], s)
    preds, obs_s, idx = predict_quantiles(model, ev, F.FEATURES, F.TARGET, F.LOOKBACK)
    if len(idx) == 0:
        return None
    preds, obs = np.clip(preds * s, 0, None), obs_s * s
    yesterday = raw[F.TARGET].reindex(idx - pd.Timedelta(days=1)).to_numpy()
    ok = np.isfinite(yesterday)
    return {
        "lid": lid, "n_days": len(idx),
        "crps_model": float(crps_ensemble(preds, obs).mean()),
        "crps_persistence": float(np.abs(yesterday[ok] - obs[ok]).mean()),
        "crps_climatology": crps_climatology(pd.Series(obs, index=idx), raw.loc[: F.FIT_END, F.TARGET]),
        "cov90": float(np.mean((preds[:, Q[0.05]] <= obs) & (obs <= preds[:, Q[0.95]]))),
        "nse_median": nse(preds[:, Q[0.50]], obs),
        "rel_spread": float((preds[:, -1] - preds[:, 0]).mean() / max(preds[:, Q[0.50]].mean(), 1e-6)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--points", default="", help="comma list of CNRFC ids; default: every trained point")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    points = [x for x in a.points.split(",") if x] or TRAINED
    if a.smoke:
        points, a.epochs = points[:2], 1
    run = f"quantile_inflow_{VERSION}_s{a.seed}" + ("_smoke" if a.smoke else "")

    frames = F.build_regional(points)
    torch.manual_seed(a.seed)
    ds = SequenceDataset(list(frames["train"].values()), F.FEATURES, F.TARGET, F.LOOKBACK)
    loader = DataLoader(ds, batch_size=F.BATCH_SIZE, shuffle=True)
    model = QuantileLSTM(n_features=len(F.FEATURES), hidden_size=F.HIDDEN)
    config = {"version": VERSION, "points": list(frames["train"]), "features": F.FEATURES, "quantiles": list(DEFAULT_QUANTILES),
              "lookback": F.LOOKBACK, "hidden": F.HIDDEN, "batch_size": F.BATCH_SIZE, "epochs": a.epochs, "lr": LR,
              "seed": a.seed, "fit_end": F.FIT_END, "eval": [F.EVAL_START, F.EVAL_END], "n_windows": len(ds)}
    art = RunArtifacts(run, config=config)
    print(f"[{time.strftime('%H:%M:%S')}] {run}: {len(frames['train'])} points, {len(ds):,} windows, "
          f"{a.epochs} epochs on {get_device()}", flush=True)
    t0 = time.time()
    losses = train_quantile_epochs(model, loader, get_device(), a.epochs, LR,
                                   callbacks=[MetricsLogger(art), ProgressLogger(frequency=5, prefix=f"{run} ")])
    print(f"[{time.strftime('%H:%M:%S')}] pinball {losses[0]:.4f} -> {losses[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)

    rows = [r for lid in frames["train"] if (r := evaluate(model, frames, lid))]
    table = pd.DataFrame(rows).set_index("lid")
    pooled = table.drop(columns="n_days").mean().rename("pooled")
    pd.set_option("display.width", 160)
    print(pd.concat([table, pooled.to_frame().T]).round(3).to_string(), flush=True)
    if not pooled.rel_spread >= 0.05 and not a.smoke:  # also catches NaN
        art.update_manifest(status="collapsed", **pooled.to_dict())
        print("quantile heads COLLAPSED (see quantile_lstm docstring on lr); refusing to save the model", file=sys.stderr)
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **config, "f_mean": frames["f_mean"].tolist(),
                "f_std": frames["f_std"].tolist(), "s_target": frames["s_target"]}, MODELS_DIR / f"{run}.pt")
    table.to_csv(RESULTS_DIR / f"eval_{run}.csv")
    art.update_manifest(status="finished", seconds=round(time.time() - t0), **pooled.to_dict())
    print(f"saved {MODELS_DIR / f'{run}.pt'} and {RESULTS_DIR / f'eval_{run}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
