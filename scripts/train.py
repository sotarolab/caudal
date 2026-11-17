"""Train one regional quantile LSTM on the reservoir inflow points and score it on the held-out water years.

v2 reads the observed inflow of the window's days beside the weather (`frames.OBS`). In training the newest
0 to `frames.MAX_HIDDEN` days of it are hidden at random, so the one model issues every lead.

Protocol (caudal-fleet's regional arm, applied to inflow): fit through FIT_END with nothing held out, evaluate
EVAL_START..EVAL_END, pinball loss over DEFAULT_QUANTILES, lr 1e-3, 30 epochs, lookback 365, hidden 128.
Scoring (all days and AR days, persistence and climatology references) is `caudal.evaluate`.

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

import torch
from torch.utils.data import DataLoader

from caudal import frames as F
from caudal.config import MODELS_DIR, RESULTS_DIR
from caudal.evaluate import evaluate_all, show
from caudal.models.quantile_lstm import DEFAULT_QUANTILES, QuantileLSTM, train_quantile_epochs
from caudal.models.sequence import SequenceDataset, get_device
from caudal.points import TRAINED
from caudal.training import MaskRecent, MetricsLogger, ProgressLogger, RunArtifacts

VERSION = "v2"
FEATURES = F.FEATURES_V2
EVAL_LEAD = 1  # the next day: the lead whose pooled scores the manifest records and the collapse check reads
LR = 1e-3
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
    ds = MaskRecent(SequenceDataset(list(frames["train"].values()), FEATURES, F.TARGET, F.LOOKBACK),
                    FEATURES, F.MAX_HIDDEN, random=True)
    loader = DataLoader(ds, batch_size=F.BATCH_SIZE, shuffle=True)
    model = QuantileLSTM(n_features=len(FEATURES), hidden_size=F.HIDDEN)
    config = {"version": VERSION, "points": list(frames["train"]), "features": FEATURES, "max_hidden": F.MAX_HIDDEN,
              "quantiles": list(DEFAULT_QUANTILES),
              "lookback": F.LOOKBACK, "hidden": F.HIDDEN, "batch_size": F.BATCH_SIZE, "epochs": a.epochs, "lr": LR,
              "seed": a.seed, "fit_end": F.FIT_END, "eval": [F.EVAL_START, F.EVAL_END], "n_windows": len(ds)}
    art = RunArtifacts(run, config=config)
    print(f"[{time.strftime('%H:%M:%S')}] {run}: {len(frames['train'])} points, {len(ds):,} windows, "
          f"{a.epochs} epochs on {get_device()}", flush=True)
    t0 = time.time()
    losses = train_quantile_epochs(model, loader, get_device(), a.epochs, LR,
                                   callbacks=[MetricsLogger(art), ProgressLogger(frequency=5, prefix=f"{run} ")])
    print(f"[{time.strftime('%H:%M:%S')}] pinball {losses[0]:.4f} -> {losses[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)

    table = evaluate_all(model, frames, DEFAULT_QUANTILES, FEATURES)
    pooled = show(table).loc[EVAL_LEAD]
    if not pooled.rel_spread >= 0.05 and not a.smoke:  # also catches NaN
        art.update_manifest(status="collapsed", eval_lead=EVAL_LEAD, **pooled.to_dict())
        print("quantile heads COLLAPSED (see quantile_lstm docstring on lr); refusing to save the model", file=sys.stderr)
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **config, "f_mean": frames["f_mean"].tolist(),
                "f_std": frames["f_std"].tolist(), "s_target": frames["s_target"]}, MODELS_DIR / f"{run}.pt")
    table.to_csv(RESULTS_DIR / f"eval_{run}.csv")
    art.update_manifest(status="finished", seconds=round(time.time() - t0), eval_lead=EVAL_LEAD, **pooled.to_dict())
    print(f"saved {MODELS_DIR / f'{run}.pt'} and {RESULTS_DIR / f'eval_{run}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
