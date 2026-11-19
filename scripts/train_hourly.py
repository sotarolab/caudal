"""Train the hourly flow model for the Russian River gauges and score it on the held-out water years.

One regional quantile LSTM over the three gauges: fit through water year 2015, the newest 0 to MAX_HIDDEN hours
of every flow input hidden at random, the target taken as its change from the newest flow left, pinball loss,
lr 1e-3. An epoch is a random draw of --samples windows.
Needs the archives: `python -m caudal.sources.usgs <sites>` and `python -m caudal.sources.openmeteo`.

    python scripts/train_hourly.py --seed 0      # writes data/models/<run>.pt and data/results/eval_<run>.csv
    python scripts/train_hourly.py --smoke       # one short epoch and two leads, to check the path
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from torch.utils.data import DataLoader, RandomSampler

from caudal import flow_hourly as H
from caudal.config import MODELS_DIR, RESULTS_DIR
from caudal.models.quantile_lstm import DEFAULT_QUANTILES, QuantileLSTM, train_quantile_epochs
from caudal.models.sequence import SequenceDataset, get_device
from caudal.points import GAUGES
from caudal.training import Anchor, MaskRecent, MetricsLogger, ProgressLogger, RunArtifacts

LR = 1e-3


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--samples", type=int, default=250_000, help="windows drawn per epoch")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--config", default="weighted_log", choices=list(H.CONFIGS), help="what the model reads")
    p.add_argument("--name", default=None, help="run name; default: the product name")
    p.add_argument("--fit-end", default=H.FIT_END)
    p.add_argument("--eval", nargs=2, default=[H.EVAL_START, H.EVAL_END], metavar=("START", "END"))
    a = p.parse_args(argv)
    H.configure(H.CONFIGS[a.config])
    H.set_years(a.fit_end, *a.eval)
    leads = H.EVAL_LEADS
    if a.smoke:
        a.epochs, a.samples, leads = 1, 20_000, (1, 24)
    run = (a.name or f"{H.PRODUCT}_s{a.seed}") + ("_smoke" if a.smoke else "")

    frames = H.build_regional(list(GAUGES))
    torch.manual_seed(a.seed)
    weight = "weight" if H.WEIGHTED else None
    ds = MaskRecent(SequenceDataset(list(frames["train"].values()), H.FEATURES, H.TARGET, H.LOOKBACK, weight),
                    H.FEATURES, H.MAX_HIDDEN, random=True, pairs=H.PAIRS)
    ds = Anchor(ds, H.FEATURES, H.PAIRS[0])  # the target is the change from the newest flow left in the window
    loader = DataLoader(ds, batch_size=H.BATCH_SIZE, sampler=RandomSampler(ds, replacement=True, num_samples=a.samples))
    model = QuantileLSTM(n_features=len(H.FEATURES), hidden_size=H.HIDDEN)
    config = {"product": H.PRODUCT, "config": H.CONFIGS[a.config], "config_name": a.config, "points": list(frames["train"]), "features": H.FEATURES, "pairs": H.PAIRS,
              "max_hidden": H.MAX_HIDDEN, "leads": H.LEADS, "quantiles": list(DEFAULT_QUANTILES), "lookback": H.LOOKBACK,
              "hidden": H.HIDDEN, "batch_size": H.BATCH_SIZE, "epochs": a.epochs, "samples": a.samples, "lr": LR,
              "seed": a.seed, "fit_end": H.FIT_END, "eval": [H.EVAL_START, H.EVAL_END], "n_windows": len(ds)}
    art = RunArtifacts(run, config=config)
    print(f"[{time.strftime('%H:%M:%S')}] {run}: {len(frames['train'])} gauges, {len(ds):,} windows, "
          f"{a.epochs} epochs of {a.samples:,} on {get_device()}", flush=True)
    t0 = time.time()
    losses = train_quantile_epochs(model, loader, get_device(), a.epochs, LR,
                                   callbacks=[MetricsLogger(art), ProgressLogger(frequency=5, prefix=f"{run} ")])
    print(f"[{time.strftime('%H:%M:%S')}] pinball {losses[0]:.4f} -> {losses[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)

    table = H.evaluate_all(model, frames, leads)
    pooled = table.groupby("lead").mean(numeric_only=True)
    with_skill = pooled.assign(skill=1 - pooled.crps_model / pooled.crps_persistence,
                               skill_high=1 - pooled.crps_model_high / pooled.crps_persistence_high)
    print(table.round(3).to_string(), "\n\nmean over the gauges, by lead hour:\n", with_skill.round(3).to_string(), sep="")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **config, "f_mean": frames["f_mean"].tolist(),
                "f_std": frames["f_std"].tolist(), "scale": frames["scale"], "qmean": frames["qmean"]},
               MODELS_DIR / f"{run}.pt")
    table.to_csv(RESULTS_DIR / f"eval_{run}.csv")
    art.update_manifest(status="finished", seconds=round(time.time() - t0), eval_lead=24, **pooled.loc[24].to_dict())
    print(f"saved {MODELS_DIR / f'{run}.pt'} and {RESULTS_DIR / f'eval_{run}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
