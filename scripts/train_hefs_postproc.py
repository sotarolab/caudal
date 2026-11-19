"""Train the HEFS post-processor (caudal.hefs_postproc) on the Russian River gauges and save it.

    python scripts/train_hefs_postproc.py --fit-end 2019-09-30 --name cand_hefspp
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from caudal import hefs_postproc as P
from caudal.config import MODELS_DIR
from caudal.sources.hefs import GAUGES


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit-end", default="2019-09-30 23:00")
    p.add_argument("--name", default=f"{P.PRODUCT}_s0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--draws", type=int, default=4, help="random gaps per issue hour")
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    xs, ys, ws = [], [], []
    for lid in GAUGES:
        src = P.gauge_inputs(lid)
        for _ in range(a.draws):
            x, y, w, *_ = P.samples(lid, src, P.FIT_START, a.fit_end, rng=rng)
            xs.append(x), ys.append(y), ws.append(w)
        print(f"{lid}: {len(x):,} samples a draw", flush=True)
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    t0 = time.time()
    model = P.train(x, y, w, epochs=a.epochs, seed=a.seed)
    print(f"{len(x):,} samples, {x.shape[1]} inputs, trained in {time.time() - t0:.0f}s", flush=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_in": x.shape[1], "quantiles": list(P.QUANTILES),
                "fit_end": a.fit_end, "product": P.PRODUCT}, MODELS_DIR / f"{a.name}.pt")
    print(f"saved {MODELS_DIR / f'{a.name}.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
