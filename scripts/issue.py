"""Issue today's quantile inflow forecasts, leads 1-14 days, into the inflow-ledger data directory.

For each point: extend the ERA5 archive, bridge from its last day to today with the forecast model's analysis
days, append the 16-day forecast, run the regional quantile LSTM, and write
<ledger>/forecasts/<model>/<lid>/<issue date>.csv with one row per day and the predicted quantiles in cfs.

Usage
-----
    python scripts/issue.py                              # model quantile_inflow_v1_s0, ledger next to this repo
    python scripts/issue.py --model quantile_inflow_v1_s0_smoke --points LAMC1 --out /tmp/x
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import torch

from embalse import forcing
from embalse import frames as F
from embalse.config import LEDGER_DATA, MODELS_DIR
from embalse.models.quantile_lstm import QuantileLSTM, predict_quantiles

LEADS = 14


def issue_point(model, ckpt: dict, lid: str, today: date) -> pd.DataFrame:
    hist = forcing.update_archive(lid, today)
    fc = forcing.fetch_forecast(lid)
    end = pd.Timestamp(today) + pd.Timedelta(days=LEADS)
    combined = pd.concat([hist, fc[fc.index > hist.index[-1]]]).loc[:end]
    frame = F.build_frame(lid, ckpt["fit_end"], f=combined)
    z = F.prep(frame, np.array(ckpt["f_mean"]), np.array(ckpt["f_std"]), ckpt["s_target"][lid])
    z[F.TARGET] = z[F.TARGET].fillna(0.0)  # windows are built only on days with a target; the value is unused here
    preds, _, idx = predict_quantiles(model, z, ckpt["features"], F.TARGET, ckpt["lookback"])
    out = pd.DataFrame(np.clip(preds * ckpt["s_target"][lid], 0, None).round(1), index=idx,
                       columns=[f"q{int(round(q * 100)):02d}" for q in ckpt["quantiles"]])
    out = out[out.index > pd.Timestamp(today)]
    out.insert(0, "lead", (out.index - pd.Timestamp(today)).days)
    out.index.name = "date"
    if len(out) != LEADS:
        raise RuntimeError(f"{lid}: {len(out)} forecast days, expected {LEADS} (forcing through {combined.index[-1]:%Y-%m-%d})")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="quantile_inflow_v1_s0")
    p.add_argument("--points", default="", help="comma list; default: the model's points")
    p.add_argument("--out", default=str(LEDGER_DATA), help="inflow-ledger data directory")
    p.add_argument("--date", default=None, help="issue date, default today UTC")
    a = p.parse_args(argv)
    today = date.fromisoformat(a.date) if a.date else datetime.now(timezone.utc).date()
    ckpt = torch.load(MODELS_DIR / f"{a.model}.pt", map_location="cpu", weights_only=False)
    model = QuantileLSTM(n_features=len(ckpt["features"]), hidden_size=ckpt["hidden"], quantiles=ckpt["quantiles"])
    model.load_state_dict(ckpt["state_dict"])
    points = [x for x in a.points.split(",") if x] or ckpt["points"]
    failed = 0
    for lid in points:
        try:
            out = issue_point(model, ckpt, lid, today)
        except Exception as e:  # one point's API hiccup must not lose the others' forecasts
            failed += 1
            print(f"FAILED {lid}: {e}", file=sys.stderr)
            continue
        path = pd.io.common.Path(a.out) / "forecasts" / a.model / lid / f"{today:%Y-%m-%d}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path)
        print(f"{lid} {today}: q50 lead1 {out.q50.iloc[0]:,.0f} cfs, lead14 {out.q50.iloc[-1]:,.0f} cfs -> {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
