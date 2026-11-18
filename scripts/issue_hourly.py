"""Issue the hourly flow forecasts of the Russian River gauges, leads 1 to 120 hours, as files.

For each gauge: the last 45 days and the next 7 of hourly weather at its basin's rainfall points from the
forecast model, the last 12 days of observed flow at the gauge and upstream from USGS, the regional quantile
LSTM, and <out>/forecasts/<model>/<gauge>/<issue hour>Z.csv with one row per hour and the quantiles in cfs.
Only the checkpoint is read from disk.

    python scripts/issue_hourly.py                                   # into ../hapi/data (or HAPI_DATA)
    python scripts/issue_hourly.py --points GUEC1 --out /tmp/x
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import torch

from caudal import flow_hourly as H
from caudal.config import LEDGER_DATA, MODELS_DIR
from caudal.models.quantile_lstm import QuantileLSTM
from caudal.points import RAIN_POINTS
from caudal.sources import openmeteo, usgs

PAST_DAYS = 45  # the 30-day rainfall total, the window, and a margin
FLOW_DAYS = 12  # the window is 10 days


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=f"{H.PRODUCT}_s0")
    p.add_argument("--points", default="", help="comma list; default: the model's gauges")
    p.add_argument("--out", default=str(LEDGER_DATA), help="hapi data directory")
    p.add_argument("--hour", default=None, help="issue hour in UTC, e.g. 2026-11-18T06; default: the hour now")
    a = p.parse_args(argv)
    hour = pd.Timestamp(a.hour) if a.hour else pd.Timestamp(datetime.now(UTC).replace(tzinfo=None)).floor("h")
    ckpt = torch.load(MODELS_DIR / f"{a.model}.pt", map_location="cpu", weights_only=False)
    model = QuantileLSTM(n_features=len(ckpt["features"]), hidden_size=ckpt["hidden"], quantiles=ckpt["quantiles"])
    model.load_state_dict(ckpt["state_dict"])
    points = [x for x in a.points.split(",") if x] or ckpt["points"]
    weather, flows, failed = {}, {}, 0
    for lid in points:
        try:
            for pt in RAIN_POINTS[lid]:
                if pt not in weather:
                    weather[pt] = openmeteo.forecast(*pt, past_days=PAST_DAYS, days=7)
            for site in H.sites(lid):
                if site not in flows:
                    flows[site] = usgs.complete_hours(site, (hour - timedelta(days=FLOW_DAYS)).date(), hour.date())
            out = H.issue_point(model, ckpt, lid, hour, openmeteo.basin_mean([weather[pt] for pt in RAIN_POINTS[lid]]),
                                [flows[s] for s in H.sites(lid)])
        except Exception as e:  # one gauge's feed must not lose the others' forecasts
            failed += 1
            print(f"FAILED {lid}: {e}", file=sys.stderr)
            continue
        path = Path(a.out) / "forecasts" / a.model / lid / f"{hour:%Y-%m-%dT%H}Z.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, date_format="%Y-%m-%dT%H:%MZ")
        newest = flows[H.sites(lid)[0]]
        print(f"{lid} {hour:%Y-%m-%d %H}Z: last observed {newest.iloc[-1]:,.0f} cfs at {newest.index[-1]:%H}Z, median "
              f"+1 h {out.q50.iloc[0]:,.0f}, +24 h {out.q50.iloc[23]:,.0f}, +120 h {out.q50.iloc[-1]:,.0f}, "
              f"peak {out.q50.max():,.0f} at +{int(out.lead[out.q50.idxmax()])} h -> {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
