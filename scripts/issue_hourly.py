"""Issue the hourly flow forecasts of the Russian River gauges, leads 1 to 120 hours, as files.

For each gauge: the last 45 days and the next 7 of hourly weather at its basin's rainfall points from the
forecast model, the last 12 days of observed flow at the gauge and upstream from USGS, the regional quantile
LSTM, and <out>/forecasts/<model>/<gauge>/<issue hour>Z.csv with one row per hour and the quantiles in cfs.
Only the checkpoint is read from disk. A feed that does not answer in a minute is given up on and named.

    python scripts/issue_hourly.py                                   # into ../hapi/data (or HAPI_DATA)
    python scripts/issue_hourly.py --points GUEC1 --out /tmp/x
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from functools import partial
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
recent_flow = partial(usgs.fetch, timeout=60, wait=5)  # an issue has minutes: three tries of 60 seconds


def feed(name: str, call):
    """The call's answer; a failure is raised again naming the feed, so the job's log says what was down."""
    try:
        return call()
    except Exception as e:
        raise RuntimeError(f"{name}: {e}") from e


# The product issued: the quantile average of three models that differ in what they read and how they weigh
# errors, chosen in caudal #27 on water years 2020 to 2025 (scripts/compare_hourly.py).
BLEND = {"quantile_flow_hourly_v2_s0": ["quantile_flow_hourly_v2_v1", "quantile_flow_hourly_v2_weighted_log",
                                         "quantile_flow_hourly_v2_full"]}


def load(name: str):
    ckpt = torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
    model = QuantileLSTM(n_features=len(ckpt["features"]), hidden_size=ckpt["hidden"], quantiles=ckpt["quantiles"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=f"{H.PRODUCT}_s0", help="a checkpoint, or a product in BLEND")
    p.add_argument("--points", default="", help="comma list; default: the model's gauges")
    p.add_argument("--out", default=str(LEDGER_DATA), help="hapi data directory")
    p.add_argument("--hour", default=None, help="issue hour in UTC, e.g. 2026-11-18T06; default: the hour now")
    a = p.parse_args(argv)
    hour = pd.Timestamp(a.hour) if a.hour else pd.Timestamp(datetime.now(UTC).replace(tzinfo=None)).floor("h")
    members = [load(n) for n in BLEND.get(a.model, [a.model])]
    points = [x for x in a.points.split(",") if x] or members[0][1]["points"]
    weather, flows, failed = {}, {}, 0
    for lid in points:
        path = Path(a.out) / "forecasts" / a.model / lid / f"{hour:%Y-%m-%dT%H}Z.csv"
        if path.exists():  # issued once, never revised: a second run in the hour leaves the first issue alone
            print(f"{lid} {hour:%Y-%m-%d %H}Z: already issued, left as it is", flush=True)
            continue
        try:
            for pt in RAIN_POINTS[lid]:
                if pt not in weather:
                    weather[pt] = feed(f"Open-Meteo weather at {pt}", lambda: openmeteo.forecast(*pt, past_days=PAST_DAYS, days=7))
            basin = openmeteo.basin_mean([weather[pt] for pt in RAIN_POINTS[lid]])
            outs = []
            for model, ckpt in members:
                H.configure(ckpt.get("config", H.V1))  # what this model reads; a checkpoint without one is v1
                for site in H.sites(lid):
                    if site not in flows:
                        first = (hour - timedelta(days=FLOW_DAYS)).date()
                        flows[site] = feed(f"USGS flow at {site}", lambda: usgs.complete_hours(site, first, hour.date(), recent_flow))
                outs.append(H.issue_point(model, ckpt, lid, hour, basin, [flows[s] for s in H.sites(lid)]))
            out = outs[0].copy()
            qs = [c for c in out.columns if c.startswith("q")]
            out[qs] = (sum(o[qs] for o in outs) / len(outs)).round(1)  # quantile averaging; still non-crossing
        except Exception as e:  # one gauge's feed must not lose the others' forecasts
            failed += 1
            print(f"FAILED {lid}: {e}", file=sys.stderr, flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, date_format="%Y-%m-%dT%H:%MZ")
        newest = flows[H.GAUGES[lid][0]]
        print(f"{lid} {hour:%Y-%m-%d %H}Z: last observed {newest.iloc[-1]:,.0f} cfs at {newest.index[-1]:%H}Z, median "
              f"+1 h {out.q50.iloc[0]:,.0f}, +24 h {out.q50.iloc[23]:,.0f}, +120 h {out.q50.iloc[-1]:,.0f}, "
              f"peak {out.q50.max():,.0f} at +{int(out.lead[out.q50.idxmax()])} h -> {path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
