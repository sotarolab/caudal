"""Past hourly HEFS issues at the Russian River gauges, from CNRFC's dated archive, as quantiles in cfs.

CNRFC keeps every 12Z hourly issue since January 2014 as one zip per river group,
`<YYYYMMDD>12_RussianNapa_hefs_csv_hourly.zip`: a header row of point ids (one column per member), a units
row, then one row per hour in kcfs, -999 where a member has no value. Only the start of each zip is read.

An issue starts at 12Z and reached hapi's archive 5 to 6.5 hours later in November 2026, so an issue counts as
known from PUBLIC hours after its start. Live, an 18Z issue is also published; the archive keeps only the 12Z one.

    python -m caudal.sources.hefs 2014-01-01 2026-11-18     # builds data/hefs_hourly/<gauge>.csv.gz
"""

from __future__ import annotations

import http.client
import struct
import sys
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

from caudal.config import HEFS_DIR

ARCHIVE = "https://www.cnrfc.noaa.gov/csv/{day:%Y%m%d}12_RussianNapa_hefs_csv_hourly.zip"
GAUGES = ["HOPC1", "HEAC1", "GUEC1"]
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
COLS = [f"hefs_q{int(q * 100):02d}" for q in QUANTILES]
HOURS = 160  # rows after the start: the latest an issue is read (start + 30 h), 120 h of leads, and a margin
PUBLIC = 7  # hours from an issue's start to when it may be used


def head(stream, lines: int) -> list[str] | None:
    """The first lines of the file inside a zip, read from the start of the stream and no further than needed."""
    h = stream.read(30)
    if h[:4] != b"PK\x03\x04" or h[8:10] != b"\x08\x00":  # CNRFC answers a missing file with an HTML page
        return None
    name, extra = struct.unpack("<HH", h[26:30])
    stream.read(name + extra)
    inflate, text = zlib.decompressobj(-15), b""
    while text.count(b"\n") < lines and (chunk := stream.read(32768)):
        text += inflate.decompress(chunk)
    return text.decode().splitlines()[:lines] if text.count(b"\n") >= lines else None


def parse(rows: list[str], lid: str) -> pd.DataFrame:
    """One issue's hours at a gauge as quantiles of its members in cfs, on a UTC index; -999 left out."""
    cells = [r.split(",") for r in rows]
    cols = [i for i, c in enumerate(cells[0]) if c == lid]
    out = {}
    for r in cells[2:]:
        m = np.array([float(r[i]) for i in cols if r[i] and float(r[i]) != -999]) * 1000  # kcfs -> cfs
        if len(m):
            out[pd.Timestamp(r[0])] = np.quantile(m, QUANTILES)
    return pd.DataFrame.from_dict(out, orient="index", columns=COLS).rename_axis("time")


def fetch(day: date) -> list[str] | None:
    req = urllib.request.Request(ARCHIVE.format(day=day), headers={"User-Agent": "caudal"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return head(r, 2 + HOURS)
        except (OSError, http.client.HTTPException, zlib.error):
            continue
    return None


def build(first: date, last: date) -> dict[str, pd.DataFrame]:
    """Every issue from `first` to `last`, one frame per gauge indexed by (issue, time), saved under HEFS_DIR."""
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    per: dict[str, list[pd.DataFrame]] = {g: [] for g in GAUGES}
    with ThreadPoolExecutor(4) as pool:
        for day, rows in zip(days, pool.map(fetch, days)):
            if rows is None or not rows[0].startswith("GMT") or not rows[2].startswith(f"{day} 12"):
                print(f"missing {day}", file=sys.stderr, flush=True)
                continue
            for g in GAUGES:
                f = parse(rows, g)
                f.insert(0, "issue", pd.Timestamp(f"{day} 12:00"))
                per[g].append(f)
    HEFS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for g, frames in per.items():
        out[g] = pd.concat(frames).reset_index().set_index(["issue", "time"]).round(1)
        out[g].to_csv(HEFS_DIR / f"{g}.csv.gz")
    return out


def load(lid: str) -> pd.DataFrame:
    return pd.read_csv(HEFS_DIR / f"{lid}.csv.gz", index_col=["issue", "time"], parse_dates=["issue", "time"])


def known_at(issues: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.Series:
    """For each hour, the start of the newest issue public by then (its start + PUBLIC hours); NaT if none."""
    starts = pd.DatetimeIndex(issues.index.get_level_values("issue").unique()).sort_values()
    public = pd.Series(starts, index=starts + pd.Timedelta(hours=PUBLIC))
    return public.reindex(hours, method="ffill")


if __name__ == "__main__":
    first, last = (date.fromisoformat(x) for x in sys.argv[1:3])
    for g, f in build(first, last).items():
        print(f"{g}: {f.index.get_level_values('issue').nunique()} issues, {len(f):,} rows", flush=True)
