"""Observed flow at a USGS gauge: 15-minute discharge from the instantaneous-values service, kept as an hourly
series in UTC that grows under data/flow_hourly/.

An hourly value is the mean of the 15-minute values in the hour ending at its stamp, the convention of
Open-Meteo's hourly precipitation, so the flow and the rain of one stamp cover the same hour.

shortcut: only the last REFRESH_DAYS days are read again on an update, and USGS revises provisional values for
months; delete a file's tail before a study that needs approved values.

    python -m caudal.sources.usgs 11462500 11464000 11467000    # build or extend the archives
"""

from __future__ import annotations

import gzip
import http.client
import io
import sys
import time
import urllib.request
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from caudal.config import FLOW_DIR

# A request older than 120 days is redirected to nwis.waterservices.usgs.gov; urllib follows it.
IV = "https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites={site}&parameterCd=00060&startDT={start}&endDT={end}"
START = date(1987, 10, 1)  # the first day of 15-minute discharge at the Russian River gauges
UTC_OFFSET = {"PST": 8, "PDT": 7}  # hours from the local time the service writes to UTC; another zone raises
REFRESH_DAYS = 2
CHUNK_DAYS = 120  # a year-long request was answered 503 on 2026-11-17; half a year, compressed, took 23 s


def parse(text: str) -> pd.Series:
    """15-minute discharge in cfs on a UTC index, from the service's tab-separated answer. A value that is not
    a number (Ice, Eqp) is missing."""
    lines = [x for x in text.splitlines() if x and not x.startswith("#")]
    if len(lines) < 3:  # the header, the column widths and nothing else
        return pd.Series(dtype=float, name="cfs", index=pd.DatetimeIndex([], name="time"))
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", skiprows=[1], dtype=str)
    offset = df["tz_cd"].map(UTC_OFFSET)
    if offset.isna().any():
        raise ValueError(f"unknown time zone code {sorted(set(df.tz_cd[offset.isna()]))}")
    when = pd.to_datetime(df["datetime"]) + pd.to_timedelta(offset, unit="h")
    flow = pd.to_numeric(df[next(c for c in df.columns if c.endswith("_00060"))], errors="coerce")
    return pd.Series(flow.to_numpy(), index=pd.DatetimeIndex(when, name="time"), name="cfs").dropna()


def hourly(q: pd.Series) -> pd.Series:
    """The mean of the values in the hour ending at each stamp. An hour with no value is absent."""
    return q.resample("1h", closed="right", label="right").mean().dropna()


def fetch(site: str, start: date, end: date) -> pd.Series:
    url = IV.format(site=site, start=start, end=end)
    for attempt in range(5):
        try:
            # compressed, the answer is a sixth of the size, and the service sends about 11 KB a second
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept-Encoding": "gzip"}), timeout=600) as r:
                body = r.read()
            return parse((gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body).decode())
        except (OSError, http.client.HTTPException):  # the service for older data is slow, drops connections
            if attempt == 4:  # and answered 503 to three sites at once on 2026-11-17; the waits add up to 7.5 min
                raise
            time.sleep(30 * 2**attempt)
    raise AssertionError("unreachable")


def complete_hours(site: str, start: date, end: date, fetch=fetch) -> pd.Series:
    """Hourly flow from `start` to `end`, whole hours only: the read starts a day early so the first hour kept
    has all its values, and an hour still in progress is left for the next read."""
    q = fetch(site, start - timedelta(days=1), end)
    h = hourly(q)
    return h.iloc[1:][h.index[1:] <= q.index[-1]] if len(h) else h


def load(site: str) -> pd.Series:
    return pd.read_csv(FLOW_DIR / f"{site}.csv", index_col="time", parse_dates=True)["cfs"]


def update_archive(site: str, today: date | None = None, fetch=fetch) -> pd.Series:
    """Extend data/flow_hourly/<site>.csv to today and return it. The file is written after every request, so
    a build that stops resumes where it was. An hour already in the file stays unless a new read replaces it."""
    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    path = FLOW_DIR / f"{site}.csv"
    out = load(site) if path.exists() else pd.Series(dtype=float, name="cfs", index=pd.DatetimeIndex([], name="time"))
    start = START if out.empty else (out.index[-1] - pd.Timedelta(days=REFRESH_DAYS)).date()
    end = today or datetime.now(UTC).date()
    while start <= end:
        stop = min(end, start + timedelta(days=CHUNK_DAYS - 1))
        out = pd.concat([out, complete_hours(site, start, stop, fetch).round(2)])
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out.to_csv(path)
        start = stop + timedelta(days=1)
    return out


if __name__ == "__main__":
    for site in sys.argv[1:]:
        t0 = time.time()
        q = update_archive(site)
        print(f"{site}: {len(q)} hours, {q.index[0]:%Y-%m-%d} to {q.index[-1]:%Y-%m-%d %H:%M}Z, "
              f"peak {q.max():,.0f} cfs at {q.idxmax():%Y-%m-%d %H:%M}Z ({time.time() - t0:.0f}s)")
