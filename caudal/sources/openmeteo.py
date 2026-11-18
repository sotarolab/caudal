"""Hourly weather at a point from Open-Meteo: the archive for the past, kept as a growing file under
data/weather_hourly/, and the forecast for the next days. Precipitation is the total of the hour ending at the
stamp, in mm; temperature is at the stamp, in degrees C. Times are UTC.

One point and year of hourly data counts as about 26 calls against the free allowance (600 a minute, 5,000 an
hour, 10,000 a day), so the first build of a point waits when the service says to.

shortcut: the archive is the reanalysis and the forecast is the best-match forecast model, as for the daily
forcing; training on one and issuing on the other is caudal issue #8.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from caudal.config import WEATHER_DIR

VARS = {"precipitation": "precip", "temperature_2m": "temp"}
COLS = list(VARS.values())
ARCHIVE = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           "&start_date={start}&end_date={end}&hourly={hourly}&timezone=UTC")
FORECAST = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&past_days={past}&forecast_days={days}&hourly={hourly}&timezone=UTC")
START = date(1987, 8, 1)  # two months before the first hourly flow, for the 30-day rainfall total
CHUNK_DAYS = 5 * 365
REFRESH_DAYS = 7  # the newest days of the archive are preliminary and are read again


def parse(answer: dict) -> pd.DataFrame:
    """One point's answer as a frame of COLS on an hourly UTC index. An hour with no value at all is dropped."""
    h = answer["hourly"]
    df = pd.DataFrame({VARS[k]: h[k] for k in VARS}, index=pd.DatetimeIndex(pd.to_datetime(h["time"]), name="time"))
    return df[COLS].astype(float).dropna(how="all")


def get(url: str, attempts: int = 8, timeout: float = 300, wait: float = 30) -> dict:
    """The service's answer. The defaults are for building an archive: a burst is refused until the minute or
    the hour turns, so the waits double from `wait` seconds and add up to 45 minutes. An issue uses `quick`."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(min(900, wait * 2**attempt))
    raise AssertionError("unreachable")


def quick(url: str) -> dict:
    """`get` for an issue, which has minutes, not an hour: four tries of 60 seconds, 5, 10 and 20 seconds
    apart. From GitHub's runners a request to Open-Meteo sometimes times out in the handshake and the next
    one goes through."""
    return get(url, attempts=4, timeout=60, wait=5)


def path(lat: float, lon: float):
    return WEATHER_DIR / f"{lat:.3f}_{lon:.3f}.csv"


def load(lat: float, lon: float) -> pd.DataFrame:
    return pd.read_csv(path(lat, lon), index_col="time", parse_dates=True)


def update_archive(lat: float, lon: float, today: date | None = None, get=get) -> pd.DataFrame:
    """Extend the point's file to today and return it. Written after every request, so a build that stops
    resumes; an hour already in the file stays unless a new read replaces it."""
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    out = load(lat, lon) if path(lat, lon).exists() else pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], name="time"))
    start = START if out.empty else (out.index[-1] - pd.Timedelta(days=REFRESH_DAYS)).date()
    end = today or datetime.now(UTC).date()
    while start <= end:
        stop = min(end, start + timedelta(days=CHUNK_DAYS - 1))
        new = parse(get(ARCHIVE.format(lat=lat, lon=lon, start=start, end=stop, hourly=",".join(VARS))))
        out = pd.concat([out, new]) if len(out) else new
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out.to_csv(path(lat, lon))
        start = stop + timedelta(days=1)
    return out


def forecast(lat: float, lon: float, past_days: int = 3, days: int = 8, get=quick) -> pd.DataFrame:
    """The last analysis days and the next `days` days from the forecast model. Read at an issue, so it gives
    up quickly."""
    return parse(get(FORECAST.format(lat=lat, lon=lon, past=past_days, days=days, hourly=",".join(VARS))))


def basin_mean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """The mean over a basin's sample points, hour by hour, on the hours every point has."""
    return pd.concat(frames, keys=range(len(frames))).groupby(level="time").mean().loc[
        lambda m: m.index.isin(set.intersection(*(set(f.index) for f in frames)))]


if __name__ == "__main__":
    from caudal.points import RAIN_POINTS

    for lat, lon in sorted({p for pts in RAIN_POINTS.values() for p in pts}, reverse=True):
        t0 = time.time()
        w = update_archive(lat, lon)
        print(f"({lat}, {lon}): {len(w):,} hours, {w.index[0]:%Y-%m-%d} to {w.index[-1]:%Y-%m-%d %H}Z, "
              f"{w.precip.sum() / (len(w) / 8766):,.0f} mm a year ({time.time() - t0:.0f}s)", flush=True)
