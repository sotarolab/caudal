"""Daily meteorological forcing from Open-Meteo: ERA5 archive for history, the forecast API for the last days
and the next 16. Train = serve: both come from the same API family with the same variables.

shortcut: forcing is sampled at the dam, not averaged over the basin; upgrade to basin-mean over the HUC
polygon once the first model scores, the Sierra reservoirs will gain the most.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import date, timedelta

import pandas as pd

from caudal.config import FORCING_DIR
from caudal.points import POINTS

VARS = {"precipitation_sum": "precip", "temperature_2m_max": "tmax", "temperature_2m_min": "tmin",
        "temperature_2m_mean": "tmean", "et0_fao_evapotranspiration": "pet"}
COLS = list(VARS.values())
ARCHIVE = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           "&start_date={start}&end_date={end}&daily={daily}&timezone=UTC")
FORECAST = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&past_days={past}&forecast_days=16&daily={daily}&timezone=UTC")
START = "1990-01-01"
ERA5_LAG_DAYS = 7  # ERA5 arrives about 5 days late; refetching a week keeps the tail final


def _daily(url: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.load(r)["daily"]
            break
        except Exception:  # Open-Meteo rate-limits bursts; back off and retry
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))
    df = pd.DataFrame({VARS[k]: v for k, v in d.items() if k in VARS}, index=pd.to_datetime(d["time"]))
    df.index.name = "date"
    return df[COLS].dropna(how="all")


def _url(template: str, lid: str, **kw) -> str:
    _, _, lat, lon, _ = POINTS[lid]
    return template.format(lat=lat, lon=lon, daily=",".join(VARS), **kw)


def update_archive(lid: str, today: date | None = None) -> pd.DataFrame:
    """Extend data/forcing/<lid>.csv to the latest ERA5 day and return it."""
    FORCING_DIR.mkdir(parents=True, exist_ok=True)
    path = FORCING_DIR / f"{lid}.csv"
    old = load(lid) if path.exists() else pd.DataFrame(columns=COLS)
    start = START if old.empty else (old.index[-1] - timedelta(days=ERA5_LAG_DAYS)).date().isoformat()
    end = (today or date.today()).isoformat()
    new = _daily(_url(ARCHIVE, lid, start=start, end=end))
    out = new if old.empty else pd.concat([old.loc[: pd.Timestamp(start) - pd.Timedelta(days=1)], new])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "date"
    out.to_csv(path, float_format="%.2f")
    return out


def load(lid: str) -> pd.DataFrame:
    return pd.read_csv(FORCING_DIR / f"{lid}.csv", index_col=0, parse_dates=True).rename_axis("date")


def fetch_forecast(lid: str, past_days: int = 14) -> pd.DataFrame:
    """Recent analysis days plus the 16-day forecast, from the forecast model (not ERA5)."""
    return _daily(_url(FORECAST, lid, past=past_days))


if __name__ == "__main__":
    import sys

    for lid in sys.argv[1:] or POINTS:
        t0 = time.time()
        df = update_archive(lid)
        print(f"{lid}: {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}, {len(df)} days ({time.time() - t0:.0f}s)", flush=True)
