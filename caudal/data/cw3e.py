"""CW3E landfalling atmospheric-river catalog - the flood-event stratifier for the US arm.

Provides the fetch/parse layer for the published catalog plus the basin-side machinery:
coastal-point assignment per basin, daily AR series, and the West Coast domain.

Wires the published catalog (https://cw3e.ucsd.edu/Projects/ARCatalog/); does NOT implement AR
detection - that logic belongs to the published catalog (Ralph et al. 2019 AR Scale), not to
this module. Layout: a fixed grid of coastal points (`json/points.json`, a JS `var point = {...}`
assignment rather than bare JSON, stripped here), each with a per-point event CSV
(`csv/eventData/CW3E_ARoutput_{lat*10}_{lon*10}.csv`) of 13 unlabeled columns documented on the
catalog page. Points span 25.0N-55.0N, so the whole HUC 17/18 study domain is covered; the
catalog record starts 1959, comfortably spanning the 2005-2014 evaluation window.

On this coast the flood-generating mechanism is landfalling ARs - roughly 80-90%% of
extreme precipitation and most major flooding in coastal CA/OR/WA (Ralph et al. 2006;
Neiman et al. 2008). Splitting the evaluation into AR-driven and non-AR days asks whether a
model holds up on the days that matter; a pooled NSE can look strong while missing every AR
flood.

Threshold notes:

- **min_ar_scale = 3.** Ralph et al. (2019)'s scale puts "hazardous" impacts at AR3+; AR1-2 are
  "primarily beneficial." AR3+ is therefore the flood-relevant regime, and the higher bar also
  cuts the dilution risk in the any-point union below.
  `run_flood_sensitivity.py` sweeps the AR floor over 2/3/4; the method ranking is
  unchanged across the grid.
- **POINT_LAT_BAND_DEG = 1.5.** A basin is assigned every coastal point within +/-1.5 deg
  latitude (~165 km) of its gauge, and a day is an AR day if ANY assigned point registers a
  qualifying AR. ARs make landfall as broad swaths, not at a single grid point, so a single
  nearest-point assignment would miss events whose core landed one point away.
  The nearest-point-only alternative is `assign_coastal_points(..., band_deg=0.0)`.

Longitude is deliberately NOT used to assign points: every catalog point sits on the coastline,
so its longitude encodes coastline shape, not the basin's exposure. Latitude is what selects
which stretch of coast an AR hit. (A Sierra Nevada basin at -119.5 is fed by ARs making landfall
at its own latitude, ~-122; matching on distance would be dominated by that irrelevant offset.)
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from caudal.config import CW3E_DIR

CW3E_BASE_URL = "https://cw3e.ucsd.edu/Projects/ARCatalog/"

EVENT_COLUMNS = [
    "start_date",
    "max_ivt_date",
    "end_date",
    "duration_hours",
    "max_ivt",
    "ar_scale",
    "avg_ivt",
    "max_ivt_direction",
    "avg_ivt_direction",
    "max_iwv",
    "avg_iwv",
    "avg_u_ivt",
    "avg_v_ivt",
]

# Study domain: the CONUS West Coast that HUC 17/18 drains to (S. California through the
# Strait of Juan de Fuca); the basin pool reaches Washington.
WEST_COAST_LAT_RANGE = (32.0, 49.0)

MIN_AR_SCALE = 3  # see module docstring
POINT_LAT_BAND_DEG = 1.5  # see module docstring

_POINTS_CACHE: pd.DataFrame | None = None
_EVENTS_CACHE: dict[str, pd.DataFrame] = {}


def _fetch(url: str, dest: Path, timeout: int = 60) -> None:
    """Download `url` to `dest` unless it is already cached. Catalog files are static."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read()
    dest.write_bytes(payload)


def fetch_points(save_dir: Path = CW3E_DIR) -> pd.DataFrame:
    """The full grid of CW3E coastal catalog points (fetched once, then cached).

    Returns
    -------
    pd.DataFrame
        Columns `lat_code`/`lon_code` (the string codes used in event-CSV filenames, e.g.
        "375", "1225") and `lat`/`lon` (float degrees, lon negative west).
    """
    global _POINTS_CACHE
    if _POINTS_CACHE is not None:
        return _POINTS_CACHE

    cache_path = save_dir / "points.json"
    _fetch(CW3E_BASE_URL + "json/points.json", cache_path, timeout=30)
    raw = cache_path.read_text()

    # Strip the "var point = " JS wrapper and the trailing comma before "]" (valid JS, invalid
    # JSON) so it parses with the stdlib rather than a lenient third-party parser.
    text = raw[raw.index("{") : raw.rindex("}") + 1]
    text = re.sub(r",\s*([\]}])", r"\1", text)
    data = json.loads(text)

    _POINTS_CACHE = pd.DataFrame(
        [
            {
                "lat_code": f["properties"]["lat"],
                "lon_code": f["properties"]["lon"],
                "lat": f["geometry"]["coordinates"][1],
                "lon": f["geometry"]["coordinates"][0],
            }
            for f in data["features"]
        ]
    ).sort_values("lat", ignore_index=True)
    return _POINTS_CACHE


def west_coast_points(save_dir: Path = CW3E_DIR) -> pd.DataFrame:
    """Catalog points over the CONUS West Coast study domain (see WEST_COAST_LAT_RANGE)."""
    points = fetch_points(save_dir)
    lo, hi = WEST_COAST_LAT_RANGE
    return points[points["lat"].between(lo, hi)].reset_index(drop=True)


def point_id(lat_code: str, lon_code: str) -> str:
    """Stable label for one catalog point, e.g. "385_1230"."""
    return f"{lat_code}_{lon_code}"


def load_ar_events(lat_code: str, lon_code: str, save_dir: Path = CW3E_DIR) -> pd.DataFrame:
    """AR event catalog for one coastal point - one row per EVENT, not per timestep.

    Parameters
    ----------
    lat_code, lon_code : str
        Codes from `fetch_points` (e.g. "375", "1225" for 37.5N, 122.5W).

    Returns
    -------
    pd.DataFrame
        EVENT_COLUMNS, with start_date/max_ivt_date/end_date parsed to datetime.
    """
    key = point_id(lat_code, lon_code)
    if key in _EVENTS_CACHE:
        return _EVENTS_CACHE[key]

    filename = f"CW3E_ARoutput_{lat_code}_{lon_code}.csv"
    cache_path = save_dir / filename
    _fetch(CW3E_BASE_URL + "csv/eventData/" + filename, cache_path)

    df = pd.read_csv(cache_path, header=None, names=EVENT_COLUMNS)
    for col in ("start_date", "max_ivt_date", "end_date"):
        df[col] = pd.to_datetime(df[col], format="%Y%m%d%H")
    _EVENTS_CACHE[key] = df
    return df


def assign_coastal_points(
    gauge_lat: float, band_deg: float = POINT_LAT_BAND_DEG, save_dir: Path = CW3E_DIR
) -> pd.DataFrame:
    """Coastal points feeding one basin: every West Coast point within `band_deg` latitude.

    Falls back to the single nearest point when the band is empty (or when band_deg == 0), so
    a basin always gets at least one point. See the module docstring on why latitude alone -
    not distance - does the assigning.

    Parameters
    ----------
    gauge_lat : float
        Basin gauge latitude, degrees north (CAMELS-US `gauge_lat`).
    band_deg : float, default POINT_LAT_BAND_DEG
        Half-width of the latitude band. 0.0 selects the nearest point only.

    Returns
    -------
    pd.DataFrame
        The selected rows of `west_coast_points`.
    """
    points = west_coast_points(save_dir)
    if band_deg > 0:
        sel = points[(points["lat"] - gauge_lat).abs() <= band_deg]
        if len(sel):
            return sel.reset_index(drop=True)
    nearest = (points["lat"] - gauge_lat).abs().idxmin()
    return points.loc[[nearest]].reset_index(drop=True)


def ar_scale_series(
    points: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    save_dir: Path = CW3E_DIR,
) -> pd.Series:
    """Daily maximum AR scale over `points`, on the full daily calendar of [start, end].

    A catalog event spans hours (`start_date` to `end_date`); every calendar day it touches
    takes that event's scale. Where several events or points overlap a day, the MAXIMUM scale
    wins - the day is characterized by the strongest AR affecting the basin's coastline, which
    is the flood-relevant quantity. Non-AR days are 0 (not NaN: the catalog is a complete
    record over its period, so "no event" is an observation, not a gap).

    Returns
    -------
    pd.Series
        Int-valued, DatetimeIndex over the full daily calendar, name "ar_scale".
    """
    calendar = pd.date_range(start, end, freq="D")
    scale = pd.Series(0, index=calendar, dtype=int, name="ar_scale")
    lo, hi = calendar[0], calendar[-1]

    for _, p in points.iterrows():
        events = load_ar_events(p["lat_code"], p["lon_code"], save_dir)
        # Clip to the window first - the catalog runs from 1959, most of it irrelevant here.
        touching = events[(events["end_date"] >= lo) & (events["start_date"] <= hi)]
        for s, e, sc in zip(touching["start_date"], touching["end_date"], touching["ar_scale"]):
            days = pd.date_range(max(s.normalize(), lo), min(e.normalize(), hi), freq="D")
            if len(days):
                scale.loc[days] = np.maximum(scale.loc[days].to_numpy(), int(sc))
    return scale


def ar_day_series(
    points: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    min_ar_scale: int = MIN_AR_SCALE,
    save_dir: Path = CW3E_DIR,
) -> pd.Series:
    """Boolean daily "a qualifying AR affected this coastline" indicator.

    Thin wrapper on `ar_scale_series` - see the module docstring on `min_ar_scale` and on the
    any-point union this inherits.

    Returns
    -------
    pd.Series
        Bool, DatetimeIndex over the full daily calendar, name "is_ar_day".
    """
    return (ar_scale_series(points, start, end, save_dir) >= min_ar_scale).rename("is_ar_day")


def basin_ar_frame(
    gauge_lat: float,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    min_ar_scale: int = MIN_AR_SCALE,
    band_deg: float = POINT_LAT_BAND_DEG,
    save_dir: Path = CW3E_DIR,
) -> pd.DataFrame:
    """One basin's daily AR context: scale, qualifying-day flag, and the points used.

    Convenience entry point for the pipeline - `assign_coastal_points` + `ar_scale_series`
    in one call.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns `ar_scale` (int) and `is_ar_day` (bool). The assigned point
        ids are attached as ``frame.attrs["points"]`` for provenance in saved outputs.
    """
    points = assign_coastal_points(gauge_lat, band_deg=band_deg, save_dir=save_dir)
    scale = ar_scale_series(points, start, end, save_dir)
    frame = pd.DataFrame({"ar_scale": scale, "is_ar_day": scale >= min_ar_scale})
    frame.attrs["points"] = [point_id(r.lat_code, r.lon_code) for r in points.itertuples()]
    frame.attrs["min_ar_scale"] = min_ar_scale
    return frame


def prefetch_west_coast(save_dir: Path = CW3E_DIR) -> int:
    """Download every West Coast point's event CSV up front (one-time, ~170 KB each).

    Called by the drivers before any parallel/looping work so the fetch happens once, visibly,
    rather than lazily inside a scoring loop.

    Returns
    -------
    int
        Number of points now cached.
    """
    points = west_coast_points(save_dir)
    for _, p in points.iterrows():
        load_ar_events(p["lat_code"], p["lon_code"], save_dir)
    return len(points)
