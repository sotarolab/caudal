"""Training frames: forcing + observed inflow per point, and the pooled normalization for one regional model.

Follows caudal's regional arm: features standardized with statistics pooled over every point's fit window;
the target scaled per point by its own fit-window standard deviation so large reservoirs do not dominate the
loss; dynamic features shifted one day so the window for day d ends with day d's forcing (what a forecast run
has in hand).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from embalse import forcing
from embalse.config import LEDGER_DATA
from embalse.points import POINTS

DYNAMIC = [*forcing.COLS, "doy_sin", "doy_cos"]
STATIC = ["static_lat", "static_elev", "static_log_qmean"]
FEATURES = DYNAMIC + STATIC
TARGET = "inflow"
LOOKBACK = 365
HIDDEN = 128
BATCH_SIZE = 256
FIT_END = "2015-09-30"
EVAL_START, EVAL_END = "2015-10-01", "2025-09-30"  # water years 2016-2025


def load_obs(lid: str) -> pd.Series:
    """Observed daily inflow in cfs from the inflow-ledger CDEC files; negative computed inflows become NaN."""
    sta = POINTS[lid][0]
    s = pd.read_csv(LEDGER_DATA / "cdec_inflow" / f"{sta}.csv", index_col="date", parse_dates=True)["cfs"]
    return s.where(s >= 0).rename(TARGET)


def build_frame(lid: str, fit_end: str = FIT_END, f: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily frame with FEATURES and TARGET (NaN where unobserved), over the forcing calendar
    (the archived one unless `f`, e.g. archive plus forecast days, is given)."""
    _, _, lat, _, elev = POINTS[lid]
    f = forcing.load(lid) if f is None else f
    out = f.join(load_obs(lid), how="left")
    angle = 2 * np.pi * out.index.day_of_year.to_numpy() / 365.25
    out["doy_sin"], out["doy_cos"] = np.sin(angle), np.cos(angle)
    out["static_lat"], out["static_elev"] = lat, elev
    out["static_log_qmean"] = np.log(out.loc[:fit_end, TARGET].mean())
    out[DYNAMIC] = out[DYNAMIC].shift(-1)
    return out.iloc[:-1].dropna(subset=FEATURES)


def prep(frame: pd.DataFrame, f_mean, f_std, s_target: float) -> pd.DataFrame:
    out = frame.copy()
    out[FEATURES] = (frame[FEATURES] - f_mean) / f_std
    out[TARGET] = frame[TARGET] / s_target
    return out


def build_regional(lids: list[str], fit_end: str = FIT_END) -> dict:
    """{"train": {lid: prepped fit frame}, "raw": {lid: full frame}, "s_target", "f_mean", "f_std"}."""
    raw = {lid: build_frame(lid, fit_end) for lid in lids}
    fit = {lid: fr.loc[:fit_end] for lid, fr in raw.items()}
    pool = pd.concat(fit.values())
    f_mean = pool[FEATURES].mean().to_numpy(dtype=np.float32)
    f_std = pool[FEATURES].std().to_numpy(dtype=np.float32)
    f_std = np.where(f_std > 0, f_std, 1.0).astype(np.float32)  # a constant channel (one point) stays finite
    train, s_target = {}, {}
    for lid, fr in fit.items():
        s = float(fr[TARGET].std())
        if np.isfinite(s) and s > 0:
            train[lid] = prep(fr, f_mean, f_std, s)
            s_target[lid] = s
    return {"train": train, "raw": raw, "s_target": s_target, "f_mean": f_mean, "f_std": f_std}
