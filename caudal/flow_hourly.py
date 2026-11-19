"""The hourly flow product for the Russian River gauges, first version: frames, scaling, scores by lead hour
and the function that issues.

One regional quantile LSTM reads, for each gauge and hour, a window of the last LOOKBACK hours: the basin mean
of the weather, running rainfall totals, and the observed flow of the gauge and of the points upstream of it,
each flow with a missing flag. The newest hours of every flow are hidden at random in training, so one model
serves every lead (`caudal.training.MaskRecent`), and the model predicts the change from the newest flow it can
see (`caudal.training.Anchor`), so a forecast starts from the last observation. Observed flow as an input follows Feng, Fang and Shen (2020)
and Nearing et al. (2022); the window setup follows Kratzert et al. (2019), Hydrol. Earth Syst. Sci. 23,
5089-5110.

shortcut: the scores feed the model observed weather at every lead, and training reads the reanalysis while
issuing reads the forecast model (caudal issue #8). shortcut: a lead L is scored with the flow of the L hours
before the target hidden, as if the hour of the issue had not arrived yet; the issue reads whatever has.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from caudal.models.quantile_lstm import predict_quantiles
from caudal.points import GAUGES, OUTFLOW, RAIN_POINTS, upstream
from caudal.scores import crps_ensemble
from caudal.sources import openmeteo, usgs
from caudal.training import Anchor, MaskRecent

PRODUCT = "quantile_flow_hourly_v2"
LOG = True  # v2 reads and predicts flow as log(1 + cfs); v1 read it in cfs
LEADS = 120  # hours
LOOKBACK = 240  # hours
MAX_HIDDEN = LEADS + 6  # the longest lead, and a few hours for a late gauge
HIDDEN = 128
BATCH_SIZE = 512
FIT_END = "2015-09-30 23:00"
EVAL_START, EVAL_END = "2015-10-01 00:00", "2025-09-30 23:00"  # water years 2016-2025
EVAL_LEADS = (1, 6, 12, 24, 48, 72, 120)
HIGH = 0.99  # the hours above this quantile of a gauge's fit-window flow are its high-flow hours
LOW = 0.25  # and the hours below this one its low-flow hours

WET = {"precip_24h": 24, "precip_72h": 72, "precip_7d": 168, "precip_30d": 720}  # running rainfall totals, hours
DYNAMIC = [*openmeteo.COLS, *WET, "doy_sin", "doy_cos"]
STATIC = ["static_log_qmean"]
SCALED = DYNAMIC + STATIC
TARGET = "flow"
RAW = [TARGET, "up1", "up2"]  # cfs: the gauge and up to two points upstream
PAIRS = [("obs_flow", "obs_missing"), ("up1_flow", "up1_missing"), ("up2_flow", "up2_missing")]
FEATURES = SCALED + [c for pair in PAIRS for c in pair]


def sites(lid: str) -> list[str]:
    """The USGS sites a gauge's frame reads: its own, then those of the points upstream that have one."""
    return [GAUGES[lid][0], *[OUTFLOW[p] for p in sorted(upstream(lid)) if p in OUTFLOW]]


def build_frame(lid: str, weather: pd.DataFrame, flows: list[pd.Series], qmean: float | None = None) -> pd.DataFrame:
    """Hourly frame of SCALED and RAW over the weather's hours. `flows` are the series of `sites(lid)`, in cfs.
    The row of an hour holds the weather of the next hour beside the flow of this one, so the window that ends
    just before a target hour has the weather up to that hour and the flow up to the hour before. `qmean` is
    the gauge's mean flow over the fit window; left out, it is computed here."""
    out = weather[openmeteo.COLS].copy()
    for name, hours in WET.items():
        out[name] = out["precip"].rolling(hours).sum()
    angle = 2 * np.pi * out.index.day_of_year.to_numpy() / 365.25
    out["doy_sin"], out["doy_cos"] = np.sin(angle), np.cos(angle)
    out[DYNAMIC] = out[DYNAMIC].shift(-1)
    for name, k in zip(RAW, range(len(RAW))):
        out[name] = flows[k].reindex(out.index) if k < len(flows) else np.nan
    out["static_log_qmean"] = np.log(out.loc[:FIT_END, TARGET].mean() if qmean is None else qmean)
    return out.iloc[:-1].dropna(subset=DYNAMIC)


def scale_of(s) -> dict:
    """A gauge's flow scale: {"log": bool, "m": mean, "s": spread}. A bare number is v1's: cfs over a spread."""
    return s if isinstance(s, dict) else {"log": False, "m": 0.0, "s": float(s)}


def to_z(q, sc: dict):
    """Flow in cfs to the model's scale. In log, a recession is a proportional fall and no value can be below 0 cfs."""
    x = np.log1p(np.clip(q, 0, None)) if sc["log"] else q
    return (x - sc["m"]) / sc["s"]


def to_cfs(z, sc: dict):
    x = np.asarray(z) * sc["s"] + sc["m"]
    return np.expm1(x) if sc["log"] else x


def prep(frame: pd.DataFrame, f_mean, f_std, scale) -> pd.DataFrame:
    """Scale a frame for the model: SCALED by the pooled statistics, every flow by the gauge's own scale
    (`scale_of`), and each flow input as its value (0 where missing) and a missing flag."""
    sc = scale_of(scale)
    out = frame.copy()
    out[SCALED] = (frame[SCALED] - f_mean) / f_std
    for (value, flag), raw in zip(PAIRS, RAW):
        scaled = to_z(frame[raw], sc)
        out[value], out[flag] = scaled.fillna(0.0), scaled.isna().astype(float)
    out[TARGET] = to_z(frame[TARGET], sc)
    return out


def load_frame(lid: str) -> pd.DataFrame:
    """A gauge's frame from the archives under data/: the basin mean of its rainfall points and its flows."""
    weather = openmeteo.basin_mean([openmeteo.load(*p) for p in RAIN_POINTS[lid]])
    return build_frame(lid, weather, [usgs.load(s) for s in sites(lid)])


def build_regional(lids: list[str], frames: dict[str, pd.DataFrame] | None = None) -> dict:
    """{"train": {lid: prepped fit frame}, "raw": {lid: full frame}, "scale", "qmean", "f_mean", "f_std"}."""
    raw = frames or {lid: load_frame(lid) for lid in lids}
    fit = {lid: fr.loc[:FIT_END] for lid, fr in raw.items()}
    pool = pd.concat(fit.values())
    f_mean = pool[SCALED].mean().to_numpy(dtype=np.float32)
    f_std = pool[SCALED].std().to_numpy(dtype=np.float32)
    f_std = np.where(f_std > 0, f_std, 1.0).astype(np.float32)
    x = {lid: np.log1p(fr[TARGET].clip(lower=0)) if LOG else fr[TARGET] for lid, fr in fit.items()}
    scale = {lid: {"log": LOG, "m": float(x[lid].mean()) if LOG else 0.0, "s": float(x[lid].std())} for lid in fit}
    qmean = {lid: float(fr[TARGET].mean()) for lid, fr in fit.items()}
    train = {lid: prep(fr, f_mean, f_std, scale[lid]) for lid, fr in fit.items()}
    return {"train": train, "raw": raw, "scale": scale, "qmean": qmean, "f_mean": f_mean, "f_std": f_std}


def windows(lead: int | None = None):
    """What a window dataset is wrapped in before the model reads it: the newest `lead` hours of every flow
    hidden (none when the frame itself has no flow there, as at an issue), then the target taken as its change
    from the newest flow left."""
    def wrap(ds):
        return Anchor(MaskRecent(ds, FEATURES, lead, pairs=PAIRS) if lead else ds, FEATURES, PAIRS[0])
    return wrap


def evaluate(model, frames: dict, lid: str, lead: int) -> dict:
    """Scores of one gauge at one lead over the evaluation years, in cfs."""
    raw, sc = frames["raw"][lid], scale_of(frames["scale"][lid])
    history = raw.loc[:EVAL_START].iloc[-(LOOKBACK + 1) : -1]
    ev = prep(pd.concat([history, raw.loc[EVAL_START:EVAL_END]]), frames["f_mean"], frames["f_std"], sc)
    change, moved, idx = predict_quantiles(model, ev, FEATURES, TARGET, LOOKBACK, batch_size=4096, wrap=windows(lead))
    z = ev[TARGET].reindex(idx).to_numpy()
    preds = np.clip(to_cfs(change + (z - moved)[:, None], sc), 0, None)  # z - moved is the newest flow the window showed
    obs = raw[TARGET].reindex(idx).to_numpy()
    last = raw[TARGET].reindex(idx - pd.Timedelta(hours=lead + 1)).to_numpy()  # the newest flow the forecast saw
    ok = np.isfinite(last)
    fit = raw.loc[:FIT_END, TARGET]
    high, low = ok & (obs >= fit.quantile(HIGH)), ok & (obs <= fit.quantile(LOW))
    crps, miss = crps_ensemble(preds, obs), np.abs(last - obs)
    q = {x: i for i, x in enumerate(model.quantiles)}
    inside = (preds[:, q[0.05]] <= obs) & (obs <= preds[:, q[0.95]])
    return {
        "lead": lead, "lid": lid, "n_hours": int(ok.sum()), "n_high": int(high.sum()),
        "crps_model": crps[ok].mean(), "crps_persistence": miss[ok].mean(),
        "crps_model_high": crps[high].mean(), "crps_persistence_high": miss[high].mean(),
        "crps_model_low": crps[low].mean(), "crps_persistence_low": miss[low].mean(),
        "cov90": inside[ok].mean(), "cov90_high": inside[high].mean(),
        # forecasts no river here makes: a lower bound at zero, a median under the lowest hour of the fit years
        "share_q05_zero": (preds[ok, q[0.05]] < 1).mean(), "share_q50_below_record": (preds[ok, q[0.5]] < fit.min()).mean(),
    }


def evaluate_all(model, frames: dict, leads=EVAL_LEADS) -> pd.DataFrame:
    return pd.DataFrame([evaluate(model, frames, lid, lead) for lead in leads for lid in frames["train"]]).set_index(["lead", "lid"])


def issue_point(model, ckpt: dict, lid: str, hour: pd.Timestamp, weather: pd.DataFrame, flows: list[pd.Series]) -> pd.DataFrame:
    """The forecast issued at `hour` (UTC, on the hour) for the LEADS hours after it: one row per hour with the
    predicted quantiles in cfs. `weather` runs from at least 40 days before the hour (the 30-day rainfall total
    and the window) to LEADS hours after it; `flows` are the series of `sites(lid)` up to the newest hour that
    has arrived. Hours with no flow yet carry the missing flag, which is how the model was trained to see them."""
    weather = weather.loc[: hour + pd.Timedelta(hours=LEADS + 1)]  # one more: build_frame's shift drops the last row
    frame = build_frame(lid, weather, [f.loc[:hour] for f in flows], qmean=ckpt["qmean"][lid])
    sc = scale_of(ckpt["scale"][lid] if "scale" in ckpt else ckpt["s_target"][lid])  # v1 checkpoints carry s_target
    z = prep(frame, np.array(ckpt["f_mean"]), np.array(ckpt["f_std"]), sc)
    z[TARGET] = z[TARGET].fillna(0.0)  # windows are built only on hours with a target; the value is never read
    change, moved, idx = predict_quantiles(model, z, ckpt["features"], TARGET, ckpt["lookback"], batch_size=BATCH_SIZE,
                                           wrap=windows())
    preds = to_cfs(change - moved[:, None], sc)  # the target is 0 here, so -moved is the newest observed flow
    out = pd.DataFrame(np.clip(preds, 0, None).round(1), index=idx,
                       columns=[f"q{int(round(q * 100)):02d}" for q in ckpt["quantiles"]])
    out = out[out.index > hour]
    out.insert(0, "lead", ((out.index - hour) / pd.Timedelta(hours=1)).astype(int))
    out.index.name = "time"
    if len(out) != LEADS:
        raise RuntimeError(f"{lid}: {len(out)} forecast hours, expected {LEADS} (weather through {weather.index[-1]})")
    return out
