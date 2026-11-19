"""An hourly flow forecast that corrects HEFS at the Russian River gauges: a model of HEFS's error.

For an issue at hour H (the newest observed flow) and a target hour t = H + g, the model reads what was known at
H: the gauge's flow and its recent changes, the flows upstream and below the dams, the rain that has fallen and the
soil moisture, and the HEFS issue public at H (its quantiles for t, and how far its value for H is from what was
observed). It predicts quantiles of the change in log(1 + flow) from H to t, so a forecast starts from the river
and cannot fall below zero, with each hour weighted by its flow so floods count as they do in cfs.

It reads no rain after H: the only forecast weather it sees is the one inside HEFS. Its scores on past years are
therefore those of a forecast made with the weather forecast of the day, not with the rain that fell.

Post-processing a hydrologic ensemble by quantile regression on its own forecast and the observed state follows
Brown and Seo (2013), Hydrol. Process. 27, 83-105, and Bogner and Pappenberger (2011), Water Resour. Res. 47,
W07524; the network and its monotone quantile head are the ones `caudal.models.quantile_lstm` uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from caudal import flow_hourly as H
from caudal.points import GAUGES, RAIN_POINTS
from caudal.sources import hefs, openmeteo, usgs

PRODUCT = "quantile_flow_hourly_hefspp_v1"
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
MAX_GAP = 126  # hours from the newest observation to the target: the longest lead and a late gauge
UPSTREAM = 3  # flows read beside the gauge's own, padded with missing
FIT_START = "2014-01-02 00:00"  # the first HEFS issue in CNRFC's archive is 2014-01-01 12Z


def log(q):
    return np.log1p(np.clip(np.asarray(q, dtype=float), 0, None))


def gauge_inputs(lid: str) -> dict:
    """Everything a gauge's samples are read from, hourly: flows, basin weather, HEFS issues."""
    H.configure(H.FULL)
    flows = [usgs.load(s) for s in H.sites(lid)]
    weather = openmeteo.basin_mean([openmeteo.load(*p) for p in RAIN_POINTS[lid]])
    return {"flows": flows, "weather": weather, "hefs": hefs.load(lid)}


def features(lid: str, src: dict, issue_hours: pd.DatetimeIndex, gaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The inputs of each (issue hour, gap) pair, and the newest observed flow at each issue hour (cfs).
    Missing values are 0 with a flag beside them."""
    h, t = issue_hours, issue_hours + pd.to_timedelta(gaps, unit="h")
    own = src["flows"][0]
    q0 = own.reindex(h).to_numpy()
    cols, flags = [], []

    def add(values, flagged=True):
        v = np.asarray(values, dtype=float)
        if flagged:
            flags.append(np.isnan(v).astype(float))
        cols.append(np.nan_to_num(v))

    add(gaps / MAX_GAP, flagged=False)
    add((log(q0) - 5.0) / 2.0, flagged=False)  # the level, roughly centred; rows without q0 are dropped
    for k in (1, 3, 24):
        add(log(q0) - log(own.reindex(h - pd.Timedelta(hours=k)).to_numpy()))
    for u in range(UPSTREAM):  # each upstream flow relative to the gauge's, and its 3-hour change
        f = src["flows"][1 + u] if 1 + u < len(src["flows"]) else pd.Series(dtype=float)
        fu = f.reindex(h).to_numpy() if len(f) else np.full(len(h), np.nan)
        add(log(fu) - log(q0))
        add(log(fu) - log(f.reindex(h - pd.Timedelta(hours=3)).to_numpy()) if len(f) else np.full(len(h), np.nan))
    w = src["weather"]
    rain = w["precip"]
    for k in (24, 72, 720):
        add(log(rain.rolling(k).sum().reindex(h).to_numpy()), flagged=False)
    for c in ("soil_0_7", "soil_7_28", "soil_28_100"):
        add(w[c].reindex(h).to_numpy() * 4 - 1.2)
    angle = 2 * np.pi * h.day_of_year.to_numpy() / 365.25
    add(np.sin(angle), flagged=False)
    add(np.cos(angle), flagged=False)
    for g in GAUGES:
        add(np.full(len(h), float(g == lid)), flagged=False)
    issue = hefs.known_at(src["hefs"], h)
    tab = src["hefs"]

    def at(times, col):
        return tab[col].reindex(pd.MultiIndex.from_arrays([issue, times])).to_numpy()

    for c in hefs.COLS:  # HEFS for the target hour, relative to the river now
        add(log(at(t, c)) - log(q0))
    add(log(q0) - log(at(h, "hefs_q50")))  # HEFS's error now, in the same issue
    return np.column_stack(cols + flags).astype(np.float32), q0


class QuantileMLP(nn.Module):
    """Two hidden layers and the monotone quantile head of `QuantileLSTM`: the first quantile is free and each
    next one adds a softplus step, so quantiles never cross."""

    def __init__(self, n_in: int, hidden: int = 256, quantiles=QUANTILES):
        super().__init__()
        self.quantiles = tuple(quantiles)
        self.register_buffer("_q", torch.tensor(self.quantiles, dtype=torch.float32))
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, len(self.quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.body(x))
        return torch.cat([raw[:, :1], raw[:, :1] + torch.cumsum(nn.functional.softplus(raw[:, 1:]), dim=1)], dim=1)


def samples(lid: str, src: dict, start: str, end: str, gaps: np.ndarray | None = None, rng=None):
    """Training or scoring samples over [start, end]: every hour with an observed flow, a random gap each
    (or every gap in `gaps`), and a target flow observed. Returns (x, change in log flow, weight, q0, t)."""
    own = src["flows"][0]
    hours = own.loc[start:end].index
    if gaps is None:
        g = rng.integers(1, MAX_GAP + 1, size=len(hours))
        h = hours
    else:
        h = hours.repeat(len(gaps))
        g = np.tile(gaps, len(hours))
    x, q0 = features(lid, src, h, g)
    t = h + pd.to_timedelta(g, unit="h")
    qt = own.reindex(t).to_numpy()
    ok = np.isfinite(q0) & np.isfinite(qt)
    y = (log(qt) - log(q0)).astype(np.float32)
    w = (qt / np.nanmean(own.loc[:end])).astype(np.float32)
    return x[ok], y[ok], w[ok], q0[ok], t[ok], g[ok]


def train(x, y, w, epochs: int = 30, lr: float = 1e-3, seed: int = 0, device="cpu") -> QuantileMLP:
    from caudal.models.quantile_lstm import pinball_loss

    torch.manual_seed(seed)
    model = QuantileMLP(x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    q = model._q
    X, Y, W = (torch.from_numpy(a).to(device) for a in (x, y, w))
    for epoch in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(X), 2048):
            b = perm[i : i + 2048]
            opt.zero_grad()
            loss = pinball_loss(model(X[b]), Y[b], q, W[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if epoch % 10 == 9:
            print(f"  epoch {epoch + 1}: pinball {loss.item():.4f}", flush=True)
    return model.cpu().eval()


@torch.no_grad()
def predict(model: QuantileMLP, x: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Quantiles of the flow in cfs."""
    change = model(torch.from_numpy(x)).numpy()
    return np.clip(np.expm1(log(q0)[:, None] + change), 0, None)
