"""Score the hourly candidates, HEFS and persistence on the same hours, and write the table.

Every product is read at five quantiles (5, 25, 50, 75 and 95%), its CRPS taken over them as an ensemble, on
the hours every product has a value for. A lead L is a target L + 1 hours after the newest observed flow.

    python scripts/compare_hourly.py --lstm cand_v1 cand_weighted_log cand_full --hefspp cand_hefspp \
        --blend cand_v1+cand_weighted_log cand_v1+cand_full cand_v1+cand_weighted_log+cand_full
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from caudal import flow_hourly as H
from caudal import hefs_postproc as P
from caudal.config import MODELS_DIR, RESULTS_DIR
from caudal.models.quantile_lstm import QuantileLSTM
from caudal.scores import crps_ensemble
from caudal.sources import hefs

Q5 = ["q05", "q25", "q50", "q75", "q95"]


def lstm_hours(name: str, leads) -> pd.DataFrame:
    ck = torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
    H.configure(ck.get("config", H.V1))
    model = QuantileLSTM(n_features=len(ck["features"]), hidden_size=ck["hidden"], quantiles=ck["quantiles"])
    model.load_state_dict(ck["state_dict"])
    frames = H.build_regional(list(hefs.GAUGES))
    frames["scale"] = ck.get("scale") or ck["s_target"]
    parts = [H.predictions(model, frames, lid, lead).assign(lid=lid, lead=lead) for lid in hefs.GAUGES for lead in leads]
    return pd.concat(parts).rename_axis("t").reset_index().set_index(["lid", "lead", "t"])


def hefspp_hours(name: str, leads) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The post-processor's quantiles and raw HEFS's, on the same hours."""
    ck = torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
    model = P.QuantileMLP(ck["n_in"], quantiles=ck["quantiles"])
    model.load_state_dict(ck["state_dict"])
    pp, raw = [], []
    for lid in hefs.GAUGES:
        src = P.gauge_inputs(lid)
        for lead in leads:
            x, _, _, q0, t, g = P.samples(lid, src, H.EVAL_START, H.EVAL_END, gaps=np.array([lead + 1]))
            idx = pd.MultiIndex.from_arrays([[lid] * len(t), [lead] * len(t), t], names=["lid", "lead", "t"])
            preds = P.predict(model, x, q0)
            pp.append(pd.DataFrame(preds, index=idx, columns=[f"q{int(round(q * 100)):02d}" for q in ck["quantiles"]]))
            h = t - pd.to_timedelta(g, unit="h")
            issue = hefs.known_at(src["hefs"], h)
            vals = src["hefs"][hefs.COLS].reindex(pd.MultiIndex.from_arrays([issue, t])).to_numpy()
            raw.append(pd.DataFrame(vals, index=idx, columns=Q5))
    return pd.concat(pp), pd.concat(raw)


def score(products: dict[str, pd.DataFrame], obs: pd.Series, last: pd.Series, limits: dict) -> pd.DataFrame:
    rows = []
    for name, f in products.items():
        for (lid, lead), g in f.groupby(level=["lid", "lead"]):
            o = obs.loc[g.index].to_numpy()
            q = g[Q5].to_numpy() if name != "persistence" else last.loc[g.index].to_numpy()[:, None]
            crps = crps_ensemble(q, o)
            hi, lo, low_record = limits[lid]
            high, low = o >= hi, o <= lo
            inside = (q[:, 0] <= o) & (o <= q[:, -1])
            rows.append({"product": name, "lid": lid, "lead": lead, "hours": len(o), "crps": crps.mean(),
                         "crps_high": crps[high].mean(), "crps_low": crps[low].mean(),
                         "cov90": inside.mean(), "cov90_high": inside[high].mean(),
                         "q05_zero": (q[:, 0] < 1).mean(), "q50_below_record": (q[:, q.shape[1] // 2] < low_record).mean()})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lstm", nargs="*", default=[])
    p.add_argument("--hefspp", default=None)
    p.add_argument("--fit-end", default="2019-09-30 23:00")
    p.add_argument("--eval", nargs=2, default=["2019-10-01 00:00", "2025-09-30 23:00"])
    p.add_argument("--out", default="compare_hourly")
    p.add_argument("--blend", nargs="*", default=[], help="products to average quantile by quantile, as a+b(+c)")
    a = p.parse_args(argv)
    H.set_years(a.fit_end, *a.eval)
    leads = H.EVAL_LEADS
    products, obs, last = {}, None, None
    for name in a.lstm:
        f = lstm_hours(name, leads)
        products[name] = f
        obs, last = f["obs"], f["last"]
    if a.hefspp:
        products[a.hefspp], products["HEFS"] = hefspp_hours(a.hefspp, leads)
    for spec in a.blend:  # quantile averaging (Lichtendahl, Grushka-Cockayne and Winkler 2013, Manage. Sci. 59)
        parts = [products[n] for n in spec.split("+")]
        idx = parts[0].index
        products[f"blend:{spec}"] = sum(p.reindex(idx)[Q5] for p in parts) / len(parts)
    products["persistence"] = next(iter(products.values()))[[]].copy()
    common = None
    for f in products.values():
        common = f.index if common is None else common.intersection(f.index)
    ok = obs.reindex(common).notna() & last.reindex(common).notna()
    for name, f in products.items():
        if name != "persistence":
            ok &= f.reindex(common)[Q5].notna().all(axis=1)
    common = common[ok.to_numpy()]
    products = {k: v.reindex(common) for k, v in products.items()}
    from caudal.points import GAUGES
    from caudal.sources import usgs
    limits = {}
    for lid in hefs.GAUGES:
        fit = usgs.load(GAUGES[lid][0]).loc[: a.fit_end]
        limits[lid] = (fit.quantile(H.HIGH), fit.quantile(H.LOW), fit.min())
    table = score(products, obs.reindex(common), last.reindex(common), limits)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / f"{a.out}.csv", index=False)
    pooled = table.groupby(["lead", "product"]).mean(numeric_only=True).round(3)
    pd.set_option("display.width", 220)
    print(f"{len(common):,} gauge-lead-hours scored by every product, {a.eval[0]} to {a.eval[1]}\n")
    print(pooled.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
