import numpy as np
import pandas as pd
import pytest

from caudal import flow_hourly as H
from caudal.models.quantile_lstm import QuantileLSTM
from caudal.models.sequence import SequenceDataset

HOURS = pd.date_range("2019-01-01", periods=24 * 50, freq="h", name="time")


def weather() -> pd.DataFrame:
    n = np.arange(len(HOURS), dtype=float)
    return pd.DataFrame({"precip": n % 7, "temp": 10 + n % 5}, index=HOURS)


def flow(offset: float = 0.0, until: int | None = None) -> pd.Series:
    return pd.Series(100 + offset + np.arange(len(HOURS), dtype=float), index=HOURS, name="cfs")[:until]


def test_frame_puts_the_next_hours_weather_beside_this_hours_flow():
    w = weather()
    fr = H.build_frame("HEAC1", w, [flow(), flow(1000)], qmean=500.0)
    t = fr.index[10]
    assert fr.index[0] == HOURS[719 - 1]  # the first hour whose next hour has a 30-day rainfall total
    assert fr.loc[t, "precip"] == w.precip[t + pd.Timedelta(hours=1)] and fr.loc[t, "flow"] == flow()[t]
    assert fr.loc[t, "precip_24h"] == w.precip[t - pd.Timedelta(hours=22) : t + pd.Timedelta(hours=1)].sum()
    z = H.prep(fr, np.zeros(len(H.SCALED)), np.ones(len(H.SCALED)), 1.0)
    ds = SequenceDataset([z], H.FEATURES, H.TARGET, lookback=5)
    x, y = ds[0]
    target = ds.timestamps[0]
    newest = x[-1, H.FEATURES.index("obs_flow")]
    assert y == flow()[target] and newest == flow()[target - pd.Timedelta(hours=1)]  # a window never holds its target's flow
    assert x[-1, H.FEATURES.index("precip")] == w.precip[target]  # and it does hold the target hour's weather


def test_a_gauge_without_upstream_points_reads_zero_and_the_missing_flag():
    assert H.sites("HOPC1") == ["11462500"] and H.sites("GUEC1") == ["11467000", "11464000", "11465350"]
    fr = H.build_frame("HOPC1", weather(), [flow()], qmean=500.0)
    z = H.prep(fr, np.zeros(len(H.SCALED)), np.ones(len(H.SCALED)), 2.0)
    assert (z.up1_flow == 0).all() and (z.up1_missing == 1).all() and (z.up2_missing == 1).all()
    assert (z.obs_missing == 0).all() and z.obs_flow.iloc[0] == fr.flow.iloc[0] / 2.0


def test_issue_writes_leads_one_to_120_from_the_issue_hour():
    hour = HOURS[24 * 44]  # 44 days into the weather; 6 days of forecast weather follow
    ckpt = {"features": H.FEATURES, "lookback": H.LOOKBACK, "quantiles": [0.05, 0.5, 0.95], "qmean": {"HEAC1": 500.0},
            "f_mean": [0.0] * len(H.SCALED), "f_std": [1.0] * len(H.SCALED), "s_target": {"HEAC1": 300.0}}
    model = QuantileLSTM(n_features=len(H.FEATURES), hidden_size=4, quantiles=ckpt["quantiles"])
    flows = [flow(until=24 * 44), flow(1000, until=24 * 44 - 3)]  # the upstream gauge is three hours late
    out = H.issue_point(model, ckpt, "HEAC1", hour, weather(), flows)
    assert out.index[0] == hour + pd.Timedelta(hours=1) and out.lead.tolist() == list(range(1, 121))
    assert list(out.columns) == ["lead", "q05", "q50", "q95"]
    assert (out.q05 <= out.q50).all() and (out.q50 <= out.q95).all() and (out[["q05", "q50", "q95"]] >= 0).all().all()
    with pytest.raises(RuntimeError, match="forecast hours"):
        H.issue_point(model, ckpt, "HEAC1", hour, weather().loc[: hour + pd.Timedelta(hours=50)], flows)
