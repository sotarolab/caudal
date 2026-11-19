import numpy as np
import pandas as pd
import pytest
import torch

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


class LearnedNothing(torch.nn.Module):
    """A model that predicts no change at any quantile."""

    quantiles = (0.05, 0.5, 0.95)

    def forward(self, x):
        return torch.zeros(len(x), 3)


def test_a_model_that_predicts_no_change_issues_the_last_observed_flow():
    hour = HOURS[24 * 44]
    ckpt = {"features": H.FEATURES, "lookback": H.LOOKBACK, "quantiles": [0.05, 0.5, 0.95], "qmean": {"HOPC1": 500.0},
            "f_mean": [0.0] * len(H.SCALED), "f_std": [1.0] * len(H.SCALED), "s_target": {"HOPC1": 300.0}}
    own = flow(until=24 * 44 - 2)  # the gauge's newest hour is three hours before the issue hour
    out = H.issue_point(LearnedNothing(), ckpt, "HOPC1", hour, weather(), [own])
    assert np.allclose(out[["q05", "q50", "q95"]].to_numpy(), own.iloc[-1], atol=0.1)  # every lead, every quantile


def test_issue_leaves_a_forecast_already_issued_alone(tmp_path, monkeypatch):
    import scripts.issue_hourly as issue
    from caudal.sources import openmeteo

    ckpt = {"features": H.FEATURES, "hidden": 4, "quantiles": [0.05, 0.5, 0.95], "points": ["HOPC1"],
            "state_dict": QuantileLSTM(n_features=len(H.FEATURES), hidden_size=4, quantiles=[0.05, 0.5, 0.95]).state_dict()}
    monkeypatch.setattr(issue.torch, "load", lambda *a, **k: ckpt)
    monkeypatch.setattr(openmeteo, "forecast", lambda *a, **k: pytest.fail("an issued hour was fetched again"))
    path = tmp_path / "forecasts" / "m" / "HOPC1" / "2026-11-18T06Z.csv"
    path.parent.mkdir(parents=True)
    path.write_text("time,lead,q05,q50,q95\n2026-11-18T07:00Z,1,1,2,3\n")
    assert issue.main(["--model", "m", "--out", str(tmp_path), "--hour", "2026-11-18T06"]) == 0
    assert path.read_text() == "time,lead,q05,q50,q95\n2026-11-18T07:00Z,1,1,2,3\n"


def test_log_target_round_trips_and_stays_above_zero():
    sc = {"log": True, "m": 5.0, "s": 1.5}
    q = np.array([0.0, 3.0, 150.0, 90000.0])
    assert np.allclose(H.to_cfs(H.to_z(q, sc), sc), q)
    assert H.to_cfs(-50.0, sc) >= -1  # the lowest the log scale can give is -1 cfs, which the issue clips to 0
    assert H.scale_of(300.0) == {"log": False, "m": 0.0, "s": 300.0}  # a v1 checkpoint's bare spread


def test_with_the_log_target_no_change_is_the_last_flow_and_nothing_falls_below_zero():
    hour = HOURS[24 * 44]
    ckpt = {"features": H.FEATURES, "lookback": H.LOOKBACK, "quantiles": [0.05, 0.5, 0.95], "qmean": {"HOPC1": 500.0},
            "f_mean": [0.0] * len(H.SCALED), "f_std": [1.0] * len(H.SCALED),
            "scale": {"HOPC1": {"log": True, "m": 5.0, "s": 1.5}}}
    own = flow(until=24 * 44 - 2)
    out = H.issue_point(LearnedNothing(), ckpt, "HOPC1", hour, weather(), [own])
    assert np.allclose(out[["q05", "q50", "q95"]].to_numpy(), own.iloc[-1], atol=0.1)
    model = QuantileLSTM(n_features=len(H.FEATURES), hidden_size=4, quantiles=ckpt["quantiles"])
    assert (H.issue_point(model, ckpt, "HOPC1", hour, weather(), [own])[["q05", "q50", "q95"]] >= 0).all().all()
