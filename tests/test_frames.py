import numpy as np
import pandas as pd

from caudal import config, frames


def test_build_frame_shifts_forcing_and_keeps_target_gaps(tmp_path, monkeypatch):
    idx = pd.date_range("2000-01-01", periods=40, freq="D")
    (tmp_path / "forcing").mkdir()
    (tmp_path / "cdec_inflow").mkdir()
    pd.DataFrame({c: np.arange(40.0) for c in frames.forcing.COLS}, index=idx).rename_axis("date").to_csv(tmp_path / "forcing" / "FOLC1.csv")
    obs = pd.Series(100.0, index=idx[5:], name="cfs").rename_axis("date")
    obs.iloc[10] = -5  # negative computed inflow -> NaN target
    obs.to_csv(tmp_path / "cdec_inflow" / "FOL.csv")
    monkeypatch.setattr(frames.forcing, "FORCING_DIR", tmp_path / "forcing")
    monkeypatch.setattr(frames, "LEDGER_DATA", tmp_path)
    monkeypatch.setattr(config, "LEDGER_DATA", tmp_path)

    fr = frames.build_frame("FOLC1", fit_end="2000-02-28")
    assert len(fr) == 39 and fr["precip"].iloc[0] == 1.0  # row d holds the forcing of day d+1
    assert fr[frames.TARGET].isna().sum() == 5 + 1 and fr[frames.TARGET].iloc[-1] == 100.0
    assert list(fr.columns[-len(frames.STATIC):]) == frames.STATIC and np.isclose(fr["static_log_qmean"].iloc[0], np.log(100.0))


def test_prep_adds_observed_inflow_and_missing_flag():
    idx = pd.date_range("2000-01-01", periods=4, freq="D")
    fr = pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0] for c in frames.FEATURES}, index=idx)
    fr[frames.TARGET] = [50.0, np.nan, 150.0, np.nan]
    n = len(frames.FEATURES)
    z = frames.prep(fr, np.full(n, 2.0), np.full(n, 2.0), s_target=100.0)
    assert z["obs_inflow"].tolist() == [0.5, 0.0, 1.5, 0.0] and z["obs_missing"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert z[frames.FEATURES[0]].tolist() == [-0.5, 0.0, 0.5, 1.0]  # v1's features are standardized as before
    assert z[frames.TARGET].isna().tolist() == [False, True, False, True]  # the target keeps its gaps
