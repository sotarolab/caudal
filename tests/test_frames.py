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
