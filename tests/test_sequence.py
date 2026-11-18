import numpy as np
import pandas as pd

from caudal.models.sequence import SequenceDataset


def frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(len(index), dtype=float), "y": np.arange(len(index), dtype=float)}, index=index)


def test_windows_follow_the_frames_own_step():
    hours = pd.date_range("2019-01-01", periods=10, freq="h")
    ds = SequenceDataset([frame(hours)], ["x"], "y", lookback=3)
    assert len(ds) == 7 and ds.timestamps[0] == hours[3]
    x, y = ds[0]
    assert x[:, 0].tolist() == [0.0, 1.0, 2.0] and y == 3.0  # the three hours before the target, not the target
    days = pd.date_range("2019-01-01", periods=10, freq="D")
    assert len(SequenceDataset([frame(days)], ["x"], "y", lookback=3)) == 7  # a daily frame as before


def test_a_missing_hour_splits_the_windows():
    hours = pd.date_range("2019-01-01", periods=12, freq="h").delete(6)  # 06:00 is missing
    ds = SequenceDataset([frame(hours)], ["x"], "y", lookback=3)
    assert [t.hour for t in ds.timestamps] == [3, 4, 5, 10, 11]  # no window reaches across the gap
