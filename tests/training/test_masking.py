from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from caudal import frames
from caudal.models.quantile_lstm import QuantileLSTM, predict_quantiles
from caudal.training import MaskRecent

FEATURES = ["rain", *frames.OBS]  # toy windows: one weather column, then the observed inflow and its missing flag
VALUE, FLAG = 1, 2


def windows() -> TensorDataset:
    x = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3) + 1  # 2 windows, 6 days, 3 features, no zeros
    x[:, :, FLAG] = 0.0
    return TensorDataset(x, torch.tensor([10.0, 20.0]))


def test_mask_recent_hides_only_the_newest_days():
    ds = windows()
    before = ds.tensors[0].clone()
    x, y = MaskRecent(ds, FEATURES, days=2)[1]
    assert x[-2:, VALUE].tolist() == [0.0, 0.0] and x[-2:, FLAG].tolist() == [1.0, 1.0]
    assert torch.equal(x[:-2], before[1, :-2]) and torch.equal(x[:, 0], before[1, :, 0]) and y == 20.0
    assert torch.equal(ds.tensors[0], before)  # the wrapped dataset is not written to


def test_mask_finds_its_columns_by_name():
    x, _ = MaskRecent(windows(), [frames.OBS[1], "rain", frames.OBS[0]], days=1)[0]  # flag first, inflow last
    assert x[-1].tolist() == [1.0, 17.0, 0.0]


def test_mask_of_zero_days_changes_nothing():
    ds = windows()
    x, _ = MaskRecent(ds, FEATURES, days=0)[0]
    assert torch.equal(x, ds.tensors[0][0])


def test_random_mask_stays_within_range():
    torch.manual_seed(0)
    masked = MaskRecent(windows(), FEATURES, days=3, random=True)
    hidden = {int(masked[0][0][:, FLAG].sum()) for _ in range(200)}
    assert hidden == {0, 1, 2, 3}


def test_predict_quantiles_reads_the_wrapped_windows():
    """Hiding 2 days through `wrap` gives the prediction of a frame whose last 2 window days were hidden by hand."""
    torch.manual_seed(0)
    lookback = 5
    idx = pd.date_range("2020-01-01", periods=lookback + 1, freq="D")  # exactly one window
    fr = pd.DataFrame({"rain": np.linspace(0, 1, 6), frames.OBS[0]: np.linspace(1, 2, 6), frames.OBS[1]: 0.0,
                       frames.TARGET: 1.0}, index=idx)
    by_hand = fr.copy()
    by_hand.iloc[lookback - 2 : lookback, [VALUE, FLAG]] = [0.0, 1.0]
    model = QuantileLSTM(n_features=3, hidden_size=4)
    args = (FEATURES, frames.TARGET, lookback)
    hidden, _, _ = predict_quantiles(model, fr, *args, wrap=partial(MaskRecent, features=FEATURES, days=2))
    assert np.allclose(hidden, predict_quantiles(model, by_hand, *args)[0])
    assert not np.allclose(hidden, predict_quantiles(model, fr, *args)[0])


def test_mask_hides_every_flow_pair_it_is_given():
    features = ["rain", "obs_flow", "obs_missing", "up1_flow", "up1_missing"]
    x = torch.ones(1, 6, 5)
    x[:, :, [2, 4]] = 0.0
    ds = TensorDataset(x, torch.tensor([1.0]))
    got, _ = MaskRecent(ds, features, days=2, pairs=[("obs_flow", "obs_missing"), ("up1_flow", "up1_missing")])[0]
    assert got[-2:, [1, 3]].sum() == 0 and got[-2:, [2, 4]].tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert got[:-2, [1, 3]].min() == 1 and got[:, 0].min() == 1  # older flow and the weather are untouched
