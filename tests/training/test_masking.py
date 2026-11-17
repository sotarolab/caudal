import torch
from torch.utils.data import TensorDataset

from caudal.training import MaskRecent

VALUE, FLAG = 1, 2  # columns of the observed inflow and of its missing flag in the toy windows


def windows() -> TensorDataset:
    x = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3) + 1  # 2 windows, 6 days, 3 features, no zeros
    x[:, :, FLAG] = 0.0
    return TensorDataset(x, torch.tensor([10.0, 20.0]))


def test_mask_recent_hides_only_the_newest_days():
    ds = windows()
    before = ds.tensors[0].clone()
    x, y = MaskRecent(ds, VALUE, FLAG, days=2)[1]
    assert x[-2:, VALUE].tolist() == [0.0, 0.0] and x[-2:, FLAG].tolist() == [1.0, 1.0]
    assert torch.equal(x[:-2], before[1, :-2]) and torch.equal(x[:, 0], before[1, :, 0]) and y == 20.0
    assert torch.equal(ds.tensors[0], before)  # the wrapped dataset is not written to


def test_mask_of_zero_days_changes_nothing():
    ds = windows()
    x, _ = MaskRecent(ds, VALUE, FLAG, days=0)[0]
    assert torch.equal(x, ds.tensors[0][0])


def test_random_mask_stays_within_range():
    torch.manual_seed(0)
    masked = MaskRecent(windows(), VALUE, FLAG, days=3, random=True)
    hidden = {int(masked[0][0][:, FLAG].sum()) for _ in range(200)}
    assert hidden == {0, 1, 2, 3}
