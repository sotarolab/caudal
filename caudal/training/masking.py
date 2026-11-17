"""Hide the newest days of observed inflow in a window, so one model serves every lead.

A forecast for lead L is made without the inflow of the L days before the target day. Hiding the newest m days
of a training window (inflow 0, missing flag 1), with m drawn at random, shows the model each of those cases,
and the same wrapper with a fixed m scores it at one lead. Lagged observed flow with missing values marked is
the autoregressive input of Nearing et al. (2022), Hydrol. Earth Syst. Sci. 26, 5493-5513.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class MaskRecent(Dataset):
    """Wrap a (window, target) dataset. `value` and `flag` are the columns of the observed inflow and of its
    missing flag; the newest `days` rows are hidden, or a uniform draw from 0..days when `random`."""

    def __init__(self, ds: Dataset, value: int, flag: int, days: int, random: bool = False):
        self.ds, self.value, self.flag, self.days, self.random = ds, value, flag, days, random

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.ds[i]
        m = int(torch.randint(self.days + 1, ())) if self.random else self.days
        if m:
            x = x.clone()  # the window is a view into the dataset's own array
            x[-m:, self.value], x[-m:, self.flag] = 0.0, 1.0
        return x, y
