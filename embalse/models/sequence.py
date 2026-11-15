"""Gap-aware sliding windows over daily frames and device selection (from caudal.models.lstm)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def get_device() -> torch.device:
    """Best available device -- CUDA, then Apple MPS, then CPU. A DataLoader/model wired to this instead of
    a hardcoded 'cpu' is what makes a rented cloud GPU actually usable
    without code changes when this moves off a laptop.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SequenceDataset(Dataset):
    """Sliding-window (lookback -> next-day-target) sequences from one or more basins' daily time series.

    (a) Each frame is split into contiguous daily runs and windows never cross a gap;
    (b) the target may be NaN (those days yield no sample -- streamflow is never an input
    feature, so a NaN target day still serves as history); (c) windows are sliced lazily in
    ``__getitem__`` (views into one array per run, no copies).
    """

    def __init__(
        self, frames: list[pd.DataFrame], feature_cols: list[str], target_col: str, lookback: int
    ):
        """
        Parameters
        ----------
        frames : list[pd.DataFrame]
            One DataFrame per basin, daily DatetimeIndex (gaps allowed -- windows will not cross
            them), columns include `feature_cols` and `target_col`. Feature columns must be
            NaN-free (caller's responsibility -- imputation is a real design decision, not
            silently handled here); the target column may contain NaN (no sample that day).
        feature_cols : list[str]
            Input feature columns -- this is exactly where the covariate ablation happens: pass
            ["precip", "temp"] for baseline, add "upslope_ivt" and/or "antecedent_moisture_pct" for the
            physical-covariate variants, same model/training code otherwise.
        target_col : str
            Streamflow column name.
        lookback : int
            Number of past days fed to the LSTM per prediction.
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self._features: list[np.ndarray] = []  # one (run_len, n_features) array per contiguous run
        self._targets: list[np.ndarray] = []
        self._samples: list[tuple[int, int]] = []  # (run_id, end position within run)
        self.timestamps: list[pd.Timestamp] = []  # target day per sample, aligned with __getitem__
        for frame in frames:
            gap = np.flatnonzero(frame.index.to_series().diff().dt.days.to_numpy()[1:] != 1) + 1
            for run in np.split(np.arange(len(frame)), gap):
                if len(run) <= lookback:
                    continue
                run_id = len(self._features)
                self._features.append(frame[feature_cols].values[run].astype(np.float32))
                target = frame[target_col].values[run].astype(np.float32)
                self._targets.append(target)
                for i in range(lookback, len(run)):
                    if np.isfinite(target[i]):
                        self._samples.append((run_id, i))
                        self.timestamps.append(frame.index[run[i]])

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        run_id, i = self._samples[idx]
        x = self._features[run_id][i - self.lookback : i]
        return torch.from_numpy(x), torch.tensor(self._targets[run_id][i], dtype=torch.float32)

