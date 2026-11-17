"""Quantile-regression LSTM: predict the tail directly instead of the conditional mean.

Tests whether a model *trained* for the tail beats reading a Gaussian model's ensemble at
a selected quantile. caudal-fleet's `GaussianLSTM` optimises E[Q | forcing]; on
AR-driven flood peaks its mean under-predicts 98% of events (median |relative peak error|
0.420), while the same ensemble read at a train-selected quantile recovers 0.224 - the
skill is present and the summary discards it. A loss that targets the relevant quantiles
directly should not need the selection step.

Design
------
- **Pinball (quantile) loss** over a fixed quantile grid. Unlike Gaussian NLL this makes no
  distributional assumption: each output head is a consistent estimator of its own quantile,
  including the far tail, where a Gaussian's symmetric shape is a poor fit for streamflow.
- **Monotone outputs by construction.** The head emits the lowest quantile plus non-negative
  softplus increments, so predicted quantiles can never cross. Crossing is common with naive
  multi-output quantile regression and would make "the q90 forecast" meaningless.
- **Same backbone, same lookback, same hidden size** as `GaussianLSTM`, so a difference against
  the frozen arm is attributable to the head and the loss and not to capacity.

The quantile grid is denser in the upper tail (0.5, 0.75, 0.9, 0.95, 0.99) because flood
peaks are the target; the low quantiles are estimated from fewer grid neighbours.

Learning rate matters more here than for the Gaussian head: at lr 5e-2 or 1e-2 the
predicted quantiles collapse together (the LSTM's tanh saturates and the head cannot
express a range); at 1e-3 the spread opens correctly. Checked against a synthetic
LogNormal target with known quantiles.
Use lr = 1e-3, which is also what the Gaussian arm uses, so the H3 comparison stays controlled.
A collapsed model still trains to a plausible-looking loss, so this failure is silent: check
that predicted quantiles actually spread before trusting a run.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Denser in the upper tail: this model exists to get flood peaks right. q50 is retained as the
# point forecast so it is directly comparable with the Gaussian arm's mean.
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


class QuantileLSTM(nn.Module):
    """LSTM with a monotone multi-quantile head, trained by pinball loss.

    Parameters
    ----------
    n_features : int
    hidden_size : int, default 128
    num_layers : int, default 1
    dropout : float, default 0.1
        Applied between LSTM layers only (no effect when num_layers == 1), matching
        `GaussianLSTM` so the two are comparable.
    quantiles : sequence of float, default DEFAULT_QUANTILES
        Must be sorted and strictly inside (0, 1).

    Notes
    -----
    `forward` returns shape (batch, n_quantiles), ordered as `quantiles`. Monotonicity is
    structural: output[:, 0] is free and each subsequent quantile adds a softplus increment.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        quantiles=DEFAULT_QUANTILES,
    ):
        super().__init__()
        q = tuple(float(x) for x in quantiles)
        if list(q) != sorted(q):
            raise ValueError("quantiles must be sorted ascending")
        if not all(0.0 < x < 1.0 for x in q):
            raise ValueError("quantiles must lie strictly inside (0, 1)")
        self.quantiles = q
        self.register_buffer("_q", torch.tensor(q, dtype=torch.float32))

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, len(q))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        raw = self.head(h_n[-1])  # (batch, n_quantiles)
        base = raw[:, :1]
        steps = nn.functional.softplus(raw[:, 1:])  # non-negative -> no crossing
        return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)

    @property
    def median_index(self) -> int:
        """Index of the 0.50 head, or the closest quantile to it."""
        return int(np.argmin(np.abs(np.asarray(self.quantiles) - 0.5)))


def pinball_loss(
    pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """Mean pinball (quantile) loss.

    For quantile q, the loss is ``max(q * e, (q - 1) * e)`` with ``e = target - pred``. It
    penalises under-prediction by ``q`` and over-prediction by ``1 - q``, so a high-q head is
    pushed upward - which is exactly the asymmetry the Gaussian mean lacks at flood peaks.

    Parameters
    ----------
    pred : (batch, n_quantiles)
    target : (batch,)
    quantiles : (n_quantiles,)
    """
    err = target.unsqueeze(1) - pred
    return torch.maximum(quantiles * err, (quantiles - 1.0) * err).mean()


def train_quantile_epochs(
    model: QuantileLSTM,
    loader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    callbacks=None,
    stage: str = "quantile",
    val_loader=None,
    start_epoch: int = 0,
):
    """Training loop for `QuantileLSTM`, callback-aware like `lstm._run_epochs`.

    Kept separate from `_run_epochs` rather than parameterised into it: that function is on the
    path that produced the frozen v4us record, and adding a loss switch to it would put the
    frozen arm one bad default away from silently changing.
    """
    from caudal.training.callbacks import dispatch

    model.to(device)
    q = model._q.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    info_base = {"model": model, "optimizer": optimizer, "stage": stage, "n_epochs": epochs}
    dispatch(callbacks, "on_train_start", dict(info_base, epoch=start_epoch))

    losses: list[float] = []
    status, epoch = "finished", start_epoch
    for i in range(epochs):
        epoch = start_epoch + i + 1
        model.train()
        batch_losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = pinball_loss(model(x), y, q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))
        losses.append(train_loss)

        if callbacks:
            info = dict(info_base, epoch=epoch, train_pinball=train_loss, lr=lr)
            if val_loader is not None:
                info["val_pinball"] = _validate_quantile(model, val_loader, device, q)
            if dispatch(callbacks, "on_epoch_end", info):
                status = "early_stopped"
                break

    dispatch(
        callbacks,
        "on_train_end",
        dict(
            info_base,
            epoch=epoch,
            status=status,
            train_pinball=losses[-1] if losses else float("nan"),
        ),
    )
    return losses


@torch.no_grad()
def _validate_quantile(model: QuantileLSTM, loader, device, q: torch.Tensor) -> float:
    model.eval()
    out = []
    for x, y in loader:
        out.append(pinball_loss(model(x.to(device)), y.to(device), q).item())
    model.train()
    return float(np.mean(out)) if out else float("nan")


@torch.no_grad()
def predict_quantiles(
    model: QuantileLSTM, frame, feature_cols, target_col, lookback, batch_size: int = 256, wrap=None
):
    """Quantile predictions over an evaluation frame.

    Returns
    -------
    (preds, obs, index) : (np.ndarray (n, n_quantiles), np.ndarray (n,), pd.DatetimeIndex)
        Aligned one-to-one, using the same gap-aware windowing as the Gaussian arm so the two
        are scored on identical days. `wrap`, if given, maps the window dataset to the one that is read
        (e.g. `caudal.training.MaskRecent`, to hide the newest inflow days); it must keep the order.
    """
    import pandas as pd
    from torch.utils.data import DataLoader

    from caudal.models.sequence import SequenceDataset, get_device

    device = get_device()
    model.to(device)
    model.eval()
    ds = SequenceDataset([frame], feature_cols, target_col, lookback)
    loader = DataLoader(wrap(ds) if wrap else ds, batch_size=batch_size, shuffle=False)

    preds, obs = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        obs.append(y.numpy())
    if not preds:
        return np.empty((0, len(model.quantiles))), np.empty(0), pd.DatetimeIndex([])
    return np.concatenate(preds), np.concatenate(obs), pd.DatetimeIndex(ds.timestamps)
