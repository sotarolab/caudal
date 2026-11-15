"""Training callbacks - per-epoch metrics, checkpointing, early stopping, live progress.

Hook names (`on_train_start` / `on_epoch_end` / `on_train_end`) match PyTorch Lightning's
`pl.Callback` so these can be wrapped as one with a trivial shim. `__call__(info: dict)` is
also supported - the shape scipy's `differential_evolution` and most calibration loops
expect - so a single callback object can be passed to both the GR4J/CemaNeige calibration
and the neural training loops.

`on_epoch_end` returns a bool: **True means stop training**. That is how `EarlyStopping`
signals the loop; every other callback returns None (falsy) and is ignored.

Callbacks never decide what a metric means, they only record it: the training loop computes
losses and validation scores, callbacks persist them. A callback that recomputes a metric its
own way is how two numbers in one repo start disagreeing.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from embalse.training.artifacts import RunArtifacts


@runtime_checkable
class Callback(Protocol):
    """Structural protocol - anything with these methods works; no base class required."""

    def on_train_start(self, info: dict[str, Any]) -> None: ...

    def on_epoch_end(self, info: dict[str, Any]) -> bool | None: ...

    def on_train_end(self, info: dict[str, Any]) -> None: ...


class BaseCallback:
    """No-op implementations so subclasses override only the hook they care about.

    Also provides `__call__(info)` -> `on_epoch_end(info)` for plain-callable call sites.
    """

    def on_train_start(self, info: dict[str, Any]) -> None:
        return None

    def on_epoch_end(self, info: dict[str, Any]) -> bool | None:
        return None

    def on_train_end(self, info: dict[str, Any]) -> None:
        return None

    def __call__(self, info: dict[str, Any]) -> bool | None:
        return self.on_epoch_end(info)


class MetricsLogger(BaseCallback):
    """Append every epoch's metrics to the run's metrics.csv.

    Parameters
    ----------
    artifacts : RunArtifacts
    exclude : sequence of str, optional
        Keys in `info` that are not metrics (e.g. "model", "optimizer"). Non-scalar values are
        dropped automatically; this is for scalars you simply do not want stored.
    """

    # `n_epochs` is a run constant carried in `info` for ProgressLogger; logging it per row
    # would put a constant column in every metrics.csv.
    NON_METRIC = ("model", "optimizer", "scheduler", "loader", "artifacts", "n_epochs")

    def __init__(self, artifacts: RunArtifacts, exclude: Sequence[str] = ()) -> None:
        self.artifacts = artifacts
        self.exclude = set(self.NON_METRIC) | set(exclude)

    def on_epoch_end(self, info: dict[str, Any]) -> None:
        # Scalars only: models/optimizers/loaders land in `info` too, and a DataFrame or a
        # tensor in a CSV cell is worse than useless. bool is a subclass of int and logs as
        # 0/1, which is what we want for flags like "improved".
        metrics = {
            k: v
            for k, v in info.items()
            if k not in self.exclude and k != "epoch" and isinstance(v, (int, float))
        }
        self.artifacts.log_epoch(int(info.get("epoch", -1)), **metrics)


class CheckpointSaver(BaseCallback):
    """Periodic + best-so-far checkpointing, with pruning so long runs do not fill the disk.

    Parameters
    ----------
    artifacts : RunArtifacts
    state_fn : callable
        ``state_fn(info) -> dict`` returning everything needed to RESUME (weights, optimizer
        state, normalization). Supplied by the caller because only it knows what its model
        needs; a weights-only checkpoint cannot continue a run.
    every_n_epochs : int, default 10
        0 disables periodic snapshots (best/last still written).
    monitor : str, optional
        Metric key in `info` tracked for ``best.pt``.
    mode : {"min", "max"}, default "min"
    keep_last : int, default 3
        Periodic snapshots retained; ``best.pt``/``last.pt`` are never pruned.
    """

    def __init__(
        self,
        artifacts: RunArtifacts,
        state_fn: Callable[[dict[str, Any]], dict[str, Any]],
        every_n_epochs: int = 10,
        monitor: str | None = None,
        mode: str = "min",
        keep_last: int = 3,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.artifacts = artifacts
        self.state_fn = state_fn
        self.every_n_epochs = every_n_epochs
        self.monitor = monitor
        self.mode = mode
        self.keep_last = keep_last
        self.best_value = math.inf if mode == "min" else -math.inf
        self.best_epoch: int | None = None

    def _improved(self, value: float) -> bool:
        return value < self.best_value if self.mode == "min" else value > self.best_value

    def on_epoch_end(self, info: dict[str, Any]) -> None:
        epoch = int(info.get("epoch", -1))
        if self.every_n_epochs and epoch > 0 and epoch % self.every_n_epochs == 0:
            self.artifacts.save_checkpoint(self.state_fn(info), epoch)
            self.artifacts.prune_checkpoints(self.keep_last)

        if self.monitor is not None:
            value = info.get(self.monitor)
            if isinstance(value, (int, float)) and math.isfinite(value) and self._improved(value):
                self.best_value = float(value)
                self.best_epoch = epoch
                self.artifacts.save_checkpoint(self.state_fn(info), epoch, tag="best")

        self.artifacts.update_manifest(
            status="training",
            epoch=epoch,
            best_metric=self.monitor,
            best_value=self.best_value if self.best_epoch else None,
            best_epoch=self.best_epoch,
        )

    def on_train_end(self, info: dict[str, Any]) -> None:
        epoch = int(info.get("epoch", -1))
        self.artifacts.save_checkpoint(self.state_fn(info), epoch, tag="last")
        self.artifacts.update_manifest(
            status=info.get("status", "finished"),
            epoch=epoch,
            best_metric=self.monitor,
            best_value=self.best_value if self.best_epoch else None,
            best_epoch=self.best_epoch,
        )


class EarlyStopping(BaseCallback):
    """Stop when a monitored metric stops improving. Returns True from `on_epoch_end`.

    Parameters
    ----------
    monitor : str
    mode : {"min", "max"}, default "min"
    patience : int, default 10
        Epochs without improvement before stopping.
    min_delta : float, default 0.0
        Improvement smaller than this does not count.

    Notes
    -----
    Only meaningful against a VALIDATION metric. Monitoring training loss just stops when the
    optimizer stalls, which is a different (and usually useless) thing - the caller is
    responsible for passing something held out.
    """

    def __init__(
        self, monitor: str, mode: str = "min", patience: int = 10, min_delta: float = 0.0
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf if mode == "min" else -math.inf
        self.bad_epochs = 0
        self.stopped_epoch: int | None = None

    def on_epoch_end(self, info: dict[str, Any]) -> bool | None:
        value = info.get(self.monitor)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = float(value)
            self.bad_epochs = 0
            return None
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.stopped_epoch = int(info.get("epoch", -1))
            return True
        return None


class ProgressLogger(BaseCallback):
    """Print a metric line every `frequency` epochs.

    Long background runs (`nohup ... > log`) then produce a readable, greppable trace instead
    of silence - which is how the overnight fleet run was actually monitored.
    """

    def __init__(self, frequency: int = 1, prefix: str = "") -> None:
        self.frequency = max(1, frequency)
        self.prefix = prefix
        self._t0 = time.time()

    def on_train_start(self, info: dict[str, Any]) -> None:
        self._t0 = time.time()
        n = info.get("n_epochs", "?")
        print(f"[{time.strftime('%H:%M:%S')}] {self.prefix}training start: {n} epochs", flush=True)

    def on_epoch_end(self, info: dict[str, Any]) -> None:
        epoch = int(info.get("epoch", -1))
        if epoch % self.frequency:
            return
        scalars = " ".join(
            f"{k}={v:.4f}"
            for k, v in info.items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and k not in ("epoch", *MetricsLogger.NON_METRIC)
        )
        print(f"[{time.strftime('%H:%M:%S')}] {self.prefix}epoch {epoch}: {scalars}", flush=True)

    def on_train_end(self, info: dict[str, Any]) -> None:
        print(
            f"[{time.strftime('%H:%M:%S')}] {self.prefix}training end "
            f"({time.time() - self._t0:.0f}s, status={info.get('status', 'finished')})",
            flush=True,
        )


def dispatch(callbacks: Sequence[Any], hook: str, info: dict[str, Any]) -> bool:
    """Call `hook` on every callback; return True if ANY asked to stop.

    Every callback runs even when one requests a stop, so a stopping epoch is still logged and
    checkpointed - losing the final epoch's artifacts to an early-stop signal would be a
    silent, annoying data loss.
    """
    stop = False
    for cb in callbacks or ():
        fn = getattr(cb, hook, None)
        if fn is None:
            continue
        stop = bool(fn(info)) or stop
    return stop
