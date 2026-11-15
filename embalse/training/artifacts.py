"""Run artifacts: one self-describing directory per training run.

Exists so a training run can be QC'd after the fact and resumed after a crash. Without it a
run's only record is its final loss value, and an interrupted multi-hour fleet loses every
model trained so far.

This is the framework-neutral core: the hand-rolled loops in `embalse.models` and the Lightning
wrappers in `embalse.training.lightning` both write the identical layout below. The callback
protocol matches Lightning's hook names (`on_epoch_end`, `on_train_end`) and also accepts the
plain `__call__(info: dict)` shape calibration loops use, so the conceptual and neural sides
share one idea of what a callback is.

Layout written per run::

    <RESULTS_DIR>/runs/<run_id>/
        config.json        # hyperparameters + environment, written before training starts
        metrics.csv        # one row per epoch, appended live (survives a kill -9)
        manifest.json      # status/timing/git sha, rewritten at each checkpoint
        checkpoints/
            epoch_0012.pt
            best.pt        # by the monitored metric, if one is configured
            last.pt        # always the most recent, for resume

Everything is plain JSON/CSV/`.pt` on purpose: QC-able with `pandas.read_csv` and a text editor,
no run-tracking service, no database, and readable years from now.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embalse.config import RESULTS_DIR

RUNS_DIR = RESULTS_DIR / "runs"


def _git_sha() -> str | None:
    """Short git SHA of the working tree, or None outside a repo.

    Recorded so a run's numbers can be traced back to the code that produced them - the single
    most useful field when a result is questioned months later.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    """True when the working tree has uncommitted changes (the run is not reproducible)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def environment_fingerprint() -> dict[str, Any]:
    """Versions that change results. Recorded in every run's config.json."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["mps"] = bool(torch.backends.mps.is_available())
        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for mod in ("numpy", "pandas"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


@dataclass
class RunArtifacts:
    """One training run's directory, created up front and appended to as it goes.

    Parameters
    ----------
    run_id : str
        Directory name under ``<RESULTS_DIR>/runs``. Callers should make this descriptive and
        deterministic (e.g. ``"regional_westcoast_v4us_s0"``) so a re-run resumes in place
        rather than spawning a new directory each time.
    config : dict
        Hyperparameters and anything else needed to reproduce the run. Written to config.json
        immediately, BEFORE training - so a run that dies in epoch 1 still says what it was.
    root : Path, optional
        Parent directory; defaults to ``<RESULTS_DIR>/runs``.

    Notes
    -----
    `metrics.csv` is flushed on every append. That is slightly wasteful and entirely
    deliberate: the point of this class is that a `kill -9` at epoch 40 of 200 leaves a
    readable record of the first 39 epochs.
    """

    run_id: str
    config: dict[str, Any] = field(default_factory=dict)
    root: Path | None = None

    def __post_init__(self) -> None:
        base = self.root or RUNS_DIR
        self.dir = Path(base) / self.run_id
        self.checkpoints = self.dir / "checkpoints"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.csv"
        self._fields: list[str] | None = None
        self._t0 = time.time()

        payload = {
            "run_id": self.run_id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": self.config,
            "environment": environment_fingerprint(),
        }
        (self.dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))
        self.update_manifest(status="created")

    # --- metrics ---------------------------------------------------------------------

    def log_epoch(self, epoch: int, **metrics: Any) -> None:
        """Append one epoch's metrics to metrics.csv, flushed immediately.

        The header is fixed by the FIRST call. Later calls with new keys would silently produce
        a ragged CSV (the same failure the fleet's METRIC_COLUMNS schema exists to prevent), so
        unknown keys are dropped and missing ones written blank.
        """
        row = {"epoch": int(epoch), "elapsed_s": round(time.time() - self._t0, 2), **metrics}
        new = not self.metrics_path.exists()
        if self._fields is None:
            self._fields = list(row) if new else _read_header(self.metrics_path)
        with open(self.metrics_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fields, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._fields})
            fh.flush()
            os.fsync(fh.fileno())

    def read_metrics(self):
        """Per-epoch metrics as a DataFrame (empty if nothing logged yet)."""
        import pandas as pd

        if not self.metrics_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.metrics_path)

    # --- checkpoints -----------------------------------------------------------------

    def save_checkpoint(self, state: dict[str, Any], epoch: int, tag: str | None = None) -> Path:
        """Write a checkpoint and refresh ``last.pt``.

        `state` should carry everything needed to RESUME, not just weights: optimizer state,
        the epoch number, and the feature normalization. A weights-only checkpoint cannot
        continue a run, which defeats the purpose.
        """
        import torch

        path = self.checkpoints / (f"{tag}.pt" if tag else f"epoch_{epoch:04d}.pt")
        torch.save({**state, "epoch": int(epoch)}, path)
        if tag != "best":
            torch.save({**state, "epoch": int(epoch)}, self.checkpoints / "last.pt")
        return path

    def load_checkpoint(self, which: str = "last") -> dict[str, Any] | None:
        """Load ``last``/``best``/a specific filename. None when absent."""
        import torch

        path = self.checkpoints / f"{which}.pt" if not which.endswith(".pt") else Path(which)
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=False)

    def prune_checkpoints(self, keep_last: int) -> int:
        """Delete all but the `keep_last` most recent ``epoch_*.pt`` (best/last are never
        touched). Returns how many were removed - hourly runs would otherwise fill a disk."""
        if keep_last <= 0:
            return 0
        found = sorted(self.checkpoints.glob("epoch_*.pt"))
        doomed = found[:-keep_last] if len(found) > keep_last else []
        for p in doomed:
            p.unlink()
        return len(doomed)

    # --- manifest --------------------------------------------------------------------

    def update_manifest(self, status: str, **extra: Any) -> None:
        """Rewrite manifest.json - the file to look at first when auditing a run."""
        payload = {
            "run_id": self.run_id,
            "status": status,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(time.time() - self._t0, 1),
            "checkpoints": sorted(p.name for p in self.checkpoints.glob("*.pt")),
            **extra,
        }
        (self.dir / "manifest.json").write_text(json.dumps(payload, indent=2, default=str))

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"RunArtifacts({self.run_id!r} -> {self.dir})"


def _read_header(path: Path) -> list[str]:
    with open(path, newline="") as fh:
        return next(csv.reader(fh))


def list_runs(root: Path | None = None):
    """Every run directory with its manifest, newest first - the QC entry point.

    Returns
    -------
    pandas.DataFrame
        Columns: run_id, status, updated, elapsed_s, n_epochs, plus any extras recorded in the
        manifest. Empty if no runs exist.
    """
    import pandas as pd

    base = Path(root or RUNS_DIR)
    if not base.exists():
        return pd.DataFrame()
    rows = []
    for d in sorted(base.iterdir()):
        manifest = d / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            info = json.loads(manifest.read_text())
        except Exception:
            continue
        metrics = d / "metrics.csv"
        info["n_epochs"] = (sum(1 for _ in open(metrics)) - 1) if metrics.exists() else 0
        info["checkpoints"] = len(info.get("checkpoints", []))
        rows.append(info)
    df = pd.DataFrame(rows)
    return df.sort_values("updated", ascending=False).reset_index(drop=True) if len(df) else df
