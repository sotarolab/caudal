"""Paths. Data under <repo>/data unless EMBALSE_DATA is set; observed inflow comes from the inflow-ledger checkout
next to this repo unless INFLOW_LEDGER_DATA points at its data/ directory."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("EMBALSE_DATA") or REPO_ROOT / "data")
FORCING_DIR = DATA_ROOT / "forcing"
RESULTS_DIR = DATA_ROOT / "results"
MODELS_DIR = DATA_ROOT / "models"
CW3E_DIR = DATA_ROOT / "cw3e"  # CW3E AR catalog cache (embalse.data.cw3e)
LEDGER_DATA = Path(os.environ.get("INFLOW_LEDGER_DATA") or REPO_ROOT.parent / "inflow-ledger" / "data")
