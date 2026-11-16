"""Paths. Data under <repo>/data unless CAUDAL_DATA is set; observed inflow comes from the hapi checkout
next to this repo unless HAPI_DATA points at its data/ directory."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CAUDAL_DATA") or REPO_ROOT / "data")
FORCING_DIR = DATA_ROOT / "forcing"
RESULTS_DIR = DATA_ROOT / "results"
MODELS_DIR = DATA_ROOT / "models"
CW3E_DIR = DATA_ROOT / "cw3e"  # CW3E AR catalog cache (caudal.data.cw3e)
LEDGER_DATA = Path(os.environ.get("HAPI_DATA") or REPO_ROOT.parent / "hapi" / "data")
