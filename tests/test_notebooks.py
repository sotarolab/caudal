"""The notebooks are checked in clean: valid, with no stored outputs, and using caudal's own model."""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "hourly_flow_lstm_lightning.ipynb"


def test_hourly_lightning_notebook_is_valid_and_has_no_outputs():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4 and nb["metadata"]["kernelspec"]["name"] == "embalse"
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(not c["outputs"] and c["execution_count"] is None for c in code)
    source = "\n".join("".join(c["source"]) for c in code)
    assert "from caudal.models.quantile_lstm import" in source and "class QuantileLSTM" not in source
    assert "L.Trainer(" in source and "pinball_loss(" in source
