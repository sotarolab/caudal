# embalse

Private ML package behind the public [inflow-ledger](../inflow-ledger): models, training, calibration and
weights for daily river-flow and reservoir-inflow forecasts at CNRFC points. Forecasts it issues are
committed to inflow-ledger before the outcome exists and scored there against observations, HEFS,
persistence and climatology.

- `embalse/points.py`: the forecast points, their CDEC station and dam coordinates, and how each treats
  regulation (`inflow`, `natural`, `handoff`).
- `embalse/forcing.py`: Open-Meteo daily forcing, ERA5 archive for history and the forecast API for the next 16 days.
- `embalse/frames.py`: training frames and the pooled normalization, caudal's regional protocol.
- `embalse/models/`: pinball-loss LSTM with a monotone quantile head and gap-aware windows, from caudal.
- `embalse/training/`: run artifacts (config, metrics.csv, manifest, checkpoints) and callbacks, from caudal.
- `embalse/scores.py`: fair ensemble CRPS, from caudal.
- `scripts/train.py`: train the regional model and score it on water years 2016-2025 against persistence and climatology.

Environment: conda env `embalse` (`conda create -n embalse python=3.11`, then `pip install -e '.[dev]'`), with
`../inflow-ledger/data/cdec_inflow/` present (run `python score.py` there, or check out its `data` branch).

    python -m embalse.forcing            # fetch or extend data/forcing/<lid>.csv for every point
    python scripts/train.py --smoke      # 2 points, 1 epoch
    python scripts/train.py              # seed 0, 30 epochs; writes data/models/ and data/results/
    python -m pytest tests
    jupyter lab notebooks/walkthrough.ipynb   # the whole thing, end to end: data, frames, model, skill, calibration, the ledger, the paper

Not yet here: the daily issue script that writes quantile forecasts into inflow-ledger, conformal calibration,
AR-day scoring, basin-mean forcing.
