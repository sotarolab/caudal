# caudal

ML package behind the public [hapi](../hapi): models, training, calibration and
weights for daily river-flow and reservoir-inflow forecasts at CNRFC points. Forecasts it issues are
committed to hapi before the outcome exists and scored there against observations, HEFS,
persistence and climatology.

- `caudal/points.py`: the forecast points, their CDEC station and dam coordinates, and how each treats
  regulation (`inflow`, `natural`, `handoff`).
- `caudal/forcing.py`: Open-Meteo daily forcing, ERA5 archive for history and the forecast API for the next 16 days.
- `caudal/frames.py`: training frames and the pooled normalization, caudal-fleet's regional protocol.
- `caudal/models/`: pinball-loss LSTM with a monotone quantile head and gap-aware windows, from caudal-fleet.
- `caudal/training/`: run artifacts (config, metrics.csv, manifest, checkpoints) and callbacks, from caudal-fleet.
- `caudal/scores.py`: fair ensemble CRPS, from caudal-fleet.
- `scripts/train.py`: train the regional model and score it on water years 2016-2025 against persistence and climatology.

Environment: conda env `caudal` (`conda create -n caudal python=3.11`, then `pip install -e '.[dev]'`), with
`../hapi/data/cdec_inflow/` present (run `python score.py` there, or check out its `data` branch).

    python -m caudal.forcing            # fetch or extend data/forcing/<lid>.csv for every point
    python scripts/train.py --smoke      # 2 points, 1 epoch
    python scripts/train.py              # seed 0, 30 epochs; writes data/models/ and data/results/
    python -m pytest tests
    jupyter lab notebooks/walkthrough.ipynb   # the whole thing, end to end: data, frames, model, skill, calibration, the ledger, the paper

Not yet here: the daily issue script that writes quantile forecasts into hapi, conformal calibration,
AR-day scoring, basin-mean forcing.
