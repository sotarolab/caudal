# embalse

Private ML package behind the public [inflow-ledger](../inflow-ledger): models, training, calibration and
weights for daily river-flow and reservoir-inflow forecasts at CNRFC points. Forecasts it issues are
committed to inflow-ledger before the outcome exists and scored there against observations, HEFS,
persistence and climatology.

- `embalse/points.py`: the forecast points and how each treats regulation (`inflow`, `natural`, `handoff`).
- `embalse/models/quantile_lstm.py`: pinball-loss LSTM with a monotone quantile head, from caudal.
- `embalse/models/sequence.py`: gap-aware sliding windows over daily frames, from caudal.
- `embalse/scores.py`: fair ensemble CRPS, from caudal.

    python -m pytest tests

Not yet here: forcing (Open-Meteo archive and forecast), training data for the points, conformal
calibration, the daily inference job.
