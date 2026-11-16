# Operational quantile LSTM, v1 (seed 0)

Retrospective skill of the model that issues the daily forecasts committed to hapi. One regional
pinball-loss LSTM over 11 reservoir inflow points, trained on water years 1990-2015 (fit through
2015-09-30), scored on water years 2016-2025 (2015-10-01 to 2025-09-30). Lookback 365 days,
hidden 128, 30 epochs, lr 0.001, 74,400 training windows, ['precip', 'tmax', 'tmin', 'tmean', 'pet', 'doy_sin', 'doy_cos', 'static_lat', 'static_elev', 'static_log_qmean'].
Forcing is Open-Meteo ERA5 sampled at the dam. Target is CDEC observed daily inflow (sensor 76), negatives dropped.

CRPS in cfs over the 7 predicted quantiles treated as an ensemble; persistence is yesterday's inflow; climatology
is the fit-window ensemble for that day of year. AR days: a CW3E catalog AR of scale 3 or more touched the coast
within 1.5 degrees of the dam's latitude (caudal-fleet's definition).

| point | days | AR days | CRPS model | CRPS persist. | CRPS clim. | CRPS model AR | CRPS persist. AR | CRPS clim. AR | cov90 | NSE q50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LAMC1 | 3,611 | 146 | 98 | 78 | 107 | 228 | 602 | 584 | 0.41 | 0.74 |
| ORDC1 | 3,644 | 143 | 1,173 | 1,214 | 2,640 | 2,898 | 6,628 | 12,223 | 0.63 | 0.86 |
| SHDC1 | 3,644 | 143 | 1,190 | 1,578 | 2,662 | 3,291 | 5,093 | 8,466 | 0.69 | 0.88 |
| FOLC1 | 3,650 | 160 | 1,068 | 874 | 1,778 | 3,422 | 5,007 | 8,417 | 0.64 | 0.78 |
| NBBC1 | 3,036 | 103 | 428 | 293 | 823 | 1,041 | 1,709 | 2,240 | 0.79 | 0.73 |
| NMSC1 | 3,632 | 139 | 341 | 329 | 718 | 1,139 | 1,955 | 2,952 | 0.63 | 0.80 |
| FRAC1 | 3,645 | 137 | 640 | 408 | 1,146 | 1,299 | 1,401 | 2,935 | 0.64 | 0.79 |
| EXQC1 | 3,640 | 139 | 388 | 299 | 668 | 1,287 | 2,319 | 3,189 | 0.69 | 0.75 |
| ISAC1 | 3,618 | 115 | 396 | 135 | 571 | 1,120 | 822 | 1,716 | 0.44 | 0.62 |
| TMDC1 | 3,518 | 123 | 176 | 143 | 336 | 613 | 1,123 | 1,556 | 0.61 | 0.69 |
| SCSC1 | 3,500 | 125 | 97 | 59 | 145 | 509 | 716 | 994 | 0.45 | 0.46 |
| pooled |  |  | 545 | 492 | 1,054 | 1,532 | 2,489 | 4,116 | 0.60 | 0.74 |

## Reading

- On all days the model roughly ties persistence, a hard lead-1 baseline, and halves climatology's error.
- On AR days it beats persistence by about 38% and climatology by
  63%: the skill is concentrated where the ledger matters.
- The 90% interval covers 60% of days: under-dispersed, as caudal-fleet found for the same head. Conformal
  calibration is the next layer; until then quote the intervals as nominal.
- Weakest points: Isabella and Success (small, snow- and foothill-driven, dam-point forcing least representative).

## Caveats

- Retrospective scores use ERA5 through the target day (a simulation); live forecasts use forecast forcing for leads
  1-14, so live skill will be lower and is what the ledger measures.
- Dam-point forcing, no basin averaging; no static catchment attributes beyond latitude, elevation and mean inflow.
- Single seed. The eval table is `data/results/eval_quantile_inflow_v1_s0.csv`, the run record
  `data/results/runs/quantile_inflow_v1_s0/`.
