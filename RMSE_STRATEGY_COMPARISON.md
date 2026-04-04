# RMSE Strategy Comparison (Existing Runs)

This repository contains both **daily** and **shift-wise** modeling tracks.

- **Daily track** artifacts live mostly under [outputs/](outputs/) and are produced by [scripts/predict.py](scripts/predict.py).
- **Shift-wise track** artifacts (the ones intended for submission) are curated under [submissions/](submissions/).

## Shift-wise strategies (curated in `submissions/`)

| Strategy                             | Leakage-safe? | OOF RMSE (liters)  | Public LB RMSE     | Artifact                                                                                                                                                                                 |
| ------------------------------------ | ------------- | ------------------ | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LightGBM baseline (telemetry-only)   | Yes           | 46.26              | 3551.50543         | [submissions/shiftwise**baseline_lgb**oof46.26**mean161.90**std92.04.csv](submissions/shiftwise__baseline_lgb__oof46.26__mean161.90__std92.04.csv)                                       |
| Clean baseline (copy)                | Yes           | (same as baseline) | (same as baseline) | [submissions/archive/shiftwise**final_clean_copy**mean161.90\_\_std92.04.csv](submissions/archive/shiftwise__final_clean_copy__mean161.90__std92.04.csv)                                 |
| Naive constant (173.4 L)             | Yes           | 97.07 (reported)   | —                  | [submissions/archive/shiftwise**naive_constant**mean173.40.csv](submissions/archive/shiftwise__naive_constant__mean173.40.csv)                                                           |
| 3-model ensemble (enhanced features) | No (leaky)    | 16.79              | ~10k (reported)    | [submissions/archive/shiftwise**ensemble_3model**oof16.79**mean140.38**std75.70\_\_leaky.csv](submissions/archive/shiftwise__ensemble_3model__oof16.79__mean140.38__std75.70__leaky.csv) |
| Best-ensemble copy                   | No (leaky)    | (same as ensemble) | (same as ensemble) | [submissions/archive/shiftwise**best_ensemble_copy**mean140.38**std75.70**leaky.csv](submissions/archive/shiftwise__best_ensemble_copy__mean140.38__std75.70__leaky.csv)                 |
| Blend (contains leaky components)    | No (likely)   | —                  | —                  | [submissions/archive/shiftwise**blend_ens0.5_stack0.3_base0.2**mean146.53\_\_std72.92.csv](submissions/archive/shiftwise__blend_ens0.5_stack0.3_base0.2__mean146.53__std72.92.csv)       |

Notes:

- The leakage callout and the LB RMSE numbers above come from [reports/fix_submission_summary.md](reports/fix_submission_summary.md).
- A quick comparison writeup is also in [submissions/archive/shiftwise\_\_comparison.txt](submissions/archive/shiftwise__comparison.txt).

## Daily track (historical)

The daily pipeline’s OOF metrics are stored under [outputs/fold_rmse.csv](outputs/fold_rmse.csv) and [outputs/time_validation_rmse.csv](outputs/time_validation_rmse.csv).

- 5-fold RMSE range (daily): ~152–188
- Time validation RMSE (daily): ~168 (Jan), ~201 (Feb)

## What to submit today

- Recommended: [submissions/shiftwise**baseline_lgb**oof46.26**mean161.90**std92.04.csv](submissions/shiftwise__baseline_lgb__oof46.26__mean161.90__std92.04.csv)
- Avoid: artifacts marked `__leaky` (they can look great in OOF but fail on the leaderboard).

# New strategies to win (using the _current_ `data/` contents)

The current [data/](data/) directory includes additional competition files that can be leveraged **without** using train-only summary fields as direct features.

## Newly available / relevant files

- Telemetry windows (train + test periods): `data/telemetry_*.parquet`
- Refuel transactions spanning into test: `data/rfid_refuels_*.parquet`
- Mine geometry: [data/mine_001_anonymized.gpkg](data/mine_001_anonymized.gpkg), [data/mine_002_anonymized.gpkg](data/mine_002_anonymized.gpkg)
- Fleet and mappings: [data/fleet.csv](data/fleet.csv), [data/id_mapping.csv](data/id_mapping.csv), [data/id_mapping_new.csv](data/id_mapping_new.csv)
- Train-only summary (treat as labels/diagnostics, not test-time features): `data/smry_*_train_ordered.csv`

## Strategy ideas (leakage-safe)

### 1) Refuel-aware shift features (from `rfid_refuels_*.parquet`)

Build shift-level features from transaction logs that exist across _both_ train and test:

- `refuel_volume_this_shift`, `refuel_count_this_shift`
- `minutes_since_last_refuel`, `refuel_volume_last_24h`
- `refuel_flag` (binary) to help model “consumption spikes” around refuel events

This replaces the leaky `arefill/initlev/endlev` summary-derived signal with a test-available proxy.

### 2) Spatial difficulty / route proxies (from `.gpkg` + telemetry)

Use mine geometries to derive consistent per-shift route features:

- Zone membership ratios (time near loading / dumping / haul roads)
- Elevation-change proxies already in telemetry features + _where_ they happen
- Repeated route clustering per vehicle (e.g., cluster start/end areas using coordinates)

Keep these as **telemetry-only** or **telemetry+geometry** features so they exist at inference.

### 3) Causal lag features computed within each vehicle

Lag features can be safe if computed strictly from _past_ shifts for the same vehicle:

- `acons_lag1`, `acons_rolling_mean_3` (training-only target-derived features are OK if built properly in CV)
- At inference, replace target-lags with **prediction-lags** (recursive) or telemetry-lags:
  - lag telemetry aggregates (e.g., `shift_km_lag1`, `moving_hours_lag1`)

Key requirement: in CV, compute lags within each fold so validation rows never see future information.

### 4) Validation that matches the leaderboard split

The leakage report strongly suggests train/test differ by time and available fields.

- Prefer **time-based validation** (train on earlier windows → validate on later windows) in addition to GroupKFold-by-vehicle.
- Track performance by month window (Jan/Feb/Mar) to detect drift.

### 5) Post-processing constraints

Use lightweight constraints that don’t leak:

- Clip per-vehicle predictions to historical quantiles (e.g., 1st–99th) computed from training shifts.
- Calibrate per-shift distributions (mean/std) using training only.

## Strategy ideas (diagnostics / train-only)

You _can_ still use `smry_*_train_ordered.csv` to:

- audit label quality
- define better evaluation splits
- sanity-check feature engineering

…but avoid using its target-adjacent fields as direct features unless you can reproduce them for test.
