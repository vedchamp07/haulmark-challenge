# Hackathon Repository Summary (HaulMark Dumper Fuel Consumption)

This repository contains multiple iterations of a hackathon solution for predicting dumper fuel consumption from high-frequency telemetry, with optional spatial context and a mix of “clean” (no leakage) and “ultra-optimized” (high OOF but leakage-prone) approaches.

## What this repo is for

- Predict fuel consumption for unseen time windows from telemetry.
- Engineer features that aggregate noisy, irregular telemetry into stable daily/shift summaries.
- Train models with vehicle-aware validation (GroupKFold by vehicle) and generate Kaggle-style submissions.
- Produce diagnostics (OOF plots, feature importances, route benchmarks).

If you’re new to this codebase, start with:

- Strategy overview: [docs/overview.txt](docs/overview.txt)
- Dataset details: [docs/about_dataset.txt](docs/about_dataset.txt)
- “What to submit” snapshot: [docs/FINAL_SUBMISSION_GUIDE.md](docs/FINAL_SUBMISSION_GUIDE.md)
- Leakage post-mortem: [reports/fix_submission_summary.md](reports/fix_submission_summary.md)

## Hackathon strategy (high level)

### 1) Data understanding + split logic

- The dataset has very high-frequency telemetry (10–60s typical) with GPS, motion, and device signals.
- “Operational day” handling matters: per the dataset notes, a “day” is anchored around night-shift boundaries (see [docs/about_dataset.txt](docs/about_dataset.txt)).
- The code standardizes timestamps, drops duplicates, filters GPS quality, bounds speed, and interpolates altitude in [utils/preprocessing.py](utils/preprocessing.py).

### 2) Feature engineering (clean daily pipeline)

The main “clean” pipeline (used by the integrated predictor) builds daily features from telemetry only:

- Cleans telemetry: [utils/preprocessing.py](utils/preprocessing.py)
- Splits train/test telemetry files by date windows: [utils/data_loader.py](utils/data_loader.py)
- Aggregates to daily features + behavior fractions + cycle features: [utils/feature_eng.py](utils/feature_eng.py)
- Optional spatial flags (dump/load/haul road zones) when enabled: [utils/spatial_utils.py](utils/spatial_utils.py)

Important note: spatial features require optional dependencies (notably `geopandas`). The code only imports spatial logic when `--enable_spatial` is used.

### 3) Modeling and validation

The modeling stack in [utils/modeling.py](utils/modeling.py) is designed to:

- Train LightGBM with GroupKFold (grouped by vehicle) and save fold metrics.
- Optionally train XGBoost (if installed) and blend OOF predictions by searching a best weight.
- Save evaluation artifacts via [utils/evaluation.py](utils/evaluation.py):
  - OOF predictions CSV
  - Actual-vs-pred scatter plot
  - Feature importances CSV + a top-30 chart
  - Route benchmark + “dumper efficiency” residual stats

### 4) Submission building

The integrated predictor:

- Generates predictions for the test period.
- Joins against an `id_mapping.csv` to build the final submission IDs.
- Fills missing predictions with vehicle historical means (and global mean fallback).

See [scripts/predict.py](scripts/predict.py).

## Important warning: leakage vs clean features

This repo includes an “ultra-optimized” track that used summary-derived fields (tank levels, refuels) which can leak target information if those fields are not present for the test period.

- The documents in [reports/fix_submission_summary.md](reports/fix_submission_summary.md) explain why an extremely strong OOF score can still fail on the leaderboard.
- Treat the “enhanced features” + some ensemble submissions as hackathon experiments, not necessarily valid for a strict no-leakage test.

## How to run (entrypoints)

### A) Integrated clean pipeline (recommended starting point)

Run end-to-end feature building → training → submission:

- Command: `python scripts/predict.py --data_dir data --output outputs/daywise__predict.csv --output_dir outputs`
- Optional spatial flags: add `--enable_spatial` (requires `geopandas`, `pyproj`, and their native deps)

Expected data layout for `--data_dir data`:

- Telemetry files named like `telemetry_YYYY-MM-DD_YYYY-MM-DD.(csv|parquet)`
- `id_mapping.csv` (required for submission IDs)
- Optional: `fleet.csv`, `*.gpkg`, refuel files

Note: the daily pipeline (`scripts/predict.py`) writes a day-wise prediction CSV. This repo keeps only shift-wise artifacts under [submissions/](submissions/), so the example output goes to [outputs/](outputs/).

### B) Shift-wise / “new_data” feature pipelines (older hackathon tracks)

These scripts generate and consume a separate `new_data/` directory of engineered CSVs:

- Clean shift features (telemetry-only): [scripts/build_clean_features.py](scripts/build_clean_features.py)
- Enhanced features (includes summary-derived fields; can be leakage-prone): [scripts/build_enhanced_features.py](scripts/build_enhanced_features.py)

Trainers that read from `new_data/` and write submissions to [submissions/](submissions/):

- Clean model: [scripts/train_clean.py](scripts/train_clean.py)
- Improved clean model: [scripts/train_improved_clean.py](scripts/train_improved_clean.py)
- Ensemble model: [scripts/train_ensemble.py](scripts/train_ensemble.py)
- Stacked model: [scripts/train_stacked.py](scripts/train_stacked.py)
- Simple clean baseline: [scripts/train_simple_clean.py](scripts/train_simple_clean.py)

Blend/aggregate submissions:

- [scripts/blend_submissions.py](scripts/blend_submissions.py)

Hyperparameter search:

- [scripts/optimize_hyperparams.py](scripts/optimize_hyperparams.py) (writes best params to [outputs/](outputs/))

## Folder-by-folder map

### assets/

- Contains static assets like the challenge PDF in [assets/](assets/) (HaulMark_Challenge-2 (2).pdf)

### data/

- Primary location for the full competition data when running the integrated pipeline.
- [data/](data/) contains the competition inputs used by pipelines in this repo:
  - Fleet: [data/fleet.csv](data/fleet.csv)
  - ID mappings: [data/id_mapping.csv](data/id_mapping.csv), [data/id_mapping_new.csv](data/id_mapping_new.csv)
  - Telemetry windows: `data/telemetry_*.parquet`
  - Refuels: `data/rfid_refuels_*.parquet`
  - Summary (train-only): `data/smry_*_train_ordered.csv`
  - Anonymized mine GeoPackages: [data/mine_001_anonymized.gpkg](data/mine_001_anonymized.gpkg), [data/mine_002_anonymized.gpkg](data/mine_002_anonymized.gpkg)
  - Telemetry sample: [data/sample_telemetry.csv](data/sample_telemetry.csv)

### docs/

- Problem context and “what to submit” guide:
  - [docs/overview.txt](docs/overview.txt)
  - [docs/about_dataset.txt](docs/about_dataset.txt)
  - [docs/FINAL_SUBMISSION_GUIDE.md](docs/FINAL_SUBMISSION_GUIDE.md)

### reports/

- Writeups and analysis notes:
  - [reports/OPTIMIZATION_REPORT.md](reports/OPTIMIZATION_REPORT.md)
  - [reports/SHIFT_WISE_SUMMARY.md](reports/SHIFT_WISE_SUMMARY.md)
  - [reports/fix_submission_summary.md](reports/fix_submission_summary.md)

### scripts/

Runnable scripts (feature engineering, training, blending, prediction):

- Integrated daily pipeline: [scripts/predict.py](scripts/predict.py)
- Shift-wise feature builders: [scripts/build_clean_features.py](scripts/build_clean_features.py), [scripts/build_enhanced_features.py](scripts/build_enhanced_features.py)
- Model training: [scripts/train_clean.py](scripts/train_clean.py), [scripts/train_improved_clean.py](scripts/train_improved_clean.py), [scripts/train_simple_clean.py](scripts/train_simple_clean.py), [scripts/train_ensemble.py](scripts/train_ensemble.py), [scripts/train_stacked.py](scripts/train_stacked.py)
- Submission blending: [scripts/blend_submissions.py](scripts/blend_submissions.py)
- Hyperparameter search: [scripts/optimize_hyperparams.py](scripts/optimize_hyperparams.py)

### utils/

Core library code used by the integrated pipeline:

- Data discovery + splitting: [utils/data_loader.py](utils/data_loader.py)
- Cleaning + operational day logic: [utils/preprocessing.py](utils/preprocessing.py)
- Feature engineering (daily aggregates, cycle features, optional spatial): [utils/feature_eng.py](utils/feature_eng.py)
- Spatial operations (optional deps): [utils/spatial_utils.py](utils/spatial_utils.py)
- Modeling + training orchestration: [utils/modeling.py](utils/modeling.py)
- Metrics + plots: [utils/evaluation.py](utils/evaluation.py)

### notebooks/

Exploration and starter notebooks:

- EDA and modeling notebooks: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb), [notebooks/02_features.ipynb](notebooks/02_features.ipynb), [notebooks/03_modeling.ipynb](notebooks/03_modeling.ipynb)
- Starter notebook: [notebooks/sample-starter-ipynb.ipynb](notebooks/sample-starter-ipynb.ipynb)

### outputs/

Generated diagnostics and artifacts from training runs, such as:

- Fold metrics: [outputs/fold_rmse.csv](outputs/fold_rmse.csv)
- OOF diagnostics: [outputs/oof_predictions.csv](outputs/oof_predictions.csv), [outputs/oof_actual_vs_pred.png](outputs/oof_actual_vs_pred.png)
- Feature importances: [outputs/feature_importance.csv](outputs/feature_importance.csv), [outputs/feature_importance_top30.png](outputs/feature_importance_top30.png)
- Route and efficiency benchmarking: [outputs/route_benchmark.csv](outputs/route_benchmark.csv), [outputs/dumper_efficiency.csv](outputs/dumper_efficiency.csv)

### submissions/

Historical and current submission CSVs (and comparisons):

- Shift-wise (current):
  - [submissions/shiftwise**baseline_lgb**oof46.26**mean161.90**std92.04.csv](submissions/shiftwise__baseline_lgb__oof46.26__mean161.90__std92.04.csv)
  - Archived (non-recommended / historical):
    - [submissions/archive/shiftwise**final_clean_copy**mean161.90\_\_std92.04.csv](submissions/archive/shiftwise__final_clean_copy__mean161.90__std92.04.csv)
    - [submissions/archive/shiftwise**naive_constant**mean173.40.csv](submissions/archive/shiftwise__naive_constant__mean173.40.csv)
    - [submissions/archive/shiftwise**blend_ens0.5_stack0.3_base0.2**mean146.53\_\_std72.92.csv](submissions/archive/shiftwise__blend_ens0.5_stack0.3_base0.2__mean146.53__std72.92.csv)
    - [submissions/archive/shiftwise**ensemble_3model**oof16.79**mean140.38**std75.70\_\_leaky.csv](submissions/archive/shiftwise__ensemble_3model__oof16.79__mean140.38__std75.70__leaky.csv)
    - [submissions/archive/shiftwise**best_ensemble_copy**mean140.38**std75.70**leaky.csv](submissions/archive/shiftwise__best_ensemble_copy__mean140.38__std75.70__leaky.csv)
    - [submissions/archive/shiftwise\_\_comparison.txt](submissions/archive/shiftwise__comparison.txt)

### logs/

Run logs for training/optimization experiments:

- [logs/archive/ensemble_training.log](logs/archive/ensemble_training.log)
- [logs/archive/optuna_log.txt](logs/archive/optuna_log.txt)
- [logs/archive/stacking_log.txt](logs/archive/stacking_log.txt)

### catboost_info/

CatBoost training artifacts (auto-generated by CatBoost).

### eda/

Scratch space for exploration (currently empty / not tracked in detail here).

### .venv/ and .venv-1/

Local Python virtual environments (machine-specific).
