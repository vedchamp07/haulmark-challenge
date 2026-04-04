#!/usr/bin/env python3
"""Hyperparameter optimization using Optuna"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_feature_columns(df, exclude_cols):
    features = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if df[c].dtype == 'object':
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        features.append(c)
    return features


def objective(trial, X, y, groups, n_splits=3):
    """Optuna objective function for LightGBM"""

    # Search space
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'seed': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 5.0),
    }

    # Cross-validate
    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y))

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )

        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

    oof_preds = np.clip(oof_preds, 0, None)
    cv_rmse = rmse(y, oof_preds)

    return cv_rmse


def main():
    data_dir = ROOT / 'new_data'

    # Load data
    print("Loading enhanced features...")
    train = pd.read_csv(data_dir / 'train_enhanced.csv')

    # Define features
    exclude_cols = {
        'vehicle', 'mine', 'date', 'shift', 'acons', 'lph', 'lph_summary',
        'veh_shift_interaction', 'date', 'ts', 'ts_first', 'ts_last'
    }
    features = get_feature_columns(train, exclude_cols)

    # Prepare data
    train = train.dropna(subset=['acons']).copy()
    X = train[features].fillna(0)
    y = train['acons'].values
    groups = train['vehicle'].astype('category').cat.codes.values

    print(f"Train shape: {X.shape}")
    print(f"Features: {len(features)}")

    # Optimize
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*60)

    study.optimize(
        lambda trial: objective(trial, X, y, groups, n_splits=3),
        n_trials=50,
        show_progress_bar=True
    )

    print("\n" + "="*60)
    print("BEST PARAMETERS")
    print("="*60)
    print(f"Best RMSE: {study.best_value:.2f}")
    print("\nBest params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*60)

    # Save best params
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df['best_rmse'] = study.best_value
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / 'best_hyperparams.csv'
    best_params_df.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")


if __name__ == '__main__':
    main()
