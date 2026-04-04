#!/usr/bin/env python3
"""Improved model on CLEAN features + safe enhancements"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = ROOT / "submissions"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def add_safe_features(df):
    """Add derived features that are SAFE (no leakage)"""
    # Efficiency ratios
    df['km_per_moving_hour'] = df['shift_km'] / (df['moving_hours'] + 1e-6)
    df['moving_vs_ignition_ratio'] = df['moving_hours'] / (df['ignition_on_hours'] + 1e-6)

    # Stop behavior
    df['stops_per_km'] = df['stop_count'] / (df['shift_km'] + 1e-6)
    df['stops_per_hour'] = df['stop_count'] / (df['ignition_on_hours'] + 1e-6)

    # Dump efficiency
    df['km_per_dump'] = df['shift_km'] / (df['dump_event_count'] + 1)
    df['dumps_per_hour'] = df['dump_event_count'] / (df['ignition_on_hours'] + 1e-6)

    # Altitude work
    df['altitude_work'] = df['total_climb_m'] * df['shift_km']
    df['climb_ratio'] = df['total_climb_m'] / (df['total_descent_m'] + 1.0)

    # Speed patterns
    df['speed_efficiency'] = df['speed_mean'] / (df['speed_max'] + 1e-6)

    return df


def add_vehicle_stats(train_df, test_df):
    """Add vehicle historical stats computed from TRAINING set only"""
    # Compute vehicle-level stats from training data
    veh_stats = train_df.groupby('vehicle').agg(
        veh_mean_km=('shift_km', 'mean'),
        veh_std_km=('shift_km', 'std'),
        veh_mean_moving_h=('moving_hours', 'mean'),
        veh_mean_dumps=('dump_event_count', 'mean'),
        veh_mean_idle_frac=('idle_fraction', 'mean'),
    ).reset_index()

    # Vehicle-shift specific stats
    if 'acons' in train_df.columns:
        veh_shift_stats = train_df.groupby(['vehicle', 'shift']).agg(
            veh_shift_mean_cons=('acons', 'mean'),
            veh_shift_std_cons=('acons', 'std'),
            veh_shift_mean_km=('shift_km', 'mean'),
        ).reset_index()
    else:
        veh_shift_stats = None

    # Merge
    train_merged = train_df.merge(veh_stats, on='vehicle', how='left')
    test_merged = test_df.merge(veh_stats, on='vehicle', how='left')

    if veh_shift_stats is not None:
        train_merged = train_merged.merge(veh_shift_stats, on=['vehicle', 'shift'], how='left')
        test_merged = test_merged.merge(veh_shift_stats, on=['vehicle', 'shift'], how='left')

    return train_merged, test_merged


def add_temporal_features(df):
    """Add temporal features"""
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['day_of_month'] = df['date'].dt.day

    # Shift encoding
    shift_map = {'C': 0, 'A': 1, 'B': 2}
    df['shift_enc'] = df['shift'].map(shift_map)

    return df


def get_feature_columns(df, exclude_cols):
    """Get all numeric feature columns"""
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


def main():
    data_dir = ROOT / 'new_data'

    print("="*60)
    print("LOADING CLEAN BASELINE FEATURES")
    print("="*60)

    # Load original clean features
    train = pd.read_csv(data_dir / 'train_shift_features.csv')
    test = pd.read_csv(data_dir / 'test_shift_features.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # Load ground truth targets
    summary = pd.concat([
        pd.read_csv(data_dir / 'smry_jan_train_ordered.csv'),
        pd.read_csv(data_dir / 'smry_feb_train_ordered.csv'),
        pd.read_csv(data_dir / 'smry_mar_train_ordered.csv'),
    ])
    summary['date'] = pd.to_datetime(summary['date']).dt.date

    # Merge targets
    summary_targets = summary[['vehicle', 'date', 'shift', 'acons']].copy()
    train['date'] = pd.to_datetime(train['date']).dt.date
    test['date'] = pd.to_datetime(test['date']).dt.date

    train = train.merge(summary_targets, on=['vehicle', 'date', 'shift'], how='left')

    print(f"\nTargets merged: {train['acons'].notna().sum()}/{len(train)}")

    # Add safe features
    print("\nAdding safe derived features...")
    train = add_safe_features(train)
    test = add_safe_features(test)

    # Add temporal features
    print("Adding temporal features...")
    train = add_temporal_features(train)
    test = add_temporal_features(test)

    # Add vehicle stats (from training only)
    print("Adding vehicle statistics...")
    train, test = add_vehicle_stats(train, test)

    print(f"\nEnhanced train shape: {train.shape}")
    print(f"Enhanced test shape: {test.shape}")

    # Define features
    exclude_cols = {
        'vehicle', 'mine', 'date', 'shift', 'acons',
    }

    features = get_feature_columns(train, exclude_cols)
    print(f"\nUsing {len(features)} features")

    # Prepare data
    train_clean = train.dropna(subset=['acons']).copy()
    X = train_clean[features].fillna(0).replace([np.inf, -np.inf], 0)
    y = train_clean['acons'].values
    groups = train_clean['vehicle'].astype('category').cat.codes.values

    print(f"\nTraining samples: {len(X)}")
    print(f"Target mean: {y.mean():.2f}L")
    print(f"Target std: {y.std():.2f}L")

    # Cross-validation
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y))
    models = []

    # Optimized parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 40,
        'max_depth': 8,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_gain_to_split': 0.01,
        'verbosity': -1,
        'seed': 42,
    }

    print("\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION")
    print("="*60)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n[Fold {fold}/5] Training...")

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
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=200)
            ]
        )

        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        models.append(model)

        fold_rmse = rmse(y_val, np.clip(oof_preds[val_idx], 0, None))
        print(f"[Fold {fold}] Val RMSE: {fold_rmse:.2f}L")

    oof_preds = np.clip(oof_preds, 0, None)
    overall_rmse = rmse(y, oof_preds)

    print("\n" + "="*60)
    print(f"OVERALL OOF RMSE: {overall_rmse:.2f}L")
    print("="*60)

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': np.mean([m.feature_importance('gain') for m in models], axis=0)
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Features:")
    print(importance.head(15).to_string(index=False))

    # Predict on test
    print("\nGenerating test predictions...")
    X_test = test[features].fillna(0).replace([np.inf, -np.inf], 0)

    test_preds = np.mean([
        m.predict(X_test, num_iteration=m.best_iteration)
        for m in models
    ], axis=0)
    test_preds = np.clip(test_preds, 0, None)

    # Create submission
    id_mapping = pd.read_csv(data_dir / 'id_mapping_new.csv')
    id_mapping['date'] = pd.to_datetime(id_mapping['date']).dt.date

    test['Predicted'] = test_preds

    submission = id_mapping.merge(
        test[['vehicle', 'date', 'shift', 'Predicted']],
        on=['vehicle', 'date', 'shift'],
        how='left'
    )

    # Fallback for missing predictions
    if 'veh_shift_mean_cons' in test.columns:
        for idx, row in submission[submission['Predicted'].isna()].iterrows():
            veh_shift_mask = (test['vehicle'] == row['vehicle']) & (test['shift'] == row['shift'])
            if veh_shift_mask.any() and 'veh_shift_mean_cons' in test.columns:
                fallback = test.loc[veh_shift_mask, 'veh_shift_mean_cons'].iloc[0]
                if pd.notna(fallback):
                    submission.at[idx, 'Predicted'] = fallback

    # Final fallback: global mean
    global_mean = train_clean['acons'].mean()
    submission['Predicted'] = submission['Predicted'].fillna(global_mean)
    submission['Predicted'] = submission['Predicted'].clip(lower=0)

    # Save
    final_submission = submission[['id', 'Predicted']].sort_values('id')
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    mean = float(final_submission['Predicted'].mean())
    std = float(final_submission['Predicted'].std())
    out_path = SUBMISSIONS_DIR / f"shiftwise__improved_clean__mean{mean:.2f}__std{std:.2f}.csv"
    final_submission.to_csv(out_path, index=False)

    print("\n" + "="*60)
    print("SUBMISSION CREATED")
    print("="*60)
    print(f"Rows: {len(final_submission)}")
    print(f"Expected: {len(id_mapping)}")
    print(f"\nPrediction stats:")
    print(final_submission['Predicted'].describe())
    print(f"\n✓ Saved: {out_path}")
    print("="*60)


if __name__ == '__main__':
    main()
