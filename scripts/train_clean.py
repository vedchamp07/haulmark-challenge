#!/usr/bin/env python3
"""Train model on CLEAN features - NO LEAKAGE"""
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


def get_feature_columns(df, exclude_cols):
    """Get all valid feature columns"""
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

    # Load clean data
    print("Loading CLEAN features (no leakage)...")
    train = pd.read_csv(data_dir / 'train_clean.csv')
    test = pd.read_csv(data_dir / 'test_clean.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # Define features
    exclude_cols = {
        'vehicle', 'mine', 'date', 'shift', 'acons',
        'date', 'ts', 'ts_first', 'ts_last'
    }

    features = get_feature_columns(train, exclude_cols)
    print(f"\nUsing {len(features)} CLEAN features (no leakage)")
    print("\nFeatures:", features[:20])

    # Prepare data
    train = train.dropna(subset=['acons']).copy()
    X = train[features].fillna(0)
    y = train['acons'].values
    groups = train['vehicle'].astype('category').cat.codes.values

    # Cross-validation
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y))
    models = []

    # Model parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 8,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.5,
        'verbosity': -1,
        'seed': 42,
    }

    print("\n" + "="*60)
    print("CROSS-VALIDATION (CLEAN FEATURES)")
    print("="*60)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n[Fold {fold}/5]")

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
                lgb.log_evaluation(period=100)
            ]
        )

        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        models.append(model)

        fold_rmse = rmse(y_val, np.clip(oof_preds[val_idx], 0, None))
        print(f"[Fold {fold}] RMSE: {fold_rmse:.2f}")

    oof_preds = np.clip(oof_preds, 0, None)
    overall_rmse = rmse(y, oof_preds)

    print("\n" + "="*60)
    print(f"OOF RMSE (CLEAN): {overall_rmse:.2f}")
    print("="*60)

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': np.mean([m.feature_importance('gain') for m in models], axis=0)
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Features:")
    print(importance.head(20).to_string(index=False))

    # Predict on test
    print("\nGenerating test predictions...")
    X_test = test[features].fillna(0)

    test_preds = np.mean([
        m.predict(X_test, num_iteration=m.best_iteration)
        for m in models
    ], axis=0)
    test_preds = np.clip(test_preds, 0, None)

    # Create submission
    id_mapping = pd.read_csv(data_dir / 'id_mapping_new.csv')
    id_mapping['date'] = pd.to_datetime(id_mapping['date']).dt.date

    test['date'] = pd.to_datetime(test['date']).dt.date
    test['Predicted'] = test_preds

    submission = id_mapping.merge(
        test[['vehicle', 'date', 'shift', 'Predicted']],
        on=['vehicle', 'date', 'shift'],
        how='left'
    )

    # Fallback for missing predictions (use historical mean by vehicle+shift)
    if 'veh_shift_mean_cons' in test.columns:
        for idx, row in submission[submission['Predicted'].isna()].iterrows():
            # Use vehicle-shift historical mean as fallback
            veh_shift_mask = (test['vehicle'] == row['vehicle']) & (test['shift'] == row['shift'])
            if 'veh_shift_mean_cons' in test.columns and veh_shift_mask.any():
                fallback = test.loc[veh_shift_mask, 'veh_shift_mean_cons'].iloc[0]
            else:
                fallback = train['acons'].mean()
            submission.at[idx, 'Predicted'] = fallback

    submission['Predicted'] = submission['Predicted'].clip(lower=0)

    # Save
    final_submission = submission[['id', 'Predicted']].sort_values('id')
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    mean = float(final_submission['Predicted'].mean())
    std = float(final_submission['Predicted'].std())
    out_path = SUBMISSIONS_DIR / f"shiftwise__baseline_lgb__oof{overall_rmse:.2f}__mean{mean:.2f}__std{std:.2f}.csv"
    final_submission.to_csv(out_path, index=False)

    print("\n" + "="*60)
    print("SUBMISSION STATISTICS (CLEAN)")
    print("="*60)
    print(f"Submission rows: {len(final_submission)}")
    print(f"Expected rows:   {len(id_mapping)}")
    print(f"Match rate:      {len(final_submission) / len(id_mapping) * 100:.1f}%")
    print(f"\nPrediction stats:")
    print(final_submission['Predicted'].describe())
    print(f"\n✓ Saved: {out_path}")
    print("="*60)

    # Compare to previous submission
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Previous (leaky) OOF RMSE:  16.79L")
    print(f"Current (clean) OOF RMSE:   {overall_rmse:.2f}L")
    print("\nNote: Clean version has higher OOF but will generalize better!")
    print("="*60)


if __name__ == '__main__':
    main()
