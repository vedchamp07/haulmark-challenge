#!/usr/bin/env python3
"""Ultra-optimized ensemble model for minimum RMSE"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = ROOT / "submissions"


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


def rmse(y_true, y_pred):
    """RMSE metric"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_lgb_model(X_train, y_train, groups, X_val, y_val, params):
    """Train LightGBM with early stopping"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=3000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    return model


def train_xgb_model(X_train, y_train, groups, X_val, y_val, params):
    """Train XGBoost with early stopping"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    return model


def train_catboost_model(X_train, y_train, groups, X_val, y_val, params):
    """Train CatBoost with early stopping"""
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    model = CatBoostRegressor(**params, verbose=100)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
    return model


def cross_validate_ensemble(df, features, target_col, n_splits=5):
    """Cross-validate ensemble of 3 models"""

    # Prepare data
    df = df.dropna(subset=[target_col]).copy()
    X = df[features].fillna(0)
    y = df[target_col].values
    groups = df['vehicle'].astype('category').cat.codes.values

    # Initialize CV
    gkf = GroupKFold(n_splits=n_splits)

    # Store OOF predictions
    oof_lgb = np.zeros(len(df))
    oof_xgb = np.zeros(len(df))
    oof_cat = np.zeros(len(df))

    models_lgb = []
    models_xgb = []
    models_cat = []

    # Model parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 63,
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

    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.01,
        'max_depth': 7,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
        'seed': 42,
    }

    cat_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.01,
        'depth': 8,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'iterations': 3000,
        'early_stopping_rounds': 100,
    }

    print("\n" + "="*60)
    print("CROSS-VALIDATION WITH ENSEMBLE")
    print("="*60)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*60}")

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        g_tr = groups[train_idx]

        # Train LightGBM
        print(f"\n[Fold {fold}] Training LightGBM...")
        lgb_model = train_lgb_model(X_tr, y_tr, g_tr, X_val, y_val, lgb_params)
        oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        models_lgb.append(lgb_model)

        # Train XGBoost
        print(f"\n[Fold {fold}] Training XGBoost...")
        xgb_model = train_xgb_model(X_tr, y_tr, g_tr, X_val, y_val, xgb_params)
        oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(X_val))
        models_xgb.append(xgb_model)

        # Train CatBoost
        print(f"\n[Fold {fold}] Training CatBoost...")
        cat_model = train_catboost_model(X_tr, y_tr, g_tr, X_val, y_val, cat_params)
        oof_cat[val_idx] = cat_model.predict(X_val)
        models_cat.append(cat_model)

        # Fold ensemble
        oof_ensemble_fold = (oof_lgb[val_idx] * 0.4 +
                             oof_xgb[val_idx] * 0.3 +
                             oof_cat[val_idx] * 0.3)

        fold_rmse = rmse(y_val, oof_ensemble_fold)
        print(f"\n[Fold {fold}] Ensemble RMSE: {fold_rmse:.2f}")

    # Overall OOF scores
    oof_lgb = np.clip(oof_lgb, 0, None)
    oof_xgb = np.clip(oof_xgb, 0, None)
    oof_cat = np.clip(oof_cat, 0, None)

    oof_ensemble = (oof_lgb * 0.4 + oof_xgb * 0.3 + oof_cat * 0.3)
    oof_ensemble = np.clip(oof_ensemble, 0, None)

    rmse_lgb = rmse(y, oof_lgb)
    rmse_xgb = rmse(y, oof_xgb)
    rmse_cat = rmse(y, oof_cat)
    rmse_ensemble = rmse(y, oof_ensemble)

    print("\n" + "="*60)
    print("OOF RESULTS")
    print("="*60)
    print(f"LightGBM RMSE:  {rmse_lgb:.2f}")
    print(f"XGBoost RMSE:   {rmse_xgb:.2f}")
    print(f"CatBoost RMSE:  {rmse_cat:.2f}")
    print(f"Ensemble RMSE:  {rmse_ensemble:.2f}")
    print("="*60)

    # Feature importance (from LightGBM)
    importance = pd.DataFrame({
        'feature': features,
        'importance': np.mean([m.feature_importance('gain') for m in models_lgb], axis=0)
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Features:")
    print(importance.head(20).to_string(index=False))

    return {
        'models_lgb': models_lgb,
        'models_xgb': models_xgb,
        'models_cat': models_cat,
        'oof_ensemble': oof_ensemble,
        'oof_lgb': oof_lgb,
        'oof_xgb': oof_xgb,
        'oof_cat': oof_cat,
        'importance': importance,
        'oof_rmse_lgb': rmse_lgb,
        'oof_rmse_xgb': rmse_xgb,
        'oof_rmse_cat': rmse_cat,
        'oof_rmse_ensemble': rmse_ensemble,
    }


def predict_ensemble(models, X_test, features):
    """Generate ensemble predictions"""
    X_test = X_test[features].fillna(0)

    # LightGBM predictions
    preds_lgb = np.mean([
        m.predict(X_test, num_iteration=m.best_iteration)
        for m in models['models_lgb']
    ], axis=0)

    # XGBoost predictions
    preds_xgb = np.mean([
        m.predict(xgb.DMatrix(X_test))
        for m in models['models_xgb']
    ], axis=0)

    # CatBoost predictions
    preds_cat = np.mean([
        m.predict(X_test)
        for m in models['models_cat']
    ], axis=0)

    # Ensemble (weighted average)
    preds_ensemble = (preds_lgb * 0.4 +
                      preds_xgb * 0.3 +
                      preds_cat * 0.3)

    return np.clip(preds_ensemble, 0, None)


def main():
    data_dir = ROOT / 'new_data'

    # Load data
    print("Loading enhanced features...")
    train = pd.read_csv(data_dir / 'train_enhanced.csv')
    test = pd.read_csv(data_dir / 'test_enhanced.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # Define features
    exclude_cols = {
        'vehicle', 'mine', 'date', 'shift', 'acons', 'lph', 'lph_summary',
        'veh_shift_interaction', 'date', 'ts', 'ts_first', 'ts_last'
    }

    features = get_feature_columns(train, exclude_cols)
    print(f"\nUsing {len(features)} features")

    # Train ensemble
    models = cross_validate_ensemble(
        train,
        features,
        target_col='acons',
        n_splits=5
    )

    # Predict on test
    print("\nGenerating test predictions...")
    test_preds = predict_ensemble(models, test, features)

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

    # Fallback for missing predictions
    vehicle_shift_means = train.groupby(['vehicle', 'shift'])['acons'].mean()
    for idx, row in submission[submission['Predicted'].isna()].iterrows():
        fallback = vehicle_shift_means.get((row['vehicle'], row['shift']), train['acons'].mean())
        submission.at[idx, 'Predicted'] = fallback

    submission['Predicted'] = submission['Predicted'].clip(lower=0)

    # Save
    final_submission = submission[['id', 'Predicted']].sort_values('id')
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    mean = float(final_submission['Predicted'].mean())
    std = float(final_submission['Predicted'].std())
    out_path = SUBMISSIONS_DIR / (
        f"shiftwise__ensemble_3model__oof{models['oof_rmse_ensemble']:.2f}__mean{mean:.2f}__std{std:.2f}__leaky.csv"
    )
    final_submission.to_csv(out_path, index=False)

    print("\n" + "="*60)
    print("SUBMISSION STATISTICS")
    print("="*60)
    print(f"Submission rows: {len(final_submission)}")
    print(f"Expected rows:   {len(id_mapping)}")
    print(f"Match rate:      {len(final_submission) / len(id_mapping) * 100:.1f}%")
    print(f"\nPrediction stats:")
    print(final_submission['Predicted'].describe())
    print(f"\n✓ Saved: {out_path}")
    print("="*60)


if __name__ == '__main__':
    main()
