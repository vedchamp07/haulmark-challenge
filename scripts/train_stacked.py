#!/usr/bin/env python3
"""Stacked ensemble with meta-learner for ultimate performance"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = ROOT / "submissions"


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


def train_base_models_with_oof(X, y, groups, n_splits=5):
    """Train base models and collect OOF predictions for stacking"""

    gkf = GroupKFold(n_splits=n_splits)

    # Store OOF predictions from each base model
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))

    models_lgb = []
    models_xgb = []
    models_cat = []

    # Model parameters (tuned from previous best runs)
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.008,
        'num_leaves': 80,
        'max_depth': 9,
        'min_data_in_leaf': 15,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 3,
        'lambda_l1': 0.2,
        'lambda_l2': 0.8,
        'verbosity': -1,
        'seed': 42,
    }

    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.008,
        'max_depth': 8,
        'min_child_weight': 4,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.2,
        'reg_lambda': 0.8,
        'seed': 42,
    }

    cat_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.008,
        'depth': 9,
        'l2_leaf_reg': 5,
        'random_seed': 42,
        'iterations': 3000,
        'early_stopping_rounds': 150,
        'verbose': 0,
    }

    print("\n" + "="*60)
    print("TRAINING BASE MODELS FOR STACKING")
    print("="*60)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n[Fold {fold}/{n_splits}] Training base models...")

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # LightGBM
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        lgb_model = lgb.train(
            lgb_params, train_data, valid_sets=[val_data],
            num_boost_round=3000,
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
        )
        oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        models_lgb.append(lgb_model)

        # XGBoost
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        xgb_model = xgb.train(
            xgb_params, dtrain, num_boost_round=3000,
            evals=[(dval, 'val')], early_stopping_rounds=150,
            verbose_eval=0
        )
        oof_xgb[val_idx] = xgb_model.predict(dval)
        models_xgb.append(xgb_model)

        # CatBoost
        train_pool = Pool(X_tr, y_tr)
        val_pool = Pool(X_val, y_val)
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(train_pool, eval_set=val_pool)
        oof_cat[val_idx] = cat_model.predict(X_val)
        models_cat.append(cat_model)

        fold_rmse_lgb = rmse(y_val, oof_lgb[val_idx])
        fold_rmse_xgb = rmse(y_val, oof_xgb[val_idx])
        fold_rmse_cat = rmse(y_val, oof_cat[val_idx])
        print(f"  LGB: {fold_rmse_lgb:.2f} | XGB: {fold_rmse_xgb:.2f} | CAT: {fold_rmse_cat:.2f}")

    return {
        'models_lgb': models_lgb,
        'models_xgb': models_xgb,
        'models_cat': models_cat,
        'oof_lgb': oof_lgb,
        'oof_xgb': oof_xgb,
        'oof_cat': oof_cat,
    }


def train_meta_learner(base_oofs, y, groups, n_splits=5):
    """Train meta-learner on base model predictions"""

    # Create meta-features matrix
    meta_features = np.column_stack([
        base_oofs['oof_lgb'],
        base_oofs['oof_xgb'],
        base_oofs['oof_cat'],
    ])

    # Cross-validate meta-learner
    gkf = GroupKFold(n_splits=n_splits)
    oof_meta = np.zeros(len(y))
    meta_models = []

    print("\n" + "="*60)
    print("TRAINING META-LEARNER (STACKING)")
    print("="*60)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(meta_features, y, groups), 1):
        X_tr, X_val = meta_features[train_idx], meta_features[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Ridge regression as meta-learner (robust, prevents overfitting)
        meta_model = Ridge(alpha=10.0)
        meta_model.fit(X_tr, y_tr)

        oof_meta[val_idx] = meta_model.predict(X_val)
        meta_models.append(meta_model)

        fold_rmse = rmse(y_val, np.clip(oof_meta[val_idx], 0, None))
        print(f"[Fold {fold}] Meta RMSE: {fold_rmse:.2f}")

    oof_meta = np.clip(oof_meta, 0, None)
    overall_rmse = rmse(y, oof_meta)

    print("\n" + "="*60)
    print(f"STACKED OOF RMSE: {overall_rmse:.2f}")
    print("="*60)

    print("\nMeta-learner weights:")
    avg_coef = np.mean([m.coef_ for m in meta_models], axis=0)
    print(f"  LightGBM: {avg_coef[0]:.4f}")
    print(f"  XGBoost:  {avg_coef[1]:.4f}")
    print(f"  CatBoost: {avg_coef[2]:.4f}")

    return meta_models, oof_meta


def predict_stacked(base_models, meta_models, X_test, features):
    """Generate stacked predictions"""

    X_test = X_test[features].fillna(0)

    # Get base model predictions
    preds_lgb = np.mean([
        m.predict(X_test, num_iteration=m.best_iteration)
        for m in base_models['models_lgb']
    ], axis=0)

    preds_xgb = np.mean([
        m.predict(xgb.DMatrix(X_test))
        for m in base_models['models_xgb']
    ], axis=0)

    preds_cat = np.mean([
        m.predict(X_test)
        for m in base_models['models_cat']
    ], axis=0)

    # Stack into meta-features
    meta_features = np.column_stack([preds_lgb, preds_xgb, preds_cat])

    # Meta-learner prediction
    preds_stacked = np.mean([
        m.predict(meta_features)
        for m in meta_models
    ], axis=0)

    return np.clip(preds_stacked, 0, None)


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

    # Prepare data
    train = train.dropna(subset=['acons']).copy()
    X = train[features].fillna(0)
    y = train['acons'].values
    groups = train['vehicle'].astype('category').cat.codes.values

    print(f"Using {len(features)} features")

    # Train base models
    base_models = train_base_models_with_oof(X, y, groups, n_splits=5)

    # Train meta-learner
    meta_models, oof_stacked = train_meta_learner(
        base_models, y, groups, n_splits=5
    )

    # Compare to simple ensemble
    oof_simple = (
        base_models['oof_lgb'] * 0.4 +
        base_models['oof_xgb'] * 0.3 +
        base_models['oof_cat'] * 0.3
    )
    oof_simple = np.clip(oof_simple, 0, None)
    simple_rmse = rmse(y, oof_simple)

    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    stacked_rmse = rmse(y, oof_stacked)
    print(f"Simple Ensemble RMSE: {simple_rmse:.2f}")
    print(f"Stacked Ensemble RMSE: {stacked_rmse:.2f}")
    print(f"Improvement: {((simple_rmse - stacked_rmse) / simple_rmse * 100):.2f}%")
    print("="*60)

    # Predict on test
    print("\nGenerating stacked predictions...")
    test_preds = predict_stacked(base_models, meta_models, test, features)

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

    # Fallback
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
    out_path = SUBMISSIONS_DIR / f"shiftwise__stacked__oof{stacked_rmse:.2f}__mean{mean:.2f}__std{std:.2f}__leaky.csv"
    final_submission.to_csv(out_path, index=False)

    print(f"\n✓ Saved: {out_path}")
    print(f"Submission rows: {len(final_submission)}")
    print(f"\nPrediction stats:")
    print(final_submission['Predicted'].describe())


if __name__ == '__main__':
    main()
