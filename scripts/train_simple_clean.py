#!/usr/bin/env python3
"""Simple improved model - fast and clean"""
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "new_data"
SUBMISSIONS_DIR = ROOT / "submissions"

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("Loading data...")
train = pd.read_csv(DATA_DIR / 'train_shift_features.csv')
test = pd.read_csv(DATA_DIR / 'test_shift_features.csv')

# Load targets
summary = pd.concat([
    pd.read_csv(DATA_DIR / 'smry_jan_train_ordered.csv'),
    pd.read_csv(DATA_DIR / 'smry_feb_train_ordered.csv'),
    pd.read_csv(DATA_DIR / 'smry_mar_train_ordered.csv'),
])
summary['date'] = pd.to_datetime(summary['date']).dt.date

# Merge targets
train['date'] = pd.to_datetime(train['date']).dt.date
test['date'] = pd.to_datetime(test['date']).dt.date
train = train.merge(summary[['vehicle', 'date', 'shift', 'acons']],
                     on=['vehicle', 'date', 'shift'], how='left')

# Add simple safe features
print("Adding features...")
for df in [train, test]:
    df['shift_enc'] = df['shift'].map({'C': 0, 'A': 1, 'B': 2})
    df['km_per_hour'] = df['shift_km'] / (df['ignition_on_hours'] + 0.001)
    df['idle_ratio'] = df['idle_hours'] / (df['ignition_on_hours'] + 0.001)
    df['work_intensity'] = df['shift_km'] * df['total_climb_m']

# Vehicle stats from training
veh_stats = train.groupby('vehicle')['shift_km'].agg(['mean', 'std']).reset_index()
veh_stats.columns = ['vehicle', 'veh_mean_km', 'veh_std_km']

train = train.merge(veh_stats, on='vehicle', how='left')
test = test.merge(veh_stats, on='vehicle', how='left')

# Prepare features
exclude = {'vehicle', 'date', 'shift', 'acons'}
features = [c for c in train.columns if c not in exclude and train[c].dtype in ['int64', 'float64']]

print(f"Using {len(features)} features")

# Clean data
train = train.dropna(subset=['acons'])
X = train[features].fillna(0)
y = train['acons'].values
groups = train['vehicle'].astype('category').cat.codes.values

print(f"Training on {len(X)} samples")

# Train model with CV
gkf = GroupKFold(n_splits=5)
models = []
oof = np.zeros(len(y))

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.5,
    'verbosity': -1,
    'seed': 42,
}

print("\nTraining with 5-fold CV...")
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.train(
        params,
        lgb.Dataset(X_tr, y_tr),
        valid_sets=[lgb.Dataset(X_val, y_val)],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    models.append(model)

    print(f"Fold {fold}: {rmse(y_val, np.clip(oof[val_idx], 0, None)):.2f}L")

oof = np.clip(oof, 0, None)
print(f"\n OOF RMSE: {rmse(y, oof):.2f}L")

# Predict
print("\nPredicting on test...")
X_test = test[features].fillna(0)
preds = np.mean([m.predict(X_test, num_iteration=m.best_iteration) for m in models], axis=0)
preds = np.clip(preds, 0, None)

# Create submission
id_map = pd.read_csv(DATA_DIR / 'id_mapping_new.csv')
id_map['date'] = pd.to_datetime(id_map['date']).dt.date

test['Predicted'] = preds
sub = id_map.merge(test[['vehicle', 'date', 'shift', 'Predicted']],
                    on=['vehicle', 'date', 'shift'], how='left')

# Fill missing
sub['Predicted'] = sub['Predicted'].fillna(train['acons'].mean()).clip(0, None)

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
out_path = SUBMISSIONS_DIR / 'shiftwise__v2_clean.csv'
sub[['id', 'Predicted']].sort_values('id').to_csv(out_path, index=False)

print(f"\n✓ Saved: {out_path}")
print(f"Rows: {len(sub)}")
print(f"Mean: {sub['Predicted'].mean():.2f}L")
print(f"Std: {sub['Predicted'].std():.2f}L")
