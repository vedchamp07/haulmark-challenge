#!/usr/bin/env python3
"""
v4 local training script.

Reads: ckpts/train_v4.parquet, ckpts/test_v4.parquet
Writes: submissions/submission_v4_oof{X}_mean{Y}.csv

Key improvements over v3 (LB MSE 957):
- Fixed spatial zone features (were all 0 due to gpkg path bug)
- frac_dump_v4 = fraction of pings in ob_dump OR mineral_stock zone (both dump targets)
- frac_haul_v4 = fraction of pings on haul road
- cumdist_km = max-min cumdist per shift (more reliable km)
- accel_std / accel_mean = road roughness / load proxy
- mine_enc, has_dump_switch = mine identity and sensor capability flags
- tankcap as feature (slight variance between mine001 and mine002 vehicles)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "ckpts"
SUBS = ROOT / "submissions"
SUBS.mkdir(exist_ok=True)

def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

# ── Load v4 features ──────────────────────────────────────────────────────────
print("Loading v4 features...")
train = pd.read_parquet(CKPT / "train_v4.parquet")
test  = pd.read_parquet(CKPT / "test_v4.parquet")
train["date"] = pd.to_datetime(train["date"]).dt.date
test["date"]  = pd.to_datetime(test["date"]).dt.date
print(f"  train: {train.shape}, test: {test.shape}")

def safe_merge(left, right, on):
    keys = on if isinstance(on, list) else [on]
    drop = [c for c in right.columns if c not in keys and c in left.columns]
    return left.merge(right.drop(columns=drop), on=on, how="left")

# ── Load labels ───────────────────────────────────────────────────────────────
import glob
smry = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "smry_*.csv"))], ignore_index=True)
smry["date"] = pd.to_datetime(smry["date"]).dt.date
train = safe_merge(train, smry[["vehicle", "date", "shift", "acons"]], on=["vehicle", "date", "shift"])
print(f"Labeled: {train['acons'].notna().sum()} / {len(train)}")

# ── RFID refuel features ──────────────────────────────────────────────────────
import datetime
def assign_shift(ts_series):
    ts = pd.to_datetime(ts_series, utc=False)
    hour = ts.dt.hour
    date = ts.dt.date
    adj = [d - datetime.timedelta(days=1) if h >= 22 else d for d, h in zip(date, hour)]
    shift = ["C" if h >= 22 or h < 6 else ("A" if h < 14 else "B") for h in hour]
    return pd.Series(adj, index=ts_series.index), pd.Series(shift, index=ts_series.index)

rfid_files = sorted(DATA.glob("rfid_refuels_*.parquet"))
if rfid_files:
    rfid = pd.concat([pd.read_parquet(f) for f in rfid_files], ignore_index=True)
    rfid["ts"] = pd.to_datetime(rfid["ts"], utc=False)
    rfid["adj_date"], rfid["shift"] = assign_shift(rfid["ts"])
    # Day-level RFID aggregation (refuels straddle shift boundaries)
    rfid_agg = rfid.groupby(["vehicle", "adj_date"]).agg(
        rfid_liters_day=("litres", "sum"),
        rfid_events_day=("litres", "count"),
    ).reset_index()
    rfid_agg.rename(columns={"adj_date": "date"}, inplace=True)
    rfid_agg["date"] = pd.to_datetime(rfid_agg["date"]).dt.date
    train = safe_merge(train, rfid_agg, on=["vehicle", "date"])
    test  = safe_merge(test,  rfid_agg, on=["vehicle", "date"])
    for df_ in [train, test]:
        df_["rfid_liters_day"] = df_["rfid_liters_day"].fillna(0)
        df_["rfid_events_day"] = df_["rfid_events_day"].fillna(0)
    print(f"RFID: non-zero in train: {(train['rfid_liters_day'] > 0).sum()}")

# ── Vehicle aggregate features ────────────────────────────────────────────────
train_clean = train[train["acons"].notna() & (train["acons"] > 0)].copy()
global_mean = train_clean["acons"].mean()

veh_shift_stats = (
    train_clean.groupby(["vehicle", "shift"])["acons"]
    .agg(["mean", "std"]).reset_index()
)
veh_shift_stats.columns = ["vehicle", "shift", "veh_shift_mean_acons", "veh_shift_std_acons"]

_lph = train_clean.copy()
_lph["lph"] = _lph["acons"] / (_lph["ign_h"] + 1e-6)
veh_lph = _lph.groupby("vehicle")["lph"].median().reset_index().rename(columns={"lph": "veh_lph"})

veh_stats = train_clean.groupby("vehicle").agg(
    veh_mean_km        = ("shift_km",      "mean"),
    veh_mean_ign_h     = ("ign_h",         "mean"),
    veh_mean_dumps     = ("haul_cycles",   "mean"),
    veh_mean_idle_frac = ("idle_fraction", "mean"),
    veh_mean_speed     = ("speed_mean",    "mean"),
    veh_mean_ext_v     = ("ext_v_mean",    "mean"),
).reset_index().merge(veh_lph, on="vehicle", how="left")

_tc = train_clean[train_clean["haul_cycles"] > 0].copy()
_tc["fuel_per_trip"] = _tc["acons"] / _tc["haul_cycles"]
veh_fpt = _tc.groupby("vehicle")["fuel_per_trip"].median().reset_index()
veh_fpt.rename(columns={"fuel_per_trip": "veh_fuel_per_trip"}, inplace=True)

for df_ in [train, test]:
    df_ = safe_merge(df_, veh_shift_stats, on=["vehicle", "shift"])
    df_ = safe_merge(df_, veh_stats,       on="vehicle")
    df_ = safe_merge(df_, veh_fpt,         on="vehicle")

# Need to reassign since safe_merge returns new df
train = safe_merge(train, veh_shift_stats, on=["vehicle", "shift"])
train = safe_merge(train, veh_stats,       on="vehicle")
train = safe_merge(train, veh_fpt,         on="vehicle")
test  = safe_merge(test,  veh_shift_stats, on=["vehicle", "shift"])
test  = safe_merge(test,  veh_stats,       on="vehicle")
test  = safe_merge(test,  veh_fpt,         on="vehicle")

# Physics predictions
for df_ in [train, test]:
    df_["physics_pred"] = df_["veh_lph"].fillna(46.0) * df_["ign_h"]
    df_["trip_pred"]    = df_["veh_fuel_per_trip"].fillna(17.0) * df_["haul_cycles"]

# Operator encoding
if "operator_mode" in train_clean.columns and train_clean["operator_mode"].notna().sum() > 0:
    op_stats = (
        train_clean.dropna(subset=["operator_mode"])
        .groupby("operator_mode")["acons"].agg(["mean", "count"]).reset_index()
    )
    op_stats.columns = ["operator_mode", "op_mean_acons", "op_count"]
    train = safe_merge(train, op_stats, on="operator_mode")
    test  = safe_merge(test,  op_stats, on="operator_mode")

for df_ in [train, test]:
    if "op_mean_acons" not in df_.columns:
        df_["op_mean_acons"] = global_mean
        df_["op_count"]      = 0
    df_["op_mean_acons"] = df_["op_mean_acons"].fillna(global_mean)
    df_["op_count"]      = df_["op_count"].fillna(0)

print(f"\nTrain final shape: {train.shape}")
print(f"Test  final shape: {test.shape}")

# ── Feature selection ─────────────────────────────────────────────────────────
EXCLUDE = {
    "vehicle", "mine_anon", "date", "date_dt", "shift",
    "acons", "is_active_label", "is_active",
    "fuel_volume", "prod_hr_dpr", "idle_hr_dpr", "km_dpr",
    "tonnage", "hmr_dpr", "operator_id", "operator_mode",
    "bd_hr_dpr", "maint_hr_dpr",
}

def get_features(train_df, test_df):
    def ok(df, c):
        return (c not in EXCLUDE
                and df[c].dtype not in ["object"]
                and not pd.api.types.is_datetime64_any_dtype(df[c]))
    tr_cols = {c for c in train_df.columns if ok(train_df, c)}
    te_cols = {c for c in test_df.columns  if ok(test_df,  c)}
    return sorted(tr_cols & te_cols)

# ── TWO-STAGE MODEL ───────────────────────────────────────────────────────────
print("\n=== Two-stage model ===")
labeled = train[train["acons"].notna()].copy()
labeled["is_active_label"] = (labeled["acons"] > 10).astype(int)
print(f"Labeled: {len(labeled)}, active: {labeled['is_active_label'].sum()}, "
      f"inactive: {(labeled['is_active_label']==0).sum()}")

# ── Rank-transform altitude features to neutralize mine-depth drift ────────────
altitude_cols = ["altitude_std", "altitude_range", "net_lift", "uphill_m"]
alt_cols_present = [c for c in altitude_cols if c in train.columns and c in test.columns]
if alt_cols_present:
    combined = pd.concat([train[alt_cols_present], test[alt_cols_present]], ignore_index=True).fillna(0)
    for col in alt_cols_present:
        all_ranks = rankdata(combined[col].values, method='average') / len(combined)
        train[f"{col}_rank"] = all_ranks[:len(train)]
        test[f"{col}_rank"] = all_ranks[len(train):]
    print(f"Rank-transformed altitude: {alt_cols_present}")

feat_cols = get_features(labeled, test)
print(f"Features: {len(feat_cols)}")
print(f"New v4 features present: {[c for c in ['frac_dump_v4','frac_haul_v4','cumdist_km','accel_std','mine_enc','has_dump_switch'] if c in feat_cols]}")

X_all    = labeled[feat_cols].fillna(0)
y_all    = labeled["acons"].values
y_active = labeled["is_active_label"].values
groups   = labeled["vehicle"].astype("category").cat.codes.values
X_test   = test[feat_cols].fillna(0)

# Forward-chained monthly CV (mirrors LB temporal structure)
labeled["month"] = pd.to_datetime(labeled["date"]).dt.month
fold_splits = [
    (labeled[labeled["month"] == 1].index.tolist(),  # Train Jan
     labeled[labeled["month"] == 2].index.tolist()), # Val Feb
    (labeled[labeled["month"].isin([1, 2])].index.tolist(),  # Train Jan+Feb
     labeled[labeled["month"] == 3].index.tolist()), # Val Mar
]

gkf      = GroupKFold(n_splits=5)  # Keep for classifier compatibility

# Stage 1: classifier
print("\n--- Stage 1: Active/Inactive classifier ---")
CLF_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "verbosity": -1, "seed": 42,
}
oof_proba  = np.zeros(len(labeled))
clf_models = []
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_active, groups), 1):
    dtrain = lgb.Dataset(X_all.iloc[tr_idx], label=y_active[tr_idx])
    dval   = lgb.Dataset(X_all.iloc[val_idx], label=y_active[val_idx], reference=dtrain)
    clf = lgb.train(CLF_PARAMS, dtrain, valid_sets=[dval], num_boost_round=1000,
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    oof_proba[val_idx] = clf.predict(X_all.iloc[val_idx], num_iteration=clf.best_iteration)
    clf_models.append(clf)
    acc = ((oof_proba[val_idx] > 0.5) == y_active[val_idx]).mean()
    print(f"  Fold {fold}: accuracy={acc:.4f}")

print(f"OOF accuracy: {((oof_proba > 0.5) == y_active).mean():.4f}")
test_active_proba = np.mean([m.predict(X_test, num_iteration=m.best_iteration) for m in clf_models], axis=0)

# Stage 2: regressor on active shifts only
print("\n--- Stage 2: Regressor (active shifts only) ---")
REGR_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.02, "num_leaves": 63, "max_depth": -1,
    "min_data_in_leaf": 15, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 1,
    "lambda_l1": 0.05, "lambda_l2": 0.3,
    "verbosity": -1, "seed": 42,
}
active_mask = labeled["is_active_label"] == 1
active_df   = labeled[active_mask].copy()
X_act = active_df[feat_cols].fillna(0)
y_act = active_df["acons"].values
g_act = active_df["vehicle"].astype("category").cat.codes.values

# ── Optuna tuning on Fold 1 (Jan+Feb → Mar, hardest fold) ────────────────────
regr_params_best = REGR_PARAMS.copy()
if OPTUNA_AVAILABLE and len(fold_splits) > 1:
    print("\n--- Optuna tuning on Fold 1 (Jan+Feb → Mar) ---")
    tr_idx_abs, val_idx_abs = fold_splits[1]
    tr_mask = active_df.index.isin(tr_idx_abs)
    val_mask = active_df.index.isin(val_idx_abs)
    tr_idx_opt = np.where(tr_mask)[0]
    val_idx_opt = np.where(val_mask)[0]
    
    if len(tr_idx_opt) > 0 and len(val_idx_opt) > 0:
        X_opt_tr, y_opt_tr = X_act.iloc[tr_idx_opt], y_act[tr_idx_opt]
        X_opt_val, y_opt_val = X_act.iloc[val_idx_opt], y_act[val_idx_opt]
        
        def objective(trial):
            p = dict(
                objective="regression", metric="rmse",
                learning_rate = trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                num_leaves    = trial.suggest_int('num_leaves', 15, 127),
                max_depth     = trial.suggest_int('max_depth', 3, 10),
                min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5, 50),
                feature_fraction = trial.suggest_float('feature_fraction', 0.5, 1.0),
                bagging_fraction = trial.suggest_float('bagging_fraction', 0.5, 1.0),
                lambda_l1 = trial.suggest_float('lambda_l1', 0.0, 1.0),
                lambda_l2 = trial.suggest_float('lambda_l2', 0.0, 1.0),
                verbosity=-1, seed=42,
            )
            dtrain = lgb.Dataset(X_opt_tr, label=y_opt_tr)
            dval = lgb.Dataset(X_opt_val, label=y_opt_val, reference=dtrain)
            m = lgb.train(p, dtrain, valid_sets=[dval], num_boost_round=5000,
                          callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
            pred = m.predict(X_opt_val, num_iteration=m.best_iteration)
            return rmse(y_opt_val, np.clip(pred, 0, None))
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100, show_progress_bar=False)
        best_optuna_params = study.best_params
        print(f"  Best Optuna RMSE: {study.best_value:.2f}L")
        regr_params_best.update({k: best_optuna_params[k] for k in best_optuna_params 
                                 if k in REGR_PARAMS})
        print(f"  Updated REGR_PARAMS with Optuna tuning")

print("\n--- Stage 2: Regressor (active shifts only, forward-chained monthly CV) ---")
oof_reg    = np.zeros(len(active_df))
reg_models = []
fold_rmses = []

for fold, (tr_idx_abs, val_idx_abs) in enumerate(fold_splits):
    # Map absolute indices to active_df indices
    tr_mask = active_df.index.isin(tr_idx_abs)
    val_mask = active_df.index.isin(val_idx_abs)
    tr_idx = np.where(tr_mask)[0]
    val_idx = np.where(val_mask)[0]
    
    if len(val_idx) == 0 or len(tr_idx) == 0:
        continue
    
    dtrain = lgb.Dataset(X_act.iloc[tr_idx], label=y_act[tr_idx])
    dval   = lgb.Dataset(X_act.iloc[val_idx], label=y_act[val_idx], reference=dtrain)
    reg = lgb.train(regr_params_best, dtrain, valid_sets=[dval], num_boost_round=5000,
                    callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(1000)])
    oof_reg[val_idx] = reg.predict(X_act.iloc[val_idx], num_iteration=reg.best_iteration)
    reg_models.append(reg)
    fr = rmse(y_act[val_idx], np.clip(oof_reg[val_idx], 0, None))
    fold_rmses.append(fr)
    val_months = labeled.loc[val_idx_abs, 'month'].unique()
    tr_months = labeled.loc[tr_idx_abs, 'month'].unique()
    print(f"  Fold {fold+1}: RMSE={fr:.2f}L  (iter={reg.best_iteration}) | "
          f"train months {sorted(tr_months)} → val month {sorted(val_months)}")

oof_reg = np.clip(oof_reg, 0, None)
oof_rmse_active = rmse(y_act, oof_reg)
print(f"\nOOF RMSE (active only):      {oof_rmse_active:.2f}L")
print(f"Fold range: {min(fold_rmses):.2f}–{max(fold_rmses):.2f}L")
print(f"Previous:   27.47L  (v3, spatial broken)")

# Combined OOF RMSE (inactive treated as 0)
oof_full = np.zeros(len(labeled))
oof_full[np.where(active_mask)[0]] = oof_reg
print(f"OOF RMSE (combined):         {rmse(y_all, oof_full):.2f}L")
print(f"Previous combined:  23.79L  (v3)")

# Feature importance
importance = pd.DataFrame({
    "feature": feat_cols,
    "gain": np.mean([m.feature_importance("gain") for m in reg_models], axis=0),
}).sort_values("gain", ascending=False)
print("\nTop 25 features:")
print(importance.head(25).to_string(index=False))
importance.to_csv(ROOT / "outputs" / "v4_feature_importance.csv", index=False)

# ── Test predictions ──────────────────────────────────────────────────────────
print("\n=== Generating submission ===")
test_reg_pred = np.mean(
    [m.predict(X_test, num_iteration=m.best_iteration) for m in reg_models], axis=0
)
test_reg_pred = np.clip(test_reg_pred, 0, None)

# Blend by classifier confidence
test_preds = test_reg_pred.copy()
low_active  = test_active_proba < 0.3
mid_active  = (test_active_proba >= 0.3) & (test_active_proba < 0.7)
test_preds[low_active] = test_reg_pred[low_active] * test_active_proba[low_active]
test_preds[mid_active] = test_reg_pred[mid_active] * test_active_proba[mid_active]
# high_active (>0.7): use full regressor prediction unchanged

test["Predicted"] = test_preds

# ── Build submission ──────────────────────────────────────────────────────────
idm = pd.read_csv(DATA / "id_mapping_new.csv")
idm["date"] = pd.to_datetime(idm["date"]).dt.date

submission = idm.merge(
    test[["vehicle", "date", "shift", "Predicted"]],
    on=["vehicle", "date", "shift"], how="left",
)
n_missing = submission["Predicted"].isna().sum()
print(f"Missing predictions: {n_missing} / {len(submission)}")

if n_missing > 0:
    vs_mean  = train[train["acons"].notna()].groupby(["vehicle", "shift"])["acons"].mean()
    glob_avg = float(train["acons"].dropna().mean())
    for idx, row in submission[submission["Predicted"].isna()].iterrows():
        key = (row["vehicle"], row["shift"])
        submission.at[idx, "Predicted"] = vs_mean.get(key, glob_avg)

fleet = pd.read_csv(DATA / "fleet.csv")
tankcap_map = dict(zip(fleet["vehicle"], fleet["tankcap"]))
submission["tankcap"] = submission["vehicle"].map(tankcap_map).fillna(1379)
submission["Predicted"] = submission.apply(
    lambda r: float(np.clip(r["Predicted"], 0, r["tankcap"])), axis=1
)

final = submission[["id", "Predicted"]].sort_values("id").reset_index(drop=True)
print(f"\nSubmission stats:\n{final['Predicted'].describe()}")
print(f"Zeros:          {(final['Predicted'] == 0).sum()}")
print(f"Near-zero <10L: {(final['Predicted'] < 10).sum()}")

mean_p = final["Predicted"].mean()
std_p  = final["Predicted"].std()
out_name = f"submission_v4_oof{oof_rmse_active:.2f}_mean{mean_p:.1f}.csv"
out_path = SUBS / out_name
final.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"\nSummary:")
print(f"  OOF RMSE active: {oof_rmse_active:.2f}L  (v3: 27.47L)")
print(f"  OOF MSE approx:  {oof_rmse_active**2:.1f}  (v3 LB: 957.06)")
