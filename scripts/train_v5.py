#!/usr/bin/env python3
"""
v5 training script — time-based CV with fold-safe vehicle history.

Key improvements over v4:
- Loaded/empty state features: frac_loaded_moving, frac_empty_moving,
  loaded_ext_v_x_h, empty_ext_v_x_h (physics-grounded energy decomposition)
- Altitude features: altitude_gain_m, loaded_alt_gain_m (corr ~0.39 with acons)
- Operator fold-safe target encoding
- Time-based CV (Jan/Feb 1-15 → 16-20) instead of GroupKFold by vehicle
  → eliminates ~7L leakage from veh_shift_mean_acons computed globally
- Removes 7 broken zero-gain stub features from v4
"""
import glob, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "ckpts"
SUBS = ROOT / "submissions"
SUBS.mkdir(exist_ok=True)

def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

# ── Load v5 features ──────────────────────────────────────────────────────────
print("Loading v5 features...")
train = pd.read_parquet(CKPT / "train_v5.parquet")
test  = pd.read_parquet(CKPT / "test_v5.parquet")
train["date"] = pd.to_datetime(train["date"]).dt.date
test["date"]  = pd.to_datetime(test["date"]).dt.date
print(f"  train: {train.shape}, test: {test.shape}")

def safe_merge(left, right, on):
    keys = on if isinstance(on, list) else [on]
    drop = [c for c in right.columns if c not in keys and c in left.columns]
    return left.merge(right.drop(columns=drop), on=on, how="left")

# ── Load labels ───────────────────────────────────────────────────────────────
smry = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "smry_*.csv"))],
                 ignore_index=True)
smry["date"] = pd.to_datetime(smry["date"]).dt.date
train = safe_merge(train, smry[["vehicle", "date", "shift", "acons"]],
                   on=["vehicle", "date", "shift"])
print(f"Labeled: {train['acons'].notna().sum()} / {len(train)}")

# ── RFID features ─────────────────────────────────────────────────────────────
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
    rfid_agg = rfid.groupby(["vehicle", "adj_date", "shift"]).agg(
        rfid_liters=("litres", "sum"),
        rfid_events=("litres", "count"),
    ).reset_index()
    rfid_agg.rename(columns={"adj_date": "date"}, inplace=True)
    rfid_agg["date"] = pd.to_datetime(rfid_agg["date"]).dt.date
    for df_ in [train, test]:
        df_ = safe_merge(df_, rfid_agg, on=["vehicle", "date", "shift"])
    train = safe_merge(train, rfid_agg, on=["vehicle", "date", "shift"])
    test  = safe_merge(test,  rfid_agg, on=["vehicle", "date", "shift"])
    for df_ in [train, test]:
        df_["rfid_liters"] = df_["rfid_liters"].fillna(0)
        df_["rfid_events"] = df_["rfid_events"].fillna(0)
    print(f"RFID non-zero in train: {(train['rfid_liters'] > 0).sum()}")

# ── Derived features from v5 state columns ───────────────────────────────────
# Decompose top feature ext_v_x_mov_h into loaded/empty components
for df_ in [train, test]:
    df_["loaded_mov_h"]    = df_["frac_loaded_moving"] * df_["mov_h"]
    df_["empty_mov_h"]     = df_["frac_empty_moving"]  * df_["mov_h"]
    df_["loaded_km"]       = df_["frac_loaded_moving"] * df_["shift_km"]
    df_["empty_km"]        = df_["frac_empty_moving"]  * df_["shift_km"]
    # Physics energy proxies split by load state
    df_["loaded_ext_v_x_h"] = df_["loaded_ext_v_mean"] * df_["loaded_mov_h"]
    df_["empty_ext_v_x_h"]  = df_["empty_ext_v_mean"]  * df_["empty_mov_h"]
    # Altitude × load interactions
    df_["loaded_uphill_km"] = df_["loaded_alt_gain_m"] / (df_["loaded_km"] + 1e-3)
    df_["empty_uphill_km"]  = df_["empty_alt_gain_m"]  / (df_["empty_km"] + 1e-3)

print(f"\nTrain final shape: {train.shape}")
print(f"Test  final shape: {test.shape}")

# ── Time-based CV folds ───────────────────────────────────────────────────────
# We have: Jan 1-20 train, Feb 1-20 train, Mar 1-11 train
# Test leakage pattern: Jan 21-31, Feb 21-28, Mar 12-20
# CV mimics this: train on 1-15, validate on 16-20 (same-month temporal split)
import datetime as dt

def make_date_folds(df):
    """
    Returns list of (train_mask, val_mask) for time-based CV.
    Folds:
      1. Jan 1-15 → Jan 16-20
      2. Feb 1-15 → Feb 16-20
      3. Jan+Feb 1-15 → Jan 16-20 + Feb 16-20 (combined — more training data)
    March always goes into training (no val data available).
    """
    dates = pd.to_datetime(df["date"])
    jan_train = (dates.dt.month == 1) & (dates.dt.day <= 15)
    jan_val   = (dates.dt.month == 1) & (dates.dt.day >= 16) & (dates.dt.day <= 20)
    feb_train = (dates.dt.month == 2) & (dates.dt.day <= 15)
    feb_val   = (dates.dt.month == 2) & (dates.dt.day >= 16) & (dates.dt.day <= 20)
    mar_train = dates.dt.month == 3

    return [
        ("Jan fold",  jan_train,                        jan_val),
        ("Feb fold",  feb_train,                        feb_val),
        ("Combined",  jan_train | feb_train | mar_train, jan_val | feb_val),
    ]


# ── Feature selection ─────────────────────────────────────────────────────────
EXCLUDE = {
    "vehicle", "mine_anon", "date", "date_dt", "shift",
    "acons", "is_active_label", "is_active",
    "fuel_volume", "prod_hr_dpr", "idle_hr_dpr", "km_dpr",
    "tonnage", "hmr_dpr", "operator_id", "operator_mode",
    "bd_hr_dpr", "maint_hr_dpr",
    # State machine internals (not useful as raw features)
    "loaded_ext_v_sum", "loaded_n_pings", "empty_ext_v_sum", "empty_n_pings",
    # operator_id_shift used internally for fold-safe encoding
    "operator_id_shift",
}

def get_features(train_df, test_df):
    def ok(df, c):
        return (c not in EXCLUDE
                and df[c].dtype not in ["object"]
                and not pd.api.types.is_datetime64_any_dtype(df[c]))
    tr_cols = {c for c in train_df.columns if ok(train_df, c)}
    te_cols = {c for c in test_df.columns  if ok(test_df,  c)}
    return sorted(tr_cols & te_cols)


# ── Build vehicle/operator global stats for TEST filling ─────────────────────
# (Used for test predictions only — train uses fold-safe versions)
train_labeled_all = train[train["acons"].notna() & (train["acons"] > 0)].copy()
global_mean = train_labeled_all["acons"].mean()

veh_shift_global = (
    train_labeled_all.groupby(["vehicle", "shift"])["acons"]
    .agg(["mean", "std"]).reset_index()
)
veh_shift_global.columns = ["vehicle", "shift", "veh_shift_mean_acons", "veh_shift_std_acons"]

_lph = train_labeled_all.copy()
_lph["lph"] = _lph["acons"] / (_lph["ign_h"] + 1e-6)
veh_lph_global = _lph.groupby("vehicle")["lph"].median().reset_index().rename(
    columns={"lph": "veh_lph"})

veh_stats_global = train_labeled_all.groupby("vehicle").agg(
    veh_mean_km        = ("shift_km",      "mean"),
    veh_mean_ign_h     = ("ign_h",         "mean"),
    veh_mean_dumps     = ("haul_cycles",   "mean"),
    veh_mean_idle_frac = ("idle_fraction", "mean"),
    veh_mean_speed     = ("speed_mean",    "mean"),
    veh_mean_ext_v     = ("ext_v_mean",    "mean"),
).reset_index().merge(veh_lph_global, on="vehicle", how="left")

_tc = train_labeled_all[train_labeled_all["haul_cycles"] > 0].copy()
_tc["fuel_per_trip"] = _tc["acons"] / _tc["haul_cycles"]
veh_fpt_global = _tc.groupby("vehicle")["fuel_per_trip"].median().reset_index()
veh_fpt_global.rename(columns={"fuel_per_trip": "veh_fuel_per_trip"}, inplace=True)

# Operator global stats
op_global = pd.DataFrame()
if "operator_id_shift" in train_labeled_all.columns:
    op_global = (train_labeled_all.dropna(subset=["operator_id_shift"])
                 .groupby("operator_id_shift")["acons"]
                 .agg(["mean", "count"]).reset_index())
    op_global.columns = ["operator_id_shift", "op_mean_acons", "op_count"]

# Apply to TEST set (global stats — no fold needed for test)
test = safe_merge(test, veh_shift_global, on=["vehicle", "shift"])
test = safe_merge(test, veh_stats_global, on="vehicle")
test = safe_merge(test, veh_fpt_global,   on="vehicle")
if len(op_global) > 0:
    test = safe_merge(test, op_global[["operator_id_shift", "op_mean_acons", "op_count"]],
                      on="operator_id_shift")

for df_ in [test]:
    df_["veh_shift_mean_acons"] = df_["veh_shift_mean_acons"].fillna(global_mean)
    df_["veh_shift_std_acons"]  = df_["veh_shift_std_acons"].fillna(0)
    df_["veh_lph"]              = df_["veh_lph"].fillna(46.0)
    df_["veh_fuel_per_trip"]    = df_.get("veh_fuel_per_trip", pd.Series(17.0, index=df_.index)).fillna(17.0)
    df_["op_mean_acons"]        = df_.get("op_mean_acons", pd.Series(global_mean, index=df_.index)).fillna(global_mean)
    df_["op_count"]             = df_.get("op_count", pd.Series(0, index=df_.index)).fillna(0)
    df_["physics_pred"]         = df_["veh_lph"] * df_["ign_h"]
    df_["trip_pred"]            = df_["veh_fuel_per_trip"] * df_["haul_cycles"]
    for col in ["veh_mean_km", "veh_mean_ign_h", "veh_mean_dumps",
                "veh_mean_idle_frac", "veh_mean_speed", "veh_mean_ext_v"]:
        if col in df_.columns:
            df_[col] = df_[col].fillna(0)

# Apply global stats to train too (for feature selection alignment)
train = safe_merge(train, veh_shift_global, on=["vehicle", "shift"])
train = safe_merge(train, veh_stats_global, on="vehicle")
train = safe_merge(train, veh_fpt_global,   on="vehicle")
if len(op_global) > 0:
    train = safe_merge(train, op_global[["operator_id_shift", "op_mean_acons", "op_count"]],
                       on="operator_id_shift")
for df_ in [train]:
    df_["veh_lph"]           = df_["veh_lph"].fillna(46.0)
    df_["veh_fuel_per_trip"] = df_.get("veh_fuel_per_trip", pd.Series(17.0, index=df_.index)).fillna(17.0)
    df_["op_mean_acons"]     = df_.get("op_mean_acons", pd.Series(global_mean, index=df_.index)).fillna(global_mean)
    df_["op_count"]          = df_.get("op_count", pd.Series(0, index=df_.index)).fillna(0)
    df_["physics_pred"]      = df_["veh_lph"] * df_["ign_h"]
    df_["trip_pred"]         = df_["veh_fuel_per_trip"] * df_["haul_cycles"]
    for col in ["veh_mean_km", "veh_mean_ign_h", "veh_mean_dumps",
                "veh_mean_idle_frac", "veh_mean_speed", "veh_mean_ext_v"]:
        if col in df_.columns:
            df_[col] = df_[col].fillna(0)

# ── TWO-STAGE MODEL with time-based CV ───────────────────────────────────────
print("\n=== Two-stage model (time-based CV) ===")
labeled = train[train["acons"].notna()].copy()
labeled["is_active_label"] = (labeled["acons"] > 10).astype(int)
print(f"Labeled: {len(labeled)}, active: {labeled['is_active_label'].sum()}, "
      f"inactive: {(labeled['is_active_label']==0).sum()}")

feat_cols = get_features(labeled, test)
print(f"Features ({len(feat_cols)}): {feat_cols[:10]}...")
print(f"New v5 features: {[c for c in feat_cols if c in ['altitude_gain_m','loaded_alt_gain_m','loaded_ext_v_x_h','frac_loaded_moving','loaded_mov_h','angle_change_mean']]}")

X_test = test[feat_cols].fillna(0)

# Stage 1 params
CLF_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "verbosity": -1, "seed": 42,
}
# Stage 2 params
REGR_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.02, "num_leaves": 63, "max_depth": -1,
    "min_data_in_leaf": 15, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 1,
    "lambda_l1": 0.05, "lambda_l2": 0.3,
    "verbosity": -1, "seed": 42,
}

# ── Time-based CV ─────────────────────────────────────────────────────────────
print("\n--- Stage 1: Active/Inactive classifier ---")
folds = make_date_folds(labeled)
oof_proba  = np.full(len(labeled), np.nan)
clf_models = []

for fold_name, tr_mask, val_mask in folds:
    tr_idx  = np.where(tr_mask.values)[0]
    val_idx = np.where(val_mask.values)[0]
    if len(val_idx) == 0:
        continue

    # Fold-safe vehicle history (computed on training portion only)
    tr_data = labeled.iloc[tr_idx]
    tr_active = tr_data[tr_data["acons"] > 0]
    fold_global_mean = tr_active["acons"].mean() if len(tr_active) > 0 else global_mean

    veh_shift_fold = (tr_active.groupby(["vehicle", "shift"])["acons"]
                     .agg(["mean", "std"]).reset_index())
    veh_shift_fold.columns = ["vehicle", "shift", "veh_shift_mean_acons", "veh_shift_std_acons"]

    _lph_f = tr_active.copy()
    _lph_f["lph"] = _lph_f["acons"] / (_lph_f["ign_h"] + 1e-6)
    veh_lph_fold = _lph_f.groupby("vehicle")["lph"].median().reset_index()
    veh_lph_fold.rename(columns={"lph": "veh_lph"}, inplace=True)

    _tc_f = tr_active[tr_active["haul_cycles"] > 0].copy()
    if len(_tc_f) > 0:
        _tc_f["fuel_per_trip"] = _tc_f["acons"] / _tc_f["haul_cycles"]
        veh_fpt_fold = _tc_f.groupby("vehicle")["fuel_per_trip"].median().reset_index()
        veh_fpt_fold.rename(columns={"fuel_per_trip": "veh_fuel_per_trip"}, inplace=True)
    else:
        veh_fpt_fold = pd.DataFrame(columns=["vehicle", "veh_fuel_per_trip"])

    op_fold = pd.DataFrame()
    if "operator_id_shift" in tr_active.columns:
        op_fold = (tr_active.dropna(subset=["operator_id_shift"])
                   .groupby("operator_id_shift")["acons"]
                   .agg(["mean", "count"]).reset_index())
        op_fold.columns = ["operator_id_shift", "op_mean_acons_fold", "op_count_fold"]

    def apply_fold_stats(df_part, vsh, vlph, vfpt, op_f, gm):
        df_part = df_part.copy()
        drop_existing = [c for c in ["veh_shift_mean_acons", "veh_shift_std_acons",
                                      "veh_lph", "veh_fuel_per_trip",
                                      "op_mean_acons_fold", "op_count_fold"] if c in df_part.columns]
        df_part.drop(columns=drop_existing, inplace=True)

        df_part = df_part.merge(vsh, on=["vehicle", "shift"], how="left")
        df_part = df_part.merge(vlph, on="vehicle", how="left")
        if len(vfpt) > 0:
            df_part = df_part.merge(vfpt, on="vehicle", how="left")
        else:
            df_part["veh_fuel_per_trip"] = 17.0
        if len(op_f) > 0:
            df_part = df_part.merge(op_f, on="operator_id_shift", how="left")
        else:
            df_part["op_mean_acons_fold"] = gm
            df_part["op_count_fold"] = 0

        df_part["veh_shift_mean_acons"] = df_part["veh_shift_mean_acons"].fillna(gm)
        df_part["veh_shift_std_acons"]  = df_part["veh_shift_std_acons"].fillna(0)
        df_part["veh_lph"]              = df_part["veh_lph"].fillna(46.0)
        df_part["veh_fuel_per_trip"]    = df_part["veh_fuel_per_trip"].fillna(17.0)
        df_part["op_mean_acons_fold"]   = df_part.get("op_mean_acons_fold", pd.Series(gm, index=df_part.index)).fillna(gm)
        df_part["op_count_fold"]        = df_part.get("op_count_fold", pd.Series(0, index=df_part.index)).fillna(0)
        df_part["physics_pred"]         = df_part["veh_lph"] * df_part["ign_h"]
        df_part["trip_pred"]            = df_part["veh_fuel_per_trip"] * df_part["haul_cycles"]
        return df_part

    tr_fold = apply_fold_stats(labeled.iloc[tr_idx], veh_shift_fold, veh_lph_fold,
                                veh_fpt_fold, op_fold, fold_global_mean)
    val_fold = apply_fold_stats(labeled.iloc[val_idx], veh_shift_fold, veh_lph_fold,
                                 veh_fpt_fold, op_fold, fold_global_mean)

    # Build feature sets for this fold (fold-specific columns may differ)
    fold_feat_cols = [c for c in feat_cols if c in tr_fold.columns and c in val_fold.columns]
    # Replace fold-safe columns in feat_cols
    fold_feat_final = []
    for c in fold_feat_cols:
        fold_feat_final.append(c)
    # Add fold-safe operator column if available
    if "op_mean_acons_fold" in tr_fold.columns:
        fold_feat_final = [c for c in fold_feat_final if c != "op_mean_acons"]
        if "op_mean_acons_fold" not in fold_feat_final:
            fold_feat_final.append("op_mean_acons_fold")

    X_tr  = tr_fold[fold_feat_final].fillna(0)
    y_tr  = labeled.iloc[tr_idx]["is_active_label"].values
    X_val = val_fold[fold_feat_final].fillna(0)
    y_val = labeled.iloc[val_idx]["is_active_label"].values

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    clf = lgb.train(CLF_PARAMS, dtrain, valid_sets=[dval], num_boost_round=1000,
                    callbacks=[lgb.early_stopping(100, verbose=False),
                                lgb.log_evaluation(500)])
    preds = clf.predict(X_val, num_iteration=clf.best_iteration)
    oof_proba[val_idx] = preds
    clf_models.append((clf, fold_feat_final))
    acc = ((preds > 0.5) == y_val).mean()
    val_dates = labeled.iloc[val_idx]["date"].astype(str)
    print(f"  {fold_name}: n_val={len(val_idx)}, accuracy={acc:.4f}")

valid_mask = ~np.isnan(oof_proba)
print(f"OOF accuracy (where computed): {((oof_proba[valid_mask] > 0.5) == labeled['is_active_label'].values[valid_mask]).mean():.4f}")

# ── Stage 2: Regressor ────────────────────────────────────────────────────────
print("\n--- Stage 2: Regressor (active shifts only, time-based CV) ---")
active_mask_all = labeled["is_active_label"] == 1
active_df = labeled[active_mask_all].copy()
print(f"Active shifts: {len(active_df)}")

oof_reg    = np.full(len(active_df), np.nan)
reg_models = []
fold_rmses = []

active_folds = make_date_folds(active_df)

for fold_name, tr_mask, val_mask in active_folds:
    tr_idx  = np.where(tr_mask.values)[0]
    val_idx = np.where(val_mask.values)[0]
    if len(val_idx) == 0:
        continue

    tr_data = active_df.iloc[tr_idx]
    fold_global_mean_act = tr_data["acons"].mean() if len(tr_data) > 0 else global_mean

    # Fold-safe stats
    veh_shift_fold = (tr_data.groupby(["vehicle", "shift"])["acons"]
                     .agg(["mean", "std"]).reset_index())
    veh_shift_fold.columns = ["vehicle", "shift", "veh_shift_mean_acons", "veh_shift_std_acons"]
    _lph_f = tr_data.copy()
    _lph_f["lph"] = _lph_f["acons"] / (_lph_f["ign_h"] + 1e-6)
    veh_lph_fold = _lph_f.groupby("vehicle")["lph"].median().reset_index()
    veh_lph_fold.rename(columns={"lph": "veh_lph"}, inplace=True)
    _tc_f = tr_data[tr_data["haul_cycles"] > 0].copy()
    if len(_tc_f) > 0:
        _tc_f["fuel_per_trip"] = _tc_f["acons"] / _tc_f["haul_cycles"]
        veh_fpt_fold = _tc_f.groupby("vehicle")["fuel_per_trip"].median().reset_index()
        veh_fpt_fold.rename(columns={"fuel_per_trip": "veh_fuel_per_trip"}, inplace=True)
    else:
        veh_fpt_fold = pd.DataFrame(columns=["vehicle", "veh_fuel_per_trip"])
    op_fold = pd.DataFrame()
    if "operator_id_shift" in tr_data.columns:
        op_fold = (tr_data.dropna(subset=["operator_id_shift"])
                   .groupby("operator_id_shift")["acons"]
                   .agg(["mean", "count"]).reset_index())
        op_fold.columns = ["operator_id_shift", "op_mean_acons_fold", "op_count_fold"]

    def apply_fold_stats_reg(df_part):
        df_part = df_part.copy()
        drop_existing = [c for c in ["veh_shift_mean_acons", "veh_shift_std_acons",
                                      "veh_lph", "veh_fuel_per_trip",
                                      "op_mean_acons_fold", "op_count_fold"] if c in df_part.columns]
        df_part.drop(columns=drop_existing, inplace=True)
        df_part = df_part.merge(veh_shift_fold, on=["vehicle", "shift"], how="left")
        df_part = df_part.merge(veh_lph_fold, on="vehicle", how="left")
        if len(veh_fpt_fold) > 0:
            df_part = df_part.merge(veh_fpt_fold, on="vehicle", how="left")
        else:
            df_part["veh_fuel_per_trip"] = 17.0
        if len(op_fold) > 0:
            df_part = df_part.merge(op_fold, on="operator_id_shift", how="left")
        else:
            df_part["op_mean_acons_fold"] = fold_global_mean_act
            df_part["op_count_fold"] = 0
        df_part["veh_shift_mean_acons"] = df_part["veh_shift_mean_acons"].fillna(fold_global_mean_act)
        df_part["veh_shift_std_acons"]  = df_part["veh_shift_std_acons"].fillna(0)
        df_part["veh_lph"]              = df_part["veh_lph"].fillna(46.0)
        df_part["veh_fuel_per_trip"]    = df_part["veh_fuel_per_trip"].fillna(17.0)
        df_part["op_mean_acons_fold"]   = df_part.get("op_mean_acons_fold", pd.Series(fold_global_mean_act, index=df_part.index)).fillna(fold_global_mean_act)
        df_part["op_count_fold"]        = df_part.get("op_count_fold", pd.Series(0, index=df_part.index)).fillna(0)
        df_part["physics_pred"]         = df_part["veh_lph"] * df_part["ign_h"]
        df_part["trip_pred"]            = df_part["veh_fuel_per_trip"] * df_part["haul_cycles"]
        return df_part

    tr_fold = apply_fold_stats_reg(active_df.iloc[tr_idx])
    val_fold = apply_fold_stats_reg(active_df.iloc[val_idx])

    fold_feat_final = [c for c in feat_cols if c in tr_fold.columns and c in val_fold.columns]
    if "op_mean_acons_fold" in tr_fold.columns:
        fold_feat_final = [c for c in fold_feat_final if c != "op_mean_acons"]
        if "op_mean_acons_fold" not in fold_feat_final:
            fold_feat_final.append("op_mean_acons_fold")

    X_tr  = tr_fold[fold_feat_final].fillna(0)
    y_tr  = active_df.iloc[tr_idx]["acons"].values
    X_val = val_fold[fold_feat_final].fillna(0)
    y_val = active_df.iloc[val_idx]["acons"].values

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    reg = lgb.train(REGR_PARAMS, dtrain, valid_sets=[dval], num_boost_round=5000,
                    callbacks=[lgb.early_stopping(300, verbose=False),
                                lgb.log_evaluation(1000)])
    preds = np.clip(reg.predict(X_val, num_iteration=reg.best_iteration), 0, None)
    oof_reg[val_idx] = preds
    reg_models.append((reg, fold_feat_final))
    fr = rmse(y_val, preds)
    fold_rmses.append(fr)
    print(f"  {fold_name}: n_val={len(val_idx)}, RMSE={fr:.2f}L (iter={reg.best_iteration})")

valid_reg = ~np.isnan(oof_reg)
if valid_reg.sum() > 0:
    oof_rmse_active = rmse(active_df["acons"].values[valid_reg], oof_reg[valid_reg])
    print(f"\nOOF RMSE (active, time-based CV): {oof_rmse_active:.2f}L")
    print(f"Fold range: {min(fold_rmses):.2f}–{max(fold_rmses):.2f}L")
    print(f"v4 OOF (GroupKFold, leaky):   27.13L")
    print(f"v3 LB RMSE (best):            ~30.9L (MSE 957)")
else:
    oof_rmse_active = 999.0
    print("WARNING: No OOF predictions computed")

# Combined OOF
oof_full = np.zeros(len(labeled))
active_indices = np.where(active_mask_all)[0]
for i, ai in enumerate(active_indices):
    if not np.isnan(oof_reg[i]):
        oof_full[ai] = oof_reg[i]
oof_labeled_valid = valid_mask & (oof_full != 0)
if oof_labeled_valid.sum() > 0:
    print(f"OOF RMSE (combined subset):   {rmse(labeled['acons'].values[oof_labeled_valid], oof_full[oof_labeled_valid]):.2f}L")

# Feature importance (from last fold's regressor)
if reg_models:
    last_reg, last_feats = reg_models[-1]
    importance = pd.DataFrame({
        "feature": last_feats,
        "gain": last_reg.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    print("\nTop 25 features (last fold):")
    print(importance.head(25).to_string(index=False))
    importance.to_csv(ROOT / "outputs" / "v5_feature_importance.csv", index=False)

# ── Test predictions ──────────────────────────────────────────────────────────
print("\n=== Generating submission ===")

# For test: map fold-safe column names to global equivalents
test_pred_df = test.copy()
if "op_mean_acons" in test_pred_df.columns:
    test_pred_df["op_mean_acons_fold"] = test_pred_df["op_mean_acons"]
    test_pred_df["op_count_fold"]      = test_pred_df.get("op_count", pd.Series(0, index=test_pred_df.index))
else:
    test_pred_df["op_mean_acons_fold"] = global_mean
    test_pred_df["op_count_fold"]      = 0

test_active_proba = np.mean(
    [m.predict(test_pred_df[[c if c in test_pred_df.columns else "op_mean_acons_fold"
                               for c in fc]].fillna(0), num_iteration=m.best_iteration)
     for m, fc in clf_models], axis=0
)

# Use all models for test predictions, average over folds
test_reg_preds = []
for reg, fc in reg_models:
    X_te = test_pred_df[[c for c in fc if c in test_pred_df.columns]].fillna(0)
    # Ensure all required columns present (fill missing with 0)
    for c in fc:
        if c not in X_te.columns:
            X_te[c] = 0.0
    X_te = X_te[fc].fillna(0)
    pred = np.clip(reg.predict(X_te, num_iteration=reg.best_iteration), 0, None)
    test_reg_preds.append(pred)
test_reg_pred = np.mean(test_reg_preds, axis=0)

# Blend by active probability
test_preds = test_reg_pred.copy()
low_active = test_active_proba < 0.3
mid_active = (test_active_proba >= 0.3) & (test_active_proba < 0.7)
test_preds[low_active] = test_reg_pred[low_active] * test_active_proba[low_active]
test_preds[mid_active] = test_reg_pred[mid_active] * test_active_proba[mid_active]

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
    vs_mean = (train[train["acons"].notna()]
               .groupby(["vehicle", "shift"])["acons"].mean())
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
out_name = f"submission_v5_oof{oof_rmse_active:.2f}_mean{mean_p:.1f}.csv"
out_path = SUBS / out_name
final.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"\nSummary:")
print(f"  OOF RMSE active (time-CV): {oof_rmse_active:.2f}L")
print(f"  v4 OOF (leaky GroupKFold): 27.13L")
print(f"  v3 LB MSE (best so far):   940.99")
