#!/usr/bin/env python3
"""
v11: Two-stage CatBoost trained on v11 features (no DPR contamination).

Key changes from v9/v10:
  - Zero DPR columns (total_trip, operator_id, km_dpr etc. NOT in test → excluded)
  - Uses telemetry_cycles instead of max_total_trip (the core bug fix)
  - Physics features split by load state (loaded vs empty)
  - Reads ckpts/train_v11.parquet + ckpts/test_v11.parquet

Writes: submissions/submission_v11_*.csv
"""
from __future__ import annotations
import datetime as dt
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "ckpts"
SUBS = ROOT / "submissions"
SUBS.mkdir(exist_ok=True)


def mse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred))


def safe_merge(left, right, on):
    keys = on if isinstance(on, list) else [on]
    drop = [c for c in right.columns if c not in keys and c in left.columns]
    return left.merge(right.drop(columns=drop), on=on, how="left")


def assign_shift(ts_series):
    ts = pd.to_datetime(ts_series, utc=False)
    hour = ts.dt.hour
    date = ts.dt.date
    adj = [d - dt.timedelta(days=1) if h >= 22 else d for d, h in zip(date, hour)]
    shift = ["C" if h >= 22 or h < 6 else ("A" if h < 14 else "B") for h in hour]
    return pd.Series(adj, index=ts_series.index), pd.Series(shift, index=ts_series.index)


def add_global_stats(train, test):
    """
    Vehicle history features — computed from training labels only.
    These are SAFE because same vehicles appear in train and test (no leakage).
    Using telemetry_cycles (not DPR haul_cycles) for trip-based stats.
    """
    train_clean = train[train["acons"].notna() & (train["acons"] > 0)].copy()
    global_mean = float(train_clean["acons"].mean())

    # Per-vehicle-shift mean and std
    veh_shift_stats = (
        train_clean.groupby(["vehicle", "shift"], observed=True)["acons"]
        .agg(["mean", "std"]).reset_index()
    )
    veh_shift_stats.columns = ["vehicle", "shift", "veh_shift_mean_acons", "veh_shift_std_acons"]

    # Vehicle litres-per-hour (from ign_h_est)
    _lph = train_clean.copy()
    _lph["lph"] = _lph["acons"] / (_lph["ign_h"] + 1e-6)
    veh_lph = (_lph.groupby("vehicle", observed=True)["lph"]
               .median().reset_index().rename(columns={"lph": "veh_lph"}))

    # Vehicle aggregate stats
    veh_stats = (train_clean.groupby("vehicle", observed=True).agg(
        veh_mean_km        = ("cumdist_km",      "mean"),
        veh_mean_ign_h     = ("ign_h",        "mean"),
        veh_mean_idle_frac = ("idle_fraction",    "mean"),
        veh_mean_speed     = ("speed_mean",       "mean"),
        veh_mean_ext_v     = ("ext_v_mean",       "mean"),
        veh_mean_cycles    = ("telemetry_cycles", "mean"),
    ).reset_index().merge(veh_lph, on="vehicle", how="left"))

    # Fuel per cycle (using telemetry_cycles — valid for test!)
    _tc = train_clean[train_clean["telemetry_cycles"] > 0].copy()
    _tc["fuel_per_cycle"] = _tc["acons"] / _tc["telemetry_cycles"]
    veh_fpc = (_tc.groupby("vehicle", observed=True)["fuel_per_cycle"]
               .median().reset_index().rename(columns={"fuel_per_cycle": "veh_fuel_per_cycle"}))

    for name, df in [("train", train), ("test", test)]:
        df2 = safe_merge(df, veh_shift_stats, on=["vehicle", "shift"])
        df2 = safe_merge(df2, veh_stats, on=["vehicle"])
        df2 = safe_merge(df2, veh_fpc, on=["vehicle"])
        df2["veh_shift_mean_acons"] = df2["veh_shift_mean_acons"].fillna(global_mean)
        df2["veh_shift_std_acons"]  = df2["veh_shift_std_acons"].fillna(0)
        df2["veh_lph"]              = df2["veh_lph"].fillna(46.0)
        df2["veh_fuel_per_cycle"]   = df2["veh_fuel_per_cycle"].fillna(17.0)

        # Physics predictions (both use telemetry-safe features)
        df2["physics_pred"] = df2["veh_lph"] * df2["ign_h"]
        df2["cycle_pred"]   = df2["veh_fuel_per_cycle"] * df2["telemetry_cycles"]

        if name == "train":
            train = df2
        else:
            test = df2

    return train, test, global_mean


def make_time_holdout_mask(df):
    """Train on Jan/Feb 1-15 + Mar; validate on Jan/Feb 16-20."""
    d = pd.to_datetime(df["date"])
    train_mask = ((d.dt.month.isin([1, 2]) & (d.dt.day <= 15)) | (d.dt.month == 3))
    val_mask   = (d.dt.month.isin([1, 2]) & (d.dt.day >= 16) & (d.dt.day <= 20))
    return train_mask.to_numpy(), val_mask.to_numpy()


def tune_blend(y_true, p, r):
    y_true = y_true.astype(float)
    p = np.clip(p.astype(float), 0, 1)
    r = np.clip(r.astype(float), 0, None)
    best = {"name": "p_pow", "params": {"gamma": 1.0}, "mse": 1e18}

    for gamma in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]:
        pred = r * (p ** gamma)
        m = mse(y_true, pred)
        if m < best["mse"]:
            best = {"name": "p_pow", "params": {"gamma": gamma}, "mse": m}

    for high_t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        w = np.clip(p / max(high_t, 1e-6), 0, 1)
        pred = r * w
        m = mse(y_true, pred)
        if m < best["mse"]:
            best = {"name": "linear_ramp", "params": {"high_t": high_t}, "mse": m}

    return best


def apply_blend(blend, p, r):
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    r = np.clip(np.asarray(r, dtype=float), 0, None)
    if blend["name"] == "p_pow":
        return r * (p ** float(blend["params"]["gamma"]))
    if blend["name"] == "linear_ramp":
        ht = float(blend["params"]["high_t"])
        return r * np.clip(p / max(ht, 1e-6), 0, 1)
    return r * p


def main():
    print("Loading v11 checkpoints...")
    train = pd.read_parquet(CKPT / "train_v11.parquet")
    test  = pd.read_parquet(CKPT / "test_v11.parquet")
    train["date"] = pd.to_datetime(train["date"]).dt.date
    test["date"]  = pd.to_datetime(test["date"]).dt.date
    print(f"  train: {train.shape}, test: {test.shape}")

    # Labels
    smry_files = sorted(DATA.glob("smry_*_train_ordered.csv"))
    if not smry_files:
        smry_files = sorted(DATA.glob("smry_*.csv"))
    smry = pd.concat([pd.read_csv(f) for f in smry_files], ignore_index=True)
    smry["date"] = pd.to_datetime(smry["date"]).dt.date
    train = safe_merge(train, smry[["vehicle", "date", "shift", "acons"]],
                       on=["vehicle", "date", "shift"])
    print(f"  Labeled: {train['acons'].notna().sum()} / {len(train)}")

    # RFID refuels (safe: available for both periods)
    rfid_files = sorted(DATA.glob("rfid_refuels_*.parquet"))
    if rfid_files:
        rfid = pd.concat([pd.read_parquet(f) for f in rfid_files], ignore_index=True)
        rfid["ts"] = pd.to_datetime(rfid["ts"], utc=False)
        rfid["adj_date"], rfid["shift"] = assign_shift(rfid["ts"])
        rfid_agg = (rfid.groupby(["vehicle", "adj_date", "shift"], observed=True)
                    .agg(rfid_liters=("litres", "sum"), rfid_events=("litres", "count"))
                    .reset_index().rename(columns={"adj_date": "date"}))
        rfid_agg["date"] = pd.to_datetime(rfid_agg["date"]).dt.date
        train = safe_merge(train, rfid_agg, on=["vehicle", "date", "shift"])
        test  = safe_merge(test,  rfid_agg, on=["vehicle", "date", "shift"])
        for df in (train, test):
            df["rfid_liters"] = df["rfid_liters"].fillna(0)
            df["rfid_events"] = df["rfid_events"].fillna(0)
        print("  RFID features added")

    # Global vehicle stats (computed from training labels, safe for test)
    train, test, global_mean = add_global_stats(train, test)

    # ── Feature list ──────────────────────────────────────────────────────────
    # Explicit DPR columns to exclude (not in test telemetry)
    DPR_EXCLUDE = {
        # DPR columns in training telemetry but NOT in test
        "max_total_trip", "min_total_trip", "n_trips_from_total",
        "haul_cycles",  # old proxy may be DPR-derived
        "km_dpr", "prod_hr_dpr", "idle_hr_dpr", "maint_hr_dpr",
        "bd_hr_dpr", "hmr_dpr", "tonnage", "rain_loss", "dense_fog",
        "operator_id_shift",  # DPR-only operator_id
        "loading_visits",  # may be DPR-derived in old checkpoints
        # Labels / targets
        "acons", "is_active",
        # Key columns
        "date", "fuel_volume", "lph",
    }

    cat_candidates = ["vehicle", "shift", "mine_anon"]
    cat_features = [c for c in cat_candidates if c in train.columns and c in test.columns]

    def numeric_cols(df):
        cols = set()
        for c in df.columns:
            if c in DPR_EXCLUDE or c in cat_candidates:
                continue
            if df[c].dtype == "object":
                continue
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                continue
            cols.add(c)
        return cols

    labeled = train[train["acons"].notna()].copy()
    labeled["is_active"] = (labeled["acons"] > 10).astype(int)

    feats = sorted((numeric_cols(labeled) & numeric_cols(test)) | set(cat_features))
    print(f"\nFeatures: {len(feats)}")
    print(f"  Categoricals: {cat_features}")

    # Verify telemetry_cycles is in features and not DPR
    assert "telemetry_cycles" in feats, "telemetry_cycles missing!"
    assert "max_total_trip" not in feats, "DPR feature leaked in!"
    assert "physics_pred" in feats, "physics_pred missing!"
    print(f"  telemetry_cycles: OK (NOT DPR)")
    print(f"  physics_pred uses ign_h_est (NOT DPR prod_hr_dpr)")

    X_all  = labeled[feats].copy()
    X_test = test[feats].copy()
    for c in cat_features:
        X_all[c]  = X_all[c].astype(str).fillna("NA")
        X_test[c] = X_test[c].astype(str).fillna("NA")

    y     = labeled["acons"].to_numpy(dtype=float)
    y_act = labeled["is_active"].to_numpy(dtype=int)

    tr_mask, va_mask = make_time_holdout_mask(labeled)
    print(f"  Train: {tr_mask.sum()}, Val: {va_mask.sum()}")
    if va_mask.sum() < 50:
        print("  WARNING: Val set too small, training on all data")
        tr_mask = np.ones(len(labeled), dtype=bool)
        va_mask = np.zeros(len(labeled), dtype=bool)

    # ── Stage 1: Active/inactive classifier ──────────────────────────────────
    print("\nTraining Stage 1 (classifier)...")
    clf = CatBoostClassifier(
        loss_function="Logloss",
        iterations=4500,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=6.0,
        random_seed=42,
        eval_metric="AUC",
        verbose=200,
    )
    pool_tr_c = Pool(X_all.loc[tr_mask], label=y_act[tr_mask], cat_features=cat_features)
    pool_va_c = Pool(X_all.loc[va_mask], label=y_act[va_mask], cat_features=cat_features) if va_mask.any() else None
    clf.fit(pool_tr_c, eval_set=pool_va_c, use_best_model=bool(va_mask.any()))

    p_test = clf.predict_proba(Pool(X_test, cat_features=cat_features))[:, 1]
    p_val  = clf.predict_proba(Pool(X_all.loc[va_mask], cat_features=cat_features))[:, 1] if va_mask.any() else np.array([])

    # ── Stage 2: RMSE regressor (active shifts only) ──────────────────────────
    print("\nTraining Stage 2 RMSE regressor...")
    reg_rmse = CatBoostRegressor(
        loss_function="RMSE",
        iterations=9000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=500,
    )
    tr_active = tr_mask & (y_act == 1)
    va_active = va_mask & (y_act == 1)
    pool_tr_r = Pool(X_all.loc[tr_active], label=y[tr_active], cat_features=cat_features)
    pool_va_r = Pool(X_all.loc[va_active], label=y[va_active], cat_features=cat_features) if va_active.any() else None
    reg_rmse.fit(pool_tr_r, eval_set=pool_va_r, use_best_model=bool(va_active.any()))

    r_test_rmse = reg_rmse.predict(Pool(X_test, cat_features=cat_features))
    r_val_rmse  = reg_rmse.predict(Pool(X_all.loc[va_mask], cat_features=cat_features)) if va_mask.any() else np.array([])

    # ── Stage 2: Tweedie regressor (robust to zeros) ──────────────────────────
    print("\nTraining Stage 2 Tweedie regressor...")
    reg_tw = CatBoostRegressor(
        loss_function="Tweedie:variance_power=1.5",
        iterations=9000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=500,
    )
    reg_tw.fit(pool_tr_r, eval_set=pool_va_r, use_best_model=bool(va_active.any()))

    r_test_tw = reg_tw.predict(Pool(X_test, cat_features=cat_features))
    r_val_tw  = reg_tw.predict(Pool(X_all.loc[va_mask], cat_features=cat_features)) if va_mask.any() else np.array([])

    # ── Tune blend on holdout ─────────────────────────────────────────────────
    if va_mask.any():
        y_val = y[va_mask]
        blend_rmse = tune_blend(y_val, p_val, r_val_rmse)
        blend_tw   = tune_blend(y_val, p_val, r_val_tw)
        pred_rmse  = np.clip(apply_blend(blend_rmse, p_val, r_val_rmse), 0, None)
        pred_tw    = np.clip(apply_blend(blend_tw,   p_val, r_val_tw),   0, None)

        # Try blending the two regressors
        best_mse = min(blend_rmse["mse"], blend_tw["mse"])
        best_alpha = 0.5
        for alpha in np.arange(0.0, 1.01, 0.1):
            r_blend = alpha * r_val_rmse + (1 - alpha) * r_val_tw
            blend_mix = tune_blend(y_val, p_val, r_blend)
            if blend_mix["mse"] < best_mse:
                best_mse   = blend_mix["mse"]
                best_alpha = alpha
                best_blend_mix = blend_mix

        print(f"\n=== Holdout Summary (v11) ===")
        print(f"RMSE blend: {blend_rmse['name']} {blend_rmse['params']} -> MSE={blend_rmse['mse']:.2f}")
        print(f"Tweedie blend: {blend_tw['name']} {blend_tw['params']} -> MSE={blend_tw['mse']:.2f}")
        print(f"Best mix: alpha={best_alpha:.1f} -> MSE={best_mse:.2f}")

        # Use best combination for test predictions
        r_test_final = best_alpha * r_test_rmse + (1 - best_alpha) * r_test_tw
        if best_mse == blend_rmse["mse"]:
            final_blend = blend_rmse
        elif best_mse == blend_tw["mse"]:
            final_blend = blend_tw
        else:
            final_blend = best_blend_mix
        test_pred = np.clip(apply_blend(final_blend, p_test, r_test_final), 0, None)
        val_mse = best_mse
    else:
        # No holdout: use RMSE with default blend
        final_blend = {"name": "p_pow", "params": {"gamma": 0.6}, "mse": float("nan")}
        test_pred = np.clip(apply_blend(final_blend, p_test, r_test_rmse), 0, None)
        val_mse = float("nan")

    # ── Feature importance ────────────────────────────────────────────────────
    fi = pd.DataFrame({
        "feature": feats,
        "gain": reg_rmse.get_feature_importance(type="PredictionValuesChange")
    }).sort_values("gain", ascending=False)
    fi_path = ROOT / "outputs" / "v11_feature_importance.csv"
    fi_path.parent.mkdir(exist_ok=True)
    fi.to_csv(fi_path, index=False)
    print(f"\nTop 15 features:\n{fi.head(15).to_string(index=False)}")

    # ── Build submission ──────────────────────────────────────────────────────
    test = test.copy()
    test["Predicted"] = test_pred

    idm = pd.read_csv(DATA / "id_mapping_new.csv")
    idm["date"] = pd.to_datetime(idm["date"]).dt.date
    submission = idm.merge(
        test[["vehicle", "date", "shift", "Predicted"]],
        on=["vehicle", "date", "shift"], how="left"
    )
    n_missing = int(submission["Predicted"].isna().sum())
    if n_missing > 0:
        print(f"  Filling {n_missing} missing predictions with vehicle×shift mean")
        labeled_clean = train[train["acons"].notna()]
        vs_mean = labeled_clean.groupby(["vehicle", "shift"], observed=True)["acons"].mean()
        for i in submission[submission["Predicted"].isna()].index:
            key = (submission.at[i, "vehicle"], submission.at[i, "shift"])
            submission.at[i, "Predicted"] = float(vs_mean.get(key, global_mean))

    # Clip to tank capacity
    fleet = pd.read_csv(DATA / "fleet.csv")
    tankcap_map = dict(zip(fleet["vehicle"], fleet["tankcap"]))
    submission["tankcap"] = submission["vehicle"].map(tankcap_map).fillna(1379)
    submission["Predicted"] = submission.apply(
        lambda r: float(np.clip(r["Predicted"], 0, r["tankcap"])), axis=1
    )

    final = submission[["id", "Predicted"]].sort_values("id").reset_index(drop=True)
    mean_p = float(final["Predicted"].mean())
    out_name = f"submission_v11_valmse{val_mse:.1f}_mean{mean_p:.1f}.csv"
    out_path = SUBS / out_name
    final.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Prediction stats: mean={mean_p:.1f}, "
          f"std={final['Predicted'].std():.1f}, "
          f"min={final['Predicted'].min():.1f}, "
          f"max={final['Predicted'].max():.1f}")


if __name__ == "__main__":
    main()
