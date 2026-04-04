#!/usr/bin/env python3
"""v7b: v6 features + tuned two-stage blend (optimize MSE).

This is a fast iteration that DOES NOT rebuild raw telemetry features.

Reads:
  - ckpts/train_v5.parquet, ckpts/test_v5.parquet
  - data/smry_*_train_ordered.csv (labels)
  - data/rfid_refuels_*.parquet (test-available refuel logs)

Writes:
  - outputs/v7b_feature_importance.csv
  - submissions/submission_v7b_mse{X}_rmse{Y}_mean{Z}.csv
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "ckpts"
OUT = ROOT / "outputs"
SUBS = ROOT / "submissions"
SUBS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on):
    keys = on if isinstance(on, list) else [on]
    drop = [c for c in right.columns if c not in keys and c in left.columns]
    return left.merge(right.drop(columns=drop), on=on, how="left")


def assign_shift(ts_series: pd.Series):
    ts = pd.to_datetime(ts_series, utc=False)
    hour = ts.dt.hour
    date = ts.dt.date
    adj = [d - dt.timedelta(days=1) if h >= 22 else d for d, h in zip(date, hour)]
    shift = ["C" if h >= 22 or h < 6 else ("A" if h < 14 else "B") for h in hour]
    return pd.Series(adj, index=ts_series.index), pd.Series(shift, index=ts_series.index)


def get_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    exclude = {
        "vehicle",
        "mine_anon",
        "date",
        "date_dt",
        "shift",
        "acons",
        "is_active_label",
        # train-only / target-adjacent / dpr
        "fuel_volume",
        "prod_hr_dpr",
        "idle_hr_dpr",
        "km_dpr",
        "tonnage",
        "hmr_dpr",
        "operator_id",
        "operator_mode",
        "bd_hr_dpr",
        "maint_hr_dpr",
        # state machine internals
        "loaded_ext_v_sum",
        "loaded_n_pings",
        "empty_ext_v_sum",
        "empty_n_pings",
        "operator_id_shift",
    }

    def ok(df: pd.DataFrame, c: str) -> bool:
        if c in exclude:
            return False
        if df[c].dtype == "object":
            return False
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return False
        return True

    tr_cols = {c for c in train_df.columns if ok(train_df, c)}
    te_cols = {c for c in test_df.columns if ok(test_df, c)}
    return sorted(tr_cols & te_cols)


def tune_blend(y_true: np.ndarray, proba: np.ndarray, reg: np.ndarray) -> dict:
    y_true = y_true.astype(float)
    proba = np.clip(proba.astype(float), 0, 1)
    reg = np.clip(reg.astype(float), 0, None)

    best = {"name": "", "params": {}, "mse": 1e18}

    # pred = reg * p^gamma
    for gamma in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        pred = reg * (proba ** gamma)
        m = mse(y_true, pred)
        if m < best["mse"]:
            best = {"name": "p_pow", "params": {"gamma": gamma}, "mse": m}

    # v4-style: use full reg if p >= high_t else reg * p^gamma
    for high_t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        for gamma in [0.8, 1.0, 1.2]:
            w = np.where(proba >= high_t, 1.0, proba ** gamma)
            pred = reg * w
            m = mse(y_true, pred)
            if m < best["mse"]:
                best = {"name": "piece_high", "params": {"high_t": high_t, "gamma": gamma}, "mse": m}

    # linear ramp to 1 at high_t
    for high_t in [0.50, 0.60, 0.70, 0.80]:
        w = np.clip(proba / max(high_t, 1e-6), 0, 1)
        pred = reg * w
        m = mse(y_true, pred)
        if m < best["mse"]:
            best = {"name": "linear_ramp", "params": {"high_t": high_t}, "mse": m}

    return best


def apply_blend(blend: dict, proba: np.ndarray, reg: np.ndarray) -> np.ndarray:
    proba = np.clip(np.asarray(proba, dtype=float), 0, 1)
    reg = np.clip(np.asarray(reg, dtype=float), 0, None)

    if blend["name"] == "p_pow":
        gamma = float(blend["params"]["gamma"])
        return reg * (proba ** gamma)
    if blend["name"] == "piece_high":
        high_t = float(blend["params"]["high_t"])
        gamma = float(blend["params"]["gamma"])
        w = np.where(proba >= high_t, 1.0, proba ** gamma)
        return reg * w
    if blend["name"] == "linear_ramp":
        high_t = float(blend["params"]["high_t"])
        w = np.clip(proba / max(high_t, 1e-6), 0, 1)
        return reg * w

    return reg * proba


def main():
    print("Loading v5 features...")
    train = pd.read_parquet(CKPT / "train_v5.parquet")
    test = pd.read_parquet(CKPT / "test_v5.parquet")
    train["date"] = pd.to_datetime(train["date"]).dt.date
    test["date"] = pd.to_datetime(test["date"]).dt.date
    print(f"  train: {train.shape}, test: {test.shape}")

    # Labels
    smry_files = sorted(DATA.glob("smry_*_train_ordered.csv"))
    if not smry_files:
        smry_files = sorted(DATA.glob("smry_*.csv"))
    smry = pd.concat([pd.read_csv(f) for f in smry_files], ignore_index=True)
    smry["date"] = pd.to_datetime(smry["date"]).dt.date
    train = safe_merge(train, smry[["vehicle", "date", "shift", "acons"]], on=["vehicle", "date", "shift"])
    print(f"Labeled: {train['acons'].notna().sum()} / {len(train)}")

    # RFID per-shift aggregates (same as v4/v6)
    rfid_files = sorted(DATA.glob("rfid_refuels_*.parquet"))
    if rfid_files:
        rfid = pd.concat([pd.read_parquet(f) for f in rfid_files], ignore_index=True)
        rfid["ts"] = pd.to_datetime(rfid["ts"], utc=False)
        rfid["adj_date"], rfid["shift"] = assign_shift(rfid["ts"])
        rfid_agg = rfid.groupby(["vehicle", "adj_date", "shift"], observed=True).agg(
            rfid_liters=("litres", "sum"),
            rfid_events=("litres", "count"),
        ).reset_index().rename(columns={"adj_date": "date"})
        rfid_agg["date"] = pd.to_datetime(rfid_agg["date"]).dt.date
        train = safe_merge(train, rfid_agg, on=["vehicle", "date", "shift"])
        test = safe_merge(test, rfid_agg, on=["vehicle", "date", "shift"])
        for df in (train, test):
            df["rfid_liters"] = df["rfid_liters"].fillna(0)
            df["rfid_events"] = df["rfid_events"].fillna(0)

    # Global vehicle aggregates (v4/v6 style)
    train_clean = train[train["acons"].notna() & (train["acons"] > 0)].copy()
    global_mean = float(train_clean["acons"].mean())

    veh_shift_stats = (
        train_clean.groupby(["vehicle", "shift"], observed=True)["acons"].agg(["mean", "std"]).reset_index()
    )
    veh_shift_stats.columns = ["vehicle", "shift", "veh_shift_mean_acons", "veh_shift_std_acons"]

    _lph = train_clean.copy()
    _lph["lph"] = _lph["acons"] / (_lph["ign_h"] + 1e-6)
    veh_lph = _lph.groupby("vehicle", observed=True)["lph"].median().reset_index().rename(columns={"lph": "veh_lph"})

    veh_stats = train_clean.groupby("vehicle", observed=True).agg(
        veh_mean_km=("shift_km", "mean"),
        veh_mean_ign_h=("ign_h", "mean"),
        veh_mean_dumps=("haul_cycles", "mean"),
        veh_mean_idle_frac=("idle_fraction", "mean"),
        veh_mean_speed=("speed_mean", "mean"),
        veh_mean_ext_v=("ext_v_mean", "mean"),
    ).reset_index().merge(veh_lph, on="vehicle", how="left")

    _tc = train_clean[train_clean["haul_cycles"] > 0].copy()
    _tc["fuel_per_trip"] = _tc["acons"] / _tc["haul_cycles"]
    veh_fpt = _tc.groupby("vehicle", observed=True)["fuel_per_trip"].median().reset_index().rename(
        columns={"fuel_per_trip": "veh_fuel_per_trip"}
    )

    # Operator encoding (global)
    op_stats = None
    op_key = None
    for col_name in ["operator_mode", "operator_id_shift"]:
        if col_name in train_clean.columns and train_clean[col_name].notna().sum() > 0:
            op_stats = (
                train_clean.dropna(subset=[col_name])
                .groupby(col_name, observed=True)["acons"].agg(["mean", "count"]).reset_index()
            )
            op_stats.columns = [col_name, "op_mean_acons", "op_count"]
            op_key = col_name
            break

    for name, df in [("train", train), ("test", test)]:
        df2 = safe_merge(df, veh_shift_stats, on=["vehicle", "shift"])
        df2 = safe_merge(df2, veh_stats, on="vehicle")
        df2 = safe_merge(df2, veh_fpt, on="vehicle")
        if op_stats is not None and op_key is not None:
            df2 = safe_merge(df2, op_stats, on=op_key)
        df2["veh_shift_mean_acons"] = df2["veh_shift_mean_acons"].fillna(global_mean)
        df2["veh_shift_std_acons"] = df2["veh_shift_std_acons"].fillna(0)
        df2["veh_lph"] = df2["veh_lph"].fillna(46.0)
        df2["veh_fuel_per_trip"] = df2["veh_fuel_per_trip"].fillna(17.0)
        df2["op_mean_acons"] = df2.get("op_mean_acons", pd.Series(global_mean, index=df2.index)).fillna(global_mean)
        df2["op_count"] = df2.get("op_count", pd.Series(0, index=df2.index)).fillna(0)
        df2["physics_pred"] = df2["veh_lph"] * df2["ign_h"]
        df2["trip_pred"] = df2["veh_fuel_per_trip"] * df2["haul_cycles"]
        if name == "train":
            train = df2
        else:
            test = df2

    # v6 derived features
    for df in (train, test):
        df["loaded_mov_h"] = df["frac_loaded_moving"] * df["mov_h"]
        df["empty_mov_h"] = df["frac_empty_moving"] * df["mov_h"]
        df["loaded_km"] = df["frac_loaded_moving"] * df["shift_km"]
        df["empty_km"] = df["frac_empty_moving"] * df["shift_km"]
        df["loaded_ext_v_x_h"] = df["loaded_ext_v_mean"] * df["loaded_mov_h"]
        df["empty_ext_v_x_h"] = df["empty_ext_v_mean"] * df["empty_mov_h"]

    labeled = train[train["acons"].notna()].copy()
    labeled["is_active_label"] = (labeled["acons"] > 10).astype(int)
    y = labeled["acons"].to_numpy(dtype=float)
    y_active = labeled["is_active_label"].to_numpy(dtype=int)
    groups = labeled["vehicle"].astype("category").cat.codes.to_numpy()

    feat_cols = get_features(labeled, test)
    print(f"Features: {len(feat_cols)}")
    X = labeled[feat_cols].fillna(0)
    X_test = test[feat_cols].fillna(0)

    gkf = GroupKFold(n_splits=5)

    CLF_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
    }
    REGR_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 15,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 0.3,
        "verbosity": -1,
        "seed": 42,
    }

    oof_proba = np.zeros(len(labeled), dtype=float)
    oof_reg_all = np.zeros(len(labeled), dtype=float)
    clf_models = []
    reg_models = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_active, groups), 1):
        X_tr = X.iloc[tr_idx]
        X_val = X.iloc[val_idx]

        # Stage 1
        dtrain_c = lgb.Dataset(X_tr, label=y_active[tr_idx])
        dval_c = lgb.Dataset(X_val, label=y_active[val_idx], reference=dtrain_c)
        clf = lgb.train(
            CLF_PARAMS,
            dtrain_c,
            valid_sets=[dval_c],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        p_val = clf.predict(X_val, num_iteration=clf.best_iteration)
        oof_proba[val_idx] = p_val
        clf_models.append(clf)

        # Stage 2 (train on active only)
        tr_active = y_active[tr_idx] == 1
        X_tr_act = X_tr.iloc[np.where(tr_active)[0]]
        y_tr_act = y[tr_idx][tr_active]
        dtrain_r = lgb.Dataset(X_tr_act, label=y_tr_act)

        val_active = y_active[val_idx] == 1
        X_val_act = X_val.iloc[np.where(val_active)[0]]
        y_val_act = y[val_idx][val_active]
        dval_r = lgb.Dataset(X_val_act, label=y_val_act, reference=dtrain_r)
        reg = lgb.train(
            REGR_PARAMS,
            dtrain_r,
            valid_sets=[dval_r],
            num_boost_round=5000,
            callbacks=[lgb.early_stopping(300, verbose=False)],
        )
        r_val = reg.predict(X_val, num_iteration=reg.best_iteration)
        oof_reg_all[val_idx] = r_val
        reg_models.append(reg)

        fold_rmse_active = rmse(y_val_act, np.clip(reg.predict(X_val_act, num_iteration=reg.best_iteration), 0, None)) if val_active.any() else 0.0
        print(f"Fold {fold}: active_rmse={fold_rmse_active:.2f}L")

    blend = tune_blend(y, oof_proba, oof_reg_all)
    oof_pred = np.clip(apply_blend(blend, oof_proba, oof_reg_all), 0, None)
    oof_mse = mse(y, oof_pred)
    oof_rmse_full = rmse(y, oof_pred)
    print("\n=== OOF Summary (v7b) ===")
    print(f"Blend: {blend['name']} {blend['params']} -> MSE={blend['mse']:.2f}")
    print(f"OOF RMSE (full): {oof_rmse_full:.2f}L")

    importance = pd.DataFrame({
        "feature": feat_cols,
        "gain": np.mean([m.feature_importance("gain") for m in reg_models], axis=0),
    }).sort_values("gain", ascending=False)
    importance.to_csv(OUT / "v7b_feature_importance.csv", index=False)

    # Test predictions
    test_proba = np.mean([m.predict(X_test, num_iteration=m.best_iteration) for m in clf_models], axis=0)
    test_reg = np.mean([m.predict(X_test, num_iteration=m.best_iteration) for m in reg_models], axis=0)
    test_pred = np.clip(apply_blend(blend, test_proba, test_reg), 0, None)

    test = test.copy()
    test["Predicted"] = test_pred

    idm = pd.read_csv(DATA / "id_mapping_new.csv")
    idm["date"] = pd.to_datetime(idm["date"]).dt.date
    submission = idm.merge(
        test[["vehicle", "date", "shift", "Predicted"]],
        on=["vehicle", "date", "shift"],
        how="left",
    )

    n_missing = int(submission["Predicted"].isna().sum())
    print(f"Missing predictions: {n_missing} / {len(submission)}")
    if n_missing > 0:
        vs_mean = train[train["acons"].notna()].groupby(["vehicle", "shift"], observed=True)["acons"].mean()
        glob_avg = float(train["acons"].dropna().mean())
        miss = submission[submission["Predicted"].isna()].index
        for i in miss:
            submission.at[i, "Predicted"] = float(vs_mean.get((submission.at[i, "vehicle"], submission.at[i, "shift"]), glob_avg))

    fleet = pd.read_csv(DATA / "fleet.csv")
    tankcap_map = dict(zip(fleet["vehicle"], fleet["tankcap"]))
    submission["tankcap"] = submission["vehicle"].map(tankcap_map).fillna(1379)
    submission["Predicted"] = submission.apply(lambda r: float(np.clip(r["Predicted"], 0, r["tankcap"])), axis=1)

    final = submission[["id", "Predicted"]].sort_values("id").reset_index(drop=True)
    print(f"\nSubmission stats:\n{final['Predicted'].describe()}")

    mean_p = float(final["Predicted"].mean())
    out_name = f"submission_v7b_mse{oof_mse:.1f}_rmse{oof_rmse_full:.2f}_mean{mean_p:.1f}.csv"
    out_path = SUBS / out_name
    final.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
