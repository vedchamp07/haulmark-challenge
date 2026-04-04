#!/usr/bin/env python3
"""Train v7 model (fast): reuse extracted checkpoints, tune blend for MSE.

Reads:
  - ckpts/train_v5.parquet
  - ckpts/test_v5.parquet

Uses (train-only labels):
  - data/smry_*_train_ordered.csv (acons)

Adds test-available refuel-history features from:
  - data/rfid_refuels_*.parquet

Writes:
  - outputs/v7_feature_importance.csv
  - submissions/submission_v7_oof<MSE/RMSE>_mean<mean>.csv

Design goals:
  - Keep v4/v6 core idea: GroupKFold by vehicle + global vehicle history
  - Improve test-time robustness via refuel history (not leaky summary fields)
  - Optimize the *actual* objective (MSE) by tuning the 2-stage blend using OOF
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


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on):
    keys = on if isinstance(on, list) else [on]
    drop = [c for c in right.columns if c not in keys and c in left.columns]
    return left.merge(right.drop(columns=drop), on=on, how="left")


def shift_start_ts(date_series: pd.Series, shift_series: pd.Series) -> pd.Series:
    d = pd.to_datetime(date_series)
    shift = shift_series.astype(str)
    hour = shift.map({"A": 6, "B": 14, "C": 22}).fillna(0).astype(int)
    return (d + pd.to_timedelta(hour, unit="h")).astype("datetime64[ns]")


def add_rfid_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rfid_files = sorted(DATA.glob("rfid_refuels_*.parquet"))
    if not rfid_files:
        return train, test

    rfid = pd.concat([pd.read_parquet(f) for f in rfid_files], ignore_index=True)
    keep = [c for c in ["vehicle", "ts", "litres", "fleet_type"] if c in rfid.columns]
    rfid = rfid[keep].copy()
    if "fleet_type" in rfid.columns:
        rfid = rfid[rfid["fleet_type"] == "Dumper"].copy()

    rfid["ts"] = pd.to_datetime(rfid["ts"], errors="coerce", utc=False)
    if getattr(rfid["ts"].dt, "tz", None) is not None:
        rfid["ts"] = rfid["ts"].dt.tz_localize(None)
    rfid = rfid.dropna(subset=["vehicle", "ts", "litres"]).copy()
    rfid["litres"] = pd.to_numeric(rfid["litres"], errors="coerce").fillna(0)
    rfid = rfid.sort_values(["vehicle", "ts"]).reset_index(drop=True)

    def compute_history_feats(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["shift_start_ts"] = shift_start_ts(out["date"], out["shift"])
        if getattr(out["shift_start_ts"].dt, "tz", None) is not None:
            out["shift_start_ts"] = out["shift_start_ts"].dt.tz_localize(None)

        # initialize
        out["hrs_since_last_refuel"] = 1e6
        out["last_refuel_liters"] = 0.0
        out["refuel_liters_last24h"] = 0.0
        out["refuel_count_last24h"] = 0.0

        # Per-vehicle searchsorted on int64 ns
        out_idx = out.index
        for veh, idx in out.groupby("vehicle", sort=False).groups.items():
            q = (
                out.loc[idx, "shift_start_ts"]
                .astype("datetime64[ns]")
                .astype("int64")
                .to_numpy()
            )
            rr = rfid[rfid["vehicle"] == veh]
            if rr.empty:
                continue
            t = rr["ts"].astype("datetime64[ns]").astype("int64").to_numpy()
            v = rr["litres"].to_numpy(dtype=float)

            # right index of first >= q
            right = np.searchsorted(t, q, side="left")
            last_i = right - 1

            has_last = last_i >= 0
            if has_last.any():
                last_ts = np.where(has_last, t[np.clip(last_i, 0, len(t) - 1)], np.nan)
                last_l = np.where(has_last, v[np.clip(last_i, 0, len(v) - 1)], 0.0)
                hrs = np.where(
                    has_last,
                    (q - last_ts) / (1e9 * 3600.0),
                    1e6,
                )
                out.loc[idx, "hrs_since_last_refuel"] = hrs
                out.loc[idx, "last_refuel_liters"] = last_l

            # last 24h window sums
            q24 = q - int(24 * 3600 * 1e9)
            left = np.searchsorted(t, q24, side="left")
            cs = np.cumsum(v)
            cc = np.cumsum(np.ones_like(v))
            sum_right = np.where(right > 0, cs[np.clip(right - 1, 0, len(cs) - 1)], 0.0)
            sum_left = np.where(left > 0, cs[np.clip(left - 1, 0, len(cs) - 1)], 0.0)
            cnt_right = np.where(right > 0, cc[np.clip(right - 1, 0, len(cc) - 1)], 0.0)
            cnt_left = np.where(left > 0, cc[np.clip(left - 1, 0, len(cc) - 1)], 0.0)
            out.loc[idx, "refuel_liters_last24h"] = (sum_right - sum_left)
            out.loc[idx, "refuel_count_last24h"] = (cnt_right - cnt_left)

        out.drop(columns=["shift_start_ts"], inplace=True)
        return out

    train2 = compute_history_feats(train)
    test2 = compute_history_feats(test)

    # Also keep the existing per-shift aggregates used in v4/v6
    def assign_shift(ts_series: pd.Series):
        ts = pd.to_datetime(ts_series, utc=False)
        hour = ts.dt.hour
        date = ts.dt.date
        adj = [d - dt.timedelta(days=1) if h >= 22 else d for d, h in zip(date, hour)]
        shift = ["C" if h >= 22 or h < 6 else ("A" if h < 14 else "B") for h in hour]
        return pd.Series(adj, index=ts_series.index), pd.Series(shift, index=ts_series.index)

    rfid2 = rfid.copy()
    rfid2["adj_date"], rfid2["shift"] = assign_shift(rfid2["ts"])
    rfid_agg = rfid2.groupby(["vehicle", "adj_date", "shift"], observed=True).agg(
        rfid_liters=("litres", "sum"),
        rfid_events=("litres", "count"),
    ).reset_index().rename(columns={"adj_date": "date"})
    rfid_agg["date"] = pd.to_datetime(rfid_agg["date"]).dt.date

    train2 = safe_merge(train2, rfid_agg, on=["vehicle", "date", "shift"])
    test2 = safe_merge(test2, rfid_agg, on=["vehicle", "date", "shift"])
    for df in (train2, test2):
        df["rfid_liters"] = df["rfid_liters"].fillna(0)
        df["rfid_events"] = df["rfid_events"].fillna(0)

    return train2, test2


def get_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    exclude = {
        "vehicle",
        "mine_anon",
        "date",
        "date_dt",
        "shift",
        "acons",
        "is_active_label",
        # dpr columns sometimes present in train-only
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
        "operator_id_shift",  # encoded via op_mean_acons in train_v6/v4 style
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
    """Grid search a small family of monotone blends to minimize MSE."""
    y_true = y_true.astype(float)
    proba = np.clip(proba.astype(float), 0, 1)
    reg = np.clip(reg.astype(float), 0, None)

    best = {"name": "", "params": {}, "mse": 1e18}

    # Family 1: pred = reg * proba^gamma
    for gamma in [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]:
        pred = reg * (proba ** gamma)
        m = mse(y_true, pred)
        if m < best["mse"]:
            best = {"name": "p_pow", "params": {"gamma": gamma}, "mse": m}

    # Family 2: v4-style piecewise (scale by p below high_t)
    for high_t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        for gamma in [0.8, 1.0, 1.2]:
            w = np.where(proba >= high_t, 1.0, proba ** gamma)
            pred = reg * w
            m = mse(y_true, pred)
            if m < best["mse"]:
                best = {"name": "piece_high", "params": {"high_t": high_t, "gamma": gamma}, "mse": m}

    # Family 3: linear ramp to 1 at high_t
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

    # Fallback to v6 behavior
    w = np.where(proba >= 0.7, 1.0, proba)
    return reg * w


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

    # Add RFID features (per-shift + history)
    train, test = add_rfid_features(train, test)

    # Global vehicle + operator stats (v4/v6 style)
    train_clean = train[train["acons"].notna() & (train["acons"] > 0)].copy()
    global_mean = float(train_clean["acons"].mean())

    veh_shift_stats = (
        train_clean.groupby(["vehicle", "shift"], observed=True)["acons"]
        .agg(["mean", "std"]).reset_index()
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

    # Operator encoding (global, consistent with test)
    op_stats = None
    for col_name in ["operator_mode", "operator_id_shift"]:
        if col_name in train_clean.columns and train_clean[col_name].notna().sum() > 0:
            op_stats = (
                train_clean.dropna(subset=[col_name])
                .groupby(col_name, observed=True)["acons"].agg(["mean", "count"]).reset_index()
            )
            op_stats.columns = [col_name, "op_mean_acons", "op_count"]
            op_key = col_name
            break
    else:
        op_key = None

    for df_name, df in [("train", train), ("test", test)]:
        df2 = safe_merge(df, veh_shift_stats, on=["vehicle", "shift"])
        df2 = safe_merge(df2, veh_stats, on="vehicle")
        df2 = safe_merge(df2, veh_fpt, on="vehicle")
        if op_stats is not None and op_key is not None:
            df2 = safe_merge(df2, op_stats, on=op_key)
        # fill
        df2["veh_shift_mean_acons"] = df2["veh_shift_mean_acons"].fillna(global_mean)
        df2["veh_shift_std_acons"] = df2["veh_shift_std_acons"].fillna(0)
        df2["veh_lph"] = df2["veh_lph"].fillna(46.0)
        df2["veh_fuel_per_trip"] = df2["veh_fuel_per_trip"].fillna(17.0)
        df2["op_mean_acons"] = df2.get("op_mean_acons", pd.Series(global_mean, index=df2.index)).fillna(global_mean)
        df2["op_count"] = df2.get("op_count", pd.Series(0, index=df2.index)).fillna(0)
        df2["physics_pred"] = df2["veh_lph"] * df2["ign_h"]
        df2["trip_pred"] = df2["veh_fuel_per_trip"] * df2["haul_cycles"]

        if df_name == "train":
            train = df2
        else:
            test = df2

    # Derived v5 features used by v6
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

    # Two-stage training with OOF reg predictions for ALL rows
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
        y_tr_active = y_active[tr_idx]

        # Stage 1
        dtrain_c = lgb.Dataset(X_tr, label=y_tr_active)
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

        # Stage 2 (train only on active rows of training fold)
        tr_active_mask = y_tr_active == 1
        X_tr_act = X_tr.iloc[np.where(tr_active_mask)[0]]
        y_tr_act = y[tr_idx][tr_active_mask]

        dtrain_r = lgb.Dataset(X_tr_act, label=y_tr_act)
        # Early stop on active-only subset from validation fold
        val_active_mask = y_active[val_idx] == 1
        X_val_act = X_val.iloc[np.where(val_active_mask)[0]]
        y_val_act = y[val_idx][val_active_mask]
        dval_r = lgb.Dataset(X_val_act, label=y_val_act, reference=dtrain_r)
        reg = lgb.train(
            REGR_PARAMS,
            dtrain_r,
            valid_sets=[dval_r],
            num_boost_round=5000,
            callbacks=[lgb.early_stopping(300, verbose=False)],
        )
        r_val_all = reg.predict(X_val, num_iteration=reg.best_iteration)
        oof_reg_all[val_idx] = r_val_all
        reg_models.append(reg)

        # Quick diagnostics
        oof_pred_fold = np.clip(r_val_all, 0, None)
        fold_rmse_active = rmse(y_val_act, np.clip(reg.predict(X_val_act, num_iteration=reg.best_iteration), 0, None)) if val_active_mask.any() else 0.0
        fold_mse_full = mse(y[val_idx], np.clip(p_val, 0, 1) * oof_pred_fold)
        print(f"Fold {fold}: active_rmse={fold_rmse_active:.2f}L, naive_full_mse={fold_mse_full:.1f}")

    # Tune blend for MSE on full label distribution
    blend = tune_blend(y, oof_proba, oof_reg_all)
    oof_pred = apply_blend(blend, oof_proba, oof_reg_all)
    oof_pred = np.clip(oof_pred, 0, None)
    oof_mse = mse(y, oof_pred)
    oof_rmse_full = rmse(y, oof_pred)

    active_mask = y_active == 1
    oof_rmse_active = rmse(y[active_mask], np.clip(oof_reg_all[active_mask], 0, None))

    print("\n=== OOF Summary (v7) ===")
    print(f"Blend: {blend['name']} {blend['params']} -> MSE={blend['mse']:.2f}")
    print(f"OOF MSE (full):   {oof_mse:.2f}")
    print(f"OOF RMSE (full):  {oof_rmse_full:.2f}L")
    print(f"OOF RMSE (active reg only, unblended): {oof_rmse_active:.2f}L")

    # Train-time feature importance from reg models
    importance = pd.DataFrame({
        "feature": feat_cols,
        "gain": np.mean([m.feature_importance("gain") for m in reg_models], axis=0),
    }).sort_values("gain", ascending=False)
    importance.to_csv(OUT / "v7_feature_importance.csv", index=False)

    # Predict test
    test_proba = np.mean(
        [m.predict(X_test, num_iteration=m.best_iteration) for m in clf_models], axis=0
    )
    test_reg = np.mean(
        [m.predict(X_test, num_iteration=m.best_iteration) for m in reg_models], axis=0
    )
    test_pred = apply_blend(blend, test_proba, test_reg)
    test_pred = np.clip(test_pred, 0, None)
    test = test.copy()
    test["Predicted"] = test_pred

    # Build submission
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
            veh = submission.at[i, "vehicle"]
            sh = submission.at[i, "shift"]
            submission.at[i, "Predicted"] = float(vs_mean.get((veh, sh), glob_avg))

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

    mean_p = float(final["Predicted"].mean())
    out_name = f"submission_v7_mse{oof_mse:.1f}_rmse{oof_rmse_full:.2f}_mean{mean_p:.1f}.csv"
    out_path = SUBS / out_name
    final.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
    raise SystemExit(0)


def shift_start_ts(date_s: pd.Series, shift_s: pd.Series) -> pd.Series:
    """Build a naive local shift-start timestamp from operational `date` + `shift`."""
    d = pd.to_datetime(date_s)
    hour = shift_s.map({"A": 6, "B": 14, "C": 22}).fillna(6).astype(int)
    out = (d + pd.to_timedelta(hour, unit="h")).dt.tz_localize(None)
    # Ensure stable dtype for merge_asof (pandas is strict about resolution)
    return pd.to_datetime(out).astype("datetime64[ns]")


def add_vehicle_code(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_veh = pd.concat([train[["vehicle"]], test[["vehicle"]]], ignore_index=True)
    codes, uniques = pd.factorize(all_veh["vehicle"].astype(str), sort=True)
    train = train.copy()
    test = test.copy()
    train["vehicle_code"] = codes[: len(train)].astype(np.int16)
    test["vehicle_code"] = codes[len(train) :].astype(np.int16)
    return train, test


def add_refuel_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add RFID refuel features that are available in both train and test.

    Adds:
      - rfid_liters / rfid_events / rfid_liters_max within the shift
      - hrs_since_last_refuel at shift start
      - liters_last_24h and events_last_24h at shift start

    Uses all `data/rfid_refuels_*.parquet` files if present.
    """

    rfid_files = sorted(DATA.glob("rfid_refuels_*.parquet"))
    if not rfid_files:
        return train, test

    rfid = pd.concat([pd.read_parquet(f) for f in rfid_files], ignore_index=True)
    if "vehicle" not in rfid.columns or "ts" not in rfid.columns or "litres" not in rfid.columns:
        return train, test

    rfid = rfid.copy()
    rfid["vehicle"] = rfid["vehicle"].astype(str)
    rfid["ts"] = pd.to_datetime(rfid["ts"], errors="coerce", utc=False).dt.tz_localize(None)
    rfid["ts"] = pd.to_datetime(rfid["ts"]).astype("datetime64[ns]")
    rfid["litres"] = pd.to_numeric(rfid["litres"], errors="coerce").fillna(0.0).clip(0)
    rfid = rfid.dropna(subset=["ts"]).sort_values(["vehicle", "ts"]).reset_index(drop=True)

    # Shift assignment for in-shift aggregates (same logic as v4/v6)
    hour = rfid["ts"].dt.hour
    rfid["shift"] = np.where((hour >= 6) & (hour < 14), "A", np.where((hour >= 14) & (hour < 22), "B", "C"))
    day0 = rfid["ts"].dt.normalize()
    nc = (rfid["shift"] == "C") & (hour < 6)
    rfid["date"] = (day0 - pd.to_timedelta(nc.astype("int8"), unit="D")).dt.date

    in_shift = (
        rfid.groupby(["vehicle", "date", "shift"], observed=True)
        .agg(
            rfid_liters=("litres", "sum"),
            rfid_events=("litres", "count"),
            rfid_liters_max=("litres", "max"),
        )
        .reset_index()
    )
    in_shift["date"] = pd.to_datetime(in_shift["date"]).dt.date

    # Refuel history: cumulative sums per vehicle
    rfid["cum_liters"] = rfid.groupby("vehicle", observed=True)["litres"].cumsum().astype(float)
    rfid["cum_events"] = rfid.groupby("vehicle", observed=True).cumcount().astype(np.int32) + 1

    def add_history(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out["shift_start"] = shift_start_ts(pd.to_datetime(out["date"]), out["shift"].astype(str))
        out["shift_start"] = pd.to_datetime(out["shift_start"]).astype("datetime64[ns]")
        out["__row_id__"] = np.arange(len(out), dtype=np.int32)

        ref = rfid[["vehicle", "ts", "cum_liters", "cum_events"]].sort_values(["vehicle", "ts"])

        def merge_asof_per_vehicle(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str) -> pd.DataFrame:
            parts = []
            for veh, lsub in left.groupby("vehicle", observed=True, sort=False):
                rsub = right[right["vehicle"] == veh]
                lsub = lsub.sort_values(left_on)
                rsub = rsub.sort_values(right_on)
                if len(rsub) == 0:
                    merged = lsub.copy()
                    for c in ["ts", "cum_liters", "cum_events"]:
                        merged[c] = np.nan
                else:
                    # Vehicle is constant in rsub; keep only left-side `vehicle` to avoid suffixes.
                    rsub = rsub.drop(columns=["vehicle"], errors="ignore")
                    merged = pd.merge_asof(
                        lsub,
                        rsub,
                        left_on=left_on,
                        right_on=right_on,
                        direction="backward",
                        allow_exact_matches=True,
                    )
                parts.append(merged)
            outm = pd.concat(parts, ignore_index=True)
            return outm

        # As-of join for last refuel before shift start
        last = merge_asof_per_vehicle(out, ref, "shift_start", "ts")
        last_ts = last["ts"]
        last["hrs_since_last_refuel"] = (
            (last["shift_start"] - last_ts).dt.total_seconds() / 3600
        ).where(last_ts.notna(), np.nan)

        # As-of join at shift_start - 24h for rolling window
        out_24 = out.copy()
        out_24["shift_start_24h"] = (out_24["shift_start"] - pd.Timedelta(hours=24)).astype("datetime64[ns]")

        at_start = merge_asof_per_vehicle(out_24, ref, "shift_start", "ts")
        at_24h = merge_asof_per_vehicle(out_24, ref, "shift_start_24h", "ts")

        liters_last_24h = at_start["cum_liters"].fillna(0.0) - at_24h["cum_liters"].fillna(0.0)
        events_last_24h = at_start["cum_events"].fillna(0.0) - at_24h["cum_events"].fillna(0.0)

        last["rfid_liters_last_24h"] = liters_last_24h.clip(lower=0.0)
        last["rfid_events_last_24h"] = events_last_24h.clip(lower=0.0)

        # Clean up / caps
        last["hrs_since_last_refuel"] = last["hrs_since_last_refuel"].fillna(999.0).clip(0, 999.0)
        last.drop(columns=["ts", "cum_liters", "cum_events"], inplace=True)
        last = last.sort_values("__row_id__").drop(columns=["__row_id__"]).reset_index(drop=True)
        return last

    train_h = add_history(train)
    test_h = add_history(test)

    # Merge in-shift aggregates
    train_h = safe_merge(train_h, in_shift, on=["vehicle", "date", "shift"])
    test_h = safe_merge(test_h, in_shift, on=["vehicle", "date", "shift"])

    for df_ in [train_h, test_h]:
        for c in [
            "rfid_liters",
            "rfid_events",
            "rfid_liters_max",
            "rfid_liters_last_24h",
            "rfid_events_last_24h",
        ]:
            if c in df_.columns:
                df_[c] = df_[c].fillna(0.0)

    return train_h, test_h


def maybe_merge_spatial_cycle_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optionally merge richer spatial/cycle outputs if they exist."""

    sp_dir = OUTPUTS / "spatial_features"
    tr_path = sp_dir / "train_spatial.csv"
    te_path = sp_dir / "test_spatial.csv"
    if not tr_path.exists() or not te_path.exists():
        return train, test

    tr = pd.read_csv(tr_path)
    te = pd.read_csv(te_path)
    for df_ in [tr, te]:
        df_["date"] = pd.to_datetime(df_["date"]).dt.date

    key = ["vehicle", "date", "shift"]

    # Only keep telemetry-derived columns (avoid duplicating label/encoding fields).
    # This is intentionally conservative.
    drop_cols = {
        "acons",
        "veh_shift_mean_acons",
        "veh_shift_std_acons",
        "veh_mean_km",
        "veh_std_km",
        "veh_mean_ign_h",
        "veh_mean_idle_frac",
        "veh_mean_cycles",
        "veh_mean_climb",
        "veh_mean_ext_v",
    }

    def select(df: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in df.columns if c in key or c not in drop_cols]
        out = df[keep].copy()
        feat_cols = [c for c in out.columns if c not in key]
        # Prefix to avoid accidental name collisions with ckpt features.
        out.rename(columns={c: f"sp_{c}" for c in feat_cols}, inplace=True)
        return out

    tr_sel = select(tr)
    te_sel = select(te)

    train2 = safe_merge(train, tr_sel, on=key)
    test2 = safe_merge(test, te_sel, on=key)

    # Fill NaNs for new sp_ numeric columns
    new_cols = [c for c in train2.columns if c.startswith("sp_")]
    for df_ in [train2, test2]:
        for c in new_cols:
            if c in df_.columns and df_[c].dtype != "object":
                df_[c] = df_[c].fillna(0)

    print(f"Merged spatial/cycle features: +{len(new_cols)} columns")
    return train2, test2


def scale_from_proba(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """Piecewise-linear scale mapping proba-> [0,1]."""
    p = np.asarray(p)
    if b <= a:
        return np.clip(p, 0, 1)
    s = (p - a) / (b - a)
    return np.clip(s, 0.0, 1.0)


def scale_v6_style(p: np.ndarray, t: float) -> np.ndarray:
    """v6-style mapping: scale = p if p < t else 1."""
    p = np.asarray(p)
    return np.where(p < t, p, 1.0)


def tune_blend(
    y_true: np.ndarray,
    oof_proba: np.ndarray,
    oof_reg: np.ndarray,
    veh_shift_mean: np.ndarray,
) -> dict:
    """Grid-search blend params to minimize overall MSE on OOF.

    We search two blend families:
      1) v6-style: scale = p if p < t else 1
      2) piecewise-linear: scale = clip((p-a)/(b-a), 0, 1)
    and also shrinkage base = alpha*reg + (1-alpha)*veh_shift_mean.
    """

    best: dict = {"mse": float("inf"), "family": "v6", "t": 0.7, "alpha": 1.0}

    alpha_grid = np.round(np.linspace(0.4, 1.0, 7), 2)

    # Family 1: v6-style threshold
    t_grid = np.round(np.linspace(0.4, 0.9, 11), 2)
    for t in t_grid:
        s = scale_v6_style(oof_proba, float(t))
        for alpha in alpha_grid:
            base = alpha * oof_reg + (1.0 - alpha) * veh_shift_mean
            pred = s * base
            cur = mse(y_true, np.clip(pred, 0, None))
            if cur < best["mse"]:
                best = {"mse": cur, "family": "v6", "t": float(t), "alpha": float(alpha)}

    # Family 2: piecewise-linear
    a_grid = np.round(np.linspace(0.05, 0.45, 9), 2)
    b_grid = np.round(np.linspace(0.55, 0.95, 9), 2)
    for a in a_grid:
        for b in b_grid:
            if b <= a + 0.05:
                continue
            s = scale_from_proba(oof_proba, float(a), float(b))
            for alpha in alpha_grid:
                base = alpha * oof_reg + (1.0 - alpha) * veh_shift_mean
                pred = s * base
                cur = mse(y_true, np.clip(pred, 0, None))
                if cur < best["mse"]:
                    best = {
                        "mse": cur,
                        "family": "piecewise",
                        "a": float(a),
                        "b": float(b),
                        "alpha": float(alpha),
                    }

    return best


def main() -> None:
    print("Loading v5 features (checkpoint)...")
    train = pd.read_parquet(CKPT / "train_v5.parquet")
    test = pd.read_parquet(CKPT / "test_v5.parquet")
    train["date"] = pd.to_datetime(train["date"]).dt.date
    test["date"] = pd.to_datetime(test["date"]).dt.date
    print(f"  train: {train.shape}, test: {test.shape}")

    # Labels
    smry = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "smry_*.csv"))], ignore_index=True)
    smry["date"] = pd.to_datetime(smry["date"]).dt.date
    train = safe_merge(train, smry[["vehicle", "date", "shift", "acons"]], on=["vehicle", "date", "shift"])
    print(f"Labeled: {train['acons'].notna().sum()} / {len(train)}")

    # Optional richer spatial/cycle features
    train, test = maybe_merge_spatial_cycle_features(train, test)

    # Refuel features (in-shift + history)
    train, test = add_refuel_features(train, test)

    # Vehicle code as numeric feature
    train, test = add_vehicle_code(train, test)

    # Vehicle stats derived from labels (global, consistent with test)
    labeled_all = train[train["acons"].notna() & (train["acons"] > 0)].copy()
    global_mean = float(labeled_all["acons"].mean())

    veh_shift_stats = (
        labeled_all.groupby(["vehicle", "shift"], observed=True)["acons"]
        .agg(["mean", "std"]).reset_index()
    )
    veh_shift_stats.columns = ["vehicle", "shift", "veh_shift_mean_acons", "veh_shift_std_acons"]

    _lph = labeled_all.copy()
    _lph["lph"] = _lph["acons"] / (_lph["ign_h"] + 1e-6)
    veh_lph = _lph.groupby("vehicle", observed=True)["lph"].median().reset_index().rename(columns={"lph": "veh_lph"})

    veh_stats = (
        labeled_all.groupby("vehicle", observed=True)
        .agg(
            veh_mean_km=("shift_km", "mean"),
            veh_mean_ign_h=("ign_h", "mean"),
            veh_mean_dumps=("haul_cycles", "mean"),
            veh_mean_idle_frac=("idle_fraction", "mean"),
            veh_mean_speed=("speed_mean", "mean"),
            veh_mean_ext_v=("ext_v_mean", "mean"),
        )
        .reset_index()
        .merge(veh_lph, on="vehicle", how="left")
    )

    _tc = labeled_all[labeled_all["haul_cycles"] > 0].copy()
    _tc["fuel_per_trip"] = _tc["acons"] / _tc["haul_cycles"]
    veh_fpt = (
        _tc.groupby("vehicle", observed=True)["fuel_per_trip"].median().reset_index()
        .rename(columns={"fuel_per_trip": "veh_fuel_per_trip"})
    )

    train = safe_merge(train, veh_shift_stats, on=["vehicle", "shift"])
    test = safe_merge(test, veh_shift_stats, on=["vehicle", "shift"])
    train = safe_merge(train, veh_stats, on="vehicle")
    test = safe_merge(test, veh_stats, on="vehicle")
    train = safe_merge(train, veh_fpt, on="vehicle")
    test = safe_merge(test, veh_fpt, on="vehicle")

    for df_ in [train, test]:
        df_["veh_shift_mean_acons"] = df_["veh_shift_mean_acons"].fillna(global_mean)
        df_["veh_shift_std_acons"] = df_["veh_shift_std_acons"].fillna(0)
        df_["veh_lph"] = df_["veh_lph"].fillna(46.0)
        df_["veh_fuel_per_trip"] = df_.get("veh_fuel_per_trip", pd.Series(17.0, index=df_.index)).fillna(17.0)
        df_["physics_pred"] = df_["veh_lph"] * df_["ign_h"]
        df_["trip_pred"] = df_["veh_fuel_per_trip"] * df_["haul_cycles"]

    # Operator encoding (use whichever ID-like column exists)
    op_col = None
    for cand in ["operator_id_shift", "operator_mode"]:
        if cand in labeled_all.columns and labeled_all[cand].notna().sum() > 0:
            op_col = cand
            break

    if op_col is not None:
        op_stats = (
            labeled_all.dropna(subset=[op_col])
            .groupby(op_col, observed=True)["acons"].agg(["mean", "count"]).reset_index()
        )
        op_stats.columns = [op_col, "op_mean_acons", "op_count"]
        train = safe_merge(train, op_stats, on=op_col)
        test = safe_merge(test, op_stats, on=op_col)

    for df_ in [train, test]:
        if "op_mean_acons" not in df_.columns:
            df_["op_mean_acons"] = global_mean
            df_["op_count"] = 0
        df_["op_mean_acons"] = df_["op_mean_acons"].fillna(global_mean)
        df_["op_count"] = df_["op_count"].fillna(0)

    # v5 derived features (keep parity with v6)
    for df_ in [train, test]:
        if "frac_loaded_moving" in df_.columns and "mov_h" in df_.columns:
            df_["loaded_mov_h"] = df_["frac_loaded_moving"] * df_["mov_h"]
            df_["empty_mov_h"] = df_["frac_empty_moving"] * df_["mov_h"]
            df_["loaded_km"] = df_["frac_loaded_moving"] * df_["shift_km"]
            df_["empty_km"] = df_["frac_empty_moving"] * df_["shift_km"]
        if "loaded_ext_v_mean" in df_.columns and "loaded_mov_h" in df_.columns:
            df_["loaded_ext_v_x_h"] = df_["loaded_ext_v_mean"] * df_["loaded_mov_h"]
        if "empty_ext_v_mean" in df_.columns and "empty_mov_h" in df_.columns:
            df_["empty_ext_v_x_h"] = df_["empty_ext_v_mean"] * df_["empty_mov_h"]

    print(f"\nTrain final shape: {train.shape}")
    print(f"Test  final shape: {test.shape}")

    # Feature selection
    EXCLUDE = {
        "vehicle",
        "mine_anon",
        "date",
        "date_dt",
        "shift",
        "shift_start",
        "acons",
        "is_active_label",
        "is_active",
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
        # v5 internals
        "loaded_ext_v_sum",
        "loaded_n_pings",
        "empty_ext_v_sum",
        "empty_n_pings",
        # used only to build op_mean_acons
        "operator_id_shift",
    }

    def get_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
        def ok(df: pd.DataFrame, c: str) -> bool:
            if c in EXCLUDE:
                return False
            if df[c].dtype == "object":
                return False
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return False
            return True

        tr_cols = {c for c in train_df.columns if ok(train_df, c)}
        te_cols = {c for c in test_df.columns if ok(test_df, c)}
        return sorted(tr_cols & te_cols)

    labeled = train[train["acons"].notna()].copy()
    labeled["is_active_label"] = (labeled["acons"] > 10).astype(int)

    feat_cols = get_features(labeled, test)
    print(f"Features: {len(feat_cols)}")

    X = labeled[feat_cols].fillna(0)
    y = labeled["acons"].to_numpy(dtype=float)
    y_active = labeled["is_active_label"].to_numpy(dtype=int)
    groups = labeled["vehicle"].astype("category").cat.codes.to_numpy()

    X_test = test[feat_cols].fillna(0)

    gkf = GroupKFold(n_splits=5)

    # Stage 1: classifier
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

    # Stage 2: regressor (trained on active shifts only)
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

    print("\n=== Two-stage model (tuned blend) ===")
    oof_proba = np.zeros(len(labeled), dtype=float)
    oof_reg = np.zeros(len(labeled), dtype=float)

    clf_models: list[lgb.Booster] = []
    reg_models: list[lgb.Booster] = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_active, groups), 1):
        X_tr = X.iloc[tr_idx]
        X_val = X.iloc[val_idx]
        y_tr_active = y_active[tr_idx]

        # Classifier
        dtrain_c = lgb.Dataset(X_tr, label=y_tr_active)
        dval_c = lgb.Dataset(X_val, label=y_active[val_idx], reference=dtrain_c)
        clf = lgb.train(
            CLF_PARAMS,
            dtrain_c,
            valid_sets=[dval_c],
            num_boost_round=1500,
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        oof_proba[val_idx] = clf.predict(X_val, num_iteration=clf.best_iteration)
        clf_models.append(clf)

        # Regressor on active-only training rows, but predict for ALL val rows.
        active_tr_mask = y_tr_active == 1
        X_tr_act = X_tr.loc[active_tr_mask]
        y_tr_act = y[tr_idx][active_tr_mask]

        dtrain_r = lgb.Dataset(X_tr_act, label=y_tr_act)
        active_val_mask = y_active[val_idx] == 1
        X_val_act = X_val.loc[active_val_mask]
        y_val_act = y[val_idx][active_val_mask]
        dval_r = lgb.Dataset(X_val_act, label=y_val_act, reference=dtrain_r)
        reg = lgb.train(
            REGR_PARAMS,
            dtrain_r,
            valid_sets=[dval_r],
            num_boost_round=6000,
            callbacks=[lgb.early_stopping(400, verbose=False), lgb.log_evaluation(0)],
        )
        val_pred = np.asarray(reg.predict(X_val, num_iteration=reg.best_iteration), dtype=float)
        oof_reg[val_idx] = np.clip(val_pred, 0, None)
        reg_models.append(reg)

        fold_acc = ((oof_proba[val_idx] > 0.5) == y_active[val_idx]).mean()
        fold_rmse_active = rmse(y[val_idx][y_active[val_idx] == 1], oof_reg[val_idx][y_active[val_idx] == 1])
        print(f"  Fold {fold}: clf_acc={fold_acc:.4f} reg_rmse_active={fold_rmse_active:.2f}")

    # Tune blend on OOF
    veh_mean = labeled["veh_shift_mean_acons"].fillna(global_mean).to_numpy(dtype=float)
    best = tune_blend(y, oof_proba, oof_reg, veh_mean)
    if best["family"] == "v6":
        print(f"\nBest blend params: family=v6 t={best['t']:.2f} alpha={best['alpha']:.2f}")
        oof_scale = scale_v6_style(oof_proba, best["t"])
    else:
        print(
            f"\nBest blend params: family=piecewise a={best['a']:.2f} b={best['b']:.2f} alpha={best['alpha']:.2f}"
        )
        oof_scale = scale_from_proba(oof_proba, best["a"], best["b"])
    oof_base = best["alpha"] * oof_reg + (1.0 - best["alpha"]) * veh_mean
    oof_pred = np.clip(oof_scale * oof_base, 0, None)

    oof_mse = mse(y, oof_pred)
    oof_rmse_all = rmse(y, oof_pred)
    oof_rmse_act = rmse(y[y_active == 1], oof_pred[y_active == 1])
    print(f"OOF MSE (all):  {oof_mse:.2f}")
    print(f"OOF RMSE (all): {oof_rmse_all:.2f}L")
    print(f"OOF RMSE (active only): {oof_rmse_act:.2f}L")

    # Feature importance (reg model gain avg)
    importance = pd.DataFrame(
        {
            "feature": feat_cols,
            "gain": np.mean([m.feature_importance("gain") for m in reg_models], axis=0),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(OUTPUTS / "v7_feature_importance.csv", index=False)

    # Test predictions
    test_proba = np.vstack([
        np.asarray(m.predict(X_test, num_iteration=m.best_iteration), dtype=float) for m in clf_models
    ]).mean(axis=0)
    test_reg = np.vstack([
        np.asarray(m.predict(X_test, num_iteration=m.best_iteration), dtype=float) for m in reg_models
    ]).mean(axis=0)
    test_reg = np.clip(test_reg, 0, None)

    test_veh_mean = test["veh_shift_mean_acons"].fillna(global_mean).to_numpy(dtype=float)
    if best["family"] == "v6":
        test_scale = scale_v6_style(test_proba, best["t"])
    else:
        test_scale = scale_from_proba(test_proba, best["a"], best["b"])
    test_base = best["alpha"] * test_reg + (1.0 - best["alpha"]) * test_veh_mean
    test_pred = np.clip(test_scale * test_base, 0, None)

    test = test.copy()
    test["Predicted"] = test_pred

    # Build submission
    idm = pd.read_csv(DATA / "id_mapping_new.csv")
    idm["date"] = pd.to_datetime(idm["date"]).dt.date

    submission = idm.merge(test[["vehicle", "date", "shift", "Predicted"]], on=["vehicle", "date", "shift"], how="left")
    n_missing = int(submission["Predicted"].isna().sum())
    print(f"\nMissing predictions: {n_missing} / {len(submission)}")

    if n_missing > 0:
        vs_mean = train[train["acons"].notna()].groupby(["vehicle", "shift"], observed=True)["acons"].mean()
        glob_avg = float(train["acons"].dropna().mean())
        for idx, row in submission[submission["Predicted"].isna()].iterrows():
            submission.at[idx, "Predicted"] = float(vs_mean.get((row["vehicle"], row["shift"]), glob_avg))

    fleet = pd.read_csv(DATA / "fleet.csv")
    tankcap_map = dict(zip(fleet["vehicle"].astype(str), fleet["tankcap"]))
    submission["tankcap"] = submission["vehicle"].astype(str).map(tankcap_map).fillna(1379)
    submission["Predicted"] = submission.apply(lambda r: float(np.clip(r["Predicted"], 0, r["tankcap"])), axis=1)

    final = submission[["id", "Predicted"]].sort_values("id").reset_index(drop=True)

    print(f"\nSubmission stats:\n{final['Predicted'].describe()}")
    print(f"Zeros:          {(final['Predicted'] == 0).sum()}")
    print(f"Near-zero <10L: {(final['Predicted'] < 10).sum()}")

    out_name = f"submission_v7_mse{oof_mse:.2f}_rmse{oof_rmse_all:.2f}_mean{final['Predicted'].mean():.1f}.csv"
    out_path = SUBS / out_name
    final.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
