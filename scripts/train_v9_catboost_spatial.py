#!/usr/bin/env python3
"""v9: Two-stage CatBoost (classifier + regressor) + spatial/cycle features.

Extends v8 by merging `outputs/spatial_features/{train,test}_spatial.csv`.

Reads:
  - ckpts/train_v5.parquet, ckpts/test_v5.parquet
  - outputs/spatial_features/train_spatial.csv, outputs/spatial_features/test_spatial.csv
  - data/smry_*_train_ordered.csv (labels)
  - data/rfid_refuels_*.parquet (optional)

Writes:
  - submissions/submission_v9_cat_spatial_mse{X}_mean{Y}.csv
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
SPATIAL = ROOT / "outputs" / "spatial_features"
SUBS = ROOT / "submissions"
SUBS.mkdir(exist_ok=True)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: list[str]):
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


def add_global_stats(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
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

    for name, df in [("train", train), ("test", test)]:
        df2 = safe_merge(df, veh_shift_stats, on=["vehicle", "shift"])
        df2 = safe_merge(df2, veh_stats, on=["vehicle"])
        df2 = safe_merge(df2, veh_fpt, on=["vehicle"])
        df2["veh_shift_mean_acons"] = df2["veh_shift_mean_acons"].fillna(global_mean)
        df2["veh_shift_std_acons"] = df2["veh_shift_std_acons"].fillna(0)
        df2["veh_lph"] = df2["veh_lph"].fillna(46.0)
        df2["veh_fuel_per_trip"] = df2["veh_fuel_per_trip"].fillna(17.0)
        df2["physics_pred"] = df2["veh_lph"] * df2["ign_h"]
        df2["trip_pred"] = df2["veh_fuel_per_trip"] * df2["haul_cycles"]

        if name == "train":
            train = df2
        else:
            test = df2

    return train, test, global_mean


def make_time_holdout_mask(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    d = pd.to_datetime(df["date"])
    train_mask = ((d.dt.month.isin([1, 2]) & (d.dt.day <= 15)) | (d.dt.month == 3))
    val_mask = (d.dt.month.isin([1, 2]) & (d.dt.day >= 16) & (d.dt.day <= 20))
    return train_mask.to_numpy(), val_mask.to_numpy()


def tune_blend(y_true: np.ndarray, p: np.ndarray, r: np.ndarray) -> dict:
    y_true = y_true.astype(float)
    p = np.clip(p.astype(float), 0, 1)
    r = np.clip(r.astype(float), 0, None)

    best = {"name": "", "params": {}, "mse": 1e18}

    for gamma in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        pred = r * (p ** gamma)
        m = mse(y_true, pred)
        if m < best["mse"]:
            best = {"name": "p_pow", "params": {"gamma": gamma}, "mse": m}

    for high_t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        w = np.clip(p / max(high_t, 1e-6), 0, 1)
        pred = r * w
        m = mse(y_true, pred)
        if m < best["mse"]:
            best = {"name": "linear_ramp", "params": {"high_t": high_t}, "mse": m}

    return best


def apply_blend(blend: dict, p: np.ndarray, r: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    r = np.clip(np.asarray(r, dtype=float), 0, None)
    if blend["name"] == "p_pow":
        return r * (p ** float(blend["params"]["gamma"]))
    if blend["name"] == "linear_ramp":
        ht = float(blend["params"]["high_t"])
        return r * np.clip(p / max(ht, 1e-6), 0, 1)
    return r * p


def load_spatial(prefix: str) -> pd.DataFrame:
    path = SPATIAL / f"{prefix}_spatial.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing spatial features: {path}")

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    keys = ["vehicle", "date", "shift"]
    feat_cols = [c for c in df.columns if c not in keys]
    ren = {c: f"sp_{c}" for c in feat_cols}
    return df.rename(columns=ren)


def main():
    print("Loading v5 checkpoints...")
    train = pd.read_parquet(CKPT / "train_v5.parquet")
    test = pd.read_parquet(CKPT / "test_v5.parquet")
    train["date"] = pd.to_datetime(train["date"]).dt.date
    test["date"] = pd.to_datetime(test["date"]).dt.date
    print(f"  train: {train.shape}, test: {test.shape}")

    print("Loading spatial features...")
    sp_tr = load_spatial("train")
    sp_te = load_spatial("test")
    train = safe_merge(train, sp_tr, on=["vehicle", "date", "shift"])
    test = safe_merge(test, sp_te, on=["vehicle", "date", "shift"])
    print(f"  merged: train={train.shape}, test={test.shape}")

    # Labels
    smry_files = sorted(DATA.glob("smry_*_train_ordered.csv"))
    if not smry_files:
        smry_files = sorted(DATA.glob("smry_*.csv"))
    smry = pd.concat([pd.read_csv(f) for f in smry_files], ignore_index=True)
    smry["date"] = pd.to_datetime(smry["date"]).dt.date
    train = safe_merge(train, smry[["vehicle", "date", "shift", "acons"]], on=["vehicle", "date", "shift"])
    print(f"Labeled: {train['acons'].notna().sum()} / {len(train)}")

    # RFID shift aggregates (safe)
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

    # Global stats + physics
    train, test, global_mean = add_global_stats(train, test)

    # v6 derived features
    for df in (train, test):
        df["loaded_mov_h"] = df["frac_loaded_moving"] * df["mov_h"]
        df["empty_mov_h"] = df["frac_empty_moving"] * df["mov_h"]
        df["loaded_km"] = df["frac_loaded_moving"] * df["shift_km"]
        df["empty_km"] = df["frac_empty_moving"] * df["shift_km"]
        df["loaded_ext_v_x_h"] = df["loaded_ext_v_mean"] * df["loaded_mov_h"]
        df["empty_ext_v_x_h"] = df["empty_ext_v_mean"] * df["empty_mov_h"]

    labeled = train[train["acons"].notna()].copy()
    labeled["is_active"] = (labeled["acons"] > 10).astype(int)

    # Feature list: numeric intersection + chosen categoricals
    exclude = {
        "acons",
        "is_active",
        "date",
        # train-only
        "fuel_volume",
        "prod_hr_dpr",
        "idle_hr_dpr",
        "km_dpr",
        "tonnage",
        "hmr_dpr",
        "bd_hr_dpr",
        "maint_hr_dpr",
    }

    cat_candidates = ["vehicle", "shift", "mine_anon", "operator_id_shift"]
    cat_features = [c for c in cat_candidates if c in labeled.columns and c in test.columns]

    def numeric_cols(df: pd.DataFrame) -> set[str]:
        cols = set()
        for c in df.columns:
            if c in exclude or c in cat_candidates:
                continue
            if df[c].dtype == "object":
                continue
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                continue
            cols.add(c)
        return cols

    feats = sorted((numeric_cols(labeled) & numeric_cols(test)) | set(cat_features))
    print(f"Features: {len(feats)} (cat={cat_features})")

    X_all = labeled[feats].copy()
    X_test = test[feats].copy()
    for c in cat_features:
        X_all[c] = X_all[c].astype(str).fillna("NA")
        X_test[c] = X_test[c].astype(str).fillna("NA")

    y = labeled["acons"].to_numpy(dtype=float)
    y_act = labeled["is_active"].to_numpy(dtype=int)

    tr_mask, va_mask = make_time_holdout_mask(labeled)
    if va_mask.sum() < 50:
        tr_mask = np.ones(len(labeled), dtype=bool)
        va_mask = np.zeros(len(labeled), dtype=bool)

    # Stage 1: classifier
    clf = CatBoostClassifier(
        loss_function="Logloss",
        iterations=4500,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=6.0,
        random_seed=42,
        eval_metric="AUC",
        verbose=False,
    )
    pool_tr_c = Pool(X_all.loc[tr_mask], label=y_act[tr_mask], cat_features=cat_features)
    pool_va_c = Pool(X_all.loc[va_mask], label=y_act[va_mask], cat_features=cat_features) if va_mask.any() else None
    clf.fit(pool_tr_c, eval_set=pool_va_c, use_best_model=bool(va_mask.any()))

    if va_mask.any():
        p_val = clf.predict_proba(Pool(X_all.loc[va_mask], cat_features=cat_features))[:, 1]
    else:
        p_val = np.array([])
    p_test = clf.predict_proba(Pool(X_test, cat_features=cat_features))[:, 1]

    # Stage 2: regressor (active only)
    reg = CatBoostRegressor(
        loss_function="RMSE",
        iterations=9000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=False,
    )
    tr_active = tr_mask & (y_act == 1)
    va_active = va_mask & (y_act == 1)
    pool_tr_r = Pool(X_all.loc[tr_active], label=y[tr_active], cat_features=cat_features)
    pool_va_r = Pool(X_all.loc[va_active], label=y[va_active], cat_features=cat_features) if va_active.any() else None
    reg.fit(pool_tr_r, eval_set=pool_va_r, use_best_model=bool(va_active.any()))

    if va_mask.any():
        r_val = reg.predict(Pool(X_all.loc[va_mask], cat_features=cat_features))
    else:
        r_val = np.array([])
    r_test = reg.predict(Pool(X_test, cat_features=cat_features))

    if va_mask.any():
        blend = tune_blend(y[va_mask], p_val, r_val)
        pred_val = np.clip(apply_blend(blend, p_val, r_val), 0, None)
        print("\n=== Holdout Summary (v9) ===")
        print(f"Blend: {blend['name']} {blend['params']} -> MSE={blend['mse']:.2f}")
        print(f"Holdout RMSE: {np.sqrt(blend['mse']):.2f}L")
    else:
        blend = {"name": "p_pow", "params": {"gamma": 1.0}, "mse": float("nan")}

    test_pred = np.clip(apply_blend(blend, p_test, r_test), 0, None)
    test = test.copy()
    test["Predicted"] = test_pred

    # Build submission
    idm = pd.read_csv(DATA / "id_mapping_new.csv")
    idm["date"] = pd.to_datetime(idm["date"]).dt.date
    submission = idm.merge(test[["vehicle", "date", "shift", "Predicted"]], on=["vehicle", "date", "shift"], how="left")
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

    mean_p = float(final["Predicted"].mean())
    out_name = f"submission_v9_cat_spatial_valmse{blend.get('mse', float('nan')):.1f}_mean{mean_p:.1f}.csv"
    out_path = SUBS / out_name
    final.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
