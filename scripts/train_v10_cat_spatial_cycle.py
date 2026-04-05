#!/usr/bin/env python3
"""v10 (final): Two-stage CatBoost + spatial v1 + fast cycle/idle features.

This is aimed at LB performance (not strict CV):
- Uses ckpts v5 base features
- Merges spatial features (outputs/spatial_features/*) as sp_* columns
- Merges new fast cycle/idle features (outputs/cycle_features_fast/*) as cy_* columns
- Trains 2-stage CatBoost (active classifier + active-only regressor)
- Trains two regressor variants (RMSE + Tweedie) and blends them on a time holdout

Writes:
- submissions/submission_v10_cat_spatial_cycle_valmse{X}_mean{Y}.csv

Note: `valmse` is local holdout MSE, not Kaggle LB.
"""

from __future__ import annotations

import argparse
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
CYCLE = ROOT / "outputs" / "cycle_features_fast"
SUBS = ROOT / "submissions"
SUBS.mkdir(exist_ok=True)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: list[str]):
    drop = [c for c in right.columns if c not in on and c in left.columns]
    return left.merge(right.drop(columns=drop), on=on, how="left")


def assign_shift(ts_series: pd.Series):
    ts = pd.to_datetime(ts_series, utc=False)
    hour = ts.dt.hour
    date = ts.dt.date
    adj = [d - dt.timedelta(days=1) if h >= 22 else d for d, h in zip(date, hour)]
    shift = ["C" if h >= 22 or h < 6 else ("A" if h < 14 else "B") for h in hour]
    return pd.Series(adj, index=ts_series.index), pd.Series(shift, index=ts_series.index)


def make_time_holdout_mask(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    d = pd.to_datetime(df["date"])
    train_mask = ((d.dt.month.isin([1, 2]) & (d.dt.day <= 15)) | (d.dt.month == 3))
    val_mask = (d.dt.month.isin([1, 2]) & (d.dt.day >= 16) & (d.dt.day <= 20))
    return train_mask.to_numpy(), val_mask.to_numpy()


def load_spatial(prefix: str) -> pd.DataFrame:
    path = SPATIAL / f"{prefix}_spatial.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    keys = ["vehicle", "date", "shift"]
    feat_cols = [c for c in df.columns if c not in keys]
    ren = {c: f"sp_{c}" for c in feat_cols}
    return df.rename(columns=ren)


def load_cycle(prefix: str) -> pd.DataFrame:
    path = CYCLE / f"{prefix}_cycle.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


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

    def apply(df: pd.DataFrame) -> pd.DataFrame:
        df2 = safe_merge(df, veh_shift_stats, on=["vehicle", "shift"])
        df2 = safe_merge(df2, veh_stats, on=["vehicle"])
        df2 = safe_merge(df2, veh_fpt, on=["vehicle"])
        df2["veh_shift_mean_acons"] = df2["veh_shift_mean_acons"].fillna(global_mean)
        df2["veh_shift_std_acons"] = df2["veh_shift_std_acons"].fillna(0)
        df2["veh_lph"] = df2["veh_lph"].fillna(46.0)
        df2["veh_fuel_per_trip"] = df2["veh_fuel_per_trip"].fillna(17.0)
        df2["physics_pred"] = df2["veh_lph"] * df2["ign_h"]
        df2["trip_pred"] = df2["veh_fuel_per_trip"] * df2["haul_cycles"]
        return df2

    return apply(train), apply(test), global_mean


def add_derived(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for df in (train, test):
        df["loaded_mov_h"] = df["frac_loaded_moving"] * df["mov_h"]
        df["empty_mov_h"] = df["frac_empty_moving"] * df["mov_h"]
        df["loaded_km"] = df["frac_loaded_moving"] * df["shift_km"]
        df["empty_km"] = df["frac_empty_moving"] * df["shift_km"]
        df["loaded_ext_v_x_h"] = df["loaded_ext_v_mean"] * df["loaded_mov_h"]
        df["empty_ext_v_x_h"] = df["empty_ext_v_mean"] * df["empty_mov_h"]

        # Simple interactions with new cycle features (idle/loaded efficiency)
        if "cy_haul_cycles_proxy" in df.columns:
            df["cy_cycles_per_km"] = df["cy_haul_cycles_proxy"] / (df["shift_km"] + 1e-6)
            df["cy_cycles_per_ign_h"] = df["cy_haul_cycles_proxy"] / (df["ign_h"] + 1e-6)
        if "cy_idle_hours" in df.columns:
            df["cy_idle_frac"] = df["cy_idle_hours"] / (df["ign_h"] + 1e-6)
        if "cy_cycle_time_min_p50" in df.columns:
            df["cy_cycle_time_x_cycles"] = df["cy_cycle_time_min_p50"] * df.get("cy_haul_cycles_proxy", 0)

    return train, test


def fit_two_stage(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    a_tr: np.ndarray,
    X_va: pd.DataFrame,
    cat_features: list[str],
    reg_loss: str,
    seed: int,
    clf_iters: int,
    reg_iters: int,
    learning_rate: float,
    clf_depth: int,
    reg_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    clf = CatBoostClassifier(
        loss_function="Logloss",
        iterations=clf_iters,
        learning_rate=learning_rate,
        depth=clf_depth,
        l2_leaf_reg=6.0,
        random_seed=seed,
        eval_metric="AUC",
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )
    clf.fit(Pool(X_tr, label=a_tr, cat_features=cat_features))
    p_va = clf.predict_proba(Pool(X_va, cat_features=cat_features))[:, 1]

    reg = CatBoostRegressor(
        loss_function=reg_loss,
        iterations=reg_iters,
        learning_rate=learning_rate,
        depth=reg_depth,
        l2_leaf_reg=6.0,
        random_seed=seed,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )
    tr_active = a_tr.astype(bool)
    reg.fit(Pool(X_tr.loc[tr_active], label=y_tr[tr_active], cat_features=cat_features))
    r_va = reg.predict(Pool(X_va, cat_features=cat_features))

    return np.clip(p_va.astype(float), 0, 1), np.clip(np.asarray(r_va, dtype=float), 0, None)


def best_gamma_alpha(y: np.ndarray, p: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    best = (1.0, 1.0, 1e18)
    for g in [0.6, 0.8, 1.0, 1.2, 1.5]:
        base = np.clip(r * (p ** g), 0, None)
        denom = float(np.dot(base, base))
        if denom <= 1e-12:
            continue
        a = float(np.dot(y, base) / denom)
        a = float(np.clip(a, 0.7, 1.3))
        pred = np.clip(a * base, 0, None)
        m = mse(y, pred)
        if m < best[2]:
            best = (float(g), float(a), float(m))
    return best


def best_weight(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    best_w = 0.5
    best_m = 1e18
    for w in np.linspace(0, 1, 41):
        pred = w * a + (1 - w) * b
        m = mse(y, pred)
        if m < best_m:
            best_m = m
            best_w = float(w)
    return best_w, best_m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--losses",
        default="rmse",
        help="Comma-separated regressor losses: rmse,tweedie (default: rmse)",
    )
    p.add_argument("--clf-iters", type=int, default=2500)
    p.add_argument("--reg-iters", type=int, default=5500)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--clf-depth", type=int, default=7)
    p.add_argument("--reg-depth", type=int, default=8)
    p.add_argument(
        "--fast",
        action="store_true",
        help="Faster run: rmse only, fewer iterations",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.fast:
        args.losses = "rmse"
        args.clf_iters = min(args.clf_iters, 1600)
        args.reg_iters = min(args.reg_iters, 2800)

    print("Loading ckpts...")
    train = pd.read_parquet(CKPT / "train_v5.parquet")
    test = pd.read_parquet(CKPT / "test_v5.parquet")
    train["date"] = pd.to_datetime(train["date"]).dt.date
    test["date"] = pd.to_datetime(test["date"]).dt.date

    print("Merging spatial v1...")
    train = safe_merge(train, load_spatial("train"), on=["vehicle", "date", "shift"])
    test = safe_merge(test, load_spatial("test"), on=["vehicle", "date", "shift"])

    print("Merging cycle/idle fast features...")
    train = safe_merge(train, load_cycle("train"), on=["vehicle", "date", "shift"])
    test = safe_merge(test, load_cycle("test"), on=["vehicle", "date", "shift"])
    for c in [col for col in train.columns if col.startswith("cy_")]:
        if c in test.columns:
            train[c] = train[c].fillna(0)
            test[c] = test[c].fillna(0)

    # Labels
    smry_files = sorted(DATA.glob("smry_*_train_ordered.csv"))
    smry = pd.concat([pd.read_csv(f) for f in smry_files], ignore_index=True)
    smry["date"] = pd.to_datetime(smry["date"]).dt.date
    train = safe_merge(train, smry[["vehicle", "date", "shift", "acons"]], on=["vehicle", "date", "shift"])

    # RFID aggregates
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

    labeled = train[train["acons"].notna()].copy()
    labeled["is_active"] = (labeled["acons"] > 10).astype(int)

    labeled, test, global_mean = add_global_stats(labeled, test)
    labeled, test = add_derived(labeled, test)

    # Feature list
    exclude = {
        "acons",
        "is_active",
        "date",
        "fuel_volume",
        "prod_hr_dpr",
        "idle_hr_dpr",
        "km_dpr",
        "tonnage",
        "hmr_dpr",
        "bd_hr_dpr",
        "maint_hr_dpr",
        # explicit leaky summary fields
        "initlev",
        "endlev",
        "arefill",
        "expected_cons_from_levels",
    }

    cat_candidates = ["vehicle", "shift", "mine_anon", "operator_id_shift"]
    cat_features = [c for c in cat_candidates if c in labeled.columns and c in test.columns]

    def numeric_cols(df: pd.DataFrame) -> set[str]:
        cols: set[str] = set()
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
    a = labeled["is_active"].to_numpy(dtype=int)

    tr_mask, va_mask = make_time_holdout_mask(labeled)
    if va_mask.sum() < 50:
        tr_mask = np.ones(len(labeled), dtype=bool)
        va_mask = np.zeros(len(labeled), dtype=bool)

    requested = [s.strip().lower() for s in str(args.losses).split(",") if s.strip()]
    variants: dict[str, str] = {}
    if "rmse" in requested:
        variants["rmse"] = "RMSE"
    if "tweedie" in requested:
        variants["tweedie"] = "Tweedie:variance_power=1.3"
    if not variants:
        raise ValueError("No valid losses requested. Use --losses rmse or --losses rmse,tweedie")

    preds_val = {}
    preds_test = {}
    maps = {}

    for name, loss in variants.items():
        print(f"\nTraining variant: {name} ({loss})")
        p_val, r_val = fit_two_stage(
            X_all.loc[tr_mask],
            y[tr_mask],
            a[tr_mask],
            X_all.loc[va_mask] if va_mask.any() else X_all.loc[tr_mask].iloc[:0],
            cat_features,
            loss,
            seed=42,
            clf_iters=args.clf_iters,
            reg_iters=args.reg_iters,
            learning_rate=args.learning_rate,
            clf_depth=args.clf_depth,
            reg_depth=args.reg_depth,
        )
        p_test, r_test = fit_two_stage(
            X_all,
            y,
            a,
            X_test,
            cat_features,
            loss,
            seed=43,
            clf_iters=args.clf_iters,
            reg_iters=args.reg_iters,
            learning_rate=args.learning_rate,
            clf_depth=args.clf_depth,
            reg_depth=args.reg_depth,
        )

        if va_mask.any():
            g, alpha, m = best_gamma_alpha(y[va_mask], p_val, r_val)
            pred_val = np.clip(alpha * r_val * (p_val ** g), 0, None)
            print(f"  holdout: gamma={g:.2f} alpha={alpha:.3f} -> MSE={m:.2f}")
        else:
            g, alpha, m = 1.0, 1.0, float('nan')
            pred_val = np.array([])

        pred_test = np.clip(alpha * r_test * (p_test ** g), 0, None)

        preds_val[name] = pred_val
        preds_test[name] = pred_test
        maps[name] = (g, alpha, m)

    if len(variants) == 1:
        only = next(iter(variants.keys()))
        w, m_blend = 1.0, maps[only][2]
        test_pred = preds_test[only]
    else:
        if va_mask.any():
            w, m_blend = best_weight(y[va_mask], preds_val["rmse"], preds_val["tweedie"])
            print(f"\nBlend: w_rmse={w:.2f} -> holdout MSE={m_blend:.2f}")
        else:
            w, m_blend = 0.5, float('nan')
        test_pred = np.clip(w * preds_test["rmse"] + (1 - w) * preds_test["tweedie"], 0, None)

    # Build submission
    idm = pd.read_csv(DATA / "id_mapping_new.csv")
    idm["date"] = pd.to_datetime(idm["date"]).dt.date
    test_out = test[["vehicle", "date", "shift"]].copy()
    test_out["Predicted"] = test_pred
    submission = idm.merge(test_out, on=["vehicle", "date", "shift"], how="left")

    n_missing = int(submission["Predicted"].isna().sum())
    print(f"Missing predictions: {n_missing} / {len(submission)}")
    if n_missing > 0:
        vs_mean = labeled.groupby(["vehicle", "shift"], observed=True)["acons"].mean()
        glob_avg = float(labeled["acons"].mean())
        miss = submission[submission["Predicted"].isna()].index
        for i in miss:
            submission.at[i, "Predicted"] = float(vs_mean.get((submission.at[i, "vehicle"], submission.at[i, "shift"]), glob_avg))

    fleet = pd.read_csv(DATA / "fleet.csv")
    tankcap_map = dict(zip(fleet["vehicle"], fleet["tankcap"]))
    submission["tankcap"] = submission["vehicle"].map(tankcap_map).fillna(1379)
    submission["Predicted"] = submission.apply(lambda r: float(np.clip(r["Predicted"], 0, r["tankcap"])), axis=1)
    final = submission[["id", "Predicted"]].sort_values("id").reset_index(drop=True)

    mean_p = float(final["Predicted"].mean())
    out_name = (
        f"submission_v10_cat_spatial_cycle_"
        f"losses{''.join(variants.keys())}_"
        f"valmse{m_blend:.1f}_"
        f"wrmse{w:.2f}_"
        f"mean{mean_p:.1f}.csv"
    )
    out_path = SUBS / out_name
    final.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
