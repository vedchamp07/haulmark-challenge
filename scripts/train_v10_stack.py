#!/usr/bin/env python3
"""v10: OOF-stacked two-stage CatBoost ensemble (revolutionary vs incremental).

Why this exists
- Prior versions tuned the classifier/regressor blend on a single time holdout.
- v10 generates **out-of-fold (OOF)** predictions via GroupKFold(by vehicle)
  for multiple fundamentally different base pipelines, then learns blend weights
  to minimize OOF MSE. This is closer to a real stack and is usually much more
  stable for LB.

Base pipelines (all two-stage):
- A: CatBoostClassifier + CatBoostRegressor (RMSE)
- B: CatBoostClassifier + CatBoostRegressor (Tweedie; nonnegative-friendly)

Reads
- ckpts/train_v5.parquet, ckpts/test_v5.parquet
- outputs/spatial_features/train_spatial.csv, outputs/spatial_features/test_spatial.csv
- data/smry_*_train_ordered.csv
- data/rfid_refuels_*.parquet (optional)

Writes
- submissions/submission_v10_stack_valmse{OOF}_mean{MEAN}.csv
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

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
    drop = [c for c in right.columns if c not in on and c in left.columns]
    return left.merge(right.drop(columns=drop), on=on, how="left")


def assign_shift(ts_series: pd.Series):
    ts = pd.to_datetime(ts_series, utc=False)
    hour = ts.dt.hour
    date = ts.dt.date
    adj = [d - dt.timedelta(days=1) if h >= 22 else d for d, h in zip(date, hour)]
    shift = ["C" if h >= 22 or h < 6 else ("A" if h < 14 else "B") for h in hour]
    return pd.Series(adj, index=ts_series.index), pd.Series(shift, index=ts_series.index)


def load_spatial(prefix: str) -> pd.DataFrame:
    path = SPATIAL / f"{prefix}_spatial.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing spatial features: {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    keys = ["vehicle", "date", "shift"]
    feat_cols = [c for c in df.columns if c not in keys]
    ren = {c: f"sp_{c}" for c in feat_cols}
    return df.rename(columns=ren)


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


def add_v6_derived(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for df in (train, test):
        df["loaded_mov_h"] = df["frac_loaded_moving"] * df["mov_h"]
        df["empty_mov_h"] = df["frac_empty_moving"] * df["mov_h"]
        df["loaded_km"] = df["frac_loaded_moving"] * df["shift_km"]
        df["empty_km"] = df["frac_empty_moving"] * df["shift_km"]
        df["loaded_ext_v_x_h"] = df["loaded_ext_v_mean"] * df["loaded_mov_h"]
        df["empty_ext_v_x_h"] = df["empty_ext_v_mean"] * df["empty_mov_h"]
    return train, test


def build_feature_list(labeled: pd.DataFrame, test: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = {
        "acons",
        "is_active",
        "date",
        # train-only / leaky
        "initlev",
        "endlev",
        "arefill",
        "expected_cons_from_levels",
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
    return feats, cat_features


@dataclass
class PipelineSpec:
    name: str
    reg_loss: str


def fit_predict_two_stage(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    act_tr: np.ndarray,
    X_va: pd.DataFrame,
    cat_features: list[str],
    spec: PipelineSpec,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit two-stage model on train split and predict on val split.

    Returns:
      p_active (proba), r_active (reg prediction)
    """
    clf = CatBoostClassifier(
        loss_function="Logloss",
        iterations=3500,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=6.0,
        random_seed=seed,
        eval_metric="AUC",
        verbose=False,
    )

    pool_tr_c = Pool(X_tr, label=act_tr, cat_features=cat_features)
    clf.fit(pool_tr_c)
    p_va = clf.predict_proba(Pool(X_va, cat_features=cat_features))[:, 1]

    reg = CatBoostRegressor(
        loss_function=spec.reg_loss,
        iterations=7000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=seed,
        verbose=False,
    )

    tr_active = act_tr.astype(bool)
    pool_tr_r = Pool(X_tr.loc[tr_active], label=y_tr[tr_active], cat_features=cat_features)
    reg.fit(pool_tr_r)
    r_va = reg.predict(Pool(X_va, cat_features=cat_features))

    return p_va.astype(float), np.clip(np.asarray(r_va, dtype=float), 0, None)


def apply_mapping(p: np.ndarray, r: np.ndarray, gamma: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    r = np.clip(np.asarray(r, dtype=float), 0, None)
    return r * (p ** gamma)


def choose_blend_weight(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    best_w = 0.5
    best_m = 1e18
    for w in np.linspace(0, 1, 51):
        pred = w * a + (1 - w) * b
        m = mse(y, pred)
        if m < best_m:
            best_m = m
            best_w = float(w)
    return best_w, best_m


def main():
    print("Loading checkpoints + spatial...")
    train = pd.read_parquet(CKPT / "train_v5.parquet")
    test = pd.read_parquet(CKPT / "test_v5.parquet")
    train["date"] = pd.to_datetime(train["date"]).dt.date
    test["date"] = pd.to_datetime(test["date"]).dt.date

    train = safe_merge(train, load_spatial("train"), on=["vehicle", "date", "shift"])
    test = safe_merge(test, load_spatial("test"), on=["vehicle", "date", "shift"])

    # Labels
    smry_files = sorted(DATA.glob("smry_*_train_ordered.csv"))
    if not smry_files:
        smry_files = sorted(DATA.glob("smry_*.csv"))
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

    # Global stats + derived
    labeled, test, _ = add_global_stats(labeled, test)
    labeled, test = add_v6_derived(labeled, test)

    feats, cat_features = build_feature_list(labeled, test)
    print(f"Features: {len(feats)} (cat={cat_features})")

    X_all = labeled[feats].copy()
    X_test = test[feats].copy()
    for c in cat_features:
        X_all[c] = X_all[c].astype(str).fillna("NA")
        X_test[c] = X_test[c].astype(str).fillna("NA")

    y = labeled["acons"].to_numpy(dtype=float)
    act = labeled["is_active"].to_numpy(dtype=int)
    groups = labeled["vehicle"].astype(str).to_numpy()

    specs = [
        PipelineSpec(name="rmse", reg_loss="RMSE"),
        PipelineSpec(name="tweedie", reg_loss="Tweedie:variance_power=1.3"),
    ]

    gammas = [0.6, 0.8, 1.0, 1.2, 1.5]

    oof_preds: dict[str, dict[float, np.ndarray]] = {s.name: {g: np.zeros(len(labeled), dtype=float) for g in gammas} for s in specs}

    print("Building OOF predictions (GroupKFold by vehicle)...")
    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y, groups=groups), start=1):
        X_tr = X_all.iloc[tr_idx]
        y_tr = y[tr_idx]
        a_tr = act[tr_idx]
        X_va = X_all.iloc[va_idx]

        for spec in specs:
            p_va, r_va = fit_predict_two_stage(X_tr, y_tr, a_tr, X_va, cat_features, spec, seed=42 + fold)
            for gamma in gammas:
                oof_preds[spec.name][gamma][va_idx] = np.clip(apply_mapping(p_va, r_va, gamma), 0, None)

        print(f"  fold {fold}/5 done")

    # Pick best gamma per spec
    best_per_spec: dict[str, tuple[float, float]] = {}
    for spec in specs:
        best_g = None
        best_m = 1e18
        for gamma in gammas:
            m = mse(y, oof_preds[spec.name][gamma])
            if m < best_m:
                best_m = m
                best_g = gamma
        assert best_g is not None
        best_per_spec[spec.name] = (float(best_g), float(best_m))
        print(f"OOF {spec.name}: best gamma={best_g} -> MSE={best_m:.2f}")

    # Blend A/B via grid search weight
    a_gamma = best_per_spec["rmse"][0]
    b_gamma = best_per_spec["tweedie"][0]
    oof_a = oof_preds["rmse"][a_gamma]
    oof_b = oof_preds["tweedie"][b_gamma]
    w_a, oof_m = choose_blend_weight(y, oof_a, oof_b)
    print(f"OOF blend: w_rmse={w_a:.2f}, w_tweedie={1-w_a:.2f} -> MSE={oof_m:.2f}")

    # Train full models for test predictions
    print("Training full models for test predictions...")
    test_preds = {}
    for spec in specs:
        # Fit on all labeled
        p_te, r_te = fit_predict_two_stage(X_all, y, act, X_test, cat_features, spec, seed=2026)
        gamma = best_per_spec[spec.name][0]
        test_preds[spec.name] = np.clip(apply_mapping(p_te, r_te, gamma), 0, None)

    test_pred = np.clip(w_a * test_preds["rmse"] + (1 - w_a) * test_preds["tweedie"], 0, None)

    test_out = test.copy()
    test_out["Predicted"] = test_pred

    # Build submission
    idm = pd.read_csv(DATA / "id_mapping_new.csv")
    idm["date"] = pd.to_datetime(idm["date"]).dt.date
    submission = idm.merge(test_out[["vehicle", "date", "shift", "Predicted"]], on=["vehicle", "date", "shift"], how="left")

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
    out_name = f"submission_v10_stack_valmse{oof_m:.1f}_mean{mean_p:.1f}.csv"
    out_path = SUBS / out_name
    final.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
