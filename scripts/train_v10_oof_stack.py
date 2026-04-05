#!/usr/bin/env python3
"""v10: OOF-stacked two-stage CatBoost with spatial + idle-derived geo features.

Goal
- Produce a materially stronger submission than v8/v9 by doing *proper* blending:
  - Generate out-of-fold (OOF) predictions with GroupKFold by vehicle
  - Optimize each two-stage model's mapping (gamma + scale) on OOF MSE
  - Optimize blend weight across models on OOF MSE

Models (two-stage)
- RMSE regressor (good general fit)
- Tweedie regressor (more robust for nonnegative / heavy-tailed targets)

Reads
- ckpts/train_v5.parquet, ckpts/test_v5.parquet
- outputs/spatial_features/train_spatial.csv, outputs/spatial_features/test_spatial.csv
- data/smry_*_train_ordered.csv
- data/rfid_refuels_*.parquet (optional)

Writes
- submissions/submission_v10_oofstack_valmse{OOF}_mean{MEAN}.csv
- submissions/submission_v10_rmse_base_valmse{OOF}_mean{MEAN}.csv
- submissions/submission_v10_tweedie_base_valmse{OOF}_mean{MEAN}.csv

Notes
- Filename `valmse` is *local OOF MSE*, not Kaggle LB MSE.
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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


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


def add_spatial_idle_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def has(*cols: str) -> bool:
        return all(c in df.columns for c in cols)

    # Idle/moving hours within zones (better than raw idle_fraction)
    if has("sp_idle_hours", "sp_frac_in_load_zone"):
        df["sp_idle_load_h"] = df["sp_idle_hours"] * df["sp_frac_in_load_zone"]
    if has("sp_idle_hours", "sp_frac_in_dump_zone"):
        df["sp_idle_dump_h"] = df["sp_idle_hours"] * df["sp_frac_in_dump_zone"]
    if has("sp_idle_hours", "sp_frac_on_haul_road"):
        df["sp_idle_haul_h"] = df["sp_idle_hours"] * df["sp_frac_on_haul_road"]

    if has("sp_moving_hours", "sp_frac_in_load_zone"):
        df["sp_mov_load_h"] = df["sp_moving_hours"] * df["sp_frac_in_load_zone"]
    if has("sp_moving_hours", "sp_frac_in_dump_zone"):
        df["sp_mov_dump_h"] = df["sp_moving_hours"] * df["sp_frac_in_dump_zone"]
    if has("sp_moving_hours", "sp_frac_on_haul_road"):
        df["sp_mov_haul_h"] = df["sp_moving_hours"] * df["sp_frac_on_haul_road"]

    # Transition rates (captures operational intensity)
    if has("sp_dump_zone_transitions", "sp_ignition_on_hours"):
        df["sp_dump_trans_per_ign_h"] = df["sp_dump_zone_transitions"] / (df["sp_ignition_on_hours"] + 1e-6)
    if has("sp_load_zone_transitions", "sp_ignition_on_hours"):
        df["sp_load_trans_per_ign_h"] = df["sp_load_zone_transitions"] / (df["sp_ignition_on_hours"] + 1e-6)
    if has("sp_haul_road_transitions", "sp_ignition_on_hours"):
        df["sp_haul_trans_per_ign_h"] = df["sp_haul_road_transitions"] / (df["sp_ignition_on_hours"] + 1e-6)

    # Cycle intensity
    if has("sp_haul_cycles_spatial", "sp_shift_km"):
        df["sp_cycles_per_km"] = df["sp_haul_cycles_spatial"] / (df["sp_shift_km"] + 1e-6)
    if has("sp_haul_cycles_spatial", "sp_moving_hours"):
        df["sp_cycles_per_mov_h"] = df["sp_haul_cycles_spatial"] / (df["sp_moving_hours"] + 1e-6)

    # Near-excavator dwell as loading proxy
    if has("sp_frac_near_excavator", "sp_ignition_on_hours"):
        df["sp_near_exc_h"] = df["sp_frac_near_excavator"] * df["sp_ignition_on_hours"]

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
        # explicit leaky summary fields if present
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


@dataclass(frozen=True)
class Spec:
    name: str
    reg_loss: str


def fit_fold_two_stage(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    act_tr: np.ndarray,
    X_va: pd.DataFrame,
    cat_features: list[str],
    spec: Spec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train 2-stage on fold train and return (p_active, reg_pred) on fold val."""

    clf = CatBoostClassifier(
        loss_function="Logloss",
        iterations=2000,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=6.0,
        random_seed=seed,
        eval_metric="AUC",
        od_type="Iter",
        od_wait=200,
        allow_writing_files=False,
        verbose=False,
    )
    pool_tr_c = Pool(X_tr, label=act_tr, cat_features=cat_features)
    pool_va_c = Pool(X_va, label=None, cat_features=cat_features)
    clf.fit(pool_tr_c)
    p_va = clf.predict_proba(pool_va_c)[:, 1]

    reg = CatBoostRegressor(
        loss_function=spec.reg_loss,
        iterations=3500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=seed,
        od_type="Iter",
        od_wait=250,
        allow_writing_files=False,
        verbose=False,
    )

    tr_active = act_tr.astype(bool)
    pool_tr_r = Pool(X_tr.loc[tr_active], label=y_tr[tr_active], cat_features=cat_features)
    reg.fit(pool_tr_r)

    r_va = reg.predict(Pool(X_va, cat_features=cat_features))

    return p_va.astype(float), np.clip(np.asarray(r_va, dtype=float), 0, None)


def mapping(p: np.ndarray, r: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    r = np.clip(np.asarray(r, dtype=float), 0, None)
    return np.clip(alpha * r * (p ** gamma), 0, None)


def best_gamma_alpha(y: np.ndarray, p: np.ndarray, r: np.ndarray, gammas: list[float]) -> tuple[float, float, float]:
    best = (1.0, 1.0, 1e18)

    for g in gammas:
        base = np.clip(r * (np.clip(p, 0, 1) ** g), 0, None)
        denom = float(np.dot(base, base))
        if denom <= 1e-12:
            continue
        alpha = float(np.dot(y, base) / denom)
        alpha = float(np.clip(alpha, 0.5, 1.5))
        pred = np.clip(alpha * base, 0, None)
        m = mse(y, pred)
        if m < best[2]:
            best = (float(g), float(alpha), float(m))

    return best


def best_blend_weight(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    best_w = 0.5
    best_m = 1e18
    for w in np.linspace(0, 1, 51):
        pred = w * a + (1 - w) * b
        m = mse(y, pred)
        if m < best_m:
            best_m = m
            best_w = float(w)
    return best_w, best_m


def train_full_predict(
    X_all: pd.DataFrame,
    y: np.ndarray,
    act: np.ndarray,
    X_test: pd.DataFrame,
    cat_features: list[str],
    spec: Spec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    clf = CatBoostClassifier(
        loss_function="Logloss",
        iterations=2500,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=6.0,
        random_seed=seed,
        eval_metric="AUC",
        allow_writing_files=False,
        verbose=False,
    )
    clf.fit(Pool(X_all, label=act, cat_features=cat_features))
    p_te = clf.predict_proba(Pool(X_test, cat_features=cat_features))[:, 1]

    reg = CatBoostRegressor(
        loss_function=spec.reg_loss,
        iterations=4500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
    )
    active_mask = act.astype(bool)
    reg.fit(Pool(X_all.loc[active_mask], label=y[active_mask], cat_features=cat_features))
    r_te = reg.predict(Pool(X_test, cat_features=cat_features))

    return p_te.astype(float), np.clip(np.asarray(r_te, dtype=float), 0, None)


def write_submission(pred_by_vehicle_date_shift: pd.DataFrame, out_name: str) -> Path:
    idm = pd.read_csv(DATA / "id_mapping_new.csv")
    idm["date"] = pd.to_datetime(idm["date"]).dt.date

    sub = idm.merge(
        pred_by_vehicle_date_shift[["vehicle", "date", "shift", "Predicted"]],
        on=["vehicle", "date", "shift"],
        how="left",
    )

    n_missing = int(sub["Predicted"].isna().sum())
    if n_missing > 0:
        raise RuntimeError(f"Missing predictions after merge: {n_missing}")

    fleet = pd.read_csv(DATA / "fleet.csv")
    tankcap_map = dict(zip(fleet["vehicle"], fleet["tankcap"]))
    sub["tankcap"] = sub["vehicle"].map(tankcap_map).fillna(1379)
    sub["Predicted"] = sub.apply(lambda r: float(np.clip(r["Predicted"], 0, r["tankcap"])), axis=1)

    final = sub[["id", "Predicted"]].sort_values("id").reset_index(drop=True)
    out_path = SUBS / out_name
    final.to_csv(out_path, index=False)
    return out_path


def main():
    print("Loading v5 checkpoints...")
    train = pd.read_parquet(CKPT / "train_v5.parquet")
    test = pd.read_parquet(CKPT / "test_v5.parquet")
    train["date"] = pd.to_datetime(train["date"]).dt.date
    test["date"] = pd.to_datetime(test["date"]).dt.date

    print("Merging spatial features...")
    train = safe_merge(train, load_spatial("train"), on=["vehicle", "date", "shift"])
    test = safe_merge(test, load_spatial("test"), on=["vehicle", "date", "shift"])

    train = add_spatial_idle_derivatives(train)
    test = add_spatial_idle_derivatives(test)

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

    # Global stats + v6 derived
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
        Spec(name="rmse", reg_loss="RMSE"),
        Spec(name="tweedie", reg_loss="Tweedie:variance_power=1.3"),
    ]
    gammas = [0.6, 0.8, 1.0, 1.2, 1.5]

    # OOF p/r storage
    oof_p: dict[str, np.ndarray] = {s.name: np.zeros(len(labeled), dtype=np.float32) for s in specs}
    oof_r: dict[str, np.ndarray] = {s.name: np.zeros(len(labeled), dtype=np.float32) for s in specs}

    print("OOF training (GroupKFold by vehicle)...")
    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y, groups=groups), start=1):
        X_tr = X_all.iloc[tr_idx]
        y_tr = y[tr_idx]
        a_tr = act[tr_idx]
        X_va = X_all.iloc[va_idx]

        for spec in specs:
            p_va, r_va = fit_fold_two_stage(
                X_tr,
                y_tr,
                a_tr,
                X_va,
                cat_features,
                spec,
                seed=1337 + 17 * fold,
            )
            oof_p[spec.name][va_idx] = p_va
            oof_r[spec.name][va_idx] = r_va

        print(f"  fold {fold}/5")

    # Choose best mapping per spec
    mapped_oof: dict[str, np.ndarray] = {}
    best_map: dict[str, tuple[float, float, float]] = {}
    for spec in specs:
        g, a, m = best_gamma_alpha(y, oof_p[spec.name], oof_r[spec.name], gammas)
        best_map[spec.name] = (g, a, m)
        pred = mapping(oof_p[spec.name], oof_r[spec.name], g, a)
        mapped_oof[spec.name] = pred
        print(f"OOF {spec.name}: gamma={g:.2f} alpha={a:.3f} -> MSE={m:.2f} RMSE={rmse(y,pred):.2f}")

    # Blend
    w_rmse, m_blend = best_blend_weight(y, mapped_oof["rmse"], mapped_oof["tweedie"])
    oof_blend = w_rmse * mapped_oof["rmse"] + (1 - w_rmse) * mapped_oof["tweedie"]
    print(f"OOF blend: w_rmse={w_rmse:.2f} -> MSE={m_blend:.2f} RMSE={rmse(y,oof_blend):.2f}")

    # Fit full + predict test
    print("Training full models for test...")
    test_preds: dict[str, np.ndarray] = {}
    for spec in specs:
        p_te, r_te = train_full_predict(X_all, y, act, X_test, cat_features, spec, seed=2026)
        g, a, _m = best_map[spec.name]
        test_preds[spec.name] = mapping(p_te, r_te, g, a)

    test_pred_blend = np.clip(w_rmse * test_preds["rmse"] + (1 - w_rmse) * test_preds["tweedie"], 0, None)

    # Attach predictions to test frame for id mapping merge
    test_out = test[["vehicle", "date", "shift"]].copy()
    test_out["Predicted"] = test_pred_blend

    # Fill missing predictions by vehicle-shift mean from labeled (rare)
    idm = pd.read_csv(DATA / "id_mapping_new.csv")
    idm["date"] = pd.to_datetime(idm["date"]).dt.date
    merged = idm.merge(test_out, on=["vehicle", "date", "shift"], how="left")
    n_missing = int(merged["Predicted"].isna().sum())
    if n_missing > 0:
        vs_mean = labeled.groupby(["vehicle", "shift"], observed=True)["acons"].mean()
        glob_avg = float(labeled["acons"].mean())
        miss = merged[merged["Predicted"].isna()].index
        for i in miss:
            merged.at[i, "Predicted"] = float(vs_mean.get((merged.at[i, "vehicle"], merged.at[i, "shift"]), glob_avg))

    # Tankcap clip + final write
    fleet = pd.read_csv(DATA / "fleet.csv")
    tankcap_map = dict(zip(fleet["vehicle"], fleet["tankcap"]))
    merged["tankcap"] = merged["vehicle"].map(tankcap_map).fillna(1379)
    merged["Predicted"] = merged.apply(lambda r: float(np.clip(r["Predicted"], 0, r["tankcap"])), axis=1)
    final = merged[["id", "Predicted"]].sort_values("id").reset_index(drop=True)

    mean_p = float(final["Predicted"].mean())
    out_path = SUBS / f"submission_v10_oofstack_valmse{m_blend:.1f}_wrmse{w_rmse:.2f}_mean{mean_p:.1f}.csv"
    final.to_csv(out_path, index=False)
    print(f"Saved best blend: {out_path}")

    # Also write best bases (often useful for manual blending on LB)
    for base in ("rmse", "tweedie"):
        pred = test_preds[base]
        tmp = test[["vehicle", "date", "shift"]].copy()
        tmp["Predicted"] = pred
        merged_b = idm.merge(tmp, on=["vehicle", "date", "shift"], how="left")
        miss = int(merged_b["Predicted"].isna().sum())
        if miss > 0:
            vs_mean = labeled.groupby(["vehicle", "shift"], observed=True)["acons"].mean()
            glob_avg = float(labeled["acons"].mean())
            idx = merged_b[merged_b["Predicted"].isna()].index
            for i in idx:
                merged_b.at[i, "Predicted"] = float(vs_mean.get((merged_b.at[i, "vehicle"], merged_b.at[i, "shift"]), glob_avg))
        merged_b["tankcap"] = merged_b["vehicle"].map(tankcap_map).fillna(1379)
        merged_b["Predicted"] = merged_b.apply(lambda r: float(np.clip(r["Predicted"], 0, r["tankcap"])), axis=1)
        final_b = merged_b[["id", "Predicted"]].sort_values("id").reset_index(drop=True)
        mean_b = float(final_b["Predicted"].mean())
        g, a, m = best_map[base]
        out_b = SUBS / f"submission_v10_{base}_base_valmse{m:.1f}_g{g:.2f}_a{a:.3f}_mean{mean_b:.1f}.csv"
        final_b.to_csv(out_b, index=False)
        print(f"Saved base {base}: {out_b}")


if __name__ == "__main__":
    main()
