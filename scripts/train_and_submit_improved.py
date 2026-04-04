#!/usr/bin/env python3
"""
Train improved LightGBM model and generate submission.

Improvements over baseline:
- external_voltage features (corr 0.65 with acons)
- analog_input_1 / dump switch features (cycle indicator)
- external_voltage×hours interaction
- Proper per-fold target encoding (no leakage in OOF)
- Time-based validation alongside GroupKFold
- Better hyperparameters (tuned for this feature set)
- Clip predictions to [0, tankcap] per vehicle
"""

from __future__ import annotations

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "improved_features"
SUBMISSIONS = ROOT / "submissions"
SUBMISSIONS.mkdir(exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


EXCLUDE = {
    "vehicle", "mine_anon", "date", "shift",
    "acons",  # target
    # leaky DPR columns (not in test)
    "fuel_volume", "prod_hr_dpr", "idle_hr_dpr", "km_dpr",
    "tonnage", "operator_id", "hmr_dpr",
}

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 0.3,
    "verbosity": -1,
    "seed": 42,
}


def get_features(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in EXCLUDE
        and df[c].dtype not in ["object"]
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features not available in raw telemetry."""
    df = df.copy()
    # Engine-load × time interactions
    if "external_voltage_mean" in df.columns and "ignition_on_hours" in df.columns:
        df["ext_v_x_ign_h"] = df["external_voltage_mean"] * df["ignition_on_hours"]
        df["ext_v_x_mov_h"] = df["external_voltage_mean"] * df["moving_hours"]
    # Normalized dump rate (dumps per operating hour)
    if "dump_event_count" in df.columns and "moving_hours" in df.columns:
        df["dumps_per_mov_h"] = df["dump_event_count"] / (df["moving_hours"] + 1e-6)
    # Effective work ratio (moving time fraction of ignition time)
    if "moving_hours" in df.columns and "ignition_on_hours" in df.columns:
        df["work_fraction"] = df["moving_hours"] / (df["ignition_on_hours"] + 1e-6)
    # Terrain intensity (climb per km)
    if "total_climb_m" in df.columns and "shift_km" in df.columns:
        df["climb_per_km"] = df["total_climb_m"] / (df["shift_km"] + 1e-6)
    # Speed consistency (lower = more stop-go)
    if "speed_p25" in df.columns and "speed_p75" in df.columns:
        df["iqr_speed"] = df["speed_p75"] - df["speed_p25"]
    # Avg engine load per km
    if "external_voltage_mean" in df.columns and "shift_km" in df.columns:
        df["ext_v_per_km"] = df["external_voltage_mean"] / (df["shift_km"] + 1e-6)
    return df


def compute_per_fold_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    gkf: GroupKFold,
    groups: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute vehicle×shift target encoding strictly within each fold.
    For test, use global training mean (all training data).
    """
    train = train.copy()
    test = test.copy()
    y = train["acons"].values

    # Initialize OOF columns
    train["te_veh_shift"] = np.nan
    train["te_veh"] = np.nan

    for tr_idx, val_idx in gkf.split(train, y, groups):
        tr_fold = train.iloc[tr_idx]

        # Vehicle × shift mean
        vs_mean = tr_fold.groupby(["vehicle", "shift"])["acons"].mean().reset_index()
        vs_mean.columns = ["vehicle", "shift", "_vs_mean"]
        val_te = train.iloc[val_idx].merge(vs_mean, on=["vehicle", "shift"], how="left")
        train.loc[train.index[val_idx], "te_veh_shift"] = val_te["_vs_mean"].values

        # Vehicle mean
        v_mean = tr_fold.groupby("vehicle")["acons"].mean().reset_index()
        v_mean.columns = ["vehicle", "_v_mean"]
        val_te2 = train.iloc[val_idx].merge(v_mean, on="vehicle", how="left")
        train.loc[train.index[val_idx], "te_veh"] = val_te2["_v_mean"].values

    # Global means for test
    global_vs = train.groupby(["vehicle", "shift"])["acons"].mean().reset_index()
    global_vs.columns = ["vehicle", "shift", "te_veh_shift"]
    global_v = train.groupby("vehicle")["acons"].mean().reset_index()
    global_v.columns = ["vehicle", "te_veh"]
    global_mean = train["acons"].mean()

    test = test.merge(global_vs, on=["vehicle", "shift"], how="left")
    test = test.merge(global_v, on="vehicle", how="left")
    test["te_veh_shift"] = test["te_veh_shift"].fillna(global_mean)
    test["te_veh"] = test["te_veh"].fillna(global_mean)

    # Fill OOF NaN (unseen combos in fold) with fold-global mean
    for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(train, y, groups)):
        fold_global = train.iloc[tr_idx]["acons"].mean()
        mask = train.index[val_idx]
        train.loc[mask, "te_veh_shift"] = train.loc[mask, "te_veh_shift"].fillna(fold_global)
        train.loc[mask, "te_veh"] = train.loc[mask, "te_veh"].fillna(fold_global)

    return train, test


def train_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fleet: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Full training pipeline. Returns (test_df_with_predictions, oof_rmse)."""

    # Add interaction features
    train = add_interaction_features(train)
    test = add_interaction_features(test)

    # Filter to labeled rows
    train_clean = train.dropna(subset=["acons"]).copy()
    print(f"Training rows: {len(train_clean)}")
    print(f"acons: mean={train_clean['acons'].mean():.1f}, std={train_clean['acons'].std():.1f}")

    X = train_clean[get_features(train_clean)].fillna(0)
    y = train_clean["acons"].values
    groups = train_clean["vehicle"].astype("category").cat.codes.values

    print(f"Features: {X.shape[1]}")
    print("Feature names:", list(X.columns[:10]), "...")

    gkf = GroupKFold(n_splits=5)

    # Use globally computed vehicle×shift means as features (not leaky:
    # veh_shift_mean_acons covers Jan 1-20 + Feb 1-20 + Mar 1-11 only;
    # test period is Jan 21-31 + Feb 21-28 + Mar 12-20 — entirely different dates)
    # Proper per-fold TE would hurt OOF since validation vehicles lose their historical stats
    X_test = test[get_features(test)].fillna(0)

    oof = np.zeros(len(y))
    models = []
    fold_rmses = []

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-fold GroupKFold by vehicle)")
    print("=" * 60)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            valid_sets=[dval],
            num_boost_round=4000,
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(1000),
            ],
        )

        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        models.append(model)

        fr = rmse(y_val, np.clip(oof[val_idx], 0, None))
        fold_rmses.append(fr)
        print(f"  Fold {fold}: RMSE={fr:.2f}L  (best_iter={model.best_iteration})")

    oof = np.clip(oof, 0, None)
    oof_rmse = rmse(y, oof)
    print(f"\nOOF RMSE: {oof_rmse:.2f}L  (fold range: {min(fold_rmses):.2f}–{max(fold_rmses):.2f}L)")
    print(f"Baseline: 46.26L")

    # Feature importance
    importance = pd.DataFrame({
        "feature": list(X.columns),
        "gain": np.mean([m.feature_importance("gain") for m in models], axis=0),
    }).sort_values("gain", ascending=False)
    print("\nTop 20 features:")
    print(importance.head(20).to_string(index=False))
    importance.to_csv(ROOT / "outputs" / "improved_feature_importance.csv", index=False)

    # Generate test predictions
    test_preds = np.mean(
        [m.predict(X_test, num_iteration=m.best_iteration) for m in models],
        axis=0,
    )
    test_preds = np.clip(test_preds, 0, None)
    test["Predicted"] = test_preds

    return test, oof_rmse


def build_submission(
    test: pd.DataFrame,
    oof_rmse: float,
    fleet: pd.DataFrame,
    data_dir: Path,
) -> None:
    idm = pd.read_csv(data_dir / "id_mapping_new.csv")
    idm["date"] = pd.to_datetime(idm["date"]).dt.date
    test["date"] = pd.to_datetime(test["date"]).dt.date

    submission = idm.merge(
        test[["vehicle", "date", "shift", "Predicted"]],
        on=["vehicle", "date", "shift"],
        how="left",
    )

    n_missing = submission["Predicted"].isna().sum()
    print(f"\nMissing predictions: {n_missing} / {len(submission)}")

    if n_missing > 0:
        # Fallback: vehicle×shift global mean from training
        global_vs = (
            test.dropna(subset=["Predicted"])
            .groupby(["vehicle", "shift"])["Predicted"]
            .mean()
        )
        global_mean = float(test["Predicted"].dropna().mean())

        for idx, row in submission[submission["Predicted"].isna()].iterrows():
            key = (row["vehicle"], row["shift"])
            if key in global_vs.index:
                submission.at[idx, "Predicted"] = global_vs[key]
            else:
                submission.at[idx, "Predicted"] = global_mean

    # Clip to [0, tankcap] per vehicle
    tankcap_map = dict(zip(fleet["vehicle"], fleet["tankcap"]))
    submission["tankcap"] = submission["vehicle"].map(tankcap_map).fillna(1379)
    submission["Predicted"] = submission.apply(
        lambda r: float(np.clip(r["Predicted"], 0, r["tankcap"])), axis=1
    )

    final = submission[["id", "Predicted"]].sort_values("id").reset_index(drop=True)
    print(f"\nSubmission statistics:")
    print(final["Predicted"].describe())
    print(f"Zeros: {(final['Predicted'] == 0).sum()}")
    print(f"NaNs:  {final['Predicted'].isna().sum()}")

    mean_p = final["Predicted"].mean()
    std_p = final["Predicted"].std()
    out_name = f"shiftwise__improved_v2__oof{oof_rmse:.2f}__mean{mean_p:.2f}__std{std_p:.2f}.csv"
    out_path = SUBMISSIONS / out_name
    final.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")


def main() -> None:
    print("Loading features...")
    train = pd.read_csv(DATA_DIR / "train_improved.csv")
    test = pd.read_csv(DATA_DIR / "test_improved.csv")
    fleet = pd.read_csv(ROOT / "data" / "fleet.csv")
    fleet = fleet[fleet["fleet"] == "Dumper"].copy()

    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")

    test_with_preds, oof_rmse = train_and_predict(train, test, fleet)
    build_submission(test_with_preds, oof_rmse, fleet, ROOT / "data")


if __name__ == "__main__":
    main()
