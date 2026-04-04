from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

from utils.evaluation import rmse, save_feature_importance, save_oof_diagnostics
from utils.feature_eng import get_feature_columns


TARGET_COL = "fuel_volume"
AUX_TARGETS = ["prod_hr_dpr", "idle_hr_dpr", "km_dpr", "tonnage"]


def _prepare_matrix(train_daily: pd.DataFrame, test_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train = train_daily.copy()
    test = test_daily.copy()

    cats = sorted(train["vehicle"].astype(str).unique().tolist())
    cmap = {c: i for i, c in enumerate(cats)}
    train["vehicle_code"] = train["vehicle"].astype(str).map(cmap).astype(float)
    test["vehicle_code"] = test["vehicle"].astype(str).map(cmap).fillna(-1).astype(float)

    feature_cols = get_feature_columns(train)

    for c in feature_cols:
        med = float(train[c].median()) if c in train.columns else 0.0
        train[c] = train[c].fillna(med)
        test[c] = test[c].fillna(med)

    return train, test, feature_cols


def train_lgb_cv(train_daily: pd.DataFrame, feature_cols: List[str], target_col: str = TARGET_COL) -> Tuple[np.ndarray, List[lgb.LGBMRegressor], pd.DataFrame]:
    train = train_daily.dropna(subset=[target_col]).copy()
    X = train[feature_cols]
    y = train[target_col]
    groups = train["vehicle"].astype(str)

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(train), dtype=float)
    models: List[lgb.LGBMRegressor] = []
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42 + fold,
            n_jobs=-1,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
        )

        pred = model.predict(X_va)
        oof[va_idx] = pred
        fr = rmse(y_va.to_numpy(), pred)
        print(f"[modeling] Fold {fold} RMSE: {fr:.5f}")

        fold_rows.append({"fold": fold, "rmse": fr})
        models.append(model)

    total = rmse(y.to_numpy(), oof)
    print(f"[modeling] OOF RMSE: {total:.5f}")

    oof_df = train[["vehicle", "date_key", target_col]].copy()
    oof_df = oof_df.rename(columns={target_col: "target"})
    oof_df["oof_pred"] = oof

    return oof, models, pd.DataFrame(fold_rows)


def train_time_validation(train_daily: pd.DataFrame, feature_cols: List[str], target_col: str = TARGET_COL) -> pd.DataFrame:
    data = train_daily.dropna(subset=[target_col]).copy()
    data["date_key"] = pd.to_datetime(data["date_key"])

    splits = [
        (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-15"), pd.Timestamp("2026-01-16"), pd.Timestamp("2026-01-20"), "jan"),
        (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-15"), pd.Timestamp("2026-02-16"), pd.Timestamp("2026-02-20"), "feb"),
    ]

    rows = []
    for tr0, tr1, va0, va1, name in splits:
        tr = data[(data["date_key"] >= tr0) & (data["date_key"] <= tr1)]
        va = data[(data["date_key"] >= va0) & (data["date_key"] <= va1)]
        if tr.empty or va.empty:
            continue

        m = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=63,
            random_state=52,
            n_jobs=-1,
        )
        m.fit(tr[feature_cols], tr[target_col])
        pred = m.predict(va[feature_cols])
        rows.append({"split": name, "rmse": rmse(va[target_col].to_numpy(), pred)})

    out = pd.DataFrame(rows)
    if not out.empty:
        print("[modeling] Time-based validation:")
        print(out)
    return out


def train_xgb_cv(train_daily: pd.DataFrame, feature_cols: List[str], target_col: str = TARGET_COL) -> Tuple[np.ndarray, List[object]]:
    if XGBRegressor is None:
        return np.zeros(train_daily[target_col].notna().sum()), []

    train = train_daily.dropna(subset=[target_col]).copy()
    X = train[feature_cols]
    y = train[target_col]
    groups = train["vehicle"].astype(str)

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(train), dtype=float)
    models = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        m = XGBRegressor(
            n_estimators=1600,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=100 + fold,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[va_idx], y.iloc[va_idx])], verbose=False)
        oof[va_idx] = m.predict(X.iloc[va_idx])
        models.append(m)

    return oof, models


def _fit_final_lgb(train_daily: pd.DataFrame, feature_cols: List[str], target_col: str = TARGET_COL) -> lgb.LGBMRegressor:
    train = train_daily.dropna(subset=[target_col]).copy()
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[feature_cols], train[target_col])
    return model


def train_aux_models(train_daily: pd.DataFrame, feature_cols: List[str], output_dir: Path) -> None:
    rows = []
    for target in AUX_TARGETS:
        if target not in train_daily.columns:
            continue
        d = train_daily.dropna(subset=[target])
        if len(d) < 50:
            continue
        model = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, random_state=99, n_jobs=-1)
        model.fit(d[feature_cols], d[target])
        pred = model.predict(d[feature_cols])
        rows.append({"target": target, "train_rmse": rmse(d[target].to_numpy(), pred)})

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "aux_targets_train_rmse.csv", index=False)


def build_route_benchmark(train_daily: pd.DataFrame, oof_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    td = train_daily.dropna(subset=[TARGET_COL]).copy()
    td["route_dist_bin"] = pd.cut(td["mean_cycle_distance_km"], bins=[-1, 1, 2, 3, 5, 10, np.inf], labels=False)
    td["route_lift_bin"] = pd.cut(td["mean_net_lift"], bins=[-np.inf, -20, -5, 5, 20, np.inf], labels=False)
    td["route_id"] = td["mine_anon"].astype(str) + "_" + td["route_dist_bin"].astype(str) + "_" + td["route_lift_bin"].astype(str)

    bench = td.groupby("route_id", observed=True).agg(
        expected_fuel_per_day=(TARGET_COL, "mean"),
        expected_fuel_per_hour=(TARGET_COL, lambda s: float(np.nanmean(s))),
        mean_cycle_count=("cycles_count", "mean"),
        rows=("vehicle", "count"),
    ).reset_index()
    bench.to_csv(output_dir / "route_benchmark.csv", index=False)

    td = td.merge(bench[["route_id", "expected_fuel_per_day"]], on="route_id", how="left")
    td["residual_route"] = td[TARGET_COL] - td["expected_fuel_per_day"]
    eff = td.groupby("vehicle", observed=True)["residual_route"].mean().reset_index(name="dumper_efficiency")
    eff.to_csv(output_dir / "dumper_efficiency.csv", index=False)

    return bench


def run_training(train_daily: pd.DataFrame, test_daily: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_daily, test_daily, feature_cols = _prepare_matrix(train_daily, test_daily)

    lgb_oof, lgb_models, fold_df = train_lgb_cv(train_daily, feature_cols, target_col=TARGET_COL)
    fold_df.to_csv(output_dir / "fold_rmse.csv", index=False)

    time_df = train_time_validation(train_daily, feature_cols, target_col=TARGET_COL)
    if not time_df.empty:
        time_df.to_csv(output_dir / "time_validation_rmse.csv", index=False)

    xgb_oof, xgb_models = train_xgb_cv(train_daily, feature_cols, target_col=TARGET_COL)

    valid_mask = train_daily[TARGET_COL].notna().to_numpy()
    y = train_daily.loc[valid_mask, TARGET_COL].to_numpy()

    if len(xgb_models) > 0:
        w_grid = np.linspace(0.0, 1.0, 21)
        best_w, best_rmse = 1.0, 1e18
        for w in w_grid:
            p = w * lgb_oof + (1 - w) * xgb_oof
            s = rmse(y, p)
            if s < best_rmse:
                best_rmse = s
                best_w = float(w)
        ensemble_w = best_w
        final_oof = ensemble_w * lgb_oof + (1 - ensemble_w) * xgb_oof
        print(f"[modeling] Ensemble weight (lgb): {ensemble_w:.2f}, OOF RMSE={best_rmse:.5f}")
    else:
        ensemble_w = 1.0
        final_oof = lgb_oof

    oof_df = train_daily.loc[valid_mask, ["vehicle", "date_key", TARGET_COL]].copy()
    oof_df = oof_df.rename(columns={TARGET_COL: "target"})
    oof_df["oof_pred"] = final_oof

    scores = save_oof_diagnostics(oof_df, output_dir)

    model_full = _fit_final_lgb(train_daily, feature_cols, TARGET_COL)

    imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": model_full.booster_.feature_importance(importance_type="gain"),
            "split": model_full.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    save_feature_importance(imp, output_dir)

    train_aux_models(train_daily, feature_cols, output_dir)
    build_route_benchmark(train_daily.loc[valid_mask], oof_df, output_dir)

    return {
        "feature_cols": feature_cols,
        "lgb_models": lgb_models,
        "xgb_models": xgb_models,
        "ensemble_w": ensemble_w,
        "full_model": model_full,
        "test_prepared": test_daily,
        "scores": scores,
    }


def predict_with_models(model_artifacts: Dict[str, object], test_daily: pd.DataFrame) -> np.ndarray:
    feature_cols = model_artifacts["feature_cols"]
    model_full = model_artifacts["full_model"]

    pred_lgb = model_full.predict(test_daily[feature_cols])

    xgb_models = model_artifacts.get("xgb_models", [])
    if xgb_models:
        pred_xgb = np.mean([m.predict(test_daily[feature_cols]) for m in xgb_models], axis=0)
        w = float(model_artifacts.get("ensemble_w", 1.0))
        pred = w * pred_lgb + (1 - w) * pred_xgb
    else:
        pred = pred_lgb

    pred = np.clip(pred, 0, None)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model from cached feature matrices")
    parser.add_argument("--cache_dir", default="data")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    train_daily = pd.read_feather(Path(args.cache_dir) / "train_features.feather")
    test_daily = pd.read_feather(Path(args.cache_dir) / "test_features.feather")

    artifacts = run_training(train_daily, test_daily, Path(args.output_dir))
    print(artifacts["scores"])


if __name__ == "__main__":
    main()
