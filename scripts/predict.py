from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_loader import discover_data_files, load_id_mapping
from utils.feature_eng import build_feature_matrices
from utils.modeling import predict_with_models, run_training


def _build_submission(test_daily: pd.DataFrame, preds: np.ndarray, id_mapping: pd.DataFrame, train_daily: pd.DataFrame) -> pd.DataFrame:
    out = test_daily[["vehicle", "date_key", "total_ignition_on_hours"]].copy()
    out["date"] = pd.to_datetime(out["date_key"]).dt.date
    out["fuel_volume"] = preds

    sub = id_mapping.copy()
    sub = sub.merge(out[["vehicle", "date", "fuel_volume"]], on=["vehicle", "date"], how="left")

    veh_hist = train_daily.groupby("vehicle", observed=True)["fuel_volume"].mean().to_dict()
    global_mean = float(train_daily["fuel_volume"].mean())

    missing = sub["fuel_volume"].isna()
    if missing.any():
        sub.loc[missing, "fuel_volume"] = sub.loc[missing, "vehicle"].map(veh_hist).fillna(global_mean)

    sub["fuel_volume"] = sub["fuel_volume"].clip(lower=0)
    sub = sub[["id", "fuel_volume"]].rename(columns={"id": "ID"})
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description="HaulMark dumper fuel prediction pipeline")
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "daywise__predict.csv"))
    parser.add_argument("--cache_dir", default=str(ROOT / "data"))
    parser.add_argument("--output_dir", default=str(ROOT / "outputs"))
    parser.add_argument("--enable_spatial", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)

    print("[predict] Step 1: Build feature matrices")
    train_daily, test_daily = build_feature_matrices(data_dir, cache_dir, enable_spatial=args.enable_spatial)

    print("[predict] Step 2: Train models")
    artifacts = run_training(train_daily, test_daily, output_dir)

    print("[predict] Step 3: Predict test")
    pred = predict_with_models(artifacts, artifacts["test_prepared"])

    files = discover_data_files(data_dir)
    if files.id_mapping_file is None:
        raise FileNotFoundError("id_mapping.csv not found; cannot build submission IDs")

    id_mapping = load_id_mapping(files.id_mapping_file)

    print("[predict] Step 4: Build submission")
    sub = _build_submission(test_daily, pred, id_mapping, train_daily)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)

    print("[predict] Submission saved:", output_path)
    print(sub.head())
    print(sub.describe(include="all"))
    print("[predict] NaN count:", int(sub.isna().sum().sum()))


if __name__ == "__main__":
    main()
