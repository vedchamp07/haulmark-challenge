from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_feature_importance(importance_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "feature_importance.csv"
    importance_df.to_csv(out_csv, index=False)

    top = importance_df.sort_values("gain", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(top["feature"], top["gain"])
    ax.invert_yaxis()
    ax.set_title("Top 30 Feature Importances (gain)")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance_top30.png", dpi=150)
    plt.close(fig)


def save_oof_diagnostics(oof_df: pd.DataFrame, output_dir: Path) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    score = rmse(oof_df["target"].to_numpy(), oof_df["oof_pred"].to_numpy())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(oof_df["target"], oof_df["oof_pred"], s=8, alpha=0.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"OOF Actual vs Predicted (RMSE={score:.4f})")
    fig.tight_layout()
    fig.savefig(output_dir / "oof_actual_vs_pred.png", dpi=150)
    plt.close(fig)

    return {"oof_rmse": score}
