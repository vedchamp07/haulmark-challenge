#!/usr/bin/env python3
"""EDA: plot acons distribution and percent-zero shifts.
Saves figure to outputs/plots/acons_zero.png and prints percent zeros.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "ckpts" / "train_v5.parquet"
OUT = ROOT / "outputs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_parquet(CKPT)
    if "acons" not in df.columns:
        # Fall back to smry files in data/ if ckpt doesn't contain acons
        smry_files = sorted((Path(ROOT) / "data").glob("smry_*_train_ordered.csv"))
        if not smry_files:
            raise RuntimeError("acons column not found in ckpt and no smry files found")
        df = pd.concat([pd.read_csv(p) for p in smry_files], ignore_index=True)
        if "acons" not in df.columns:
            raise RuntimeError("acons column not found in smry files either")

    total = len(df)
    zeros = (df["acons"] == 0).sum()
    pct = 100.0 * zeros / total

    print(f"Total shifts: {total}")
    print(f"Shifts with acons==0: {zeros} ({pct:.2f}%)")

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    sns.histplot(df["acons"].clip(upper=200), bins=60, ax=axes[0])
    axes[0].set_title("acons (clipped at 200) - histogram")
    axes[0].set_xlabel("acons")

    axes[1].bar([0,1],[zeros, total-zeros], color=["#d62728","#1f77b4"])
    axes[1].set_xticks([0,1])
    axes[1].set_xticklabels(["acons==0","acons>0"])
    axes[1].set_ylabel("Count of shifts")
    axes[1].set_title(f"Zero vs Non-zero shifts ({pct:.2f}% zeros)")

    plt.tight_layout()
    out_path = OUT / "acons_zero.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

if __name__ == '__main__':
    main()
