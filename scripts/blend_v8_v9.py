#!/usr/bin/env python3
"""Blend v8 and v9 submissions.

This is a lightweight post-processing helper:
  - Loads two submission CSVs (id, Predicted)
  - Optionally finds best weight on a local holdout if you provide oof preds (not required)
  - Writes blended submission

Default behavior: simple convex blend with a fixed weight.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBS = ROOT / "submissions"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default=str(SUBS / "submission_v8_cat_mse403.9_mean153.9.csv"))
    p.add_argument("--b", default=str(SUBS / "submission_v9_cat_spatial_mse396.0_mean153.8.csv"))
    p.add_argument("--wa", type=float, default=0.5, help="weight for A; B gets (1-wa)")
    p.add_argument("--out", default=str(SUBS / "submission_blend_v8_v9.csv"))
    args = p.parse_args()

    a = pd.read_csv(args.a)
    b = pd.read_csv(args.b)
    if set(a.columns) != {"id", "Predicted"}:
        a = a[["id", "Predicted"]]
    if set(b.columns) != {"id", "Predicted"}:
        b = b[["id", "Predicted"]]

    df = a.merge(b, on="id", how="inner", suffixes=("_a", "_b"))
    wa = float(args.wa)
    df["Predicted"] = wa * df["Predicted_a"].to_numpy(dtype=float) + (1 - wa) * df["Predicted_b"].to_numpy(dtype=float)
    out = df[["id", "Predicted"]].sort_values("id").reset_index(drop=True)

    out_path = Path(args.out)
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} (wa={wa:.3f})")


if __name__ == "__main__":
    main()
