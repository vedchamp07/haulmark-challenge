#!/usr/bin/env python3
"""Final submission blender - combine all best models"""
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = ROOT / "submissions"


def _find_first(patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(SUBMISSIONS_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None

def load_submission(fpath: Path):
    """Load and validate submission"""
    df = pd.read_csv(fpath)
    assert 'id' in df.columns and 'Predicted' in df.columns
    assert len(df) == 1735
    return df.sort_values('id').reset_index(drop=True)

def main():
    print("="*60)
    print("FINAL SUBMISSION BLENDER")
    print("="*60)

    # Load all available submissions
    submissions = {}

    # Ensemble submission (often leaky)
    fpath = _find_first([
        "shiftwise__ensemble_3model__*__leaky.csv",
        "shiftwise__ensemble_3model__*.csv",
        "submission_ensemble.csv",
    ])
    if fpath is not None:
        submissions['ensemble'] = load_submission(fpath)
        print(f"✓ Loaded: {fpath}")
    else:
        print("✗ Not found: ensemble")

    # Stacked submission (meta-learner; may not exist)
    fpath = _find_first([
        "shiftwise__stacked__*__leaky.csv",
        "shiftwise__stacked__*.csv",
        "submission_stacked.csv",
    ])
    if fpath is not None:
        submissions['stacked'] = load_submission(fpath)
        print(f"✓ Loaded: {fpath}")
    else:
        print("✗ Not found: stacked")

    # Baseline shift-wise submission (leakage-safe)
    fpath = _find_first([
        "shiftwise__baseline_lgb__*.csv",
        "shiftwise__final_clean_copy__*.csv",
        "submission_shift_wise.csv",
        "submission_final_clean.csv",
    ])
    if fpath is not None:
        submissions['baseline'] = load_submission(fpath)
        print(f"✓ Loaded: {fpath}")
    else:
        print("✗ Not found: baseline")

    if len(submissions) == 0:
        print("\nNo submissions found!")
        return

    print(f"\nTotal submissions loaded: {len(submissions)}")

    # Strategy 1: Weighted average (favor ensemble/stacked)
    if 'ensemble' in submissions:
        weights = {'ensemble': 0.5, 'stacked': 0.3, 'baseline': 0.2}
    else:
        weights = {'baseline': 1.0}

    # Only use available submissions
    weights = {k: v for k, v in weights.items() if k in submissions}
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}

    print("\nBlending weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.2%}")

    # Blend predictions
    blended = submissions[list(submissions.keys())[0]][['id']].copy()
    blended['Predicted'] = 0.0

    for name, weight in weights.items():
        blended['Predicted'] += submissions[name]['Predicted'] * weight

    blended['Predicted'] = blended['Predicted'].clip(lower=0)

    # Save
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    mean = float(blended['Predicted'].mean())
    std = float(blended['Predicted'].std())
    if set(weights.keys()) == {'ensemble', 'stacked', 'baseline'}:
        blend_tag = "ens0.5_stack0.3_base0.2"
    elif set(weights.keys()) == {'ensemble', 'baseline'}:
        blend_tag = "ens_plus_base"
    elif set(weights.keys()) == {'baseline'}:
        blend_tag = "baseline_only"
    else:
        blend_tag = "custom"
    blended_path = SUBMISSIONS_DIR / f"shiftwise__blend_{blend_tag}__mean{mean:.2f}__std{std:.2f}.csv"
    blended.to_csv(blended_path, index=False)

    print("\n" + "="*60)
    print("FINAL BLENDED STATISTICS")
    print("="*60)
    print(f"Submission rows: {len(blended)}")
    print(f"\nPrediction stats:")
    print(blended['Predicted'].describe())

    print("\n" + "="*60)
    print("INDIVIDUAL SUBMISSION STATS")
    print("="*60)
    for name, sub in submissions.items():
        print(f"\n{name}:")
        print(f"  Mean: {sub['Predicted'].mean():.2f}L")
        print(f"  Std:  {sub['Predicted'].std():.2f}L")
        print(f"  Min:  {sub['Predicted'].min():.2f}L")
        print(f"  Max:  {sub['Predicted'].max():.2f}L")

    print("\n" + "="*60)
    print(f"✓ Saved: {blended_path}")
    print("="*60)

    # Also create best individual submission copy for convenience
    if 'stacked' in submissions:
        best = submissions['stacked']
        best_model = 'stacked'
    elif 'ensemble' in submissions:
        best = submissions['ensemble']
        best_model = 'ensemble'
    else:
        best = submissions['baseline']
        best_model = 'baseline'

    best_mean = float(best['Predicted'].mean())
    best_std = float(best['Predicted'].std())
    maybe_leaky = "__leaky" if best_model in {"ensemble", "stacked"} else ""
    best_path = SUBMISSIONS_DIR / f"shiftwise__best_{best_model}_copy__mean{best_mean:.2f}__std{best_std:.2f}{maybe_leaky}.csv"
    best.to_csv(best_path, index=False)
    print(f"\n✓ Best individual saved as: {best_path}")

if __name__ == '__main__':
    main()
