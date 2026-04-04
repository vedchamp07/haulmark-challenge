# 🔧 RMSE Regression Fix - Summary

## What Happened

### Submission 1: 3551.50543 RMSE ✅ (CORRECT)

- **File**: `submissions/shiftwise__baseline_lgb__oof46.26__mean161.90__std92.04.csv`
- **Features**: Clean telemetry only (shift_km, moving_hours, altitude, etc.)
- **OOF RMSE**: 46.26L
- **Leaderboard**: 3551.50543 RMSE (6th place)
- **Status**: VALID - no data leakage

### Submission 2: 10k RMSE ❌ (DATA LEAKAGE)

- **File**: `submissions/archive/shiftwise__ensemble_3model__oof16.79__mean140.38__std75.70__leaky.csv`

(Archived at `submissions/archive/shiftwise__ensemble_3model__oof16.79__mean140.38__std75.70__leaky.csv`.)

- **Features**: Included `expected_cons_from_levels`, `arefill`, `initlev`, `endlev`, `runhrs`
- **OOF RMSE**: 16.79L (misleadingly good!)
- **Leaderboard**: 10k RMSE (MUCH WORSE)
- **Problem**: These features are derived from summary data that contains the target!

## The Data Leakage Explained

```python
# LEAKY FEATURE (from summary data):
expected_cons_from_levels = (initlev - endlev) + arefill

# This IS the target we're predicting!
# Summary data has:
# - initlev: tank level at shift start
# - endlev: tank level at shift end
# - arefill: refuel amount
# - acons: ACTUAL consumption (the target!)
```

**Problem**: These fields are NOT available for the test set (Mar 12-20). The model learned
to rely on them during training (hence amazing 16L OOF), but they're missing for predictions!

## Why OOF Was Good But Leaderboard Was Bad

- **Training**: Model had access to `expected_cons_from_levels` → just copied the answer
- **Cross-validation**: Still had access in validation folds → still good
- **Test set**: NO ACCESS to these fields → predictions were garbage

## ✅ Solution: Use Clean Submission

**SUBMIT**: `submissions/shiftwise__baseline_lgb__oof46.26__mean161.90__std92.04.csv` (or the identical copy `submissions/archive/shiftwise__final_clean_copy__mean161.90__std92.04.csv`)

Path: `submissions/shiftwise__baseline_lgb__oof46.26__mean161.90__std92.04.csv`

This is the CORRECT baseline using only telemetry features that are truly available
at prediction time.

## 📈 How to Improve (Without Leakage)

To beat 3551.50543 RMSE, you need to engineer better features from **telemetry data only**:

### Safe Features to Add:

1. **Better temporal patterns**: Day of week, month, time-of-day effects
2. **Vehicle personality**: Historical consumption patterns per vehicle (from training only!)
3. **Route characteristics**: Altitude variance, speed patterns, cycle detection
4. **Interactions**: vehicle × shift, altitude × speed, etc.
5. **Ensemble**: Multiple models (LightGBM + XGBoost) with different parameters

### DO NOT USE (Leakage):

- ❌ `arefill`, `initlev`, `endlev`, `runhrs` from summary files
- ❌ `expected_cons_from_levels`
- ❌ ANY lag features that use future test data
- ❌ Target encodings without proper CV splitting

## Current Status

✅ **Clean submission ready**: `submissions/shiftwise__baseline_lgb__oof46.26__mean161.90__std92.04.csv`

- Same as your 3551.50543 RMSE baseline
- No data leakage
- Safe to submit

🎯 **Next steps**: Submit this, then work on safe feature engineering to improve incrementally

---

Generated: 2026-04-03
