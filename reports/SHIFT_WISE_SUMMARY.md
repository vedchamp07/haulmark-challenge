# SHIFT-WISE FUEL CONSUMPTION PREDICTION - SUMMARY

## Key Changes from Previous Approach

### 1. **Evaluation Level**

- **OLD**: Daily predictions (1 per vehicle per day)
- **NEW**: Shift-wise predictions (3 per vehicle per day: C, A, B)

### 2. **Shifts Definition**

- **Shift C** (Night): 22:00 - 05:59 (8 hours)
- **Shift A** (Morning): 06:00 - 13:59 (8 hours)
- **Shift B** (Afternoon): 14:00 - 21:59 (8 hours)

### 3. **Target Variable**

- **Formula**: `acons = initlev - endlev + arefill`
- `initlev`: Initial fuel level at shift start
- `endlev`: Final fuel level at shift end
- `arefill`: Refuel amount during shift
- **acons**: Actual consumption per shift (TARGET)

### 4. **Target Statistics**

- Mean: 173.4L per shift
- Std: 97.1L
- Range: 0-602.6L
- Much smaller than daily (mean was ~444L daily)

### 5. **Data Sources**

- **Training**: Jan 1-20, Feb 1-20, March 1-11 (all with DPR columns)
- **Test**: Jan 21-31, Feb 21-28, March 12-20 (no DPR columns)
- **Ground Truth**: Summary files (smry_jan/feb/mar_train_ordered.csv)
- **Predictions needed**: 1,735 (vehicle × date × shift combinations)

### 6. **Model Performance**

- **OOF RMSE**: 46.26L
- **Naive baseline RMSE**: 97.07L
- **Improvement**: 52.3%

### 7. **Top Features**

1. shift_enc (shift A/B/C encoding)
2. altitude_std (terrain variability)
3. altitude_mean (terrain level)
4. n_pings (data coverage)
5. veh_shift_mean (vehicle-shift historical average)
6. idle_fraction
7. total_climb_m
8. veh_shift_std
9. day_of_month
10. km_per_hour

### 8. **Files Created**

- `submissions/shiftwise__baseline_lgb__oof46.26__mean161.90__std92.04.csv` - Main shift-wise baseline (LightGBM)
- `submissions/archive/shiftwise__naive_constant__mean173.40.csv` - Naive baseline (constant = 173.4L)

## Submission Format

```csv
id,Predicted
0,127.55
1,147.53
2,77.93
...
```

## Next Steps to Improve RMSE

1. **Add refuel event detection**
   - Parse refuel transactions more carefully
   - Flag shifts with refueling events

2. **Improve shift boundary handling**
   - Currently using simple hour-based rules
   - Could use actual shift_dpr from DPR when available

3. **Add cycle-level features**
   - Detect haul cycles using dump switch signal (analog_input_1)
   - Compute fuel per cycle instead of per shift

4. **Operator features**
   - operator_id is available in training
   - Could capture operator-specific behavior

5. **Weather features**
   - rain_loss and dense_fog columns exist
   - Could impact fuel consumption

6. **Spatial features**
   - Use mine geometry (haul roads, benches, dump zones)
   - Compute route-specific difficulty

7. **Ensemble methods**
   - Blend LightGBM + CatBoost + XGBoost
   - Stack with simpler models

8. **Feature engineering**
   - Rolling statistics across shifts
   - Lag features (previous shift consumption)
   - Interaction features (shift × vehicle type)
