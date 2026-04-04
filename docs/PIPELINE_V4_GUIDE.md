# Best Pipeline — v4 (940 MSE on Leaderboard)

## What This Is

Complete knowledge dump for the next engineer picking this up. v4 is the best LB score (940 MSE ≈ 30.7L RMSE). We are ~24th place; 1st place is 391 MSE (≈19.8L RMSE).

---

## Competition Setup

- **Target**: `acons` — actual fuel consumed per shift (litres), computed as `initlev - endlev + arefill`
- **Metric**: MSE (NOT RMSE). `sqrt(940) ≈ 30.7L RMSE`. Top: 391 MSE ≈ 19.8L RMSE.
- **Vehicles**: 32 Dump trucks (Dump001–Dump045, with gaps), 10 Excavators (Exc005–Exc017), ~7 Wheeled
- **Mines**: mine001 (15 dumpers), mine002 (15 dumpers), different terrains and activity patterns
- **Shifts**: C=22:00–05:59, A=06:00–13:59, B=14:00–21:59. Night shift C: timestamps before 06:00 belong to the PREVIOUS calendar date.

### Train / Test Split

| Period | Role |
|---|---|
| Jan 1–20, Feb 1–20, Mar 1–11 | Train |
| Jan 21–31, Feb 21–28, Mar 12–20 | Test |

Same vehicles appear in both train and test (critical — historical vehicle features are valid for test).

---

## Run Order

```bash
source ~/.venv/bin/activate

# Step 1: Re-extract spatial + accel features from raw parquets (~5 min)
python scripts/build_v4_patch_features.py
# → ckpts/train_v4.parquet (4654, 73), ckpts/test_v4.parquet (2610, 73)

# Step 2: Train and generate submission (~3 min)
python scripts/train_v4.py
# → submissions/submission_v4_oof27.13_mean152.8.csv
```

---

## Two-Stage Model Architecture

### Stage 1: Active/Inactive Classifier
LightGBM binary: is this an active shift? (`acons > 10L`)
- ~18% of training shifts have `acons ≈ 0` (vehicle offline/parked)
- OOF accuracy: 0.973
- Parameters: lr=0.05, num_leaves=31, early_stopping=100

### Stage 2: Regressor on Active Shifts Only
LightGBM on active shifts (acons > 10L):
- OOF RMSE reported on active shifts only
- Parameters: lr=0.02, num_leaves=63, num_boost_round=5000, early_stopping=300
- feature_fraction=0.85, bagging_fraction=0.85, lambda_l1=0.05, lambda_l2=0.3

### Prediction Blend
```python
if active_proba < 0.3:  pred = reg_pred * active_proba   # scale down
elif active_proba < 0.7: pred = reg_pred * active_proba  # scale proportionally
else:                    pred = reg_pred                  # use full regressor
```

### CV Strategy: GroupKFold by vehicle (5 folds)
**Important**: The OOF RMSE of 27.13L is inflated (~5L) because `veh_shift_mean_acons` is computed globally before CV splits. This is INTENTIONAL — see "Why Not Fix the Leakage" section.

---

## Feature Engineering (73 features in v4)

### Motion / Time
`ign_h`, `mov_h`, `idle_h`, `idle_fraction`, `work_fraction`, `shift_km`, `cumdist_km`, `km_per_hour`, `n_pings`

### Terrain
`altitude_mean/std/max/min/range`, `total_climb_m`, `total_descent_m`, `net_lift`, `climb_per_km`

### Speed
`speed_mean/std/max/p25/p50/p75`, `speed_cv`, `iqr_speed`

### External Voltage (engine load proxy) — TOP FEATURES
`ext_v_mean/std/max`, `ext_v_x_mov_h` **#1 feature by 10× margin**, `ext_v_x_ign_h`, `ext_v_per_km`

### Dump Signal (mine001 only)
`dump_sig_events`, `dump_sig_mean/max`, `dump_time_h`
Note: Only valid Feb 17+ in training, all 15 mine001 dumpers wired by Mar 1.

### Spatial Zones (v4, fixed)
`frac_dump_v4` (88.7% non-zero), `frac_haul_v4` (92.5% non-zero)
Computed from `mine_001_anonymized.gpkg` (LOCAL filename — underscore before number)

### Cycle Features
`haul_cycles` (from total_trip delta), `max_total_trip`, `min_total_trip`, `n_trips_from_total`, `km_per_cycle`, `ign_h_per_cycle`, `loading_visits`

### Accelerometer
`accel_std`, `accel_mean` (from axis_x/y/z magnitude)

### Physics Predictions
`physics_pred = veh_lph × ign_h` — vehicle's typical fuel rate × hours on
`trip_pred = veh_fuel_per_trip × haul_cycles`

### Vehicle History (leaky but valid for test — same vehicles)
`veh_shift_mean_acons`, `veh_shift_std_acons` — vehicle's historical mean per shift type
`veh_lph` — vehicle's median litres/hour (from acons/ign_h)
`veh_mean_km/ign_h/dumps/idle_frac/speed/ext_v`
`veh_fuel_per_trip`

### Operator Encoding
`op_mean_acons`, `op_count` — operator's historical consumption

### Mine / Fleet
`mine_enc` (0=mine001, 1=mine002), `has_dump_switch`, `tankcap`

### RFID Refuels
`rfid_liters`, `rfid_events` — ground-truth refueling events per shift

### Temporal
`shift_enc`, `day_of_week`, `day_of_month`, `week_number`, `month`, `is_weekend`, `hour_start`, `hour_end`

---

## Top Features (v4, by gain)

1. `ext_v_x_mov_h` — 141.8M — **#1 by 2× margin**
2. `mov_h` — 74.4M
3. `shift_km` — 63.3M
4. `ext_v_per_km` — 17.0M
5. `physics_pred` — 8.2M
6. `max_total_trip` — 7.2M
7. `total_climb_m` — 5.5M
8. `total_descent_m` — 3.5M
9. `min_total_trip` — 2.6M
10. `idle_fraction` — 2.6M
11. `ext_v_std` — 2.2M
12. `n_pings` — 2.0M
13. `cumdist_km` — 2.0M
14. `altitude_mean` — 1.9M
15. `frac_haul_v4` — 1.8M
16. `frac_dump_v4` — 1.7M
17. `veh_lph` — 1.5M (leaky but valid)
18. `climb_per_km` — 1.2M

---

## Why Not Fix the "Leakage" (IMPORTANT)

v5 attempted to fix `veh_shift_mean_acons` "leakage" by computing it fold-safely inside CV loops, and switched to time-based CV. **v5 got a WORSE LB score.**

### Why the "leakage" is actually fine:
- Train and test use THE SAME vehicles (Dump001–Dump045)
- `veh_shift_mean_acons` computed from all training data = vehicle's true historical consumption
- When predicting test, we apply this feature computed from training — this is a legitimate look-up, not leakage
- The OOF inflation (~5L) happens because validation fold rows saw their own data in the mean
- But the TEST predictions are clean: training data computes the mean, test uses it

### Why v5's fix made things worse:
- Time-based CV trains on half the data (15 days) → vehicle history computed from 15 days instead of 60 days
- This means training feature values (from 15-day history) ≠ test feature values (from 60-day history)
- The model learned patterns from a different distribution of features than what it saw at test time
- **Always compute vehicle history from ALL training data, apply to both train and test consistently**

### Takeaway:
Keep GroupKFold + global vehicle history. OOF 27.13L is inflated but model is sound.

---

## Known Data Issues

### gpkg Filename Bug (CRITICAL)
- Local: `data/mine_001_anonymized.gpkg`, `data/mine_002_anonymized.gpkg` (underscore before number)
- Kaggle: `mine001_anonymized.gpkg`, `mine002_anonymized.gpkg` (no underscore)
- v3 had all spatial features = 0 because Kaggle script used wrong local path
- Always verify this when rerunning on Kaggle

### Mine002 Has No Dump Switch
All mine002 vehicles (Dump029–Dump043) have `dump_switch=NaN`
- `analog_input_1` is only meaningful for mine001
- Dump switch signal NOT usable for mine002 cycle detection

### Analog Input Availability
- Jan training data: ~0.37% non-null (sensor not yet installed)
- Feb 17+ training: partial mine001 coverage
- Mar 1–11 training: all 15 mine001 dumpers wired (79.8% non-null)
- For test: Jan 21–31 = sparse, Feb 21–28 = partial, Mar 12–20 = full mine001 coverage

### Near-Zero Shifts (18% of training)
836 rows with `acons < 1L` — vehicle parked/offline. Handled by two-stage classifier.

### 30 No-Telemetry Test Rows
Vehicles: Dump020 (Jan 22–27), Dump040 (Feb 24), Dump041 (Feb 25–28), Dump044/045 (Mar 12)
These vehicles have 43–56% inactive rate in training → fallback to `vehicle×shift mean` is appropriate.

### fuel_volume Is Unreliable
57% of active shifts have >100L discrepancy between `acons` and `fuel_volume`-derived consumption. Not in test parquets anyway. Do not use.

### tonnage, DPR Columns Not In Test Parquets
`tonnage`, `hmr_dpr`, `prod_hr_dpr`, `idle_hr_dpr`, `km_dpr`, `maint_hr_dpr`, `bd_hr_dpr` exist in training parquets but NOT in test parquets. Correctly excluded.

### total_trip Semantics
`total_trip` is a DPR (daily production report) field — constant within most shifts, representing total haul trips completed. `max_total_trip - min_total_trip` per shift captures the shift's trip increment.

---

## Spatial Zone Detection Method

Layers in gpkg are **LineStrings** (not filled polygons). Three approaches:
1. `polygonize()` — converts closed LineString rings to proper Polygons (most accurate, use first)
2. `buffer()` — pad LineStrings outward (fallback)

Zone assignments:
- **Dump zone**: `ob_dump` + `mineral_stock` combined (both are valid dump destinations), 50m buffer
- **Haul road**: `haul_road` layer, 25m buffer
- **Loading zone**: excavator proximity (distance to nearest Exc* vehicle < 150m in UTM)
- **bench layer**: covers 43% of pings — too noisy to use as a zone

Excavator positions: 5 excavators per mine, very stable (lat/lon std ~100–200m UTM). Can compute mean position from training parquets for use in loading zone detection.

---

## Mine Differences (Important for Modeling)

| Mine | Vehicles | Mean Active acons | Active Rate | Notes |
|---|---|---|---|---|
| mine001 | Dump001–Dump028 (15) | 218.5L | 84% | Has dump switch, longer hauls |
| mine002 | Dump029–Dump043 (15) | 163.2L | 62% | No dump switch, shorter/flatter hauls |

Mine001 consumes ~34% more fuel per active shift. The `mine_enc` feature captures this but ranks only ~#72 — the other features (ext_v, km) already encode mine differences implicitly.

---

## What Actually Works (Evidence-Backed)

| Feature | Correlation with acons | Notes |
|---|---|---|
| `ext_v_x_mov_h` | 0.894 | Far and away #1 |
| `mov_h` | 0.85 | Collinear with above |
| `shift_km` | 0.80 | Strong |
| `altitude_gain_m` | 0.387 | NEW — GPS altitude diffs, not yet in v4 |
| `altitude_loss_m` | -0.387 | NEW |
| `frac_dump_v4` | -0.255 | Negative (at dump = not moving = less fuel) |
| `frac_haul_v4` | 0.140 | Weak positive |
| `n_loading_events` | — | State machine cycles, mild signal |
| `angle_change_mean` | 0.15 | Congestion/driver behavior |

---

## Features NOT Yet In v4 (Try These Next)

### High Priority: altitude_gain_m and altitude_loss_m
- Correlation 0.387 — stronger than all spatial zone features
- Computed from GPS `altitude` column (diff per ping, sum positive/negative diffs per shift)
- Both train AND test have `altitude` column
- In v5 these ranked #7 and #8 by gain
- **To add**: just add to build_v4_patch_features.py alongside existing features

### Medium Priority: n_dump_events and n_loading_events  
- From state machine: count dump zone entries, count loading zone exits
- Ranked #11 and #19 in v5
- Provide better haul cycle counts than total_trip for some vehicles

### Lower Priority: angle_change_mean
- Ranked #17 in v5 (1.57M gain)
- `mean |Δangle|` per shift — captures route complexity/congestion

### How to Add Without Breaking v4:
Add these 4 features to `build_v4_patch_features.py`, save as `ckpts/train_v4b.parquet`, and run `train_v4.py` pointing at v4b. Keep GroupKFold and global vehicle history UNCHANGED.

---

## Organizer Hints (from orientation_notes.txt)

1. **Dump switch** (`analog_input_1`): 1V=moving, 0.1V=idling, >2.5V=dumping. mine001 only. Reliable from Feb 17; all 15 mine001 dumpers wired by Mar 1.
2. **Full haul cycle**: loading → haul (loaded) → dump → empty haul → loading
3. **Fuel order**: haul loaded > empty haul > idling
4. **Loading zone**: match dumper to nearest Exc* vehicle (excavators barely move)
5. **Each dumper assigned to one excavator** (fixed)
6. **Trip count** = primary fuel driver
7. **Tonnage irrelevant** — organizer explicitly said don't use
8. **RFID = ground truth** for refueling
9. **hdop**: discard pings with hdop > 4 (GPS quality filter — not yet implemented)
10. **Congestion**: stoppage density, speed variations are driver behavior proxies

---

## What NOT to Do

1. **Do NOT use `initlev`, `endlev`, `arefill` from smry_*.csv as features** — these are test-time leakage (not available for test period)
2. **Do NOT use `fuel_volume` from telemetry** — unreliable sensor, not in test parquets
3. **Do NOT use `tonnage`, `hmr_dpr`, `prod_hr_dpr`** — not in test parquets
4. **Do NOT change CV to time-based** — v5 proved this makes things worse (see above)
5. **Do NOT use `bench` layer from gpkg** — covers 43% of pings, no discriminatory power
6. **Do NOT submit the 16.79L OOF ensemble** — it uses `initlev/endlev/arefill` which are leaky

---

## File Reference

| File | Purpose |
|---|---|
| `scripts/build_v4_patch_features.py` | Extract spatial + accel + fleet features from raw parquets |
| `scripts/train_v4.py` | **Current best: two-stage LGB, GroupKFold, global vehicle history** |
| `ckpts/train_v4.parquet` | v4 train features (4654×73) — start here |
| `ckpts/test_v4.parquet` | v4 test features (2610×73) |
| `outputs/v4_feature_importance.csv` | Feature importances from v4 training |
| `data/mine_001_anonymized.gpkg` | Spatial layers mine001 (local path — underscore!) |
| `data/mine_002_anonymized.gpkg` | Spatial layers mine002 |
| `data/fleet.csv` | 75 vehicles: type, tankcap, mine_anon, dump_switch |
| `data/id_mapping_new.csv` | 1,735 test rows to predict |
| `data/smry_*.csv` | Training labels (acons) — DO NOT use other columns as features |

---

## Submission History

| v | OOF RMSE | LB MSE | Key Change |
|---|---|---|---|
| v1 | 46.26L | 3551 | Baseline telemetry features |
| v2 | 50.73L | 3183 | Added ext_voltage interactions |
| v3 | 27.47L | 957 | Two-stage model, spatial zones (but all 0 due to gpkg bug) |
| v4 | 27.13L | **941** | Fixed gpkg path → real spatial features; accel, cumdist, mine_enc |
| v5 | 22.24L | WORSE | Altitude+state machine features (good) but time-based CV (bad) |
| v6 | 27.18L | pending | v4 pipeline + v5 features (altitude, state machine, angle) |

v6 = correct approach: GroupKFold + global vehicle history (v4) + new feature signals (v5).
**Next step to try: XGBoost/CatBoost ensemble on top of v6 features.**
