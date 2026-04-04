# CLAUDE.md — HaulMark Dumper Fuel Consumption Hackathon

## Competition Summary

**Task**: Predict per-shift fuel consumption (`acons`, litres) for 32 dumper vehicles from high-frequency telemetry.

**Metric**: RMSE (litres)

**Organizer**: Mindshift Analytics × AI & SW Guild, IIT Madras

**Submission format**: `id, Predicted` — 1,735 rows from `data/id_mapping_new.csv`

---

## Current Best Result

| Strategy | OOF RMSE (active) | LB MSE | File |
|---|---|---|---|
| **v6: v4 pipeline + v5 features** | **27.18L** | pending | `submissions/submission_v6_oof27.18_mean152.9.csv` |
| v5: State machine + altitude + broken CV | 22.24L (time-CV, inconsistent) | WORSE | `submissions/submission_v5_oof22.24_mean152.3.csv` |
| v4: Two-stage + fixed spatial + accel | 27.13L (GroupKFold) | **940.99 ← BEST LB** | `submissions/submission_v4_oof27.13_mean152.8.csv` |
| v3: Two-stage + spatial (broken) + operator | 27.47L (leaky) | **940.99 / 957.06** | `submissions/submission_v3_oof27.47_mean152.9.csv` |
| v2: LGB + ext_voltage + interactions | 50.73L | 3182.71 | archive |
| v1: LGB baseline (telemetry-only) | 46.26L | 3551.51 | archive |
| Leaky ensemble (DO NOT SUBMIT) | 16.79L | ~10,000 | archive |

**LB metric is MSE (not RMSE)**. Top score on leaderboard: **391 MSE** (≈19.8L RMSE). 10th place: 594 MSE.

**v5 got WORSE LB score** despite better OOF. Root cause: time-based CV makes training use 15-day vehicle history but test uses 60-day history → train/test feature distributions differ. GroupKFold is correct; the "leakage" in veh_shift_mean_acons is actually valid because same vehicles appear in train and test.

**v5 features ARE good** (altitude_gain_m ranked #7, n_dump_events #11 in v5). The CV change is what broke it.
**Recommended next step: add altitude_gain_m + altitude_loss_m to v4, keep everything else identical.**

**Key lessons**:
- OOF RMSE (active only) is the right CV metric — the two-stage model separates inactive shifts (acons≈0) from active ones, so OOF on active shifts only is meaningful
- **GroupKFold by vehicle is correct** — `veh_shift_mean_acons` is a legitimate look-up (same vehicles in test), not real leakage
- `altitude_gain_m` (corr 0.387) is the strongest untapped signal — stronger than all spatial zone features
- `external_voltage × moving_hours` is consistently the #1 feature
- See `docs/PIPELINE_V4_GUIDE.md` for comprehensive pipeline documentation
- Leaky features (`initlev`, `endlev`, `arefill`) cause catastrophic LB failure

---

## Data Layout

```
data/
  fleet.csv                         # 75 vehicles: type, tankcap (741-1379L), mine_anon, dump_switch
  id_mapping_new.csv                # 1735 test rows: id, vehicle, date, shift
  smry_jan/feb/mar_train_ordered.csv  # Train labels: acons + tank levels (TRAIN ONLY)
  rfid_refuels_*.parquet            # Refuel events (available for both train AND test)
  telemetry_2026-01-01_*.parquet    # Train telemetry (Jan 1-20, Feb 1-20, Mar 1-11)
  telemetry_2026-01-21_*.parquet    # TEST telemetry (Jan 21-31, Feb 21-28, Mar 12-20)
  mine_001_anonymized.gpkg          # Spatial layers for mine001 (UTM EPSG:32645)
  mine_002_anonymized.gpkg          # Spatial layers for mine002
```

**Key rule**: `smry_*` fields (`initlev`, `endlev`, `arefill`, `runhrs`) are NOT available for the test period. Using them as features causes data leakage.

---

## Target Variable

**`acons`** (actual fuel consumption per shift, litres)
- Formula: `acons = initlev - endlev + arefill`
- Training mean: **141.6L**, std: **109.9L**, range: 0–602.6L
- Computed from 3 shifts per day: C (22:00–05:59), A (06:00–13:59), B (14:00–21:59)

---

## Time Split

| Period | Role | Telemetry File(s) |
|---|---|---|
| Jan 1–20 | Train | `telemetry_2026-01-01_*.parquet` × 2 |
| **Jan 21–31** | **TEST** | `telemetry_2026-01-21_2026-01-31.parquet` |
| Feb 1–20 | Train | `telemetry_2026-02-01_*.parquet` × 2 |
| **Feb 21–28** | **TEST** | `telemetry_2026-02-21_2026-02-28.parquet` |
| Mar 1–11 | Train | `telemetry_2026-03-01_2026-03-11.parquet` |
| **Mar 12–20** | **TEST** | `telemetry_2026-03-12_2026-03-20.parquet` |

**Operational day**: starts at 22:00 of previous calendar day, ends 21:59. Shift C night records before 06:00 are attributed to the previous calendar date.

---

## Shift Definitions

| Shift | Hours (IST) | Encoding |
|---|---|---|
| C (Night) | 22:00–05:59 | 0 |
| A (Morning) | 06:00–13:59 | 1 |
| B (Afternoon) | 14:00–21:59 | 2 |

---

## Current Pipeline (v4)

Two-stage approach:
1. **Stage 1 classifier**: LightGBM binary — is this an active shift (acons > 10L)?
2. **Stage 2 regressor**: LightGBM on active shifts only (acons > 10L)
3. **Prediction blend**: regressor output scaled by classifier probability for uncertain shifts

### Feature Set (82 features in v4)

**Motion/time:** `ign_h`, `mov_h`, `idle_h`, `idle_fraction`, `work_fraction`, `shift_km`, `cumdist_km`, `km_per_hour`, `n_pings`

**Terrain:** `altitude_mean/std/max/min/range`, `total_climb_m`, `total_descent_m`, `net_lift`, `climb_per_km`

**Speed:** `speed_mean/std/max/p25/p50/p75`, `speed_cv`, `iqr_speed`

**External voltage (engine load proxy):** `ext_v_mean/std/max`, `ext_v_x_mov_h` **(#1 feature)**, `ext_v_x_ign_h`, `ext_v_per_km`

**Dump signal** (`analog_input_1`, mine001 only): `dump_sig_events`, `dump_sig_mean/max`, `dump_time_h`

**Spatial zones (v4, fixed):** `frac_dump_v4` (ob_dump + mineral_stock, 50m buffer), `frac_haul_v4` (haul_road, 25m buffer), `dump_zone_h`, `haul_road_h`

**Cycle features:** `haul_cycles`, `max_total_trip`, `min_total_trip`, `n_trips_from_total`, `km_per_cycle`, `ign_h_per_cycle`, `loading_visits`

**Accelerometer:** `accel_std`, `accel_mean` (axis_x/y/z magnitude)

**Physics predictions:** `physics_pred = veh_lph × ign_h`, `trip_pred = veh_fuel_per_trip × haul_cycles`

**Vehicle history:** `veh_shift_mean_acons`, `veh_shift_std_acons`, `veh_lph`, `veh_mean_km`, `veh_mean_ext_v`, `veh_fuel_per_trip`

**Operator:** `op_mean_acons`, `op_count`

**Mine/fleet:** `mine_enc` (0=mine001, 1=mine002), `has_dump_switch`, `tankcap`

**RFID refuels:** `rfid_liters`, `rfid_events`

**Temporal:** `shift_enc`, `day_of_week`, `day_of_month`, `week_number`, `month`, `is_weekend`, `hour_start`, `hour_end`

### Model Parameters

```python
# Stage 2 regressor
LightGBM: objective='regression', metric='rmse'
learning_rate=0.02, num_leaves=63, min_data_in_leaf=15
feature_fraction=0.85, bagging_fraction=0.85, lambda_l1=0.05, lambda_l2=0.3
num_boost_round=5000, early_stopping=300
```

### CV Strategy

5-fold `GroupKFold` by `vehicle`. OOF RMSE reported on **active shifts only** (acons > 10L) — this is the right metric for the two-stage setup.

---

## Top Features (v4, by gain)

1. `ext_v_x_mov_h` — engine load × moving hours (physics energy proxy)
2. `mov_h` — moving hours
3. `shift_km` — distance driven
4. `ext_v_per_km` — engine load per km (terrain/load difficulty)
5. `physics_pred` — veh_lph × ign_h
6. `max_total_trip` — trip counter from DPR
7. `total_climb_m` — elevation gain
8. `total_descent_m` — elevation loss
9. `min_total_trip` — trip counter at shift start
10. `idle_fraction` — idling proportion
11. `ext_v_std` — engine load variability
12. `n_pings` — telemetry density
13. `cumdist_km` — cumulative distance (more reliable than disthav sum)
14. `frac_haul_v4` — fraction of time on haul road ← **new in v4**
15. `frac_dump_v4` — fraction of time in dump zone ← **new in v4**

---

## Critical Implementation Notes

### gpkg Filenames (IMPORTANT)
- Local files: `data/mine_001_anonymized.gpkg`, `data/mine_002_anonymized.gpkg` (underscore before number)
- Kaggle files: `mine001_anonymized.gpkg`, `mine002_anonymized.gpkg` (no underscore)
- **v3 bug**: Kaggle script searched for wrong filename → all spatial features were 0. Fixed in v4 by using correct local path.
- **For Kaggle**: update path in `kaggle_v3_submission.py` to use `mine001_anonymized.gpkg`

### Mine002 has NO dump switch
- All mine002 vehicles (Dump029–Dump043) have `dump_switch=NaN` in `fleet.csv`
- `analog_input_1` is only reliable for mine001 vehicles
- Use `has_dump_switch` feature to let model differentiate
- For mine002, cycle detection relies on `total_trip` + spatial zones only

### total_trip semantics
- `total_trip` is a DPR (daily production report) field — the value is constant within most shifts, stamped once per shift from the DPR system
- `max_total_trip - min_total_trip` per shift captures any intra-shift increments (mostly shift C)
- The max value itself = total trips completed in that shift

### Near-zero acons (18% of training data)
- 836 rows have acons ≈ 0 (runhrs ≈ 0) — vehicle was parked/offline that shift
- The two-stage model handles this: classifier predicts inactive → scale prediction toward 0
- 30 test rows have zero telemetry (Dump020, Dump040, Dump041, Dump044, Dump045 on specific dates)
- These vehicles have 43-56% inactive shift rates in training → fallback to vehicle×shift mean is appropriate

### fuel_volume is unreliable (train only, DO NOT USE)
- 57% of active shifts have >100L discrepancy between acons and fuel_volume-derived consumption
- Confirmed noisy sensor. `fuel_volume` column not in test telemetry anyway.

### Spatial zone detection method
- Layers are **LineString** boundaries (not filled polygons)
- `polygonize()` recovers closed rings into proper polygons — use first, then buffer as fallback
- `bench` layer covers nearly the entire mine (43% of pings) — too noisy for zone detection, skip it
- Use `ob_dump` + `mineral_stock` (combined) for dump zone detection (both are valid dump destinations)
- Buffer 50m for dump/stock zones, 25m for haul road

---

## Organizer Key Insights

From `docs/orientation_notes.txt`:

1. **Dump switch** (`analog_input_1`): 1V=moving, 0.1V=idling, **>2.5V=dumping**. Mine001 only. Reliable from Feb 17; all 15 mine001 dumpers wired by Mar 1.
2. **Full haul cycle**: loading → haul (loaded) → dump → empty haul → loading
3. **Fuel order**: haul loaded > empty haul > idling
4. **Loading zone**: match dumper position to nearest Exc* vehicle (excavator barely moves)
5. **Each dumper assigned to one excavator** (fixed)
6. **Dump detection**: distance=0 at dump location + spatial zone (ob_dump / stock)
7. **Trip count** = primary fuel driver
8. **Congestion/stoppage density** = driver behavior proxy
9. **RFID = ground truth** for refueling; fuel_volume sensor unreliable
10. **Tonnage irrelevant** — don't use as feature

---

## Running v6 Locally (CURRENT BEST)

```bash
source ~/.venv/bin/activate

# Step 1: Extract v5 patch features from raw parquets (~8-12 min)
# (Already done if ckpts/train_v5.parquet exists)
python scripts/build_v5_features.py
# Output: ckpts/train_v5.parquet (83 cols), ckpts/test_v5.parquet

# Step 2: Train v6 model (v4 pipeline + v5 features) (~3 min)
python scripts/train_v6.py
# Output: submissions/submission_v6_oof27.18_mean152.9.csv
```

### Checkpoints
- `ckpts/train_feats_checkpoint.parquet` — v3 features (68 cols, spatial broken)
- `ckpts/train_v4.parquet` — v4 features (73 cols, spatial fixed, broken stubs)
- `ckpts/train_v5.parquet` — v5 features (83 cols, + state machine + altitude) ← START HERE
- `ckpts/test_v5.parquet`  — v5 test features (83 cols)

**v5 builds on v4**: `build_v5_features.py` loads v4 checkpoints and adds new features on top.

---

## For Kaggle Submission

Use `scripts/kaggle_v3_submission.py` as the base. Key fix needed for v5:
- Change gpkg path from `mine001_anonymized.gpkg` to `mine_001_anonymized.gpkg` if running locally
- On Kaggle the filename is `mine001_anonymized.gpkg` — verify before running

---

## File Reference

| File | Purpose |
|---|---|
| `scripts/build_v5_features.py` | **Re-extracts all features (v4 + altitude + state machine)** |
| `scripts/train_v6.py` | **Current best: v4 pipeline on v5 checkpoints** |
| `scripts/build_v4_patch_features.py` | v4 spatial+accel feature extraction (superseded by v5) |
| `scripts/train_v4.py` | v4 training script (superseded by v6) |
| `scripts/kaggle_v3_submission.py` | Kaggle-ready end-to-end script (spatial bug fixed in v4 branch) |
| `scripts/analyze_for_improvement.py` | EDA script — acons distribution, total_trip, lph, missing rows |
| `scripts/eda_for_final_run.py` | EDA script — spatial geometry audit, gpkg bbox check |
| `ckpts/train_v4.parquet` | v4 train features (start here, don't reprocess parquets) |
| `ckpts/test_v4.parquet` | v4 test features |
| `outputs/v4_feature_importance.csv` | Feature importances from v4 training |
| `docs/orientation_notes.txt` | Organizer hints — read before changing strategy |
| `docs/about_dataset.txt` | Full column reference |

---

## Environment

```bash
source ~/.venv/bin/activate  # project venv
```

Key packages: `lightgbm`, `geopandas`, `pyproj`, `shapely`, `pyarrow`, `pandas`, `numpy`, `sklearn`
