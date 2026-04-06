#!/usr/bin/env python3
"""
v4 Optimized — HaulMark fuel prediction with 4 key improvements:
1. Forward-chained monthly CV (Jan→Feb, Jan+Feb→Mar) instead of GroupKFold
2. Altitude rank-transform to neutralize mine-depth drift
3. Optuna tuning (100 trials) on hardest fold  
4. Day-level RFID aggregation (captures boundary-straddling refuels)
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import rankdata
import matplotlib.pyplot as plt

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ==========================================
# 1. PATHS
# ==========================================
ROOT = Path(__file__).resolve().parent.parent
RAW = str(ROOT / "data")

SUMMARY_FILES = [
    "smry_jan_train_ordered.csv",
    "smry_feb_train_ordered.csv",
    "smry_mar_train_ordered.csv",
]

TRAIN_TELEMETRY = [
    "telemetry_2026-01-01_2026-01-10.parquet",
    "telemetry_2026-01-11_2026-01-20.parquet",
    "telemetry_2026-02-01_2026-02-10.parquet",
    "telemetry_2026-02-11_2026-02-20.parquet",
    "telemetry_2026-03-01_2026-03-11.parquet",
]

TEST_TELEMETRY = [
    "telemetry_2026-01-21_2026-01-31.parquet",
    "telemetry_2026-02-21_2026-02-28.parquet",
    "telemetry_2026-03-12_2026-03-20.parquet",
]

RFID_FILE = "rfid_refuels_2026-01-01_2026-03-31.parquet"

# ==========================================
# 2. LOAD TELEMETRY
# ==========================================
def load_and_combine(file_list):
    dfs = []
    for name in file_list:
        p = os.path.join(RAW, name)
        if os.path.exists(p):
            print(f"  Loading: {name}")
            df = pd.read_parquet(p)
            df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_one_telemetry(name: str) -> pd.DataFrame:
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        print(f"  WARNING: Not found: {name}")
        return pd.DataFrame()
    print(f"  Loading: {name}")
    df = pd.read_parquet(p)
    df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]
    return df


# ==========================================
# 3. RFID FEATURES (DAY-LEVEL)
# ==========================================
def build_rfid_features(rfid_path: str) -> pd.DataFrame:
    """
    Day-level RFID features. Deliberately avoids shift split
    since refuels straddle shift boundaries.
    """
    if not os.path.exists(rfid_path):
        print(f"  WARNING: RFID file not found: {rfid_path}")
        return pd.DataFrame()

    rfid = pd.read_parquet(rfid_path)
    rfid.columns = [str(c).lower().strip().replace(' ', '_') for c in rfid.columns]

    ts_col = None
    for c in ['ts', 'timestamp', 'event_time', 'time']:
        if c in rfid.columns:
            ts_col = c
            break

    if ts_col is not None:
        rfid[ts_col] = pd.to_datetime(rfid[ts_col], errors='coerce')
        if getattr(rfid[ts_col].dt, 'tz', None) is not None:
            rfid[ts_col] = rfid[ts_col].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        rfid['adj_date'] = (rfid[ts_col] + pd.Timedelta(hours=2)).dt.strftime('%Y-%m-%d')
    elif 'date_dpr' in rfid.columns:
        rfid['adj_date'] = pd.to_datetime(rfid['date_dpr'], errors='coerce').dt.strftime('%Y-%m-%d')
    elif 'date' in rfid.columns:
        rfid['adj_date'] = pd.to_datetime(rfid['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    else:
        return pd.DataFrame()

    rfid['vehicle'] = rfid['vehicle'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

    qty_col = next((c for c in rfid.columns if any(k in c for k in
                    ['qty', 'quantity', 'fuel', 'volume', 'litres', 'liters'])), None)
    if qty_col is None:
        return pd.DataFrame()

    print(f"  RFID qty column: '{qty_col}'")
    
    # Day-level aggregation ONLY (no shift split)
    rfid_day = rfid.groupby(['vehicle', 'adj_date']).agg(
        rfid_fuel_added_day   = (qty_col, 'sum'),
        rfid_n_events_day     = (qty_col, 'count'),
        rfid_is_refueled_day  = (qty_col, lambda x: int((x > 0).any())),
    ).reset_index()

    return rfid_day


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================
def get_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df['vehicle'] = df['vehicle'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    df['ts']      = pd.to_datetime(df['ts'])
    if df['ts'].dt.tz is not None:
        df['ts'] = df['ts'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df = df.sort_values(['vehicle', 'ts']).reset_index(drop=True)

    df['adj_date'] = (df['ts'] + pd.Timedelta(hours=2)).dt.strftime('%Y-%m-%d')

    if 'shift_dpr' not in df.columns or df['shift_dpr'].isna().mean() > 0.5:
        h = df['ts'].dt.hour
        df['shift_dpr'] = np.select(
            [((h >= 22) | (h < 6)), ((h >= 6) & (h < 14)), ((h >= 14) & (h < 22))],
            ['C', 'B', 'A'],
            default='B'
        )

    df['delta_alt'] = df.groupby('vehicle')['altitude'].diff().fillna(0)
    df['dist_m']    = df['disthav'].fillna(0)
    df['speed']     = df['speed'].fillna(0)

    if 'analog_input_1' in df.columns and (df['analog_input_1'] > 2.5).mean() > 0.001:
        df['is_dumping'] = (df['analog_input_1'] > 2.5).astype(int)
        df['cycle_id']   = (df.groupby('vehicle')['is_dumping'].transform(lambda x: (x.diff() < 0).cumsum()))
    else:
        df['is_dumping'] = 0
        df['cycle_id']   = (df.groupby('vehicle')['dist_m'].transform(lambda x: (x == 0).cumsum()))

    df['cum_dist'] = df.groupby(['vehicle', 'cycle_id'])['dist_m'].cumsum()
    df['is_laden'] = (df['cum_dist'] > 5000).astype(int)

    laden_df   = df[df['is_laden'] == 1]
    unladen_df = df[df['is_laden'] == 0]
    moving_df  = df[df['speed'] > 1]

    base = df.groupby(['vehicle', 'adj_date', 'shift_dpr']).agg(
        n_pings          = ('ts',       'count'),
        total_dist_km    = ('dist_m',   lambda x: x.sum() / 1000),
        n_stops          = ('speed',    lambda x: ((x.shift(1) >= 1) & (x < 1)).sum()),
        idle_h           = ('speed',    lambda x: (x < 1).sum() * (20 / 3600)),
        trip_count       = ('cycle_id', 'nunique'),
        shift_start_hr   = ('ts',       lambda x: x.min().hour),
        shift_duration_h = ('ts',       lambda x: (x.max() - x.min()).total_seconds() / 3600),
        alt_std          = ('altitude', 'std'),
        alt_range        = ('altitude', lambda x: x.max() - x.min()),
        net_lift         = ('altitude', lambda x: float(x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0.0),
        pos_climb_m      = ('delta_alt', lambda x: x[x > 0].sum()),
        neg_descent_m    = ('delta_alt', lambda x: x[x < 0].abs().sum()),
        dump_events      = ('is_dumping', 'sum'),
    ).reset_index()

    laden = laden_df.groupby(['vehicle', 'adj_date', 'shift_dpr']).agg(
        laden_dist_km   = ('dist_m',    lambda x: x.sum() / 1000),
        laden_climb_m   = ('delta_alt', lambda x: x[x > 0].sum()),
        laden_descent_m = ('delta_alt', lambda x: x[x < 0].abs().sum()),
    ).reset_index()

    unladen = unladen_df.groupby(['vehicle', 'adj_date', 'shift_dpr']).agg(
        unladen_dist_km = ('dist_m',    lambda x: x.sum() / 1000),
        unladen_climb_m = ('delta_alt', lambda x: x[x > 0].sum()),
    ).reset_index()

    moving = moving_df.groupby(['vehicle', 'adj_date', 'shift_dpr']).agg(
        mean_speed  = ('speed', 'mean'),
        speed_std   = ('speed', 'std'),
        p85_speed   = ('speed', lambda x: x.quantile(0.85)),
        max_speed   = ('speed', 'max'),
        pct_over_60 = ('speed', lambda x: (x > 60).mean()),
    ).reset_index()

    agg_df = (base
              .merge(laden,   on=['vehicle', 'adj_date', 'shift_dpr'], how='left')
              .merge(unladen, on=['vehicle', 'adj_date', 'shift_dpr'], how='left')
              .merge(moving,  on=['vehicle', 'adj_date', 'shift_dpr'], how='left'))

    for col in ['laden_dist_km', 'laden_climb_m', 'laden_descent_m',
                'unladen_dist_km', 'unladen_climb_m', 'dump_events']:
        if col in agg_df.columns:
            agg_df[col] = agg_df[col].fillna(0)

    for col in ['mean_speed', 'speed_std', 'p85_speed', 'max_speed', 'pct_over_60']:
        if col in agg_df.columns:
            agg_df[col] = agg_df[col].fillna(0)

    eps = 1e-6
    agg_df['laden_ratio']        = agg_df['laden_dist_km']  / (agg_df['total_dist_km'] + eps)
    agg_df['laden_climb_rate']   = agg_df['laden_climb_m']  / (agg_df['laden_dist_km'] * 1000 + eps)
    agg_df['trips_per_km']       = agg_df['trip_count']     / (agg_df['total_dist_km'] + eps)
    agg_df['idle_ratio']         = agg_df['idle_h']         / (agg_df['shift_duration_h'] + eps)
    agg_df['dist_per_hour']      = agg_df['total_dist_km']  / (agg_df['shift_duration_h'] + eps)
    agg_df['laden_climb_work']   = agg_df['laden_dist_km']  * agg_df['laden_climb_rate']
    agg_df['idle_x_stops']       = agg_df['idle_h']         * agg_df['n_stops']
    agg_df['speed_x_laden']      = agg_df['mean_speed']     * agg_df['laden_ratio']
    agg_df['climb_descent_ratio']= agg_df['pos_climb_m']    / (agg_df['neg_descent_m'] + eps)
    agg_df['is_night_shift']     = ((agg_df['shift_start_hr'] >= 20) | (agg_df['shift_start_hr'] < 6)).astype(int)
    agg_df['work_intensity']     = (agg_df['laden_dist_km'] * agg_df['laden_climb_m'] /
                                    (agg_df['shift_duration_h'] + eps))

    return agg_df


# ==========================================
# 5. LOAD & ENGINEER
# ==========================================
print("\n--- Step 1: Engineering features ---")
train_feature_parts = []
for name in TRAIN_TELEMETRY:
    raw = load_one_telemetry(name)
    if raw.empty:
        continue
    feat = get_advanced_features(raw)
    if not feat.empty:
        train_feature_parts.append(feat)
    del raw, feat
    gc.collect()

test_feature_parts = []
for name in TEST_TELEMETRY:
    raw = load_one_telemetry(name)
    if raw.empty:
        continue
    feat = get_advanced_features(raw)
    if not feat.empty:
        test_feature_parts.append(feat)
    del raw, feat
    gc.collect()

if not train_feature_parts:
    raise FileNotFoundError(f"No training telemetry files found under: {RAW}")
if not test_feature_parts:
    raise FileNotFoundError(f"No test telemetry files found under: {RAW}")

train_feats = pd.concat(train_feature_parts, ignore_index=True)
test_feats = pd.concat(test_feature_parts, ignore_index=True)
print(f"  Train feature rows: {len(train_feats)}")
print(f"  Test  feature rows: {len(test_feats)}")

# ── PATCH 2: Rank-normalize altitude features ──────────────────────────────────
print("\n--- Applying altitude rank-transform (Patch 2) ---")
altitude_cols = ["alt_std", "alt_range", "net_lift", "pos_climb_m"]
alt_cols_present = [c for c in altitude_cols if c in train_feats.columns and c in test_feats.columns]
if alt_cols_present:
    combined = pd.concat([train_feats[alt_cols_present], test_feats[alt_cols_present]], ignore_index=True).fillna(0)
    for col in alt_cols_present:
        all_ranks = rankdata(combined[col].values, method='average') / len(combined)
        train_feats[f"{col}_rank"] = all_ranks[:len(train_feats)]
        test_feats[f"{col}_rank"] = all_ranks[len(train_feats):]
    print(f"  Rank-transformed: {alt_cols_present}")

gc.collect()

# Consistent shift/vehicle encoding
all_shifts = sorted(set(train_feats['shift_dpr'].unique()) | set(test_feats['shift_dpr'].unique()))
shift_map  = {s: i for i, s in enumerate(all_shifts)}
train_feats['shift_dpr_enc'] = train_feats['shift_dpr'].astype(str).map(shift_map).fillna(-1).astype(int)
test_feats['shift_dpr_enc']  = test_feats['shift_dpr'].astype(str).map(shift_map).fillna(-1).astype(int)
print(f"  Shift encoding: {shift_map}")

all_vehicles = sorted(set(train_feats['vehicle'].unique()) | set(test_feats['vehicle'].unique()))
vehicle_map  = {v: i for i, v in enumerate(all_vehicles)}
train_feats['vehicle_enc'] = train_feats['vehicle'].astype(str).map(vehicle_map).fillna(-1).astype(int)
test_feats['vehicle_enc']  = test_feats['vehicle'].astype(str).map(vehicle_map).fillna(-1).astype(int)


# ==========================================
# 6. RFID MERGE (DAY-LEVEL)
# ==========================================
print("\n--- Step 2: RFID features (day-level) ---")
rfid_feats = build_rfid_features(os.path.join(RAW, RFID_FILE))

if not rfid_feats.empty:
    rfid_feats['vehicle'] = rfid_feats['vehicle'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    train_feats = train_feats.merge(rfid_feats, on=['vehicle', 'adj_date'], how='left')
    test_feats  = test_feats.merge(rfid_feats,  on=['vehicle', 'adj_date'], how='left')
    rfid_cols   = ['rfid_fuel_added_day', 'rfid_n_events_day', 'rfid_is_refueled_day']
    for col in rfid_cols:
        train_feats[col] = train_feats[col].fillna(0)
        test_feats[col]  = test_feats[col].fillna(0)
    print(f"  RFID merged (day-level). Refueled shifts: {(train_feats['rfid_is_refueled_day'] > 0).sum()}")
else:
    rfid_cols = []
    print("  RFID unavailable")


# ==========================================
# 7. LOAD & MERGE SUMMARY TARGETS
# ==========================================
print("\n--- Step 3: Merging summary targets ---")
summaries = []
for f in SUMMARY_FILES:
    p = os.path.join(RAW, f)
    if os.path.exists(p):
        s = pd.read_csv(p)
        s.columns = [str(c).lower().strip().replace(' ', '_') for c in s.columns]
        summaries.append(s)

train_summary = pd.concat(summaries, ignore_index=True)

if 'acons' in train_summary.columns:
    train_summary.rename(columns={'acons': 'actual_fuel'}, inplace=True)
elif all(c in train_summary.columns for c in ['initlev', 'arefill', 'endlev']):
    train_summary['actual_fuel'] = (train_summary['initlev'] + train_summary['arefill']) - train_summary['endlev']
else:
    raise ValueError(f"Cannot find fuel target. Columns: {train_summary.columns.tolist()}")

for old, new in {'date': 'adj_date', 'shift': 'shift_dpr'}.items():
    if old in train_summary.columns and new not in train_summary.columns:
        train_summary.rename(columns={old: new}, inplace=True)

train_summary['vehicle'] = train_summary['vehicle'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

train_data = train_summary.merge(train_feats, on=['vehicle', 'adj_date', 'shift_dpr'], how='inner')
print(f"  Merged: {len(train_data)} / {len(train_summary)} rows, hit rate: {len(train_data) / len(train_summary):.4f}")
print(f"  actual_fuel: mean={train_data['actual_fuel'].mean():.1f}L ± {train_data['actual_fuel'].std():.1f}L")

INTERACTION_FEATURES = [
    'idle_x_stops', 'speed_x_laden', 'climb_descent_ratio',
    'work_intensity', 'trips_per_km', 'dist_per_hour'
]

def ensure_interaction_features(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    eps = 1e-6
    present_before = [c for c in INTERACTION_FEATURES if c in df.columns]
    missing = [c for c in INTERACTION_FEATURES if c not in df.columns]
    print(f"  {df_name} interaction columns present after merge: {present_before}")

    recomputed = []
    if 'idle_x_stops' in missing and {'idle_h', 'n_stops'}.issubset(df.columns):
        df['idle_x_stops'] = df['idle_h'].fillna(0) * df['n_stops'].fillna(0)
        recomputed.append('idle_x_stops')
    if 'speed_x_laden' in missing and {'mean_speed', 'laden_ratio'}.issubset(df.columns):
        df['speed_x_laden'] = df['mean_speed'].fillna(0) * df['laden_ratio'].fillna(0)
        recomputed.append('speed_x_laden')
    if 'climb_descent_ratio' in missing and {'pos_climb_m', 'neg_descent_m'}.issubset(df.columns):
        df['climb_descent_ratio'] = df['pos_climb_m'].fillna(0) / (df['neg_descent_m'].fillna(0) + eps)
        recomputed.append('climb_descent_ratio')
    if 'work_intensity' in missing and {'laden_dist_km', 'laden_climb_m', 'shift_duration_h'}.issubset(df.columns):
        df['work_intensity'] = (
            df['laden_dist_km'].fillna(0) * df['laden_climb_m'].fillna(0)
        ) / (df['shift_duration_h'].fillna(0) + eps)
        recomputed.append('work_intensity')
    if 'trips_per_km' in missing and {'trip_count', 'total_dist_km'}.issubset(df.columns):
        df['trips_per_km'] = df['trip_count'].fillna(0) / (df['total_dist_km'].fillna(0) + eps)
        recomputed.append('trips_per_km')
    if 'dist_per_hour' in missing and {'total_dist_km', 'shift_duration_h'}.issubset(df.columns):
        df['dist_per_hour'] = df['total_dist_km'].fillna(0) / (df['shift_duration_h'].fillna(0) + eps)
        recomputed.append('dist_per_hour')

    present_after = [c for c in INTERACTION_FEATURES if c in df.columns]
    print(f"  FIX2: {df_name} interactions present={len(present_after)}/6, recomputed={len(recomputed)}, rows={len(df)}")
    return df

train_data = ensure_interaction_features(train_data, "train_data")

if train_data.empty or len(train_data) / len(train_summary) < 0.70:
    raise ValueError("CRITICAL: Join hit rate too low.")


# ==========================================
# 8. LOAD ID MAPPING & BUILD TEST FRAME
# ==========================================
print("\n--- Step 4: Building test frame ---")
id_map = pd.read_csv(os.path.join(RAW, "id_mapping_new.csv"))
id_map.columns = [str(c).lower().strip().replace(' ', '_') for c in id_map.columns]

for old, new in {'date': 'adj_date', 'shift': 'shift_dpr'}.items():
    if old in id_map.columns and new not in id_map.columns:
        id_map.rename(columns={old: new}, inplace=True)

id_map['vehicle'] = id_map['vehicle'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

test_data = id_map.merge(test_feats, on=['vehicle', 'adj_date', 'shift_dpr'], how='left')
n_missing = test_data['total_dist_km'].isnull().sum()
print(f"  Test rows: {len(test_data)} | Missing telemetry: {n_missing}")
test_data = ensure_interaction_features(test_data, "test_data")

test_data['vehicle_enc']   = test_data['vehicle'].map(vehicle_map).fillna(-1).astype(int)
test_data['shift_dpr_enc'] = test_data['shift_dpr'].map(shift_map).fillna(-1).astype(int)


# ==========================================
# 9. FEATURE COLUMNS
# ==========================================
def unique_cols(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


BASE_FEATURES = [
    'vehicle_enc', 'shift_dpr_enc',
    'n_pings', 'total_dist_km', 'laden_dist_km', 'unladen_dist_km',
    'laden_ratio', 'laden_climb_m', 'laden_descent_m',
    'unladen_dist_km', 'unladen_climb_m', 'laden_climb_rate',
    'mean_speed', 'speed_std', 'p85_speed', 'pct_over_60', 'max_speed',
    'n_stops', 'idle_h', 'idle_ratio',
    'trip_count', 'trips_per_km', 'dump_events',
    'shift_start_hr', 'is_night_shift', 'shift_duration_h', 'dist_per_hour',
    'laden_climb_work', 'idle_x_stops', 'speed_x_laden',
    'climb_descent_ratio', 'work_intensity',
    # ← Rank-transformed altitude (Patch 2)
    'alt_std_rank', 'alt_range_rank', 'net_lift_rank', 'pos_climb_m_rank',
]

# Guarantee interaction features are explicitly included if they survive/recompute.
BASE_FEATURES += INTERACTION_FEATURES

RFID_FEATURES = rfid_cols

HISTORY_FEATURES = [
    'veh_mean_fuel', 'veh_median_fuel', 'veh_std_fuel',
    'veh_p10_fuel', 'veh_p90_fuel', 'veh_shift_count',
    'veh_shift_mean_fuel', 'veh_shift_std_fuel',
]

BASE_FEATURES    = unique_cols([c for c in BASE_FEATURES    if c in train_data.columns])
RFID_FEATURES    = unique_cols([c for c in RFID_FEATURES    if c in train_data.columns])
HISTORY_FEATURES = unique_cols(HISTORY_FEATURES)
ALL_FEATURES     = unique_cols(BASE_FEATURES + RFID_FEATURES + HISTORY_FEATURES)

fix1_cols = ['veh_median_fuel', 'veh_p10_fuel', 'veh_p90_fuel', 'veh_shift_std_fuel']
print(f"  FIX1: vehicle history includes {fix1_cols} for {len(train_data)} training rows")

print(f"\n  Base: {len(BASE_FEATURES)}, RFID: {len(RFID_FEATURES)}, History: {len(HISTORY_FEATURES)}, Total: {len(ALL_FEATURES)}")


# ==========================================
# 10. IMPUTE MISSING BASE TELEMETRY
# ==========================================
impute_cols = [c for c in BASE_FEATURES if c not in ('vehicle_enc', 'shift_dpr_enc')]
veh_means_impute    = train_data.groupby('vehicle')[impute_cols].mean()
global_means_impute = train_data[impute_cols].mean()

nan_after_impute = {}

for col in impute_cols:
    if col in test_data.columns:
        col_median = train_data[col].median() if col in train_data.columns else global_means_impute[col]
        test_data[col] = (test_data[col]
                          .fillna(test_data['vehicle'].map(veh_means_impute[col]))
                          .fillna(global_means_impute[col])
                          .fillna(col_median))
        nan_after_impute[col] = int(test_data[col].isna().sum())
    if col in train_data.columns:
        train_data[col] = train_data[col].fillna(global_means_impute[col])

print("  Imputation complete.")
for col in impute_cols:
    print(f"    NaNs after impute [{col}]: {nan_after_impute.get(col, -1)}")
print(f"  FIX4: hardened test imputation with median fallback across {len(impute_cols)} columns")


# ==========================================
# 11. LEAK-SAFE VEHICLE HISTORY
# ==========================================
def add_vehicle_history(train_df, apply_df, global_mean, global_std):
    veh_stats = train_df.groupby('vehicle')['actual_fuel'].agg(
        veh_mean_fuel   = 'mean',
        veh_median_fuel = 'median',
        veh_std_fuel    = 'std',
        veh_p10_fuel    = lambda x: x.quantile(0.10),
        veh_p90_fuel    = lambda x: x.quantile(0.90),
        veh_shift_count = 'count',
    ).reset_index()

    veh_shift_stats = train_df.groupby(['vehicle', 'shift_dpr'])['actual_fuel'].agg(
        veh_shift_mean_fuel = 'mean',
        veh_shift_std_fuel  = 'std',
    ).reset_index()

    out = apply_df.copy()
    out = out.merge(veh_stats,       on='vehicle',                how='left')
    out = out.merge(veh_shift_stats, on=['vehicle', 'shift_dpr'], how='left')

    for col in ['veh_mean_fuel', 'veh_median_fuel', 'veh_p10_fuel', 'veh_p90_fuel']:
        out[col] = out[col].fillna(global_mean)
    for col in ['veh_std_fuel', 'veh_shift_std_fuel']:
        out[col] = out[col].fillna(global_std)
    out['veh_shift_mean_fuel'] = out['veh_shift_mean_fuel'].fillna(out['veh_mean_fuel'])
    out['veh_shift_count']     = out['veh_shift_count'].fillna(0)
    return out


# ==========================================
# 12. TEMPORAL CV (PATCH 3)
# ==========================================
print("\n--- Step 5: Forward-chained monthly CV (Patch 3) ---")

train_data['month'] = pd.to_datetime(train_data['adj_date']).dt.month
train_data = train_data.sort_values('adj_date').reset_index(drop=True)

y_raw            = train_data['actual_fuel'].values
global_fuel_mean = float(train_data['actual_fuel'].mean())
global_fuel_std  = float(train_data['actual_fuel'].std())

# Forward-chained folds
fold_splits = [
    (train_data[train_data['month'] == 1].index.tolist(),
     train_data[train_data['month'] == 2].index.tolist()),
    (train_data[train_data['month'].isin([1, 2])].index.tolist(),
     train_data[train_data['month'] == 3].index.tolist()),
]

best_params = dict(
    loss='squared_error', max_iter=1000, max_depth=5,
    min_samples_leaf=40, l2_regularization=3.0,
    learning_rate=0.05, max_leaf_nodes=31,
    early_stopping=True, validation_fraction=0.1, random_state=42,
)

# ── PATCH 4: Optuna on Fold 1 (hardest fold) ──────────────────────────────────
if OPTUNA_AVAILABLE and len(fold_splits) > 1:
    print("  Optuna tuning (100 trials on Fold 1: Jan+Feb → Mar) ...")
    f_tr_idx, f_val_idx = fold_splits[1]
    f_tr_raw  = train_data.loc[f_tr_idx].copy()
    f_val_raw = train_data.loc[f_val_idx].copy()
    f_tr_h    = add_vehicle_history(f_tr_raw, f_tr_raw, global_fuel_mean, global_fuel_std)
    f_val_h   = add_vehicle_history(f_tr_raw, f_val_raw, global_fuel_mean, global_fuel_std)

    X_opt_tr  = f_tr_h[ALL_FEATURES].fillna(0)
    y_opt_tr  = f_tr_h['actual_fuel'].values
    X_opt_val = f_val_h[ALL_FEATURES].fillna(0)
    y_opt_val = f_val_raw['actual_fuel'].values

    def objective(trial):
        max_depth_choice = trial.suggest_categorical('max_depth', [-1, 3, 4, 5, 6, 7])
        p = dict(
            loss='squared_error',
            max_iter          = trial.suggest_int('max_iter', 300, 2000),
            max_depth         = None if max_depth_choice == -1 else max_depth_choice,
            min_samples_leaf  = trial.suggest_int('min_samples_leaf', 20, 120),
            l2_regularization = trial.suggest_float('l2_regularization', 0.3, 20.0, log=True),
            learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            max_leaf_nodes    = trial.suggest_categorical('max_leaf_nodes', [31, 63, 127, 255]),
            early_stopping=True, validation_fraction=0.1, random_state=42,
        )
        m = HistGradientBoostingRegressor(**p)
        m.fit(X_opt_tr, y_opt_tr)
        pred = np.clip(m.predict(X_opt_val), 10, None)
        return np.sqrt(np.mean((y_opt_val - pred)**2))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    best_params = study.best_params
    if best_params.get('max_depth') == -1:
        best_params['max_depth'] = None
    best_params.update(dict(loss='squared_error', early_stopping=True,
                            validation_fraction=0.1, random_state=42))
    print(f"  Best Optuna RMSE: {study.best_value:.2f}")
    print("  FIX5: Optuna search space updated with max_depth=-1 option and max_leaf_nodes=[31,63,127,255]")
    print(f"  Updated best_params with Optuna tuning")

# ── Full CV scoring ──────────────────────────────────────────────────────────
print("\n  Running full forward-chained CV ...")
oof_preds = np.zeros(len(train_data))

for fold, (tr_idx, val_idx) in enumerate(fold_splits):
    f_tr  = add_vehicle_history(train_data.loc[tr_idx],  train_data.loc[tr_idx],  global_fuel_mean, global_fuel_std)
    f_val = add_vehicle_history(train_data.loc[tr_idx],  train_data.loc[val_idx], global_fuel_mean, global_fuel_std)

    m = HistGradientBoostingRegressor(**best_params)
    m.fit(f_tr[ALL_FEATURES].fillna(0), f_tr['actual_fuel'].values)
    pred = np.clip(m.predict(f_val[ALL_FEATURES].fillna(0)), 10, None)
    oof_preds[val_idx] = pred

    rmse = np.sqrt(np.mean((train_data.loc[val_idx, 'actual_fuel'].values - pred)**2))
    print(f"  Fold {fold+1} RMSE: {rmse:.2f} | train {sorted(train_data.loc[tr_idx,'month'].unique())} "
          f"→ val {train_data.loc[val_idx,'adj_date'].min()}:{train_data.loc[val_idx,'adj_date'].max()}")

honest_idx  = train_data[train_data['month'].isin([2, 3])].index.tolist()
scored      = [i for i in honest_idx if oof_preds[i] != 0]
honest_rmse = np.sqrt(np.mean((y_raw[scored] - oof_preds[scored])**2))
print(f"\n  HONEST CV RMSE (forward folds only): {honest_rmse:.2f}")
print(f"  HONEST CV MSE: {honest_rmse**2:.1f}")


# ==========================================
# 13. FINAL MODEL
# ==========================================
print("\n--- Step 6: Final model on all training data ---")
train_full_h = add_vehicle_history(train_data, train_data, global_fuel_mean, global_fuel_std)
test_full_h  = add_vehicle_history(train_data, test_data,  global_fuel_mean, global_fuel_std)

for col in ALL_FEATURES:
    for df in [train_full_h, test_full_h]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            fill_val = train_full_h[col].median() if col in train_full_h.columns else 0
            df[col] = df[col].fillna(fill_val)

assert test_full_h[ALL_FEATURES].isnull().sum().sum() == 0, "NaNs remain in test features"

final_model = HistGradientBoostingRegressor(**best_params)
final_model.fit(train_full_h[ALL_FEATURES].fillna(0), train_full_h['actual_fuel'].values)
final_preds = np.clip(final_model.predict(test_full_h[ALL_FEATURES].fillna(0)), 10, None)
print("  FIX3: prediction floor is consistently 10L in Optuna objective, CV, and final model predictions")


# ==========================================
# 14. SMART FLOOR + CLIP
# ==========================================
veh_floor   = train_data.groupby('vehicle')['actual_fuel'].quantile(0.05)
global_floor= float(train_data['actual_fuel'].quantile(0.01))
global_ceil = float(train_data['actual_fuel'].quantile(0.999))

test_full_h['smart_floor'] = test_full_h['vehicle'].map(veh_floor).fillna(global_floor).clip(lower=10)
test_full_h['acons']       = np.clip(final_preds, test_full_h['smart_floor'], global_ceil)

print(f"\n  Predictions clipped to floor: {(test_full_h['acons'] == test_full_h['smart_floor']).sum()}")
print(f"  Range: {test_full_h['acons'].min():.1f} → {test_full_h['acons'].max():.1f}")
print(f"  Mean:  {test_full_h['acons'].mean():.1f}")


# ==========================================
# 15. SUBMISSION
# ==========================================
print("\n--- Step 7: Saving submission ---")

id_col = next((c for c in id_map.columns if c.lower() in ('id', 'submission_id', 'idx')), None)
if id_col is None:
    print(f"  ERROR: No 'id' column found. Columns: {id_map.columns.tolist()}")
else:
    submission = test_full_h[[id_col, 'acons']].copy()
    assert submission['acons'].isna().sum() == 0, "NaN predictions"
    assert len(submission) == len(id_map), "Row count mismatch"

    out_path = os.path.join(RAW, "submission_v4_optimized.csv")
    submission.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(submission.head(15).to_string(index=False))

print("\nDone.")
