import pandas as pd
import numpy as np
import os
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ==========================================
# 1. PATHS — CORRECTED TEST PERIOD
# ==========================================
ROOT = Path(__file__).resolve().parent
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

# THE ACTUAL TEST SET
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
        else:
            print(f"  WARNING: Not found: {name}")
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
# 3. RFID FEATURES
# ==========================================
def build_rfid_features(rfid_path: str) -> pd.DataFrame:
    """
    Computes per-vehicle per-shift RFID refuel features.
    Uses the same adj_date (+2h) and shift_dpr convention as telemetry.
    """
    if not os.path.exists(rfid_path):
        print(f"  WARNING: RFID file not found: {rfid_path}")
        return pd.DataFrame()

    rfid = pd.read_parquet(rfid_path)
    rfid.columns = [str(c).lower().strip().replace(' ', '_') for c in rfid.columns]

    # Normalize timestamp/date. Prefer event timestamp, fallback to date_dpr/date.
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
        print("  WARNING: RFID has no timestamp/date column to build adj_date")
        return pd.DataFrame()

    # Derive shift from hour — must match whatever shift_dpr convention
    # the telemetry uses. We will join on shift_dpr so we need to match it.
    # Print unique hours per shift to verify after first run.
    rfid['vehicle']   = rfid['vehicle'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

    # Identify fuel quantity column
    qty_col = next((c for c in rfid.columns if any(k in c for k in
                    ['qty', 'quantity', 'fuel', 'volume', 'litres', 'liters'])), None)
    if qty_col is None:
        print(f"  WARNING: Cannot find fuel qty column in RFID. Columns: {rfid.columns.tolist()}")
        return pd.DataFrame()

    print(f"  RFID qty column: '{qty_col}'")
    print(f"  RFID sample:\n{rfid[['vehicle', 'adj_date', qty_col]].head(5)}")

    # Aggregate per vehicle × adj_date
    # We deliberately do NOT split by shift here — refueling events often
    # straddle shift boundaries. Day-level features are safer.
    rfid_day = rfid.groupby(['vehicle', 'adj_date']).agg(
        rfid_fuel_added_day   = (qty_col, 'sum'),
        rfid_n_events_day     = (qty_col, 'count'),
        rfid_is_refueled_day  = (qty_col, lambda x: int((x > 0).any())),
        rfid_max_single_fill  = (qty_col, 'max'),
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

    # adj_date: the +2h convention (same as -22h mathematically)
    # A "day" starts at 22:00 previous night → adding 2h shifts 22:xx into the next date
    df['adj_date'] = (df['ts'] + pd.Timedelta(hours=2)).dt.strftime('%Y-%m-%d')

    # shift_dpr: use the column directly if present (it IS in telemetry for training rows)
    # For test rows it may or may not be present — check and handle
    if 'shift_dpr' not in df.columns or df['shift_dpr'].isna().mean() > 0.5:
        print("  WARNING: shift_dpr missing or mostly NaN — deriving from hour")
        # Fallback derivation — adjust labels to match your actual shift_dpr unique values
        # (Step 1 revealed: ['C', 'A', 'B'] → Night/Morning/Afternoon or similar)
        # Print the unique values and adjust these strings accordingly
        h = df['ts'].dt.hour
        df['shift_dpr'] = np.select(
            [((h >= 22) | (h < 6)),   # Night
             ((h >= 6)  & (h < 14)),  # Morning
             ((h >= 14) & (h < 22))], # Afternoon
            ['C', 'B', 'A'],           # ← ADJUST to match your actual shift_dpr strings
            default='B'
        )

    df['delta_alt'] = df.groupby('vehicle')['altitude'].diff().fillna(0)
    df['dist_m']    = df['disthav'].fillna(0)
    df['speed']     = df['speed'].fillna(0)

    # Dump signal
    if 'analog_input_1' in df.columns and (df['analog_input_1'] > 2.5).mean() > 0.001:
        df['is_dumping'] = (df['analog_input_1'] > 2.5).astype(int)
        df['cycle_id']   = (df.groupby('vehicle')['is_dumping']
                              .transform(lambda x: (x.diff() < 0).cumsum()))
    else:
        df['is_dumping'] = 0
        df['cycle_id']   = (df.groupby('vehicle')['dist_m']
                              .transform(lambda x: (x == 0).cumsum()))

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
        # Altitude — keep RELATIVE features only (neutralize depth drift)
        alt_std          = ('altitude', 'std'),
        alt_range        = ('altitude', lambda x: x.max() - x.min()),
        net_lift         = ('altitude', lambda x: float(x.iloc[-1] - x.iloc[0])
                                                  if len(x) > 1 else 0.0),
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

    fill_zero = ['laden_dist_km', 'laden_climb_m', 'laden_descent_m',
                 'unladen_dist_km', 'unladen_climb_m', 'dump_events']
    for col in fill_zero:
        if col in agg_df.columns:
            agg_df[col] = agg_df[col].fillna(0)

    fill_speed = ['mean_speed', 'speed_std', 'p85_speed', 'max_speed', 'pct_over_60']
    for col in fill_speed:
        if col in agg_df.columns:
            agg_df[col] = agg_df[col].fillna(0)

    # ── Derived features ────────────────────────────────────────────────────
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
    agg_df['is_night_shift']     = ((agg_df['shift_start_hr'] >= 20) |
                                     (agg_df['shift_start_hr'] < 6)).astype(int)
    agg_df['work_intensity']     = (agg_df['laden_dist_km'] * agg_df['laden_climb_m'] /
                                    (agg_df['shift_duration_h'] + eps))

    # RANK-TRANSFORM the altitude features to neutralize mine-depth drift
    # This was the fix for adversarial AUC 0.639 on altitude_mean/std/net_lift
    for col in ['alt_std', 'alt_range', 'net_lift', 'pos_climb_m']:
        if col in agg_df.columns:
            from scipy.stats import rankdata
            agg_df[f'{col}_rank'] = rankdata(agg_df[col].fillna(0), method='average') / len(agg_df)

    return agg_df


# ==========================================
# 5. LOAD & ENGINEER
# ==========================================
print("\n--- Step 1: Engineering features ---")
train_feature_parts = []
train_shift_values = set()
for name in TRAIN_TELEMETRY:
    raw = load_one_telemetry(name)
    if raw.empty:
        continue
    if 'shift_dpr' in raw.columns:
        train_shift_values.update(raw['shift_dpr'].dropna().astype(str).unique().tolist())
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
    raise FileNotFoundError(f"No training telemetry files were found under: {RAW}")
if not test_feature_parts:
    raise FileNotFoundError(f"No test telemetry files were found under: {RAW}")

train_feats = pd.concat(train_feature_parts, ignore_index=True)
test_feats = pd.concat(test_feature_parts, ignore_index=True)

print("\n  AUDIT shift_dpr unique values in train telemetry:")
print(" ", sorted(train_shift_values) if train_shift_values else "MISSING")

print(f"\n  Train feature rows: {len(train_feats)}")
print(f"  Test  feature rows: {len(test_feats)}")

gc.collect()

# Re-rank altitude features jointly across train+test (prevents train/test rank gap)
from scipy.stats import rankdata
for col in ['alt_std', 'alt_range', 'net_lift', 'pos_climb_m']:
    rank_col = f'{col}_rank'
    if rank_col in train_feats.columns and rank_col in test_feats.columns:
        combined = pd.concat([train_feats[col], test_feats[col]]).fillna(0)
        all_ranks = rankdata(combined, method='average') / len(combined)
        train_feats[rank_col] = all_ranks[:len(train_feats)]
        test_feats[rank_col]  = all_ranks[len(train_feats):]

# Consistent shift encoding across train+test
all_shifts = sorted(set(train_feats['shift_dpr'].unique()) | set(test_feats['shift_dpr'].unique()))
shift_map  = {s: i for i, s in enumerate(all_shifts)}
train_feats['shift_dpr_enc'] = train_feats['shift_dpr'].astype(str).map(shift_map).fillna(-1).astype(int)
test_feats['shift_dpr_enc']  = test_feats['shift_dpr'].astype(str).map(shift_map).fillna(-1).astype(int)
print(f"\n  Shift encoding: {shift_map}")

# Consistent vehicle encoding
all_vehicles = sorted(set(train_feats['vehicle'].unique()) | set(test_feats['vehicle'].unique()))
vehicle_map  = {v: i for i, v in enumerate(all_vehicles)}
train_feats['vehicle_enc'] = train_feats['vehicle'].astype(str).map(vehicle_map).fillna(-1).astype(int)
test_feats['vehicle_enc']  = test_feats['vehicle'].astype(str).map(vehicle_map).fillna(-1).astype(int)


# ==========================================
# 6. RFID MERGE
# ==========================================
print("\n--- Step 2: RFID features ---")
rfid_feats = build_rfid_features(os.path.join(RAW, RFID_FILE))

if not rfid_feats.empty:
    rfid_feats['vehicle'] = rfid_feats['vehicle'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    train_feats = train_feats.merge(rfid_feats, on=['vehicle', 'adj_date'], how='left')
    test_feats  = test_feats.merge(rfid_feats,  on=['vehicle', 'adj_date'], how='left')
    rfid_cols   = ['rfid_fuel_added_day', 'rfid_n_events_day',
                   'rfid_is_refueled_day', 'rfid_max_single_fill']
    for col in rfid_cols:
        train_feats[col] = train_feats[col].fillna(0)
        test_feats[col]  = test_feats[col].fillna(0)
    print(f"  RFID merged. Refueled shifts in train: {(train_feats['rfid_is_refueled_day'] > 0).sum()}")
else:
    rfid_cols = []
    print("  RFID skipped — no file or columns found")


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

print("\n  SUMMARY COLUMNS:", train_summary.columns.tolist())
print("  SUMMARY HEAD:\n", train_summary.head(3).to_string())

# Build actual_fuel from available columns
if 'acons' in train_summary.columns:
    train_summary.rename(columns={'acons': 'actual_fuel'}, inplace=True)
elif all(c in train_summary.columns for c in ['initlev', 'arefill', 'endlev']):
    train_summary['actual_fuel'] = (train_summary['initlev'] + train_summary['arefill']) - train_summary['endlev']
else:
    raise ValueError(f"Cannot find fuel target. Columns: {train_summary.columns.tolist()}")

# Normalize join keys — print what we're joining on
rename_candidates = {'date': 'adj_date', 'shift': 'shift_dpr'}
for old, new in rename_candidates.items():
    if old in train_summary.columns and new not in train_summary.columns:
        train_summary.rename(columns={old: new}, inplace=True)

train_summary['vehicle'] = train_summary['vehicle'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

print(f"\n  Summary adj_date range: {train_summary['adj_date'].min()} → {train_summary['adj_date'].max()}")
print(f"  Summary shift_dpr unique: {sorted(train_summary['shift_dpr'].dropna().unique().tolist())}")
print(f"  Train feats adj_date range: {train_feats['adj_date'].min()} → {train_feats['adj_date'].max()}")
print(f"  Train feats shift_dpr unique: {sorted(train_feats['shift_dpr'].dropna().unique().tolist())}")

train_data = train_summary.merge(
    train_feats,
    on=['vehicle', 'adj_date', 'shift_dpr'],
    how='inner'
)

print(f"\n  Merged training rows: {len(train_data)}")
print(f"  Join hit rate: {len(train_data) / len(train_summary):.4f}")
print(f"  actual_fuel stats:\n{train_data['actual_fuel'].describe()}")

if train_data.empty:
    raise ValueError("CRITICAL: train_data is empty. Check that adj_date and shift_dpr formats match between summary and telemetry.")
if len(train_data) / len(train_summary) < 0.70:
    print("WARNING: Join hit rate < 70%. Print adj_date and shift_dpr samples from both sides to debug.")
    print("  Summary adj_date samples:", train_summary['adj_date'].head(5).tolist())
    print("  Feats adj_date samples:  ", train_feats['adj_date'].head(5).tolist())
    print("  Summary shift_dpr samples:", train_summary['shift_dpr'].head(5).tolist())
    print("  Feats shift_dpr samples:  ", train_feats['shift_dpr'].head(5).tolist())


# ==========================================
# 8. LOAD ID MAPPING & BUILD TEST FRAME
# ==========================================
print("\n--- Step 4: Building test frame ---")
id_map = pd.read_csv(os.path.join(RAW, "id_mapping_new.csv"))
id_map.columns = [str(c).lower().strip().replace(' ', '_') for c in id_map.columns]
print("  id_map columns:", id_map.columns.tolist())
print("  id_map head:\n", id_map.head(5).to_string())

# Rename to match join keys
for old, new in {'date': 'adj_date', 'shift': 'shift_dpr'}.items():
    if old in id_map.columns and new not in id_map.columns:
        id_map.rename(columns={old: new}, inplace=True)

id_map['vehicle'] = id_map['vehicle'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

test_data = id_map.merge(test_feats, on=['vehicle', 'adj_date', 'shift_dpr'], how='left')
n_missing = test_data['total_dist_km'].isnull().sum()
print(f"\n  Test rows: {len(test_data)} | Missing telemetry: {n_missing}")

# Re-apply encodings to test_data (vehicle/shift may have changed after merge)
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
    'n_pings',
    'total_dist_km', 'laden_dist_km', 'unladen_dist_km',
    'laden_ratio', 'laden_climb_m', 'laden_descent_m',
    'unladen_dist_km', 'unladen_climb_m', 'laden_climb_rate',
    'mean_speed', 'speed_std', 'p85_speed', 'pct_over_60', 'max_speed',
    'n_stops', 'idle_h', 'idle_ratio',
    'trip_count', 'trips_per_km', 'dump_events',
    'shift_start_hr', 'is_night_shift',
    'shift_duration_h', 'dist_per_hour',
    'laden_climb_work', 'idle_x_stops', 'speed_x_laden',
    'climb_descent_ratio', 'work_intensity',
    # Rank-transformed altitude (drift-neutralized)
    'alt_std_rank', 'alt_range_rank', 'net_lift_rank', 'pos_climb_m_rank',
]

RFID_FEATURES = rfid_cols  # empty list if RFID unavailable

HISTORY_FEATURES = [
    'veh_mean_fuel', 'veh_median_fuel', 'veh_std_fuel',
    'veh_p10_fuel', 'veh_p90_fuel', 'veh_shift_count',
    'veh_shift_mean_fuel', 'veh_shift_std_fuel',
]

# Only keep features that exist in the data
BASE_FEATURES    = unique_cols([c for c in BASE_FEATURES    if c in train_data.columns])
RFID_FEATURES    = unique_cols([c for c in RFID_FEATURES    if c in train_data.columns])
HISTORY_FEATURES = unique_cols(HISTORY_FEATURES)
ALL_FEATURES     = unique_cols(BASE_FEATURES + RFID_FEATURES + HISTORY_FEATURES)

print(f"\n  Base features:    {len(BASE_FEATURES)}")
print(f"  RFID features:    {len(RFID_FEATURES)}")
print(f"  History features: {len(HISTORY_FEATURES)}")
print(f"  Total:            {len(ALL_FEATURES)}")


# ==========================================
# 10. IMPUTE MISSING BASE TELEMETRY IN TEST
# ==========================================
impute_cols = [c for c in BASE_FEATURES if c not in ('vehicle_enc', 'shift_dpr_enc')]

veh_means_impute    = train_data.groupby('vehicle')[impute_cols].mean()
global_means_impute = train_data[impute_cols].mean()

for col in impute_cols:
    if col in test_data.columns:
        test_data[col] = (test_data[col]
                          .fillna(test_data['vehicle'].map(veh_means_impute[col]))
                          .fillna(global_means_impute[col]))
    if col in train_data.columns:
        train_data[col] = train_data[col].fillna(global_means_impute[col])

print("  Imputation complete.")


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
# 12. TEMPORAL CV
# ==========================================
print("\n--- Step 5: Cross-Validation ---")

train_data['month'] = pd.to_datetime(train_data['adj_date']).dt.month
train_data = train_data.sort_values('adj_date').reset_index(drop=True)

y_raw            = train_data['actual_fuel'].values
global_fuel_mean = float(train_data['actual_fuel'].mean())
global_fuel_std  = float(train_data['actual_fuel'].std())

# Forward-chained folds — no data from the future leaks into training
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

# ── Optuna on Fold 1 (Jan+Feb → Mar, most signal, mirrors LB gap) ──────────
if OPTUNA_AVAILABLE:
    print("\n  Optuna tuning (Fold 1: Jan+Feb → Mar) ...")
    f_tr_idx, f_val_idx = fold_splits[1]
    f_tr_raw  = train_data.loc[f_tr_idx].copy()
    f_val_raw = train_data.loc[f_val_idx].copy()
    f_tr_h    = add_vehicle_history(f_tr_raw, f_tr_raw,  global_fuel_mean, global_fuel_std)
    f_val_h   = add_vehicle_history(f_tr_raw, f_val_raw, global_fuel_mean, global_fuel_std)

    X_opt_tr  = f_tr_h[ALL_FEATURES]
    y_opt_tr  = f_tr_h['actual_fuel'].values
    X_opt_val = f_val_h[ALL_FEATURES]
    y_opt_val = f_val_raw['actual_fuel'].values

    def objective(trial):
        p = dict(
            loss='squared_error',
            max_iter          = trial.suggest_int('max_iter', 300, 2000),
            max_depth         = trial.suggest_int('max_depth', 3, 7),
            min_samples_leaf  = trial.suggest_int('min_samples_leaf', 20, 120),
            l2_regularization = trial.suggest_float('l2_regularization', 0.3, 20.0, log=True),
            learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            max_leaf_nodes    = trial.suggest_categorical('max_leaf_nodes', [15, 31, 63, 127]),
            early_stopping=True, validation_fraction=0.1, random_state=42,
        )
        m = HistGradientBoostingRegressor(**p)
        m.fit(X_opt_tr, y_opt_tr)
        pred = np.clip(m.predict(X_opt_val), 10, None)
        return np.sqrt(mean_squared_error(y_opt_val, pred))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    best_params = study.best_params
    best_params.update(dict(loss='squared_error', early_stopping=True,
                            validation_fraction=0.1, random_state=42))
    print(f"  Best Optuna RMSE: {study.best_value:.2f}")
    print(f"  Best params: {best_params}")

# ── Full CV scoring ──────────────────────────────────────────────────────────
print("\n  Running full CV ...")
oof_preds = np.zeros(len(train_data))

for fold, (tr_idx, val_idx) in enumerate(fold_splits):
    f_tr  = add_vehicle_history(train_data.loc[tr_idx],  train_data.loc[tr_idx],  global_fuel_mean, global_fuel_std)
    f_val = add_vehicle_history(train_data.loc[tr_idx],  train_data.loc[val_idx], global_fuel_mean, global_fuel_std)

    m = HistGradientBoostingRegressor(**best_params)
    m.fit(f_tr[ALL_FEATURES], f_tr['actual_fuel'].values)
    pred = np.clip(m.predict(f_val[ALL_FEATURES]), 10, None)
    oof_preds[val_idx] = pred

    rmse = np.sqrt(mean_squared_error(train_data.loc[val_idx, 'actual_fuel'].values, pred))
    print(f"  Fold {fold} RMSE: {rmse:.2f} | "
          f"train months {sorted(train_data.loc[tr_idx,'month'].unique())} "
          f"→ val {train_data.loc[val_idx,'adj_date'].min()}:{train_data.loc[val_idx,'adj_date'].max()}")

honest_idx  = train_data[train_data['month'].isin([2, 3])].index.tolist()
scored      = [i for i in honest_idx if oof_preds[i] != 0]
honest_rmse = np.sqrt(mean_squared_error(y_raw[scored], oof_preds[scored]))
print(f"\n  HONEST CV RMSE (forward folds only): {honest_rmse:.2f}")
print(f"  HONEST CV MSE:                        {honest_rmse**2:.1f}")


# ==========================================
# 13. FINAL MODEL
# ==========================================
print("\n--- Step 6: Final model on all training data ---")
train_full_h = add_vehicle_history(train_data, train_data, global_fuel_mean, global_fuel_std)
test_full_h  = add_vehicle_history(train_data, test_data,  global_fuel_mean, global_fuel_std)

# Belt-and-suspenders NaN fill
for col in ALL_FEATURES:
    for df in [train_full_h, test_full_h]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            fill_val = train_full_h[col].median() if col in train_full_h.columns else 0
            df[col] = df[col].fillna(fill_val)

assert test_full_h[ALL_FEATURES].isnull().sum().sum() == 0, "NaNs remain in test features"

final_model = HistGradientBoostingRegressor(**best_params)
final_model.fit(train_full_h[ALL_FEATURES], train_full_h['actual_fuel'].values)
final_preds = final_model.predict(test_full_h[ALL_FEATURES])


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

# Find the ID column in id_map
id_col = next((c for c in id_map.columns if c.lower() in ('id', 'submission_id', 'idx')), None)
if id_col is None:
    print(f"  ERROR: No 'id' column found. id_map columns: {id_map.columns.tolist()}")
else:
    submission = test_full_h[[id_col, 'acons']].copy()
    assert submission['acons'].isna().sum() == 0, "NaN predictions in submission"
    assert len(submission) == len(id_map),        "Row count mismatch with id_mapping"

    out_path = os.path.join(RAW, "submission.csv")
    submission.to_csv(out_path, index=False)
    # Also keep a versioned artifact for traceability.
    submission.to_csv(os.path.join(RAW, "submission_v4_fixed.csv"), index=False)
    print(f"  Saved: {out_path}")
    print(submission.head(10).to_string(index=False))


# ==========================================
# 16. DIAGNOSTICS
# ==========================================
try:
    fi = (pd.Series(final_model.feature_importances_, index=ALL_FEATURES)
          .sort_values(ascending=False))
    print(f"\n--- Top 20 Feature Importances ---")
    print(fi.head(20).to_string())
except Exception:
    pass

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(f"v4 Fixed | Honest CV RMSE: {honest_rmse:.2f}  MSE: {honest_rmse**2:.1f}",
             fontsize=13, fontweight='bold')

test_full_h['acons'].hist(bins=40, color='teal', edgecolor='black', ax=axes[0])
axes[0].set_title("Test prediction distribution"); axes[0].set_xlabel("Predicted fuel (L)")

actual_s = y_raw[scored]; pred_s = oof_preds[scored]
axes[1].scatter(actual_s, pred_s, alpha=0.3, s=10, color='steelblue')
lo, hi = min(actual_s.min(), pred_s.min()), max(actual_s.max(), pred_s.max())
axes[1].plot([lo, hi], [lo, hi], 'r--', lw=1)
axes[1].set_title(f"OOF actual vs predicted | RMSE={honest_rmse:.2f}")
axes[1].set_xlabel("Actual fuel (L)"); axes[1].set_ylabel("Predicted fuel (L)")

(actual_s - pred_s).tolist()
pd.Series(actual_s - pred_s).hist(bins=40, color='coral', edgecolor='black', ax=axes[2])
axes[2].axvline(0, color='black', lw=1, ls='--')
axes[2].set_title("Residuals"); axes[2].set_xlabel("Actual − Predicted (L)")

plt.tight_layout()
plt.savefig(os.path.join(RAW, "diagnostics_v4.png"), dpi=120)
plt.show()
print("\nDone.")