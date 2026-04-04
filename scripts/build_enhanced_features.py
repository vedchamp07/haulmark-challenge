#!/usr/bin/env python3
"""Enhanced feature engineering for ultra-low RMSE"""
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]


def assign_shift(ts):
    """Assign shift: C (22:00-05:59), A (06:00-13:59), B (14:00-21:59)"""
    hour = ts.hour
    if 6 <= hour < 14:
        return 'A'
    elif 14 <= hour < 22:
        return 'B'
    else:
        return 'C'


def assign_working_date(ts, shift):
    """Night shift C belongs to next day"""
    if shift == 'C' and ts.hour < 6:
        return (ts - pd.Timedelta(days=1)).date()
    return ts.date()


def detect_refuels_from_fuel_signal(fuel_series, threshold=50):
    """Detect refueling events from sudden fuel level jumps"""
    fuel_diff = fuel_series.diff()
    return (fuel_diff > threshold).sum()


def detect_dump_events(analog_series, threshold=2.5):
    """Detect dump events from analog_input_1 signal"""
    if analog_series.isna().all():
        return 0
    prev = analog_series.shift(1)
    crossings = ((analog_series > threshold) & ((prev <= threshold) | prev.isna()))
    return int(crossings.sum())


def segment_haul_cycles(df):
    """Identify haul cycles: load → travel → dump"""
    if 'analog_input_1' not in df.columns or df['analog_input_1'].isna().all():
        return pd.DataFrame()

    # Detect dump events
    df = df.sort_values('ts').copy()
    df['is_dump'] = (df['analog_input_1'] > 2.5).astype(int)
    df['dump_event'] = df['is_dump'].diff().gt(0).astype(int)

    # Assign cycle ID
    df['cycle_id'] = df['dump_event'].cumsum()

    if df['cycle_id'].max() == 0:
        return pd.DataFrame()

    # Aggregate per cycle
    cycles = df.groupby('cycle_id').agg(
        cycle_duration_min=('ts', lambda x: (x.max() - x.min()).total_seconds() / 60),
        cycle_distance_km=('disthav', lambda x: x.sum() / 1000),
        cycle_net_lift=('altitude', lambda x: x.max() - x.min()),
        cycle_max_speed=('speed', 'max'),
        cycle_mean_speed=('speed', 'mean'),
    ).reset_index()

    return cycles


def aggregate_shift_features(group):
    """Comprehensive shift-level feature extraction"""
    g = group.sort_values('ts').copy()

    # Basic metrics
    n = len(g)
    if n == 0:
        return pd.Series()

    # Time calculations
    g['time_gap'] = g['ts'].diff().dt.total_seconds().fillna(0).clip(0, 300)
    total_time = g['time_gap'].sum()

    # Ignition and movement
    ignition_on = (g['ignition'] == 1)
    moving = (g['speed'] > 2)
    idle = ignition_on & ~moving

    ignition_time = g.loc[ignition_on, 'time_gap'].sum() / 3600
    moving_time = g.loc[moving, 'time_gap'].sum() / 3600
    idle_time = g.loc[idle, 'time_gap'].sum() / 3600

    # Distance
    shift_km = g['disthav'].sum() / 1000

    # Altitude features
    alt_mean = g['altitude'].mean()
    alt_std = g['altitude'].std()
    alt_max = g['altitude'].max()
    alt_min = g['altitude'].min()
    alt_range = alt_max - alt_min

    g['alt_diff'] = g['altitude'].diff().fillna(0)
    net_lift = g['altitude'].iloc[-1] - g['altitude'].iloc[0] if len(g) > 1 else 0
    total_climb = g['alt_diff'].clip(lower=0).sum()
    total_descent = (-g['alt_diff']).clip(lower=0).sum()

    # Speed features
    speed_data = g.loc[g['speed'] > 0, 'speed']
    speed_mean = speed_data.mean() if len(speed_data) > 0 else 0
    speed_std = speed_data.std() if len(speed_data) > 0 else 0
    speed_max = g['speed'].max()
    speed_p50 = speed_data.quantile(0.5) if len(speed_data) > 0 else 0
    speed_p75 = speed_data.quantile(0.75) if len(speed_data) > 0 else 0
    speed_p90 = speed_data.quantile(0.9) if len(speed_data) > 0 else 0

    # Stop-go behavior
    prev_speed = g['speed'].shift(1).fillna(0)
    stop_events = ((prev_speed > 2) & (g['speed'] <= 2) & ignition_on).sum()

    # Dump events
    dump_count = detect_dump_events(g['analog_input_1'])

    # Fuel features
    fuel_refills = 0
    fuel_volatility = 0
    if 'fuel_volume' in g.columns and g['fuel_volume'].notna().sum() > 5:
        fuel_smoothed = g['fuel_volume'].rolling(window=5, min_periods=1).median()
        fuel_refills = detect_refuels_from_fuel_signal(fuel_smoothed)
        fuel_volatility = fuel_smoothed.std()

    # Derived metrics
    idle_fraction = idle_time / (ignition_time + 1e-6)
    stop_density = stop_events / (shift_km + 1e-6)
    km_per_hour = shift_km / (ignition_time + 1e-6)
    climb_descent_ratio = total_climb / (total_descent + 1.0)

    # Haul cycles
    cycles = segment_haul_cycles(g)
    cycles_count = len(cycles) if len(cycles) > 0 else 0
    mean_cycle_duration = cycles['cycle_duration_min'].mean() if cycles_count > 0 else 0
    mean_cycle_distance = cycles['cycle_distance_km'].mean() if cycles_count > 0 else 0
    mean_cycle_lift = cycles['cycle_net_lift'].mean() if cycles_count > 0 else 0

    # Signal quality
    avg_satellites = g['satellites'].mean()
    avg_hdop = g['gnss_hdop'].mean()

    # Accelerometer (vibration/road quality proxy)
    axis_x_std = g['axis_x'].std() if 'axis_x' in g.columns else 0
    axis_y_std = g['axis_y'].std() if 'axis_y' in g.columns else 0
    axis_z_std = g['axis_z'].std() if 'axis_z' in g.columns else 0
    vibration_magnitude = np.sqrt(axis_x_std**2 + axis_y_std**2 + axis_z_std**2)

    # Battery (vehicle health proxy)
    battery_mean = g['battery_level'].mean() if 'battery_level' in g.columns else 0

    # Angle variance (route zigzag)
    angle_std = g['angle'].std() if 'angle' in g.columns else 0

    # Speed variability (aggressive driving)
    speed_cv = (speed_std / (speed_mean + 1e-6)) if speed_mean > 0 else 0

    # Efficiency proxies
    km_per_climb = shift_km / (total_climb + 1.0)
    distance_per_dump = shift_km / (dump_count + 1.0)

    # Time of day features
    hour_start = g['ts'].iloc[0].hour
    hour_end = g['ts'].iloc[-1].hour

    return pd.Series({
        'ignition_on_hours': ignition_time,
        'moving_hours': moving_time,
        'idle_hours': idle_time,
        'shift_km': shift_km,
        'altitude_mean': alt_mean,
        'altitude_std': alt_std,
        'altitude_max': alt_max,
        'altitude_min': alt_min,
        'altitude_range': alt_range,
        'net_lift': net_lift,
        'total_climb_m': total_climb,
        'total_descent_m': total_descent,
        'speed_mean': speed_mean,
        'speed_std': speed_std,
        'speed_max': speed_max,
        'speed_p50': speed_p50,
        'speed_p75': speed_p75,
        'speed_p90': speed_p90,
        'stop_count': stop_events,
        'dump_event_count': dump_count,
        'fuel_refills_detected': fuel_refills,
        'fuel_volatility': fuel_volatility,
        'n_pings': n,
        'idle_fraction': idle_fraction,
        'stop_density': stop_density,
        'km_per_hour': km_per_hour,
        'climb_descent_ratio': climb_descent_ratio,
        'cycles_count': cycles_count,
        'mean_cycle_duration_min': mean_cycle_duration,
        'mean_cycle_distance_km': mean_cycle_distance,
        'mean_cycle_lift_m': mean_cycle_lift,
        'avg_satellites': avg_satellites,
        'avg_hdop': avg_hdop,
        'vibration_magnitude': vibration_magnitude,
        'battery_mean': battery_mean,
        'angle_std': angle_std,
        'speed_cv': speed_cv,
        'km_per_climb_m': km_per_climb,
        'distance_per_dump_km': distance_per_dump,
        'hour_start': hour_start,
        'hour_end': hour_end,
    })


def add_refuel_features(shift_df, summary_df):
    """Add refuel information from summary data"""
    refuel_cols = ['vehicle', 'date', 'shift', 'arefill', 'initlev', 'endlev', 'runhrs', 'lph']
    summary_sub = summary_df[refuel_cols].copy()
    summary_sub['date'] = pd.to_datetime(summary_sub['date']).dt.date

    merged = shift_df.merge(
        summary_sub,
        on=['vehicle', 'date', 'shift'],
        how='left'
    )

    # Fill missing refuels with 0
    for col in ['arefill', 'initlev', 'endlev', 'runhrs']:
        merged[col] = merged[col].fillna(0)

    # Compute expected consumption from tank levels
    merged['expected_cons_from_levels'] = (
        merged['initlev'] - merged['endlev'] + merged['arefill']
    )

    # Efficiency: liters per hour (from summary)
    merged['lph_summary'] = merged['lph']

    # Binary: did refuel happen?
    merged['refuel_flag'] = (merged['arefill'] > 10).astype(int)

    return merged


def add_lag_features(df, target_col='acons'):
    """Add lagged consumption features"""
    df = df.sort_values(['vehicle', 'date', 'shift']).copy()

    # Shift order: C, A, B
    shift_order = {'C': 0, 'A': 1, 'B': 2}

    for vehicle in df['vehicle'].unique():
        mask = (df['vehicle'] == vehicle)
        veh_df = df.loc[mask].copy()

        # Previous shift consumption
        if target_col in veh_df.columns:
            df.loc[mask, 'cons_lag1'] = veh_df[target_col].shift(1)
            df.loc[mask, 'cons_lag2'] = veh_df[target_col].shift(2)
            df.loc[mask, 'cons_lag3'] = veh_df[target_col].shift(3)

            # Rolling mean
            df.loc[mask, 'cons_rolling_mean_3'] = (
                veh_df[target_col].shift(1).rolling(window=3, min_periods=1).mean()
            )

    return df


def add_vehicle_aggregates(train_df, test_df):
    """Add vehicle-level historical aggregates"""
    # Compute from training data
    veh_stats = train_df.groupby('vehicle').agg(
        veh_mean_cons=('acons', 'mean'),
        veh_std_cons=('acons', 'std'),
        veh_mean_km=('shift_km', 'mean'),
        veh_mean_ignition_h=('ignition_on_hours', 'mean'),
        veh_mean_dumps=('dump_event_count', 'mean'),
        veh_mean_idle_frac=('idle_fraction', 'mean'),
    ).reset_index()

    # Shift-specific stats
    shift_stats = train_df.groupby(['vehicle', 'shift']).agg(
        veh_shift_mean=('acons', 'mean'),
        veh_shift_std=('acons', 'std'),
    ).reset_index()

    # Merge into train and test
    train_merged = train_df.merge(veh_stats, on='vehicle', how='left')
    train_merged = train_merged.merge(shift_stats, on=['vehicle', 'shift'], how='left')

    test_merged = test_df.merge(veh_stats, on='vehicle', how='left')
    test_merged = test_merged.merge(shift_stats, on=['vehicle', 'shift'], how='left')

    return train_merged, test_merged


def add_temporal_features(df):
    """Add time-based features"""
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['week_number'] = df['date'].dt.isocalendar().week
    df['day_of_month'] = df['date'].dt.day

    # Shift encoding
    shift_map = {'C': 0, 'A': 1, 'B': 2}
    df['shift_enc'] = df['shift'].map(shift_map)

    # Interaction: vehicle type × shift
    df['veh_shift_interaction'] = df['vehicle'].astype(str) + '_' + df['shift']

    return df


def process_telemetry_file(fpath, train_mode=True):
    """Process single telemetry file into shift features"""
    print(f"  Loading {fpath.name}...")
    df = pd.read_parquet(fpath)

    required_cols = ['vehicle', 'ts', 'latitude', 'longitude', 'altitude', 'speed',
                     'ignition', 'disthav', 'analog_input_1', 'satellites', 'gnss_hdop',
                     'angle', 'battery_level', 'axis_x', 'axis_y', 'axis_z']

    # Add fuel_volume only for train
    if train_mode and 'fuel_volume' in df.columns:
        required_cols.append('fuel_volume')

    available_cols = [c for c in required_cols if c in df.columns]
    df = df[available_cols].copy()

    # Convert timestamp
    df['ts'] = pd.to_datetime(df['ts'])
    if df['ts'].dt.tz is None:
        df['ts'] = df['ts'].dt.tz_localize('UTC')
    df['ts'] = df['ts'].dt.tz_convert('Asia/Kolkata')

    # Assign shift and date
    df['shift'] = df['ts'].apply(assign_shift)
    df['date'] = df.apply(lambda row: assign_working_date(row['ts'], row['shift']), axis=1)

    # Basic cleaning
    df = df[df['speed'] <= 80].copy()
    df['speed'] = df['speed'].clip(lower=0, upper=60)

    # Aggregate by (vehicle, date, shift)
    shift_feats = df.groupby(['vehicle', 'date', 'shift']).apply(
        aggregate_shift_features
    ).reset_index()

    return shift_feats


def main():
    data_dir = ROOT / 'new_data'

    # Load summary data (ground truth)
    print("Loading summary data...")
    summary = pd.concat([
        pd.read_csv(data_dir / 'smry_jan_train_ordered.csv'),
        pd.read_csv(data_dir / 'smry_feb_train_ordered.csv'),
        pd.read_csv(data_dir / 'smry_mar_train_ordered.csv'),
    ])
    summary['date'] = pd.to_datetime(summary['date']).dt.date
    print(f"  Summary rows: {len(summary)}")

    # Process training telemetry (Jan 1 - Mar 11)
    train_files = [
        'telemetry_2026-01-01_2026-01-10.parquet',
        'telemetry_2026-01-11_2026-01-20.parquet',
        'telemetry_2026-01-21_2026-01-31.parquet',
        'telemetry_2026-02-01_2026-02-10.parquet',
        'telemetry_2026-02-11_2026-02-20.parquet',
        'telemetry_2026-02-21_2026-02-28.parquet',
        'telemetry_2026-03-01_2026-03-11.parquet',
    ]

    print("Processing training telemetry...")
    train_chunks = []
    for fname in train_files:
        chunk = process_telemetry_file(data_dir / fname, train_mode=True)
        train_chunks.append(chunk)
        gc.collect()

    train_features = pd.concat(train_chunks, ignore_index=True)
    print(f"  Train features shape: {train_features.shape}")

    # Process test telemetry (Mar 12-20)
    test_files = ['telemetry_2026-03-12_2026-03-20.parquet']

    print("Processing test telemetry...")
    test_chunks = []
    for fname in test_files:
        chunk = process_telemetry_file(data_dir / fname, train_mode=False)
        test_chunks.append(chunk)
        gc.collect()

    test_features = pd.concat(test_chunks, ignore_index=True)
    print(f"  Test features shape: {test_features.shape}")

    # Add refuel and tank level features
    print("Adding refuel features...")
    train_features = add_refuel_features(train_features, summary)
    test_features = add_refuel_features(test_features, summary)

    # Merge ground truth targets
    target_cols = ['vehicle', 'date', 'shift', 'acons']
    targets = summary[target_cols].copy()
    train_features = train_features.merge(targets, on=['vehicle', 'date', 'shift'], how='left')

    # Add temporal features
    print("Adding temporal features...")
    train_features = add_temporal_features(train_features)
    test_features = add_temporal_features(test_features)

    # Add vehicle aggregates
    print("Adding vehicle aggregates...")
    train_features, test_features = add_vehicle_aggregates(train_features, test_features)

    # Add lag features
    print("Adding lag features...")
    train_features = add_lag_features(train_features, target_col='acons')
    test_features = add_lag_features(test_features, target_col='veh_shift_mean')

    # Save
    train_features.to_csv(data_dir / 'train_enhanced.csv', index=False)
    test_features.to_csv(data_dir / 'test_enhanced.csv', index=False)

    print(f"\n✓ Enhanced train: {train_features.shape}")
    print(f"✓ Enhanced test: {test_features.shape}")
    print(f"\nTrain columns: {len(train_features.columns)}")
    print("\nSample features:")
    print(train_features.head(2))

    # Check target distribution
    print(f"\nTarget 'acons' stats:")
    print(train_features['acons'].describe())
    print(f"Missing targets: {train_features['acons'].isna().sum()}")


if __name__ == '__main__':
    main()
