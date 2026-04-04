#!/usr/bin/env python3
"""Clean shift-wise feature engineering (NO LEAKAGE).

Builds shift-level aggregates from raw telemetry, plus safe auxiliary features that
exist for both train and test:
- refuel aggregates from RFID refuel events (with optional ±N minute alignment)
- optional spatial zone flags from mine GeoPackages (OB dump vs ROM stock, etc.)

Outputs:
- outputs/shiftwise_features/train_clean_shiftwise.csv
- outputs/shiftwise_features/test_clean_shiftwise.csv
"""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.data_loader import discover_data_files, split_train_test_files

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]


def assign_shift(ts: pd.Timestamp) -> str:
    """Assign shift: C (22:00-05:59), A (06:00-13:59), B (14:00-21:59)."""
    hour = ts.hour
    if 6 <= hour < 14:
        return "A"
    if 14 <= hour < 22:
        return "B"
    return "C"


def assign_working_date(ts: pd.Timestamp, shift: str):
    """Night shift C belongs to previous day for records before 06:00."""
    if shift == "C" and ts.hour < 6:
        return (ts - pd.Timedelta(days=1)).date()
    return ts.date()


def detect_dump_events(analog_series: pd.Series, threshold: float = 2.5) -> int:
    """Detect dump events from analog_input_1 rising edge over threshold."""
    if analog_series is None or analog_series.isna().all():
        return 0
    prev = analog_series.shift(1)
    crossings = (analog_series > threshold) & ((prev <= threshold) | prev.isna())
    return int(crossings.sum())


def _ensure_local_ts(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts, errors="coerce")
    if out.dt.tz is None:
        out = out.dt.tz_localize("UTC")
    return out.dt.tz_convert("Asia/Kolkata")


def _mine_gpkg_map(data_dir: Path) -> dict[str, Path]:
    gpkg_files = sorted(data_dir.glob("*.gpkg"))
    out: dict[str, Path] = {}
    for g in gpkg_files:
        name = g.name.lower()
        if "mine_001" in name or "mine001" in name:
            out["mine001"] = g
        elif "mine_002" in name or "mine002" in name:
            out["mine002"] = g
    return out


def _load_refuels(data_dir: Path) -> pd.DataFrame | None:
    candidates = sorted(data_dir.glob("rfid_refuels_*.parquet"))
    if not candidates:
        return None
    return pd.read_parquet(candidates[-1])


def _aggregate_refuels_by_shift(refuels: pd.DataFrame, minutes_shift: int) -> pd.DataFrame:
    r = refuels.copy()
    if "fleet_type" in r.columns:
        r = r[r["fleet_type"] == "Dumper"].copy()

    r["ts_local"] = _ensure_local_ts(r["ts"])
    r["ts_local_shifted"] = r["ts_local"] + pd.Timedelta(minutes=minutes_shift)
    r["shift"] = r["ts_local_shifted"].apply(assign_shift)
    r["date"] = r.apply(lambda row: assign_working_date(row["ts_local_shifted"], row["shift"]), axis=1)

    r = r.sort_values(["vehicle", "ts_local"]).copy()
    r["prev_refuel_ts_local"] = r.groupby("vehicle", observed=True)["ts_local"].shift(1)
    r["hrs_since_prev_refuel"] = (r["ts_local"] - r["prev_refuel_ts_local"]).dt.total_seconds() / 3600.0

    agg = (
        r.groupby(["vehicle", "date", "shift"], observed=True)
        .agg(
            refuel_liters=("litres", "sum"),
            refuel_count=("litres", "count"),
            refuel_liters_max=("litres", "max"),
            hrs_since_prev_refuel_min=("hrs_since_prev_refuel", "min"),
            hrs_since_prev_refuel_mean=("hrs_since_prev_refuel", "mean"),
        )
        .reset_index()
        .fillna(0)
    )

    suffix = f"shift_{minutes_shift:+d}m".replace("+", "p").replace("-", "m")
    rename = {
        "refuel_liters": f"refuel_liters_{suffix}",
        "refuel_count": f"refuel_count_{suffix}",
        "refuel_liters_max": f"refuel_liters_max_{suffix}",
        "hrs_since_prev_refuel_min": f"hrs_since_prev_refuel_min_{suffix}",
        "hrs_since_prev_refuel_mean": f"hrs_since_prev_refuel_mean_{suffix}",
    }
    return agg.rename(columns=rename)


def aggregate_shift_features(group: pd.DataFrame) -> pd.Series:
    g = group.sort_values("ts").copy()

    n = len(g)
    if n == 0:
        return pd.Series(dtype=float)

    g["time_gap"] = g["ts"].diff().dt.total_seconds().fillna(0).clip(0, 300)

    ignition_on = g["ignition"] == 1
    moving = g["speed"] > 2
    idle = ignition_on & ~moving

    ignition_time_h = g.loc[ignition_on, "time_gap"].sum() / 3600.0
    moving_time_h = g.loc[moving, "time_gap"].sum() / 3600.0
    idle_time_h = g.loc[idle, "time_gap"].sum() / 3600.0

    shift_km = g["disthav"].sum() / 1000.0

    alt_mean = float(g["altitude"].mean())
    alt_std = float(g["altitude"].std())
    alt_max = float(g["altitude"].max())
    alt_min = float(g["altitude"].min())
    alt_range = alt_max - alt_min

    g["alt_diff"] = g["altitude"].diff().fillna(0)
    net_lift = float(g["altitude"].iloc[-1] - g["altitude"].iloc[0]) if len(g) > 1 else 0.0
    total_climb = float(g["alt_diff"].clip(lower=0).sum())
    total_descent = float((-g["alt_diff"]).clip(lower=0).sum())

    speed_data = g.loc[g["speed"] > 0, "speed"]
    speed_mean = float(speed_data.mean()) if len(speed_data) > 0 else 0.0
    speed_std = float(speed_data.std()) if len(speed_data) > 0 else 0.0
    speed_max = float(g["speed"].max())
    speed_p50 = float(speed_data.quantile(0.5)) if len(speed_data) > 0 else 0.0
    speed_p75 = float(speed_data.quantile(0.75)) if len(speed_data) > 0 else 0.0
    speed_p90 = float(speed_data.quantile(0.9)) if len(speed_data) > 0 else 0.0

    prev_speed = g["speed"].shift(1).fillna(0)
    stop_events = int(((prev_speed > 2) & (g["speed"] <= 2) & ignition_on).sum())

    dump_count = detect_dump_events(g["analog_input_1"]) if "analog_input_1" in g.columns else 0
    dump_signal_mean = float(g["analog_input_1"].mean()) if "analog_input_1" in g.columns else 0.0
    dump_signal_std = float(g["analog_input_1"].std()) if "analog_input_1" in g.columns else 0.0
    dump_signal_max = float(g["analog_input_1"].max()) if "analog_input_1" in g.columns else 0.0
    dump_high_time_h = (
        float(g.loc[g["analog_input_1"] > 2.5, "time_gap"].sum()) / 3600.0 if "analog_input_1" in g.columns else 0.0
    )

    frac_in_dump_zone = float(g["in_dump_zone"].mean()) if "in_dump_zone" in g.columns else 0.0
    frac_in_ob_dump_zone = float(g["in_ob_dump_zone"].mean()) if "in_ob_dump_zone" in g.columns else 0.0
    frac_in_rom_stock_zone = float(g["in_rom_stock_zone"].mean()) if "in_rom_stock_zone" in g.columns else 0.0
    frac_in_load_zone = float(g["in_load_zone"].mean()) if "in_load_zone" in g.columns else 0.0
    frac_on_haul_road = float(g["on_haul_road"].mean()) if "on_haul_road" in g.columns else 0.0

    if "dump_edge" in g.columns:
        dump_in_ob = int(((g["dump_edge"] == 1) & (g.get("in_ob_dump_zone", 0) == 1)).sum())
        dump_in_rom = int(((g["dump_edge"] == 1) & (g.get("in_rom_stock_zone", 0) == 1)).sum())
    else:
        dump_in_ob = 0
        dump_in_rom = 0
    dump_outside = max(int(dump_count - dump_in_ob - dump_in_rom), 0)

    total_time_s = float(g["time_gap"].sum())
    top_cell_dwell_frac = 0.0
    unique_cells = 0
    if "x_utm" in g.columns and "y_utm" in g.columns and total_time_s > 0:
        grid = 50.0
        gx = np.floor(g["x_utm"] / grid).astype("int64")
        gy = np.floor(g["y_utm"] / grid).astype("int64")
        cell = gx * 1_000_000 + gy
        dwell = g.groupby(cell, observed=True)["time_gap"].sum()
        unique_cells = int(dwell.shape[0])
        top_cell_dwell_frac = float(dwell.max() / (total_time_s + 1e-6)) if not dwell.empty else 0.0

    idle_fraction = float(idle_time_h / (ignition_time_h + 1e-6))
    stop_density = float(stop_events / (shift_km + 1e-6))
    km_per_hour = float(shift_km / (ignition_time_h + 1e-6))
    climb_descent_ratio = float(total_climb / (total_descent + 1.0))

    avg_satellites = float(g["satellites"].mean()) if "satellites" in g.columns else 0.0
    avg_hdop = float(g["gnss_hdop"].mean()) if "gnss_hdop" in g.columns else 0.0

    axis_x_std = float(g["axis_x"].std()) if "axis_x" in g.columns else 0.0
    axis_y_std = float(g["axis_y"].std()) if "axis_y" in g.columns else 0.0
    axis_z_std = float(g["axis_z"].std()) if "axis_z" in g.columns else 0.0
    vibration_magnitude = float(np.sqrt(axis_x_std**2 + axis_y_std**2 + axis_z_std**2))

    battery_mean = float(g["battery_level"].mean()) if "battery_level" in g.columns else 0.0
    angle_std = float(g["angle"].std()) if "angle" in g.columns else 0.0
    speed_cv = float(speed_std / (speed_mean + 1e-6)) if speed_mean > 0 else 0.0

    km_per_climb = float(shift_km / (total_climb + 1.0))
    distance_per_dump = float(shift_km / (dump_count + 1.0))

    hour_start = int(g["ts"].iloc[0].hour)
    hour_end = int(g["ts"].iloc[-1].hour)

    return pd.Series(
        {
            "ignition_on_hours": ignition_time_h,
            "moving_hours": moving_time_h,
            "idle_hours": idle_time_h,
            "shift_km": shift_km,
            "altitude_mean": alt_mean,
            "altitude_std": alt_std,
            "altitude_max": alt_max,
            "altitude_min": alt_min,
            "altitude_range": alt_range,
            "net_lift": net_lift,
            "total_climb_m": total_climb,
            "total_descent_m": total_descent,
            "speed_mean": speed_mean,
            "speed_std": speed_std,
            "speed_max": speed_max,
            "speed_p50": speed_p50,
            "speed_p75": speed_p75,
            "speed_p90": speed_p90,
            "stop_count": stop_events,
            "dump_event_count": dump_count,
            "dump_signal_mean": dump_signal_mean,
            "dump_signal_std": dump_signal_std,
            "dump_signal_max": dump_signal_max,
            "dump_high_time_h": dump_high_time_h,
            "frac_in_dump_zone": frac_in_dump_zone,
            "frac_in_ob_dump_zone": frac_in_ob_dump_zone,
            "frac_in_rom_stock_zone": frac_in_rom_stock_zone,
            "frac_in_load_zone": frac_in_load_zone,
            "frac_on_haul_road": frac_on_haul_road,
            "dump_events_in_ob_dump": dump_in_ob,
            "dump_events_in_rom_stock": dump_in_rom,
            "dump_events_outside_zones": dump_outside,
            "hotspot_top_cell_dwell_frac": top_cell_dwell_frac,
            "hotspot_unique_cells": unique_cells,
            "n_pings": int(n),
            "idle_fraction": idle_fraction,
            "stop_density": stop_density,
            "km_per_hour": km_per_hour,
            "climb_descent_ratio": climb_descent_ratio,
            "avg_satellites": avg_satellites,
            "avg_hdop": avg_hdop,
            "vibration_magnitude": vibration_magnitude,
            "battery_mean": battery_mean,
            "angle_std": angle_std,
            "speed_cv": speed_cv,
            "km_per_climb_m": km_per_climb,
            "distance_per_dump_km": distance_per_dump,
            "hour_start": hour_start,
            "hour_end": hour_end,
        }
    )


def add_vehicle_aggregates(train_df: pd.DataFrame, test_df: pd.DataFrame):
    veh_stats = (
        train_df.groupby("vehicle", observed=True)
        .agg(
            veh_mean_km=("shift_km", "mean"),
            veh_mean_ignition_h=("ignition_on_hours", "mean"),
            veh_mean_dumps=("dump_event_count", "mean"),
            veh_mean_idle_frac=("idle_fraction", "mean"),
            veh_mean_speed=("speed_mean", "mean"),
            veh_std_km=("shift_km", "std"),
        )
        .reset_index()
    )

    if "acons" in train_df.columns:
        shift_stats = (
            train_df.groupby(["vehicle", "shift"], observed=True)
            .agg(
                veh_shift_mean_cons=("acons", "mean"),
                veh_shift_std_cons=("acons", "std"),
            )
            .reset_index()
        )
    else:
        shift_stats = pd.DataFrame()

    train_merged = train_df.merge(veh_stats, on="vehicle", how="left")
    test_merged = test_df.merge(veh_stats, on="vehicle", how="left")

    if not shift_stats.empty:
        train_merged = train_merged.merge(shift_stats, on=["vehicle", "shift"], how="left")
        test_merged = test_merged.merge(shift_stats, on=["vehicle", "shift"], how="left")

    return train_merged, test_merged


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["day_of_week"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["week_number"] = out["date"].dt.isocalendar().week.astype(int)
    out["day_of_month"] = out["date"].dt.day.astype(int)

    shift_map = {"C": 0, "A": 1, "B": 2}
    out["shift_enc"] = out["shift"].map(shift_map).astype(int)
    return out


def process_telemetry_file(
    fpath: Path,
    *,
    enable_spatial: bool = False,
    layers_by_mine: dict[str, object] | None = None,
) -> pd.DataFrame:
    print(f"  Loading {fpath.name}...")

    required_cols = [
        "vehicle",
        "mine_anon",
        "ts",
        "received_ts",
        "latitude",
        "longitude",
        "altitude",
        "speed",
        "ignition",
        "disthav",
        "analog_input_1",
        "satellites",
        "gnss_hdop",
        "angle",
        "battery_level",
        "axis_x",
        "axis_y",
        "axis_z",
    ]

    try:
        df = pd.read_parquet(fpath, columns=required_cols)
    except Exception:
        df = pd.read_parquet(fpath)
        df = df[[c for c in required_cols if c in df.columns]].copy()

    mandatory = {"vehicle", "ts", "speed", "disthav", "ignition", "altitude"}
    if not mandatory.issubset(set(df.columns)):
        missing = sorted(mandatory - set(df.columns))
        print(f"  Skipping {fpath.name} (missing {missing})")
        return pd.DataFrame(columns=["vehicle", "date", "shift"])

    df["ts"] = _ensure_local_ts(df["ts"])
    if "received_ts" in df.columns:
        df["received_ts"] = _ensure_local_ts(df["received_ts"])
        df["rx_delay_s"] = (df["received_ts"] - df["ts"]).dt.total_seconds().clip(lower=0, upper=3600)
    else:
        df["rx_delay_s"] = 0.0

    # Vectorized shift and working-date assignment
    hour = df["ts"].dt.hour
    df["shift"] = np.where((hour >= 6) & (hour < 14), "A", np.where((hour >= 14) & (hour < 22), "B", "C"))
    day0 = df["ts"].dt.tz_localize(None).dt.normalize()
    is_night_carry = (df["shift"] == "C") & (hour < 6)
    work_day = day0 - pd.to_timedelta(is_night_carry.astype("int8"), unit="D")
    df["date"] = work_day.dt.date

    df = df[df["speed"] <= 80].copy()
    df["speed"] = df["speed"].clip(lower=0, upper=60)

    if "analog_input_1" in df.columns:
        prev = df.groupby(["vehicle", "date", "shift"], observed=True)["analog_input_1"].shift(1)
        df["dump_edge"] = ((df["analog_input_1"] > 2.5) & ((prev <= 2.5) | prev.isna())).astype("int8")
    else:
        df["dump_edge"] = 0

    if enable_spatial:
        try:
            from utils.spatial_utils import add_utm_coordinates, add_zone_flags

            if "latitude" in df.columns and "longitude" in df.columns:
                df = add_utm_coordinates(df)

            for c in ["in_dump_zone", "in_ob_dump_zone", "in_rom_stock_zone", "in_load_zone", "on_haul_road"]:
                if c not in df.columns:
                    df[c] = 0

            if layers_by_mine and "mine_anon" in df.columns:
                parts = []
                for mine, part in df.groupby("mine_anon", observed=True):
                    layers = layers_by_mine.get(mine) or layers_by_mine.get(str(mine).lower())
                    if layers is not None and len(part) > 0:
                        part = add_zone_flags(part, layers)  # type: ignore[arg-type]
                    parts.append(part)
                df = pd.concat(parts, ignore_index=True)
        except Exception as e:
            print(f"  Spatial disabled for {fpath.name} due to error: {e}")

    shift_feats = df.groupby(["vehicle", "date", "shift"], observed=True).apply(aggregate_shift_features).reset_index()

    rx = (
        df.groupby(["vehicle", "date", "shift"], observed=True)["rx_delay_s"]
        .agg(
            rx_delay_mean_s="mean",
            rx_delay_p90_s=lambda s: float(np.nanpercentile(s.to_numpy(), 90)) if len(s) else 0.0,
        )
        .reset_index()
    )
    return shift_feats.merge(rx, on=["vehicle", "date", "shift"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLEAN shift-wise features (telemetry + safe aux)")
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    parser.add_argument("--out_dir", default=str(ROOT / "outputs" / "shiftwise_features"))
    parser.add_argument("--enable_spatial", action="store_true")
    parser.add_argument("--refuel_shift_minutes", type=int, default=2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading summary data for targets...")
    summary = pd.concat(
        [
            pd.read_csv(data_dir / "smry_jan_train_ordered.csv"),
            pd.read_csv(data_dir / "smry_feb_train_ordered.csv"),
            pd.read_csv(data_dir / "smry_mar_train_ordered.csv"),
        ],
        ignore_index=True,
    )
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce").dt.date
    print(f"  Summary rows: {len(summary)}")

    files = discover_data_files(data_dir)
    train_paths, test_paths = split_train_test_files(files.telemetry_files)

    print(f"Train telemetry windows: {len(train_paths)}")
    for p in train_paths:
        print(f" - {p.name}")
    print(f"Test telemetry windows: {len(test_paths)}")
    for p in test_paths:
        print(f" - {p.name}")

    layers_by_mine: dict[str, object] = {}
    if args.enable_spatial:
        gpkg_map = _mine_gpkg_map(data_dir)
        if gpkg_map:
            from utils.spatial_utils import load_layers

            for mine, gpkg in gpkg_map.items():
                print(f"Loading gpkg layers for {mine}: {gpkg.name}")
                layers_by_mine[mine] = load_layers(gpkg)
        else:
            print("No gpkg files found; spatial features will be empty.")

    refuels = _load_refuels(data_dir)
    ref_aggs: list[pd.DataFrame] = []
    if refuels is not None:
        print(f"Loaded refuels: {len(refuels)}")
        ref_aggs.append(_aggregate_refuels_by_shift(refuels, 0))
        if args.refuel_shift_minutes:
            ref_aggs.append(_aggregate_refuels_by_shift(refuels, args.refuel_shift_minutes))
            ref_aggs.append(_aggregate_refuels_by_shift(refuels, -args.refuel_shift_minutes))

    print("Processing training telemetry...")
    train_chunks: list[pd.DataFrame] = []
    for fpath in train_paths:
        chunk = process_telemetry_file(
            fpath,
            enable_spatial=args.enable_spatial,
            layers_by_mine=layers_by_mine if args.enable_spatial else None,
        )
        train_chunks.append(chunk)
        gc.collect()

    train_features = pd.concat(train_chunks, ignore_index=True)
    print(f"  Train features shape: {train_features.shape}")

    print("Processing test telemetry...")
    test_chunks: list[pd.DataFrame] = []
    for fpath in test_paths:
        chunk = process_telemetry_file(
            fpath,
            enable_spatial=args.enable_spatial,
            layers_by_mine=layers_by_mine if args.enable_spatial else None,
        )
        test_chunks.append(chunk)
        gc.collect()

    test_features = pd.concat(test_chunks, ignore_index=True)
    print(f"  Test features shape: {test_features.shape}")

    targets = summary[["vehicle", "date", "shift", "acons"]].copy()
    train_features = train_features.merge(targets, on=["vehicle", "date", "shift"], how="left")

    if ref_aggs:
        for agg in ref_aggs:
            train_features = train_features.merge(agg, on=["vehicle", "date", "shift"], how="left")
            test_features = test_features.merge(agg, on=["vehicle", "date", "shift"], how="left")

        ref_cols = [
            c for c in train_features.columns if c.startswith("refuel_") or c.startswith("hrs_since_prev_refuel")
        ]
        train_features[ref_cols] = train_features[ref_cols].fillna(0)
        test_features[ref_cols] = test_features[ref_cols].fillna(0)

    print("Adding temporal features...")
    train_features = add_temporal_features(train_features)
    test_features = add_temporal_features(test_features)

    print("Adding vehicle aggregates...")
    train_features, test_features = add_vehicle_aggregates(train_features, test_features)

    train_out = out_dir / "train_clean_shiftwise.csv"
    test_out = out_dir / "test_clean_shiftwise.csv"
    train_features.to_csv(train_out, index=False)
    test_features.to_csv(test_out, index=False)

    print(f"\n✓ Clean train: {train_features.shape} -> {train_out}")
    print(f"✓ Clean test: {test_features.shape} -> {test_out}")


if __name__ == "__main__":
    main()
