from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.cycle_segment import aggregate_cycle_features
from utils.data_loader import discover_data_files, load_telemetry, split_train_test_files
from utils.preprocessing import DPR_TARGET_COLS, clean_telemetry


FEATURE_EXCLUDE = {
    "vehicle",
    "mine_anon",
    "date_key",
    "fuel_volume",
    "prod_hr_dpr",
    "idle_hr_dpr",
    "maint_hr_dpr",
    "bd_hr_dpr",
    "hmr_dpr",
    "km_dpr",
    "tonnage",
    "total_trip",
}


def _required_columns(include_targets: bool) -> List[str]:
    cols = [
        "vehicle",
        "mine_anon",
        "ts",
        "date_dpr",
        "latitude",
        "longitude",
        "altitude",
        "speed",
        "ignition",
        "analog_input_1",
        "satellites",
        "gnss_pdop",
        "gnss_hdop",
        "disthav",
        "cumdist",
        "angle",
        "battery_level",
        "battery_current",
        "battery_voltage",
        "external_voltage",
        "axis_x",
        "axis_y",
        "axis_z",
        "gsm_signal",
        "gsm_operator",
    ]
    if include_targets:
        cols.extend(["fuel_volume", "prod_hr_dpr", "idle_hr_dpr", "km_dpr", "tonnage", "total_trip"])
    return cols


def _safe_pct(s: pd.Series, q: float) -> float:
    if s.dropna().empty:
        return 0.0
    return float(np.nanpercentile(s.to_numpy(), q))


def _daily_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values(["vehicle", "date_key", "ts"])

    d["is_moving"] = (d["speed"] > 2).astype(np.int8)
    d["is_idle"] = ((d["ignition"] == 1) & (d["speed"] <= 2)).astype(np.int8)

    d["alt_diff"] = d.groupby(["vehicle", "date_key"], observed=True)["altitude"].diff().fillna(0)
    d["positive_climb_inc"] = d["alt_diff"].clip(lower=0)
    d["negative_descent_inc"] = (-d["alt_diff"]).clip(lower=0)

    prev_speed = d.groupby(["vehicle", "date_key"], observed=True)["speed"].shift(1).fillna(0)
    d["stop_event"] = ((prev_speed > 2) & (d["speed"] <= 2) & (d["ignition"] == 1)).astype(np.int16)

    if "analog_input_1" in d.columns:
        prev_analog = d.groupby(["vehicle", "date_key"], observed=True)["analog_input_1"].shift(1)
        d["dump_event"] = ((d["analog_input_1"] > 2.5) & ((prev_analog <= 2.5) | (prev_analog.isna()))).astype(np.int16)
    else:
        d["dump_event"] = 0

    d["stopped"] = ((d["ignition"] == 1) & (d["speed"] <= 2)).astype(np.int8)
    d["stop_seg"] = d.groupby(["vehicle", "date_key"], observed=True)["stopped"].transform(
        lambda s: (s != s.shift()).cumsum()
    )

    stop_durations = (
        d[d["stopped"] == 1]
        .groupby(["vehicle", "date_key", "stop_seg"], observed=True)["time_gap"]
        .sum()
        .reset_index(name="stop_duration_s")
    )
    stop_stats = stop_durations.groupby(["vehicle", "date_key"], observed=True).agg(
        avg_stop_duration_s=("stop_duration_s", "mean"),
        total_stop_time_s=("stop_duration_s", "sum"),
    )

    agg = d.groupby(["vehicle", "mine_anon", "date_key"], observed=True).agg(
        ts_first=("ts", "min"),
        ts_last=("ts", "max"),
        total_ignition_on_hours=("time_gap", lambda s: float((s[d.loc[s.index, "ignition"] == 1].sum()) / 3600.0)),
        total_moving_hours=("time_gap", lambda s: float((s[d.loc[s.index, "is_moving"] == 1].sum()) / 3600.0)),
        total_idle_hours=("time_gap", lambda s: float((s[d.loc[s.index, "is_idle"] == 1].sum()) / 3600.0)),
        daily_km=("disthav", lambda s: float(np.nansum(s) / 1000.0)),
        net_lift=("altitude", lambda s: float(s.iloc[-1] - s.iloc[0]) if len(s) > 1 else 0.0),
        gross_elevation_change=("alt_diff", lambda s: float(np.nansum(np.abs(s)))),
        altitude_std=("altitude", "std"),
        altitude_mean=("altitude", "mean"),
        altitude_max=("altitude", "max"),
        altitude_min=("altitude", "min"),
        positive_climb=("positive_climb_inc", "sum"),
        negative_descent=("negative_descent_inc", "sum"),
        speed_mean=("speed", lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0),
        speed_std=("speed", "std"),
        speed_max=("speed", "max"),
        speed_p50=("speed", lambda s: _safe_pct(s, 50)),
        speed_p75=("speed", lambda s: _safe_pct(s, 75)),
        speed_p90=("speed", lambda s: _safe_pct(s, 90)),
        stop_count=("stop_event", "sum"),
        dump_events=("dump_event", "sum"),
        avg_gnss_hdop=("gnss_hdop", "mean"),
        avg_satellites=("satellites", "mean"),
        battery_level_mean=("battery_level", "mean"),
        axis_x_std=("axis_x", "std"),
        axis_y_std=("axis_y", "std"),
        axis_z_std=("axis_z", "std"),
        coverage_rows=("ts", "count"),
    ).reset_index()

    agg["data_coverage_hours"] = (agg["ts_last"] - agg["ts_first"]).dt.total_seconds().clip(lower=0) / 3600.0
    agg["idle_fraction"] = agg["total_idle_hours"] / (agg["total_ignition_on_hours"] + 1e-6)
    agg["climb_to_descent_ratio"] = agg["positive_climb"] / (agg["negative_descent"] + 1.0)
    agg["stop_density"] = agg["stop_count"] / (agg["daily_km"] + 1e-6)
    agg["fuel_efficiency_proxy"] = agg["daily_km"] / (agg["total_ignition_on_hours"] + 1e-6)

    agg = agg.merge(stop_stats.reset_index(), on=["vehicle", "date_key"], how="left")
    agg[["avg_stop_duration_s", "total_stop_time_s"]] = agg[["avg_stop_duration_s", "total_stop_time_s"]].fillna(0)

    # Time features from operational day
    agg["day_of_week"] = pd.to_datetime(agg["date_key"]).dt.dayofweek
    agg["week_number"] = pd.to_datetime(agg["date_key"]).dt.isocalendar().week.astype(int)
    agg["is_weekend"] = (agg["day_of_week"] >= 5).astype(np.int8)

    return agg


def _extract_targets(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["vehicle", "date_key"] + [c for c in ["fuel_volume", "prod_hr_dpr", "idle_hr_dpr", "km_dpr", "tonnage", "total_trip"] if c in df.columns]
    t = df[cols].copy()

    grp = t.groupby(["vehicle", "date_key"], observed=True)
    out = grp.agg(
        fuel_volume=("fuel_volume", "first"),
        prod_hr_dpr=("prod_hr_dpr", "first"),
        idle_hr_dpr=("idle_hr_dpr", "first"),
        km_dpr=("km_dpr", "first"),
        tonnage=("tonnage", "first"),
        total_trip=("total_trip", "first"),
    ).reset_index()
    return out


def _add_behavior_fraction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_at_low_speed"] = np.clip(out["speed_p50"] / 10.0, 0, 1)
    out["time_at_high_speed"] = np.clip((out["speed_p90"] - 30.0) / 30.0, 0, 1)
    return out


def _vehicle_history_features(train_daily: pd.DataFrame, test_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train_daily.sort_values(["vehicle", "date_key"]).copy()
    test = test_daily.copy()

    vh = train.groupby("vehicle", observed=True).agg(
        vehicle_hist_mean_fuel=("fuel_volume", "mean"),
        vehicle_hist_std_fuel=("fuel_volume", "std"),
        vehicle_hist_mean_km=("daily_km", "mean"),
        vehicle_hist_mean_ign_h=("total_ignition_on_hours", "mean"),
    )
    vh["vehicle_efficiency_rank"] = vh["vehicle_hist_mean_fuel"].rank(method="dense")

    train = train.merge(vh.reset_index(), on="vehicle", how="left")
    test = test.merge(vh.reset_index(), on="vehicle", how="left")

    global_ratio = (train["fuel_volume"].sum() / (train["total_ignition_on_hours"].sum() + 1e-6))
    test["fallback_fuel_from_ignition"] = test["total_ignition_on_hours"] * global_ratio

    return train, test


def _compute_bad_weather_feature(train_daily: pd.DataFrame, test_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_daily.copy()
    te = test_daily.copy()

    per_day_speed = tr.groupby("date_key", observed=True)["speed_mean"].mean()
    threshold = np.nanpercentile(per_day_speed.to_numpy(), 25) if len(per_day_speed) else 0
    bad_days = set(per_day_speed[per_day_speed <= threshold].index)

    tr["bad_weather_day"] = tr["date_key"].isin(bad_days).astype(np.int8)
    te["bad_weather_day"] = te["date_key"].isin(bad_days).astype(np.int8)
    return tr, te


def _enrich_with_spatial_flags(df: pd.DataFrame, gpkg_by_mine: Dict[str, Path]) -> pd.DataFrame:
    try:
        from utils.spatial_utils import add_zone_flags, load_layers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Spatial features require optional dependencies (e.g. geopandas). "
            "Install geopandas or run with enable_spatial=False."
        ) from exc

    chunks = []
    for mine, g in df.groupby("mine_anon", observed=True):
        gpkg = gpkg_by_mine.get(str(mine))
        if gpkg is None:
            gg = g.copy()
            gg["in_dump_zone"] = 0
            gg["in_load_zone"] = 0
            gg["on_haul_road"] = 0
            chunks.append(gg)
            continue
        layers = load_layers(gpkg)
        gg = add_zone_flags(g.copy(), layers)
        chunks.append(gg)
    return pd.concat(chunks, ignore_index=True)


def _mine_gpkg_mapping(gpkg_files: List[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for g in gpkg_files:
        name = g.name.lower()
        if "mine_001" in name or "mine001" in name:
            out["mine001"] = g
        elif "mine_002" in name or "mine002" in name:
            out["mine002"] = g
    return out


def _build_daily(files: List[Path], include_targets: bool, gpkg_map: Dict[str, Path], enable_spatial: bool) -> pd.DataFrame:
    daily_frames = []
    target_frames = []
    for i, f in enumerate(files, start=1):
        print(f"[feature_eng] Processing {i}/{len(files)}: {f.name}")
        df = load_telemetry(f, columns=_required_columns(include_targets=include_targets))
        df = clean_telemetry(df)

        if enable_spatial:
            df = _enrich_with_spatial_flags(df, gpkg_map)
        else:
            df["in_dump_zone"] = 0
            df["in_load_zone"] = 0
            df["on_haul_road"] = 0

        base_daily = _daily_aggregate(df)
        zone_daily = (
            df.groupby(["vehicle", "mine_anon", "date_key"], observed=True)
            .agg(
                time_in_dump_zone_s=("time_gap", lambda s: float(s[df.loc[s.index, "in_dump_zone"] == 1].sum())),
                time_in_load_zone_s=("time_gap", lambda s: float(s[df.loc[s.index, "in_load_zone"] == 1].sum())),
                time_on_haul_road_s=("time_gap", lambda s: float(s[df.loc[s.index, "on_haul_road"] == 1].sum())),
                shift_composition=("date_key", "count"),
            )
            .reset_index()
        )

        cycles_daily = aggregate_cycle_features(df[["vehicle", "date_key", "ts", "speed", "ignition", "analog_input_1", "disthav", "altitude", "time_gap"]])

        merged = base_daily.merge(zone_daily, on=["vehicle", "mine_anon", "date_key"], how="left")
        merged = merged.merge(cycles_daily, on=["vehicle", "date_key"], how="left")
        merged = _add_behavior_fraction_features(merged)

        for c in [
            "cycles_count",
            "mean_cycle_duration_min",
            "mean_cycle_distance_km",
            "mean_net_lift",
            "total_cycle_distance_km",
        ]:
            if c in merged.columns:
                merged[c] = merged[c].fillna(0)

        daily_frames.append(merged)

        if include_targets and "fuel_volume" in df.columns:
            target_frames.append(_extract_targets(df))

    out_daily = pd.concat(daily_frames, ignore_index=True)
    out_daily = out_daily.sort_values(["vehicle", "date_key"]).drop_duplicates(["vehicle", "date_key"], keep="last")

    if include_targets:
        target_daily = pd.concat(target_frames, ignore_index=True)
        target_daily = target_daily.sort_values(["vehicle", "date_key"]).drop_duplicates(["vehicle", "date_key"], keep="last")
        out_daily = out_daily.merge(target_daily, on=["vehicle", "date_key"], how="left")

    return out_daily


def build_feature_matrices(data_dir: Path, cache_dir: Path, enable_spatial: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = discover_data_files(data_dir)
    train_files, test_files = split_train_test_files(files.telemetry_files)
    gpkg_map = _mine_gpkg_mapping(files.gpkg_files)

    train_daily = _build_daily(train_files, include_targets=True, gpkg_map=gpkg_map, enable_spatial=enable_spatial)
    test_daily = _build_daily(test_files, include_targets=False, gpkg_map=gpkg_map, enable_spatial=enable_spatial)

    train_daily, test_daily = _vehicle_history_features(train_daily, test_daily)
    train_daily, test_daily = _compute_bad_weather_feature(train_daily, test_daily)

    train_daily = train_daily.reset_index(drop=True)
    test_daily = test_daily.reset_index(drop=True)

    train_daily.to_feather(cache_dir / "train_features.feather")
    test_daily.to_feather(cache_dir / "test_features.feather")

    print(f"[feature_eng] train_daily shape={train_daily.shape}")
    print(f"[feature_eng] test_daily shape={test_daily.shape}")
    return train_daily, test_daily


def get_feature_columns(train_daily: pd.DataFrame) -> List[str]:
    features = []
    for c in train_daily.columns:
        if c in FEATURE_EXCLUDE:
            continue
        if train_daily[c].dtype == "object":
            continue
        if is_datetime64_any_dtype(train_daily[c]):
            continue
        features.append(c)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily train/test feature matrices")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="data")
    parser.add_argument("--enable_spatial", action="store_true")
    args = parser.parse_args()

    train_daily, test_daily = build_feature_matrices(Path(args.data_dir), Path(args.cache_dir), enable_spatial=args.enable_spatial)
    print(train_daily.head(3))
    print(train_daily.isnull().sum().sort_values(ascending=False).head(20))
    print(test_daily.head(3))


if __name__ == "__main__":
    main()
