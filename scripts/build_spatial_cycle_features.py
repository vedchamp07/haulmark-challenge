#!/usr/bin/env python3
"""Spatial + Cycle Detection Feature Pipeline.

This script is intentionally self-contained and produces *test-available* features.

Key features:
    - Time fractions in dump/load/haul zones (from mine .gpkg)
    - Excavator proximity as a loading proxy (mean excavator anchors per mine)
    - Dump/load/haul transitions (zone entry counts)
    - Cycle proxies from dump switch (when present) and spatial transitions

Outputs:
    - outputs/spatial_features/train_spatial.csv
    - outputs/spatial_features/test_spatial.csv

Performance notes:
    - Uses cached per-file parts under outputs/spatial_features/parts/ so you can
        interrupt and resume without losing progress.
"""

from __future__ import annotations

import gc
import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "outputs" / "spatial_features"
DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
DEFAULT_PARTS_DIR = DEFAULT_OUTPUT / "parts"
DEFAULT_PARTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILES = {
    "telemetry_2026-01-21_2026-01-31.parquet",
    "telemetry_2026-02-21_2026-02-28.parquet",
    "telemetry_2026-03-12_2026-03-20.parquet",
}

# Buffers for zone membership (in metres, UTM)
DUMP_BUFFER_M = 30.0    # OB dump / ROM stock
LOAD_BUFFER_M = 30.0    # bench / cpu
HAUL_BUFFER_M = 20.0    # haul road
EXC_PROXIMITY_M = 80.0  # excavator proximity = loading zone


# ─────────────────────────────────────────────────────────────────
# Spatial helpers
# ─────────────────────────────────────────────────────────────────

TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)


def to_utm(lat: np.ndarray, lon: np.ndarray):
    x, y = TRANSFORMER.transform(lon, lat)
    return x, y


def load_mine_layers(data_dir: Path) -> dict[str, dict]:
    """Load all gpkg layers for each mine, pre-compute buffered unions."""
    result = {}
    for gpkg in sorted(data_dir.glob("*.gpkg")):
        name = gpkg.name.lower()
        if "mine_001" in name or "mine001" in name:
            mine_key = "mine001"
        elif "mine_002" in name or "mine002" in name:
            mine_key = "mine002"
        else:
            continue

        layers = {}
        for lname in gpd.list_layers(gpkg).name.tolist():
            gdf = gpd.read_file(gpkg, layer=lname)
            if len(gdf) == 0:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:32645")
            elif str(gdf.crs).upper() != "EPSG:32645":
                gdf = gdf.to_crs("EPSG:32645")
            layers[lname] = gdf

        def safe_union(key, buf):
            gdf = layers.get(key)
            if gdf is None:
                return None
            return gdf.geometry.buffer(buf).union_all()

        result[mine_key] = {
            "raw": layers,
            "ob_dump_union":  safe_union("ob_dump",       DUMP_BUFFER_M),
            "stock_union":    safe_union(
                "mineral_stock" if "mineral_stock" in layers else "stock",
                DUMP_BUFFER_M,
            ),
            "bench_union":    safe_union("bench",         LOAD_BUFFER_M),
            "cpu_union":      safe_union("cpu",           LOAD_BUFFER_M),
            "haul_union":     safe_union("haul_road",     HAUL_BUFFER_M),
        }
        print(f"  Loaded spatial layers for {mine_key}: {list(layers.keys())}")

    return result


def add_zone_flags_vectorized(df: pd.DataFrame, mine_layers: dict) -> pd.DataFrame:
    """Add spatial zone flag columns to df (in-place friendly, returns copy)."""
    df = df.copy()
    for col in ["in_dump_zone", "in_ob_dump_zone", "in_rom_stock_zone",
                "in_load_zone", "on_haul_road"]:
        df[col] = 0

    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    # Convert to UTM
    x, y = to_utm(df["latitude"].to_numpy(dtype=float),
                  df["longitude"].to_numpy(dtype=float))
    df["x_utm"] = x
    df["y_utm"] = y

    # Process per mine
    for mine, layers in mine_layers.items():
        mask = df["mine_anon"].astype(str).str.lower() == mine
        n_mine = int(mask.sum())
        if n_mine == 0:
            continue

        x_m = x[mask]
        y_m = y[mask]

        if "speed" in df.columns:
            spd_m = pd.to_numeric(df.loc[mask, "speed"], errors="coerce").fillna(0).to_numpy(dtype=float)
        else:
            spd_m = np.zeros(n_mine, dtype=float)

        def within(union_geom, extra_mask: np.ndarray | None = None):
            if union_geom is None:
                return np.zeros(n_mine, dtype=bool)

            minx, miny, maxx, maxy = union_geom.bounds
            bbox = (x_m >= minx) & (x_m <= maxx) & (y_m >= miny) & (y_m <= maxy)
            if extra_mask is not None:
                bbox = bbox & extra_mask
            if bbox.sum() == 0:
                return np.zeros(n_mine, dtype=bool)

            pts_sub = gpd.GeoSeries(gpd.points_from_xy(x_m[bbox], y_m[bbox]), crs="EPSG:32645")
            inside_sub = pts_sub.within(union_geom).to_numpy()
            out = np.zeros(n_mine, dtype=bool)
            out[bbox] = inside_sub
            return out

        in_ob  = within(layers["ob_dump_union"])
        in_rom = within(layers["stock_union"])
        in_dum = in_ob | in_rom
        in_lod = within(layers["bench_union"]) | within(layers["cpu_union"])
        moving = spd_m > 2.0
        on_hau = within(layers["haul_union"], extra_mask=moving)

        df.loc[mask, "in_dump_zone"]     = in_dum.astype(np.int8)
        df.loc[mask, "in_ob_dump_zone"]  = in_ob.astype(np.int8)
        df.loc[mask, "in_rom_stock_zone"] = in_rom.astype(np.int8)
        df.loc[mask, "in_load_zone"]     = in_lod.astype(np.int8)
        df.loc[mask, "on_haul_road"]     = on_hau.astype(np.int8)

    return df


# ─────────────────────────────────────────────────────────────────
# Excavator helpers (FAST)
# ─────────────────────────────────────────────────────────────────

def compute_excavator_mean_positions(data_dir: Path, telemetry_files: list[Path]) -> dict[str, np.ndarray]:
    """Compute stable excavator anchors per mine (UTM coordinates).

    We intentionally use a single anchor set per mine (not per shift) because:
      - excavators move little relative to the mine scale
      - it avoids expensive per-(date,shift) nested loops
    """
    frames = []
    use_cols = ["vehicle", "mine_anon", "latitude", "longitude", "speed"]
    for f in telemetry_files:
        try:
            df = pd.read_parquet(f, columns=use_cols)
        except Exception:
            df = pd.read_parquet(f)
            df = df[[c for c in use_cols if c in df.columns]].copy()

        if "vehicle" not in df.columns:
            continue
        exc = df[df["vehicle"].astype(str).str.startswith("Exc")].copy()
        if exc.empty:
            continue
        if "speed" in exc.columns:
            exc = exc[pd.to_numeric(exc["speed"], errors="coerce").fillna(0) < 2].copy()
        frames.append(exc[[c for c in ["vehicle", "mine_anon", "latitude", "longitude"] if c in exc.columns]])

    if not frames:
        return {"mine001": np.empty((0, 2), dtype=float), "mine002": np.empty((0, 2), dtype=float)}

    exc_all = pd.concat(frames, ignore_index=True)
    exc_all = exc_all.dropna(subset=["vehicle", "mine_anon", "latitude", "longitude"]).copy()
    if exc_all.empty:
        return {"mine001": np.empty((0, 2), dtype=float), "mine002": np.empty((0, 2), dtype=float)}

    mean = (
        exc_all.groupby(["vehicle", "mine_anon"], observed=True)
        .agg(lat=("latitude", "mean"), lon=("longitude", "mean"))
        .reset_index()
    )
    x, y = to_utm(mean["lat"].to_numpy(dtype=float), mean["lon"].to_numpy(dtype=float))
    mean["x_utm"] = x
    mean["y_utm"] = y

    out: dict[str, np.ndarray] = {}
    for mine_id in ["mine001", "mine002"]:
        sub = mean[mean["mine_anon"].astype(str).str.lower() == mine_id]
        out[mine_id] = sub[["x_utm", "y_utm"]].to_numpy(dtype=float)
    return out


def add_excavator_proximity_fast(df: pd.DataFrame, exc_positions_utm: dict[str, np.ndarray]) -> pd.DataFrame:
    """Vectorized proximity-to-excavator using per-mine anchor points."""
    df = df.copy()
    df["near_excavator"] = 0
    if "x_utm" not in df.columns or "y_utm" not in df.columns:
        return df

    for mine_id, anchors in exc_positions_utm.items():
        if anchors is None or len(anchors) == 0:
            continue
        mask = df["mine_anon"].astype(str).str.lower() == mine_id
        if mask.sum() == 0:
            continue

        x = df.loc[mask, "x_utm"].to_numpy(dtype=float)
        y = df.loc[mask, "y_utm"].to_numpy(dtype=float)
        # distance to nearest anchor
        dx = x[:, None] - anchors[:, 0][None, :]
        dy = y[:, None] - anchors[:, 1][None, :]
        dist2 = dx * dx + dy * dy
        near = (dist2.min(axis=1) < EXC_PROXIMITY_M**2)
        df.loc[mask, "near_excavator"] = near.astype(np.int8)

    return df


# ─────────────────────────────────────────────────────────────────
# Shift-level aggregation
# ─────────────────────────────────────────────────────────────────

def aggregate_shift_spatial(g: pd.DataFrame) -> pd.Series:
    """Aggregate per-shift spatial + cycle features (kept lean for speed)."""
    g = g.sort_values("ts").copy()
    n = len(g)
    if n == 0:
        return pd.Series(dtype=float)

    g["tgap"] = g["ts"].diff().dt.total_seconds().fillna(0).clip(0, 300)
    total_s = g["tgap"].sum()

    # ── Basic motion ──────────────────────────────────────────────
    ign = g["ignition"] == 1
    moving = g["speed"] > 2
    ign_s  = g.loc[ign, "tgap"].sum()
    mov_s  = g.loc[moving, "tgap"].sum()
    idle_s = g.loc[ign & ~moving, "tgap"].sum()
    shift_km = g["disthav"].sum() / 1000

    # ── Altitude ──────────────────────────────────────────────────
    ad = g["altitude"].diff().fillna(0)
    total_climb   = float(ad.clip(lower=0).sum())
    total_descent = float((-ad).clip(lower=0).sum())

    # ── Dump switch ──────────────────────────────────────────────
    dump_count_sig = 0
    if "analog_input_1" in g.columns and not g["analog_input_1"].isna().all():
        a = g["analog_input_1"].fillna(0)
        prev = a.shift(1).fillna(0)
        dump_count_sig = int(((a > 2.5) & (prev <= 2.5)).sum())

    # ── Spatial zone times ────────────────────────────────────────
    time_dump_s    = 0.0
    time_load_s    = 0.0
    time_haul_s    = 0.0
    time_ob_s      = 0.0
    time_rom_s     = 0.0
    dump_transitions = 0
    load_transitions = 0
    haul_transitions = 0

    if "in_dump_zone" in g.columns:
        time_dump_s = float(g.loc[g["in_dump_zone"] == 1, "tgap"].sum())
        time_ob_s   = float(g.loc[g["in_ob_dump_zone"] == 1, "tgap"].sum())
        time_rom_s  = float(g.loc[g["in_rom_stock_zone"] == 1, "tgap"].sum())

        # Spatial dump events: transitions INTO dump zone (0→1)
        prev_dump = g["in_dump_zone"].shift(1).fillna(0)
        dump_transitions = int(((g["in_dump_zone"] == 1) & (prev_dump == 0)).sum())

    if "in_load_zone" in g.columns:
        time_load_s = float(g.loc[g["in_load_zone"] == 1, "tgap"].sum())
        prev_load   = g["in_load_zone"].shift(1).fillna(0)
        load_transitions = int(((g["in_load_zone"] == 1) & (prev_load == 0)).sum())

    if "on_haul_road" in g.columns:
        time_haul_s = float(g.loc[g["on_haul_road"] == 1, "tgap"].sum())
        prev_haul   = g["on_haul_road"].shift(1).fillna(0)
        haul_transitions = int(((g["on_haul_road"] == 1) & (prev_haul == 0)).sum())

    # ── Excavator proximity (loading events) ──────────────────────
    loading_visits   = 0
    loading_dwell_s  = 0.0
    if "near_excavator" in g.columns:
        loading_dwell_s = float(g.loc[g["near_excavator"] == 1, "tgap"].sum())
        prev_exc = g["near_excavator"].shift(1).fillna(0)
        loading_visits = int(((g["near_excavator"] == 1) & (prev_exc == 0)).sum())

    # ── Haul cycle count proxy ────────────────────────────────────
    # Prefer dump switch signal when available (most reliable), else spatial transitions.
    haul_cycles = max(dump_count_sig, dump_transitions, loading_visits)

    # ── Loaded vs empty haul behavior + cycle time proxy ─────────
    loaded_haul_speed_mean = 0.0
    empty_haul_speed_mean = 0.0
    loaded_haul_speed_p90 = 0.0
    empty_haul_speed_p90 = 0.0
    loaded_haul_km = 0.0
    empty_haul_km = 0.0
    load_events = 0
    dump_events = 0
    cycle_time_min_p50 = 0.0

    if "near_excavator" in g.columns and "in_dump_zone" in g.columns:
        near = g["near_excavator"].to_numpy(dtype=int)
        dum = g["in_dump_zone"].to_numpy(dtype=int)
        near_prev = np.r_[0, near[:-1]]
        dum_prev = np.r_[0, dum[:-1]]
        load_evt = (near == 1) & (near_prev == 0)
        dump_evt = (dum == 1) & (dum_prev == 0)
        load_events = int(load_evt.sum())
        dump_events = int(dump_evt.sum())

        score = np.cumsum(load_evt.astype(int) - dump_evt.astype(int))
        score = np.clip(score, 0, None)
        loaded = score > 0
        empty = (score == 0) & (np.cumsum(dump_evt.astype(int)) > 0)

        if "on_haul_road" in g.columns:
            on_haul = g["on_haul_road"].to_numpy(dtype=int) == 1
        else:
            on_haul = np.ones(n, dtype=bool)

        spd_all = pd.to_numeric(g["speed"], errors="coerce").fillna(0).to_numpy(dtype=float)
        moving_mask = (spd_all > 2.0) & on_haul

        spd_loaded = spd_all[moving_mask & loaded]
        spd_empty = spd_all[moving_mask & empty]

        if spd_loaded.size:
            loaded_haul_speed_mean = float(spd_loaded.mean())
            loaded_haul_speed_p90 = float(np.percentile(spd_loaded, 90))
        if spd_empty.size:
            empty_haul_speed_mean = float(spd_empty.mean())
            empty_haul_speed_p90 = float(np.percentile(spd_empty, 90))

        if "disthav" in g.columns:
            dist = pd.to_numeric(g["disthav"], errors="coerce").fillna(0).to_numpy(dtype=float)
            loaded_haul_km = float(dist[on_haul & loaded].sum() / 1000.0)
            empty_haul_km = float(dist[on_haul & empty].sum() / 1000.0)

        if load_events > 0 and dump_events > 0:
            ts_ns = g["ts"].astype("int64").to_numpy()
            load_idx = np.where(load_evt)[0]
            dump_idx = np.where(dump_evt)[0]
            li = 0
            deltas = []
            for di in dump_idx:
                while li + 1 < len(load_idx) and load_idx[li + 1] <= di:
                    li += 1
                if load_idx[li] <= di:
                    dt_min = (ts_ns[di] - ts_ns[load_idx[li]]) / 1e9 / 60.0
                    if 0 <= dt_min <= 300:
                        deltas.append(dt_min)
            if deltas:
                cycle_time_min_p50 = float(np.median(deltas))

    # ── Congestion proxy ──────────────────────────────────────────
    # Stop-and-go on haul road = congestion
    stops = int(((g["speed"].shift(1).fillna(0) > 2) & (g["speed"] <= 2) & ign).sum())
    stop_density = stops / (shift_km + 1e-6)

    # ── External voltage ──────────────────────────────────────────
    ext_v = 0.0
    ext_v_max = 0.0
    if "external_voltage" in g.columns and not g["external_voltage"].isna().all():
        ev = g["external_voltage"].dropna()
        ext_v = float(ev.mean())
        ext_v_max = float(ev.max())

    # ── Speed stats ───────────────────────────────────────────────
    spd = g.loc[moving, "speed"]
    speed_mean = float(spd.mean()) if len(spd) > 0 else 0.0
    speed_p50  = float(spd.quantile(0.50)) if len(spd) > 0 else 0.0
    speed_p90  = float(spd.quantile(0.90)) if len(spd) > 0 else 0.0

    # ── Fractions ─────────────────────────────────────────────────
    frac_dump  = time_dump_s  / (total_s + 1e-6)
    frac_load  = time_load_s  / (total_s + 1e-6)
    frac_haul  = time_haul_s  / (total_s + 1e-6)
    frac_exc   = loading_dwell_s / (total_s + 1e-6)

    return pd.Series({
        # Motion
        "ignition_on_hours":   ign_s  / 3600,
        "moving_hours":        mov_s  / 3600,
        "idle_hours":          idle_s / 3600,
        "shift_km":            shift_km,
        "idle_fraction":       idle_s / (ign_s + 1e-6),
        "km_per_hour":         shift_km / (ign_s / 3600 + 1e-6),
        "stop_count":          stops,
        "stop_density":        stop_density,
        "n_pings":             n,
        # Altitude
        "total_climb_m":      total_climb,
        "total_descent_m":    total_descent,
        "altitude_mean":      float(g["altitude"].mean()),
        "altitude_std":       float(g["altitude"].std()),
        "altitude_range":     float(g["altitude"].max() - g["altitude"].min()),
        "net_lift":           float(g["altitude"].iloc[-1] - g["altitude"].iloc[0]) if n > 1 else 0.0,
        "climb_descent_ratio": total_climb / (total_descent + 1.0),
        "km_per_climb_m":     shift_km / (total_climb + 1.0),
        # Speed
        "speed_mean":  speed_mean,
        "speed_p50":   speed_p50,
        "speed_p90":   speed_p90,
        # Dump switch signal
        "dump_count_signal":  dump_count_sig,
        # Spatial zones
        "time_in_dump_zone_h":  time_dump_s  / 3600,
        "time_in_load_zone_h":  time_load_s  / 3600,
        "time_on_haul_road_h":  time_haul_s  / 3600,
        "time_in_ob_dump_h":    time_ob_s    / 3600,
        "time_in_rom_stock_h":  time_rom_s   / 3600,
        "frac_in_dump_zone":    frac_dump,
        "frac_in_load_zone":    frac_load,
        "frac_on_haul_road":    frac_haul,
        "frac_near_excavator":  frac_exc,
        "dump_zone_transitions": dump_transitions,
        "load_zone_transitions": load_transitions,
        "haul_road_transitions": haul_transitions,
        # Cycle proxies
        "haul_cycles_spatial":   haul_cycles,
        "loading_visits":        loading_visits,
        "loading_dwell_h":       loading_dwell_s / 3600,
        "km_per_cycle_spatial":  shift_km / (haul_cycles + 1.0),
        # External voltage
        "external_voltage_mean": ext_v,
        "external_voltage_max":  ext_v_max,
        # Loaded/empty haul behavior
        "loaded_haul_speed_mean": loaded_haul_speed_mean,
        "empty_haul_speed_mean":  empty_haul_speed_mean,
        "loaded_haul_speed_p90":  loaded_haul_speed_p90,
        "empty_haul_speed_p90":   empty_haul_speed_p90,
        "loaded_haul_km":         loaded_haul_km,
        "empty_haul_km":          empty_haul_km,
        "loaded_empty_haul_speed_ratio": loaded_haul_speed_mean / (empty_haul_speed_mean + 1e-6),
        "load_events":            load_events,
        "dump_events":            dump_events,
        "cycle_time_min_p50":     cycle_time_min_p50,
        # Time anchors
        "hour_start": int(g["ts"].iloc[0].hour),
        "hour_end":   int(g["ts"].iloc[-1].hour),
    })


# ─────────────────────────────────────────────────────────────────
# File processing
# ─────────────────────────────────────────────────────────────────

LOAD_COLS = [
    "vehicle",
    "mine_anon",
    "ts",
    "latitude",
    "longitude",
    "altitude",
    "speed",
    "ignition",
    "disthav",
    "analog_input_1",
    "external_voltage",
]


def ensure_tz(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts, errors="coerce", utc=True)
    return out.dt.tz_convert("Asia/Kolkata")


def process_file(
    fpath: Path,
    mine_layers: dict,
    dumper_set: set,
    exc_positions_utm: dict[str, np.ndarray],
) -> pd.DataFrame:
    print(f"  {fpath.name}...")
    try:
        df = pd.read_parquet(fpath, columns=LOAD_COLS)
    except Exception:
        df = pd.read_parquet(fpath)
        df = df[[c for c in LOAD_COLS if c in df.columns]].copy()

    df["ts"] = ensure_tz(df["ts"])
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0).clip(0, 80)
    df["altitude"] = pd.to_numeric(df["altitude"], errors="coerce").fillna(0)
    df["ignition"] = pd.to_numeric(df["ignition"], errors="coerce").fillna(0).astype(int)
    df["disthav"]  = pd.to_numeric(df["disthav"], errors="coerce").fillna(0)

    # Shift + working date
    hour = df["ts"].dt.hour
    df["shift"] = np.where((hour >= 6) & (hour < 14), "A",
                  np.where((hour >= 14) & (hour < 22), "B", "C"))
    day0 = df["ts"].dt.tz_localize(None).dt.normalize()
    nc   = (df["shift"] == "C") & (hour < 6)
    df["date"] = (day0 - pd.to_timedelta(nc.astype("int8"), unit="D")).dt.date

    # Speed filter
    df = df[df["speed"] <= 80].copy()
    df["speed"] = df["speed"].clip(0, 60)

    # Filter to dumpers
    df = df[df["vehicle"].isin(dumper_set)].copy()
    if len(df) == 0:
        return pd.DataFrame()

    # Add spatial zone flags
    if mine_layers:
        df = add_zone_flags_vectorized(df, mine_layers)

    # Add excavator proximity (fast anchors)
    if exc_positions_utm:
        df = add_excavator_proximity_fast(df, exc_positions_utm)

    # Aggregate per shift
    gkey = ["vehicle", "date", "shift"]
    feats = (
        df.groupby(gkey, observed=True)
        .apply(aggregate_shift_spatial)
        .reset_index()
    )
    gc.collect()
    return feats


# ─────────────────────────────────────────────────────────────────
# Post-processing: temporal + vehicle stats
# ─────────────────────────────────────────────────────────────────

def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    df = df.copy()
    df["day_of_week"]  = d.dt.dayofweek
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["week_number"]  = d.dt.isocalendar().week.astype(int)
    df["day_of_month"] = d.dt.day
    df["month"]        = d.dt.month
    df["shift_enc"]    = df["shift"].map({"C": 0, "A": 1, "B": 2})
    return df


def add_vehicle_stats(labeled: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame):
    veh = (
        labeled.groupby("vehicle")
        .agg(
            veh_mean_km=("shift_km", "mean"),
            veh_std_km=("shift_km", "std"),
            veh_mean_ign_h=("ignition_on_hours", "mean"),
            veh_mean_idle_frac=("idle_fraction", "mean"),
            veh_mean_cycles_spatial=("haul_cycles_spatial", "mean"),
            veh_mean_climb=("total_climb_m", "mean"),
            veh_mean_ext_v=("external_voltage_mean", "mean"),
        )
        .reset_index()
    )

    veh_shift = (
        labeled.groupby(["vehicle", "shift"])
        .agg(
            veh_shift_mean_acons=("acons", "mean"),
            veh_shift_std_acons=("acons", "std"),
        )
        .reset_index()
    )

    for df in [train, test]:
        for col in list(veh.columns[1:]) + list(veh_shift.columns[2:]):
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    tr = train.merge(veh, on="vehicle", how="left").merge(veh_shift, on=["vehicle", "shift"], how="left")
    te = test.merge(veh, on="vehicle", how="left").merge(veh_shift, on=["vehicle", "shift"], how="left")
    return tr, te


def aggregate_refuels(refuels: pd.DataFrame, offset_min: int = 0) -> pd.DataFrame:
    r = refuels.copy()
    if "fleet_type" in r.columns:
        r = r[r["fleet_type"] == "Dumper"].copy()

    r["ts"] = ensure_tz(r["ts"])
    r["ts_adj"] = r["ts"] + pd.Timedelta(minutes=offset_min)
    hour = r["ts_adj"].dt.hour
    r["shift"] = np.where((hour >= 6) & (hour < 14), "A",
                 np.where((hour >= 14) & (hour < 22), "B", "C"))
    day0 = r["ts_adj"].dt.tz_localize(None).dt.normalize()
    nc   = (r["shift"] == "C") & (hour < 6)
    r["date"] = (day0 - pd.to_timedelta(nc.astype("int8"), unit="D")).dt.date

    r = r.sort_values(["vehicle", "ts"])
    r["prev_ts"] = r.groupby("vehicle")["ts"].shift(1)
    r["hrs_since_prev"] = (r["ts"] - r["prev_ts"]).dt.total_seconds() / 3600

    agg = (
        r.groupby(["vehicle", "date", "shift"])
        .agg(
            refuel_liters=("litres", "sum"),
            refuel_count=("litres", "count"),
            refuel_liters_max=("litres", "max"),
            hrs_since_prev_refuel=("hrs_since_prev", "min"),
        )
        .reset_index()
        .fillna(0)
    )

    sfx = f"_off{offset_min:+d}m".replace("+", "p").replace("-", "m")
    return agg.rename(columns={
        "refuel_liters": f"refuel_liters{sfx}",
        "refuel_count":  f"refuel_count{sfx}",
        "refuel_liters_max": f"refuel_liters_max{sfx}",
        "hrs_since_prev_refuel": f"hrs_since_refuel{sfx}",
    })


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT),
        help="Output directory (default: outputs/spatial_features)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    fleet = pd.read_csv(DATA / "fleet.csv")
    dumper_set = set(fleet[fleet["fleet"] == "Dumper"]["vehicle"].tolist())
    print(f"Dumpers: {len(dumper_set)}")

    print("Loading spatial layers...")
    mine_layers = load_mine_layers(DATA)

    all_tel = sorted(DATA.glob("telemetry_2026-0*.parquet"))
    train_files = [f for f in all_tel if f.name not in TEST_FILES]
    test_files  = [f for f in all_tel if f.name in TEST_FILES]

    print(f"\nTrain files: {[f.name for f in train_files]}")
    print(f"Test files:  {[f.name for f in test_files]}")

    print("\nComputing excavator anchor positions (fast)...")
    exc_positions_utm = compute_excavator_mean_positions(DATA, all_tel)
    for mine_id, anchors in exc_positions_utm.items():
        print(f"  {mine_id}: {len(anchors)} exc anchors")

    # ── Train ────────────────────────────────────────────────────
    print("\nProcessing training telemetry (spatial + cycles)...")
    train_chunks = []
    for f in train_files:
        out_part = parts_dir / f"train__{f.stem}.parquet"
        if out_part.exists():
            chunk = pd.read_parquet(out_part)
        else:
            chunk = process_file(f, mine_layers, dumper_set, exc_positions_utm)
            if len(chunk) > 0:
                chunk.to_parquet(out_part, index=False)
        if len(chunk) > 0:
            train_chunks.append(chunk)
        gc.collect()
    train_feats = pd.concat(train_chunks, ignore_index=True)
    print(f"  Train raw: {train_feats.shape}")

    # ── Test ─────────────────────────────────────────────────────
    print("\nProcessing test telemetry (spatial + cycles)...")
    test_chunks = []
    for f in test_files:
        out_part = parts_dir / f"test__{f.stem}.parquet"
        if out_part.exists():
            chunk = pd.read_parquet(out_part)
        else:
            chunk = process_file(f, mine_layers, dumper_set, exc_positions_utm)
            if len(chunk) > 0:
                chunk.to_parquet(out_part, index=False)
        if len(chunk) > 0:
            test_chunks.append(chunk)
        gc.collect()
    test_feats = pd.concat(test_chunks, ignore_index=True)
    print(f"  Test raw:  {test_feats.shape}")

    # ── Targets ──────────────────────────────────────────────────
    smry = pd.concat([
        pd.read_csv(DATA / "smry_jan_train_ordered.csv"),
        pd.read_csv(DATA / "smry_feb_train_ordered.csv"),
        pd.read_csv(DATA / "smry_mar_train_ordered.csv"),
    ], ignore_index=True)
    smry["date"] = pd.to_datetime(smry["date"]).dt.date

    train_feats["date"] = pd.to_datetime(train_feats["date"]).dt.date
    test_feats["date"]  = pd.to_datetime(test_feats["date"]).dt.date

    train_feats = train_feats.merge(
        smry[["vehicle", "date", "shift", "acons"]],
        on=["vehicle", "date", "shift"], how="left",
    )
    print(f"  Labeled: {train_feats['acons'].notna().sum()} / {len(train_feats)}")

    # ── Fleet features ───────────────────────────────────────────
    fleet_feats = (
        fleet[fleet["fleet"] == "Dumper"][["vehicle", "mine_anon"]]
        .assign(mine_enc=lambda d: d["mine_anon"].map({"mine001": 0, "mine002": 1}))
    )
    for df in [train_feats, test_feats]:
        if "mine_enc" in df.columns:
            df.drop(columns=["mine_enc"], inplace=True)
    train_feats = train_feats.merge(fleet_feats[["vehicle", "mine_enc"]], on="vehicle", how="left")
    test_feats  = test_feats.merge( fleet_feats[["vehicle", "mine_enc"]], on="vehicle", how="left")

    # ── Temporal ─────────────────────────────────────────────────
    train_feats = add_temporal(train_feats)
    test_feats  = add_temporal(test_feats)

    # ── Vehicle stats ─────────────────────────────────────────────
    labeled = train_feats.dropna(subset=["acons"]).copy()
    train_feats, test_feats = add_vehicle_stats(labeled, train_feats, test_feats)

    # ── Refuel features ───────────────────────────────────────────
    ref_files = sorted(DATA.glob("rfid_refuels_*.parquet"))
    if ref_files:
        refuels = pd.read_parquet(ref_files[-1])
        for offset in [0, 2, -2]:
            agg = aggregate_refuels(refuels, offset)
            train_feats = train_feats.merge(agg, on=["vehicle", "date", "shift"], how="left")
            test_feats  = test_feats.merge( agg, on=["vehicle", "date", "shift"], how="left")
        ref_cols = [c for c in train_feats.columns
                    if c.startswith("refuel_") or c.startswith("hrs_since_refuel")]
        train_feats[ref_cols] = train_feats[ref_cols].fillna(0)
        test_feats[ref_cols]  = test_feats[ref_cols].fillna(0)
        print(f"  Refuel features added: {len(ref_cols)}")

    print(f"\nFinal train: {train_feats.shape}")
    print(f"Final test:  {test_feats.shape}")
    print(f"Labeled rows: {train_feats['acons'].notna().sum()}")

    # ── Spatial feature coverage ──────────────────────────────────
    for col in ["frac_in_dump_zone", "frac_in_load_zone", "frac_on_haul_road",
                "haul_cycles_spatial", "loading_visits"]:
        if col in train_feats.columns:
            nz = (train_feats[col] > 0).sum()
            print(f"  {col}: non-zero in {nz}/{len(train_feats)} rows ({100*nz/len(train_feats):.1f}%)")

    train_feats.to_csv(out_dir / "train_spatial.csv", index=False)
    test_feats.to_csv(out_dir / "test_spatial.csv",   index=False)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
