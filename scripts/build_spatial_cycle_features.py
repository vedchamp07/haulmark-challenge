#!/usr/bin/env python3
"""
Spatial + Cycle Detection Feature Pipeline.

New features based on orientation notes:
1. Spatial zone features from .gpkg (time in dump/load/haul zones)
2. Haul cycle detection (load → haul → dump → empty → load)
3. Excavator proximity for loading event detection
4. Load-state features (loaded vs empty haul speed/distance)
5. Dump event count via spatial transitions (for Jan/Feb without analog signal)

Outputs:
- outputs/spatial_features/train_spatial.csv
- outputs/spatial_features/test_spatial.csv
"""

from __future__ import annotations

import gc
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
OUTPUT = ROOT / "outputs" / "spatial_features"
OUTPUT.mkdir(parents=True, exist_ok=True)

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
        if mask.sum() == 0:
            continue

        pts = gpd.GeoSeries(
            gpd.points_from_xy(x[mask], y[mask]), crs="EPSG:32645"
        )

        def within(union):
            if union is None:
                return np.zeros(mask.sum(), dtype=bool)
            return pts.within(union).to_numpy()

        in_ob  = within(layers["ob_dump_union"])
        in_rom = within(layers["stock_union"])
        in_dum = in_ob | in_rom
        in_lod = within(layers["bench_union"]) | within(layers["cpu_union"])
        on_hau = within(layers["haul_union"])

        df.loc[mask, "in_dump_zone"]     = in_dum.astype(np.int8)
        df.loc[mask, "in_ob_dump_zone"]  = in_ob.astype(np.int8)
        df.loc[mask, "in_rom_stock_zone"] = in_rom.astype(np.int8)
        df.loc[mask, "in_load_zone"]     = in_lod.astype(np.int8)
        df.loc[mask, "on_haul_road"]     = on_hau.astype(np.int8)

    return df


# ─────────────────────────────────────────────────────────────────
# Excavator helpers
# ─────────────────────────────────────────────────────────────────

def extract_excavator_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a telemetry chunk, extract per-(mine, date, shift) excavator positions.
    Excavators barely move — median lat/lon is their operating location.
    """
    exc = df[df["vehicle"].astype(str).str.startswith("Exc")].copy()
    if exc.empty:
        return pd.DataFrame(columns=["mine_anon", "date", "shift",
                                     "exc_x_utm", "exc_y_utm", "exc_vehicle"])
    exc = exc[exc["speed"] < 2].copy()  # only pings where stationary
    if exc.empty:
        return pd.DataFrame(columns=["mine_anon", "date", "shift",
                                     "exc_x_utm", "exc_y_utm", "exc_vehicle"])

    x, y = to_utm(exc["latitude"].to_numpy(dtype=float),
                  exc["longitude"].to_numpy(dtype=float))
    exc["x_utm"] = x
    exc["y_utm"] = y

    pos = (
        exc.groupby(["vehicle", "mine_anon", "date", "shift"], observed=True)
        .agg(exc_x_utm=("x_utm", "median"), exc_y_utm=("y_utm", "median"))
        .reset_index()
        .rename(columns={"vehicle": "exc_vehicle"})
    )
    return pos


def add_excavator_proximity(df: pd.DataFrame, exc_pos: pd.DataFrame) -> pd.DataFrame:
    """Mark each dumper ping as near-excavator (loading zone) or not."""
    df = df.copy()
    df["near_excavator"] = 0

    if exc_pos.empty or "x_utm" not in df.columns:
        return df

    for mine, mine_exc in exc_pos.groupby("mine_anon", observed=True):
        # Get all unique excavator positions for this mine
        mine_mask = df["mine_anon"].astype(str).str.lower() == str(mine).lower()
        if mine_mask.sum() == 0:
            continue

        # Per date+shift, check proximity to each excavator operating in that shift
        for (date, shift), shift_exc in mine_exc.groupby(["date", "shift"], observed=True):
            row_mask = mine_mask & (df["date"] == date) & (df["shift"] == shift)
            if row_mask.sum() == 0:
                continue

            dx = df.loc[row_mask, "x_utm"].to_numpy() - shift_exc["exc_x_utm"].to_numpy()[:, None]
            dy = df.loc[row_mask, "y_utm"].to_numpy() - shift_exc["exc_y_utm"].to_numpy()[:, None]
            dist2 = dx**2 + dy**2  # shape: (n_exc, n_pings)
            near = (dist2 < EXC_PROXIMITY_M**2).any(axis=0)
            df.loc[row_mask, "near_excavator"] = near.astype(np.int8)

    return df


# ─────────────────────────────────────────────────────────────────
# Shift-level aggregation
# ─────────────────────────────────────────────────────────────────

def aggregate_shift_spatial(g: pd.DataFrame) -> pd.Series:
    """
    Aggregate per-shift spatial + cycle features.
    Input: all telemetry rows for one (vehicle, date, shift).
    """
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

    # ── Haul cycle count ─────────────────────────────────────────
    # Best estimate: max of dump_count_sig, dump_transitions, loading_visits
    # Prefer dump switch signal when available (most reliable)
    haul_cycles = max(dump_count_sig, dump_transitions)

    # ── Load state speed features ─────────────────────────────────
    # Loaded = between load_zone (or exc_proximity) departure and dump_zone arrival
    # Heuristic: points ON haul road + speed > 2 → differentiate by direction
    # Simple proxy: use speed quantiles on haul road pings
    loaded_speed_mean = 0.0
    empty_speed_mean  = 0.0
    loaded_km = 0.0
    empty_km  = 0.0

    if "near_excavator" in g.columns and "in_dump_zone" in g.columns:
        # Assign load state using spatial transitions
        state = np.zeros(n, dtype=int)  # 0=unknown, 1=loaded, 2=empty
        cur = 0
        for i in range(n):
            if g["near_excavator"].iloc[i] == 1:
                cur = 1  # just loaded
            elif g["in_dump_zone"].iloc[i] == 1:
                cur = 2  # just dumped → empty
            state[i] = cur

        loaded_mask = state == 1
        empty_mask  = state == 2

        spd_loaded = g.loc[loaded_mask & moving, "speed"]
        spd_empty  = g.loc[empty_mask  & moving, "speed"]
        loaded_speed_mean = float(spd_loaded.mean()) if len(spd_loaded) > 0 else 0.0
        empty_speed_mean  = float(spd_empty.mean())  if len(spd_empty)  > 0 else 0.0
        loaded_km = g.loc[loaded_mask, "disthav"].sum() / 1000
        empty_km  = g.loc[empty_mask,  "disthav"].sum() / 1000

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
        # Cycle
        "haul_cycles":          haul_cycles,
        "loading_visits":       loading_visits,
        "loading_dwell_h":      loading_dwell_s / 3600,
        "km_per_cycle":         shift_km / (haul_cycles + 1.0),
        # Load-state speed
        "loaded_speed_mean":    loaded_speed_mean,
        "empty_speed_mean":     empty_speed_mean,
        "loaded_km":            loaded_km,
        "empty_km":             empty_km,
        "loaded_empty_speed_ratio": loaded_speed_mean / (empty_speed_mean + 1e-6),
        # External voltage
        "external_voltage_mean": ext_v,
        "external_voltage_max":  ext_v_max,
        # Time anchors
        "hour_start": int(g["ts"].iloc[0].hour),
        "hour_end":   int(g["ts"].iloc[-1].hour),
    })


# ─────────────────────────────────────────────────────────────────
# File processing
# ─────────────────────────────────────────────────────────────────

LOAD_COLS = [
    "vehicle", "mine_anon", "ts",
    "latitude", "longitude", "altitude", "speed", "ignition", "disthav",
    "analog_input_1", "external_voltage", "satellites", "gnss_hdop",
    "axis_x", "axis_y", "axis_z", "battery_level", "angle",
]


def ensure_tz(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts, errors="coerce", utc=True)
    return out.dt.tz_convert("Asia/Kolkata")


def process_file(
    fpath: Path,
    mine_layers: dict,
    dumper_set: set,
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

    # Extract excavator positions BEFORE filtering to dumpers
    exc_pos = extract_excavator_positions(df)

    # Filter to dumpers
    df = df[df["vehicle"].isin(dumper_set)].copy()
    if len(df) == 0:
        return pd.DataFrame()

    # Add spatial zone flags
    if mine_layers:
        df = add_zone_flags_vectorized(df, mine_layers)

    # Add excavator proximity
    if not exc_pos.empty:
        df = add_excavator_proximity(df, exc_pos)

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
            veh_mean_cycles=("haul_cycles", "mean"),
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

    # ── Train ────────────────────────────────────────────────────
    print("\nProcessing training telemetry (spatial + cycles)...")
    train_chunks = []
    for f in train_files:
        chunk = process_file(f, mine_layers, dumper_set)
        if len(chunk) > 0:
            train_chunks.append(chunk)
        gc.collect()
    train_feats = pd.concat(train_chunks, ignore_index=True)
    print(f"  Train raw: {train_feats.shape}")

    # ── Test ─────────────────────────────────────────────────────
    print("\nProcessing test telemetry (spatial + cycles)...")
    test_chunks = []
    for f in test_files:
        chunk = process_file(f, mine_layers, dumper_set)
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
                "haul_cycles", "loading_visits"]:
        if col in train_feats.columns:
            nz = (train_feats[col] > 0).sum()
            print(f"  {col}: non-zero in {nz}/{len(train_feats)} rows ({100*nz/len(train_feats):.1f}%)")

    train_feats.to_csv(OUTPUT / "train_spatial.csv", index=False)
    test_feats.to_csv(OUTPUT / "test_spatial.csv",   index=False)
    print(f"\nSaved to {OUTPUT}/")


if __name__ == "__main__":
    main()
