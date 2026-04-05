#!/usr/bin/env python3
"""
v11 feature builder — builds ENTIRELY from raw telemetry (no DPR columns).

Key fixes vs v4/v5:
  1. Zero DPR contamination: does NOT use total_trip, operator_id, km_dpr etc.
     (those are in training telemetry but NOT in test → the core bug causing ~800 MSE)
  2. Per-dumper excavator assignment: each dumper is assigned to 1 excavator (fixed),
     using all excavators creates false loading detections
  3. Validated cycle count: telemetry-derived trip count replaces DPR total_trip
  4. Physics features split by load state (loaded vs empty haul)

Outputs: ckpts/train_v11.parquet, ckpts/test_v11.parquet
Runtime: ~15-25 minutes
"""
import datetime
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.ops import polygonize, unary_union

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "ckpts"
CKPT.mkdir(exist_ok=True)

TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)

TRAIN_FILES = [
    "telemetry_2026-01-01_2026-01-10.parquet",
    "telemetry_2026-01-11_2026-01-20.parquet",
    "telemetry_2026-02-01_2026-02-10.parquet",
    "telemetry_2026-02-11_2026-02-20.parquet",
    "telemetry_2026-03-01_2026-03-11.parquet",
]
TEST_FILES = [
    "telemetry_2026-01-21_2026-01-31.parquet",
    "telemetry_2026-02-21_2026-02-28.parquet",
    "telemetry_2026-03-12_2026-03-20.parquet",
]
ALL_FILES = TRAIN_FILES + TEST_FILES

# Only use columns available in test telemetry — no DPR columns
SAFE_COLS = [
    "vehicle", "ts", "latitude", "longitude", "altitude",
    "speed", "ignition", "angle", "mine_anon",
    "external_voltage", "axis_x", "axis_y", "axis_z",
    "disthav", "cumdist", "satellites", "gnss_hdop",
]
# analog_input_1 loaded separately (dump switch, mine001 only)


# ── Load spatial zones ────────────────────────────────────────────────────────
print("Loading spatial layers...")
mine_zones = {}
for mine_id, fname in [("mine001", "mine_001_anonymized.gpkg"),
                        ("mine002", "mine_002_anonymized.gpkg")]:
    gpkg = DATA / fname

    def load_layer(lname):
        try:
            gdf = gpd.read_file(str(gpkg), layer=lname)
            if gdf.crs and gdf.crs.to_epsg() != 32645:
                gdf = gdf.to_crs("EPSG:32645")
            return gdf
        except Exception:
            return None

    def make_zone(gdf, buf):
        if gdf is None or len(gdf) == 0:
            return None
        polys = list(polygonize(gdf.geometry.union_all()))
        if polys:
            return unary_union(polys).buffer(buf * 0.5)
        return gdf.geometry.buffer(buf).union_all()

    ob = load_layer("ob_dump")
    stk = load_layer("mineral_stock")
    hr = load_layer("haul_road")
    bench = load_layer("bench")  # noisy but useful for mine-boundary exclusion

    dump_parts = []
    if ob is not None and len(ob) > 0:
        dump_parts.append(make_zone(ob, 50))
    if stk is not None and len(stk) > 0:
        dump_parts.append(make_zone(stk, 50))

    mine_zones[mine_id] = {
        "any_dump": unary_union([z for z in dump_parts if z is not None]) if dump_parts else None,
        "haul_road": hr.geometry.buffer(25).union_all() if hr is not None and len(hr) > 0 else None,
    }
    print(f"  {mine_id}: dump={mine_zones[mine_id]['any_dump'] is not None}, "
          f"haul={mine_zones[mine_id]['haul_road'] is not None}")


# ── Shift assignment ──────────────────────────────────────────────────────────
def assign_shift(ts_series: pd.Series):
    ts = pd.to_datetime(ts_series, utc=False)
    hour = ts.dt.hour
    date = ts.dt.date
    adj = [d - datetime.timedelta(days=1) if h >= 22 else d for d, h in zip(date, hour)]
    shift = ["C" if h >= 22 or h < 6 else ("A" if h < 14 else "B") for h in hour]
    return pd.Series(adj, index=ts_series.index), pd.Series(shift, index=ts_series.index)


# ── Step 1: Find per-dumper excavator assignment ──────────────────────────────
print("\nBuilding per-dumper excavator assignment from all telemetry...")
exc_pos_frames = []
dump_slow_frames = []

for fname in ALL_FILES:
    path = DATA / fname
    print(f"  {fname} (excavator scan)...")
    # Load just the location + vehicle cols
    try:
        df = pd.read_parquet(path, columns=["vehicle", "mine_anon", "latitude", "longitude", "speed", "ignition"])
    except Exception as e:
        print(f"  Warning: {e}")
        continue

    # Excavator positions (mean is fine, they barely move)
    exc = df[df["vehicle"].str.startswith("Exc")].copy()
    if len(exc) > 0:
        exc_pos_frames.append(exc[["vehicle", "mine_anon", "latitude", "longitude"]])

    # Dumper slow pings (speed < 5 km/h, ignition on) = potential loading or stopping
    dump_slow = df[df["vehicle"].str.startswith("Dump") & (df["speed"] < 5) & (df["ignition"] == 1)].copy()
    if len(dump_slow) > 0:
        dump_slow_frames.append(dump_slow[["vehicle", "mine_anon", "latitude", "longitude"]])

exc_all = pd.concat(exc_pos_frames, ignore_index=True)
dump_slow_all = pd.concat(dump_slow_frames, ignore_index=True)
del exc_pos_frames, dump_slow_frames
gc.collect()

# Mean excavator position per vehicle (global — they barely move)
exc_mean = exc_all.groupby(["vehicle", "mine_anon"])[["latitude", "longitude"]].mean().reset_index()
ex_x, ex_y = TRANSFORMER.transform(exc_mean["longitude"].values, exc_mean["latitude"].values)
exc_mean["x_utm"] = ex_x
exc_mean["y_utm"] = ex_y
print(f"  Found {len(exc_mean)} excavators")

# For each dumper's slow pings, find nearest excavator
dump_x, dump_y = TRANSFORMER.transform(dump_slow_all["longitude"].values, dump_slow_all["latitude"].values)
dump_slow_all["x_utm"] = dump_x
dump_slow_all["y_utm"] = dump_y

# Build per-mine excavator arrays
exc_by_mine = {}
for mine_id in ["mine001", "mine002"]:
    sub = exc_mean[exc_mean["mine_anon"] == mine_id]
    exc_by_mine[mine_id] = sub[["vehicle", "x_utm", "y_utm"]].reset_index(drop=True)

# For each dumper slow ping, find the nearest excavator in its mine
LOADING_RADIUS_M = 100.0  # Maximum distance to count as "near excavator"

nearest_exc = []
for mine_id, group in dump_slow_all.groupby("mine_anon"):
    if mine_id not in exc_by_mine or len(exc_by_mine[mine_id]) == 0:
        continue
    exc_xy = exc_by_mine[mine_id][["x_utm", "y_utm"]].values
    exc_names = exc_by_mine[mine_id]["vehicle"].values
    dx = group["x_utm"].values[:, None] - exc_xy[:, 0]
    dy = group["y_utm"].values[:, None] - exc_xy[:, 1]
    dists = np.sqrt(dx**2 + dy**2)
    min_dist_idx = dists.argmin(axis=1)
    min_dist = dists.min(axis=1)
    # Only count pings within loading radius
    mask = min_dist < LOADING_RADIUS_M
    if mask.sum() == 0:
        continue
    sub = group[mask].copy()
    sub["nearest_exc"] = exc_names[min_dist_idx[mask]]
    sub["exc_dist_m"] = min_dist[mask]
    nearest_exc.append(sub[["vehicle", "nearest_exc", "exc_dist_m"]])

nearest_exc_df = pd.concat(nearest_exc, ignore_index=True) if nearest_exc else pd.DataFrame()

# Per-dumper primary excavator = most frequent in near-loading pings
if len(nearest_exc_df) > 0:
    dumper_exc_map = (
        nearest_exc_df.groupby("vehicle")["nearest_exc"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
else:
    dumper_exc_map = {}

print(f"  Assigned excavators for {len(dumper_exc_map)} dumpers")
print(f"  Sample: {dict(list(dumper_exc_map.items())[:5])}")

# Per-dumper excavator UTM position lookup
dumper_exc_xy = {}
for dumper, exc_name in dumper_exc_map.items():
    row = exc_mean[exc_mean["vehicle"] == exc_name]
    if len(row) > 0:
        dumper_exc_xy[dumper] = (float(row["x_utm"].iloc[0]), float(row["y_utm"].iloc[0]))

del dump_slow_all, nearest_exc_df
gc.collect()


# ── State machine per vehicle ─────────────────────────────────────────────────
def run_state_machine(group: pd.DataFrame) -> np.ndarray:
    """
    Returns loaded_state array:
      1.0 = LOADED (after leaving loading zone, before dump zone)
      0.0 = EMPTY  (after dump, or default)
    Uses per-dumper excavator assignment for loading detection.
    """
    at_l = group["at_loading"].values
    at_d = group["at_dump"].values
    is_d = group["is_dumping"].values

    # Transitions: just-left-loading = loaded, at-dump or dumping = empty
    just_left_loading = (~at_l) & np.concatenate([[False], at_l[:-1]])
    now_empty = at_d | is_d

    state = pd.Series(np.nan, index=range(len(group)))
    state[just_left_loading] = 1.0  # became loaded
    state[now_empty] = 0.0          # became empty
    state[at_l] = np.nan            # in loading zone: transitioning

    return state.ffill().bfill().fillna(0.0).values


# ── Main feature extraction ───────────────────────────────────────────────────
def extract_features(file_list: list, label: str) -> pd.DataFrame:
    parts = []
    for fname in file_list:
        path = DATA / fname
        print(f"  {fname}...")

        # Detect available columns
        import pyarrow.parquet as pq
        try:
            all_avail = pq.read_schema(str(path)).names
        except Exception:
            all_avail = SAFE_COLS + ["analog_input_1"]

        use_cols = [c for c in SAFE_COLS if c in all_avail]
        if "analog_input_1" in all_avail:
            use_cols.append("analog_input_1")

        df = pd.read_parquet(path, columns=use_cols)
        df = df[df["vehicle"].str.startswith("Dump")].copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=False)
        df["adj_date"], df["shift"] = assign_shift(df["ts"])
        df = df.sort_values(["vehicle", "ts"]).reset_index(drop=True)

        # UTM coordinates
        x, y = TRANSFORMER.transform(df["longitude"].values, df["latitude"].values)
        df["x_utm"] = x
        df["y_utm"] = y

        # ── Zone flags ────────────────────────────────────────────────────────
        df["at_dump"] = False
        df["on_haul"] = False
        for mine_id, zones in mine_zones.items():
            mask = df["mine_anon"] == mine_id
            if mask.sum() == 0 or zones["any_dump"] is None:
                continue
            pts = gpd.GeoSeries(
                gpd.points_from_xy(df.loc[mask, "x_utm"].values,
                                   df.loc[mask, "y_utm"].values),
                crs="EPSG:32645"
            )
            if zones["any_dump"] is not None:
                df.loc[mask, "at_dump"] = pts.within(zones["any_dump"]).values
            if zones["haul_road"] is not None:
                df.loc[mask, "on_haul"] = pts.within(zones["haul_road"]).values

        # ── Per-dumper excavator proximity (assigned excavator only) ──────────
        df["at_loading"] = False
        for veh in df["vehicle"].unique():
            if veh not in dumper_exc_xy:
                continue
            exc_x, exc_y = dumper_exc_xy[veh]
            vmask = df["vehicle"] == veh
            vdf = df[vmask]
            dist = np.sqrt(
                (vdf["x_utm"].values - exc_x)**2 +
                (vdf["y_utm"].values - exc_y)**2
            )
            # At loading = within 100m of assigned excavator
            df.loc[vmask, "at_loading"] = dist < LOADING_RADIUS_M

        # Dump switch (mine001 Feb17+, sparse otherwise — gracefully handled)
        if "analog_input_1" in df.columns:
            df["is_dumping"] = df["analog_input_1"].fillna(0) > 2.5
        else:
            df["is_dumping"] = False

        # ── State machine per vehicle ─────────────────────────────────────────
        state_parts = []
        for veh, vdf in df.groupby("vehicle", sort=False):
            vdf_sorted = vdf.sort_values("ts")
            st = run_state_machine(vdf_sorted)
            state_parts.append(pd.Series(st, index=vdf_sorted.index))
        df["loaded_state"] = pd.concat(state_parts).sort_index()

        # ── Derived ping-level signals ────────────────────────────────────────
        df["is_moving"] = (df["speed"] > 1.0) & (df["ignition"] == 1)
        df["is_idle"] = (df["speed"] <= 1.0) & (df["ignition"] == 1)
        df["loaded_moving"] = (df["loaded_state"] == 1.0) & df["is_moving"]
        df["empty_moving"] = (df["loaded_state"] == 0.0) & df["is_moving"]

        # Altitude diffs (per vehicle)
        df["d_alt"] = df.groupby("vehicle")["altitude"].transform(
            lambda x: x.diff().fillna(0).clip(-100, 100)
        )
        df["alt_gain"] = df["d_alt"].clip(lower=0)
        df["alt_loss"] = (-df["d_alt"]).clip(lower=0)
        df["loaded_alt_gain"] = df["alt_gain"] * df["loaded_moving"].astype(float)
        df["empty_alt_gain"] = df["alt_gain"] * df["empty_moving"].astype(float)

        # External voltage × load state
        ev = df["external_voltage"].fillna(0).values
        df["loaded_ext_v"] = ev * df["loaded_moving"].astype(float)
        df["empty_ext_v"] = ev * df["empty_moving"].astype(float)

        # Heading change (congestion proxy)
        if "angle" in df.columns:
            df["d_angle"] = df.groupby("vehicle")["angle"].transform(
                lambda x: x.diff().abs().fillna(0).clip(0, 180)
            )
        else:
            df["d_angle"] = 0.0

        # Cycle transitions
        df["loading_exit"] = df.groupby("vehicle")["at_loading"].transform(
            lambda x: (~x) & x.shift(1, fill_value=False)
        )
        df["dump_entry"] = df.groupby("vehicle")["at_dump"].transform(
            lambda x: x & (~x.shift(1, fill_value=False))
        )
        df["dump_exit"] = df.groupby("vehicle")["at_dump"].transform(
            lambda x: (~x) & x.shift(1, fill_value=False)
        )
        df["dump_active_edge"] = df.groupby("vehicle")["is_dumping"].transform(
            lambda x: x & (~x.shift(1, fill_value=False))
        )

        # ── Time intervals from actual timestamps (capped at 5 min for gaps) ────
        MAX_INTERVAL_H = 5 / 60.0  # 5-minute cap
        df["dt_h"] = df.groupby("vehicle")["ts"].transform(
            lambda x: x.diff().dt.total_seconds().fillna(30).clip(0, 300) / 3600.0
        )
        df["ign_dt_h"]    = df["dt_h"] * (df["ignition"] == 1).astype(float)
        df["mov_dt_h"]    = df["dt_h"] * df["is_moving"].astype(float)
        df["idle_dt_h"]   = df["dt_h"] * df["is_idle"].astype(float)
        df["loaded_dt_h"] = df["dt_h"] * df["loaded_moving"].astype(float)
        df["empty_dt_h"]  = df["dt_h"] * df["empty_moving"].astype(float)

        # ── Aggregate to shift level ──────────────────────────────────────────
        key = ["vehicle", "adj_date", "shift"]
        agg = df.groupby(key).agg(
            # Time fractions (derived from pings, not DPR)
            n_pings        = ("ts",              "count"),
            n_ign_pings    = ("ignition",         "sum"),
            n_moving_pings = ("is_moving",        "sum"),
            n_idle_pings   = ("is_idle",          "sum"),
            n_loaded_pings = ("loaded_moving",    "sum"),
            n_empty_pings  = ("empty_moving",     "sum"),

            # Distance
            cumdist_km     = ("cumdist",  lambda x: x.max() - x.min()),
            disthav_km     = ("disthav",  lambda x: x.sum() / 1000.0),

            # Speed stats (moving only)
            speed_mean     = ("speed",   "mean"),
            speed_std      = ("speed",   "std"),
            speed_max      = ("speed",   "max"),
            speed_p25      = ("speed",   lambda x: x.quantile(0.25)),
            speed_p50      = ("speed",   lambda x: x.quantile(0.50)),
            speed_p75      = ("speed",   lambda x: x.quantile(0.75)),

            # Altitude
            altitude_mean  = ("altitude", "mean"),
            altitude_std   = ("altitude", "std"),
            altitude_max   = ("altitude", "max"),
            altitude_min   = ("altitude", "min"),
            altitude_range = ("altitude", lambda x: x.max() - x.min()),
            altitude_gain_m= ("alt_gain", "sum"),
            altitude_loss_m= ("alt_loss", "sum"),

            # Load-state altitude
            loaded_alt_gain_m= ("loaded_alt_gain", "sum"),
            empty_alt_gain_m = ("empty_alt_gain",  "sum"),

            # External voltage
            ext_v_mean     = ("external_voltage", "mean"),
            ext_v_std      = ("external_voltage", "std"),
            ext_v_max      = ("external_voltage", "max"),
            loaded_ext_v_sum= ("loaded_ext_v",    "sum"),
            empty_ext_v_sum = ("empty_ext_v",     "sum"),

                # Time accumulators (actual intervals, capped at 5 min to handle gaps)
            ign_h_actual   = ("ign_dt_h",    "sum"),
            mov_h_actual   = ("mov_dt_h",    "sum"),
            idle_h_actual  = ("idle_dt_h",   "sum"),
            loaded_h_actual= ("loaded_dt_h", "sum"),
            empty_h_actual = ("empty_dt_h",  "sum"),

            # Dump switch (mine001 only; 0 elsewhere)
            dump_events    = ("dump_active_edge", "sum"),
            dump_pings_active = ("is_dumping",    "sum"),

            # Spatial zone fractions
            frac_dump_zone = ("at_dump",    "mean"),
            frac_haul_road = ("on_haul",    "mean"),
            frac_loading   = ("at_loading", "mean"),

            # Cycles (telemetry-derived, works for both train and test)
            n_loading_events = ("loading_exit",    "sum"),
            n_dump_entries   = ("dump_entry",      "sum"),
            n_dump_exits     = ("dump_exit",       "sum"),

            # Congestion
            angle_change_mean = ("d_angle", "mean"),

        ).reset_index()

        # GPS quality (optional column)
        if "satellites" in df.columns:
            sat_agg = df.groupby(key)["satellites"].mean().reset_index().rename(columns={"satellites": "n_satellites"})
            agg = agg.merge(sat_agg, on=key, how="left")
        else:
            agg["n_satellites"] = 0.0

        parts.append(agg)
        del df
        gc.collect()

    result = pd.concat(parts, ignore_index=True)
    result.rename(columns={"adj_date": "date"}, inplace=True)
    result["date"] = pd.to_datetime(result["date"]).dt.date

    # ── Derived shift-level features ──────────────────────────────────────────
    # Use actual timestamp-based hours (much more accurate than ping counts)
    result["ign_h"]    = result["ign_h_actual"]
    result["mov_h"]    = result["mov_h_actual"]
    result["idle_h"]   = result["idle_h_actual"]
    result["loaded_h"] = result["loaded_h_actual"]
    result["empty_h"]  = result["empty_h_actual"]
    # Also keep ping-count estimates as fallback features
    AVG_INTERVAL_H = 30 / 3600.0
    result["ign_h_est"]    = result["n_ign_pings"] * AVG_INTERVAL_H
    result["mov_h_est"]    = result["n_moving_pings"] * AVG_INTERVAL_H
    result["dump_h_est"]   = result["dump_pings_active"] * AVG_INTERVAL_H

    # Fractions (use actual time)
    result["idle_fraction"]    = result["idle_h"]   / (result["ign_h"] + 1e-6)
    result["work_fraction"]    = result["mov_h"]    / (result["ign_h"] + 1e-6)
    result["loaded_fraction"]  = result["loaded_h"] / (result["mov_h"] + 1e-6)
    result["empty_fraction"]   = result["empty_h"]  / (result["mov_h"] + 1e-6)

    # Speed features
    result["iqr_speed"] = result["speed_p75"] - result["speed_p25"]
    result["speed_cv"]  = result["speed_std"] / (result["speed_mean"] + 1e-6)
    result["km_per_h"]  = result["cumdist_km"] / (result["ign_h"] + 1e-6)

    # Altitude
    result["net_lift"]     = result["altitude_gain_m"] - result["altitude_loss_m"]
    result["climb_per_km"] = result["altitude_gain_m"] / (result["cumdist_km"] + 1e-6)

    # External voltage cross features (physics energy proxy)
    # Using actual time-based hours → much more accurate than ping count
    result["ext_v_x_mov_h"]   = result["ext_v_mean"] * result["mov_h"]
    result["ext_v_x_ign_h"]   = result["ext_v_mean"] * result["ign_h"]
    result["ext_v_per_km"]    = result["ext_v_mean"] / (result["cumdist_km"] + 1e-6)

    # Load-state external voltage (sum / actual loaded hours)
    result["loaded_ext_v_mean"] = result["loaded_ext_v_sum"] / (result["loaded_h"] + 1e-6)
    result["empty_ext_v_mean"]  = result["empty_ext_v_sum"]  / (result["empty_h"]  + 1e-6)
    # The revolutionary feature: ext_v × hours split by load state
    result["loaded_ext_v_x_h"] = result["loaded_ext_v_mean"] * result["loaded_h"]
    result["empty_ext_v_x_h"]  = result["empty_ext_v_mean"]  * result["empty_h"]

    # Telemetry-derived cycle count (REPLACES DPR total_trip — valid for test!)
    result["telemetry_cycles"] = np.maximum(result["n_loading_events"], result["n_dump_entries"])
    result["telemetry_cycles_min"] = np.minimum(result["n_loading_events"], result["n_dump_entries"])
    result["cycle_balance"]   = result["n_dump_entries"] / (result["n_loading_events"] + 1e-6)
    result["km_per_cycle"]    = result["cumdist_km"] / (result["telemetry_cycles"] + 1e-6)
    result["h_per_cycle"]     = result["ign_h"] / (result["telemetry_cycles"] + 1e-6)

    # Physics: loaded climb work = altitude gain × haul distance while loaded
    result["loaded_alt_gain_m"] = result["loaded_alt_gain_m"]  # already computed in agg
    result["loaded_km"]    = result["loaded_fraction"] * result["cumdist_km"]
    result["empty_km"]     = result["empty_fraction"]  * result["cumdist_km"]

    # Dump switch features (mine001 Feb17+; 0 for mine002 and early mine001)
    result["has_dump_signal"] = (result["dump_events"] > 0).astype(float)

    # GPS quality
    if "n_satellites" not in result.columns:
        result["n_satellites"] = 0.0

    print(f"  {label}: shape {result.shape}")
    nz = (result["telemetry_cycles"] > 0).sum()
    print(f"  telemetry_cycles non-zero: {nz}/{len(result)} ({nz/len(result):.1%})")
    nz2 = (result["loaded_alt_gain_m"] > 0).sum()
    print(f"  loaded_alt_gain_m non-zero: {nz2}/{len(result)} ({nz2/len(result):.1%})")

    return result


# ── Run extraction ────────────────────────────────────────────────────────────
print("\n=== Extracting v11 features: TRAIN ===")
train_feats = extract_features(TRAIN_FILES, "train")

print("\n=== Extracting v11 features: TEST ===")
test_feats = extract_features(TEST_FILES, "test")

# ── Fleet metadata (no parquet needed) ───────────────────────────────────────
fleet = pd.read_csv(DATA / "fleet.csv")
fleet_d = fleet[fleet["fleet"] == "Dumper"][["vehicle", "mine_anon", "dump_switch", "tankcap"]].copy()
fleet_d["mine_enc"] = (fleet_d["mine_anon"] == "mine002").astype(int)
fleet_d["has_dump_switch"] = fleet_d["dump_switch"].fillna(0).astype(int)
fleet_feat = fleet_d[["vehicle", "mine_enc", "has_dump_switch", "tankcap"]]

def safe_merge(left, right, on):
    keys = on if isinstance(on, list) else [on]
    drop = [c for c in right.columns if c not in keys and c in left.columns]
    return left.merge(right.drop(columns=drop), on=on, how="left")

train_feats = safe_merge(train_feats, fleet_feat, on="vehicle")
test_feats  = safe_merge(test_feats,  fleet_feat, on="vehicle")

# ── Temporal features (from date + shift) ────────────────────────────────────
SHIFT_ENC = {"C": 0, "A": 1, "B": 2}
for df in (train_feats, test_feats):
    d = pd.to_datetime(df["date"])
    df["shift_enc"]     = df["shift"].map(SHIFT_ENC).fillna(1).astype(int)
    df["day_of_week"]   = d.dt.dayofweek
    df["day_of_month"]  = d.dt.day
    df["week_number"]   = d.dt.isocalendar().week.astype(int)
    df["month"]         = d.dt.month
    df["is_weekend"]    = (d.dt.dayofweek >= 5).astype(int)
    df["hour_start"]    = [{"C": 22, "A": 6, "B": 14}.get(s, 6) for s in df["shift"]]
    df["hour_end"]      = [{"C": 5,  "A": 13, "B": 21}.get(s, 13) for s in df["shift"]]

# ── Fill NaN in numeric cols ──────────────────────────────────────────────────
numeric_fill_zero = [
    "n_loading_events", "n_dump_entries", "n_dump_exits", "telemetry_cycles",
    "loaded_alt_gain_m", "empty_alt_gain_m", "loaded_ext_v_sum", "empty_ext_v_sum",
    "loaded_ext_v_mean", "empty_ext_v_mean", "loaded_ext_v_x_mov_h", "empty_ext_v_x_mov_h",
    "frac_dump_zone", "frac_haul_road", "frac_loading",
    "dump_events", "dump_pings_active", "dump_h_est",
    "loaded_climb_work", "loaded_km", "empty_km", "loaded_fraction", "empty_fraction",
    "accel_std", "accel_mean",
]
for col in numeric_fill_zero:
    for df in (train_feats, test_feats):
        if col in df.columns:
            df[col] = df[col].fillna(0)

# ── Save ──────────────────────────────────────────────────────────────────────
train_feats.to_parquet(CKPT / "train_v11.parquet", index=False)
test_feats.to_parquet( CKPT / "test_v11.parquet",  index=False)
print(f"\nSaved: ckpts/train_v11.parquet {train_feats.shape}")
print(f"Saved: ckpts/test_v11.parquet  {test_feats.shape}")
print(f"Train features: {list(train_feats.columns)}")
