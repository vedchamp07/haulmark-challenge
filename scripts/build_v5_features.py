#!/usr/bin/env python3
"""
v5 feature patch — builds on v4 checkpoints (~8-12 min).

New features added:
1. Loaded/empty state machine per ping
   (excavator proximity → LOADED; dump zone / analog_input_1>2.5V → EMPTY)
2. Per-state aggregates: frac_loaded_moving, frac_empty_moving,
   loaded/empty ext_v means, n_loading_events, n_dump_events
3. Altitude features: altitude_gain_m, altitude_loss_m,
   loaded_alt_gain_m, empty_alt_gain_m (corr ~0.39 with acons)
4. Operator ID per shift (for fold-safe target encoding in train_v5.py)
5. Heading change (angle_change_mean)

Drops broken zero-gain v4 stub features:
  dump_zone_h, haul_road_h, dump_zone_enters, haul_road_enters,
  load_zone_h, loaded_km_est, empty_km_est

Saves: ckpts/train_v5.parquet, ckpts/test_v5.parquet
"""
import datetime, gc
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.ops import polygonize, unary_union

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "ckpts"

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

NEEDED_COLS = [
    "vehicle", "ts", "latitude", "longitude", "altitude",
    "speed", "mine_anon", "external_voltage", "angle", "operator_id",
]
# analog_input_1 loaded separately (not in all parquets cleanly)

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

    ob  = load_layer("ob_dump")
    stk = load_layer("mineral_stock")
    hr  = load_layer("haul_road")

    mine_zones[mine_id] = {
        "ob_dump":   make_zone(ob,  50),
        "stock":     make_zone(stk, 50),
        "haul_road": hr.geometry.buffer(25).union_all() if hr is not None and len(hr) > 0 else None,
    }
    dump_parts = [z for z in [mine_zones[mine_id]["ob_dump"], mine_zones[mine_id]["stock"]] if z is not None]
    mine_zones[mine_id]["any_dump"] = unary_union(dump_parts) if dump_parts else None
    print(f"  {mine_id}: ob_dump={mine_zones[mine_id]['ob_dump'] is not None}, "
          f"stock={mine_zones[mine_id]['stock'] is not None}, "
          f"haul={mine_zones[mine_id]['haul_road'] is not None}")


# ── Extract excavator mean positions (stable anchors for loading detection) ───
print("\nExtracting excavator positions from all parquets...")
exc_frames = []
for fname in ALL_FILES:
    path = DATA / fname
    try:
        df = pd.read_parquet(path, columns=["vehicle", "mine_anon", "latitude", "longitude"])
        exc = df[df["vehicle"].str.startswith("Exc")].copy()
        if len(exc) > 0:
            exc_frames.append(exc)
    except Exception as e:
        print(f"  Warning reading {fname}: {e}")

exc_all = pd.concat(exc_frames, ignore_index=True)
exc_mean = exc_all.groupby(["vehicle", "mine_anon"]).agg(
    lat=("latitude", "mean"), lon=("longitude", "mean")
).reset_index()
ex_x, ex_y = TRANSFORMER.transform(exc_mean["lon"].values, exc_mean["lat"].values)
exc_mean["x_utm"] = ex_x
exc_mean["y_utm"] = ex_y

exc_positions_utm = {}
for mine_id in ["mine001", "mine002"]:
    sub = exc_mean[exc_mean["mine_anon"] == mine_id]
    exc_positions_utm[mine_id] = sub[["x_utm", "y_utm"]].values if len(sub) > 0 else None
    print(f"  {mine_id}: {len(sub)} excavators found")


# ── Shift assignment ──────────────────────────────────────────────────────────
def assign_shift(ts_series: pd.Series):
    ts = pd.to_datetime(ts_series, utc=False)
    hour = ts.dt.hour
    date = ts.dt.date
    adj = [d - datetime.timedelta(days=1) if h >= 22 else d for d, h in zip(date, hour)]
    shift = ["C" if h >= 22 or h < 6 else ("A" if h < 14 else "B") for h in hour]
    return pd.Series(adj, index=ts_series.index), pd.Series(shift, index=ts_series.index)


# ── State machine per vehicle ─────────────────────────────────────────────────
def run_state_machine(group: pd.DataFrame) -> np.ndarray:
    """
    Given a vehicle's pings sorted by ts, returns loaded_state array:
      1.0 = LOADED (left loading zone, before dump)
      0.0 = EMPTY  (after dump, or default)
    """
    at_l = group["at_loading"].values
    at_d = group["at_dump"].values
    is_d = group["is_dumping"].values

    # Transition events
    just_left_loading = (~at_l) & np.concatenate([[False], at_l[:-1]])
    now_empty = at_d | is_d

    # Build state series with known transitions; NaN = unknown
    state = pd.Series(np.nan, index=range(len(group)))
    state[just_left_loading] = 1.0   # just became loaded
    state[now_empty]         = 0.0   # just became empty
    state[at_l]              = np.nan  # in loading zone: transitioning

    # Forward fill past known events, then backward fill start
    return state.ffill().bfill().fillna(0.0).values


# ── Process each parquet ──────────────────────────────────────────────────────
def extract_v5_features(file_list: list, label: str) -> pd.DataFrame:
    parts = []
    for fname in file_list:
        path = DATA / fname
        print(f"  {fname}...")

        # Load needed columns, gracefully handling missing ones
        available = pd.read_parquet(path, columns=["vehicle"]).index  # dummy read for schema
        try:
            available_cols = pd.read_parquet(path).columns.tolist()
        except Exception:
            available_cols = NEEDED_COLS
        use_cols = [c for c in NEEDED_COLS if c in available_cols]
        if "analog_input_1" in available_cols:
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
        for mine_id, zones in mine_zones.items():
            mask = df["mine_anon"] == mine_id
            if mask.sum() == 0:
                continue
            pts = gpd.GeoSeries(
                gpd.points_from_xy(df.loc[mask, "x_utm"].values, df.loc[mask, "y_utm"].values),
                crs="EPSG:32645"
            )
            if zones["any_dump"] is not None:
                df.loc[mask, "at_dump"] = pts.within(zones["any_dump"]).values

        # ── Excavator proximity flags ─────────────────────────────────────────
        df["at_loading"] = False
        for mine_id, exc_xy in exc_positions_utm.items():
            if exc_xy is None or len(exc_xy) == 0:
                continue
            mask = df["mine_anon"] == mine_id
            if mask.sum() == 0:
                continue
            x_m = df.loc[mask, "x_utm"].values
            y_m = df.loc[mask, "y_utm"].values
            # Vectorized distance to all excavators in this mine
            dists = np.sqrt(
                ((x_m[:, None] - exc_xy[:, 0]) ** 2 +
                 (y_m[:, None] - exc_xy[:, 1]) ** 2)
            )
            df.loc[mask, "at_loading"] = dists.min(axis=1) < 150.0

        # Dump switch signal (mine001 Feb17+; sparse otherwise)
        if "analog_input_1" in df.columns:
            df["is_dumping"] = df["analog_input_1"].fillna(0) > 2.5
        else:
            df["is_dumping"] = False

        # ── State machine per vehicle ─────────────────────────────────────────
        state_parts = []
        for veh, vdf in df.groupby("vehicle", sort=False):
            vdf = vdf.sort_values("ts")
            st = run_state_machine(vdf)
            state_parts.append(pd.Series(st, index=vdf.index))
        df["loaded_state"] = pd.concat(state_parts).sort_index()

        # ── Derived per-ping signals ──────────────────────────────────────────
        df["is_moving"] = df["speed"] > 1.0
        df["loaded_moving"] = (df["loaded_state"] == 1.0) & df["is_moving"]
        df["empty_moving"]  = (df["loaded_state"] == 0.0) & df["is_moving"]

        # Altitude diffs (per vehicle, clip extremes)
        df["d_alt"] = df.groupby("vehicle")["altitude"].transform(
            lambda x: x.diff().fillna(0).clip(-100, 100)
        )
        df["alt_gain"]         = df["d_alt"].clip(lower=0)
        df["alt_loss"]         = df["d_alt"].clip(upper=0)
        df["loaded_alt_gain"]  = df["alt_gain"]  * df["loaded_moving"].astype(float)
        df["loaded_alt_loss"]  = df["alt_loss"]  * df["loaded_moving"].astype(float)
        df["empty_alt_gain"]   = df["alt_gain"]  * df["empty_moving"].astype(float)

        # External voltage × load state (for later: loaded_ext_v_x_h)
        ev = df["external_voltage"].fillna(0).values
        df["loaded_ext_v"] = ev * df["loaded_moving"].astype(float)
        df["empty_ext_v"]  = ev * df["empty_moving"].astype(float)

        # Transition events (for counting haul cycles from state machine)
        df["loading_exit"] = df.groupby("vehicle")["at_loading"].transform(
            lambda x: (~x) & x.shift(1, fill_value=False)
        )
        df["dump_entry"] = df.groupby("vehicle")["at_dump"].transform(
            lambda x: x & (~x.shift(1, fill_value=False))
        )

        # Heading change
        if "angle" in df.columns:
            df["d_angle"] = df.groupby("vehicle")["angle"].transform(
                lambda x: x.diff().abs().fillna(0).clip(0, 180)
            )
        else:
            df["d_angle"] = 0.0

        # ── Aggregate to shift level ──────────────────────────────────────────
        key = ["vehicle", "adj_date", "shift"]
        agg = df.groupby(key).agg(
            altitude_gain_m   = ("alt_gain",       "sum"),
            altitude_loss_m   = ("alt_loss",        "sum"),
            loaded_alt_gain_m = ("loaded_alt_gain", "sum"),
            loaded_alt_loss_m = ("loaded_alt_loss", "sum"),
            empty_alt_gain_m  = ("empty_alt_gain",  "sum"),
            frac_loaded_moving= ("loaded_moving",   "mean"),
            frac_empty_moving = ("empty_moving",    "mean"),
            n_loading_events  = ("loading_exit",    "sum"),
            n_dump_events     = ("dump_entry",      "sum"),
            loaded_ext_v_sum  = ("loaded_ext_v",    "sum"),
            loaded_n_pings    = ("loaded_moving",   "sum"),
            empty_ext_v_sum   = ("empty_ext_v",     "sum"),
            empty_n_pings     = ("empty_moving",    "sum"),
            angle_change_mean = ("d_angle",         "mean"),
        ).reset_index()

        # Operator ID: most common non-null per shift
        if "operator_id" in df.columns:
            op = (df.dropna(subset=["operator_id"])
                  .groupby(key)["operator_id"]
                  .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
                  .reset_index()
                  .rename(columns={"operator_id": "operator_id_shift"}))
            agg = agg.merge(op, on=key, how="left")
        else:
            agg["operator_id_shift"] = np.nan

        agg.rename(columns={"adj_date": "date"}, inplace=True)
        agg["date"] = pd.to_datetime(agg["date"]).dt.date

        parts.append(agg)
        gc.collect()

    result = pd.concat(parts, ignore_index=True)

    # Derived means (after concat to get global stats)
    result["loaded_ext_v_mean"] = result["loaded_ext_v_sum"] / (result["loaded_n_pings"] + 1e-6)
    result["empty_ext_v_mean"]  = result["empty_ext_v_sum"]  / (result["empty_n_pings"] + 1e-6)

    print(f"  {label}: v5 features shape {result.shape}")
    nz = (result["altitude_gain_m"] > 0).sum()
    print(f"  altitude_gain_m non-zero: {nz}/{len(result)} ({nz/len(result):.1%})")
    nz2 = (result["n_loading_events"] > 0).sum()
    print(f"  n_loading_events non-zero: {nz2}/{len(result)} ({nz2/len(result):.1%})")
    nz3 = (result["frac_loaded_moving"] > 0).sum()
    print(f"  frac_loaded_moving non-zero: {nz3}/{len(result)} ({nz3/len(result):.1%})")
    return result


# ── Run extraction ────────────────────────────────────────────────────────────
print("\n=== Extracting v5 features: TRAIN ===")
train_v5_patch = extract_v5_features(TRAIN_FILES, "train")
print("\n=== Extracting v5 features: TEST ===")
test_v5_patch  = extract_v5_features(TEST_FILES,  "test")

# ── Load v4 checkpoints ───────────────────────────────────────────────────────
print("\n=== Loading v4 checkpoints ===")
train = pd.read_parquet(CKPT / "train_v4.parquet")
test  = pd.read_parquet(CKPT / "test_v4.parquet")
train["date"] = pd.to_datetime(train["date"]).dt.date
test["date"]  = pd.to_datetime(test["date"]).dt.date
print(f"  train_v4: {train.shape}, test_v4: {test.shape}")

# Drop broken zero-gain stub features from v4
DROP_BROKEN = [
    "dump_zone_h", "haul_road_h", "dump_zone_enters",
    "haul_road_enters", "load_zone_h", "loaded_km_est", "empty_km_est",
]
for df_ in [train, test]:
    cols_to_drop = [c for c in DROP_BROKEN if c in df_.columns]
    if cols_to_drop:
        df_.drop(columns=cols_to_drop, inplace=True)
print(f"  Dropped broken features: {DROP_BROKEN}")

# ── Merge v5 patch features ───────────────────────────────────────────────────
def safe_merge(left, right, on):
    keys = on if isinstance(on, list) else [on]
    drop = [c for c in right.columns if c not in keys and c in left.columns]
    return left.merge(right.drop(columns=drop), on=on, how="left")

train = safe_merge(train, train_v5_patch, on=["vehicle", "date", "shift"])
test  = safe_merge(test,  test_v5_patch,  on=["vehicle", "date", "shift"])

# Fill NaN in new numerical cols
NEW_COLS = [
    "altitude_gain_m", "altitude_loss_m", "loaded_alt_gain_m",
    "loaded_alt_loss_m", "empty_alt_gain_m",
    "frac_loaded_moving", "frac_empty_moving",
    "n_loading_events", "n_dump_events",
    "loaded_ext_v_mean", "empty_ext_v_mean",
    "angle_change_mean",
]
for col in NEW_COLS:
    for df_ in [train, test]:
        if col in df_.columns:
            df_[col] = df_[col].fillna(0)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== v5 feature summary (train) ===")
for col in ["altitude_gain_m", "loaded_alt_gain_m", "frac_loaded_moving",
            "frac_empty_moving", "n_loading_events", "loaded_ext_v_mean"]:
    if col in train.columns:
        nz = (train[col] > 0).sum()
        print(f"  {col}: non-zero {nz}/{len(train)} ({nz/len(train):.1%}), "
              f"mean={train[col].mean():.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
train.to_parquet(CKPT / "train_v5.parquet", index=False)
test.to_parquet( CKPT / "test_v5.parquet",  index=False)
print(f"\nSaved: ckpts/train_v5.parquet {train.shape}")
print(f"Saved: ckpts/test_v5.parquet  {test.shape}")
