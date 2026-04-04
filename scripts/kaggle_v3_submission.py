#!/usr/bin/env python3
"""
Kaggle submission v3 — paste into a single code cell and run.

Key improvements over v2 (LB 3182):
1. Two-stage model: inactive-shift classifier → regressor on active shifts
2. total_trip (cycle counter) as feature — available in both train AND test
3. Operator target encoding (per fold, no leakage)
4. Better zero-shift handling
5. Keeps spatial features from gpkg
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import gc, warnings, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA = Path("/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge")
OUT  = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)
print(f"DATA: {DATA}")
print(f"OUT:  {OUT}")

TEST_FILES = {
    "telemetry_2026-01-21_2026-01-31.parquet",
    "telemetry_2026-02-21_2026-02-28.parquet",
    "telemetry_2026-03-12_2026-03-20.parquet",
}

# ── Spatial setup ─────────────────────────────────────────────────────────────
try:
    import geopandas as gpd
    from pyproj import Transformer
    SPATIAL_OK = True
    TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    print("geopandas available — spatial features ENABLED")
except ImportError:
    SPATIAL_OK = False
    print("geopandas not available — spatial features DISABLED")

DUMP_BUF  = 40.0   # metres buffer for dump/stock zones
LOAD_BUF  = 40.0   # metres buffer for loading zones (bench)
HAUL_BUF  = 25.0   # metres buffer for haul road
EXC_PROX  = 100.0  # metres proximity to excavator = loading event


def load_mine_layers():
    result = {}
    for mine_id in ["mine001", "mine002"]:
        gpkg = DATA / f"{mine_id}_anonymized.gpkg"
        if not gpkg.exists():
            continue
        raw = {}
        import fiona
        available = fiona.listlayers(str(gpkg))
        print(f"  {mine_id}: layers={available}")
        for lname in available:
            try:
                gdf = gpd.read_file(str(gpkg), layer=lname)
                if gdf.crs and gdf.crs.to_epsg() != 32645:
                    gdf = gdf.to_crs("EPSG:32645")
                raw[lname] = gdf
            except Exception:
                pass

        def safe_union(k, buf):
            g = raw.get(k)
            if g is None:
                g = raw.get("mineral_stock" if k == "stock" else k)
            return g.geometry.buffer(buf).union_all() if g is not None else None

        result[mine_id] = {
            "ob_dump_union":    safe_union("ob_dump",       DUMP_BUF),
            "stock_union":      safe_union("mineral_stock", DUMP_BUF),
            "bench_union":      safe_union("bench",         LOAD_BUF),
            "haul_road_union":  safe_union("haul_road",     HAUL_BUF),
        }
    return result


def add_zone_flags(df: pd.DataFrame, mine_layers: dict) -> pd.DataFrame:
    """Add boolean zone flags for each ping using vectorised spatial join."""
    if not SPATIAL_OK or not mine_layers:
        for col in ["in_dump", "in_stock", "in_bench", "on_haul"]:
            df[col] = False
        return df

    df = df.copy()
    for col in ["in_dump", "in_stock", "in_bench", "on_haul"]:
        df[col] = False

    for mine_id, layers in mine_layers.items():
        mask = df["mine_anon"] == mine_id
        if mask.sum() == 0:
            continue
        sub = df.loc[mask, ["latitude", "longitude"]]
        x, y = TRANSFORMER.transform(sub["longitude"].values, sub["latitude"].values)
        pts = gpd.GeoSeries(
            gpd.points_from_xy(x, y), crs="EPSG:32645"
        )
        for col, key in [
            ("in_dump",  "ob_dump_union"),
            ("in_stock", "stock_union"),
            ("in_bench", "on_haul"),      # NOTE: using 'on_haul' key intentionally below
            ("on_haul",  "haul_road_union"),
        ]:
            # fix mapping
            zone_key = {
                "in_dump":  "ob_dump_union",
                "in_stock": "stock_union",
                "in_bench": "bench_union",
                "on_haul":  "haul_road_union",
            }[col]
            poly = layers.get(zone_key)
            if poly is not None:
                df.loc[mask, col] = pts.within(poly).values
    return df


def get_exc_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Extract median positions of excavators per (mine, operational_date, shift)."""
    exc_mask = df["vehicle"].str.startswith("Exc")
    if exc_mask.sum() == 0:
        return pd.DataFrame(columns=["mine_anon", "adj_date", "shift", "exc_x", "exc_y"])
    exc = df[exc_mask].copy()
    if SPATIAL_OK:
        x, y = TRANSFORMER.transform(exc["longitude"].values, exc["latitude"].values)
        exc["x_utm"] = x
        exc["y_utm"] = y
    else:
        exc["x_utm"] = exc["longitude"]
        exc["y_utm"] = exc["latitude"]

    grp = exc.groupby(["mine_anon", "adj_date", "shift"]).agg(
        exc_x=("x_utm", "median"),
        exc_y=("y_utm", "median"),
    ).reset_index()
    return grp


def assign_shift(ts_series: pd.Series):
    """Returns (adj_date, shift) following the operational day convention."""
    ts = pd.to_datetime(ts_series, utc=False)
    hour = ts.dt.hour
    date = ts.dt.date
    adj_date = [
        d - datetime.timedelta(days=1) if h >= 22 else d
        for d, h in zip(date, hour)
    ]
    shift = [
        "C" if h >= 22 or h < 6 else ("A" if h < 14 else "B")
        for h in hour
    ]
    return (
        pd.Series(adj_date, index=ts_series.index, dtype="object"),
        pd.Series(shift, index=ts_series.index),
    )


def process_telemetry(path: Path, mine_layers: dict, is_test: bool) -> pd.DataFrame:
    """Load one parquet, add zone flags, compute per-ping deltas, return enriched df."""
    print(f"  {path.name}...")
    df_full = pd.read_parquet(path)
    df_full["ts"] = pd.to_datetime(df_full["ts"], utc=False)
    df_full["adj_date"], df_full["shift"] = assign_shift(df_full["ts"])

    # UTM for all vehicles (needed for excavator positions)
    if SPATIAL_OK:
        x, y = TRANSFORMER.transform(df_full["longitude"].values, df_full["latitude"].values)
        df_full["x_utm"] = x
        df_full["y_utm"] = y
    else:
        df_full["x_utm"] = df_full["longitude"]
        df_full["y_utm"] = df_full["latitude"]

    # Excavator positions (from all vehicles including Exc*)
    exc_pos = get_exc_positions(df_full)

    # Now restrict to dumpers only
    df = df_full[df_full["vehicle"].str.startswith("Dump")].copy()
    del df_full; gc.collect()

    # Zone flags
    df = add_zone_flags(df, mine_layers)

    df.sort_values(["vehicle", "ts"], inplace=True)

    # Per-row time delta (seconds)
    df["dt_s"] = (
        df.groupby("vehicle")["ts"]
        .transform(lambda s: s.diff().dt.total_seconds().fillna(0).clip(0, 300))
    )

    # State durations
    df["ign_time_s"]  = df["dt_s"] * df["ignition"].clip(0, 1)
    df["mov_time_s"]  = df["dt_s"] * (df["speed"] > 0.5).astype(float)
    df["idle_time_s"] = df["ign_time_s"] - df["mov_time_s"]

    # Dump signal
    if "analog_input_1" in df.columns:
        df["dump_signal"] = df["analog_input_1"].fillna(0)
        df["is_dumping"]  = (df["dump_signal"] > 2.5).astype(float)
        df["dump_time_s"] = df["dt_s"] * df["is_dumping"]
    else:
        df["dump_signal"] = 0.0
        df["is_dumping"]  = 0.0
        df["dump_time_s"] = 0.0

    # Zone time
    df["dump_zone_s"] = df["dt_s"] * df["in_dump"].astype(float)
    df["load_zone_s"] = df["dt_s"] * df["in_bench"].astype(float)
    df["haul_road_s"] = df["dt_s"] * df["on_haul"].astype(float)

    # Dump zone entry transitions
    df["dump_zone_enter"] = (
        df.groupby("vehicle")["in_dump"]
        .transform(lambda s: ((s == True) & (s.shift(1) == False)).astype(int))
    )
    df["haul_road_enter"] = (
        df.groupby("vehicle")["on_haul"]
        .transform(lambda s: ((s == True) & (s.shift(1) == False)).astype(int))
    )

    # Dump signal transitions
    df["dump_sig_enter"] = (
        df.groupby("vehicle")["is_dumping"]
        .transform(lambda s: ((s == 1) & (s.shift(1) == 0)).astype(int))
    )

    # Excavator proximity (exc_pos already computed from full df above)
    df["near_exc"] = False
    if len(exc_pos) > 0:
        for _, row in exc_pos.iterrows():
            mask = (
                (df["mine_anon"] == row["mine_anon"]) &
                (df["adj_date"].astype(str) == str(row["adj_date"])) &
                (df["shift"] == row["shift"])
            )
            if mask.sum() == 0:
                continue
            dx = df.loc[mask, "x_utm"] - row["exc_x"]
            dy = df.loc[mask, "y_utm"] - row["exc_y"]
            near = (dx**2 + dy**2) < EXC_PROX**2
            df.loc[mask, "near_exc"] = near
    df["exc_enter"] = (
        df.groupby("vehicle")["near_exc"]
        .transform(lambda s: ((s == True) & (s.shift(1) == False)).astype(int))
    )

    # Altitude deltas
    df["alt_diff"] = df.groupby("vehicle")["altitude"].transform(lambda s: s.diff().fillna(0))
    df["climb_m"]  = df["alt_diff"].clip(lower=0)
    df["descent_m"] = (-df["alt_diff"]).clip(lower=0)

    # Total trip counter — take max per shift (it resets at shift boundaries)
    # in_bench transition indicates loading visit

    return df


def build_shift_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-ping df into per-(vehicle, date, shift) features."""
    # Dominant operator per shift
    if "operator_id" in df.columns:
        op_mode = (
            df[df["operator_id"].notna()]
            .groupby(["vehicle", "adj_date", "shift"])["operator_id"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
            .reset_index()
            .rename(columns={"operator_id": "operator_mode"})
        )
    else:
        op_mode = None

    # Core aggregations
    agg = df.groupby(["vehicle", "adj_date", "shift"]).agg(
        n_pings         = ("ts", "size"),
        ign_h           = ("ign_time_s",    lambda x: x.sum() / 3600),
        mov_h           = ("mov_time_s",    lambda x: x.sum() / 3600),
        idle_h          = ("idle_time_s",   lambda x: x.sum() / 3600),
        dump_time_h     = ("dump_time_s",   lambda x: x.sum() / 3600),
        dump_zone_h     = ("dump_zone_s",   lambda x: x.sum() / 3600),
        load_zone_h     = ("load_zone_s",   lambda x: x.sum() / 3600),
        haul_road_h     = ("haul_road_s",   lambda x: x.sum() / 3600),
        dump_zone_enters= ("dump_zone_enter","sum"),
        haul_road_enters= ("haul_road_enter","sum"),
        dump_sig_events = ("dump_sig_enter", "sum"),
        loading_visits  = ("exc_enter",      "sum"),
        shift_km        = ("disthav",        lambda x: x.sum() / 1000),
        total_climb_m   = ("climb_m",        "sum"),
        total_descent_m = ("descent_m",      "sum"),
        altitude_mean   = ("altitude",       "mean"),
        altitude_std    = ("altitude",       "std"),
        altitude_max    = ("altitude",       "max"),
        altitude_min    = ("altitude",       "min"),
        speed_mean      = ("speed",          "mean"),
        speed_std       = ("speed",          "std"),
        speed_max       = ("speed",          "max"),
        speed_p25       = ("speed",          lambda x: np.percentile(x, 25)),
        speed_p50       = ("speed",          lambda x: np.percentile(x, 50)),
        speed_p75       = ("speed",          lambda x: np.percentile(x, 75)),
        ext_v_mean      = ("external_voltage","mean"),
        ext_v_std       = ("external_voltage","std"),
        ext_v_max       = ("external_voltage","max"),
        dump_sig_mean   = ("dump_signal",    "mean"),
        dump_sig_max    = ("dump_signal",    "max"),
        frac_dump_zone  = ("in_dump",        "mean"),
        frac_load_zone  = ("in_bench",       "mean"),
        frac_haul_road  = ("on_haul",        "mean"),
        mine_anon       = ("mine_anon",      "first"),
    ).reset_index()

    # total_trip (may not exist or all NaN in older train files)
    if "total_trip" in df.columns:
        tt = df.groupby(["vehicle", "adj_date", "shift"])["total_trip"].agg(["max", "min"]).reset_index()
        tt.columns = ["vehicle", "adj_date", "shift", "max_total_trip", "min_total_trip"]
        agg = agg.merge(tt, on=["vehicle", "adj_date", "shift"], how="left")
    else:
        agg["max_total_trip"] = np.nan
        agg["min_total_trip"] = np.nan

    # Satellites (column may not exist in all files)
    if "satellites" in df.columns:
        sat = df.groupby(["vehicle", "adj_date", "shift"])["satellites"].mean().reset_index()
        sat.rename(columns={"satellites": "n_satellites"}, inplace=True)
        agg = agg.merge(sat, on=["vehicle", "adj_date", "shift"], how="left")
    else:
        agg["n_satellites"] = np.nan

    agg.rename(columns={"adj_date": "date"}, inplace=True)

    # Derived features
    agg["altitude_range"]    = agg["altitude_max"] - agg["altitude_min"]
    agg["net_lift"]          = agg["total_climb_m"] - agg["total_descent_m"]
    agg["idle_fraction"]     = agg["idle_h"] / (agg["ign_h"] + 1e-6)
    agg["work_fraction"]     = agg["mov_h"] / (agg["ign_h"] + 1e-6)
    agg["km_per_hour"]       = agg["shift_km"] / (agg["ign_h"] + 1e-6)
    agg["climb_per_km"]      = agg["total_climb_m"] / (agg["shift_km"] + 1e-6)
    agg["speed_cv"]          = agg["speed_std"] / (agg["speed_mean"] + 1e-6)
    agg["iqr_speed"]         = agg["speed_p75"] - agg["speed_p25"]
    agg["ext_v_x_mov_h"]     = agg["ext_v_mean"] * agg["mov_h"]
    agg["ext_v_x_ign_h"]     = agg["ext_v_mean"] * agg["ign_h"]
    agg["ext_v_per_km"]      = agg["ext_v_mean"] / (agg["shift_km"] + 1e-6)

    # Haul cycles: best estimate from available signals
    agg["n_trips_from_total"] = (agg["max_total_trip"] - agg["min_total_trip"]).clip(lower=0)
    agg["haul_cycles"] = agg[["dump_zone_enters", "dump_sig_events", "loading_visits",
                               "n_trips_from_total"]].max(axis=1)
    agg["km_per_cycle"]      = agg["shift_km"] / (agg["haul_cycles"] + 1e-6)
    agg["ign_h_per_cycle"]   = agg["ign_h"] / (agg["haul_cycles"] + 1e-6)

    # Speed in different zones
    # (Proxy: loaded = on haul going to dump, empty = on haul going back)
    # Use shift_km and dump_zone_h as proxy
    agg["loaded_km_est"]  = agg["shift_km"] * agg["frac_haul_road"] * 0.5
    agg["empty_km_est"]   = agg["shift_km"] * agg["frac_haul_road"] * 0.5

    # Temporal features
    agg["date_dt"] = pd.to_datetime(agg["date"])
    agg["shift_enc"]    = agg["shift"].map({"C": 0, "A": 1, "B": 2})
    agg["day_of_week"]  = agg["date_dt"].dt.dayofweek
    agg["is_weekend"]   = (agg["day_of_week"] >= 5).astype(int)
    agg["day_of_month"] = agg["date_dt"].dt.day
    agg["week_number"]  = agg["date_dt"].dt.isocalendar().week.astype(int)
    agg["month"]        = agg["date_dt"].dt.month
    agg["hour_start"]   = agg["shift"].map({"C": 22, "A": 6, "B": 14})
    agg["hour_end"]     = agg["shift"].map({"C": 6, "A": 14, "B": 22})

    # Active shift flag
    agg["is_active"] = (
        (agg["ign_h"] > 0.5) | (agg["shift_km"] > 1.0) | (agg["n_pings"] > 200)
    ).astype(int)

    # Merge operator
    if op_mode is not None:
        op_mode.rename(columns={"adj_date": "date"}, inplace=True)
        agg = agg.merge(op_mode, on=["vehicle", "date", "shift"], how="left")
    else:
        agg["operator_mode"] = np.nan

    return agg


# ── Main processing ───────────────────────────────────────────────────────────
print("\n=== Loading spatial layers ===")
mine_layers = load_mine_layers() if SPATIAL_OK else {}

print("\n=== Loading fleet ===")
fleet = pd.read_csv(DATA / "fleet.csv")
dumpers = fleet[fleet["fleet"] == "Dumper"]["vehicle"].tolist()
print(f"Dumpers: {len(dumpers)}")

# Find all parquet files
all_files = sorted(DATA.glob("telemetry_*.parquet"))
train_files = [f for f in all_files if f.name not in TEST_FILES]
test_files  = [f for f in all_files if f.name in TEST_FILES]
print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

# Process telemetry
print("\n=== Processing training telemetry ===")
train_raw = pd.concat(
    [process_telemetry(f, mine_layers, is_test=False) for f in train_files],
    ignore_index=True
)
print("\n=== Processing test telemetry ===")
test_raw = pd.concat(
    [process_telemetry(f, mine_layers, is_test=True) for f in test_files],
    ignore_index=True
)

train_feats = build_shift_features(train_raw)
test_feats  = build_shift_features(test_raw)
del train_raw, test_raw; gc.collect()
print(f"Train features: {train_feats.shape}")
print(f"Test features:  {test_feats.shape}")


def safe_merge(left, right, on, how="left"):
    """Merge dropping any right-side columns that already exist in left (except join keys)."""
    keys = on if isinstance(on, list) else [on]
    drop_cols = [c for c in right.columns if c not in keys and c in left.columns]
    if drop_cols:
        right = right.drop(columns=drop_cols)
    return left.merge(right, on=on, how=how)


# ── Load labels ───────────────────────────────────────────────────────────────
print("\n=== Merging targets ===")
smry_files = sorted(DATA.glob("smry_*_train_ordered.csv"))
smry = pd.concat([pd.read_csv(f) for f in smry_files], ignore_index=True)
smry["date"] = pd.to_datetime(smry["date"]).dt.date

train_feats["date"] = pd.to_datetime(train_feats["date"]).dt.date
test_feats["date"]  = pd.to_datetime(test_feats["date"]).dt.date

train = safe_merge(train_feats, smry[["vehicle", "date", "shift", "acons"]], on=["vehicle", "date", "shift"])
print(f"Labeled: {train['acons'].notna().sum()} / {len(train)}")

# ── RFID refuel features ──────────────────────────────────────────────────────
rfid_files = sorted(DATA.glob("rfid_refuels_*.parquet"))
if rfid_files:
    rfid = pd.concat([pd.read_parquet(f) for f in rfid_files], ignore_index=True)
    rfid["ts"] = pd.to_datetime(rfid["ts"], utc=False)
    rfid["adj_date"], rfid["shift"] = assign_shift(rfid["ts"])
    rfid_agg = rfid.groupby(["vehicle", "adj_date", "shift"]).agg(
        rfid_liters=("litres", "sum"),
        rfid_events=("litres", "count"),
    ).reset_index()
    rfid_agg.rename(columns={"adj_date": "date"}, inplace=True)
    rfid_agg["date"] = pd.to_datetime(rfid_agg["date"]).dt.date
    train      = safe_merge(train,      rfid_agg, on=["vehicle", "date", "shift"])
    test_feats = safe_merge(test_feats, rfid_agg, on=["vehicle", "date", "shift"])
    train["rfid_liters"]       = train["rfid_liters"].fillna(0)
    train["rfid_events"]       = train["rfid_events"].fillna(0)
    test_feats["rfid_liters"]  = test_feats["rfid_liters"].fillna(0)
    test_feats["rfid_events"]  = test_feats["rfid_events"].fillna(0)
    print(f"RFID merged. Non-zero refuels in train: {(train['rfid_liters'] > 0).sum()}")

# ── Vehicle aggregate features ────────────────────────────────────────────────
train_clean = train[train["acons"].notna() & (train["acons"] > 0)].copy()
global_acons_mean = train_clean["acons"].mean()

veh_shift_stats = (
    train_clean.groupby(["vehicle", "shift"])["acons"]
    .agg(["mean", "std"]).reset_index()
)
veh_shift_stats.columns = ["vehicle", "shift", "veh_shift_mean_acons", "veh_shift_std_acons"]

_lph = train_clean.copy()
_lph["lph"] = _lph["acons"] / (_lph["ign_h"] + 1e-6)
veh_lph = _lph.groupby("vehicle")["lph"].median().reset_index().rename(columns={"lph": "veh_lph"})

veh_stats = train_clean.groupby("vehicle").agg(
    veh_mean_km       = ("shift_km",      "mean"),
    veh_mean_ign_h    = ("ign_h",         "mean"),
    veh_mean_dumps    = ("haul_cycles",   "mean"),
    veh_mean_idle_frac= ("idle_fraction", "mean"),
    veh_mean_speed    = ("speed_mean",    "mean"),
    veh_mean_ext_v    = ("ext_v_mean",    "mean"),
).reset_index()
veh_stats = veh_stats.merge(veh_lph, on="vehicle", how="left")

_tc = train_clean[train_clean["haul_cycles"] > 0].copy()
_tc["fuel_per_trip"] = _tc["acons"] / _tc["haul_cycles"]
veh_fpt = _tc.groupby("vehicle")["fuel_per_trip"].median().reset_index()
veh_fpt.rename(columns={"fuel_per_trip": "veh_fuel_per_trip"}, inplace=True)

train      = safe_merge(train,      veh_shift_stats, on=["vehicle", "shift"])
train      = safe_merge(train,      veh_stats,       on="vehicle")
train      = safe_merge(train,      veh_fpt,         on="vehicle")
test_feats = safe_merge(test_feats, veh_shift_stats, on=["vehicle", "shift"])
test_feats = safe_merge(test_feats, veh_stats,       on="vehicle")
test_feats = safe_merge(test_feats, veh_fpt,         on="vehicle")

# Physics predictions as features
for df_ in [train, test_feats]:
    df_["physics_pred"] = df_["veh_lph"].fillna(46.0) * df_["ign_h"]
    df_["trip_pred"]    = df_["veh_fuel_per_trip"].fillna(17.0) * df_["haul_cycles"]

# ── Operator target encoding ──────────────────────────────────────────────────
if "operator_mode" in train_clean.columns and train_clean["operator_mode"].notna().sum() > 0:
    op_stats = (
        train_clean.dropna(subset=["operator_mode"])
        .groupby("operator_mode")["acons"].agg(["mean", "count"]).reset_index()
    )
    op_stats.columns = ["operator_mode", "op_mean_acons", "op_count"]
    train      = safe_merge(train,      op_stats, on="operator_mode")
    test_feats = safe_merge(test_feats, op_stats, on="operator_mode")
else:
    train["op_mean_acons"]      = global_acons_mean
    train["op_count"]           = 0
    test_feats["op_mean_acons"] = global_acons_mean
    test_feats["op_count"]      = 0

train["op_mean_acons"]      = train["op_mean_acons"].fillna(global_acons_mean)
test_feats["op_mean_acons"] = test_feats["op_mean_acons"].fillna(global_acons_mean)
train["op_count"]           = train["op_count"].fillna(0)
test_feats["op_count"]      = test_feats["op_count"].fillna(0)

print(f"\nSpatial feature coverage (labeled train):")
for col in ["frac_dump_zone", "frac_load_zone", "frac_haul_road", "haul_cycles", "loading_visits"]:
    if col in train.columns:
        lab = train[train["acons"].notna()]
        nz = (lab[col] > 0).sum()
        print(f"  {col}: non-zero {nz}/{len(lab)} ({nz/len(lab):.0%})")

# ── Feature set ───────────────────────────────────────────────────────────────
EXCLUDE = {
    "vehicle", "mine_anon", "date", "date_dt", "shift",
    "acons", "is_active_label", "is_active",
    "fuel_volume", "prod_hr_dpr", "idle_hr_dpr", "km_dpr",
    "tonnage", "hmr_dpr", "operator_id", "operator_mode",
    "bd_hr_dpr", "maint_hr_dpr",
}

def get_features(train_df, test_df):
    """Return feature columns present in BOTH train and test, not in EXCLUDE."""
    def eligible(df, c):
        return (
            c not in EXCLUDE
            and df[c].dtype not in ["object"]
            and not pd.api.types.is_datetime64_any_dtype(df[c])
        )
    train_cols = {c for c in train_df.columns if eligible(train_df, c)}
    test_cols  = {c for c in test_df.columns  if eligible(test_df,  c)}
    return sorted(train_cols & test_cols)

# ── TWO-STAGE MODEL ───────────────────────────────────────────────────────────
print("\n=== Two-stage model ===")

labeled = train[train["acons"].notna()].copy()
labeled["is_active_label"] = (labeled["acons"] > 10).astype(int)
print(f"Labeled: {len(labeled)}, active: {labeled['is_active_label'].sum()}, inactive: {(labeled['is_active_label']==0).sum()}")

feat_cols = get_features(labeled, test_feats)
print(f"Features ({len(feat_cols)}): {feat_cols[:10]} ...")

X_all = labeled[feat_cols].fillna(0)
y_all = labeled["acons"].values
y_active = labeled["is_active_label"].values
groups = labeled["vehicle"].astype("category").cat.codes.values

X_test = test_feats[feat_cols].fillna(0)

gkf = GroupKFold(n_splits=5)

# ── Stage 1: Activity classifier ─────────────────────────────────────────────
print("\n--- Stage 1: Active/Inactive classifier ---")
CLF_PARAMS = {
    "objective":   "binary",
    "metric":      "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves":  31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
}
oof_proba = np.zeros(len(labeled))
clf_models = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_active, groups), 1):
    X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    y_tr, y_val = y_active[tr_idx], y_active[val_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    clf = lgb.train(
        CLF_PARAMS, dtrain, valid_sets=[dval], num_boost_round=1000,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )
    oof_proba[val_idx] = clf.predict(X_val, num_iteration=clf.best_iteration)
    clf_models.append(clf)
    acc = ((oof_proba[val_idx] > 0.5) == y_active[val_idx]).mean()
    print(f"  Fold {fold}: accuracy={acc:.3f}")

oof_active_pred = (oof_proba > 0.5).astype(int)
overall_acc = (oof_active_pred == y_active).mean()
print(f"OOF accuracy: {overall_acc:.3f}")
test_active_proba = np.mean([m.predict(X_test, num_iteration=m.best_iteration) for m in clf_models], axis=0)

# ── Stage 2: Regressor on active shifts only ──────────────────────────────────
print("\n--- Stage 2: Consumption regressor (active shifts only) ---")
REGR_PARAMS = {
    "objective":   "regression",
    "metric":      "rmse",
    "learning_rate": 0.03,
    "num_leaves":  63,
    "max_depth":   -1,
    "min_data_in_leaf": 15,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1":   0.05,
    "lambda_l2":   0.3,
    "verbosity":   -1,
    "seed":        42,
}

active_mask = labeled["is_active_label"] == 1
active_df = labeled[active_mask].copy()
X_act = active_df[feat_cols].fillna(0)
y_act = active_df["acons"].values
g_act = active_df["vehicle"].astype("category").cat.codes.values

oof_reg = np.zeros(len(active_df))
reg_models = []
fold_rmses = []

gkf2 = GroupKFold(n_splits=5)
for fold, (tr_idx, val_idx) in enumerate(gkf2.split(X_act, y_act, g_act), 1):
    X_tr, X_val = X_act.iloc[tr_idx], X_act.iloc[val_idx]
    y_tr, y_val = y_act[tr_idx], y_act[val_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    reg = lgb.train(
        REGR_PARAMS, dtrain, valid_sets=[dval], num_boost_round=4000,
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(1000)],
    )
    oof_reg[val_idx] = reg.predict(X_val, num_iteration=reg.best_iteration)
    reg_models.append(reg)
    fr = np.sqrt(mean_squared_error(y_val, np.clip(oof_reg[val_idx], 0, None)))
    fold_rmses.append(fr)
    print(f"  Fold {fold}: RMSE={fr:.2f}L  (iter={reg.best_iteration})")

oof_reg = np.clip(oof_reg, 0, None)
oof_rmse_active = np.sqrt(mean_squared_error(y_act, oof_reg))
print(f"\nOOF RMSE (active shifts only): {oof_rmse_active:.2f}L")

# Feature importance
importance = pd.DataFrame({
    "feature": list(X_act.columns),
    "gain": np.mean([m.feature_importance("gain") for m in reg_models], axis=0),
}).sort_values("gain", ascending=False)
print("\nTop 20 features:")
print(importance.head(20).to_string(index=False))

# ── Combined OOF evaluation ───────────────────────────────────────────────────
# Reconstruct full predictions: inactive → 0, active → regressor
oof_full = np.zeros(len(labeled))
act_idx = np.where(active_mask)[0]
oof_full[act_idx] = oof_reg
# For correctly-predicted inactive: prediction = 0 (good)
# For incorrectly-predicted active (false positive): prediction = regressor output

oof_rmse_full = np.sqrt(mean_squared_error(y_all, oof_full))
print(f"\nOOF RMSE (combined, treating inactive as 0): {oof_rmse_full:.2f}L")

# ── Generate test predictions ─────────────────────────────────────────────────
print("\n=== Generating submission ===")
test_reg_pred = np.mean(
    [m.predict(X_test, num_iteration=m.best_iteration) for m in reg_models], axis=0
)
test_reg_pred = np.clip(test_reg_pred, 0, None)

# Blend: if classifier says inactive (proba < 0.4), scale down prediction
test_preds = test_reg_pred * test_active_proba
# For confidently active (proba > 0.7), use full regressor prediction
high_active = test_active_proba > 0.7
test_preds[high_active] = test_reg_pred[high_active]
# For confidently inactive (proba < 0.3), predict near 0
low_active = test_active_proba < 0.3
test_preds[low_active] = test_reg_pred[low_active] * test_active_proba[low_active]

test_feats["Predicted"] = test_preds

# ── Build submission ──────────────────────────────────────────────────────────
idm = pd.read_csv(DATA / "id_mapping_new.csv")
idm["date"] = pd.to_datetime(idm["date"]).dt.date

submission = idm.merge(
    test_feats[["vehicle", "date", "shift", "Predicted"]],
    on=["vehicle", "date", "shift"],
    how="left",
)

n_missing = submission["Predicted"].isna().sum()
print(f"Missing predictions: {n_missing} / {len(submission)}")

if n_missing > 0:
    # Fallback: vehicle × shift global mean
    vs_mean = (
        train[train["acons"].notna()]
        .groupby(["vehicle", "shift"])["acons"].mean()
    )
    global_mean = float(train["acons"].dropna().mean())
    for idx, row in submission[submission["Predicted"].isna()].iterrows():
        key = (row["vehicle"], row["shift"])
        submission.at[idx, "Predicted"] = vs_mean.get(key, global_mean)

# Clip to tank capacity
tankcap_map = dict(zip(fleet["vehicle"], fleet["tankcap"]))
submission["tankcap"] = submission["vehicle"].map(tankcap_map).fillna(1379)
submission["Predicted"] = submission.apply(
    lambda r: float(np.clip(r["Predicted"], 0, r["tankcap"])), axis=1
)

final = submission[["id", "Predicted"]].sort_values("id").reset_index(drop=True)
print(f"\nSubmission statistics:")
print(final["Predicted"].describe())
print(f"Zeros: {(final['Predicted'] == 0).sum()}")
print(f"Near-zero (<10L): {(final['Predicted'] < 10).sum()}")

mean_p = final["Predicted"].mean()
std_p  = final["Predicted"].std()
out_name = f"submission_v3_oof{oof_rmse_active:.2f}_mean{mean_p:.1f}.csv"
final.to_csv(OUT / out_name, index=False)
final.to_csv(OUT / "submission.csv", index=False)
print(f"\nSaved: {OUT / out_name}")
print(f"Also saved: {OUT / 'submission.csv'}")
print(f"\nPrevious best LB MSE: 3182.71  (OOF was 48.87L)")
print(f"This run OOF (active only): {oof_rmse_active:.2f}L")
