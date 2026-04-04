#!/usr/bin/env python3
"""
Self-contained Kaggle notebook script.
Paste this entire file into a single code cell and run.

Builds spatial + cycle features, trains LightGBM, saves submission.csv.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import gc, warnings, os
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ── Path detection ────────────────────────────────────────────────────────────
DATA = Path("/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge")
OUT  = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/spatial_features")
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
    print("geopandas NOT available — spatial features DISABLED")

DUMP_BUF  = 30.0
LOAD_BUF  = 30.0
HAUL_BUF  = 20.0
EXC_PROX  = 80.0   # metres — excavator proximity = loading zone


def load_mine_layers():
    if not SPATIAL_OK:
        return {}
    result = {}
    for gpkg in sorted(DATA.glob("*.gpkg")):
        n = gpkg.name.lower()
        key = "mine001" if "mine_001" in n or "mine001" in n else (
              "mine002" if "mine_002" in n or "mine002" in n else None)
        if key is None:
            continue
        raw = {}
        for lname in gpd.list_layers(gpkg).name.tolist():
            gdf = gpd.read_file(gpkg, layer=lname)
            if len(gdf) == 0:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:32645")
            elif str(gdf.crs).upper() != "EPSG:32645":
                gdf = gdf.to_crs("EPSG:32645")
            raw[lname] = gdf

        def safe_union(k, buf):
            g = raw.get(k)
            if g is None:
                g = raw.get("mineral_stock" if k == "stock" else k)
            return g.geometry.buffer(buf).union_all() if g is not None else None

        result[key] = {
            "ob_dump_union": safe_union("ob_dump",        DUMP_BUF),
            "stock_union":   safe_union("mineral_stock",  DUMP_BUF),
            "bench_union":   safe_union("bench",          LOAD_BUF),
            "cpu_union":     safe_union("cpu",            LOAD_BUF),
            "haul_union":    safe_union("haul_road",      HAUL_BUF),
        }
        print(f"  {key}: layers={list(raw.keys())}")
    return result


def add_zone_flags(df, mine_layers):
    """Add in_dump_zone, in_load_zone, on_haul_road columns (vectorized)."""
    for col in ["in_dump_zone","in_ob_dump_zone","in_rom_stock_zone","in_load_zone","on_haul_road"]:
        df[col] = np.int8(0)
    if not SPATIAL_OK or not mine_layers:
        return df
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    x, y = TRANSFORMER.transform(df["longitude"].to_numpy(float),
                                  df["latitude"].to_numpy(float))
    df["x_utm"] = x
    df["y_utm"]  = y

    for mine, layers in mine_layers.items():
        mask = df["mine_anon"].astype(str).str.lower() == mine
        if not mask.any():
            continue
        pts = gpd.GeoSeries(gpd.points_from_xy(x[mask], y[mask]), crs="EPSG:32645")
        def w(u):
            return pts.within(u).to_numpy() if u is not None else np.zeros(mask.sum(), bool)
        in_ob  = w(layers["ob_dump_union"])
        in_rom = w(layers["stock_union"])
        in_lod = w(layers["bench_union"]) | w(layers["cpu_union"])
        on_hau = w(layers["haul_union"])
        df.loc[mask, "in_dump_zone"]      = (in_ob | in_rom).astype(np.int8)
        df.loc[mask, "in_ob_dump_zone"]   = in_ob.astype(np.int8)
        df.loc[mask, "in_rom_stock_zone"] = in_rom.astype(np.int8)
        df.loc[mask, "in_load_zone"]      = in_lod.astype(np.int8)
        df.loc[mask, "on_haul_road"]      = on_hau.astype(np.int8)
    return df


def extract_exc_positions(df):
    """Median UTM position of each excavator per (mine, date, shift)."""
    exc = df[df["vehicle"].astype(str).str.startswith("Exc")].copy()
    exc = exc[exc["speed"] < 2]
    if exc.empty or "x_utm" not in exc.columns:
        return pd.DataFrame(columns=["exc_vehicle","mine_anon","date","shift","exc_x","exc_y"])
    return (exc.groupby(["vehicle","mine_anon","date","shift"], observed=True)
               .agg(exc_x=("x_utm","median"), exc_y=("y_utm","median"))
               .reset_index().rename(columns={"vehicle":"exc_vehicle"}))


def add_exc_proximity(df, exc_pos):
    df["near_excavator"] = np.int8(0)
    if exc_pos.empty or "x_utm" not in df.columns:
        return df
    for mine, me in exc_pos.groupby("mine_anon", observed=True):
        mm = df["mine_anon"].astype(str).str.lower() == str(mine).lower()
        for (date, shift), se in me.groupby(["date","shift"], observed=True):
            rm = mm & (df["date"] == date) & (df["shift"] == shift)
            if not rm.any():
                continue
            dx = df.loc[rm,"x_utm"].to_numpy()[:,None] - se["exc_x"].to_numpy()
            dy = df.loc[rm,"y_utm"].to_numpy()[:,None] - se["exc_y"].to_numpy()
            near = ((dx**2 + dy**2) < EXC_PROX**2).any(axis=1)
            df.loc[rm, "near_excavator"] = near.astype(np.int8)
    return df


# ── Per-row signal columns (pre-compute before groupby) ──────────────────────
def add_per_row_signals(df):
    """Add time-weighted and event signals at row level (vectorized)."""
    gkey = ["vehicle","date","shift"]
    df = df.sort_values(["vehicle","ts"]).copy()

    df["tgap"] = (df.groupby(gkey, observed=True)["ts"]
                    .diff().dt.total_seconds().fillna(0).clip(0, 300))

    df["alt_diff"] = df.groupby(gkey, observed=True)["altitude"].diff().fillna(0)
    df["climb"]    = df["alt_diff"].clip(lower=0)
    df["descent"]  = (-df["alt_diff"]).clip(lower=0)

    df["ign_time_s"] = df["tgap"] * (df["ignition"] == 1)
    df["mov_time_s"] = df["tgap"] * (df["speed"] > 2)
    df["idle_time_s"]= df["tgap"] * ((df["ignition"] == 1) & (df["speed"] <= 2))

    prev_spd = df.groupby(gkey, observed=True)["speed"].shift(1).fillna(0)
    df["stop_event"] = ((prev_spd > 2) & (df["speed"] <= 2) & (df["ignition"] == 1)).astype(np.int8)

    # Dump switch signal (analog_input_1)
    if "analog_input_1" in df.columns:
        a = df["analog_input_1"].fillna(0)
        prev_a = df.groupby(gkey, observed=True)["analog_input_1"].shift(1).fillna(0)
        df["dump_edge"]     = ((a > 2.5) & (prev_a <= 2.5)).astype(np.int8)
        df["dump_high"]     = (a > 2.5).astype(np.int8)
        df["dump_high_s"]   = df["tgap"] * df["dump_high"]
    else:
        df["dump_edge"] = df["dump_high"] = df["dump_high_s"] = np.int8(0)

    # Spatial zone time signals
    for zone_col, time_col in [
        ("in_dump_zone",     "dump_zone_s"),
        ("in_ob_dump_zone",  "ob_dump_s"),
        ("in_rom_stock_zone","rom_stock_s"),
        ("in_load_zone",     "load_zone_s"),
        ("on_haul_road",     "haul_road_s"),
        ("near_excavator",   "exc_prox_s"),
    ]:
        if zone_col in df.columns:
            df[time_col] = df["tgap"] * df[zone_col]
        else:
            df[zone_col] = np.int8(0)
            df[time_col] = 0.0

    # Spatial transition events (0→1)
    for zone_col, ev_col in [
        ("in_dump_zone",  "dump_zone_enter"),
        ("in_load_zone",  "load_zone_enter"),
        ("on_haul_road",  "haul_road_enter"),
        ("near_excavator","exc_enter"),
    ]:
        prev = df.groupby(gkey, observed=True)[zone_col].shift(1).fillna(0)
        df[ev_col] = ((df[zone_col] == 1) & (prev == 0)).astype(np.int8)

    return df


def build_shift_features(df):
    """Vectorized groupby.agg() — fast, no Python loop per group."""
    gkey = ["vehicle","date","shift"]

    # Speed percentiles need a custom approach
    spd_moving = df[df["speed"] > 2].copy()
    spd_grp = spd_moving.groupby(gkey, observed=True)["speed"]
    spd_stats = pd.DataFrame({
        "speed_mean": spd_grp.mean(),
        "speed_std":  spd_grp.std(),
        "speed_max":  spd_grp.max(),
        "speed_p50":  spd_grp.quantile(0.5),
        "speed_p75":  spd_grp.quantile(0.75),
        "speed_p90":  spd_grp.quantile(0.9),
    }).reset_index()

    # Altitude first/last for net_lift
    alt_fl = (df.groupby(gkey, observed=True)["altitude"]
                .agg(alt_first="first", alt_last="last")
                .reset_index())

    # External voltage
    ext_cols = {}
    if "external_voltage" in df.columns:
        ev = df.groupby(gkey, observed=True)["external_voltage"]
        ext_cols = {"ext_v_mean": ev.mean(), "ext_v_max": ev.max()}

    agg_dict = {
        "n_pings":          ("ts",           "size"),
        "ign_h":            ("ign_time_s",   "sum"),
        "mov_h":            ("mov_time_s",   "sum"),
        "idle_h":           ("idle_time_s",  "sum"),
        "shift_km":         ("disthav",      lambda x: x.sum() / 1000),
        "total_climb_m":    ("climb",        "sum"),
        "total_descent_m":  ("descent",      "sum"),
        "altitude_mean":    ("altitude",     "mean"),
        "altitude_std":     ("altitude",     "std"),
        "altitude_max":     ("altitude",     "max"),
        "altitude_min":     ("altitude",     "min"),
        "stop_count":       ("stop_event",   "sum"),
        # Dump switch
        "dump_count_sig":   ("dump_edge",    "sum"),
        "dump_high_h":      ("dump_high_s",  lambda x: x.sum() / 3600),
        # Spatial zone times
        "dump_zone_h":      ("dump_zone_s",  lambda x: x.sum() / 3600),
        "ob_dump_h":        ("ob_dump_s",    lambda x: x.sum() / 3600),
        "rom_stock_h":      ("rom_stock_s",  lambda x: x.sum() / 3600),
        "load_zone_h":      ("load_zone_s",  lambda x: x.sum() / 3600),
        "haul_road_h":      ("haul_road_s",  lambda x: x.sum() / 3600),
        "exc_prox_h":       ("exc_prox_s",   lambda x: x.sum() / 3600),
        # Spatial events
        "dump_zone_enters": ("dump_zone_enter","sum"),
        "load_zone_enters": ("load_zone_enter","sum"),
        "haul_road_enters": ("haul_road_enter","sum"),
        "loading_visits":   ("exc_enter",     "sum"),
        # Vibration proxy
        "axis_x_std":       ("axis_x",       "std"),
        "axis_y_std":       ("axis_y",       "std"),
        "axis_z_std":       ("axis_z",       "std"),
        # Quality
        "avg_hdop":         ("gnss_hdop",    "mean"),
        "avg_satellites":   ("satellites",   "mean"),
        "battery_mean":     ("battery_level","mean"),
        "hour_start":       ("ts",           lambda x: x.iloc[0].hour if len(x) else 0),
        "hour_end":         ("ts",           lambda x: x.iloc[-1].hour if len(x) else 0),
    }
    # Only include columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if v[0] in df.columns}

    feats = df.groupby(gkey, observed=True).agg(**agg_dict).reset_index()

    # Convert time sums to hours
    for col in ["ign_h", "mov_h", "idle_h"]:
        if col in feats.columns:
            feats[col] = feats[col] / 3600

    feats = feats.merge(spd_stats, on=gkey, how="left")
    feats = feats.merge(alt_fl,    on=gkey, how="left")
    if ext_cols:
        ext_df = pd.DataFrame(ext_cols).reset_index()
        ext_df.columns = gkey + list(ext_cols.keys())
        feats = feats.merge(ext_df, on=gkey, how="left")

    # Derived columns
    feats["net_lift"]            = (feats["alt_last"] - feats["alt_first"]).fillna(0)
    feats["altitude_range"]      = feats["altitude_max"] - feats["altitude_min"]
    feats["idle_fraction"]       = feats["idle_h"]   / (feats["ign_h"]   + 1e-6)
    feats["km_per_hour"]         = feats["shift_km"] / (feats["ign_h"]   + 1e-6)
    feats["climb_descent_ratio"] = feats["total_climb_m"] / (feats["total_descent_m"] + 1.0)
    feats["km_per_climb_m"]      = feats["shift_km"] / (feats["total_climb_m"] + 1.0)
    feats["stop_density"]        = feats["stop_count"] / (feats["shift_km"] + 1e-6)
    feats["speed_cv"]            = feats["speed_std"] / (feats["speed_mean"] + 1e-6)
    feats["distance_per_dump"]   = feats["shift_km"] / (feats["dump_count_sig"] + 1.0)

    # Spatial fractions (of ignition time)
    ign_h = feats["ign_h"] + 1e-6
    feats["frac_dump_zone"]  = feats["dump_zone_h"]  / ign_h
    feats["frac_load_zone"]  = feats["load_zone_h"]  / ign_h
    feats["frac_haul_road"]  = feats["haul_road_h"]  / ign_h
    feats["frac_exc_prox"]   = feats["exc_prox_h"]   / ign_h

    # Cycle count = max(dump signal, spatial dump entries, loading visits)
    # Prefer analog signal when available; fall back to spatial
    feats["haul_cycles"] = feats[["dump_count_sig","dump_zone_enters","loading_visits"]].max(axis=1)
    feats["km_per_cycle"]= feats["shift_km"] / (feats["haul_cycles"] + 1.0)

    # Vibration
    feats["vibration"] = np.sqrt(
        feats.get("axis_x_std", 0)**2 +
        feats.get("axis_y_std", 0)**2 +
        feats.get("axis_z_std", 0)**2
    )

    # External voltage × time interaction
    if "ext_v_mean" in feats.columns:
        feats["ext_v_x_mov_h"] = feats["ext_v_mean"] * feats["mov_h"]
        feats["ext_v_x_ign_h"] = feats["ext_v_mean"] * feats["ign_h"]

    feats.drop(columns=["alt_first","alt_last"], errors="ignore", inplace=True)
    return feats


# ── File processing ───────────────────────────────────────────────────────────
def ensure_tz(ts):
    out = pd.to_datetime(ts, errors="coerce", utc=True)
    return out.dt.tz_convert("Asia/Kolkata")


LOAD_COLS = [
    "vehicle","mine_anon","ts",
    "latitude","longitude","altitude","speed","ignition","disthav",
    "analog_input_1","external_voltage","satellites","gnss_hdop",
    "axis_x","axis_y","axis_z","battery_level","angle",
]

def process_file(fpath, mine_layers, dumper_set):
    print(f"  {fpath.name}...")
    try:
        df = pd.read_parquet(fpath, columns=LOAD_COLS)
    except Exception:
        df = pd.read_parquet(fpath)
        df = df[[c for c in LOAD_COLS if c in df.columns]].copy()

    df["ts"] = ensure_tz(df["ts"])
    for col in ["speed","altitude","disthav"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["ignition"] = pd.to_numeric(df["ignition"], errors="coerce").fillna(0).astype(int)
    df["speed"] = df["speed"].clip(0, 60)
    df = df[df["speed"] <= 80].copy()

    # Shift + working date
    hour = df["ts"].dt.hour
    df["shift"] = np.where((hour >= 6) & (hour < 14), "A",
                  np.where((hour >= 14) & (hour < 22), "B", "C"))
    day0 = df["ts"].dt.tz_localize(None).dt.normalize()
    nc   = (df["shift"] == "C") & (hour < 6)
    df["date"] = (day0 - pd.to_timedelta(nc.astype("int8"), unit="D")).dt.date

    # Spatial zones on full df (includes excavators for UTM conversion)
    if mine_layers:
        df = add_zone_flags(df, mine_layers)

    # Extract excavator positions (before filtering to dumpers)
    exc_pos = extract_exc_positions(df)

    # Filter to dumpers
    df = df[df["vehicle"].isin(dumper_set)].copy()
    if len(df) == 0:
        return pd.DataFrame()

    # Excavator proximity
    df = add_exc_proximity(df, exc_pos)

    # Per-row derived signals
    df = add_per_row_signals(df)

    # Aggregate to shift level
    feats = build_shift_features(df)
    gc.collect()
    return feats


# ── Temporal + vehicle features ───────────────────────────────────────────────
def add_temporal(df):
    d = pd.to_datetime(df["date"])
    df["day_of_week"]  = d.dt.dayofweek
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["week_number"]  = d.dt.isocalendar().week.astype(int)
    df["day_of_month"] = d.dt.day
    df["month"]        = d.dt.month
    df["shift_enc"]    = df["shift"].map({"C": 0, "A": 1, "B": 2})
    return df


def add_vehicle_stats(labeled, train, test):
    veh = labeled.groupby("vehicle").agg(
        veh_mean_km=("shift_km","mean"), veh_std_km=("shift_km","std"),
        veh_mean_ign_h=("ign_h","mean"), veh_mean_idle_frac=("idle_fraction","mean"),
        veh_mean_cycles=("haul_cycles","mean"), veh_mean_climb=("total_climb_m","mean"),
    ).reset_index()
    vs = labeled.groupby(["vehicle","shift"]).agg(
        veh_shift_mean_acons=("acons","mean"),
        veh_shift_std_acons=("acons","std"),
    ).reset_index()
    drop_cols = list(veh.columns[1:]) + list(vs.columns[2:])
    for df in [train, test]:
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    tr = train.merge(veh, on="vehicle", how="left").merge(vs, on=["vehicle","shift"], how="left")
    te = test.merge( veh, on="vehicle", how="left").merge(vs, on=["vehicle","shift"], how="left")
    return tr, te


def agg_refuels(refuels, offset_min=0):
    r = refuels.copy()
    if "fleet_type" in r.columns:
        r = r[r["fleet_type"] == "Dumper"].copy()
    r["ts"] = ensure_tz(r["ts"])
    r["ts_adj"] = r["ts"] + pd.Timedelta(minutes=offset_min)
    hour = r["ts_adj"].dt.hour
    r["shift"] = np.where((hour>=6)&(hour<14),"A", np.where((hour>=14)&(hour<22),"B","C"))
    day0 = r["ts_adj"].dt.tz_localize(None).dt.normalize()
    nc = (r["shift"]=="C")&(hour<6)
    r["date"] = (day0 - pd.to_timedelta(nc.astype("int8"), unit="D")).dt.date
    r = r.sort_values(["vehicle","ts"])
    r["prev_ts"] = r.groupby("vehicle")["ts"].shift(1)
    r["hrs_since"] = (r["ts"]-r["prev_ts"]).dt.total_seconds()/3600
    agg = (r.groupby(["vehicle","date","shift"]).agg(
        refuel_liters=("litres","sum"), refuel_count=("litres","count"),
        refuel_max=("litres","max"), hrs_since_refuel=("hrs_since","min"),
    ).reset_index().fillna(0))
    sfx = f"_r{offset_min:+d}".replace("+","p").replace("-","m")
    return agg.rename(columns={c: c+sfx for c in ["refuel_liters","refuel_count","refuel_max","hrs_since_refuel"]})


# ── Model ─────────────────────────────────────────────────────────────────────
EXCLUDE = {"vehicle","mine_anon","date","shift","acons",
           "fuel_volume","x_utm","y_utm","mine_enc"}

LGB_PARAMS = {
    "objective":"regression","metric":"rmse","learning_rate":0.03,
    "num_leaves":63,"max_depth":-1,"min_data_in_leaf":20,
    "feature_fraction":0.85,"bagging_fraction":0.85,"bagging_freq":1,
    "lambda_l1":0.05,"lambda_l2":0.3,"verbosity":-1,"seed":42,
}

def rmse(y, yp):
    return float(np.sqrt(mean_squared_error(y, yp)))

def get_features(df):
    return [c for c in df.columns
            if c not in EXCLUDE
            and df[c].dtype not in ["object"]
            and not pd.api.types.is_datetime64_any_dtype(df[c])]


# ── MAIN ──────────────────────────────────────────────────────────────────────
print("\n=== Loading spatial layers ===")
mine_layers = load_mine_layers()

print("\n=== Loading fleet ===")
fleet = pd.read_csv(DATA / "fleet.csv")
dumper_set = set(fleet[fleet["fleet"]=="Dumper"]["vehicle"].tolist())
print(f"Dumpers: {len(dumper_set)}")

all_tel     = sorted(DATA.glob("telemetry_2026-0*.parquet"))
train_files = [f for f in all_tel if f.name not in TEST_FILES]
test_files  = [f for f in all_tel if f.name in TEST_FILES]
print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

print("\n=== Processing training telemetry ===")
train_chunks = []
for f in train_files:
    c = process_file(f, mine_layers, dumper_set)
    if len(c): train_chunks.append(c)
    gc.collect()
train_feats = pd.concat(train_chunks, ignore_index=True)
print(f"Train: {train_feats.shape}")

print("\n=== Processing test telemetry ===")
test_chunks = []
for f in test_files:
    c = process_file(f, mine_layers, dumper_set)
    if len(c): test_chunks.append(c)
    gc.collect()
test_feats = pd.concat(test_chunks, ignore_index=True)
print(f"Test: {test_feats.shape}")

print("\n=== Merging targets ===")
smry = pd.concat([
    pd.read_csv(DATA / "smry_jan_train_ordered.csv"),
    pd.read_csv(DATA / "smry_feb_train_ordered.csv"),
    pd.read_csv(DATA / "smry_mar_train_ordered.csv"),
], ignore_index=True)
smry["date"] = pd.to_datetime(smry["date"]).dt.date
train_feats["date"] = pd.to_datetime(train_feats["date"]).dt.date
test_feats["date"]  = pd.to_datetime(test_feats["date"]).dt.date

train_feats = train_feats.merge(smry[["vehicle","date","shift","acons"]],
                                on=["vehicle","date","shift"], how="left")
print(f"Labeled: {train_feats['acons'].notna().sum()} / {len(train_feats)}")

# Temporal
train_feats = add_temporal(train_feats)
test_feats  = add_temporal(test_feats)

# Vehicle stats
labeled = train_feats.dropna(subset=["acons"]).copy()
train_feats, test_feats = add_vehicle_stats(labeled, train_feats, test_feats)

# Refuel
ref_files = sorted(DATA.glob("rfid_refuels_*.parquet"))
if ref_files:
    ref = pd.read_parquet(ref_files[-1])
    for off in [0, 2, -2]:
        a = agg_refuels(ref, off)
        train_feats = train_feats.merge(a, on=["vehicle","date","shift"], how="left")
        test_feats  = test_feats.merge( a, on=["vehicle","date","shift"], how="left")
    rc = [c for c in train_feats.columns if "refuel" in c or "hrs_since" in c]
    train_feats[rc] = train_feats[rc].fillna(0)
    test_feats[rc]  = test_feats[rc].fillna(0)
    print(f"Refuel features: {len(rc)}")

print("\n=== Spatial feature coverage ===")
for col in ["frac_dump_zone","frac_load_zone","frac_haul_road","haul_cycles","loading_visits"]:
    if col in train_feats.columns:
        labeled_col = train_feats.loc[train_feats["acons"].notna(), col]
        nz = (labeled_col > 0).sum()
        print(f"  {col}: non-zero {nz}/{len(labeled_col)} ({100*nz/len(labeled_col):.0f}%)")

print("\n=== Training model ===")
train_clean = train_feats.dropna(subset=["acons"]).copy()
features = get_features(train_clean)
print(f"Features ({len(features)}): {features[:10]} ...")

X = train_clean[features].fillna(0)
y = train_clean["acons"].values
groups = train_clean["vehicle"].astype("category").cat.codes.values
X_test = test_feats[features].fillna(0)

gkf = GroupKFold(5)
oof = np.zeros(len(y))
models = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
    m = lgb.train(LGB_PARAMS,
                  lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx]),
                  valid_sets=[lgb.Dataset(X.iloc[val_idx], label=y[val_idx])],
                  num_boost_round=4000,
                  callbacks=[lgb.early_stopping(200,verbose=False), lgb.log_evaluation(1000)])
    oof[val_idx] = m.predict(X.iloc[val_idx], num_iteration=m.best_iteration)
    models.append(m)
    fr = rmse(y[val_idx], np.clip(oof[val_idx],0,None))
    print(f"  Fold {fold}: RMSE={fr:.2f}L  (iter={m.best_iteration})")

oof = np.clip(oof, 0, None)
oof_rmse = rmse(y, oof)
print(f"\nOOF RMSE: {oof_rmse:.2f}L")

# Feature importance
imp = pd.DataFrame({"feature":features,
                    "gain":np.mean([m.feature_importance("gain") for m in models],axis=0)
                   }).sort_values("gain",ascending=False)
print("\nTop 20 features:")
print(imp.head(20).to_string(index=False))
imp.to_csv(OUT / "feature_importance.csv", index=False)

print("\n=== Generating submission ===")
test_preds = np.clip(
    np.mean([m.predict(X_test, num_iteration=m.best_iteration) for m in models], axis=0),
    0, None)
test_feats["Predicted"] = test_preds

idm = pd.read_csv(DATA / "id_mapping_new.csv")
idm["date"] = pd.to_datetime(idm["date"]).dt.date

sub = idm.merge(test_feats[["vehicle","date","shift","Predicted"]],
                on=["vehicle","date","shift"], how="left")

# Fallback for missing predictions
n_miss = sub["Predicted"].isna().sum()
print(f"Missing predictions: {n_miss}")
if n_miss > 0:
    global_vs = train_clean.groupby(["vehicle","shift"])["acons"].mean()
    gm = float(train_clean["acons"].mean())
    for idx, row in sub[sub["Predicted"].isna()].iterrows():
        k = (row["vehicle"], row["shift"])
        sub.at[idx,"Predicted"] = global_vs.get(k, gm)

sub["Predicted"] = sub["Predicted"].clip(lower=0)

final = sub[["id","Predicted"]].sort_values("id").reset_index(drop=True)
print(f"Submission stats:\n{final['Predicted'].describe()}")

mean_p = final["Predicted"].mean()
std_p  = final["Predicted"].std()
out_name = f"submission_spatial_oof{oof_rmse:.2f}_mean{mean_p:.1f}.csv"
final.to_csv(OUT / out_name, index=False)
# Also save as standard name
final.to_csv(OUT / "submission.csv", index=False)
print(f"\nSaved: {OUT/out_name}")
print(f"Also saved: {OUT/'submission.csv'}")
print(f"\nPrevious best LB: 3375.61  (OOF was 50.73L)")
print(f"This run OOF:     {oof_rmse:.2f}L")
