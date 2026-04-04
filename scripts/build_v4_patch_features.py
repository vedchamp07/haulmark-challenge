#!/usr/bin/env python3
"""
v4 feature patch — runs locally in ~5 minutes.

Loads checkpoints from ckpts/, then re-extracts ONLY the missing/broken features:
  1. Spatial zone fractions (were all 0 due to wrong gpkg filename on Kaggle)
  2. Accelerometer std (axis_x/y/z → accel_std per shift)
  3. cumdist_km (more reliable than disthav sum)
  4. mine_enc, has_dump_switch (no parquet needed — from fleet.csv)

Saves patched checkpoints to ckpts/train_v4.parquet and ckpts/test_v4.parquet.
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

# ── Load spatial layers with correct filenames ────────────────────────────────
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

    # ob_dump and mineral_stock are both valid dump destinations
    ob  = load_layer("ob_dump")
    stk = load_layer("mineral_stock")
    hr  = load_layer("haul_road")

    # Use polygonize first (exact), fall back to buffer
    def make_zone(gdf, buf):
        if gdf is None or len(gdf) == 0:
            return None
        polys = list(polygonize(gdf.geometry.union_all()))
        if polys:
            return unary_union(polys).buffer(buf * 0.5)   # small extra buffer
        return gdf.geometry.buffer(buf).union_all()

    mine_zones[mine_id] = {
        "ob_dump":   make_zone(ob,  50),
        "stock":     make_zone(stk, 50),
        "haul_road": hr.geometry.buffer(25).union_all() if hr is not None and len(hr) > 0 else None,
    }
    # Combine dump zones
    dump_parts = [z for z in [mine_zones[mine_id]["ob_dump"], mine_zones[mine_id]["stock"]] if z is not None]
    mine_zones[mine_id]["any_dump"] = unary_union(dump_parts) if dump_parts else None
    print(f"  {mine_id}: ob_dump={mine_zones[mine_id]['ob_dump'] is not None}, "
          f"stock={mine_zones[mine_id]['stock'] is not None}, "
          f"haul={mine_zones[mine_id]['haul_road'] is not None}")


def assign_shift(ts_series: pd.Series):
    ts = pd.to_datetime(ts_series, utc=False)
    hour = ts.dt.hour
    date = ts.dt.date
    adj = [d - datetime.timedelta(days=1) if h >= 22 else d for d, h in zip(date, hour)]
    shift = ["C" if h >= 22 or h < 6 else ("A" if h < 14 else "B") for h in hour]
    return pd.Series(adj, index=ts_series.index), pd.Series(shift, index=ts_series.index)


def extract_patch_features(file_list: list, label: str) -> pd.DataFrame:
    """
    Fast targeted pass: load only needed columns, compute spatial fracs + accel + cumdist.
    Returns shift-level patch df keyed on (vehicle, date, shift).
    """
    needed_cols = [
        "vehicle", "ts", "latitude", "longitude", "mine_anon",
        "axis_x", "axis_y", "axis_z",
        "cumdist",
    ]
    parts = []
    for fname in file_list:
        path = DATA / fname
        print(f"  {fname}...")
        df = pd.read_parquet(path, columns=[c for c in needed_cols
                                             if c in pd.read_parquet(path, columns=["vehicle"]).columns
                                             or True])
        # Read only needed cols robustly
        all_cols = pd.read_parquet(path, columns=["vehicle"]).index  # just to get schema trick
        try:
            df = pd.read_parquet(path, columns=needed_cols)
        except Exception:
            available = pd.read_parquet(path).columns.tolist()
            use = [c for c in needed_cols if c in available]
            df = pd.read_parquet(path, columns=use)

        df = df[df["vehicle"].str.startswith("Dump")].copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=False)
        df["adj_date"], df["shift"] = assign_shift(df["ts"])

        # UTM
        x, y = TRANSFORMER.transform(df["longitude"].values, df["latitude"].values)
        df["x_utm"] = x
        df["y_utm"] = y

        # Zone flags (vectorised per mine)
        df["in_any_dump"] = False
        df["on_haul_road"] = False
        for mine_id, zones in mine_zones.items():
            mask = df["mine_anon"] == mine_id
            if mask.sum() == 0:
                continue
            sub_x = df.loc[mask, "x_utm"].values
            sub_y = df.loc[mask, "y_utm"].values
            pts = gpd.GeoSeries(gpd.points_from_xy(sub_x, sub_y), crs="EPSG:32645")
            if zones["any_dump"] is not None:
                df.loc[mask, "in_any_dump"] = pts.within(zones["any_dump"]).values
            if zones["haul_road"] is not None:
                df.loc[mask, "on_haul_road"] = pts.within(zones["haul_road"]).values

        parts.append(df)
        gc.collect()

    df_all = pd.concat(parts, ignore_index=True)

    # Accel magnitude (if columns exist)
    has_accel = all(c in df_all.columns for c in ["axis_x", "axis_y", "axis_z"])

    # Aggregate to shift level
    agg_dict = {
        "frac_dump_v4":  ("in_any_dump",  "mean"),
        "frac_haul_v4":  ("on_haul_road", "mean"),
        "cumdist_km":    ("cumdist",       lambda x: (x.max() - x.min())),
    }
    if has_accel:
        # compute per-ping accel magnitude first
        df_all["accel_mag"] = np.sqrt(
            df_all["axis_x"].fillna(0)**2 +
            df_all["axis_y"].fillna(0)**2 +
            df_all["axis_z"].fillna(0)**2
        )
        agg_dict["accel_std"]  = ("accel_mag", "std")
        agg_dict["accel_mean"] = ("accel_mag", "mean")

    agg = df_all.groupby(["vehicle", "adj_date", "shift"]).agg(**agg_dict).reset_index()
    agg.rename(columns={"adj_date": "date"}, inplace=True)
    agg["date"] = pd.to_datetime(agg["date"]).dt.date

    print(f"  {label}: patch features shape {agg.shape}")
    nz_dump = (agg["frac_dump_v4"] > 0).sum()
    nz_haul = (agg["frac_haul_v4"] > 0).sum()
    print(f"  frac_dump_v4 non-zero: {nz_dump}/{len(agg)} ({nz_dump/len(agg):.1%})")
    print(f"  frac_haul_v4 non-zero: {nz_haul}/{len(agg)} ({nz_haul/len(agg):.1%})")
    return agg


# ── Run extraction ────────────────────────────────────────────────────────────
print("\n=== Extracting patch features: TRAIN ===")
train_patch = extract_patch_features(TRAIN_FILES, "train")
print("\n=== Extracting patch features: TEST ===")
test_patch  = extract_patch_features(TEST_FILES, "test")

# ── Load checkpoints ──────────────────────────────────────────────────────────
print("\n=== Loading checkpoints ===")
train = pd.read_parquet(CKPT / "train_feats_checkpoint.parquet")
test  = pd.read_parquet(CKPT / "test_feats_checkpoint.parquet")
train["date"] = pd.to_datetime(train["date"]).dt.date
test["date"]  = pd.to_datetime(test["date"]).dt.date
print(f"  train: {train.shape}, test: {test.shape}")

# Drop the broken zero-filled spatial columns
drop_broken = ["frac_dump_zone", "frac_load_zone", "frac_haul_road"]
train.drop(columns=[c for c in drop_broken if c in train.columns], inplace=True)
test.drop( columns=[c for c in drop_broken if c in test.columns],  inplace=True)

# ── Merge patch features ──────────────────────────────────────────────────────
def safe_merge(left, right, on):
    keys = on if isinstance(on, list) else [on]
    drop = [c for c in right.columns if c not in keys and c in left.columns]
    return left.merge(right.drop(columns=drop), on=on, how="left")

train = safe_merge(train, train_patch, on=["vehicle", "date", "shift"])
test  = safe_merge(test,  test_patch,  on=["vehicle", "date", "shift"])

# ── Fleet-derived features (no parquet needed) ────────────────────────────────
fleet = pd.read_csv(DATA / "fleet.csv")
fleet_d = fleet[fleet["fleet"] == "Dumper"][["vehicle", "mine_anon", "dump_switch", "tankcap"]].copy()
fleet_d["mine_enc"]       = (fleet_d["mine_anon"] == "mine002").astype(int)
fleet_d["has_dump_switch"] = fleet_d["dump_switch"].fillna(0).astype(int)
fleet_feat = fleet_d[["vehicle", "mine_enc", "has_dump_switch", "tankcap"]]

train = safe_merge(train, fleet_feat, on="vehicle")
test  = safe_merge(test,  fleet_feat, on="vehicle")

# Fill any remaining NaN in new cols
for col in ["frac_dump_v4", "frac_haul_v4", "cumdist_km", "accel_std", "accel_mean",
            "mine_enc", "has_dump_switch", "tankcap"]:
    for df_ in [train, test]:
        if col in df_.columns:
            df_[col] = df_[col].fillna(0)

print("\n=== Patched feature summary ===")
for col in ["frac_dump_v4", "frac_haul_v4", "cumdist_km", "accel_std", "mine_enc", "has_dump_switch"]:
    if col in train.columns:
        nz = (train[col] > 0).sum()
        print(f"  train {col}: non-zero {nz}/{len(train)} ({nz/len(train):.1%}), mean={train[col].mean():.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
train.to_parquet(CKPT / "train_v4.parquet", index=False)
test.to_parquet( CKPT / "test_v4.parquet",  index=False)
print(f"\nSaved: ckpts/train_v4.parquet {train.shape}")
print(f"Saved: ckpts/test_v4.parquet  {test.shape}")
