#!/usr/bin/env python3
"""
EDA to find biggest remaining gains before final submission.
"""
import glob, datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer

ROOT = "/Users/vedantn/Hackathon"
DATA = f"{ROOT}/data"

# ── 1. Fleet: dump_switch coverage ───────────────────────────────────────────
print("=" * 60)
print("1. FLEET — dump_switch field")
print("=" * 60)
fleet = pd.read_csv(f"{DATA}/fleet.csv")
print(fleet[fleet["fleet"] == "Dumper"][["vehicle", "mine_anon", "dump_switch", "tankcap"]].to_string())

# ── 2. Spatial geometry audit ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. SPATIAL GEOMETRY AUDIT")
print("=" * 60)
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)

for mine_id in ["mine001", "mine002"]:
    gpkg = f"{DATA}/{mine_id}_anonymized.gpkg"
    import fiona
    layers = fiona.listlayers(gpkg)
    print(f"\n{mine_id}: {layers}")
    for lname in layers:
        gdf = gpd.read_file(gpkg, layer=lname)
        if gdf.crs and gdf.crs.to_epsg() != 32645:
            gdf = gdf.to_crs("EPSG:32645")
        geom_types = gdf.geometry.geom_type.value_counts().to_dict()
        bbox = gdf.total_bounds  # minx, miny, maxx, maxy
        area_km2 = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) / 1e6
        print(f"  {lname}: {len(gdf)} features, types={geom_types}, bbox_area≈{area_km2:.1f}km²")
        # Try buffering and see if any actual telemetry points land inside
        if lname in ["ob_dump", "bench", "mineral_stock"]:
            from shapely.ops import polygonize, unary_union
            from shapely.geometry import MultiPolygon
            # Method 1: buffer LineString
            buf40  = gdf.geometry.buffer(40).union_all()
            buf100 = gdf.geometry.buffer(100).union_all()
            buf200 = gdf.geometry.buffer(200).union_all()
            # Method 2: polygonize (convert closed LineString rings to Polygons)
            try:
                polys = list(polygonize(gdf.geometry.union_all()))
                poly_union = unary_union(polys) if polys else None
                print(f"    polygonize → {len(polys)} polygons, area={poly_union.area/1e6:.3f}km²" if poly_union else "    polygonize → 0 polygons")
            except Exception as e:
                print(f"    polygonize failed: {e}")
            print(f"    buffer 40m area:  {buf40.area/1e6:.3f}km²")
            print(f"    buffer 100m area: {buf100.area/1e6:.3f}km²")
            print(f"    buffer 200m area: {buf200.area/1e6:.3f}km²")

# ── 3. Test telemetry points vs spatial zones ─────────────────────────────────
print("\n" + "=" * 60)
print("3. DO TELEMETRY POINTS FALL INSIDE ZONES? (sample check)")
print("=" * 60)
# Load a small sample of telemetry for mine001 vehicles
df_sample = pd.read_parquet(f"{DATA}/telemetry_2026-01-01_2026-01-10.parquet")
m1 = df_sample[df_sample["mine_anon"] == "mine001"].sample(min(5000, len(df_sample[df_sample["mine_anon"]=="mine001"])), random_state=42)
x, y = TRANSFORMER.transform(m1["longitude"].values, m1["latitude"].values)
pts = gpd.GeoSeries(gpd.points_from_xy(x, y), crs="EPSG:32645")

for mine_id in ["mine001"]:
    gpkg = f"{DATA}/{mine_id}_anonymized.gpkg"
    for lname in ["ob_dump", "bench", "mineral_stock", "haul_road"]:
        try:
            gdf = gpd.read_file(gpkg, layer=lname).to_crs("EPSG:32645")
            from shapely.ops import polygonize, unary_union
            # Method 1: buffer
            for buf in [40, 100, 200, 500]:
                zone = gdf.geometry.buffer(buf).union_all()
                hits = pts.within(zone).sum()
                pct = hits / len(pts) * 100
                print(f"  {lname} buf={buf}m: {hits}/{len(pts)} = {pct:.1f}%")
            # Method 2: polygonize
            polys = list(polygonize(gdf.geometry.union_all()))
            if polys:
                zone_poly = unary_union(polys)
                hits = pts.within(zone_poly).sum()
                print(f"  {lname} polygonize: {hits}/{len(pts)} = {hits/len(pts)*100:.1f}%")
            print()
        except Exception as e:
            print(f"  {lname}: error {e}")

# ── 4. rain_loss and dense_fog ────────────────────────────────────────────────
print("=" * 60)
print("4. RAIN_LOSS / DENSE_FOG AVAILABILITY")
print("=" * 60)
for fname in sorted(glob.glob(f"{DATA}/telemetry_*.parquet")):
    df = pd.read_parquet(fname, columns=["ts", "rain_loss", "dense_fog"] if "rain_loss" in pd.read_parquet(fname, columns=["ts"]).columns or True else ["ts"])
    cols_present = [c for c in ["rain_loss", "dense_fog"] if c in df.columns]
    if cols_present:
        for c in cols_present:
            nn = df[c].notna().sum()
            nz = (df[c] > 0).sum() if nn > 0 else 0
            print(f"  {fname.split('/')[-1]}: {c} non-null={nn}, nonzero={nz}")
    break  # just check one

import os
for fname in sorted(glob.glob(f"{DATA}/telemetry_*.parquet")):
    df = pd.read_parquet(fname)
    for c in ["rain_loss", "dense_fog"]:
        if c in df.columns:
            nn = df[c].notna().sum()
            nz = (df[c].fillna(0) > 0).sum()
            tag = "TRAIN" if "01_01" in fname or "02_01" in fname or "03_01" in fname else "TEST"
            print(f"  [{tag}] {os.path.basename(fname)}: {c} non-null={nn}/{len(df)}, nonzero={nz}")

# ── 5. No-telemetry test rows — what should they be? ──────────────────────────
print("\n" + "=" * 60)
print("5. NO-TELEMETRY TEST ROWS — EXPECTED TRUE LABEL")
print("=" * 60)
smry = pd.concat([pd.read_csv(f) for f in glob.glob(f"{DATA}/smry_*.csv")], ignore_index=True)
smry["date"] = pd.to_datetime(smry["date"]).dt.date

# Vehicles that had no telemetry: Dump020 (Jan 22-27), Dump040 (Feb 24), Dump041 (Feb 25-28), Dump044/045 (Mar 12)
# Check their training history — are they frequently offline?
offline_vehs = ["Dump020", "Dump040", "Dump041", "Dump044", "Dump045"]
for v in offline_vehs:
    sub = smry[smry["vehicle"] == v]
    zeros = (sub["acons"] < 5).sum()
    total = len(sub)
    print(f"  {v}: {total} training shifts, {zeros} near-zero ({zeros/total:.0%}), mean_active={sub[sub['acons']>10]['acons'].mean():.1f}L")

# ── 6. OOF error analysis by mine ────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. TRAINING LABEL QUALITY CHECK")
print("=" * 60)
# Check if acons matches fuel_volume-derived consumption
df_train = pd.read_parquet(f"{DATA}/telemetry_2026-01-01_2026-01-10.parquet",
                           columns=["vehicle", "ts", "fuel_volume", "mine_anon"])
df_train["ts"] = pd.to_datetime(df_train["ts"])

def assign_shift(ts_series):
    ts = pd.to_datetime(ts_series, utc=False)
    hour = ts.dt.hour
    date = ts.dt.date
    adj_date = [d - datetime.timedelta(days=1) if h >= 22 else d for d, h in zip(date, hour)]
    shift = ["C" if h >= 22 or h < 6 else ("A" if h < 14 else "B") for h in hour]
    return pd.Series(adj_date, index=ts_series.index), pd.Series(shift, index=ts_series.index)

df_train["adj_date"], df_train["shift"] = assign_shift(df_train["ts"])
# fuel_volume start and end per shift
fv = df_train.groupby(["vehicle", "adj_date", "shift"]).agg(
    fv_start=("fuel_volume", "first"),
    fv_end=("fuel_volume", "last"),
    fv_min=("fuel_volume", "min"),
    fv_max=("fuel_volume", "max"),
).reset_index()
fv["fv_consumed"] = fv["fv_start"] - fv["fv_end"]
fv["date"] = pd.to_datetime(fv["adj_date"]).dt.date

merged = smry.merge(fv[["vehicle","date","shift","fv_consumed","fv_start","fv_end","fv_min","fv_max"]],
                    on=["vehicle","date","shift"], how="inner")
merged = merged[merged["acons"] > 10]
merged["discrepancy"] = merged["acons"] - merged["fv_consumed"]
print(f"Label vs fuel_volume discrepancy (where acons>10):")
print(merged["discrepancy"].describe())
big_disc = merged[merged["discrepancy"].abs() > 100]
print(f"Rows with |discrepancy| > 100L: {len(big_disc)} / {len(merged)}")
print("\nBig discrepancy breakdown:")
print(big_disc[["vehicle","date","shift","acons","fv_consumed","fv_start","fv_end"]].head(15).to_string())

# ── 7. Per-vehicle error in training (identify hard vehicles) ─────────────────
print("\n" + "=" * 60)
print("7. VARIANCE IN acons BY VEHICLE×SHIFT (how predictable is each?)")
print("=" * 60)
smry_active = smry[smry["acons"] > 10].copy()
veh_shift_cv = (
    smry_active.groupby(["vehicle", "shift"])["acons"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
veh_shift_cv["cv"] = veh_shift_cv["std"] / (veh_shift_cv["mean"] + 1e-6)
veh_shift_cv = veh_shift_cv[veh_shift_cv["count"] >= 5]
print("Most variable vehicle×shift combos (high CV = hard to predict):")
print(veh_shift_cv.sort_values("cv", ascending=False).head(15).to_string(index=False))
print("\nLeast variable (easy to predict):")
print(veh_shift_cv.sort_values("cv").head(10).to_string(index=False))

# ── 8. cumdist vs disthav ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. CUMDIST (better km source?)")
print("=" * 60)
# cumdist is cumulative and available in both train and test
# shift km from cumdist: max(cumdist) - min(cumdist) per shift
df_cud = pd.read_parquet(f"{DATA}/telemetry_2026-01-01_2026-01-10.parquet",
                          columns=["vehicle","ts","cumdist","disthav","mine_anon"])
df_cud["ts"] = pd.to_datetime(df_cud["ts"])
df_cud["adj_date"], df_cud["shift"] = assign_shift(df_cud["ts"])
km_agg = df_cud.groupby(["vehicle","adj_date","shift"]).agg(
    km_disthav = ("disthav", lambda x: x.sum()/1000),
    km_cumdist  = ("cumdist", lambda x: (x.max() - x.min())),
).reset_index()
km_agg["date"] = pd.to_datetime(km_agg["adj_date"]).dt.date
merged_km = smry.merge(km_agg[["vehicle","date","shift","km_disthav","km_cumdist"]], on=["vehicle","date","shift"])
merged_km = merged_km[merged_km["acons"]>10]
print(f"Correlation acons vs km_disthav: {merged_km['km_disthav'].corr(merged_km['acons']):.4f}")
print(f"Correlation acons vs km_cumdist: {merged_km['km_cumdist'].corr(merged_km['acons']):.4f}")

print("\n=== EDA DONE ===")
