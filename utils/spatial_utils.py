from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.data_loader import discover_data_files, split_train_test_files, load_telemetry


def add_utm_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    x, y = transformer.transform(out["longitude"].to_numpy(), out["latitude"].to_numpy())
    out["x_utm"] = x
    out["y_utm"] = y
    return out


def load_layers(gpkg_path: Path) -> Dict[str, gpd.GeoDataFrame]:
    layers: Dict[str, gpd.GeoDataFrame] = {}
    for lname in gpd.list_layers(gpkg_path).name.tolist():
        gdf = gpd.read_file(gpkg_path, layer=lname)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:32645")
        elif str(gdf.crs).upper() != "EPSG:32645":
            gdf = gdf.to_crs("EPSG:32645")
        layers[lname] = gdf
    return layers


def _safe_union(gdf: Optional[gpd.GeoDataFrame], buffer_m: float) -> Optional[object]:
    if gdf is None or gdf.empty:
        return None
    geom = gdf.geometry.buffer(buffer_m)
    return geom.union_all()


def add_zone_flags(
    df: pd.DataFrame,
    mine_layers: Dict[str, gpd.GeoDataFrame],
    dump_buffer_m: float = 20.0,
    load_buffer_m: float = 20.0,
    haul_buffer_m: float = 15.0,
) -> pd.DataFrame:
    out = df.copy()
    if "x_utm" not in out.columns or "y_utm" not in out.columns:
        out = add_utm_coordinates(out)

    points = gpd.GeoSeries(gpd.points_from_xy(out["x_utm"], out["y_utm"]), crs="EPSG:32645")

    ob_dump_union = _safe_union(mine_layers.get("ob_dump"), dump_buffer_m)
    stock_union = _safe_union(mine_layers.get("stock", mine_layers.get("mineral_stock")), dump_buffer_m)
    bench_union = _safe_union(mine_layers.get("bench"), load_buffer_m)
    cpu_union = _safe_union(mine_layers.get("cpu"), load_buffer_m)
    haul_union = _safe_union(mine_layers.get("haul_road"), haul_buffer_m)

    in_ob_dump = points.within(ob_dump_union).to_numpy() if ob_dump_union is not None else np.zeros(len(out), dtype=bool)
    in_rom_stock = points.within(stock_union).to_numpy() if stock_union is not None else np.zeros(len(out), dtype=bool)
    in_dump = in_ob_dump | in_rom_stock

    in_load = np.zeros(len(out), dtype=bool)
    for zone in [bench_union, cpu_union]:
        if zone is not None:
            in_load |= points.within(zone).to_numpy()

    on_haul = points.within(haul_union).to_numpy() if haul_union is not None else np.zeros(len(out), dtype=bool)

    out["in_dump_zone"] = in_dump.astype(np.int8)
    out["in_ob_dump_zone"] = in_ob_dump.astype(np.int8)
    out["in_rom_stock_zone"] = in_rom_stock.astype(np.int8)
    out["in_load_zone"] = in_load.astype(np.int8)
    out["on_haul_road"] = on_haul.astype(np.int8)
    return out


def run_spatial_diagnostics(data_dir: Path) -> None:
    files = discover_data_files(data_dir)
    train_files, _ = split_train_test_files(files.telemetry_files)
    if not files.gpkg_files:
        print("[spatial_utils] No gpkg file found.")
        return
    gpkg = files.gpkg_files[0]
    layers = load_layers(gpkg)
    print(f"[spatial_utils] Loaded {gpkg.name}")
    for k, v in layers.items():
        print(f" - {k}: shape={v.shape}, crs={v.crs}")

    cols = ["vehicle", "mine_anon", "ts", "latitude", "longitude", "speed", "disthav", "date_dpr"]
    sample = load_telemetry(train_files[0], columns=cols).head(50000)
    sample = add_utm_coordinates(sample)
    sample = add_zone_flags(sample, layers)

    print("[spatial_utils] zone rates")
    print(sample[["in_dump_zone", "in_load_zone", "on_haul_road"]].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial layer diagnostics")
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()
    run_spatial_diagnostics(Path(args.data_dir))
