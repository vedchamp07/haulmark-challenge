#!/usr/bin/env python3
"""Fast cycle/idle feature builder (no geopandas).

Purpose
- Add *revolutionary* signal quickly without slow polygon `within()` calls:
  - loaded vs empty haul behavior on haul segments
  - cycle-time proxies from (load event -> dump event)

How
- Load telemetry parquet files
- Assign (date, shift) with the same shift logic
- Filter to dumpers
- Compute excavator proximity using mean excavator anchors per mine (UTM)
- Use analog dump switch (`analog_input_1`) as dump event when present
- Aggregate per (vehicle, date, shift)

Outputs
- outputs/cycle_features_fast/train_cycle.csv
- outputs/cycle_features_fast/test_cycle.csv

These can be merged on (vehicle, date, shift) alongside ckpt + spatial v1 features.
"""

from __future__ import annotations

import gc
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pyproj import Transformer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "cycle_features_fast"
OUT.mkdir(parents=True, exist_ok=True)

TEST_FILES = {
    "telemetry_2026-01-21_2026-01-31.parquet",
    "telemetry_2026-02-21_2026-02-28.parquet",
    "telemetry_2026-03-12_2026-03-20.parquet",
}

EXC_PROXIMITY_M = 80.0
DUMP_SWITCH_T = 2.5

TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)


def to_utm(lat: np.ndarray, lon: np.ndarray):
    x, y = TRANSFORMER.transform(lon, lat)
    return x, y


def ensure_tz(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts, errors="coerce", utc=True)
    return out.dt.tz_convert("Asia/Kolkata")


def compute_excavator_mean_positions(telemetry_files: list[Path]) -> dict[str, np.ndarray]:
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
    df = df.copy()
    df["near_excavator"] = 0
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    x, y = to_utm(df["latitude"].to_numpy(dtype=float), df["longitude"].to_numpy(dtype=float))
    df["x_utm"] = x
    df["y_utm"] = y

    for mine_id, anchors in exc_positions_utm.items():
        if anchors is None or len(anchors) == 0:
            continue
        mask = df["mine_anon"].astype(str).str.lower() == mine_id
        if mask.sum() == 0:
            continue

        xm = df.loc[mask, "x_utm"].to_numpy(dtype=float)
        ym = df.loc[mask, "y_utm"].to_numpy(dtype=float)
        dx = xm[:, None] - anchors[:, 0][None, :]
        dy = ym[:, None] - anchors[:, 1][None, :]
        dist2 = dx * dx + dy * dy
        near = (dist2.min(axis=1) < EXC_PROXIMITY_M**2)
        df.loc[mask, "near_excavator"] = near.astype(np.int8)

    return df


def aggregate_shift(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("ts").copy()
    n = len(g)
    if n == 0:
        return pd.Series(dtype=float)

    ts = g["ts"]
    tgap = ts.diff().dt.total_seconds().fillna(0).clip(0, 300).to_numpy(dtype=float)

    spd = pd.to_numeric(g["speed"], errors="coerce").fillna(0).to_numpy(dtype=float)
    ign = pd.to_numeric(g["ignition"], errors="coerce").fillna(0).to_numpy(dtype=int) == 1
    moving = spd > 2.0

    ign_s = float(tgap[ign].sum())
    mov_s = float(tgap[moving].sum())
    idle_s = float(tgap[(ign) & (~moving)].sum())

    dist = pd.to_numeric(g["disthav"], errors="coerce").fillna(0).to_numpy(dtype=float)
    shift_km = float(dist.sum() / 1000.0)

    near = g["near_excavator"].to_numpy(dtype=int) if "near_excavator" in g.columns else np.zeros(n, dtype=int)
    near_prev = np.r_[0, near[:-1]]
    load_evt = (near == 1) & (near_prev == 0)

    dump_evt = np.zeros(n, dtype=bool)
    if "analog_input_1" in g.columns and not g["analog_input_1"].isna().all():
        a = pd.to_numeric(g["analog_input_1"], errors="coerce").fillna(0).to_numpy(dtype=float)
        prev = np.r_[0.0, a[:-1]]
        dump_evt = (a > DUMP_SWITCH_T) & (prev <= DUMP_SWITCH_T)

    load_events = int(load_evt.sum())
    dump_events = int(dump_evt.sum())

    score = np.cumsum(load_evt.astype(int) - dump_evt.astype(int))
    score = np.clip(score, 0, None)
    loaded = score > 0
    empty = (score == 0) & (np.cumsum(dump_evt.astype(int)) > 0)

    spd_loaded = spd[moving & loaded]
    spd_empty = spd[moving & empty]

    loaded_speed_mean = float(spd_loaded.mean()) if spd_loaded.size else 0.0
    empty_speed_mean = float(spd_empty.mean()) if spd_empty.size else 0.0
    loaded_speed_p90 = float(np.percentile(spd_loaded, 90)) if spd_loaded.size else 0.0
    empty_speed_p90 = float(np.percentile(spd_empty, 90)) if spd_empty.size else 0.0

    loaded_km = float(dist[loaded].sum() / 1000.0)
    empty_km = float(dist[empty].sum() / 1000.0)

    cycle_time_min_p50 = 0.0
    if load_events > 0 and dump_events > 0:
        ts_ns = ts.astype("int64").to_numpy()
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

    haul_cycles_proxy = max(load_events, dump_events)

    return pd.Series({
        "cy_ignition_on_hours": ign_s / 3600.0,
        "cy_moving_hours": mov_s / 3600.0,
        "cy_idle_hours": idle_s / 3600.0,
        "cy_shift_km": shift_km,
        "cy_load_events": load_events,
        "cy_dump_events": dump_events,
        "cy_haul_cycles_proxy": haul_cycles_proxy,
        "cy_cycle_time_min_p50": cycle_time_min_p50,
        "cy_loaded_speed_mean": loaded_speed_mean,
        "cy_empty_speed_mean": empty_speed_mean,
        "cy_loaded_speed_p90": loaded_speed_p90,
        "cy_empty_speed_p90": empty_speed_p90,
        "cy_loaded_km": loaded_km,
        "cy_empty_km": empty_km,
        "cy_loaded_empty_speed_ratio": loaded_speed_mean / (empty_speed_mean + 1e-6),
    })


def process_file(fpath: Path, dumper_set: set, exc_positions_utm: dict[str, np.ndarray]) -> pd.DataFrame:
    print(f"  {fpath.name}...")
    cols = [
        "vehicle",
        "mine_anon",
        "ts",
        "latitude",
        "longitude",
        "speed",
        "ignition",
        "disthav",
        "analog_input_1",
    ]
    try:
        df = pd.read_parquet(fpath, columns=cols)
    except Exception:
        df = pd.read_parquet(fpath)
        df = df[[c for c in cols if c in df.columns]].copy()

    if df.empty:
        return pd.DataFrame()

    df["ts"] = ensure_tz(df["ts"])
    df["speed"] = pd.to_numeric(df.get("speed"), errors="coerce").fillna(0).clip(0, 60)
    df["ignition"] = pd.to_numeric(df.get("ignition"), errors="coerce").fillna(0).astype(int)
    df["disthav"] = pd.to_numeric(df.get("disthav"), errors="coerce").fillna(0)

    hour = df["ts"].dt.hour
    df["shift"] = np.where((hour >= 6) & (hour < 14), "A", np.where((hour >= 14) & (hour < 22), "B", "C"))
    day0 = df["ts"].dt.tz_localize(None).dt.normalize()
    nc = (df["shift"] == "C") & (hour < 6)
    df["date"] = (day0 - pd.to_timedelta(nc.astype("int8"), unit="D")).dt.date

    df = df[df["vehicle"].isin(dumper_set)].copy()
    if df.empty:
        return pd.DataFrame()

    df = add_excavator_proximity_fast(df, exc_positions_utm)

    feats = df.groupby(["vehicle", "date", "shift"], observed=True).apply(aggregate_shift).reset_index()
    return feats


def main():
    fleet = pd.read_csv(DATA / "fleet.csv")
    dumper_set = set(fleet[fleet["fleet"] == "Dumper"]["vehicle"].tolist())
    print(f"Dumpers: {len(dumper_set)}")

    tel_files = sorted(DATA.glob("telemetry_2026-0*.parquet"))
    train_files = [f for f in tel_files if f.name not in TEST_FILES]
    test_files = [f for f in tel_files if f.name in TEST_FILES]

    print("Computing excavator anchors...")
    exc_positions_utm = compute_excavator_mean_positions(tel_files)
    for mine, anchors in exc_positions_utm.items():
        print(f"  {mine}: {len(anchors)}")

    print("\nProcessing train...")
    tr_parts = []
    for f in train_files:
        tr_parts.append(process_file(f, dumper_set, exc_positions_utm))
        gc.collect()
    tr = pd.concat([x for x in tr_parts if len(x)], ignore_index=True)

    print("\nProcessing test...")
    te_parts = []
    for f in test_files:
        te_parts.append(process_file(f, dumper_set, exc_positions_utm))
        gc.collect()
    te = pd.concat([x for x in te_parts if len(x)], ignore_index=True)

    tr.to_csv(OUT / "train_cycle.csv", index=False)
    te.to_csv(OUT / "test_cycle.csv", index=False)
    print(f"\nSaved: {OUT / 'train_cycle.csv'}")
    print(f"Saved: {OUT / 'test_cycle.csv'}")


if __name__ == "__main__":
    main()
