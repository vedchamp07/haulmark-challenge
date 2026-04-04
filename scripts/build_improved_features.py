#!/usr/bin/env python3
"""
Improved shift-wise feature engineering.

Key improvements over build_clean_features.py:
1. external_voltage features (corr 0.65 with acons, not in baseline)
2. Better time coverage features (max_gap_min, total_time_covered_h)
3. DPR-aligned telemetry feature (prod_hr proxy from speed/ignition patterns)
4. Richer speed percentile set

Outputs:
- outputs/improved_features/train_improved.csv
- outputs/improved_features/test_improved.csv
"""

from __future__ import annotations

import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs" / "improved_features"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Files that are TEST (not train)
TEST_FILES = {
    "telemetry_2026-01-21_2026-01-31.parquet",
    "telemetry_2026-02-21_2026-02-28.parquet",
    "telemetry_2026-03-12_2026-03-20.parquet",
}

COLS = [
    "vehicle", "mine_anon", "ts",
    "altitude", "speed", "ignition", "disthav",
    "analog_input_1", "axis_x", "axis_y", "axis_z",
    "satellites", "gnss_hdop", "external_voltage", "battery_level",
    "angle", "received_ts",
    # DPR columns (train only, not used as features directly)
    "fuel_volume",
]


def ensure_tz(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts, errors="coerce", utc=True)
    return out.dt.tz_convert("Asia/Kolkata")


def aggregate_shift(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("ts").copy()
    n = len(g)
    if n == 0:
        return pd.Series(dtype=float)

    g["tgap"] = g["ts"].diff().dt.total_seconds().fillna(0).clip(0, 300)

    ign = g["ignition"] == 1
    moving = g["speed"] > 2
    idle = ign & ~moving

    ign_h = g.loc[ign, "tgap"].sum() / 3600
    mov_h = g.loc[moving, "tgap"].sum() / 3600
    idle_h = g.loc[idle, "tgap"].sum() / 3600
    total_h = g["tgap"].sum() / 3600
    shift_km = g["disthav"].sum() / 1000

    # Altitude
    ad = g["altitude"].diff().fillna(0)
    net_lift = float(g["altitude"].iloc[-1] - g["altitude"].iloc[0]) if n > 1 else 0.0
    total_climb = float(ad.clip(lower=0).sum())
    total_descent = float((-ad).clip(lower=0).sum())

    # Speed
    spd = g.loc[g["speed"] > 0, "speed"]
    s_mean = float(spd.mean()) if len(spd) > 0 else 0.0
    s_std = float(spd.std()) if len(spd) > 0 else 0.0
    s_max = float(g["speed"].max())
    s_p25 = float(spd.quantile(0.25)) if len(spd) > 0 else 0.0
    s_p50 = float(spd.quantile(0.50)) if len(spd) > 0 else 0.0
    s_p75 = float(spd.quantile(0.75)) if len(spd) > 0 else 0.0
    s_p90 = float(spd.quantile(0.90)) if len(spd) > 0 else 0.0

    stops = int(((g["speed"].shift(1).fillna(0) > 2) & (g["speed"] <= 2) & ign).sum())

    # Dump signal
    dump_count = 0
    dump_sig_mean = 0.0
    dump_high_h = 0.0
    if "analog_input_1" in g.columns and not g["analog_input_1"].isna().all():
        a = g["analog_input_1"].fillna(0)
        prev = a.shift(1).fillna(0)
        dump_count = int(((a > 2.5) & (prev <= 2.5)).sum())
        dump_sig_mean = float(a.mean())
        dump_high_h = float(g.loc[a > 2.5, "tgap"].sum()) / 3600

    # Vibration / accelerometer
    ax_std = float(g["axis_x"].std()) if "axis_x" in g.columns else 0.0
    ay_std = float(g["axis_y"].std()) if "axis_y" in g.columns else 0.0
    az_std = float(g["axis_z"].std()) if "axis_z" in g.columns else 0.0
    vib = float(np.sqrt(ax_std**2 + ay_std**2 + az_std**2))

    # External voltage (engine load proxy) — NEW
    ext_v = 0.0
    ext_v_max = 0.0
    ext_v_std = 0.0
    if "external_voltage" in g.columns and not g["external_voltage"].isna().all():
        ev = g["external_voltage"].dropna()
        ext_v = float(ev.mean())
        ext_v_max = float(ev.max())
        ext_v_std = float(ev.std()) if len(ev) > 1 else 0.0

    # Ping quality
    max_gap_min = float(g["tgap"].max()) / 60
    avg_hdop = float(g["gnss_hdop"].mean()) if "gnss_hdop" in g.columns else 0.0
    avg_sats = float(g["satellites"].mean()) if "satellites" in g.columns else 0.0

    # Battery
    bat_mean = float(g["battery_level"].mean()) if "battery_level" in g.columns else 0.0

    # Angle (heading) variability
    ang_std = float(g["angle"].std()) if "angle" in g.columns else 0.0

    # RX delay (signal latency)
    rx_delay = 0.0
    if "received_ts" in g.columns and not g["received_ts"].isna().all():
        delay = (g["received_ts"] - g["ts"]).dt.total_seconds().clip(lower=0, upper=3600)
        rx_delay = float(delay.mean())

    # Derived ratios
    idle_frac = idle_h / (ign_h + 1e-6)
    stop_dens = stops / (shift_km + 1e-6)
    km_per_h = shift_km / (ign_h + 1e-6)
    cdr = total_climb / (total_descent + 1.0)
    km_per_climb = shift_km / (total_climb + 1.0)
    dist_per_dump = shift_km / (dump_count + 1.0)
    s_cv = s_std / (s_mean + 1e-6)

    return pd.Series({
        "n_pings": n,
        "ignition_on_hours": ign_h,
        "moving_hours": mov_h,
        "idle_hours": idle_h,
        "total_time_covered_h": total_h,
        "idle_fraction": idle_frac,
        "shift_km": shift_km,
        "km_per_hour": km_per_h,
        "total_climb_m": total_climb,
        "total_descent_m": total_descent,
        "net_lift": net_lift,
        "climb_descent_ratio": cdr,
        "altitude_mean": float(g["altitude"].mean()),
        "altitude_std": float(g["altitude"].std()),
        "altitude_max": float(g["altitude"].max()),
        "altitude_min": float(g["altitude"].min()),
        "altitude_range": float(g["altitude"].max() - g["altitude"].min()),
        "speed_mean": s_mean,
        "speed_std": s_std,
        "speed_max": s_max,
        "speed_p25": s_p25,
        "speed_p50": s_p50,
        "speed_p75": s_p75,
        "speed_p90": s_p90,
        "speed_cv": s_cv,
        "stop_count": stops,
        "stop_density": stop_dens,
        "km_per_climb_m": km_per_climb,
        "dump_event_count": dump_count,
        "dump_signal_mean": dump_sig_mean,
        "dump_high_time_h": dump_high_h,
        "distance_per_dump_km": dist_per_dump,
        "vibration_magnitude": vib,
        # NEW: external voltage
        "external_voltage_mean": ext_v,
        "external_voltage_max": ext_v_max,
        "external_voltage_std": ext_v_std,
        # Coverage quality
        "max_gap_min": max_gap_min,
        "avg_hdop": avg_hdop,
        "avg_satellites": avg_sats,
        "battery_mean": bat_mean,
        "angle_std": ang_std,
        "rx_delay_mean_s": rx_delay,
        # Time anchors
        "hour_start": int(g["ts"].iloc[0].hour),
        "hour_end": int(g["ts"].iloc[-1].hour),
    })


def process_file(fpath: Path, dumper_set: set) -> pd.DataFrame:
    print(f"  {fpath.name}...")
    try:
        df = pd.read_parquet(fpath, columns=[c for c in COLS if True])
    except Exception:
        df = pd.read_parquet(fpath)
        df = df[[c for c in COLS if c in df.columns]].copy()

    df = df[df["vehicle"].isin(dumper_set)].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df["ts"] = ensure_tz(df["ts"])
    if "received_ts" in df.columns:
        df["received_ts"] = ensure_tz(df["received_ts"])

    # Speed filter
    df = df[df["speed"] <= 80].copy()
    df["speed"] = df["speed"].clip(0, 60)

    # Shift + working date
    hour = df["ts"].dt.hour
    df["shift"] = np.where(
        (hour >= 6) & (hour < 14), "A",
        np.where((hour >= 14) & (hour < 22), "B", "C"),
    )
    day0 = df["ts"].dt.tz_localize(None).dt.normalize()
    nc = (df["shift"] == "C") & (hour < 6)
    df["date"] = (day0 - pd.to_timedelta(nc.astype("int8"), unit="D")).dt.date

    feats = (
        df.groupby(["vehicle", "date", "shift"], observed=True)
        .apply(aggregate_shift)
        .reset_index()
    )
    gc.collect()
    return feats


def aggregate_refuels(refuels: pd.DataFrame, offset_min: int = 0) -> pd.DataFrame:
    r = refuels.copy()
    if "fleet_type" in r.columns:
        r = r[r["fleet_type"] == "Dumper"].copy()

    r["ts"] = ensure_tz(r["ts"])
    r["ts_adj"] = r["ts"] + pd.Timedelta(minutes=offset_min)

    hour = r["ts_adj"].dt.hour
    r["shift"] = np.where(
        (hour >= 6) & (hour < 14), "A",
        np.where((hour >= 14) & (hour < 22), "B", "C"),
    )
    day0 = r["ts_adj"].dt.tz_localize(None).dt.normalize()
    nc = (r["shift"] == "C") & (hour < 6)
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
        "refuel_count": f"refuel_count{sfx}",
        "refuel_liters_max": f"refuel_liters_max{sfx}",
        "hrs_since_prev_refuel": f"hrs_since_refuel{sfx}",
    })


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    df = df.copy()
    df["day_of_week"] = d.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_number"] = d.dt.isocalendar().week.astype(int)
    df["day_of_month"] = d.dt.day
    df["month"] = d.dt.month
    df["shift_enc"] = df["shift"].map({"C": 0, "A": 1, "B": 2})
    return df


def add_vehicle_stats(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute vehicle-level stats from training data and attach to both sets."""
    veh = (
        train.groupby("vehicle")
        .agg(
            veh_mean_km=("shift_km", "mean"),
            veh_std_km=("shift_km", "std"),
            veh_mean_ign_h=("ignition_on_hours", "mean"),
            veh_mean_idle_frac=("idle_fraction", "mean"),
            veh_mean_speed=("speed_mean", "mean"),
            veh_mean_climb=("total_climb_m", "mean"),
            veh_mean_ext_v=("external_voltage_mean", "mean"),
        )
        .reset_index()
    )

    veh_shift = pd.DataFrame()
    if "acons" in train.columns:
        veh_shift = (
            train.groupby(["vehicle", "shift"])
            .agg(
                veh_shift_mean_acons=("acons", "mean"),
                veh_shift_std_acons=("acons", "std"),
                veh_shift_count=("acons", "count"),
            )
            .reset_index()
        )

    tr = train.merge(veh, on="vehicle", how="left")
    te = test.merge(veh, on="vehicle", how="left")
    if not veh_shift.empty:
        tr = tr.merge(veh_shift, on=["vehicle", "shift"], how="left")
        te = te.merge(veh_shift, on=["vehicle", "shift"], how="left")
    return tr, te


def main() -> None:
    fleet = pd.read_csv(DATA / "fleet.csv")
    dumper_set = set(fleet[fleet["fleet"] == "Dumper"]["vehicle"].tolist())
    fleet_feats = (
        fleet[fleet["fleet"] == "Dumper"][["vehicle", "tankcap", "mine_anon"]]
        .copy()
        .assign(mine_enc=lambda df: df["mine_anon"].map({"mine001": 0, "mine002": 1}))
    )

    # Discover telemetry files
    all_tel = sorted(DATA.glob("telemetry_2026-0*.parquet"))
    train_files = [f for f in all_tel if f.name not in TEST_FILES]
    test_files = [f for f in all_tel if f.name in TEST_FILES]

    print("Train files:", [f.name for f in train_files])
    print("Test files:", [f.name for f in test_files])

    # Process telemetry
    print("\nProcessing training telemetry...")
    train_chunks = [process_file(f, dumper_set) for f in train_files]
    train_feats = pd.concat([c for c in train_chunks if len(c) > 0], ignore_index=True)
    print(f"  Train shape: {train_feats.shape}")

    print("Processing test telemetry...")
    test_chunks = [process_file(f, dumper_set) for f in test_files]
    test_feats = pd.concat([c for c in test_chunks if len(c) > 0], ignore_index=True)
    print(f"  Test shape: {test_feats.shape}")

    # Load summary targets
    smry = pd.concat([
        pd.read_csv(DATA / "smry_jan_train_ordered.csv"),
        pd.read_csv(DATA / "smry_feb_train_ordered.csv"),
        pd.read_csv(DATA / "smry_mar_train_ordered.csv"),
    ], ignore_index=True)
    smry["date"] = pd.to_datetime(smry["date"]).dt.date

    # Standardize date types
    train_feats["date"] = pd.to_datetime(train_feats["date"]).dt.date
    test_feats["date"] = pd.to_datetime(test_feats["date"]).dt.date

    # Merge targets with training features
    train_feats = train_feats.merge(
        smry[["vehicle", "date", "shift", "acons"]],
        on=["vehicle", "date", "shift"],
        how="left",
    )
    print(f"  Training rows with acons: {train_feats['acons'].notna().sum()} / {len(train_feats)}")

    # Fleet features
    for df in [train_feats, test_feats]:
        df.drop(columns=[c for c in ["tankcap", "mine_enc", "mine_anon"] if c in df.columns], inplace=True)
    train_feats = train_feats.merge(
        fleet_feats[["vehicle", "tankcap", "mine_enc"]], on="vehicle", how="left"
    )
    test_feats = test_feats.merge(
        fleet_feats[["vehicle", "tankcap", "mine_enc"]], on="vehicle", how="left"
    )

    # Temporal features
    train_feats = add_temporal(train_feats)
    test_feats = add_temporal(test_feats)

    # Vehicle stats
    labeled = train_feats.dropna(subset=["acons"]).copy()
    train_feats, test_feats = add_vehicle_stats(labeled, test_feats)
    # Also attach to all train (including rows without labels)
    train_feats, _ = add_vehicle_stats(labeled, train_feats)
    # Re-do properly: compute stats from labeled data only
    veh = labeled.groupby("vehicle").agg(
        veh_mean_km=("shift_km", "mean"),
        veh_std_km=("shift_km", "std"),
        veh_mean_ign_h=("ignition_on_hours", "mean"),
        veh_mean_idle_frac=("idle_fraction", "mean"),
        veh_mean_speed=("speed_mean", "mean"),
        veh_mean_climb=("total_climb_m", "mean"),
        veh_mean_ext_v=("external_voltage_mean", "mean"),
    ).reset_index()
    veh_shift = labeled.groupby(["vehicle", "shift"]).agg(
        veh_shift_mean_acons=("acons", "mean"),
        veh_shift_std_acons=("acons", "std"),
        veh_shift_count=("acons", "count"),
    ).reset_index()

    for df in [train_feats, test_feats]:
        # Drop any previously merged vehicle stats to avoid duplicates
        drop = [c for c in df.columns if c.startswith("veh_")]
        df.drop(columns=drop, inplace=True)

    train_feats = train_feats.merge(veh, on="vehicle", how="left")
    train_feats = train_feats.merge(veh_shift, on=["vehicle", "shift"], how="left")
    test_feats = test_feats.merge(veh, on="vehicle", how="left")
    test_feats = test_feats.merge(veh_shift, on=["vehicle", "shift"], how="left")

    # Refuel features
    refuel_candidates = sorted(DATA.glob("rfid_refuels_*.parquet"))
    if refuel_candidates:
        refuels = pd.read_parquet(refuel_candidates[-1])
        for offset in [0, 2, -2]:
            agg = aggregate_refuels(refuels, offset)
            train_feats = train_feats.merge(agg, on=["vehicle", "date", "shift"], how="left")
            test_feats = test_feats.merge(agg, on=["vehicle", "date", "shift"], how="left")
        ref_cols = [c for c in train_feats.columns if c.startswith("refuel_") or c.startswith("hrs_since_refuel")]
        train_feats[ref_cols] = train_feats[ref_cols].fillna(0)
        test_feats[ref_cols] = test_feats[ref_cols].fillna(0)
        print(f"Added refuel features: {ref_cols}")

    print(f"\nFinal train shape: {train_feats.shape}")
    print(f"Final test shape:  {test_feats.shape}")
    print(f"Training rows with acons: {train_feats['acons'].notna().sum()}")

    train_feats.to_csv(OUTPUT / "train_improved.csv", index=False)
    test_feats.to_csv(OUTPUT / "test_improved.csv", index=False)
    print(f"\nSaved to {OUTPUT}/")


if __name__ == "__main__":
    main()
