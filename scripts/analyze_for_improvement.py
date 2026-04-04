#!/usr/bin/env python3
"""
Deep dive analysis to understand what's needed to go from MSE 3182 → sub-1000.
sqrt(3182) ≈ 56.4L RMSE currently, target sqrt(1000) ≈ 31.6L RMSE.
"""
import glob
import numpy as np
import pandas as pd

# ── Load summary labels ───────────────────────────────────────────────────────
smry = pd.concat([pd.read_csv(f) for f in glob.glob("data/smry_*.csv")], ignore_index=True)
smry["date"] = pd.to_datetime(smry["date"]).dt.date
print(f"Summary rows: {len(smry)}")
print(f"Vehicles:     {smry['vehicle'].nunique()}")
print()

# ── acons distribution ────────────────────────────────────────────────────────
print("=== acons distribution ===")
bins = [0, 1, 10, 50, 100, 200, 300, 400, 700]
labels = ["0", "1-10", "10-50", "50-100", "100-200", "200-300", "300-400", "400+"]
smry["bucket"] = pd.cut(smry["acons"], bins=bins, labels=labels, right=False)
print(smry["bucket"].value_counts().sort_index().to_string())
print()

# Key question: are near-zero shifts truly "off" or data quality issues?
zero_rows = smry[smry["acons"] < 1]
active_rows = smry[smry["acons"] >= 50]
print(f"Near-zero shifts (acons<1): {len(zero_rows)} ({len(zero_rows)/len(smry):.1%})")
print(f"Active shifts (acons≥50):   {len(active_rows)} ({len(active_rows)/len(smry):.1%})")
print()
print("Near-zero: runhrs distribution:")
print(zero_rows["runhrs"].value_counts().head(10).to_string())
print()

# ── Test set: which shifts are we predicting? ─────────────────────────────────
idmap = pd.read_csv("data/id_mapping_new.csv")
idmap["date"] = pd.to_datetime(idmap["date"]).dt.date
print(f"=== Test set: {len(idmap)} rows ===")
print("Shift distribution:", idmap["shift"].value_counts().to_dict())
print("Vehicles in test:", idmap["vehicle"].nunique())

# How many test (vehicle, date, shift) have known-zero history?
smry_zero = smry[smry["acons"] < 1][["vehicle", "shift"]].drop_duplicates()
print(f"\nVehicle+shift combos that EVER had near-zero in training: {len(smry_zero)}")
# Could these test rows also be zero?

# ── operator_id analysis ──────────────────────────────────────────────────────
print("\n=== operator_id availability ===")
test_tele = pd.read_parquet("data/telemetry_2026-01-21_2026-01-31.parquet",
                            columns=["vehicle", "ts", "operator_id"])
print(f"Test telemetry rows: {len(test_tele)}")
print(f"operator_id non-null: {test_tele['operator_id'].notna().sum()} ({test_tele['operator_id'].notna().mean():.1%})")
print(f"Unique operators in test: {test_tele['operator_id'].nunique()}")
# Is operator_id stable within a shift (same driver all shift)?
op_per_shift = test_tele.groupby(["vehicle", test_tele["ts"].dt.date])["operator_id"].nunique()
print(f"Mean unique operators per (vehicle, day): {op_per_shift.mean():.2f}")

# ── total_trip as trip counter ────────────────────────────────────────────────
print("\n=== total_trip analysis ===")
train1 = pd.read_parquet("data/telemetry_2026-01-01_2026-01-10.parquet",
                         columns=["vehicle", "ts", "total_trip", "speed", "ignition"])

def assign_op_date_shift(ts_series):
    ts = pd.to_datetime(ts_series, utc=False)
    hour = ts.dt.hour
    date = ts.dt.date
    import datetime
    adj_date = pd.array([
        d - datetime.timedelta(days=1) if h >= 22 else d
        for d, h in zip(date, hour)
    ])
    shift = pd.array([
        "C" if h >= 22 or h < 6 else ("A" if h < 14 else "B")
        for h in hour
    ])
    return pd.Series(adj_date, index=ts_series.index), pd.Series(shift, index=ts_series.index)

train1["adj_date"], train1["shift_calc"] = assign_op_date_shift(train1["ts"])
# Max total_trip per shift = number of trips completed
max_trip = train1.groupby(["vehicle", "adj_date", "shift_calc"])["total_trip"].max().reset_index()
max_trip.columns = ["vehicle", "date", "shift", "n_trips"]
merged = smry.merge(max_trip, on=["vehicle", "date", "shift"], how="inner")
active = merged[(merged["acons"] > 50) & (merged["n_trips"].notna()) & (merged["n_trips"] > 0)]
corr = active["n_trips"].corr(active["acons"])
print(f"total_trip max vs acons correlation (active shifts): {corr:.4f}")

# Fuel per trip
active = active.copy()
active["fuel_per_trip"] = active["acons"] / active["n_trips"]
print(f"\nFuel per trip stats (active shifts):")
print(active["fuel_per_trip"].describe())
print()
# Per-vehicle fuel per trip
print("Per-vehicle mean fuel per trip:")
print(active.groupby("vehicle")["fuel_per_trip"].mean().sort_values().to_string())

# ── ignition-hours based model ────────────────────────────────────────────────
print("\n=== Physics-based lph model ===")
# Load ignition hours from feature file if available
try:
    train_feat = pd.read_csv("outputs/improved_features/train_improved.csv")
    if "ignition_on_hours" in train_feat.columns:
        act = train_feat[(train_feat["acons"] > 50) & (train_feat["ignition_on_hours"] > 0)].copy()
        act["lph"] = act["acons"] / act["ignition_on_hours"]
        print("lph = acons / ignition_on_hours (active shifts):")
        print(act["lph"].describe())
        print()
        # Predict using just lph * ignition_hours
        global_lph = act["lph"].median()
        pred_simple = global_lph * act["ignition_on_hours"]
        rmse_simple = np.sqrt(np.mean((act["acons"] - pred_simple)**2))
        print(f"Global median lph: {global_lph:.2f}")
        print(f"RMSE of (lph_median × ign_h): {rmse_simple:.2f}L")
        # Per-vehicle lph
        veh_lph = act.groupby("vehicle")["lph"].median()
        act["pred_veh_lph"] = act["vehicle"].map(veh_lph) * act["ignition_on_hours"]
        rmse_veh = np.sqrt(np.mean((act["acons"] - act["pred_veh_lph"])**2))
        print(f"RMSE of (veh_lph × ign_h): {rmse_veh:.2f}L")
except Exception as e:
    print(f"Could not load features: {e}")

# ── Missing predictions investigation ─────────────────────────────────────────
print("\n=== Missing predictions (92 in last submission) ===")
# Which test rows have no telemetry?
test_all = pd.concat([
    pd.read_parquet(f, columns=["vehicle", "ts"])
    for f in ["data/telemetry_2026-01-21_2026-01-31.parquet",
              "data/telemetry_2026-02-21_2026-02-28.parquet",
              "data/telemetry_2026-03-12_2026-03-20.parquet"]
])
import datetime
test_all["ts"] = pd.to_datetime(test_all["ts"], utc=False)
hour = test_all["ts"].dt.hour
adj_date = [d - datetime.timedelta(days=1) if h >= 22 else d
            for d, h in zip(test_all["ts"].dt.date, hour)]
shift = ["C" if h >= 22 or h < 6 else ("A" if h < 14 else "B") for h in hour]
test_all["adj_date"] = pd.to_datetime(adj_date).date if False else [str(x) for x in adj_date]
test_all["shift_calc"] = shift
test_tele_coverage = test_all.groupby(["vehicle", "adj_date", "shift_calc"]).size().reset_index()
test_tele_coverage.columns = ["vehicle", "date_str", "shift", "n_pings"]
idmap["date_str"] = idmap["date"].astype(str)
merged_cov = idmap.merge(test_tele_coverage, left_on=["vehicle", "date_str", "shift"],
                          right_on=["vehicle", "date_str", "shift"], how="left")
missing = merged_cov[merged_cov["n_pings"].isna()]
print(f"Test rows with NO telemetry: {len(missing)} / {len(idmap)}")
print(missing[["vehicle", "date_str", "shift"]].to_string())

print("\n=== Done ===")
