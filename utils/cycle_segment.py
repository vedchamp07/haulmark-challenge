from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.data_loader import discover_data_files, split_train_test_files, load_telemetry
from utils.preprocessing import clean_telemetry


@dataclass
class CycleSummary:
    cycles_count: int
    mean_cycle_duration_min: float
    mean_cycle_distance_km: float
    mean_net_lift: float
    total_cycle_distance_km: float


def segment_cycles_one_group(group: pd.DataFrame) -> List[Dict]:
    g = group.sort_values("ts").copy()
    g["time_gap"] = g["time_gap"].fillna(0)
    g["speed"] = g["speed"].fillna(0)

    if "analog_input_1" in g.columns and g["analog_input_1"].notna().any():
        prev = g["analog_input_1"].shift(1).fillna(-np.inf)
        g["dump_edge"] = (g["analog_input_1"] > 2.5) & (prev <= 2.5)
        dump_times = g.loc[g["dump_edge"], "ts"].sort_values().tolist()
        cycles = []
        for i in range(1, len(dump_times)):
            c = g[(g["ts"] >= dump_times[i - 1]) & (g["ts"] <= dump_times[i])]
            if len(c) < 2:
                continue
            cycles.append(
                {
                    "start_ts": c["ts"].iloc[0],
                    "end_ts": c["ts"].iloc[-1],
                    "duration_min": (c["ts"].iloc[-1] - c["ts"].iloc[0]).total_seconds() / 60.0,
                    "distance_km": c["disthav"].fillna(0).sum() / 1000.0,
                    "net_lift": c["altitude"].iloc[-1] - c["altitude"].iloc[0],
                }
            )
        return cycles

    g["is_slow"] = g["speed"] < 3
    g["slow_segment"] = (g["is_slow"] != g["is_slow"].shift()).cumsum()

    dwell = (
        g[g["is_slow"]]
        .groupby("slow_segment", observed=True)
        .agg(start_ts=("ts", "first"), end_ts=("ts", "last"), duration=("time_gap", "sum"))
        .reset_index(drop=True)
    )
    dwell = dwell[dwell["duration"] > 120]

    cycles: List[Dict] = []
    for i in range(0, max(len(dwell) - 1, 0), 2):
        start_ts = dwell.iloc[i]["start_ts"]
        end_ts = dwell.iloc[i + 1]["end_ts"]
        c = g[(g["ts"] >= start_ts) & (g["ts"] <= end_ts)]
        if len(c) < 2:
            continue
        cycles.append(
            {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_min": (end_ts - start_ts).total_seconds() / 60.0,
                "distance_km": c["disthav"].fillna(0).sum() / 1000.0,
                "net_lift": c["altitude"].iloc[-1] - c["altitude"].iloc[0],
            }
        )
    return cycles


def aggregate_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (vehicle, date_key), g in df.groupby(["vehicle", "date_key"], observed=True):
        cycles = segment_cycles_one_group(g)
        if cycles:
            rows.append(
                {
                    "vehicle": vehicle,
                    "date_key": pd.to_datetime(date_key),
                    "cycles_count": len(cycles),
                    "mean_cycle_duration_min": float(np.mean([c["duration_min"] for c in cycles])),
                    "mean_cycle_distance_km": float(np.mean([c["distance_km"] for c in cycles])),
                    "mean_net_lift": float(np.mean([c["net_lift"] for c in cycles])),
                    "total_cycle_distance_km": float(np.sum([c["distance_km"] for c in cycles])),
                }
            )
        else:
            rows.append(
                {
                    "vehicle": vehicle,
                    "date_key": pd.to_datetime(date_key),
                    "cycles_count": 0,
                    "mean_cycle_duration_min": 0.0,
                    "mean_cycle_distance_km": 0.0,
                    "mean_net_lift": 0.0,
                    "total_cycle_distance_km": 0.0,
                }
            )
    return pd.DataFrame(rows)


def run_cycle_diagnostics(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_data_files(data_dir)
    train_files, _ = split_train_test_files(files.telemetry_files)
    use_file = train_files[0]
    print(f"[cycle_segment] Using {use_file.name} for diagnostics")

    cols = [
        "vehicle",
        "ts",
        "date_dpr",
        "speed",
        "ignition",
        "analog_input_1",
        "disthav",
        "altitude",
        "gnss_hdop",
        "satellites",
    ]
    df = load_telemetry(use_file, columns=cols)
    df = clean_telemetry(df)

    top_vehicles = df["vehicle"].value_counts().head(3).index.tolist()
    for vehicle in top_vehicles:
        subset = df[df["vehicle"] == vehicle].head(30000).copy()
        if subset.empty:
            continue
        cycles = aggregate_cycle_features(subset)
        print(f"[cycle_segment] {vehicle}: days={len(cycles)} avg_cycles={cycles['cycles_count'].mean():.2f}")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(subset["ts"], subset["speed"], lw=0.8, label="speed")
        if "analog_input_1" in subset.columns:
            ax2 = ax.twinx()
            ax2.plot(subset["ts"], subset["analog_input_1"], color="tab:orange", alpha=0.4, lw=0.7, label="analog_input_1")
            ax2.axhline(2.5, color="red", ls="--", lw=1)
            ax2.set_ylabel("analog_input_1")
        ax.set_title(f"Cycle Diagnostics {vehicle}")
        ax.set_ylabel("speed")
        ax.set_xlabel("ts")
        fig.tight_layout()
        fig.savefig(output_dir / f"cycle_diagnostics_{vehicle}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cycle segmentation diagnostics")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="eda")
    args = parser.parse_args()
    run_cycle_diagnostics(Path(args.data_dir), Path(args.output_dir))
