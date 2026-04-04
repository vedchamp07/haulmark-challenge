from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


BASE_TELEMETRY_COLS: List[str] = [
    "vehicle",
    "mine_anon",
    "ts",
    "received_ts",
    "latitude",
    "longitude",
    "altitude",
    "speed",
    "ignition",
    "angle",
    "satellites",
    "gnss_pdop",
    "gnss_hdop",
    "gsm_signal",
    "gsm_operator",
    "battery_level",
    "battery_current",
    "battery_voltage",
    "external_voltage",
    "axis_x",
    "axis_y",
    "axis_z",
    "analog_input_1",
    "disthav",
    "cumdist",
    "fuel_volume",
    "date_dpr",
]

DPR_TARGET_COLS = [
    "fuel_volume",
    "prod_hr_dpr",
    "idle_hr_dpr",
    "maint_hr_dpr",
    "bd_hr_dpr",
    "hmr_dpr",
    "km_dpr",
    "tonnage",
    "total_trip",
]


NON_FEATURE_DPR_COLUMNS = [
    "shift_dpr",
    "date_dpr",
    "operator_id",
    "hmr_dpr",
    "prod_hr_dpr",
    "idle_hr_dpr",
    "maint_hr_dpr",
    "bd_hr_dpr",
    "km_dpr",
    "total_trip",
    "tonnage",
    "rain_loss",
    "dense_fog",
]


def assign_operational_day(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    op_day = (out[ts_col] + pd.Timedelta(hours=2)).dt.date
    out["op_day"] = pd.to_datetime(op_day)

    if "date_dpr" in out.columns:
        date_dpr = pd.to_datetime(out["date_dpr"], errors="coerce")
        out["date_dpr"] = date_dpr
        out["date_key"] = date_dpr.fillna(out["op_day"])
    else:
        out["date_key"] = out["op_day"]

    return out


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cast_map = {
        "speed": "float32",
        "altitude": "float32",
        "ignition": "int8",
        "satellites": "float32",
        "gnss_pdop": "float32",
        "gnss_hdop": "float32",
        "disthav": "float32",
        "cumdist": "float32",
        "battery_level": "float32",
        "axis_x": "float32",
        "axis_y": "float32",
        "axis_z": "float32",
        "analog_input_1": "float32",
        "fuel_volume": "float32",
    }
    for c, t in cast_map.items():
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(t)
    for c in ["vehicle", "mine_anon"]:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def clean_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = assign_operational_day(out)
    out = optimize_dtypes(out)

    out = out.sort_values(["vehicle", "ts"])
    out = out.drop_duplicates(subset=["vehicle", "ts"], keep="last")

    if "gnss_hdop" in out.columns:
        out = out[(out["gnss_hdop"].isna()) | (out["gnss_hdop"] <= 5)]
    if "satellites" in out.columns:
        out = out[(out["satellites"].isna()) | (out["satellites"] >= 4)]

    if "speed" in out.columns:
        out["speed_raw"] = out["speed"]
        out = out[out["speed"] <= 80]
        out["speed"] = out["speed"].clip(lower=0, upper=60)

    if "altitude" in out.columns:
        out["altitude"] = (
            out.groupby("vehicle", observed=True)["altitude"]
            .transform(lambda s: s.interpolate(method="linear", limit=30, limit_direction="both"))
        )

    out["time_gap"] = (
        out.groupby("vehicle", observed=True)["ts"].diff().dt.total_seconds().fillna(0).clip(lower=0, upper=300)
    )

    if "fuel_volume" in out.columns:
        out["fuel_volume_med5"] = (
            out.groupby("vehicle", observed=True)["fuel_volume"]
            .transform(lambda s: s.rolling(window=5, min_periods=1).median())
            .astype("float32")
        )

    return out


def split_train_test_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out = assign_operational_day(out)

    train_mask = (
        ((out["date_key"] >= pd.Timestamp("2026-01-01")) & (out["date_key"] <= pd.Timestamp("2026-01-20")))
        | ((out["date_key"] >= pd.Timestamp("2026-02-01")) & (out["date_key"] <= pd.Timestamp("2026-02-20")))
    )
    test_mask = (
        ((out["date_key"] >= pd.Timestamp("2026-01-21")) & (out["date_key"] <= pd.Timestamp("2026-01-31")))
        | ((out["date_key"] >= pd.Timestamp("2026-02-21")) & (out["date_key"] <= pd.Timestamp("2026-02-28")))
    )

    return out[train_mask].copy(), out[test_mask].copy()
