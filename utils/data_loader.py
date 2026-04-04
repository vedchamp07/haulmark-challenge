import argparse
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import fiona
except Exception:  # pragma: no cover
    fiona = None


TRAIN_WINDOWS = [
    (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-20")),
    (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-20")),
    (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-11")),
]
TEST_WINDOWS = [
    (pd.Timestamp("2026-01-21"), pd.Timestamp("2026-01-31")),
    (pd.Timestamp("2026-02-21"), pd.Timestamp("2026-02-28")),
    (pd.Timestamp("2026-03-12"), pd.Timestamp("2026-03-20")),
]


@dataclass
class DataFiles:
    telemetry_files: List[Path]
    fleet_file: Optional[Path]
    id_mapping_file: Optional[Path]
    refuel_file: Optional[Path]
    gpkg_files: List[Path]


def unzip_archives(data_dir: Path) -> None:
    zips = list(data_dir.parent.glob("*.zip")) + list(data_dir.glob("*.zip"))
    for zf in zips:
        print(f"[data_loader] Extracting: {zf}")
        with zipfile.ZipFile(zf, "r") as zobj:
            zobj.extractall(data_dir)


def discover_data_files(data_dir: Path) -> DataFiles:
    telemetry_files = sorted(
        [p for p in data_dir.glob("**/*") if p.is_file() and "telemetry_" in p.name and p.suffix in {".parquet", ".csv"}]
    )
    fleet_file = next(iter(sorted(data_dir.glob("**/fleet.csv"))), None)
    id_mapping_file = next(iter(sorted(data_dir.glob("**/id_mapping.csv"))), None)
    refuel_candidates = sorted(data_dir.glob("**/*refuel*.parquet")) + sorted(data_dir.glob("**/*refuel*.csv"))
    refuel_file = refuel_candidates[0] if refuel_candidates else None
    gpkg_files = sorted(data_dir.glob("**/*.gpkg"))
    return DataFiles(
        telemetry_files=telemetry_files,
        fleet_file=fleet_file,
        id_mapping_file=id_mapping_file,
        refuel_file=refuel_file,
        gpkg_files=gpkg_files,
    )


def _parse_file_window(path: Path) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", path.name)
    if not m:
        return None
    return pd.Timestamp(m.group(1)), pd.Timestamp(m.group(2))


def _in_any_window(start: pd.Timestamp, end: pd.Timestamp, windows: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
    for w0, w1 in windows:
        if start >= w0 and end <= w1:
            return True
    return False


def split_train_test_files(files: Iterable[Path]) -> Tuple[List[Path], List[Path]]:
    train_files: List[Path] = []
    test_files: List[Path] = []
    for f in files:
        wnd = _parse_file_window(f)
        if wnd is None:
            continue
        start, end = wnd
        if _in_any_window(start, end, TRAIN_WINDOWS):
            train_files.append(f)
        elif _in_any_window(start, end, TEST_WINDOWS):
            test_files.append(f)
    return sorted(train_files), sorted(test_files)


def read_table(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def load_telemetry(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = read_table(path, columns=columns)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if "received_ts" in df.columns:
        df["received_ts"] = pd.to_datetime(df["received_ts"], errors="coerce")
    return df


def load_id_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def inspect_file(path: Path) -> None:
    print(f"\n[data_loader] Inspecting {path.name}")
    df = read_table(path)
    print(f"shape={df.shape}")
    print(df.dtypes)
    print(df.head(3))


def print_gpkg_layers(gpkg_files: List[Path]) -> None:
    if fiona is None:
        print("[data_loader] fiona unavailable; skipping gpkg layer listing.")
        return
    for gpkg in gpkg_files:
        print(f"\n[data_loader] GeoPackage: {gpkg}")
        for layer in fiona.listlayers(gpkg):
            print(f" - {layer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unzip and inspect challenge data files")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    unzip_archives(data_dir)
    files = discover_data_files(data_dir)

    print("[data_loader] ===== Inventory =====")
    for p in files.telemetry_files:
        print(f"telemetry: {p}")
    print(f"fleet: {files.fleet_file}")
    print(f"id_mapping: {files.id_mapping_file}")
    print(f"refuel: {files.refuel_file}")
    for g in files.gpkg_files:
        print(f"gpkg: {g}")

    train_files, test_files = split_train_test_files(files.telemetry_files)
    print("\n[data_loader] Train telemetry files:")
    for p in train_files:
        print(f" - {p.name}")
    print("[data_loader] Test telemetry files:")
    for p in test_files:
        print(f" - {p.name}")

    for p in [files.fleet_file, files.id_mapping_file, files.refuel_file]:
        if p is not None:
            inspect_file(p)

    for p in train_files[:1] + test_files[:1]:
        inspect_file(p)

    print_gpkg_layers(files.gpkg_files)
    print("\n[data_loader] Done.")


if __name__ == "__main__":
    main()
