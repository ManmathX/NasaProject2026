"""Flow 1: load each NASA CSV exactly as published and stamp every row with its provenance.

Run with ``python -m flamelens.ingest``.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from flamelens import paths

PROVENANCE = ["_dataset_id", "_source_file", "_source_row", "_record_id"]


def sha256(path: Path) -> str:
    """Return the hex sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_header(path: Path, encoding: str) -> list[str]:
    """Return the CSV header exactly as written in the file."""
    with open(path, encoding=encoding, newline="") as fh:
        return next(csv.reader(fh))


def read_raw(entry: dict[str, Any]) -> pd.DataFrame:
    """Read one registered CSV with every value kept as text.

    Fails if the file hash no longer matches the registry, or if the header has
    duplicate names (pandas would silently rename them).
    """
    path = paths.RAW / entry["file"]
    actual = sha256(path)
    if actual != entry["sha256"]:
        raise ValueError(f"{path.name}: sha256 {actual} does not match registry {entry['sha256']}")
    header = raw_header(path, entry["encoding"])
    if len(set(header)) != len(header):
        raise ValueError(f"{path.name}: duplicate column names in header {header}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=entry["encoding"])
    if list(df.columns) != header:
        raise ValueError(f"{path.name}: pandas changed the header")
    return df


def ingest_one(entry: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest one dataset; return the stamped frame and its report entry."""
    df = read_raw(entry)
    rows_in = len(df)
    df.insert(0, "_source_row", range(1, rows_in + 1))
    raw_cols = [c for c in df.columns if c != "_source_row"]
    blank = df[raw_cols].apply(lambda col: col.str.strip().eq("")).all(axis=1)
    df = df.loc[~blank].copy()

    dataset_id = entry["dataset_id"]
    df["_dataset_id"] = dataset_id
    df["_source_file"] = entry["file"]
    df["_record_id"] = dataset_id + ":" + df["_source_row"].astype(str)
    df = df[raw_cols + PROVENANCE].reset_index(drop=True)

    report = {
        "file": entry["file"],
        "rows_in": rows_in,
        "blank_dropped": int(blank.sum()),
        "rows_out": len(df),
        "sha256": entry["sha256"],
    }
    return df, report


def run() -> dict[str, dict[str, Any]]:
    """Ingest every registered dataset into data/ingested/."""
    paths.INGESTED.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict[str, Any]] = {}
    for entry in paths.registry():
        df, report[entry["dataset_id"]] = ingest_one(entry)
        df.to_parquet(paths.INGESTED / f"{entry['dataset_id']}.parquet", index=False)
    paths.write_json(paths.INGESTED / "ingest_report.json", report)
    return report


def load(dataset_id: str) -> pd.DataFrame:
    """Load one ingested dataset."""
    return pd.read_parquet(paths.INGESTED / f"{dataset_id}.parquet")


if __name__ == "__main__":
    for dataset_id, r in run().items():
        print(f"{dataset_id:24s} in={r['rows_in']:5d} blank={r['blank_dropped']:4d} out={r['rows_out']:4d}")
