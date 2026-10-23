"""Project paths and small YAML/JSON helpers shared by every flow."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("FLAMELENS_ROOT", Path(__file__).resolve().parents[2]))

DATA = ROOT / "data"
RAW = DATA / "raw"
INGESTED = DATA / "ingested"
NORMALIZED = DATA / "normalized"
TASKS = DATA / "tasks"
LABELS = DATA / "labels"
REGISTRY = DATA / "registry.yaml"

SCHEMA = ROOT / "schema"
COLUMN_MAP = SCHEMA / "column_map.yaml"
TARGETS = SCHEMA / "targets.yaml"
VALID_RANGES = SCHEMA / "valid_ranges.yaml"

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def load_yaml(path: Path) -> Any:
    """Read a YAML file."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_json(path: Path, obj: Any) -> None:
    """Write pretty JSON, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
        fh.write("\n")


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def registry() -> list[dict[str, Any]]:
    """Return the dataset entries from data/registry.yaml."""
    return load_yaml(REGISTRY)["datasets"]


def registry_entry(dataset_id: str) -> dict[str, Any]:
    """Return one registry entry by dataset_id."""
    for entry in registry():
        if entry["dataset_id"] == dataset_id:
            return entry
    raise KeyError(f"{dataset_id} is not in {REGISTRY}")
