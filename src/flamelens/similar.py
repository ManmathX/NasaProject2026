"""Flow 6: nearest real NASA tests for any request, with or without a model.

Extinction reuses the Flow 5 coverage space (same scaling and fuel + diluent filters).
Spread and soot explorers use Gower distance over mixed numeric/categorical columns.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import pandas as pd

from flamelens import coverage, ingest, normalize, paths, tasks
from flamelens.features import CLASS_NAMES, COVERAGE_NUMERIC, TARGET


class NotAvailable(RuntimeError):
    """The module's explorer table has not been built yet."""


@lru_cache(maxsize=None)
def _raw(dataset_id: str) -> pd.DataFrame:
    return ingest.load(dataset_id).set_index("_record_id")


@lru_cache(maxsize=None)
def _task(task: str) -> pd.DataFrame:
    return tasks.load(task).set_index("_record_id")


@lru_cache(maxsize=None)
def _coverage(task: str) -> coverage.CoverageModel:
    return coverage.load(task)


def _raw_inputs(dataset_id: str) -> list[str]:
    """Raw headers of the ID, INPUT and date/time META columns, in file order."""
    cmap = normalize.column_map()[dataset_id]
    return [raw for raw, s in cmap.items()
            if s["role"] in ("ID", "INPUT") or (s["role"] == "META" and s["type"] in ("date", "time"))]


def record(record_id: str) -> dict[str, Any]:
    """One NASA record in its original columns and text, plus provenance."""
    dataset_id = record_id.rsplit(":", 1)[0]
    try:
        row = _raw(dataset_id).loc[record_id]
    except (KeyError, FileNotFoundError):
        raise KeyError(f"unknown record {record_id!r}") from None
    return {"record_id": record_id, "dataset_id": dataset_id, "source_file": row["_source_file"],
            "source_row": int(row["_source_row"]),
            "values": {c: row[c] for c in row.index if not c.startswith("_")}}


def similar_extinction(query: pd.Series | dict, task: str = "ext_flex", k: int = 5) -> list[dict[str, Any]]:
    """Top-k nearest training tests in the coverage space of the query's fuel + diluent group."""
    cov = _coverage(task)
    group = cov.group_for(query)
    if group is None:
        return []
    values = np.array([float(query[c]) for c in COVERAGE_NUMERIC[task]])
    dist, idx = group.kneighbors(values, k)
    table = _task(task)
    results = []
    for d, i in zip(dist, idx):
        rid = group.record_ids[i]
        dataset_id = rid.rsplit(":", 1)[0]
        raw = _raw(dataset_id).loc[rid]
        label = int(table.loc[rid, TARGET[task]])
        results.append({
            "record_id": rid,
            "distance": float(d),
            "similarity": float(1.0 / (1.0 + d)),
            "conditions": {c: raw[c] for c in _raw_inputs(dataset_id)},
            "outcome": str(table.loc[rid, "test_end"]),
            "label": label,
            "label_name": CLASS_NAMES[task][label],
            "source_file": raw["_source_file"],
            "source_row": int(raw["_source_row"]),
        })
    return results


def similar(task: str, conditions: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    """Nearest NASA tests for any module: 'ext_flex', 'spread' or 'soot'."""
    if task == "ext_flex":
        return similar_extinction(tasks.query_frame(task, conditions).iloc[0], task, k)
    if task in EXPLORERS:
        return similar_explorer(task, conditions, k)
    raise NotAvailable(f"no similarity search for {task!r}")


def neighbor_summary(results: Sequence[dict[str, Any]], task: str = "ext_flex") -> str:
    """e.g. '4 of 5 extinguished'."""
    if not results:
        return "no comparable NASA tests"
    n_pos = sum(r["label"] == 1 for r in results)
    return f"{n_pos} of {len(results)} {CLASS_NAMES[task][1].replace('_', ' ')}"


def agreement(prob: float, results: Sequence[dict[str, Any]]) -> bool:
    """False when the model says >= 70% but most neighbours did not extinguish, or <= 30% and most did."""
    if not results:
        return True
    share = sum(r["label"] == 1 for r in results) / len(results)
    if prob >= 0.7 and share < 0.5:
        return False
    if prob <= 0.3 and share > 0.5:
        return False
    return True


# ---------------------------------------------------------------- Gower (spread / soot explorers)

EXPLORERS = {
    "spread": {"table": "spread_explorer",
               "numeric": ["x_o2", "flow_cm_s", "thickness_mm"],
               "categorical": ["material", "flow_direction"]},
    "soot": {"table": "soot_explorer",
             "numeric": ["nozzle_mm", "fuel_rate_mg_s", "air_rate_cm_s"],
             "categorical": ["fuel"]},
}


def gower_distances(table: pd.DataFrame, query: dict[str, Any], numeric: list[str], categorical: list[str],
                    weights: dict[str, float] | None = None) -> np.ndarray:
    """Gower distance from one query to every row.

    Numeric columns score |a - b| / range; categorical score 0 if equal, 1 if not.
    A column that is null in the query or in a row is skipped for that pair and the
    remaining weights are renormalized.
    """
    weights = weights or {}
    num = np.zeros(len(table))
    den = np.zeros(len(table))
    for col in numeric:
        q = query.get(col)
        if q is None or pd.isna(q):
            continue
        vals = table[col].astype(float).to_numpy()
        span = np.nanmax(vals) - np.nanmin(vals) if np.isfinite(vals).any() else 0.0
        part = np.abs(vals - float(q)) / span if span > 0 else np.where(vals == float(q), 0.0, 1.0)
        ok = ~np.isnan(vals)
        w = weights.get(col, 1.0)
        num[ok] += w * np.minimum(part[ok], 1.0)
        den[ok] += w
    for col in categorical:
        q = query.get(col)
        if q is None or pd.isna(q):
            continue
        vals = table[col]
        ok = vals.notna().to_numpy()
        w = weights.get(col, 1.0)
        diff = (vals.astype(str).str.lower() != str(q).lower()).to_numpy(dtype=float)
        num[ok] += w * diff[ok]
        den[ok] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def similar_explorer(task: str, conditions: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    """Gower nearest neighbours over an explorer table. No prediction is made."""
    spec = EXPLORERS[task]
    path = paths.TASKS / f"{spec['table']}.parquet"
    if not path.exists():
        raise NotAvailable(f"{spec['table']} has not been built (its NASA CSVs are not in data/raw yet)")
    table = pd.read_parquet(path)
    dist = gower_distances(table, conditions, spec["numeric"], spec["categorical"])
    order = np.argsort(np.where(np.isnan(dist), np.inf, dist))[:k]
    out = []
    for i in order:
        if np.isnan(dist[i]):
            break
        rid = table["record_id"].iloc[i]
        out.append({"record_id": rid, "distance": float(dist[i]), "similarity": float(1 / (1 + dist[i])),
                    **{k_: v for k_, v in record(rid).items() if k_ != "record_id"}})
    return out
