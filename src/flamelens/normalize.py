"""Flow 2: map raw NASA columns to canonical names, units and roles.

Reads data/ingested/, writes data/normalized/<dataset_id>.parquet, a long table
(record_id, dataset, variable, value, unit, role) and normalize_report.json.
Raw files are never modified. Run with ``python -m flamelens.normalize``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from flamelens import ingest, paths, units

ROLES = {"INPUT", "OUTPUT", "ID", "META", "IGNORE"}
TYPES = {"float", "int", "category", "text", "date", "time"}
DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d")
REGISTRY_FIELDS = ["experiment", "category", "platform", "gravity"]


class SchemaError(ValueError):
    """The column map does not match the data."""


def column_map() -> dict[str, dict[str, dict[str, Any]]]:
    """Load schema/column_map.yaml."""
    return paths.load_yaml(paths.COLUMN_MAP)


def valid_ranges() -> dict[str, dict[str, Any]]:
    """Load schema/valid_ranges.yaml."""
    return paths.load_yaml(paths.VALID_RANGES)


def check_map(dataset_id: str, raw_columns: list[str], cmap: dict[str, dict[str, Any]]) -> None:
    """Fail loudly if any raw column is unmapped, a mapping is stale, or a spec is invalid."""
    unmapped = [c for c in raw_columns if c not in cmap]
    stale = [c for c in cmap if c not in raw_columns]
    problems = []
    if unmapped:
        problems.append(f"raw columns missing from column_map.yaml: {unmapped}")
    if stale:
        problems.append(f"column_map.yaml lists columns not in the data: {stale}")
    names: dict[str, str] = {}
    for raw, spec in cmap.items():
        if spec.get("role") not in ROLES:
            problems.append(f"{raw!r}: role must be one of {sorted(ROLES)}")
        if spec.get("type") not in TYPES:
            problems.append(f"{raw!r}: type must be one of {sorted(TYPES)}")
        if spec.get("role") != "IGNORE":
            if spec["name"] in names:
                problems.append(f"{raw!r} and {names[spec['name']]!r} both map to {spec['name']!r}")
            names[spec["name"]] = raw
    if problems:
        raise SchemaError(f"{dataset_id}: " + "; ".join(problems))


def _parse_date(text: str) -> str | None:
    text = text.strip()
    if units.is_missing(text):
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date {text!r}")


def _category(raw: pd.Series, values: dict[str, str] | None, column: str) -> pd.Series:
    text = raw.str.strip()
    out = text.where(text != "", None)
    if values is None:
        return out.astype("string")
    lookup = {str(k).strip().lower(): v for k, v in values.items()}
    unknown = sorted({v for v in out.dropna() if v.lower() not in lookup})
    if unknown:
        raise SchemaError(f"{column!r}: values {unknown} have no entry in column_map 'values'")
    return out.map(lambda v: lookup[v.lower()] if isinstance(v, str) else None).astype("string")


def normalize_dataset(dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one ingested dataset. Returns the wide frame and a report."""
    df = ingest.load(dataset_id)
    cmap = column_map()[dataset_id]
    raw_columns = [c for c in df.columns if c not in ingest.PROVENANCE]
    check_map(dataset_id, raw_columns, cmap)
    ranges = valid_ranges()
    entry = paths.registry_entry(dataset_id)

    out = df[ingest.PROVENANCE].copy()
    report: dict[str, Any] = {"flags": {}, "unparseable": {}, "approximate": {}}

    for raw, spec in cmap.items():
        if spec["role"] == "IGNORE":
            continue
        name, kind, col = spec["name"], spec["type"], df[raw]
        flag = pd.Series(False, index=df.index)

        if kind in ("float", "int"):
            values, bad = units.parse_series(col, spec.get("unit_in"), spec.get("unit"))
            if bad.any():
                report["unparseable"][name] = _examples(df, col, bad)
            flag |= bad
            approx = col.map(units.is_approximate)
            if approx.any():
                out[f"approx_{name}"] = approx
                report["approximate"][name] = df.loc[approx, "_record_id"].tolist()
            limits = ranges.get(name)
            if limits:
                too_low = values < limits["min"] if not limits.get("exclusive_min") else values <= limits["min"]
                outside = (too_low | (values > limits["max"])).fillna(False).astype(bool)
                if outside.any():
                    report["flags"][name] = _examples(df, col, outside)
                    values = values.mask(outside)
                    flag |= outside
            if kind == "int" and values.dropna().mod(1).eq(0).all():
                values = values.astype("Int64")
            out[name] = values
        elif kind == "category":
            out[name] = _category(col, spec.get("values"), raw)
        elif kind == "text":
            text = col.str.strip()
            out[name] = text.where(text != "", None).astype("string")
        elif kind == "date":
            parsed = []
            for text in col:
                try:
                    parsed.append(_parse_date(text))
                except ValueError:
                    parsed.append(None)
            out[name] = pd.Series(parsed, index=df.index, dtype="string")
            bad = out[name].isna() & ~col.map(units.is_missing)
            if bad.any():
                report["unparseable"][name] = _examples(df, col, bad)
            flag |= bad
        elif kind == "time":
            text = col.str.strip()
            out[name] = text.where(~text.map(units.is_missing), None).astype("string")

        if flag.any():
            out[f"flag_{name}"] = flag

    for field in REGISTRY_FIELDS:
        out[field] = entry[field]

    out, dropped = _drop_duplicates(out, cmap)
    report.update(rows_in=len(df), rows_out=len(out), duplicates_dropped=dropped)
    return out.reset_index(drop=True), report


def _examples(df: pd.DataFrame, col: pd.Series, mask: pd.Series) -> list[dict[str, str]]:
    return [{"record_id": r, "raw": v} for r, v in zip(df.loc[mask, "_record_id"], col[mask])]


def _drop_duplicates(out: pd.DataFrame, cmap: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    """Drop exact duplicates on INPUT columns plus timestamps.

    Datasets without a date or time column use their ID columns instead, so two
    distinct tests at the same conditions are never merged.
    """
    specs = [s for s in cmap.values() if s["role"] != "IGNORE"]
    keys = [s["name"] for s in specs if s["role"] == "INPUT"]
    stamps = [s["name"] for s in specs if s["role"] == "META" and s["type"] in ("date", "time")]
    keys += stamps or [s["name"] for s in specs if s["role"] == "ID"]
    dup = out.duplicated(subset=keys, keep="first")
    return out.loc[~dup], out.loc[dup, "_record_id"].tolist()


def long_table(wide: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Melt one normalized dataset into (record_id, dataset, variable, value, value_num, unit, role)."""
    cmap = column_map()[dataset_id]
    rows = []
    for spec in cmap.values():
        if spec["role"] == "IGNORE":
            continue
        name = spec["name"]
        series = wide[name]
        numeric = spec["type"] in ("float", "int")
        rows.append(pd.DataFrame({
            "record_id": wide["_record_id"],
            "dataset": dataset_id,
            "variable": name,
            "value": series.astype("string"),
            "value_num": series.astype("Float64") if numeric else pd.Series(pd.NA, index=wide.index, dtype="Float64"),
            "unit": spec.get("unit"),
            "role": spec["role"],
        }))
    long = pd.concat(rows, ignore_index=True)
    return long[long["value"].notna()].reset_index(drop=True)


def load(dataset_id: str) -> pd.DataFrame:
    """Load one normalized dataset."""
    return pd.read_parquet(paths.NORMALIZED / f"{dataset_id}.parquet")


def run() -> dict[str, dict[str, Any]]:
    """Normalize every registered dataset."""
    paths.NORMALIZED.mkdir(parents=True, exist_ok=True)
    reports, longs = {}, []
    for entry in paths.registry():
        dataset_id = entry["dataset_id"]
        wide, reports[dataset_id] = normalize_dataset(dataset_id)
        wide.to_parquet(paths.NORMALIZED / f"{dataset_id}.parquet", index=False)
        longs.append(long_table(wide, dataset_id))
    pd.concat(longs, ignore_index=True).to_parquet(paths.NORMALIZED / "long.parquet", index=False)
    paths.write_json(paths.NORMALIZED / "normalize_report.json", reports)
    return reports


if __name__ == "__main__":
    for dataset_id, r in run().items():
        flagged = {k: len(v) for k, v in r["flags"].items()}
        print(f"{dataset_id:24s} rows={r['rows_out']:4d} dup_dropped={len(r['duplicates_dropped'])} "
              f"flags={flagged} unparseable={ {k: len(v) for k, v in r['unparseable'].items()} }")
