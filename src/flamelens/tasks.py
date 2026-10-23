"""Flow 3: build one table per prediction task with a defined target and group column.

Run with ``python -m flamelens.tasks``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from flamelens import normalize, paths
from flamelens.features import (
    CATEGORICAL, CV_SPLITS, FEATURES, GROUP, NULLABLE, OUTPUTS, SOURCES, TARGET, TARGET_SOURCE,
)

PROVENANCE = ["_record_id", "_dataset_id", "_source_file", "_source_row"]
MIN_GROUPS_PER_FOLD = 5


class TargetNotDefined(RuntimeError):
    """schema/targets.yaml has an unmapped or TODO outcome value."""


def diluent(x_he: pd.Series, x_co2: pd.Series) -> pd.Series:
    """He if x_he > 0.01, CO2 if x_co2 > 0.01, otherwise N2."""
    out = np.where(x_he > 0.01, "He", np.where(x_co2 > 0.01, "CO2", "N2"))
    return pd.Series(out, index=x_he.index, dtype="string")


def add_flex_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the FLEX derived inputs. Works on task rows and on user queries alike."""
    df = df.copy()
    df["p_o2_atm"] = df["x_o2"] * df["p_atm"]
    df["inert_to_o2"] = (1 - df["x_o2"]) / df["x_o2"]
    df["co2_to_o2"] = df["x_co2"] / df["x_o2"]
    df["he_to_o2"] = df["x_he"] / df["x_o2"]
    df["d0_sq"] = df["d0_mm"] ** 2
    if "diluent" not in df or df["diluent"].isna().any():
        df["diluent"] = diluent(df["x_he"], df["x_co2"])
    return df


def target_map(task: str) -> dict[str, Any]:
    """Return the targets.yaml entry for a task."""
    return paths.load_yaml(paths.TARGETS)[task]


def apply_target(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach label/censored columns and drop excluded rows. Stops on TODO or unmapped values."""
    spec = target_map(task)
    source = TARGET_SOURCE[task]
    labels = {str(k): v for k, v in spec["labels"].items()}
    counts = df[source].value_counts(dropna=False)
    unmapped = [str(v) for v in counts.index if str(v) not in labels]
    todo = [k for k, v in labels.items() if str(v).upper() == "TODO"]
    if unmapped or todo:
        raise TargetNotDefined(
            f"{task}: label these values in schema/targets.yaml before training: {sorted(set(unmapped + todo))}\n"
            f"value counts:\n{counts.to_string()}"
        )
    label = df[source].astype(str).map(labels)
    keep = label != "exclude"
    out = df.loc[keep].copy()
    out[TARGET[task]] = label[keep].astype(int)
    out["censored"] = out[source].isin(spec.get("censored") or [])
    info = {
        "value_counts": {str(k): int(v) for k, v in counts.items()},
        "excluded_by_label": int((~keep).sum()),
        "reviewed_by": spec.get("reviewed_by"),
    }
    return out, info


def build_ext_flex() -> tuple[pd.DataFrame, dict[str, Any]]:
    """data/tasks/ext_flex.parquet: FLEX droplet extinction."""
    df = normalize.load("PSI-69_FLEX")
    df = add_flex_derived(df)
    return _finish(df, "ext_flex")


def build_ext_cfig() -> tuple[pd.DataFrame, dict[str, Any]]:
    """data/tasks/ext_cfig.parquet: CFI-G normal and inverse gas-flame extinction."""
    parts = [normalize.load(d) for d in SOURCES["ext_cfig"]]
    df = pd.concat(parts, ignore_index=True)
    # Day numbers restart per campaign, so the group is flame type + day.
    df["group"] = df["flame_type"] + "-" + df["day"].astype(str)
    return _finish(df, "ext_cfig")


def _finish(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, info = apply_target(df, task)
    required = [c for c in FEATURES[task] if c not in NULLABLE[task]]
    missing = df[required].isna().any(axis=1)
    info["excluded_missing_inputs"] = df.loc[missing, "_record_id"].tolist()
    df = df.loc[~missing]

    keep = PROVENANCE + [GROUP[task]] + FEATURES[task] + [TARGET[task], "censored", TARGET_SOURCE[task]]
    out = df[list(dict.fromkeys(keep))].reset_index(drop=True)
    for col in CATEGORICAL[task]:
        out[col] = out[col].astype("string")
    for col in FEATURES[task]:
        if col not in CATEGORICAL[task]:
            out[col] = out[col].astype("float64")
    info.update(
        n_rows=len(out),
        class_balance={int(k): int(v) for k, v in out[TARGET[task]].value_counts().sort_index().items()},
        n_groups=int(out[GROUP[task]].nunique()),
        min_groups_per_fold=min_groups_per_fold(out, task),
    )
    return out, info


def min_groups_per_fold(df: pd.DataFrame, task: str, seeds=range(5)) -> int:
    """Smallest number of distinct groups in any test fold across the CV seeds."""
    smallest = np.inf
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=CV_SPLITS[task], shuffle=True, random_state=seed)
        for _, test in cv.split(df, df[TARGET[task]], df[GROUP[task]]):
            smallest = min(smallest, df[GROUP[task]].iloc[test].nunique())
    return int(smallest)


def load(task: str) -> pd.DataFrame:
    """Load one task table."""
    return pd.read_parquet(paths.TASKS / f"{task}.parquet")


class InvalidQuery(ValueError):
    """A prediction request has unknown categories or missing inputs (API answers 422)."""


def fuel_aliases(task: str) -> dict[str, str]:
    """Accepted fuel spellings (lower-case) -> canonical fuel name, from column_map.yaml."""
    cmap = normalize.column_map()
    aliases: dict[str, str] = {}
    for dataset in SOURCES[task]:
        for spec in cmap[dataset].values():
            if spec["name"] == "fuel":
                for raw, canonical in (spec.get("values") or {}).items():
                    aliases[str(raw).lower()] = canonical
                    aliases[canonical.lower()] = canonical
    if task == "ext_flex":
        aliases["heptane"] = "n-heptane"
    return aliases


def query_frame(task: str, conditions: dict[str, Any]) -> pd.DataFrame:
    """Turn user conditions into a one-row frame with every model input.

    Requires the base inputs; derived inputs are computed here, never taken from the caller.
    """
    if task != "ext_flex":
        raise InvalidQuery(f"predictions are only available for ext_flex, not {task}")
    fuel = str(conditions.get("fuel", "")).strip().lower()
    aliases = fuel_aliases(task)
    if fuel not in aliases:
        raise InvalidQuery(f"unknown fuel {conditions.get('fuel')!r}; known fuels: {sorted(set(aliases.values()))}")
    row: dict[str, Any] = {"fuel": aliases[fuel]}
    gases = {"x_co2", "x_he"}
    for col in ["x_o2", "x_co2", "x_he", "p_atm", "d0_mm"]:
        value = conditions.get(col)
        if value is None and col in gases:
            value = 0.0  # an unstated diluent fraction means that gas is absent, as in the NASA tables
        if value is None:
            raise InvalidQuery(f"missing input {col!r}")
        try:
            row[col] = float(value)
        except (TypeError, ValueError):
            raise InvalidQuery(f"{col!r} must be a number, got {value!r}") from None
    if not 0 < row["x_o2"] <= 1:
        raise InvalidQuery("x_o2 must be a mole fraction between 0 and 1")
    if row["x_o2"] + row["x_co2"] + row["x_he"] > 1.0 + 1e-6:
        raise InvalidQuery("x_o2 + x_co2 + x_he cannot exceed 1")
    if row["p_atm"] <= 0 or row["d0_mm"] <= 0:
        raise InvalidQuery("p_atm and d0_mm must be positive")
    df = pd.DataFrame([row])
    df["fuel"] = df["fuel"].astype("string")
    if conditions.get("diluent") is not None:
        df["diluent"] = pd.Series([str(conditions["diluent"])], dtype="string")
    return add_flex_derived(df)


BUILDERS = {"ext_flex": build_ext_flex, "ext_cfig": build_ext_cfig}


def run(verbose: bool = True) -> dict[str, dict[str, Any]]:
    """Build every task table and data/tasks/tasks_report.json."""
    paths.TASKS.mkdir(parents=True, exist_ok=True)
    reports = {}
    for task, build in BUILDERS.items():
        df, info = build()
        leak = set(FEATURES[task]) & set(OUTPUTS[task])
        if leak:
            raise RuntimeError(f"{task}: post-test fields in FEATURES: {sorted(leak)}")
        if info["min_groups_per_fold"] < MIN_GROUPS_PER_FOLD:
            raise RuntimeError(f"{task}: a validation fold has only {info['min_groups_per_fold']} groups")
        df.to_parquet(paths.TASKS / f"{task}.parquet", index=False)
        reports[task] = info
        if verbose:
            print(f"\n== {task}: {info['n_rows']} rows, {info['n_groups']} groups "
                  f"(min {info['min_groups_per_fold']} per fold)")
            print(f"   raw outcome counts: {info['value_counts']}")
            print(f"   class balance: {info['class_balance']}")
            print(f"   excluded by label: {info['excluded_by_label']}, "
                  f"missing inputs: {len(info['excluded_missing_inputs'])}")
            if info["reviewed_by"] is None:
                print("   NOTE: target mapping in schema/targets.yaml is provisional (reviewed_by is null)")
    paths.write_json(paths.TASKS / "tasks_report.json", reports)
    return reports


if __name__ == "__main__":
    run()
