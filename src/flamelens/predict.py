"""Gate, predict and gather evidence for one extinction request (Flows 4-6 combined).

This is what /api/extinction/predict serves. No probability leaves this module unless
the coverage gate passes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from flamelens import coverage, similar, tasks, train

TASK = "ext_flex"


@lru_cache(maxsize=None)
def bundle(task: str = TASK) -> train.ModelBundle:
    return train.load_bundle(task)


@lru_cache(maxsize=None)
def coverage_model(task: str = TASK) -> coverage.CoverageModel:
    return coverage.load(task)


def band(prob: float) -> str:
    """Display band (not a physical threshold): Low < 0.30 <= Medium <= 0.70 < High."""
    if prob < 0.3:
        return "Low"
    if prob > 0.7:
        return "High"
    return "Medium"


def score_frame(df: pd.DataFrame, task: str = TASK) -> pd.DataFrame:
    """Probability, prediction set and coverage level for many query rows.

    Probability and set are withheld (None) where coverage is Low.
    """
    cov = coverage_model(task).assess_frame(df)
    out = cov.copy()
    out["probability"] = None
    out["prediction_set"] = None
    ok = (cov["level"] != "Low").to_numpy()
    if ok.any():
        model = bundle(task)
        rows = df.loc[ok]
        out.loc[ok, "probability"] = model.proba(rows)
        out.loc[ok, "prediction_set"] = pd.Series(model.prediction_sets(rows), index=rows.index)
    return out


def predict(conditions: dict[str, Any], task: str = TASK, k: int = 5) -> dict[str, Any]:
    """Full prediction response: status, coverage, neighbours and, when allowed, probability."""
    query = tasks.query_frame(task, conditions)
    row = query.iloc[0]
    model = bundle(task)
    cov = coverage_model(task).assess(row)
    neighbours = similar.similar_extinction(row, task, k)
    response: dict[str, Any] = {
        "coverage": {key: cov[key] for key in ("level", "score", "evidence_count", "out_of_range")},
        "coverage_detail": {key: cov.get(key) for key in ("reason", "out_of_range_detail", "group", "n_group")},
        "neighbors": [n["record_id"] for n in neighbours],
        "neighbor_details": neighbours,
        "neighbor_summary": similar.neighbor_summary(neighbours, task),
        "conditions": {c: (None if pd.isna(row[c]) else (row[c] if isinstance(row[c], str) else float(row[c])))
                       for c in model.features},
        "model_card": model.version,
        "warnings": [],
    }
    if cov["level"] == "Low":
        return {"status": "unavailable", "reason": cov["reason"], **response}

    prob = float(model.proba(query)[0])
    pset = model.prediction_sets(query)[0]
    status = coverage.status(cov, pset)
    if not similar.agreement(prob, neighbours):
        response["warnings"].append("Model and nearest experiments disagree")
        if status == "supported":
            status = "uncertain"
    if status == "uncertain" and len(pset) != 1:
        response["warnings"].append("The 90% prediction set holds both outcomes")
    if status == "limited":
        response["warnings"].append("Limited NASA evidence near these conditions")
    return {"status": status, "probability": prob, "band": band(prob), "prediction_set": pset, **response}
