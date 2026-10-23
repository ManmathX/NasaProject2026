"""Flow 5: data-coverage score and prediction availability.

Before any model output is shown, a request must sit inside the tested range of its
fuel + diluent group and close enough to real NASA tests. Distance thresholds are
calibrated from the data itself (leave-one-out 5-NN distances). Run with
``python -m flamelens.coverage`` after training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from flamelens import paths, tasks
from flamelens.features import COVERAGE_FILTERS, COVERAGE_MIN_SCALE, COVERAGE_NUMERIC

K = 5
LEVELS = ("High", "Medium", "Low")


@dataclass
class GroupSpace:
    """Scaled coverage space for one filter group (e.g. methanol + CO2)."""

    key: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    X: np.ndarray
    record_ids: np.ndarray
    nn: NearestNeighbors
    k: int
    loo: np.ndarray
    d75: float
    d95: float
    d99: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.median) / self.scale

    def kneighbors(self, values: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Distances and indices of the k nearest training tests to one raw-unit query."""
        k = min(k or self.k, len(self.X))
        dist, idx = self.nn.kneighbors(self.transform(values).reshape(1, -1), n_neighbors=k)
        return dist[0], idx[0]


class CoverageModel:
    """Per-group coverage spaces for one task."""

    def __init__(self, task: str, groups: dict[tuple[str, ...], GroupSpace]):
        self.task = task
        self.numeric = COVERAGE_NUMERIC[task]
        self.filters = COVERAGE_FILTERS[task]
        self.groups = groups

    @classmethod
    def fit(cls, df: pd.DataFrame, task: str = "ext_flex", k: int = K) -> "CoverageModel":
        """Fit median/IQR scaling, tested min/max and a k-NN index per filter group."""
        numeric, filters = COVERAGE_NUMERIC[task], COVERAGE_FILTERS[task]
        floor = np.array([COVERAGE_MIN_SCALE[task][c] for c in numeric])
        groups = {}
        for key, g in df.groupby(filters, sort=True):
            key = tuple(str(v) for v in (key if isinstance(key, tuple) else (key,)))
            raw = g[numeric].to_numpy(dtype=float)
            if len(raw) < 2:
                continue
            median = np.median(raw, axis=0)
            q75, q25 = np.percentile(raw, [75, 25], axis=0)
            scale = np.maximum(q75 - q25, floor)
            X = (raw - median) / scale
            kk = min(k, len(X) - 1)
            nn = NearestNeighbors().fit(X)
            loo = _leave_one_out(nn, X, kk)
            d75, d95, d99 = np.percentile(loo, [75, 95, 99])
            groups[key] = GroupSpace(key, median, scale, raw.min(axis=0), raw.max(axis=0), X,
                                     g["_record_id"].to_numpy(), nn, kk, loo, float(d75), float(d95), float(d99))
        return cls(task, groups)

    def group_for(self, query: pd.Series | dict) -> GroupSpace | None:
        key = tuple(str(query[f]) for f in self.filters)
        return self.groups.get(key)

    def assess(self, query: pd.Series | dict) -> dict[str, Any]:
        """Coverage level, score, evidence count and out-of-range variables for one query row."""
        group = self.group_for(query)
        key = {f: str(query[f]) for f in self.filters}
        if group is None:
            return {"level": "Low", "score": 0.0, "evidence_count": 0, "out_of_range": [],
                    "mean_distance": None, "group": key, "n_group": 0,
                    "reason": f"no NASA tests with {', '.join(f'{k}={v}' for k, v in key.items())}"}
        values = np.array([float(query[c]) for c in self.numeric])
        if np.isnan(values).any():
            raise tasks.InvalidQuery("coverage needs every numeric input")
        tol = 1e-9 * np.maximum(1.0, np.abs(group.hi))
        out = [{"variable": c, "value": float(v), "tested_min": float(lo), "tested_max": float(hi),
                "direction": "below" if v < lo else "above"}
               for c, v, lo, hi, t in zip(self.numeric, values, group.lo, group.hi, tol) if v < lo - t or v > hi + t]
        dist, _ = group.kneighbors(values)
        mean_d = float(dist.mean())
        all_d = np.linalg.norm(group.X - group.transform(values), axis=1)
        if out:
            level = "Low"
        elif mean_d <= group.d75:
            level = "High"
        elif mean_d <= group.d95:
            level = "Medium"
        else:
            level = "Low"
        reason = None
        if out:
            reason = "; ".join(f"{o['variable']} {o['direction']} the tested range "
                               f"[{o['tested_min']:g}, {o['tested_max']:g}]" for o in out)
        elif level == "Low":
            reason = "too far from any tested combination of conditions"
        return {
            "level": level,
            "score": float((group.loo > mean_d).mean()),
            "evidence_count": int((all_d <= group.d95).sum()),
            "out_of_range": [o["variable"] for o in out],
            "out_of_range_detail": out,
            "mean_distance": mean_d,
            "thresholds": {"d75": group.d75, "d95": group.d95, "d99": group.d99},
            "group": key,
            "n_group": int(len(group.X)),
            "reason": reason,
        }

    def assess_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized assess() for many rows: level, score, evidence_count, mean_distance, n_out_of_range."""
        out = pd.DataFrame({"level": "Low", "score": 0.0, "evidence_count": 0,
                            "mean_distance": np.nan, "n_out_of_range": 0}, index=df.index)
        for key, rows in df.groupby([df[f].astype(str) for f in self.filters], sort=False):
            group = self.groups.get(key if isinstance(key, tuple) else (key,))
            if group is None:
                continue
            values = rows[self.numeric].to_numpy(dtype=float)
            tol = 1e-9 * np.maximum(1.0, np.abs(group.hi))
            n_out = ((values < group.lo - tol) | (values > group.hi + tol)).sum(axis=1)
            scaled = group.transform(values)
            dist, _ = group.nn.kneighbors(scaled, n_neighbors=group.k)
            mean_d = dist.mean(axis=1)
            all_d = np.linalg.norm(scaled[:, None, :] - group.X[None, :, :], axis=2)
            level = np.where(n_out > 0, "Low",
                             np.where(mean_d <= group.d75, "High", np.where(mean_d <= group.d95, "Medium", "Low")))
            out.loc[rows.index, "level"] = level
            out.loc[rows.index, "score"] = (group.loo[None, :] > mean_d[:, None]).mean(axis=1)
            out.loc[rows.index, "evidence_count"] = (all_d <= group.d95).sum(axis=1)
            out.loc[rows.index, "mean_distance"] = mean_d
            out.loc[rows.index, "n_out_of_range"] = n_out
        return out


def _leave_one_out(nn: NearestNeighbors, X: np.ndarray, k: int) -> np.ndarray:
    """Mean distance from each training row to its k nearest *other* rows."""
    dist, idx = nn.kneighbors(X, n_neighbors=k + 1)
    out = np.empty(len(X))
    for i in range(len(X)):
        others = dist[i][idx[i] != i]
        out[i] = others[:k].mean()
    return out


def status(coverage: dict[str, Any], prediction_set: list[str]) -> str:
    """Spec Flow 5 table: supported | uncertain | limited | unavailable."""
    if coverage["level"] == "Low":
        return "unavailable"
    if coverage["level"] == "Medium":
        return "limited"
    return "supported" if len(prediction_set) == 1 else "uncertain"


def model_path(task: str = "ext_flex"):
    return paths.MODELS / task / "coverage.joblib"


def load(task: str = "ext_flex") -> CoverageModel:
    return joblib.load(model_path(task))


def validate(df: pd.DataFrame, task: str = "ext_flex") -> dict[str, Any]:
    """Error rate of the chosen model's out-of-fold predictions at each coverage level.

    Uses the outer CV folds saved by train.py: coverage is fitted on each fold's training
    rows and applied to its test rows, so no test row is ever in its own coverage space.
    """
    card = paths.read_json(paths.MODELS / task / "model_card.json")
    oof = pd.read_parquet(paths.MODELS / task / "oof.parquet")
    oof = oof[oof["model"] == card["served_model_cv"]]
    threshold = paths.read_json(paths.MODELS / task / "cv_report.json")["threshold"]
    by_id = df.set_index("_record_id")
    parts = []
    for (seed, fold), test in oof.groupby(["seed", "fold"]):
        train_ids = by_id.index.difference(test["_record_id"])
        cov = CoverageModel.fit(by_id.loc[train_ids].reset_index(), task)
        assessed = cov.assess_frame(by_id.loc[test["_record_id"]])
        parts.append(test.assign(level=assessed["level"].to_numpy()))
    res = pd.concat(parts, ignore_index=True)
    res["error"] = ((res["prob"] >= threshold).astype(int) != res["label"]).astype(int)
    res["sq_error"] = (res["prob"] - res["label"]) ** 2
    levels = {}
    for level in LEVELS:
        g = res[res["level"] == level]
        levels[level] = {"n": int(len(g)),
                         "error_rate": float(g["error"].mean()) if len(g) else None,
                         "brier": float(g["sq_error"].mean()) if len(g) else None}
    rates = [levels[l]["error_rate"] for l in LEVELS if levels[l]["error_rate"] is not None]
    report = {"model": card["served_model_cv"], "threshold": threshold, "levels": levels,
              "error_rises_high_to_low": all(a <= b for a, b in zip(rates, rates[1:]))}
    paths.write_json(paths.MODELS / task / "coverage_validation.json", report)
    return report


def run(task: str = "ext_flex") -> dict[str, Any]:
    """Fit the coverage model on the full task table, save it, and validate it on the CV folds."""
    df = tasks.load(task)
    model = CoverageModel.fit(df, task)
    joblib.dump(model, model_path(task))
    report = validate(df, task)
    for key, g in model.groups.items():
        print(f"{'/'.join(key):20s} n={len(g.X):3d} d75={g.d75:.3f} d95={g.d95:.3f} d99={g.d99:.3f}")
    print("\nError by coverage level (out-of-fold):")
    for level, v in report["levels"].items():
        rate = "n/a" if v["error_rate"] is None else f"{v['error_rate']:.3f}"
        brier = "n/a" if v["brier"] is None else f"{v['brier']:.3f}"
        print(f"  {level:7s} n={v['n']:4d} error_rate={rate} brier={brier}")
    print(f"  error rises High -> Low: {report['error_rises_high_to_low']}")
    return report


if __name__ == "__main__":
    # Call through the package so CoverageModel pickles as flamelens.coverage.CoverageModel.
    from flamelens import coverage as _coverage

    _coverage.run()
