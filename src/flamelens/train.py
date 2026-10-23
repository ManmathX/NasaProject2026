"""Flow 4: train and evaluate the FLEX extinction model.

Nested, grouped cross-validation over four model families; keeps the simplest model
within one SD of the best mean ROC-AUC, calibrates it, and wraps it with MAPIE for
90% conformal prediction sets. Run with ``python -m flamelens.train``.
"""

from __future__ import annotations

import subprocess
import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import joblib
import numpy as np
import pandas as pd
from mapie.classification import CrossConformalClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from flamelens import paths, tasks
from flamelens.features import CATEGORICAL, CLASS_NAMES, CV_SPLITS, FEATURES, GROUP, NUMERIC, TARGET

TASK = "ext_flex"
SEEDS = range(5)
INNER_SPLITS = 3
THRESHOLD = 0.5
CONFIDENCE = 0.9
# Simplest first: used to pick among models whose ROC-AUC is within 1 SD of the best.
SIMPLICITY = ["logreg", "rf", "hgb"]


def model_dir(task: str = TASK):
    return paths.MODELS / task


def _onehot(task: str):
    return ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL[task])


def candidates(task: str = TASK) -> dict[str, tuple[Pipeline, dict[str, list]]]:
    """Model pipelines and their inner-search grids."""
    num = NUMERIC[task]
    linear = ColumnTransformer([
        _onehot(task),
        ("num", Pipeline([("spline", SplineTransformer(n_knots=4, degree=3)), ("scale", StandardScaler())]), num),
    ])
    trees = ColumnTransformer([_onehot(task), ("num", "passthrough", num)])
    return {
        "majority": (Pipeline([("pre", trees), ("clf", DummyClassifier(strategy="most_frequent"))]), {}),
        "logreg": (
            Pipeline([("pre", linear), ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]),
            {"clf__C": [0.1, 1, 10]},
        ),
        "rf": (
            Pipeline([("pre", trees), ("clf", RandomForestClassifier(
                n_estimators=500, class_weight="balanced", random_state=0, n_jobs=1))]),
            {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [3, 5, 10]},
        ),
        "hgb": (
            Pipeline([("pre", trees), ("clf", HistGradientBoostingClassifier(
                class_weight="balanced", random_state=0))]),
            {"clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.03, 0.1]},
        ),
    }


def _search(pipe: Pipeline, grid: dict, X, y, groups, seed: int):
    """Inner grid search on training folds only; returns the fitted best estimator and params."""
    if not grid:
        return clone(pipe).fit(X, y), {}
    inner = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    search = GridSearchCV(clone(pipe), grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_


def fold_metrics(y_true, prob) -> dict[str, float]:
    """Spec Flow 4 metrics for one fold (positive class = 1, extinguished)."""
    pred = (prob >= THRESHOLD).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "average_precision": average_precision_score(y_true, prob),
        "average_precision_minority": average_precision_score(1 - np.asarray(y_true), 1 - prob),
        "brier": brier_score_loss(y_true, prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
    }


def cross_validate(df: pd.DataFrame, task: str = TASK) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Repeated outer StratifiedGroupKFold. Returns (fold metrics, out-of-fold predictions, best params seen)."""
    X, y, groups = df[FEATURES[task]], df[TARGET[task]].to_numpy(), df[GROUP[task]].to_numpy()
    rows, oof, params_seen = [], [], {}
    for seed in SEEDS:
        outer = StratifiedGroupKFold(n_splits=CV_SPLITS[task], shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
            for name, (pipe, grid) in candidates(task).items():
                est, params = _search(pipe, grid, X.iloc[tr], y[tr], groups[tr], seed)
                prob = est.predict_proba(X.iloc[te])[:, 1]
                rows.append({"model": name, "seed": seed, "fold": fold, **fold_metrics(y[te], prob),
                             "confusion": confusion_matrix(y[te], (prob >= THRESHOLD).astype(int),
                                                           labels=[0, 1]).tolist(), "params": params})
                params_seen.setdefault(name, []).append(params)
                oof.append(pd.DataFrame({"_record_id": df["_record_id"].iloc[te].to_numpy(), "model": name,
                                         "seed": seed, "fold": fold, "prob": prob, "label": y[te]}))
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True), params_seen


def calibrated_cv(df: pd.DataFrame, folds: pd.DataFrame, name: str, task: str = TASK
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-score the chosen model as served: same outer folds and tuned params, plus sigmoid calibration."""
    pipe, _ = candidates(task)[name]
    X, y, groups = df[FEATURES[task]], df[TARGET[task]].to_numpy(), df[GROUP[task]].to_numpy()
    params = {(r["seed"], r["fold"]): r["params"] for _, r in folds[folds["model"] == name].iterrows()}
    rows, oof = [], []
    label = f"{name}_calibrated"
    for seed in SEEDS:
        outer = StratifiedGroupKFold(n_splits=CV_SPLITS[task], shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
            tuned = clone(pipe).set_params(**params[(seed, fold)])
            inner = list(StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
                         .split(X.iloc[tr], y[tr], groups[tr]))
            cal = CalibratedClassifierCV(tuned, method="sigmoid", cv=inner).fit(X.iloc[tr], y[tr])
            prob = cal.predict_proba(X.iloc[te])[:, 1]
            rows.append({"model": label, "seed": seed, "fold": fold, **fold_metrics(y[te], prob),
                         "confusion": confusion_matrix(y[te], (prob >= THRESHOLD).astype(int),
                                                       labels=[0, 1]).tolist(), "params": params[(seed, fold)]})
            oof.append(pd.DataFrame({"_record_id": df["_record_id"].iloc[te].to_numpy(), "model": label,
                                     "seed": seed, "fold": fold, "prob": prob, "label": y[te]}))
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def reliability(oof: pd.DataFrame, model: str, n_bins: int = 5) -> dict[str, list[float]]:
    """Reliability curve (quantile bins) of out-of-fold probabilities."""
    g = oof[oof["model"] == model]
    frac_pos, mean_pred = calibration_curve(g["label"], g["prob"], n_bins=n_bins, strategy="quantile")
    return {"mean_predicted": mean_pred.round(4).tolist(), "fraction_positive": frac_pos.round(4).tolist()}


def summarize(folds: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Mean and SD of each metric per model, plus per-seed ROC-AUC and the summed confusion matrix."""
    metrics = ["roc_auc", "average_precision", "average_precision_minority", "brier", "balanced_accuracy"]
    out = {}
    for name, g in folds.groupby("model", sort=False):
        out[name] = {m: {"mean": float(g[m].mean()), "sd": float(g[m].std(ddof=1))} for m in metrics}
        out[name]["roc_auc_by_seed"] = g.groupby("seed")["roc_auc"].mean().round(4).tolist()
        out[name]["confusion_at_threshold"] = np.sum(np.array(g["confusion"].tolist()), axis=0).tolist()
    return out


def choose(summary: dict[str, dict[str, Any]]) -> str:
    """Simplest model whose mean ROC-AUC is within 1 SD of the best; must beat the majority class."""
    best = max(SIMPLICITY, key=lambda m: summary[m]["roc_auc"]["mean"])
    floor = summary[best]["roc_auc"]["mean"] - summary[best]["roc_auc"]["sd"]
    chosen = next(m for m in SIMPLICITY if summary[m]["roc_auc"]["mean"] >= floor)
    baseline = summary["majority"]["roc_auc_by_seed"]
    if not all(a > b for a, b in zip(summary[chosen]["roc_auc_by_seed"], baseline)):
        raise RuntimeError(f"{chosen} does not beat the majority class on ROC-AUC in every repeat")
    return chosen


def leave_one_fuel_out(df: pd.DataFrame, name: str, task: str = TASK) -> dict[str, dict[str, Any]]:
    """Train on one fuel, test on the other. Reported whatever the score: it shows where the model stops."""
    pipe, grid = candidates(task)[name]
    X, y, groups = df[FEATURES[task]], df[TARGET[task]].to_numpy(), df[GROUP[task]].to_numpy()
    out = {}
    for fuel in sorted(df["fuel"].unique()):
        te = (df["fuel"] == fuel).to_numpy()
        est, _ = _search(pipe, grid, X[~te], y[~te], groups[~te], seed=0)
        prob = est.predict_proba(X[te])[:, 1]
        result = {"n_test": int(te.sum()), "test_class_balance": np.bincount(y[te], minlength=2).tolist(),
                  "brier": float(brier_score_loss(y[te], prob)),
                  "balanced_accuracy": float(balanced_accuracy_score(y[te], (prob >= THRESHOLD).astype(int)))}
        result["roc_auc"] = float(roc_auc_score(y[te], prob)) if len(set(y[te])) == 2 else None
        out[f"test_on_{fuel}"] = result
    return out


@dataclass
class ModelBundle:
    """Everything needed to serve one task's predictions."""

    task: str
    version: str
    model_name: str
    features: list[str]
    calibrated: Any
    conformal: Any
    classes: dict[int, str]
    threshold: float = THRESHOLD
    confidence: float = CONFIDENCE
    params: dict[str, Any] = field(default_factory=dict)

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of class 1."""
        return self.calibrated.predict_proba(X[self.features])[:, 1]

    def prediction_sets(self, X: pd.DataFrame) -> list[list[str]]:
        """90% conformal prediction set per row, as class names."""
        _, sets = self.conformal.predict_set(X[self.features])
        return [[self.classes[c] for c in (0, 1) if row[c, 0]] for row in sets]


def fit_final(df: pd.DataFrame, name: str, task: str = TASK) -> tuple[ModelBundle, dict]:
    """Refit the chosen model on all rows, calibrate (sigmoid) and conformalize (90%)."""
    pipe, grid = candidates(task)[name]
    X, y, groups = df[FEATURES[task]], df[TARGET[task]].to_numpy(), df[GROUP[task]].to_numpy()
    _, params = _search(pipe, grid, X, y, groups, seed=0)
    tuned = clone(pipe).set_params(**params)
    splits = list(StratifiedGroupKFold(n_splits=CV_SPLITS[task], shuffle=True, random_state=0).split(X, y, groups))
    calibrated = CalibratedClassifierCV(tuned, method="sigmoid", cv=splits).fit(X, y)
    conformal = CrossConformalClassifier(
        estimator=CalibratedClassifierCV(tuned, method="sigmoid", cv=3),
        confidence_level=CONFIDENCE, conformity_score="lac",
        cv=StratifiedGroupKFold(n_splits=CV_SPLITS[task], shuffle=True, random_state=1), random_state=0,
    ).fit_conformalize(X, y, groups=groups)
    bundle = ModelBundle(task=task, version=f"{task}@{date.today().isoformat()}", model_name=name,
                         features=FEATURES[task], calibrated=calibrated, conformal=conformal,
                         classes=CLASS_NAMES[task], params=params)
    return bundle, params


def input_ranges(df: pd.DataFrame, task: str = TASK) -> dict[str, dict[str, dict[str, float]]]:
    """Per-fuel min and max of every numeric input."""
    return {
        fuel: {c: {"min": float(g[c].min()), "max": float(g[c].max())} for c in NUMERIC[task]}
        for fuel, g in df.groupby("fuel")
    }


def git_commit() -> dict[str, Any]:
    """Commit hash of the repo and whether the project folder has uncommitted changes."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain", "--", "."))}
    except OSError:
        return {"commit": None, "dirty": None}


def run(task: str = TASK) -> dict[str, Any]:
    """Cross-validate, choose, refit and save model.joblib, cv_report.json and model_card.json."""
    warnings.filterwarnings("ignore", category=UserWarning)
    df = tasks.load(task)
    task_report = paths.read_json(paths.TASKS / "tasks_report.json")[task]
    folds, oof, params_seen = cross_validate(df, task)
    chosen = choose(summarize(folds))
    cal_folds, cal_oof = calibrated_cv(df, folds, chosen, task)
    folds = pd.concat([folds, cal_folds], ignore_index=True)
    oof = pd.concat([oof, cal_oof], ignore_index=True)
    summary = summarize(folds)
    served = f"{chosen}_calibrated"
    lofo = leave_one_fuel_out(df, chosen, task)
    bundle, params = fit_final(df, chosen, task)

    out = model_dir(task)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "model.joblib")
    oof.to_parquet(out / "oof.parquet", index=False)
    cv_report = {"task": task, "seeds": list(SEEDS), "n_splits": CV_SPLITS[task], "inner_splits": INNER_SPLITS,
                 "group": GROUP[task], "threshold": THRESHOLD, "chosen": chosen, "served": served,
                 "summary": summary, "reliability": {m: reliability(oof, m) for m in (chosen, served)},
                 "folds": folds.drop(columns="confusion").to_dict(orient="records"),
                 "inner_best_params": params_seen, "leave_one_fuel_out": lofo}
    paths.write_json(out / "cv_report.json", cv_report)
    card = {
        "version": bundle.version,
        "task": task,
        "date": date.today().isoformat(),
        "git": git_commit(),
        "target": {"source": "test_end", "positive": CLASS_NAMES[task][1], "negative": CLASS_NAMES[task][0],
                   "mapping": task_report["value_counts"], "reviewed_by": task_report["reviewed_by"]},
        "n_rows": int(len(df)),
        "n_excluded": {"by_label": task_report["excluded_by_label"],
                       "missing_inputs": len(task_report["excluded_missing_inputs"])},
        "class_balance": task_report["class_balance"],
        "features": FEATURES[task],
        "input_ranges_by_fuel": input_ranges(df, task),
        "cv_metrics": {m: summary[m] for m in summary},
        "chosen_model": chosen,
        "served_model_cv": served,
        "chosen_params": params,
        "calibration": "CalibratedClassifierCV(method='sigmoid') on grouped folds",
        "conformal": {"method": "MAPIE CrossConformalClassifier, LAC score", "confidence_level": CONFIDENCE},
        "display_bands": {"Low": "< 0.30", "Medium": "0.30-0.70", "High": "> 0.70",
                          "note": "Display bands, not physical thresholds."},
        "leave_one_fuel_out": lofo,
    }
    paths.write_json(out / "model_card.json", card)
    _print_table(summary, chosen, lofo)
    return card


def _print_table(summary, chosen, lofo) -> None:
    print(f"\n{'model':10s} {'ROC-AUC':>15s} {'PR-AUC':>15s} {'PR-AUC(min)':>15s} {'Brier':>15s} {'Bal.acc':>15s}")
    for name, s in summary.items():
        cells = [f"{s[m]['mean']:.3f} ± {s[m]['sd']:.3f}" for m in
                 ["roc_auc", "average_precision", "average_precision_minority", "brier", "balanced_accuracy"]]
        mark = "  <- chosen" if name == chosen else ""
        print(f"{name:10s} " + " ".join(f"{c:>15s}" for c in cells) + mark)
    print("\nLeave-one-fuel-out (chosen model):")
    for k, v in lofo.items():
        auc = "n/a" if v["roc_auc"] is None else f"{v['roc_auc']:.3f}"
        print(f"  {k:22s} n={v['n_test']:3d} roc_auc={auc} bal_acc={v['balanced_accuracy']:.3f} brier={v['brier']:.3f}")


def load_bundle(task: str = TASK) -> ModelBundle:
    """Load the saved model bundle."""
    return joblib.load(model_dir(task) / "model.joblib")


if __name__ == "__main__":
    # Call through the package so ModelBundle pickles as flamelens.train.ModelBundle, not __main__.
    from flamelens import train as _train

    _train.run()
