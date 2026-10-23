"""Flow 7: what-if sweeps and O2 x pressure extinction maps, with coverage at every point.

Run with ``python -m flamelens.whatif`` to precompute the cached maps.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from flamelens import paths, predict, tasks
from flamelens.features import COVERAGE_NUMERIC, TARGET

TASK = "ext_flex"
SWEEPABLE = COVERAGE_NUMERIC[TASK]
# How close a real test's other conditions must be to count as "near the fixed conditions".
TOLERANCE = {"x_o2": 0.015, "x_co2": 0.03, "x_he": 0.03, "p_atm": 0.1, "d0_mm": 0.5}
MAP_D0 = (2.0, 3.0, 4.0)
MAPS_DIR = paths.MODELS / TASK / "maps"


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["fuel"] = df["fuel"].astype("string")
    df["diluent"] = df["diluent"].astype("string")
    return tasks.add_flex_derived(df)


def _base(conditions: dict[str, Any]) -> dict[str, Any]:
    q = tasks.query_frame(TASK, conditions).iloc[0]
    return {"fuel": q["fuel"], "diluent": q["diluent"], **{c: float(q[c]) for c in SWEEPABLE}}


def tested_range(fuel: str, variable: str) -> tuple[float, float]:
    """That fuel's tested min and max of one input, from the model card."""
    card = paths.read_json(paths.MODELS / TASK / "model_card.json")
    r = card["input_ranges_by_fuel"][fuel][variable]
    return r["min"], r["max"]


def nearby_tests(base: dict[str, Any], free: list[str]) -> pd.DataFrame:
    """Real tests with the same fuel and diluent whose fixed conditions are within TOLERANCE of base."""
    df = tasks.load(TASK)
    mask = (df["fuel"] == base["fuel"]) & (df["diluent"] == base["diluent"])
    for c in SWEEPABLE:
        if c not in free:
            mask &= (df[c] - base[c]).abs() <= TOLERANCE[c]
    return df.loc[mask]


def sweep(conditions: dict[str, Any], variable: str, lo: float | None = None, hi: float | None = None,
          steps: int = 30) -> dict[str, Any]:
    """Vary one input across a range with the others fixed. Fuel and diluent never change."""
    if variable not in SWEEPABLE:
        raise tasks.InvalidQuery(f"cannot sweep {variable!r}; choose one of {SWEEPABLE}")
    base = _base(conditions)
    t_lo, t_hi = tested_range(base["fuel"], variable)
    lo = t_lo if lo is None else float(lo)
    hi = t_hi if hi is None else float(hi)
    if not lo < hi:
        raise tasks.InvalidQuery("the sweep range needs lo < hi")
    xs = np.linspace(lo, hi, steps)
    rows = [{**base, variable: float(x)} for x in xs]
    df = _frame(rows)
    valid = (df["x_o2"] + df["x_co2"] + df["x_he"] <= 1.0 + 1e-9) & (df["x_o2"] > 0)
    scored = predict.score_frame(df[valid])
    points = []
    for i, x in enumerate(xs):
        if not valid.iloc[i]:
            points.append({"value": float(x), "probability": None, "prediction_set": None, "coverage": "Low",
                           "note": "gas fractions exceed 1"})
            continue
        s = scored.loc[df.index[i]]
        points.append({"value": float(x),
                       "probability": None if s["probability"] is None else float(s["probability"]),
                       "prediction_set": s["prediction_set"], "coverage": s["level"]})
    near = nearby_tests(base, [variable])
    result = {
        "variable": variable, "fixed": base, "range": [lo, hi], "tested_range": [t_lo, t_hi], "points": points,
        "nasa_tests": [{"record_id": r["_record_id"], "value": float(r[variable]), "label": int(r[TARGET[TASK]])}
                       for _, r in near.iterrows()],
        "model_card": predict.bundle().version,
    }
    if variable == "x_o2":
        result["physics_check"] = physics_check(result)
    return result


def physics_check(sweep_result: dict[str, Any], eps: float = 0.005) -> dict[str, Any]:
    """Flag an O2 sweep where extinction probability rises with O2 over > 30% of the covered range."""
    pts = [p for p in sweep_result["points"] if p["probability"] is not None]
    rises, total = 0.0, 0.0
    for a, b in zip(pts, pts[1:]):
        width = b["value"] - a["value"]
        total += width
        if b["probability"] - a["probability"] > eps:
            rises += width
    share = rises / total if total else 0.0
    return {"flagged": share > 0.3, "share_rising": round(share, 3), "covered_points": len(pts),
            "rule": "more O2 should not raise extinction probability over most of the range"}


def grid_map(fuel: str, diluent: str, d0_mm: float, x_var: str = "x_o2", y_var: str = "p_atm", n: int = 50,
             fixed: dict[str, float] | None = None) -> dict[str, Any]:
    """Probability grid over x_var x y_var for one fuel, diluent and droplet size.

    Other inputs are held at the fuel + diluent group median unless given in `fixed`.
    Cells with Low coverage carry no probability (shown grey).
    """
    df = tasks.load(TASK)
    g = df[(df["fuel"] == fuel) & (df["diluent"] == diluent)]
    if g.empty:
        raise tasks.InvalidQuery(f"no NASA tests for {fuel} with {diluent}")
    held = {c: float(g[c].median()) for c in SWEEPABLE}
    held.update({"d0_mm": float(d0_mm), **(fixed or {})})
    xs = np.linspace(g[x_var].min(), g[x_var].max(), n)
    ys = np.linspace(g[y_var].min(), g[y_var].max(), n)
    rows = [{"fuel": fuel, "diluent": diluent, **held, x_var: float(x), y_var: float(y)} for y, x in product(ys, xs)]
    grid = _frame(rows)
    valid = (grid["x_o2"] + grid["x_co2"] + grid["x_he"] <= 1.0 + 1e-9)
    scored = predict.score_frame(grid[valid]).reindex(grid.index)
    prob = scored["probability"].where(valid, None).to_numpy().reshape(n, n)
    level = scored["level"].fillna("Low").to_numpy().reshape(n, n)
    pts = g[(g["d0_mm"] - d0_mm).abs() <= 0.5]
    return {
        "fuel": fuel, "diluent": diluent, "d0_mm": d0_mm, "x_var": x_var, "y_var": y_var,
        "held": held, "x": xs.tolist(), "y": ys.tolist(),
        "probability": [[None if p is None or pd.isna(p) else round(float(p), 4) for p in r] for r in prob],
        "coverage": level.tolist(),
        "nasa_tests": [{"record_id": r["_record_id"], "x": float(r[x_var]), "y": float(r[y_var]),
                        "d0_mm": float(r["d0_mm"]), "label": int(r[TARGET[TASK]])} for _, r in pts.iterrows()],
        "model_card": predict.bundle().version,
    }


def map_path(fuel: str, diluent: str, d0_mm: float):
    return MAPS_DIR / f"{fuel}_{diluent}_d{d0_mm:g}.json"


def precompute_maps() -> list[dict[str, Any]]:
    """Cache a map for every fuel x diluent at d0 in MAP_D0, plus maps/index.json."""
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    df = tasks.load(TASK)
    index = []
    for (fuel, diluent), _ in df.groupby(["fuel", "diluent"]):
        for d0 in MAP_D0:
            m = grid_map(fuel, diluent, d0)
            paths.write_json(map_path(fuel, diluent, d0), m)
            levels = pd.Series(np.ravel(m["coverage"])).value_counts().to_dict()
            index.append({"fuel": fuel, "diluent": diluent, "d0_mm": d0, "file": map_path(fuel, diluent, d0).name,
                          "coverage_cells": levels, "nasa_tests": len(m["nasa_tests"])})
    paths.write_json(MAPS_DIR / "index.json", index)
    return index


def load_map(fuel: str, diluent: str, d0_mm: float) -> dict[str, Any]:
    """Read a cached map; raises FileNotFoundError when it was not precomputed."""
    return paths.read_json(map_path(fuel, diluent, d0_mm))


def physics_report() -> list[dict[str, Any]]:
    """O2 sweeps at each training row's conditions (one per fuel+diluent+pressure band) and their flags."""
    df = tasks.load(TASK)
    out = []
    for (fuel, diluent), g in df.groupby(["fuel", "diluent"]):
        base = g.iloc[(g["p_atm"] - g["p_atm"].median()).abs().argsort().iloc[0]]
        conds = {"fuel": fuel, "diluent": diluent, **{c: float(base[c]) for c in SWEEPABLE}}
        s = sweep(conds, "x_o2")
        out.append({"fuel": fuel, "diluent": diluent, "base_record": base["_record_id"], **s["physics_check"]})
    paths.write_json(paths.MODELS / TASK / "physics_check.json", out)
    return out


if __name__ == "__main__":
    for entry in precompute_maps():
        print(f"{entry['file']:32s} cells={entry['coverage_cells']} tests={entry['nasa_tests']}")
    print("\nO2 sweep physics check:")
    for r in physics_report():
        print(f"  {r['fuel']:10s} {r['diluent']:4s} flagged={r['flagged']} share_rising={r['share_rising']} "
              f"covered_points={r['covered_points']}")
