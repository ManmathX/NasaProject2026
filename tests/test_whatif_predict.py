"""Flow 7 checks (sweeps carry coverage, maps cache) and the gated prediction response."""

from __future__ import annotations

import pytest

from flamelens import predict, tasks, whatif

pytestmark = pytest.mark.slow

BASE = {"fuel": "methanol", "x_o2": 0.18, "x_co2": 0.0, "x_he": 0.0, "p_atm": 1.0, "d0_mm": 2.5}


def test_sweep_points_carry_coverage(trained):
    s = whatif.sweep(BASE, "x_o2", steps=30)
    assert len(s["points"]) == 30
    assert all(p["coverage"] in {"High", "Medium", "Low"} for p in s["points"])
    assert all(p["probability"] is None for p in s["points"] if p["coverage"] == "Low")
    assert "physics_check" in s


def test_sweep_rejects_unknown_variable(trained):
    with pytest.raises(tasks.InvalidQuery):
        whatif.sweep(BASE, "burn_s")


def test_physics_check_flags_rising_curve():
    rising = {"points": [{"value": v / 10, "probability": v / 10} for v in range(10)]}
    falling = {"points": [{"value": v / 10, "probability": 1 - v / 10} for v in range(10)]}
    assert whatif.physics_check(rising)["flagged"]
    assert not whatif.physics_check(falling)["flagged"]


def test_grid_map_shape(trained):
    m = whatif.grid_map("methanol", "N2", 2.5, n=8)
    assert len(m["probability"]) == 8 and len(m["probability"][0]) == 8
    flat = [(p, c) for pr, cr in zip(m["probability"], m["coverage"]) for p, c in zip(pr, cr)]
    assert all(p is None for p, c in flat if c == "Low")


def test_maps_load_from_cache(trained):
    if not whatif.map_path("methanol", "N2", 2.0).exists():
        whatif.precompute_maps()
    m = whatif.load_map("methanol", "N2", 2.0)
    assert m["fuel"] == "methanol" and len(m["x"]) == 50


def test_low_coverage_request_is_unavailable(trained):
    r = predict.predict({**BASE, "x_o2": 0.05})
    assert r["status"] == "unavailable"
    assert "probability" not in r
    assert "x_o2" in r["coverage"]["out_of_range"]
    assert r["neighbors"]


def test_covered_request_has_probability(trained):
    r = predict.predict({**BASE, "x_o2": 0.21, "d0_mm": 2.0})
    assert r["status"] in {"supported", "uncertain", "limited"}
    assert 0 <= r["probability"] <= 1
    assert r["band"] in {"Low", "Medium", "High"}
    assert len(r["neighbors"]) == 5
    assert r["model_card"].startswith("ext_flex@")


def test_invalid_requests():
    with pytest.raises(tasks.InvalidQuery):
        tasks.query_frame("ext_flex", {**BASE, "fuel": "kerosene"})
    with pytest.raises(tasks.InvalidQuery):
        tasks.query_frame("ext_flex", {k: v for k, v in BASE.items() if k != "p_atm"})
