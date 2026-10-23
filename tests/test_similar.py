"""Flow 6 checks: an exact training row is its own nearest neighbour; Gower handles nulls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flamelens import similar, tasks


@pytest.mark.slow
def test_exact_training_row_first(trained):
    df = tasks.load("ext_flex")
    for _, row in df.sample(10, random_state=1).iterrows():
        results = similar.similar_extinction(row)
        assert results[0]["distance"] == pytest.approx(0.0)
        at_zero = [r["record_id"] for r in results if r["distance"] == pytest.approx(0.0)]
        assert row["_record_id"] in at_zero
        assert results[0]["similarity"] == pytest.approx(1.0)
        # Conditions come back in original NASA columns and text.
        assert "Ambient pressure; mmHg" in results[0]["conditions"]


@pytest.mark.slow
def test_record_lookup(trained):
    rec = similar.record("PSI-69_FLEX:1")
    assert rec["source_row"] == 1
    assert rec["values"]["Test end"] == "Disruption"


def test_neighbor_summary_and_agreement():
    res = [{"label": 1}] * 4 + [{"label": 0}]
    assert similar.neighbor_summary(res) == "4 of 5 extinguished"
    assert similar.agreement(0.8, res)
    assert not similar.agreement(0.2, res)
    assert not similar.agreement(0.9, [{"label": 0}] * 3 + [{"label": 1}] * 2)
    assert similar.agreement(0.5, [{"label": 0}] * 5)


def test_gower():
    table = pd.DataFrame({"x": [0.0, 5.0, 10.0, np.nan], "m": ["a", "b", "a", "a"]})
    d = similar.gower_distances(table, {"x": 0.0, "m": "a"}, ["x"], ["m"])
    assert d[0] == 0.0
    assert d[1] == pytest.approx((0.5 + 1) / 2)
    assert d[2] == pytest.approx((1.0 + 0) / 2)
    assert d[3] == 0.0  # x is null for that row: only the category counts
    d2 = similar.gower_distances(table, {"x": None, "m": "b"}, ["x"], ["m"])
    assert list(d2) == [1.0, 0.0, 1.0, 1.0]
