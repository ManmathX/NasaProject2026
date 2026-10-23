"""Flow 5 checks: training rows score High, out-of-range requests score Low and name the variable."""

from __future__ import annotations

import numpy as np
import pytest

from flamelens import coverage, paths, tasks

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cov(trained):
    return coverage.load()


@pytest.fixture(scope="module")
def flex(task_reports):
    return tasks.load("ext_flex")


def test_training_row_scores_high(cov, flex):
    group = cov.groups[("methanol", "N2")]
    typical = group.record_ids[np.argsort(group.loo)[len(group.loo) // 2]]
    row = flex.set_index("_record_id").loc[typical]
    assert cov.assess(row)["level"] == "High"
    levels = cov.assess_frame(flex)["level"]
    assert (levels == "High").mean() > 0.6


def test_o2_below_tested_minimum_is_low(cov, flex):
    row = flex.iloc[0].copy()
    group = cov.group_for(row)
    row["x_o2"] = group.lo[cov.numeric.index("x_o2")] - 0.02
    result = cov.assess(row)
    assert result["level"] == "Low"
    assert "x_o2" in result["out_of_range"]
    assert "below" in result["reason"]


def test_unknown_group_is_low(cov, flex):
    row = flex.iloc[0].copy()
    row["diluent"] = "Ar"
    assert cov.assess(row)["level"] == "Low"


def test_assess_frame_matches_assess(cov, flex):
    sample = flex.sample(15, random_state=0)
    batch = cov.assess_frame(sample)
    for idx, row in sample.iterrows():
        single = cov.assess(row)
        assert batch.loc[idx, "level"] == single["level"]
        assert batch.loc[idx, "evidence_count"] == single["evidence_count"]
        assert batch.loc[idx, "score"] == pytest.approx(single["score"])


def test_status_table():
    assert coverage.status({"level": "High"}, ["extinguished"]) == "supported"
    assert coverage.status({"level": "High"}, ["extinguished", "not_extinguished"]) == "uncertain"
    assert coverage.status({"level": "Medium"}, ["extinguished"]) == "limited"
    assert coverage.status({"level": "Low"}, []) == "unavailable"


def test_validation_report(trained):
    report = paths.read_json(paths.MODELS / "ext_flex" / "coverage_validation.json")
    assert set(report["levels"]) == {"High", "Medium", "Low"}
    print(report)
