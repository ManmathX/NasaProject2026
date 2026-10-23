"""Flow 4 checks: model card written, beats the majority class, bundle serves sets."""

from __future__ import annotations

import numpy as np
import pytest

from flamelens import paths, tasks, train

pytestmark = pytest.mark.slow


def test_model_card(trained):
    for key in ["n_rows", "n_excluded", "features", "input_ranges_by_fuel", "cv_metrics", "chosen_model",
                "date", "git", "version"]:
        assert key in trained
    assert trained["features"] == train.FEATURES["ext_flex"]
    assert (paths.MODELS / "ext_flex" / "cv_report.json").exists()


def test_beats_majority_every_repeat(trained):
    metrics = trained["cv_metrics"]
    chosen = metrics[trained["chosen_model"]]["roc_auc_by_seed"]
    majority = metrics["majority"]["roc_auc_by_seed"]
    assert all(a > b for a, b in zip(chosen, majority))


def test_leave_one_fuel_out_reported(trained):
    assert set(trained["leave_one_fuel_out"]) == {"test_on_methanol", "test_on_n-heptane"}


def test_bundle_outputs(trained):
    bundle = train.load_bundle()
    df = tasks.load("ext_flex").head(20)
    prob = bundle.proba(df)
    assert np.all((prob >= 0) & (prob <= 1))
    sets = bundle.prediction_sets(df)
    assert len(sets) == 20
    assert all(set(s) <= {"extinguished", "not_extinguished"} for s in sets)
