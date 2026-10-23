"""Flow 3/14 leakage checks: no post-test field can reach a model."""

from __future__ import annotations

import pytest

from flamelens import features, normalize, tasks


def _roles(task):
    """canonical name -> role, over every source dataset of the task."""
    cmap = normalize.column_map()
    return {
        spec["name"]: spec["role"]
        for dataset in features.SOURCES[task]
        for spec in cmap[dataset].values()
    }


@pytest.mark.parametrize("task", sorted(features.FEATURES))
def test_features_disjoint_from_outputs(task):
    assert not set(features.FEATURES[task]) & set(features.OUTPUTS[task])


@pytest.mark.parametrize("task", sorted(features.FEATURES))
def test_outputs_list_covers_every_output_column(task):
    outputs = {name for name, role in _roles(task).items() if role == "OUTPUT"}
    assert outputs <= set(features.OUTPUTS[task])


@pytest.mark.parametrize("task", sorted(features.FEATURES))
def test_features_are_inputs_or_derived_from_inputs(task):
    roles = _roles(task)
    derived = features.DERIVED[task]
    for col in features.FEATURES[task]:
        if col in derived:
            assert all(roles.get(src) == "INPUT" for src in derived[col]), col
        else:
            assert roles.get(col) == "INPUT", f"{task}: {col} has role {roles.get(col)}"


def test_x_n2_not_a_feature():
    assert "x_n2" not in features.FEATURES["ext_flex"]


@pytest.mark.parametrize("task", sorted(features.FEATURES))
def test_task_tables(task_reports, task):
    df = tasks.load(task)
    target = features.TARGET[task]
    assert df[target].notna().all()
    assert set(df[target].unique()) <= {0, 1}
    assert task_reports[task]["min_groups_per_fold"] >= 5
    required = [c for c in features.FEATURES[task] if c not in features.NULLABLE[task]]
    assert df[required].notna().all().all()
    print(f"{task} class balance: {task_reports[task]['class_balance']}")


def test_todo_target_stops_the_build(normalized, monkeypatch):
    monkeypatch.setattr(tasks, "target_map", lambda task: {"labels": {"Extinction": "TODO"}, "censored": []})
    with pytest.raises(tasks.TargetNotDefined):
        tasks.build_ext_flex()
