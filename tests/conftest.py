"""Shared fixtures: build each pipeline stage once per test session."""

from __future__ import annotations

import pytest

from flamelens import paths


@pytest.fixture(scope="session")
def ingested():
    from flamelens import ingest

    return ingest.run()


@pytest.fixture(scope="session")
def normalized(ingested):
    from flamelens import normalize

    return normalize.run()


@pytest.fixture(scope="session")
def task_reports(normalized):
    from flamelens import tasks

    return tasks.run(verbose=False)


@pytest.fixture(scope="session")
def trained(task_reports):
    """Train ext_flex and fit coverage if their artifacts are missing (slow: ~3 min)."""
    from flamelens import coverage, train

    d = paths.MODELS / "ext_flex"
    if not (d / "model.joblib").exists() or not (d / "model_card.json").exists():
        train.run()
    if not (d / "coverage.joblib").exists():
        coverage.run()
    return paths.read_json(d / "model_card.json")


@pytest.fixture(scope="session")
def registered_ids():
    return {e["dataset_id"] for e in paths.registry()}
