"""Flow 1 checks: row counts, untouched headers, unique record ids."""

from __future__ import annotations

import pytest

from flamelens import ingest, paths

# Expected rows after blank rows are dropped (spec Flow 1). (low, high) for files with a range.
EXPECTED_ROWS = {
    "PSI-69_FLEX": (274, 274),
    "PSI-159_CFI-G_Normal": (179, 179),
    "PSI-159_CFI-G_Inverse": (140, 170),
    "PSI-25_BASS-II": (129, 129),
    "PSI-98_SAFFIRE-1": (4, 4),
    "PSI-99_SAFFIRE-2": (15, 15),
    "PSI-100_SAFFIRE-3": (2, 2),
    "PSI-107_SPICE": (469, 469),
    "PSI-102_SAME": (8, 8),
}


@pytest.mark.parametrize("dataset_id", sorted(EXPECTED_ROWS))
def test_row_counts(ingested, registered_ids, dataset_id):
    if dataset_id not in registered_ids:
        pytest.skip(f"{dataset_id} CSV not in data/raw yet")
    lo, hi = EXPECTED_ROWS[dataset_id]
    assert lo <= ingested[dataset_id]["rows_out"] <= hi


def test_headers_unchanged(ingested):
    for entry in paths.registry():
        header = ingest.raw_header(paths.RAW / entry["file"], entry["encoding"])
        df = ingest.load(entry["dataset_id"])
        assert list(df.columns[: len(header)]) == header
        assert list(df.columns[len(header):]) == ingest.PROVENANCE


def test_record_ids_unique_and_traceable(ingested):
    for entry in paths.registry():
        df = ingest.load(entry["dataset_id"])
        assert df["_record_id"].is_unique
        assert (df["_record_id"] == entry["dataset_id"] + ":" + df["_source_row"].astype(str)).all()


def test_values_kept_as_text(ingested):
    df = ingest.load("PSI-69_FLEX")
    # The FLEX file marks some missing diameters with an en dash; ingestion must not touch it.
    assert (df["Droplet initial diameter; mm"] == "–").sum() == 6
    assert df.loc[df["_source_row"] == 1, "Ambient pressure; mmHg"].item() == "757.7"
