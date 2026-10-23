"""Flow 2 checks: unit conversions, column-map coverage, flagged values."""

from __future__ import annotations

import pandas as pd
import pytest

from flamelens import ingest, normalize, paths, units


def test_unit_conversions():
    assert units.convert(760.0, "mmHg", "atm") == pytest.approx(1.0)
    assert units.convert(1.01325, "bar", "atm") == pytest.approx(1.0, abs=1e-3)
    assert units.convert(21.0, "%", "fraction") == pytest.approx(0.21)
    assert units.convert(25.0, "C", "K") == pytest.approx(298.15)
    assert units.parse_value("0.27 mm") == (0.27, "mm")
    assert units.parse_value("0.27") == (0.27, None)
    assert units.parse_value("") == (None, None)
    assert units.parse_value("–") == (None, None)
    assert units.parse_value("#DIV/0!") == (None, None)
    assert units.parse_value("~9") == (9.0, None) and units.is_approximate("~9")
    with pytest.raises(ValueError):
        units.parse_value("hello")


def test_every_raw_column_is_mapped(ingested):
    cmap = normalize.column_map()
    for entry in paths.registry():
        df = ingest.load(entry["dataset_id"])
        raw = [c for c in df.columns if c not in ingest.PROVENANCE]
        normalize.check_map(entry["dataset_id"], raw, cmap[entry["dataset_id"]])


def test_unmapped_column_fails():
    with pytest.raises(normalize.SchemaError):
        normalize.check_map("X", ["a", "b"], {"a": {"name": "a", "role": "INPUT", "type": "float"}})


def test_flex_zero_pressure_flagged_and_row_kept(normalized):
    df = normalize.load("PSI-69_FLEX")
    assert len(df) == 274
    row = df[df["_record_id"] == "PSI-69_FLEX:114"]
    assert len(row) == 1
    assert pd.isna(row["p_atm"].item())
    assert bool(row["flag_p_atm"].item())


def test_canonical_units(normalized):
    flex = normalize.load("PSI-69_FLEX")
    assert flex["p_atm"].max() < 3.1          # 2319.9 mmHg -> 3.05 atm
    assert flex["x_o2"].between(0, 1).all()
    assert set(flex["fuel"].dropna()) == {"methanol", "n-heptane"}
    assert flex["test_date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    inv = normalize.load("PSI-159_CFI-G_Inverse")
    assert set(inv["flame_type"]) == {"inverse"}
    assert inv["peak_burner_t_k"].dropna().between(200, 1500).all()


def test_missing_values_stay_null(normalized):
    flex = normalize.load("PSI-69_FLEX")
    # 12 droplet diameters are blank or an en dash in the raw file; none may become 0.
    assert flex["d0_mm"].isna().sum() == 12
    assert (flex["d0_mm"] == 0).sum() == 0


def test_long_table(normalized):
    long = pd.read_parquet(paths.NORMALIZED / "long.parquet")
    assert {"record_id", "dataset", "variable", "value", "unit", "role"} <= set(long.columns)
    assert "IGNORE" not in set(long["role"])
