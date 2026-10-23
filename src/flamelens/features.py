"""The only place feature, output, target and group lists live (spec Flow 3).

Every model, coverage check and leakage test reads its columns from here.
"""

from __future__ import annotations

# Source datasets behind each task table.
SOURCES: dict[str, list[str]] = {
    "ext_flex": ["PSI-69_FLEX"],
    "ext_cfig": ["PSI-159_CFI-G_Normal", "PSI-159_CFI-G_Inverse"],
}

# Inputs computed only from other inputs: derived name -> the columns it reads.
DERIVED: dict[str, dict[str, list[str]]] = {
    "ext_flex": {
        "p_o2_atm": ["x_o2", "p_atm"],
        "inert_to_o2": ["x_o2"],
        "co2_to_o2": ["x_co2", "x_o2"],
        "he_to_o2": ["x_he", "x_o2"],
        "d0_sq": ["d0_mm"],
        "diluent": ["x_he", "x_co2"],
    },
    "ext_cfig": {},
}

CATEGORICAL: dict[str, list[str]] = {
    "ext_flex": ["fuel", "diluent"],
    "ext_cfig": ["fuel", "flame_type"],
}

# Measured inputs, before derivation. x_n2 is left out: the four gas fractions sum to 1.
BASE_NUMERIC: dict[str, list[str]] = {
    "ext_flex": ["x_o2", "x_co2", "x_he", "p_atm", "d0_mm"],
    "ext_cfig": ["x_o2", "x_f", "p_atm", "fuel_flow_mg_s", "o2_flow_mg_s", "tad_k", "zst"],
}

NUMERIC: dict[str, list[str]] = {
    "ext_flex": BASE_NUMERIC["ext_flex"] + ["p_o2_atm", "inert_to_o2", "co2_to_o2", "he_to_o2", "d0_sq"],
    "ext_cfig": BASE_NUMERIC["ext_cfig"],
}

FEATURES: dict[str, list[str]] = {task: CATEGORICAL[task] + NUMERIC[task] for task in NUMERIC}

# Inputs allowed to be null in a training row (structurally absent, not missing).
NULLABLE: dict[str, list[str]] = {
    "ext_flex": [],
    "ext_cfig": ["o2_flow_mg_s"],  # normal flames draw oxygen from the ambient, not a metered flow
}

# Coverage space (Flow 5): numeric inputs scaled and compared within each filter group.
COVERAGE_NUMERIC: dict[str, list[str]] = {"ext_flex": BASE_NUMERIC["ext_flex"]}
COVERAGE_FILTERS: dict[str, list[str]] = {"ext_flex": ["fuel", "diluent"]}
# Floor on the median/IQR scale: one scaled unit is never smaller than a meaningful change.
# Without it, N2 tests clustered at 1.00 +/- 0.01 atm put the 0.7 atm tests 26 "IQRs" away.
COVERAGE_MIN_SCALE: dict[str, dict[str, float]] = {
    "ext_flex": {"x_o2": 0.02, "x_co2": 0.05, "x_he": 0.05, "p_atm": 0.1, "d0_mm": 0.25},
}

# Post-test fields. None of these may ever reach a model.
OUTPUTS: dict[str, list[str]] = {
    "ext_flex": ["dext_mm", "k_mm2_s", "burn_s", "test_end"],
    "ext_cfig": ["x_o2_after", "x_f_after", "hot_ignition", "cool_flame", "burn_s", "ext_outcome",
                 "peak_burner_t_k", "peak_burner_t_k_reported"],
}

TARGET: dict[str, str] = {"ext_flex": "label", "ext_cfig": "label"}
TARGET_SOURCE: dict[str, str] = {"ext_flex": "test_end", "ext_cfig": "ext_outcome"}
GROUP: dict[str, str] = {"ext_flex": "test_date", "ext_cfig": "group"}
# CFI-G has only 23 day-groups, so 4 folds keep at least 5 groups in every fold.
CV_SPLITS: dict[str, int] = {"ext_flex": 5, "ext_cfig": 4}

CLASS_NAMES: dict[str, dict[int, str]] = {
    "ext_flex": {0: "not_extinguished", 1: "extinguished"},
    "ext_cfig": {0: "flow_terminated", 1: "self_extinguished"},
}
