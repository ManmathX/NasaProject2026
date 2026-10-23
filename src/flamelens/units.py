"""Parse text values with units and convert them to FlameLens canonical units.

Canonical units: oxygen and other gas amounts as a 0-1 fraction, pressure in atm,
lengths in mm, flow speed in cm/s, mass flow in mg/s, temperature in K, time in s.
"""

from __future__ import annotations

import math
import re
from typing import Callable

import numpy as np
import pandas as pd

# Text that means "no value" in a numeric column: blanks, dashes, and Excel error cells.
MISSING_TOKENS = {"", "-", "–", "—", "na", "n/a", "nan", "none", "null",
                  "#value!", "#div/0!", "#n/a", "#ref!", "#num!"}

_NUMBER = re.compile(
    r"""^\s*(?P<approx>[~≈])?\s*
        (?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
        \s*(?P<unit>[^\d\s].*?)?\s*$""",
    re.VERBOSE,
)

_UNIT_ALIASES = {
    "fraction": "fraction", "mole fraction": "fraction", "-": "fraction",
    "%": "%", "percent": "%", "% by vol": "%", "vol%": "%",
    "ppm": "ppm",
    "atm": "atm", "bar": "bar", "mmhg": "mmHg", "torr": "mmHg", "kpa": "kPa", "pa": "Pa",
    "psi": "psia", "psia": "psia",
    "mm": "mm", "cm": "cm", "m": "m", "um": "um", "µm": "um",
    "s": "s", "sec": "s", "ms": "ms", "min": "min",
    "cm/s": "cm/s", "mm/s": "mm/s", "m/s": "m/s",
    "mg/s": "mg/s", "g/s": "g/s", "g/min": "g/min",
    "k": "K", "c": "C", "°c": "C",
    "mm2/s": "mm2/s", "mm^2/s": "mm2/s", "mm²/s": "mm2/s",
    "w": "W", "j": "J",
}

# (unit_in, unit_out) -> function on floats or numpy arrays
_LINEAR: dict[tuple[str, str], float] = {
    ("%", "fraction"): 0.01,
    ("ppm", "fraction"): 1e-6,
    ("mmHg", "atm"): 1 / 760.0,
    ("bar", "atm"): 1 / 1.01325,
    ("kPa", "atm"): 1 / 101.325,
    ("Pa", "atm"): 1 / 101325.0,
    ("psia", "atm"): 1 / 14.6959,
    ("cm", "mm"): 10.0,
    ("m", "mm"): 1000.0,
    ("um", "mm"): 1e-3,
    ("ms", "s"): 1e-3,
    ("min", "s"): 60.0,
    ("mm/s", "cm/s"): 0.1,
    ("m/s", "cm/s"): 100.0,
    ("g/s", "mg/s"): 1000.0,
    ("g/min", "mg/s"): 1000.0 / 60.0,
}
_OFFSET: dict[tuple[str, str], Callable] = {
    ("C", "K"): lambda v: v + 273.15,
}


def canonical_unit(unit: str | None) -> str | None:
    """Return the canonical spelling of a unit, or None for no unit."""
    if unit is None:
        return None
    key = unit.strip().lower()
    if key == "":
        return None
    if key not in _UNIT_ALIASES:
        raise ValueError(f"unknown unit {unit!r}")
    return _UNIT_ALIASES[key]


def is_missing(text: object) -> bool:
    """True when a raw cell means 'no value'."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return True
    return str(text).strip().lower() in MISSING_TOKENS


def is_approximate(text: object) -> bool:
    """True when a raw number is marked approximate, as in '~9'."""
    return not is_missing(text) and str(text).strip()[:1] in ("~", "≈")


def parse_value(text: object) -> tuple[float | None, str | None]:
    """Split a raw cell into (number, unit).

    >>> parse_value("0.27 mm")
    (0.27, 'mm')
    >>> parse_value("")
    (None, None)

    Raises ValueError when the text is neither missing nor a number.
    """
    if is_missing(text):
        return None, None
    match = _NUMBER.match(str(text))
    if not match:
        raise ValueError(f"not a number: {text!r}")
    unit = match.group("unit")
    return float(match.group("num")), (canonical_unit(unit) if unit else None)


def convert(value, unit_in: str | None, unit_out: str | None):
    """Convert a float, array or Series from unit_in to unit_out."""
    a, b = canonical_unit(unit_in), canonical_unit(unit_out)
    if a == b or a is None or b is None:
        if (a is None) != (b is None):
            raise ValueError(f"cannot convert between {unit_in!r} and {unit_out!r}")
        return value
    if (a, b) in _LINEAR:
        return value * _LINEAR[(a, b)]
    if (a, b) in _OFFSET:
        return _OFFSET[(a, b)](value)
    raise ValueError(f"no conversion from {unit_in!r} to {unit_out!r}")


def parse_series(
    raw: pd.Series, unit_in: str | None, unit_out: str | None
) -> tuple[pd.Series, pd.Series]:
    """Parse a text column into canonical-unit floats.

    Returns (values, unparseable) where unparseable marks cells that held text
    which is neither a number nor a missing marker; those values become NA.
    A unit written in the cell overrides unit_in for that cell.
    """
    values = np.full(len(raw), np.nan)
    bad = np.zeros(len(raw), dtype=bool)
    for i, text in enumerate(raw.tolist()):
        try:
            number, cell_unit = parse_value(text)
        except ValueError:
            bad[i] = True
            continue
        if number is None:
            continue
        values[i] = convert(number, cell_unit or unit_in, unit_out)
    return pd.Series(values, index=raw.index, dtype="Float64"), pd.Series(bad, index=raw.index)
