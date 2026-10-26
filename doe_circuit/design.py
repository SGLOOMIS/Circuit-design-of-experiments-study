"""Experimental designs for the three circuit factors.

Factors and their tested bounds (natural units)
-----------------------------------------------
| Code | Column    | Meaning                     | Low (-1) | High (+1) |
|------|-----------|-----------------------------|----------|-----------|
| xR   | R_ohm     | series output resistance    | 100 ohm  | 500 ohm   |
| xC   | C_pF      | load capacitance            | 1.0 pF   | 5.0 pF    |
| xV   | V_supply  | supply voltage              | 1.2 V    | 3.3 V     |

Coded units map each factor linearly onto [-1, +1]:

    x = (value - center) / half_range,   center = (low + high) / 2,
                                         half_range = (high - low) / 2

Regression is done in coded units so that every coefficient is on the same
scale and equals half of the classic "low-to-high" effect.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

# code -> (natural column, low, high, unit)
FACTORS = {
    "xR": ("R_ohm", 100.0, 500.0, "ohm"),
    "xC": ("C_pF", 1.0, 5.0, "pF"),
    "xV": ("V_supply", 1.2, 3.3, "V"),
}
CODES = list(FACTORS)


def factor_table() -> pd.DataFrame:
    """The documented low/high settings as a DataFrame (for reports)."""
    rows = []
    for code, (col, lo, hi, unit) in FACTORS.items():
        rows.append({"code": code, "column": col, "unit": unit,
                     "low (-1)": lo, "center (0)": (lo + hi) / 2, "high (+1)": hi})
    return pd.DataFrame(rows)


def coded_to_natural(coded: pd.DataFrame) -> pd.DataFrame:
    """Add natural-unit columns to a frame of coded columns xR, xC, xV."""
    out = coded.copy()
    for code, (col, lo, hi, _) in FACTORS.items():
        # Same as center + half_range * x, written as an interpolation so that
        # x = -1 and x = +1 give exactly the documented low/high floats.
        t = (out[code] + 1) / 2
        out[col] = lo * (1 - t) + hi * t
    return out


def natural_to_coded(natural: pd.DataFrame) -> pd.DataFrame:
    """Add coded columns xR, xC, xV to a frame of natural-unit columns."""
    out = natural.copy()
    for code, (col, lo, hi, _) in FACTORS.items():
        center, half = (lo + hi) / 2, (hi - lo) / 2
        out[code] = (out[col] - center) / half
    return out


def full_factorial(replicates: int = 1, center_points: int = 0) -> pd.DataFrame:
    """2^3 full factorial in standard (Yates) order, optionally replicated.

    Standard order varies xR fastest, then xC, then xV:
        run 1: (-,-,-), run 2: (+,-,-), run 3: (-,+,-), ..., run 8: (+,+,+)

    ``center_points`` all-zero runs are appended; they are not needed to fit
    the two-level model but give a pure-error estimate and a curvature check.
    """
    corners = [levels[::-1] for levels in itertools.product([-1.0, 1.0], repeat=3)]
    rows = []
    for rep in range(1, replicates + 1):
        for std_order, levels in enumerate(corners, start=1):
            rows.append({"std_order": std_order, "replicate": rep, "point_type": "factorial",
                         **dict(zip(CODES, levels))})
    for i in range(center_points):
        rows.append({"std_order": 9 + i, "replicate": 1, "point_type": "center",
                     **dict.fromkeys(CODES, 0.0)})
    return coded_to_natural(pd.DataFrame(rows))


def axial_points(alpha: float = 1.0) -> pd.DataFrame:
    """The 6 axial ("star") runs that turn a 2^3 factorial into a central composite design.

    alpha = 1 gives a face-centred CCD, which keeps every run inside the
    tested low/high bounds (important: we must not extrapolate the circuit).
    """
    rows = []
    for code in CODES:
        for sign in (-1.0, 1.0):
            point = dict.fromkeys(CODES, 0.0)
            point[code] = sign * alpha
            rows.append({"point_type": "axial", "replicate": 1, **point})
    return coded_to_natural(pd.DataFrame(rows))


def random_validation_points(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Uniformly random points inside the bounds, for out-of-sample validation."""
    coded = pd.DataFrame(rng.uniform(-1.0, 1.0, size=(n, 3)), columns=CODES)
    coded.insert(0, "point_type", "validation")
    return coded_to_natural(coded)
