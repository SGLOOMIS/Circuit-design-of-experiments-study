import numpy as np
import pandas as pd
import pytest

from doe_circuit.design import (CODES, FACTORS, axial_points, coded_to_natural, full_factorial,
                                natural_to_coded, random_validation_points)


def test_full_factorial_has_all_eight_corners_in_standard_order():
    d = full_factorial()
    assert len(d) == 8
    expected = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=float)
    np.testing.assert_array_equal(d[CODES].to_numpy(), expected)
    assert list(d["std_order"]) == list(range(1, 9))


def test_design_columns_are_balanced_and_orthogonal():
    x = full_factorial()[CODES].to_numpy()
    np.testing.assert_array_equal(x.sum(axis=0), 0)          # balanced: 4 low, 4 high
    np.testing.assert_array_equal(x.T @ x, 8 * np.eye(3))    # main effects orthogonal
    inter = np.column_stack([x[:, 0] * x[:, 1], x[:, 0] * x[:, 2], x[:, 1] * x[:, 2]])
    np.testing.assert_array_equal(x.T @ inter, 0)            # main effects vs interactions


def test_natural_units_match_documented_low_high_settings():
    d = full_factorial()
    for code, (col, lo, hi, _) in FACTORS.items():
        assert set(d.loc[d[code] == -1, col]) == {lo}
        assert set(d.loc[d[code] == 1, col]) == {hi}


def test_replicates_and_center_points():
    d = full_factorial(replicates=2, center_points=4)
    assert len(d) == 20
    assert (d["point_type"] == "center").sum() == 4
    center = d[d["point_type"] == "center"].iloc[0]
    assert center["R_ohm"] == pytest.approx(300.0)
    assert center["C_pF"] == pytest.approx(3.0)
    assert center["V_supply"] == pytest.approx(2.25)


def test_coding_round_trip():
    nat = pd.DataFrame({"R_ohm": [100, 250, 500], "C_pF": [1, 2.5, 5], "V_supply": [1.2, 2.0, 3.3]})
    coded = natural_to_coded(nat)
    np.testing.assert_allclose(coded["xR"], [-1, -0.25, 1])
    back = coded_to_natural(coded[CODES])
    np.testing.assert_allclose(back[["R_ohm", "C_pF", "V_supply"]], nat)


def test_axial_points_stay_inside_bounds_for_face_centred_ccd():
    a = axial_points(alpha=1.0)
    assert len(a) == 6
    assert (a[CODES].abs().sum(axis=1) == 1).all()   # exactly one factor off-centre
    assert a[CODES].abs().max().max() == 1


def test_validation_points_inside_box():
    v = random_validation_points(50, np.random.default_rng(1))
    assert (v[CODES].abs() <= 1).all().all()
