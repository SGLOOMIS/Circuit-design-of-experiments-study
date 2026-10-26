import numpy as np
import pandas as pd
import pytest

from doe_circuit.circuit_model import (NoiseModel, driver_on_resistance, junction_capacitance,
                                       propagation_delay, simulate_runs, switching_energy)


def test_on_resistance_hand_calculation():
    # 400 / (3.3 - 0.5)^1.3
    assert driver_on_resistance(3.3) == pytest.approx(104.89494, rel=1e-6)
    assert driver_on_resistance(1.2) == pytest.approx(400 / 0.7 ** 1.3)


def test_junction_capacitance_hand_calculation():
    assert junction_capacitance(0.0) == pytest.approx(0.4)
    assert junction_capacitance(2.1) == pytest.approx(0.4 / 2.0)   # sqrt(1 + 2.1/0.7) = 2


def test_delay_hand_calculation():
    expected = np.log(2) * (100 + 104.894943) * (1 + 0.4 / np.sqrt(1 + 3.3 / 0.7))
    assert propagation_delay(100, 1, 3.3) == pytest.approx(expected, rel=1e-6)
    assert propagation_delay(100, 1, 3.3) == pytest.approx(165.787, abs=1e-3)


def test_delay_monotonic_in_each_factor():
    assert propagation_delay(500, 3, 2) > propagation_delay(100, 3, 2)
    assert propagation_delay(300, 5, 2) > propagation_delay(300, 1, 2)
    assert propagation_delay(300, 3, 3.3) < propagation_delay(300, 3, 1.2)


def test_delay_is_linear_in_r():
    d = [propagation_delay(r, 2, 2) for r in (100, 200, 300)]
    assert d[2] - d[1] == pytest.approx(d[1] - d[0])


def test_delay_vectorised():
    out = propagation_delay(np.array([100, 500]), np.array([1, 5]), np.array([3.3, 1.2]))
    assert out.shape == (2,)


def test_supply_below_threshold_rejected():
    with pytest.raises(ValueError):
        propagation_delay(100, 1, 0.4)


def test_switching_energy():
    assert switching_energy(2.0, 2.1) == pytest.approx((2.0 + 0.2) * 2.1 ** 2)


def test_noise_free_simulation_equals_equation():
    design = pd.DataFrame({"R_ohm": [100, 500], "C_pF": [1, 5], "V_supply": [3.3, 1.2]})
    out = simulate_runs(design, NoiseModel(0, 0, 0), np.random.default_rng(0))
    np.testing.assert_allclose(out["delay_ps"], out["delay_true_ps"])


def test_noise_is_reproducible_and_has_expected_size():
    design = pd.DataFrame({"R_ohm": [300] * 4000, "C_pF": [3] * 4000, "V_supply": [2.25] * 4000})
    a = simulate_runs(design, NoiseModel(), np.random.default_rng(7))
    b = simulate_runs(design, NoiseModel(), np.random.default_rng(7))
    np.testing.assert_array_equal(a["delay_ps"], b["delay_ps"])
    err = a["delay_ps"] - a["delay_true_ps"]
    true = a["delay_true_ps"].iloc[0]
    # Expected SD ~ sqrt((1% of R share)^2 + (2% of C share)^2 + 10^2); check loosely
    assert abs(err.mean()) < 3.0
    assert 10 < err.std() < 0.05 * true
