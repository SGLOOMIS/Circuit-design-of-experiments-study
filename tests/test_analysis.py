import numpy as np
import pandas as pd
import pytest

from doe_circuit import analysis
from doe_circuit.circuit_model import NoiseModel, propagation_delay, simulate_runs
from doe_circuit.design import CODES, axial_points, full_factorial

NO_NOISE = NoiseModel(0, 0, 0)


@pytest.fixture
def clean_factorial():
    return simulate_runs(full_factorial(), NO_NOISE, np.random.default_rng(0))


def test_coefficients_equal_half_the_classic_effects(clean_factorial):
    # For a 2^3 design: effect = mean(y at +1) - mean(y at -1) = 2 * coefficient.
    fit = analysis.fit_model(clean_factorial.assign(delay_ps=clean_factorial["delay_true_ps"]))
    y, x = clean_factorial["delay_true_ps"].to_numpy(), clean_factorial[CODES].to_numpy()
    for j, code in enumerate(CODES):
        effect = y[x[:, j] > 0].mean() - y[x[:, j] < 0].mean()
        assert 2 * fit.params[code] == pytest.approx(effect)
    xrxc = x[:, 0] * x[:, 1]
    assert 2 * fit.params["xR:xC"] == pytest.approx(y[xrxc > 0].mean() - y[xrxc < 0].mean())


def test_known_physics_recovered_from_noise_free_data(clean_factorial):
    fit = analysis.fit_model(clean_factorial.assign(delay_ps=clean_factorial["delay_true_ps"]))
    # delay contains ln2*R*C = ln2*(300 + 200 xR)*(3 + 2 xC), so the xR*xC coefficient is ln2*400
    assert fit.params["xR:xC"] == pytest.approx(np.log(2) * 400)
    # three-factor interaction is zero, so the 7-term model leaves an exactly-zero residual
    assert fit.ssr == pytest.approx(0, abs=1e-12)


def test_predictions_match_manual_linear_combination(clean_factorial):
    fit = analysis.fit_model(clean_factorial.assign(delay_ps=clean_factorial["delay_true_ps"]))
    point = pd.DataFrame({"R_ohm": [200.0], "C_pF": [4.0], "V_supply": [1.5]})
    pred = analysis.predict(fit, point).iloc[0]
    b = fit.params
    xr, xc, xv = pred["xR"], pred["xC"], pred["xV"]
    manual = (b["Intercept"] + b["xR"] * xr + b["xC"] * xc + b["xV"] * xv
              + b["xR:xC"] * xr * xc + b["xR:xV"] * xr * xv + b["xC:xV"] * xc * xv)
    assert pred["pred_ps"] == pytest.approx(manual)
    assert pred["pi_low_ps"] <= pred["pred_ps"] <= pred["pi_high_ps"]


def test_factorial_model_reproduces_corners_exactly(clean_factorial):
    fit = analysis.fit_model(clean_factorial.assign(delay_ps=clean_factorial["delay_true_ps"]))
    pred = analysis.predict(fit, clean_factorial[["R_ohm", "C_pF", "V_supply"]])
    np.testing.assert_allclose(pred["pred_ps"], clean_factorial["delay_true_ps"], rtol=1e-9)


def test_log_model_predictions_are_back_transformed():
    rng = np.random.default_rng(3)
    ccd = pd.concat([simulate_runs(full_factorial(center_points=2), NO_NOISE, rng),
                     simulate_runs(axial_points(), NO_NOISE, rng)], ignore_index=True)
    fit = analysis.fit_model(ccd, analysis.LOG_QUADRATIC_FORMULA)
    assert analysis.is_log_model(fit)
    pred = analysis.predict(fit, ccd[["R_ohm", "C_pF", "V_supply"]])
    np.testing.assert_allclose(pred["pred_ps"], np.exp(fit.fittedvalues), rtol=1e-9)
    # a decent approximation of the true equation at the design points
    rel = pred["pred_ps"] / ccd["delay_true_ps"] - 1
    assert rel.abs().max() < 0.10


def test_curvature_test_detects_curvature():
    data = simulate_runs(full_factorial(replicates=2, center_points=4), NoiseModel(),
                         np.random.default_rng(11))
    res = analysis.curvature_test(data)
    assert res["mean_center"] < res["mean_factorial"]
    assert res["p_value"] < 0.001


def test_lack_of_fit_partition():
    rng = np.random.default_rng(5)
    data = simulate_runs(full_factorial(replicates=2, center_points=3), NoiseModel(), rng)
    fit = analysis.fit_model(data, analysis.FACTORIAL_FORMULA)
    lof = analysis.lack_of_fit_test(fit, data)
    assert lof["df_pure_error"] == 8 + 2              # 8 corner pairs + 3 centres
    assert lof["df_lack_of_fit"] + lof["df_pure_error"] == fit.df_resid
    assert lof["ss_lack_of_fit"] + lof["ss_pure_error"] == pytest.approx(fit.ssr)


def test_validation_metrics():
    m = analysis.validation_metrics([1, 2, 3, 4], [1, 2, 3, 6])
    assert m["rmse_ps"] == pytest.approx(1.0)
    assert m["mae_ps"] == pytest.approx(0.5)
    assert m["max_abs_err_ps"] == pytest.approx(2.0)
    assert m["bias_ps"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(1 - 4 / 5)


def test_effects_table_flags():
    rng = np.random.default_rng(2)
    data = simulate_runs(full_factorial(replicates=2), NoiseModel(), rng)
    eff = analysis.effects_table(analysis.fit_model(data), practical_ps=50)
    assert eff.loc["xC", "statistically_significant"] and eff.loc["xC", "practically_significant"]
    assert not eff.loc["xR:xV", "practically_significant"]
    assert (eff["change_ps"] == 2 * eff["coef_ps"]).all()


def test_optimiser_respects_bounds_and_energy_budget():
    rng = np.random.default_rng(9)
    ccd = pd.concat([simulate_runs(full_factorial(center_points=2), NO_NOISE, rng),
                     simulate_runs(axial_points(), NO_NOISE, rng)], ignore_index=True)
    fit = analysis.fit_model(ccd, analysis.LOG_QUADRATIC_FORMULA)
    best = analysis.optimise_delay(fit, energy_budget_pj=6.0)
    assert best["energy_pJ"] <= 6.0 + 1e-6
    assert all(-1 - 1e-9 <= best[c] <= 1 + 1e-9 for c in CODES)
    # within 15 % of the noise-free simulator at the chosen point
    true = propagation_delay(best["R_ohm"], best["C_pF"], best["V_supply"])
    assert best["pred_delay_ps"] == pytest.approx(true, rel=0.15)
