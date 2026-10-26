"""Regression, ANOVA, curvature test, response-surface optimisation and validation.

All models are fitted in coded units (xR, xC, xV in [-1, +1]) with plain
statsmodels formulas, so each term reads exactly like the textbook model:

    y = b0 + bR*xR + bC*xC + bV*xV                 (main effects)
           + bRC*xR*xC + bRV*xR*xV + bCV*xC*xV       (two-factor interactions)
           [+ bRR*xR^2 + bCC*xC^2 + bVV*xV^2]        (quadratic, response surface only)

The three-factor interaction xR:xC:xV is deliberately left out of the
factorial model; its degree of freedom is pooled into the error estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize, stats

from .circuit_model import switching_energy
from .design import CODES, FACTORS, coded_to_natural, natural_to_coded

FACTORIAL_FORMULA = "delay_ps ~ xR + xC + xV + xR:xC + xR:xV + xC:xV"
QUADRATIC_TERMS = " + I(xR**2) + I(xC**2) + I(xV**2)"
QUADRATIC_FORMULA = FACTORIAL_FORMULA + QUADRATIC_TERMS
# Response-surface model on log(delay).  Delay is a *product* of a resistance
# term and a capacitance term, so log(delay) is close to additive and a
# quadratic approximates it far better than it approximates delay itself.
# The log also turns the multiplicative component-tolerance noise into
# roughly constant-variance noise, which OLS assumes.
LOG_QUADRATIC_FORMULA = "np.log(delay_ps) ~" + QUADRATIC_FORMULA.split("~", 1)[1]

# A hypothetical engineering threshold: changes smaller than this are treated
# as irrelevant to the design (e.g. below the timing-margin granularity).
PRACTICAL_THRESHOLD_PS = 50.0


def fit_model(data: pd.DataFrame, formula: str = FACTORIAL_FORMULA):
    """Ordinary least squares fit of ``formula`` to ``data`` (coded columns)."""
    return smf.ols(formula, data=data).fit()


def is_log_model(fit) -> bool:
    """True when the fitted response is np.log(delay_ps)."""
    return fit.model.formula.replace(" ", "").startswith("np.log(")


def predict_ps(fit, coded: pd.DataFrame) -> np.ndarray:
    """Point predictions in picoseconds for coded points (undoes the log if any).

    exp(mean of log) is the *median* delay; with ~2 % scatter the gap to the
    mean is ~0.02 %, far below anything that matters here.
    """
    pred = fit.predict(coded[CODES]).to_numpy()
    return np.exp(pred) if is_log_model(fit) else pred


def anova_table(fit) -> pd.DataFrame:
    """Type-II ANOVA (sums of squares, F, p) for every model term.

    For an orthogonal two-level design Type I, II and III sums of squares are
    identical; Type II is used so the table is also sensible for the CCD.
    """
    table = sm.stats.anova_lm(fit, typ=2)
    table["mean_sq"] = table["sum_sq"] / table["df"]
    return table[["df", "sum_sq", "mean_sq", "F", "PR(>F)"]]


def effects_table(fit, practical_ps: float = PRACTICAL_THRESHOLD_PS,
                  alpha: float = 0.05) -> pd.DataFrame:
    """Coefficients with statistical *and* practical significance flags.

    Intended for models of delay in ps (not the log model).
    ``change_ps`` is how much the term moves the predicted delay across the
    tested range: for a linear or interaction term going from -1 to +1 the
    change is 2 * coefficient (the classic DOE "effect"); for a squared term
    x^2 only spans 0..1, so the change is the coefficient itself.
    """
    ci = fit.conf_int(alpha)
    rows = []
    for term in fit.params.index:
        if term == "Intercept":
            continue
        coef = fit.params[term]
        scale = 1.0 if term.startswith("I(") else 2.0
        rows.append({
            "term": term,
            "coef_ps": coef,
            "ci_low_ps": ci.loc[term, 0],
            "ci_high_ps": ci.loc[term, 1],
            "change_ps": scale * coef,
            "p_value": fit.pvalues[term],
            "statistically_significant": fit.pvalues[term] < alpha,
            "practically_significant": abs(scale * coef) >= practical_ps,
        })
    return pd.DataFrame(rows).set_index("term")


def curvature_test(data: pd.DataFrame, response: str = "delay_ps") -> dict:
    """Single-degree-of-freedom curvature test using centre points.

    A two-level model predicts the centre of the box to equal the mean of the
    corner runs.  If the centre-point mean differs by more than noise allows,
    the response is curved and a quadratic (response-surface) model is needed.

        curvature = mean(factorial) - mean(center)
        se        = sqrt(MSE * (1/n_F + 1/n_C))
        t         = curvature / se,   df = residual df of the factorial fit

    MSE comes from the factorial model fitted to the corner runs only.
    """
    factorial = data[data["point_type"] == "factorial"]
    center = data[data["point_type"] == "center"]
    fit = fit_model(factorial, FACTORIAL_FORMULA.replace("delay_ps", response))
    n_f, n_c = len(factorial), len(center)
    curvature = factorial[response].mean() - center[response].mean()
    se = np.sqrt(fit.mse_resid * (1 / n_f + 1 / n_c))
    t = curvature / se
    p = 2 * stats.t.sf(abs(t), fit.df_resid)
    return {"mean_factorial": factorial[response].mean(), "mean_center": center[response].mean(),
            "curvature_ps": curvature, "se_ps": se, "t": t, "df": fit.df_resid, "p_value": p}


def lack_of_fit_test(fit, data: pd.DataFrame) -> dict:
    """Split residual error into pure error (replicates) and lack of fit.

    Runs at identical settings differ only by noise, so their scatter around
    their own mean is *pure error*.  Whatever residual is left over is error
    the model's shape cannot explain:

        SS_LOF = SS_resid - SS_PE,   F = (SS_LOF / df_LOF) / (SS_PE / df_PE)

    A small p-value means the model is biased somewhere in the region and its
    prediction intervals (which only account for noise) will be too narrow.
    ``data`` must be the frame the model was fitted on, in the same row order.
    """
    y = pd.Series(np.asarray(fit.model.endog), index=data.index)
    groups = data[CODES].round(9).apply(tuple, axis=1)
    ss_pe = float(((y - y.groupby(groups).transform("mean")) ** 2).sum())
    df_pe = int(len(y) - groups.nunique())
    ss_lof = float(fit.ssr - ss_pe)
    df_lof = int(fit.df_resid - df_pe)
    f = (ss_lof / df_lof) / (ss_pe / df_pe)
    return {"ss_lack_of_fit": ss_lof, "df_lack_of_fit": df_lof, "ss_pure_error": ss_pe,
            "df_pure_error": df_pe, "F": f, "p_value": float(stats.f.sf(f, df_lof, df_pe))}


def predict(fit, natural: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Predictions with confidence and prediction intervals for natural-unit points."""
    coded = natural_to_coded(natural)
    frame = fit.get_prediction(coded).summary_frame(alpha=alpha)
    back = np.exp if is_log_model(fit) else (lambda a: a)  # interval ends map through exp
    out = coded.copy()
    out["pred_ps"] = back(frame["mean"].to_numpy())
    out["ci_low_ps"] = back(frame["mean_ci_lower"].to_numpy())
    out["ci_high_ps"] = back(frame["mean_ci_upper"].to_numpy())
    out["pi_low_ps"] = back(frame["obs_ci_lower"].to_numpy())
    out["pi_high_ps"] = back(frame["obs_ci_upper"].to_numpy())
    return out


def validation_metrics(observed, predicted) -> dict:
    """Out-of-sample error summary (all in ps except R^2)."""
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    err = predicted - observed
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)
    return {"n": len(observed), "rmse_ps": float(np.sqrt(np.mean(err ** 2))),
            "mae_ps": float(np.mean(np.abs(err))), "max_abs_err_ps": float(np.max(np.abs(err))),
            "bias_ps": float(np.mean(err)), "r2": float(1 - ss_res / ss_tot)}


def optimise_delay(fit, energy_budget_pj: float, n_starts: int = 20,
                   rng: np.random.Generator | None = None) -> pd.Series:
    """Minimise *predicted* delay inside the tested box subject to an energy budget.

    The search stays inside the coded box [-1, 1]^3, i.e. within the tested
    bounds; the fitted model is never asked to extrapolate.  Energy is not a
    fitted response: it is the deterministic formula E = (C + C_J(V)) V^2.
    SLSQP is restarted from several random points to avoid local minima.
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    def to_frame(x):
        return coded_to_natural(pd.DataFrame([x], columns=CODES))

    def objective(x):
        return float(predict_ps(fit, pd.DataFrame([x], columns=CODES))[0])

    def energy_slack(x):
        row = to_frame(x).iloc[0]
        return energy_budget_pj - float(switching_energy(row["C_pF"], row["V_supply"]))

    best = None
    for x0 in rng.uniform(-1, 1, size=(n_starts, 3)):
        res = optimize.minimize(objective, x0, method="SLSQP", bounds=[(-1, 1)] * 3,
                                constraints=[{"type": "ineq", "fun": energy_slack}])
        if res.success and energy_slack(res.x) >= -1e-6 and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("No feasible configuration found within the tested bounds")
    row = to_frame(np.clip(best.x, -1, 1)).iloc[0]
    row["pred_delay_ps"] = best.fun
    row["energy_pJ"] = float(switching_energy(row["C_pF"], row["V_supply"]))
    return row


def surface_grid(fit, fixed: dict, x_code: str, y_code: str, n: int = 61) -> tuple:
    """Predicted delay on an n x n grid over two coded factors, third held fixed.

    Returns (x_natural, y_natural, Z) suitable for matplotlib.contour.
    """
    g = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(g, g)
    coded = pd.DataFrame({x_code: xx.ravel(), y_code: yy.ravel()})
    for code, value in fixed.items():
        coded[code] = value
    z = predict_ps(fit, coded).reshape(xx.shape)
    nat = coded_to_natural(coded[CODES])
    xcol, ycol = FACTORS[x_code][0], FACTORS[y_code][0]
    return nat[xcol].to_numpy().reshape(xx.shape), nat[ycol].to_numpy().reshape(xx.shape), z
