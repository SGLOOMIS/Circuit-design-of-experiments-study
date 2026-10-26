"""Run the full DOE study on the hypothetical RC-driver simulator.

Usage:  python run_study.py [--seed 2026] [--out outputs]

Stages (each builds on the previous one):
  1. Response equation sanity check + 2^3 factorial with centre points
  2. Factorial regression, ANOVA, curvature test, effect plots
  3. Augment to a face-centred CCD, fit a response surface, optimise
  4. Validate on fresh random points that were not used for fitting
  5. Statistical vs practical significance demonstration
Everything is written to the output folder: CSVs, PNGs and report.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from doe_circuit import analysis, plots
from doe_circuit.circuit_model import (NoiseModel, propagation_delay, simulate_runs,
                                       switching_energy)
from doe_circuit.design import (FACTORS, axial_points, factor_table, full_factorial,
                                random_validation_points)

ENERGY_BUDGET_PJ = 6.0
N_VALIDATION = 30


def fmt(df: pd.DataFrame, floatfmt: str = "{:,.4g}") -> str:
    """DataFrame as a fenced plain-text block for the markdown report."""
    return "```\n" + df.to_string(float_format=floatfmt.format) + "\n```"


def max_voltage_for_budget(c_pf: float, budget: float) -> float:
    """Largest V in the tested range with switching energy <= budget (bisection)."""
    lo, hi = FACTORS["xV"][1], FACTORS["xV"][2]
    if switching_energy(c_pf, hi) <= budget:
        return hi
    if switching_energy(c_pf, lo) > budget:
        return lo
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if switching_energy(c_pf, mid) <= budget else (lo, mid)
    return lo


def main(seed: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    noise = NoiseModel()
    report = []

    # ---- Stage 1: equation check and small factorial --------------------------------
    check = propagation_delay(100.0, 1.0, 3.3)
    stage1 = simulate_runs(full_factorial(replicates=2, center_points=4), noise, rng)
    stage1.to_csv(out / "stage1_factorial_runs.csv", index=False)

    # ---- Stage 2: factorial regression and ANOVA -----------------------------------
    factorial_only = stage1[stage1["point_type"] == "factorial"]
    fact_fit = analysis.fit_model(factorial_only, analysis.FACTORIAL_FORMULA)
    fact_anova = analysis.anova_table(fact_fit)
    fact_effects = analysis.effects_table(fact_fit)
    curvature = analysis.curvature_test(stage1)
    fact_anova.to_csv(out / "factorial_anova.csv")
    fact_effects.to_csv(out / "factorial_effects.csv")
    plots.main_effects(stage1, out / "main_effects.png")
    plots.interaction_effects(stage1, out / "interaction_effects.png")

    # ---- Stage 3: response surface ------------------------------------------------
    axial = simulate_runs(axial_points(alpha=1.0), noise, rng)
    ccd = pd.concat([stage1, axial], ignore_index=True)
    ccd.to_csv(out / "ccd_runs.csv", index=False)
    rsm_fit = analysis.fit_model(ccd, analysis.LOG_QUADRATIC_FORMULA)
    rsm_anova = analysis.anova_table(rsm_fit)
    rsm_lof = analysis.lack_of_fit_test(rsm_fit, ccd)
    raw_quad_fit = analysis.fit_model(ccd, analysis.QUADRATIC_FORMULA)  # for comparison only
    rsm_anova.to_csv(out / "rsm_anova.csv")

    best = analysis.optimise_delay(rsm_fit, ENERGY_BUDGET_PJ, rng=np.random.default_rng(seed))
    best_true = float(propagation_delay(best["R_ohm"], best["C_pF"], best["V_supply"]))
    confirm = simulate_runs(pd.DataFrame([best[["R_ohm", "C_pF", "V_supply"]]] * 5), noise, rng)

    c_panels = [1.0, 3.0]
    panels = []
    for c_val in c_panels:
        xc = (c_val - (FACTORS["xC"][1] + FACTORS["xC"][2]) / 2) / ((FACTORS["xC"][2] - FACTORS["xC"][1]) / 2)
        panels.append((c_val, *analysis.surface_grid(rsm_fit, {"xC": xc}, "xR", "xV")))
    plots.response_surface(panels, [max_voltage_for_budget(c, ENERGY_BUDGET_PJ) for c in c_panels],
                           best, out / "response_surface.png")

    # ---- Stage 4: out-of-sample validation -----------------------------------------
    val = simulate_runs(random_validation_points(N_VALIDATION, rng), noise, rng)
    val_pred = analysis.predict(rsm_fit, val)
    val = pd.concat([val, val_pred[["pred_ps", "pi_low_ps", "pi_high_ps"]]], axis=1)
    val.to_csv(out / "validation_points.csv", index=False)
    m_obs = analysis.validation_metrics(val["delay_ps"], val["pred_ps"])
    m_true = analysis.validation_metrics(val["delay_true_ps"], val["pred_ps"])
    m_raw = analysis.validation_metrics(val["delay_ps"], analysis.predict(raw_quad_fit, val)["pred_ps"])
    coverage = float(((val["delay_ps"] >= val["pi_low_ps"]) & (val["delay_ps"] <= val["pi_high_ps"])).mean())
    plots.validation(val, out / "validation.png")

    # ---- Stage 5: statistical vs practical significance ----------------------------
    big = simulate_runs(full_factorial(replicates=200), noise, rng)
    big_effects = analysis.effects_table(analysis.fit_model(big))
    # "True" effects: the same factorial model fitted to the noise-free equation
    true_effects = analysis.effects_table(analysis.fit_model(
        simulate_runs(full_factorial(), NoiseModel(0, 0, 0), rng).assign(
            delay_ps=lambda d: d["delay_true_ps"])))
    combined = pd.concat([fact_effects,
                          big_effects.loc[["xR:xV"]].rename(index={"xR:xV": "xR:xV (1600 runs)"})])
    plots.effect_estimates(combined, analysis.PRACTICAL_THRESHOLD_PS, out / "effect_estimates.png")

    # ---- Report --------------------------------------------------------------------
    sig = combined[["change_ps", "p_value", "statistically_significant", "practically_significant"]]
    sig.insert(1, "true_change_ps", [true_effects.loc[t.split(" ")[0], "change_ps"] for t in sig.index])
    report += [
        "# DOE study report: hypothetical RC-driver propagation delay",
        "",
        f"Generated by `run_study.py --seed {seed}`. **Every number below comes from a hypothetical "
        "closed-form simulator** (`doe_circuit/circuit_model.py`), not from SPICE or hardware. "
        "See the README section *Hypothetical Simulator and Real Circuits*.",
        "",
        "## 1. Factors and response",
        fmt(factor_table().set_index("code")),
        "Response: `t_pd = ln2 * (R + R_on(V)) * (C + C_J(V))` in ps, "
        "with `R_on = 400/(V-0.5)^1.3` ohm and `C_J = 0.4/sqrt(1+V/0.7)` pF.",
        f"Hand check: t_pd(100 ohm, 1 pF, 3.3 V) = {check:.2f} ps.",
        "",
        "Noise: each run draws actual R ~ R*(1+N(0,1 %)), C ~ C*(1+N(0,2 %)), and adds "
        f"N(0, 10 ps) measurement noise. Seeded with `numpy.random.default_rng({seed})`.",
        "",
        "## 2. Factorial study (2^3, 2 replicates, + 4 centre points = 20 runs)",
        f"Model: `{analysis.FACTORIAL_FORMULA}` (coded units, fitted to the 16 corner runs).",
        "",
        "### ANOVA",
        fmt(fact_anova),
        "### Effects (change in delay from low to high level)",
        fmt(fact_effects),
        f"R^2 = {fact_fit.rsquared:.5f}, residual SD = {np.sqrt(fact_fit.mse_resid):.1f} ps.",
        "",
        "### Curvature check",
        f"Mean of corner runs = {curvature['mean_factorial']:.1f} ps, mean of centre runs = "
        f"{curvature['mean_center']:.1f} ps, difference = {curvature['curvature_ps']:.1f} ps "
        f"(t = {curvature['t']:.1f}, p = {curvature['p_value']:.2g}). "
        + ("Significant curvature: a two-level model cannot describe the interior, so the design "
           "is augmented to a response surface." if curvature["p_value"] < 0.05 else
           "No significant curvature detected."),
        "",
        "Figures: `main_effects.png`, `interaction_effects.png`.",
        "",
        "## 3. Response surface (face-centred CCD, 26 runs)",
        f"Model: `{analysis.LOG_QUADRATIC_FORMULA}`",
        "",
        fmt(rsm_anova),
        f"R^2 (log scale) = {rsm_fit.rsquared:.5f}.",
        "",
        "Note that interactions depend on the scale of the response: on the log scale R x C "
        "vanishes (log of a product is a sum) while R x V becomes large (log(R + R_on(V)) is not "
        "additive in R and V). 'Is there an interaction?' always means 'on which scale?'.",
        "",
        f"Lack-of-fit test: F = {rsm_lof['F']:.2f} on ({rsm_lof['df_lack_of_fit']}, "
        f"{rsm_lof['df_pure_error']}) df, p = {rsm_lof['p_value']:.2g}. "
        + ("**The quadratic shape is detectably wrong somewhere in the box**, so prediction "
           "intervals will understate the real error (see validation)." if rsm_lof["p_value"] < 0.05
           else "No significant lack of fit."),
        "",
        f"### Proposed configuration (minimise predicted delay, energy <= {ENERGY_BUDGET_PJ:g} pJ)",
        fmt(best.to_frame("value")),
        f"Noise-free simulator at that point: {best_true:.1f} ps "
        f"(model error {100 * (best['pred_delay_ps'] / best_true - 1):+.1f} %). "
        f"Five confirmation runs: mean {confirm['delay_ps'].mean():.1f} ps, "
        f"SD {confirm['delay_ps'].std():.1f} ps.",
        "",
        "Within the tested bounds the optimiser pushes R and C to their lower limits and raises V "
        "until the energy budget binds. In a real design C is usually fixed by the load being "
        "driven, so the `C = 3 pF` panel of `response_surface.png` is the more realistic view.",
        "",
        "## 4. Validation on held-out points",
        f"{N_VALIDATION} uniformly random points inside the box, simulated with fresh noise.",
        "",
        fmt(pd.DataFrame({"log-quadratic vs observed": m_obs, "log-quadratic vs noise-free": m_true,
                          "raw-ps quadratic vs observed": m_raw})),
        f"95 % prediction-interval coverage: {coverage:.0%} (nominal 95 %).",
        "",
        "The log-scale model is clearly better than the same quadratic fitted to delay in ps. "
        "Coverage below nominal is the practical symptom of the lack of fit found above: the "
        "intervals account for noise, not for the quadratic's systematic error. The right panel "
        "of `validation.png` shows the pattern: errors trend from positive at low V to negative "
        "at high V, because R_on ~ (V - 0.5)^-1.3 is not quadratic even on the log scale.",
        "",
        "## 5. Statistical vs practical significance",
        f"Practical threshold: {analysis.PRACTICAL_THRESHOLD_PS:g} ps.",
        fmt(sig),
        f"The true R x V effect built into the simulator is "
        f"{true_effects.loc['xR:xV', 'change_ps']:+.2f} ps (from the C_J(V) parasitic). In the 16-run "
        "study it is not statistically detectable. Repeating the same factorial 200 times "
        "(1600 runs, cheap in simulation, absurd on a bench) makes it highly significant, yet it is "
        "still far below anything an engineer would act on. Figure: `effect_estimates.png`.",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    main(args.seed, args.out)
