"""Hypothetical circuit model: a CMOS-style driver charging an RC load.

THIS IS A TOY SIMULATOR. It is a closed-form equation chosen so that the
design-of-experiments workflow has something with realistic *structure*
(multiplicative R*C behaviour, a nonlinear supply-voltage effect) to study.
It is not a SPICE model, it ignores input slew, short-circuit current,
velocity saturation details, interconnect, temperature, and layout parasitics.
Numbers it produces are illustrative only.

Response equation
-----------------
The driver transistor is approximated by an "on" resistance that falls as the
supply voltage rises (alpha-power-law style):

    R_on(V) = K_ON / (V - V_T) ** ALPHA

The driver's own drain junction adds a small parasitic capacitance that
shrinks as the reverse bias (here, the supply) grows, like a diode junction:

    C_J(V) = C_J0 / sqrt(1 + V / PHI_B)

The driver resistance is in series with an external resistance R, and the pair
charges the load C plus the parasitic C_J.  The 50 % propagation delay of a
first-order RC step response is ln(2) * tau, so

    t_pd(R, C, V) = ln(2) * (R + R_on(V)) * (C + C_J(V))

Units: R and R_on in ohms, C in picofarads, V in volts -> ohm * pF = ps,
so t_pd comes out directly in picoseconds.

What this structure implies (useful for checking the DOE later):
  * R*C product        -> a strong R x C interaction
  * R_on(V)*C product  -> a strong C x V interaction and curvature in V
  * R*C_J(V)           -> a real but *tiny* R x V interaction (a few ps)
  * no R*C*V product   -> the three-factor interaction is exactly zero

A secondary (deterministic) quantity, switching energy per transition, is
used only as a design constraint:

    E_sw(C, V) = (C + C_J(V)) * V**2          (pF * V^2 = pJ)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LN2 = np.log(2.0)

# Hypothetical device constants (documented, not extracted from any real process)
K_ON = 400.0   # ohm * V**ALPHA, scale of the driver on-resistance
V_T = 0.5      # V, threshold voltage
ALPHA = 1.3    # velocity-saturation exponent (1 = fully saturated, 2 = long channel)
C_J0 = 0.4     # pF, zero-bias parasitic junction capacitance of the driver
PHI_B = 0.7    # V, junction built-in potential


def driver_on_resistance(v_supply):
    """Driver on-resistance in ohms for supply voltage(s) in volts."""
    v_supply = np.asarray(v_supply, dtype=float)
    if np.any(v_supply <= V_T):
        raise ValueError(f"Supply voltage must exceed the threshold V_T={V_T} V")
    return K_ON / (v_supply - V_T) ** ALPHA


def junction_capacitance(v_supply):
    """Driver parasitic capacitance in pF: C_J0 / sqrt(1 + V / PHI_B)."""
    return C_J0 / np.sqrt(1.0 + np.asarray(v_supply, dtype=float) / PHI_B)


def propagation_delay(r_ohm, c_pf, v_supply):
    """Noise-free delay in ps: ln(2) * (R + R_on(V)) * (C + C_J(V))."""
    r_ohm = np.asarray(r_ohm, dtype=float)
    c_pf = np.asarray(c_pf, dtype=float)
    return (LN2 * (r_ohm + driver_on_resistance(v_supply))
            * (c_pf + junction_capacitance(v_supply)))


def switching_energy(c_pf, v_supply):
    """Energy per 0->1 transition in pJ: (C + C_J(V)) * V^2."""
    v_supply = np.asarray(v_supply, dtype=float)
    return (np.asarray(c_pf, dtype=float) + junction_capacitance(v_supply)) * v_supply ** 2


@dataclass(frozen=True)
class NoiseModel:
    """How simulated "measurement" scatter is generated.

    Two independent sources, both Gaussian:

    * Component tolerance: each run gets its *actual* R and C drawn around the
      nominal setting, R_actual = R * (1 + N(0, r_tol_sd)) and likewise for C.
      This mimics building each run with a fresh resistor and capacitor.
    * Measurement noise: additive N(0, meas_sd_ps) picoseconds on the delay,
      mimicking oscilloscope / timing-measurement jitter.

    Supply voltage is assumed to be regulated precisely and gets no scatter.
    Set all three values to 0 for a deterministic simulator.
    """

    r_tol_sd: float = 0.01     # 1 % (1-sigma) resistor scatter
    c_tol_sd: float = 0.02     # 2 % (1-sigma) capacitor scatter
    meas_sd_ps: float = 10.0   # 10 ps (1-sigma) measurement noise


def simulate_runs(design: pd.DataFrame, noise: NoiseModel | None = None,
                  rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Evaluate the hypothetical simulator at every row of a natural-units design.

    ``design`` must contain columns ``R_ohm``, ``C_pF`` and ``V_supply``.
    Returns a copy with added columns:
      ``delay_true_ps`` (noise-free equation), ``delay_ps`` (noisy observation)
      and ``energy_pJ`` (nominal switching energy).
    """
    noise = noise or NoiseModel()
    rng = rng if rng is not None else np.random.default_rng()
    out = design.copy()
    n = len(out)

    r_actual = out["R_ohm"].to_numpy() * (1 + rng.normal(0.0, noise.r_tol_sd, n))
    c_actual = out["C_pF"].to_numpy() * (1 + rng.normal(0.0, noise.c_tol_sd, n))
    v = out["V_supply"].to_numpy()

    out["delay_true_ps"] = propagation_delay(out["R_ohm"], out["C_pF"], v)
    out["delay_ps"] = (propagation_delay(r_actual, c_actual, v)
                       + rng.normal(0.0, noise.meas_sd_ps, n))
    out["energy_pJ"] = switching_energy(out["C_pF"], v)
    return out
