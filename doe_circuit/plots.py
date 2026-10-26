"""Matplotlib figures for the DOE study (static PNGs, light theme)."""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .design import CODES, FACTORS  # noqa: E402

# Colours by job: categorical identity (fixed order), one sequential hue, ink for text.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.labelsize": 10, "font.size": 9, "lines.linewidth": 2,
})


def _label(code: str) -> str:
    col, lo, hi, unit = FACTORS[code]
    return f"{col}  [{lo:g} to {hi:g} {unit}]"


def main_effects(data: pd.DataFrame, path, response: str = "delay_ps"):
    """Mean response at the low and high level of each factor (factorial runs).

    The centre-point mean is drawn as a hollow marker: if the response were
    linear it would sit on the line, so its distance from the line is curvature.
    """
    fact = data[data["point_type"] == "factorial"]
    center = data[data["point_type"] == "center"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, code in zip(axes, CODES):
        means = fact.groupby(code)[response].mean()
        ax.plot([-1, 1], means.to_numpy(), color=SERIES[0], marker="o", markersize=8,
                label="factorial mean")
        if len(center):
            ax.plot([0], [center[response].mean()], marker="o", markersize=9, mfc=SURFACE,
                    mec=SERIES[1], mew=2, ls="none", label="centre-point mean")
        ax.axhline(fact[response].mean(), color=INK_2, lw=1, ls="--")
        ax.set_xticks([-1, 0, 1], ["low", "centre", "high"])
        ax.set_xlabel(_label(code))
    axes[0].set_ylabel("mean delay (ps)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Main effects on propagation delay (hypothetical simulator)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def interaction_effects(data: pd.DataFrame, path, response: str = "delay_ps"):
    """Two-factor interaction plots: non-parallel lines indicate an interaction."""
    fact = data[data["point_type"] == "factorial"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (a, b) in zip(axes, itertools.combinations(CODES, 2)):
        for colour, level, name in ((SERIES[0], -1, "low"), (SERIES[1], 1, "high")):
            means = fact[fact[b] == level].groupby(a)[response].mean()
            ax.plot([-1, 1], means.to_numpy(), color=colour, marker="o", markersize=8,
                    label=f"{FACTORS[b][0]} {name}")
            ax.annotate(f"{FACTORS[b][0]} {name}", (1, means.iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK_2, fontsize=8)
        ax.set_xticks([-1, 1], ["low", "high"])
        ax.set_xlim(-1.3, 1.9)
        ax.set_xlabel(_label(a))
        ax.set_title(f"{FACTORS[a][0]} x {FACTORS[b][0]}", loc="left")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    axes[0].set_ylabel("mean delay (ps)")
    fig.suptitle("Two-factor interactions (parallel lines = no interaction)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def effect_estimates(effects: pd.DataFrame, threshold_ps: float, path):
    """Dot-and-whisker plot of effect estimates with the practical-threshold band."""
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(effects) + 1.4))
    y = np.arange(len(effects))[::-1]
    ax.axvspan(-threshold_ps, threshold_ps, color=GRID, alpha=0.9, lw=0,
               label=f"|effect| < {threshold_ps:g} ps: practically negligible")
    ax.axvline(0, color=INK_2, lw=1)
    for yi, (term, row) in zip(y, effects.iterrows()):
        colour = SERIES[0] if row["statistically_significant"] else INK_2
        ax.plot([2 * row["ci_low_ps"], 2 * row["ci_high_ps"]], [yi, yi], color=colour)
        ax.plot(row["change_ps"], yi, "o", color=colour, markersize=8, mec=SURFACE, mew=2)
        ax.annotate(f"{row['change_ps']:+.1f} ps, p={row['p_value']:.2g}", (row["change_ps"], yi),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, effects.index)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("effect over tested range (ps, symlog scale) with 95 % CI\n"
                  "blue = statistically significant at 5 %, grey = not")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_title("Statistical vs practical significance", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def response_surface(panels: list, energy_vmax: list, optimum: pd.Series, path):
    """Contours of predicted delay over R and V, one panel per fixed C.

    ``panels`` holds (C value, R grid, V grid, Z) tuples from analysis.surface_grid.
    ``energy_vmax`` gives, per panel, the highest V allowed by the energy budget.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2), sharey=True)
    levels = np.geomspace(min(p[3].min() for p in panels), max(p[3].max() for p in panels), 14)
    for ax, (c_val, rr, vv, zz), vmax in zip(np.atleast_1d(axes), panels, energy_vmax):
        cf = ax.contourf(rr, vv, zz, levels=levels, cmap=BLUES)
        cs = ax.contour(rr, vv, zz, levels=levels[::2], colors=SURFACE, linewidths=0.8)
        ax.clabel(cs, fmt="%.0f", fontsize=7, colors=INK)
        if vmax < vv.max():
            ax.fill_between([rr.min(), rr.max()], vmax, vv.max(), color=SURFACE, alpha=0.65,
                            hatch="//", edgecolor=INK_2, lw=0)
            ax.axhline(vmax, color=SERIES[1], lw=2)
            ax.annotate("above: energy budget exceeded", (rr.min(), vmax), xytext=(4, 6),
                        textcoords="offset points", ha="left", color=INK, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
        if np.isclose(optimum["C_pF"], c_val):
            ax.plot(optimum["R_ohm"], optimum["V_supply"], marker="*", markersize=16,
                    color=SERIES[1], mec=INK, mew=1, ls="none", clip_on=False)
            ax.annotate(f"proposed: {optimum['pred_delay_ps']:.0f} ps", (optimum["R_ohm"],
                        optimum["V_supply"]), xytext=(10, -14), textcoords="offset points",
                        fontsize=8, color=INK)
        ax.set_title(f"C = {c_val:g} pF", loc="left")
        ax.set_xlabel("R (ohm)")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("supply voltage V (V)")
    fig.colorbar(cf, ax=axes, label="predicted delay (ps)", format="%.0f")
    fig.suptitle("Fitted response surface (log-quadratic model), within tested bounds only",
                 x=0.01, ha="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def validation(val: pd.DataFrame, path):
    """Predicted vs observed delay on held-out points, and relative error."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    inside = (val["delay_ps"] >= val["pi_low_ps"]) & (val["delay_ps"] <= val["pi_high_ps"])
    for mask, colour, name in ((inside, SERIES[0], "inside 95 % PI"),
                               (~inside, SERIES[1], "outside 95 % PI")):
        sub = val[mask]
        ax1.errorbar(sub["delay_ps"], sub["pred_ps"],
                     yerr=[sub["pred_ps"] - sub["pi_low_ps"], sub["pi_high_ps"] - sub["pred_ps"]],
                     fmt="o", color=colour, ms=7, mec=SURFACE, elinewidth=1, capsize=0,
                     label=f"{name} ({mask.sum()})")
    lims = [val[["delay_ps", "pred_ps"]].min().min() * 0.85, val[["delay_ps", "pred_ps"]].max().max() * 1.15]
    ax1.plot(lims, lims, color=INK_2, lw=1, ls="--")
    ax1.set(xscale="log", yscale="log", xlim=lims, ylim=lims,
            xlabel="observed delay (ps, noisy simulator)", ylabel="predicted delay (ps)")
    ax1.set_title("Held-out validation points", loc="left")
    ax1.legend(frameon=False, loc="upper left")

    rel = 100 * (val["pred_ps"] / val["delay_ps"] - 1)
    ax2.axhline(0, color=INK_2, lw=1)
    ax2.scatter(val["V_supply"], rel, color=SERIES[0], s=40, edgecolor=SURFACE)
    ax2.set(xlabel="supply voltage V (V)", ylabel="prediction error (%)")
    ax2.set_title("Relative error vs supply voltage", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
