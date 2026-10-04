"""Build AAAI-column-layout figures for the WolfBench paper.

The manuscript uses Figure 2 as the only full-width system figure. The other
paper figures are intentionally single-column figures so they fit the AAAI
two-column format without dominating page flow.

Usage:
    python -m experiments.scaling_theory.paper_column_figures
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.special import expit

from experiments._common import OUTPUTS_ROOT, SCALING_THEORY_OUTPUTS_ROOT


OUT_DIR = SCALING_THEORY_OUTPUTS_ROOT / "paper_column_figures"
EXP2_DIR = SCALING_THEORY_OUTPUTS_ROOT / "exp2_society_size_scaling"
EXP12_DIR = SCALING_THEORY_OUTPUTS_ROOT / "exp12_canonical_scaling_refined"
EXP13_DIR = SCALING_THEORY_OUTPUTS_ROOT / "exp13_social_dynamics"

INK = "#24172B"
MUTED = "#6F6078"
GRID = "#E9DEEE"
RULE = "#CDBBD8"
PINK = "#D83A8B"
ROSE = "#F06BAA"
VIOLET = "#7C4DFF"
PURPLE = "#4B1D73"
GREEN = "#059669"
ORANGE = "#F59E0B"
BLUE = "#2563EB"
GRAY = "#9B91A3"

SCENARIOS = ["s1", "s2", "s3", "s4"]
SCENARIO_LABELS = {
    "s1": "S1\nsocial\npump",
    "s2": "S2\nfinfluencer",
    "s3": "S3\nspoofing",
    "s4": "S4\nwash\ntrading",
}
SCENARIO_COLORS = {"s1": PINK, "s2": VIOLET, "s3": PURPLE, "s4": ROSE}

VARIANT_ORDER = [
    "no_social",
    "baseline",
    "high_exposure",
    "high_reach",
    "hub_placement",
    "reflexive_cascade",
]
VARIANT_LABELS = {
    "no_social": "NoSocial",
    "baseline": "Base",
    "high_exposure": "HighExp.",
    "high_reach": "HighReach",
    "hub_placement": "Hub",
    "reflexive_cascade": "Reflex.",
}
VARIANT_COLORS = {
    "no_social": GRAY,
    "baseline": PINK,
    "high_exposure": ROSE,
    "high_reach": VIOLET,
    "hub_placement": PURPLE,
    "reflexive_cascade": "#C084FC",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 5.9,
            "axes.labelsize": 6.0,
            "axes.titlesize": 6.6,
            "legend.fontsize": 5.2,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, color=GRID, linewidth=0.55)


def alpha_label(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    if value >= 0.01:
        return f"{value:.3g}".replace("0.", ".")
    return f"{value:.4g}".replace("0.", ".")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required figure input: {path}")
    return pd.read_csv(path)


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def save(fig: plt.Figure, stem: str, sources: list[Path]) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=420)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "figure": stem,
        "png": str(png.relative_to(OUTPUTS_ROOT.parent)),
        "pdf": str(pdf.relative_to(OUTPUTS_ROOT.parent)),
        "sources": [str(source.relative_to(OUTPUTS_ROOT.parent)) for source in sources],
    }


def add_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    title: str,
    lines: list[str],
    face: str,
    edge: str,
    title_color: str = INK,
) -> None:
    x, y = xy
    w, h = size
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.15,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(card)
    ax.text(x + 0.035 * w, y + h - 0.18 * h, title, transform=ax.transAxes, ha="left", va="center",
            fontsize=7.0, fontweight="bold", color=title_color)
    ax.text(x + 0.04 * w, y + h - 0.38 * h, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=5.55, color=MUTED, linespacing=1.14)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = MUTED,
    lw: float = 1.15,
    rad: float = 0.0,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=lw,
        color=color,
        transform=ax.transAxes,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)


def figure1_compact_scaling_teaser() -> dict[str, Any]:
    curves = numeric(read_csv(EXP2_DIR / "collapse_rate_wilson_ci.csv"), ["n_society", "alpha", "mean"])
    thresholds = numeric(read_csv(EXP2_DIR / "alpha_critical_summary.csv"), ["n_society", "alpha_c_logistic", "logistic_slope"])

    fig = plt.figure(figsize=(3.35, 2.42))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.12], wspace=0.32)
    ax_loop = fig.add_subplot(grid[0, 0])
    ax_curve = fig.add_subplot(grid[0, 1])
    ax_loop.axis("off")

    ax_loop.text(0.0, 1.025, "A  amplification loop", transform=ax_loop.transAxes, ha="left", va="bottom",
                 fontsize=6.8, fontweight="bold", color=INK)
    ax_loop.add_patch(
        FancyBboxPatch(
            (0.05, 0.14),
            0.86,
            0.72,
            boxstyle="round,pad=0.018,rounding_size=0.050",
            linewidth=0.85,
            edgecolor="#E6D3EC",
            facecolor="#FFF8FD",
            transform=ax_loop.transAxes,
            zorder=0,
        )
    )

    # Population strip: the missing harmful-fraction axis.
    dot_x = np.linspace(0.18, 0.78, 11)
    harmful = {8, 9, 10}
    for idx, x in enumerate(dot_x):
        face = PINK if idx in harmful else "#D9D1DE"
        radius = 0.018 if idx not in harmful else 0.021
        ax_loop.add_patch(
            Circle(
                (x, 0.755),
                radius,
                transform=ax_loop.transAxes,
                facecolor=face,
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
        )
    ax_loop.plot([0.16, 0.80], [0.755, 0.755], color="#D7B8DE", lw=1.0, alpha=0.55,
                 transform=ax_loop.transAxes, zorder=1)
    ax_loop.text(0.48, 0.705, r"harmful fraction $\alpha$", transform=ax_loop.transAxes,
                 ha="center", va="center", fontsize=4.65, color=MUTED)

    # A compact loop with labels outside the nodes avoids boxed-text crowding.
    loop_nodes = {
        "exposure": (0.50, 0.63),
        "amplify": (0.72, 0.50),
        "feedback": (0.50, 0.32),
        "failure": (0.28, 0.50),
    }
    for start, end, rad in [
        ("exposure", "amplify", -0.20),
        ("amplify", "feedback", -0.20),
        ("feedback", "failure", -0.20),
        ("failure", "exposure", -0.20),
    ]:
        arrow(ax_loop, loop_nodes[start], loop_nodes[end], color=PURPLE, lw=1.25, rad=rad)

    for key, (x, y) in loop_nodes.items():
        ax_loop.add_patch(
            Circle(
                (x, y),
                0.039,
                transform=ax_loop.transAxes,
                facecolor="#FFFFFF",
                edgecolor="#D7B8DE",
                linewidth=0.9,
                zorder=4,
            )
        )
        ax_loop.add_patch(
            Circle(
                (x, y),
                0.018,
                transform=ax_loop.transAxes,
                facecolor=SCENARIO_COLORS.get("s1", PINK) if key in {"exposure", "failure"} else PURPLE,
                edgecolor="white",
                linewidth=0.45,
                zorder=5,
            )
        )

    ax_loop.add_patch(
        Circle(
            (0.50, 0.48),
            0.112,
            transform=ax_loop.transAxes,
            facecolor="#FFFFFF",
            edgecolor="#D7B8DE",
            linewidth=0.95,
            zorder=1,
        )
    )
    ax_loop.text(
        0.50,
        0.505,
        r"$\alpha=K/N$",
        transform=ax_loop.transAxes,
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=PINK,
        zorder=2,
    )
    ax_loop.text(
        0.50,
        0.440,
        "sweep",
        transform=ax_loop.transAxes,
        ha="center",
        va="center",
        fontsize=5.6,
        color=MUTED,
        zorder=2,
    )
    ax_loop.text(0.50, 0.215, "exposure -> amplify",
                 transform=ax_loop.transAxes, ha="center", va="center",
                 fontsize=4.15, color=MUTED)
    ax_loop.text(0.50, 0.180, "feedback -> failure",
                 transform=ax_loop.transAxes, ha="center", va="center",
                 fontsize=4.15, color=MUTED)

    for n_value, color in [(1000, VIOLET), (5000, PURPLE)]:
        rows = curves[curves["n_society"] == n_value].sort_values("alpha")
        trow = thresholds[thresholds["n_society"] == n_value]
        if rows.empty or trow.empty:
            continue
        ac = float(trow.iloc[0]["alpha_c_logistic"])
        slope = float(trow.iloc[0]["logistic_slope"])
        xs = np.linspace(float(rows["alpha"].min()), float(rows["alpha"].max()), 200)
        ax_curve.plot(xs, expit(slope * (xs - ac)), color=color, lw=1.35, label=f"N={n_value}")
        ax_curve.plot(rows["alpha"], rows["mean"], "o", color=color, ms=2.3, alpha=0.75)
        ax_curve.axvline(ac, color=color, lw=0.75, ls=":")
    ax_curve.axhline(0.5, color=MUTED, lw=0.65, ls="--")
    ax_curve.text(0.0, 1.025, "B  critical fraction", transform=ax_curve.transAxes,
                  ha="left", va="bottom", fontsize=6.8, fontweight="bold", color=INK)
    ax_curve.set_xlabel("harmful fraction alpha")
    ax_curve.set_ylabel("P(failure)")
    ax_curve.set_xlim(-0.001, 0.041)
    ax_curve.set_ylim(-0.03, 1.03)
    ax_curve.set_xticks([0, 0.02, 0.04])
    ax_curve.set_xticklabels(["0", ".02", ".04"])
    ax_curve.legend(frameon=False, loc="lower right", handlelength=1.5)
    clean_axis(ax_curve)
    return save(fig, "figure1_compact_scaling_teaser", [EXP2_DIR / "collapse_rate_wilson_ci.csv", EXP2_DIR / "alpha_critical_summary.csv"])


def figure2_measurement_instrument() -> dict[str, Any]:
    curves = numeric(read_csv(EXP2_DIR / "collapse_rate_wilson_ci.csv"), ["n_society", "alpha", "mean"])
    thresholds = numeric(
        read_csv(EXP2_DIR / "alpha_critical_summary.csv"),
        ["n_society", "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high", "transition_width_10_90"],
    ).sort_values("n_society")

    fig = plt.figure(figsize=(7.25, 3.45))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.02, 0.97, "Harm scales through a missing population axis",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(0.02, 0.895, "WolfBench turns harmful-agent fraction into a measured critical-regime estimand.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6, color=MUTED)

    # Panel A: the missing axis.
    ax.text(0.035, 0.785, "A  population axis", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.3, fontweight="bold", color=INK)
    ax.add_patch(FancyBboxPatch((0.035, 0.55), 0.23, 0.18, boxstyle="round,pad=0.014,rounding_size=0.028",
                                linewidth=1.0, edgecolor="#D8B4DD", facecolor="#FFF7FC", transform=ax.transAxes))
    ax.text(0.055, 0.68, r"$\alpha=K/N$", transform=ax.transAxes, ha="left", va="center",
            fontsize=10.0, fontweight="bold", color=PINK)
    ax.text(0.055, 0.61, "small harmful minorities\ncan move a shared system",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color=MUTED, linespacing=1.15)
    ax.plot([0.055, 0.24], [0.48, 0.48], color=RULE, lw=2.0, transform=ax.transAxes, solid_capstyle="round")
    for i, frac in enumerate(np.linspace(0, 1, 8)):
        xdot = 0.055 + 0.185 * frac
        color = PINK if i >= 2 else GREEN
        ax.add_patch(Circle((xdot, 0.48), 0.009 + 0.002 * i, transform=ax.transAxes, facecolor=color, edgecolor="white", lw=0.45))
    ax.text(0.055, 0.42, "benign", transform=ax.transAxes, ha="left", va="top", fontsize=5.7, color=GREEN)
    ax.text(0.24, 0.42, "harmful", transform=ax.transAxes, ha="right", va="top", fontsize=5.7, color=PINK)

    arrow(ax, (0.28, 0.61), (0.335, 0.61), color=MUTED, lw=1.2)

    # Panel B: actual measured critical transition surface.
    ax.text(0.365, 0.785, "B  transition surface", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.3, fontweight="bold", color=INK)
    heat_ax = fig.add_axes([0.392, 0.370, 0.220, 0.315])
    n_values = sorted(curves["n_society"].dropna().unique())
    alpha_values = sorted(curves["alpha"].dropna().unique())
    heat = np.full((len(n_values), len(alpha_values)), np.nan)
    for i, n_society in enumerate(n_values):
        for j, alpha in enumerate(alpha_values):
            row = curves[(curves["n_society"] == n_society) & np.isclose(curves["alpha"], alpha)]
            if not row.empty:
                heat[i, j] = float(row.iloc[0]["mean"])
    image = heat_ax.imshow(heat, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    heat_ax.set_xticks([0, 4, 8, 12])
    heat_ax.set_xticklabels(["0", ".01", ".02", ".04"])
    heat_ax.set_yticks([0, 6, 10])
    heat_ax.set_yticklabels(["100", "1k", "5k"])
    heat_ax.set_xlabel("alpha", labelpad=0)
    heat_ax.set_ylabel("")
    heat_ax.tick_params(length=2, pad=1)
    for spine in heat_ax.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(0.65)
    cax = fig.add_axes([0.618, 0.410, 0.009, 0.245])
    cb = fig.colorbar(image, cax=cax)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=5.0, length=2, pad=1)
    arrow(ax, (0.645, 0.61), (0.700, 0.61), color=MUTED, lw=1.2)

    # Panel C: estimands and implication, with the Figure 3 result embedded.
    ax.text(0.705, 0.785, "C  safety targets", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.3, fontweight="bold", color=INK)
    ax.add_patch(FancyBboxPatch((0.705, 0.565), 0.255, 0.165, boxstyle="round,pad=0.014,rounding_size=0.026",
                                linewidth=1.0, edgecolor="#D8B4DD", facecolor="#FFF7FC", transform=ax.transAxes))
    first = thresholds.iloc[0]
    last = thresholds.iloc[-1]
    ax.text(0.725, 0.685, r"$\alpha_c(N)$: " + f"{first['alpha_c_logistic']:.4f} -> {last['alpha_c_logistic']:.4f}",
            transform=ax.transAxes, ha="left", va="center", fontsize=6.4, color=INK, fontweight="bold")
    ax.text(0.725, 0.635, r"$W_N$: " + f"{first['transition_width_10_90']:.4f} -> {last['transition_width_10_90']:.4f}",
            transform=ax.transAxes, ha="left", va="center", fontsize=6.4, color=INK)
    ax.text(0.725, 0.585, "defense shifts boundary right", transform=ax.transAxes,
            ha="left", va="center", fontsize=5.6, color=MUTED)

    scale_ax = fig.add_axes([0.735, 0.345, 0.215, 0.155])
    xs = thresholds["n_society"].to_numpy(dtype=float)
    ys = thresholds["alpha_c_logistic"].to_numpy(dtype=float)
    scale_ax.plot(xs, ys, "o-", color=PINK, ms=2.2, lw=1.0)
    scale_ax.set_xscale("log")
    scale_ax.set_yscale("log")
    scale_ax.set_xticks([100, 1000, 5000])
    scale_ax.set_xticklabels(["100", "1k", "5k"])
    scale_ax.set_yticks([0.01, 0.02])
    scale_ax.set_yticklabels([".01", ".02"])
    scale_ax.set_title("", pad=0)
    scale_ax.grid(True, color=GRID, lw=0.45)
    for spine in scale_ax.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(0.55)

    # Keep operational details as a compact footer, not the main visual claim.
    ax.add_patch(FancyBboxPatch((0.07, 0.085), 0.86, 0.10, boxstyle="round,pad=0.01,rounding_size=0.020",
                                linewidth=0.8, edgecolor="#CBD5E1", facecolor="#F8FAFC", transform=ax.transAxes))
    footer = "instantiate society  ->  run 30-day public loop  ->  align event  ->  fit response curve"
    ax.text(0.50, 0.136, footer, transform=ax.transAxes, ha="center", va="center", fontsize=6.2, color=MUTED)

    return save(fig, "figure2_measurement_instrument", [EXP2_DIR / "collapse_rate_wilson_ci.csv", EXP2_DIR / "alpha_critical_summary.csv"])


def figure3_s1_scaling_compact() -> dict[str, Any]:
    curves = numeric(read_csv(EXP2_DIR / "collapse_rate_wilson_ci.csv"), ["n_society", "alpha", "mean", "ci_low", "ci_high"])
    thresholds = numeric(
        read_csv(EXP2_DIR / "alpha_critical_summary.csv"),
        ["n_society", "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high", "logistic_slope", "transition_width_10_90"],
    ).sort_values("n_society")

    fig = plt.figure(figsize=(3.35, 3.45))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.28, 0.82], hspace=0.54, wspace=0.35)
    ax_curve = fig.add_subplot(grid[0, :])
    ax_alpha = fig.add_subplot(grid[1, 0])
    ax_width = fig.add_subplot(grid[1, 1])

    for n_value, color, line_width in [(100, "#F4A6CF", 1.0), (1000, VIOLET, 1.25), (5000, PURPLE, 1.45)]:
        rows = curves[curves["n_society"] == n_value].sort_values("alpha")
        trow = thresholds[thresholds["n_society"] == n_value]
        if rows.empty or trow.empty:
            continue
        ax_curve.errorbar(
            rows["alpha"],
            rows["mean"],
            yerr=[
                np.maximum(0, rows["mean"].to_numpy() - rows["ci_low"].to_numpy()),
                np.maximum(0, rows["ci_high"].to_numpy() - rows["mean"].to_numpy()),
            ],
            fmt="o",
            color=color,
            ms=2.2,
            lw=0.65,
            capsize=1.2,
            alpha=0.78,
            label=f"N={n_value}",
        )
        ac = float(trow.iloc[0]["alpha_c_logistic"])
        slope = float(trow.iloc[0]["logistic_slope"])
        xs = np.linspace(float(rows["alpha"].min()), float(rows["alpha"].max()), 200)
        ax_curve.plot(xs, expit(slope * (xs - ac)), color=color, lw=line_width)
        ax_curve.axvline(ac, color=color, ls=":", lw=0.7)
    ax_curve.axhline(0.5, color=MUTED, ls="--", lw=0.65)
    ax_curve.axvspan(0.009, 0.020, color="#FCE7F3", alpha=0.38, zorder=0)
    ax_curve.annotate(
        "small alpha change\nlarge collapse jump",
        xy=(0.014, 0.56),
        xytext=(0.023, 0.28),
        fontsize=5.9,
        color=INK,
        arrowprops=dict(arrowstyle="->", lw=0.8, color=PINK),
    )
    ax_curve.text(0.010, 0.96, "near-critical band", fontsize=5.6, color=PINK, ha="left", va="top")
    ax_curve.set_title("A  a narrow harmful-fraction band flips the society", loc="left")
    ax_curve.set_xlabel("harmful-agent fraction alpha")
    ax_curve.set_ylabel("P(collapse)")
    ax_curve.set_xlim(-0.001, 0.041)
    ax_curve.set_ylim(-0.04, 1.04)
    ax_curve.set_xticks([0, 0.01, 0.02, 0.04])
    ax_curve.set_xticklabels(["0", ".01", ".02", ".04"])
    ax_curve.legend(frameon=False, ncol=3, loc="lower right", handlelength=1.3, columnspacing=0.8)
    clean_axis(ax_curve)

    x = thresholds["n_society"].to_numpy(dtype=float)
    ac = thresholds["alpha_c_logistic"].to_numpy(dtype=float)
    low = thresholds["alpha_c_ci_low"].to_numpy(dtype=float)
    high = thresholds["alpha_c_ci_high"].to_numpy(dtype=float)
    width = thresholds["transition_width_10_90"].to_numpy(dtype=float)
    ax_alpha.errorbar(
        x,
        ac,
        yerr=[np.maximum(0, ac - low), np.maximum(0, high - ac)],
        fmt="o-",
        color=PINK,
        ecolor=ROSE,
        lw=1.1,
        ms=2.7,
        capsize=1.5,
        label=r"$\alpha_c$",
    )
    ax_alpha.set_xscale("log")
    ax_alpha.set_yscale("log")
    ax_alpha.set_xlabel("society size N")
    ax_alpha.set_ylabel(r"$\alpha_c(N)$")
    ax_alpha.set_title(r"B  $\alpha_c$ moves down", loc="left")
    ax_alpha.set_xticks([100, 1000, 5000])
    ax_alpha.set_xticklabels(["100", "1k", "5k"])
    clean_axis(ax_alpha)
    ax_alpha.text(0.08, 0.08, ".0144 -> .0094", transform=ax_alpha.transAxes, ha="left", va="bottom",
                  fontsize=5.7, color=PINK, fontweight="bold")

    ax_width.plot(x, width, "s--", color=VIOLET, lw=1.0, ms=2.5)
    ax_width.set_xscale("log")
    ax_width.set_yscale("log")
    ax_width.set_xlabel("society size N")
    ax_width.set_ylabel(r"$W_N$")
    ax_width.set_title(r"C  transition narrows", loc="left")
    ax_width.set_xticks([100, 1000, 5000])
    ax_width.set_xticklabels(["100", "1k", "5k"])
    ax_width.spines["top"].set_visible(False)
    clean_axis(ax_width)
    ax_width.text(0.08, 0.08, ".0366 -> .0183", transform=ax_width.transAxes, ha="left", va="bottom",
                  fontsize=5.7, color=VIOLET, fontweight="bold")
    return save(fig, "figure3_s1_finite_size_compact", [EXP2_DIR / "collapse_rate_wilson_ci.csv", EXP2_DIR / "alpha_critical_summary.csv"])


def figure4_mechanism_compact() -> dict[str, Any]:
    thresholds = numeric(
        read_csv(EXP12_DIR / "alpha_c_by_scenario_n.csv"),
        ["n_society", "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high", "transition_width_10_90"],
    )
    summary = read_csv(EXP12_DIR / "scenario_law_summary.csv")
    n_focus = 1000
    rows = thresholds[thresholds["n_society"] == n_focus].copy().set_index("scenario").reindex(SCENARIOS).reset_index()

    fig = plt.figure(figsize=(3.35, 2.95))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.3, 0.82], hspace=0.68)
    ax = fig.add_subplot(grid[0, 0])
    ax_tbl = fig.add_subplot(grid[1, 0])

    ypos = np.arange(len(rows))[::-1]
    for i, row in rows.iterrows():
        scenario = str(row["scenario"])
        color = SCENARIO_COLORS[scenario]
        ac = float(row["alpha_c_logistic"])
        low = float(row["alpha_c_ci_low"]) if pd.notna(row["alpha_c_ci_low"]) else ac
        high = float(row["alpha_c_ci_high"]) if pd.notna(row["alpha_c_ci_high"]) else ac
        detection_floor = 1.0 / n_focus
        censored_first_count = scenario == "s2" and ac < detection_floor
        if censored_first_count:
            ax.plot(
                detection_floor,
                ypos[i],
                marker="<",
                color=color,
                ms=5.0,
                linestyle="None",
            )
            ax.text(
                detection_floor * 1.18,
                ypos[i],
                r"$<1/N$",
                va="center",
                ha="left",
                fontsize=6.2,
                color=INK,
            )
        else:
            ax.errorbar(
                ac,
                ypos[i],
                xerr=[[max(0, ac - low)], [max(0, high - ac)]],
                fmt="o",
                color=color,
                lw=1.0,
                ms=4.0,
                capsize=2.0,
            )
            ax.text(
                high * 1.18 if high > 0 else ac + 1e-4,
                ypos[i],
                f"{ac:.3g}",
                va="center",
                ha="left",
                fontsize=6.2,
                color=INK,
            )
    ax.set_xscale("log")
    ax.set_xlim(1.7e-4, 9e-2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.set_xlabel(r"critical harmful fraction $\alpha_c$ at $N=1000$")
    ax.set_title("A  mechanism-specific critical regimes", loc="left")
    ax.text(
        0.02,
        -0.31,
        r"S2 marker is censored at the first feasible nonzero count $(1/N=.001)$.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.05,
        color=MUTED,
    )
    clean_axis(ax)

    ax_tbl.axis("off")
    event_short = {
        "generic_collapse": "generic",
        "spoof_liquidity_failure": "spoof/liq.",
        "fake_liquidity_failure": "fake-liq.",
    }
    claim_short = {"strong": "strong", "partial": "partial", "mechanism-strong": "mech.-strong"}
    cell_rows = []
    for scenario in SCENARIOS:
        srow = summary[summary["scenario"] == scenario]
        if srow.empty:
            cell_rows.append([scenario.upper(), "", ""])
        else:
            row = srow.iloc[0]
            cell_rows.append([
                scenario.upper(),
                event_short.get(str(row["primary_metric"]), str(row["primary_metric"])),
                claim_short.get(str(row["evidence_grade"]), str(row["evidence_grade"])),
            ])
    table = ax_tbl.table(
        cellText=cell_rows,
        colLabels=["B/S", "event", "grade"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.18, 0.46, 0.36],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.45)
    table.scale(1.0, 1.08)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_facecolor("#F3D8F3")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("#FFF7FC")
            cell.set_text_props(color=INK)
    return save(
        fig,
        "figure4_mechanism_heterogeneity_compact",
        [EXP12_DIR / "alpha_c_by_scenario_n.csv", EXP12_DIR / "scenario_law_summary.csv"],
    )


def figure5_social_dynamics_compact() -> dict[str, Any]:
    table = numeric(
        read_csv(EXP13_DIR / "social_dynamic_table.csv"),
        [
            "alpha_c",
            "alpha_c_ci_low",
            "alpha_c_ci_high",
            "delta_alpha_c",
            "cascade_peak_at_trajectory_alpha",
            "cascade_velocity_at_trajectory_alpha",
        ],
    )
    table["variant"] = table["variant"].astype(str)
    table = table.set_index("variant").reindex(VARIANT_ORDER).reset_index()

    fig = plt.figure(figsize=(3.35, 3.35))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.18, 0.9], hspace=0.72)
    ax_alpha = fig.add_subplot(grid[0, 0])
    ax_cascade = fig.add_subplot(grid[1, 0])

    y = np.arange(len(table))[::-1]
    for i, row in table.iterrows():
        key = str(row["variant"])
        color = VARIANT_COLORS.get(key, GRAY)
        if key == "no_social":
            ax_alpha.scatter([0.04], [y[i]], marker=">", s=32, color=color, zorder=3)
            ax_alpha.text(0.0408, y[i], "> .040", va="center", ha="left", fontsize=6.2, color=MUTED)
            continue
        ac = float(row["alpha_c"])
        low = float(row["alpha_c_ci_low"]) if pd.notna(row["alpha_c_ci_low"]) else ac
        high = float(row["alpha_c_ci_high"]) if pd.notna(row["alpha_c_ci_high"]) else ac
        ax_alpha.errorbar(ac, y[i], xerr=[[max(0, ac - low)], [max(0, high - ac)]],
                          fmt="o", color=color, ms=3.5, capsize=2.0, lw=0.9)
    baseline = table[table["variant"] == "baseline"]
    if not baseline.empty:
        ax_alpha.axvline(float(baseline.iloc[0]["alpha_c"]), color=PINK, ls="--", lw=0.8)
    ax_alpha.set_yticks(y)
    ax_alpha.set_yticklabels([VARIANT_LABELS.get(str(v), str(v)) for v in table["variant"]])
    ax_alpha.set_xlim(0, 0.043)
    ax_alpha.set_xlabel(r"critical fraction $\alpha_c$")
    ax_alpha.set_title("A  social regime shifts the threshold", loc="left")
    clean_axis(ax_alpha)

    cascade = table["cascade_peak_at_trajectory_alpha"].to_numpy(dtype=float)
    colors = [VARIANT_COLORS.get(str(v), GRAY) for v in table["variant"]]
    ax_cascade.bar(np.arange(len(table)), cascade, color=colors, edgecolor=INK, linewidth=0.45)
    ax_cascade.axhline(0.4, color=MUTED, ls="--", lw=0.75)
    ax_cascade.set_xticks(np.arange(len(table)))
    ax_cascade.set_xticklabels([VARIANT_LABELS.get(str(v), str(v)) for v in table["variant"]], rotation=28, ha="right")
    ax_cascade.set_ylim(0, 1.05)
    ax_cascade.set_ylabel("max cascade")
    ax_cascade.set_title(r"B  near-critical cascade at $\alpha=.0125$", loc="left")
    clean_axis(ax_cascade)
    return save(fig, "figure5_social_dynamics_compact", [EXP13_DIR / "social_dynamic_table.csv"])


def write_manifest(entries: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figure_manifest.json").write_text(json.dumps({"figures": entries}, indent=2) + "\n")
    lines = ["# AAAI Column Layout Figures", "", "| Figure | PNG | PDF | Sources |", "|---|---|---|---|"]
    for entry in entries:
        lines.append(
            f"| {entry['figure']} | `{entry['png']}` | `{entry['pdf']}` | "
            + ", ".join(f"`{source}`" for source in entry["sources"])
            + " |"
        )
    (OUT_DIR / "figure_manifest.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    configure_style()
    entries = [
        figure1_compact_scaling_teaser(),
        figure2_measurement_instrument(),
        figure3_s1_scaling_compact(),
        figure4_mechanism_compact(),
        figure5_social_dynamics_compact(),
    ]
    write_manifest(entries)
    for entry in entries:
        print(entry["png"])
        print(entry["pdf"])


if __name__ == "__main__":
    main()
