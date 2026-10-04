"""Story-first paper figures for WolfBench.

These figures are intentionally schematic: Figure 1 is a compact single-column
first-page teaser that combines the core claim with the strongest current
evidence, while Figure 2 explains the benchmark loop with visual agents and
metrics.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wolfbench-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd

from .io_utils import OUTPUTS, ROOT, ensure_dir


INK = "#24112F"
MUTED = "#6B5873"
GRID = "#E9D7F0"
PINK = "#DB2777"
ROSE = "#F472B6"
MAGENTA = "#C026D3"
PURPLE = "#7E22CE"
DEEP = "#4C1D95"
VIOLET = "#A855F7"
TEAL = "#14B8A6"
MINT = "#8EE6D6"
ORANGE = "#F97316"
BLUE = "#2563EB"
SAFE = "#DFF8F2"
CRITICAL = "#FFF2B8"
COLLAPSED = "#FDE2E2"
PANEL = "#FFF8FC"
PALE = "#F8ECFF"


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig_dir = ensure_dir(ROOT / "figures")
    paper_dir = ensure_dir(ROOT.parent / "AuthorKit27" / "Figures")
    for directory in (fig_dir, paper_dir):
        fig.savefig(directory / f"{stem}.pdf")
        fig.savefig(directory / f"{stem}.png", dpi=360)
    plt.close(fig)


def robot(ax: plt.Axes, x: float, y: float, scale: float = 1.0, color: str = TEAL,
          edge: str = "white", llm: bool = False, harmful: bool = False) -> None:
    w, h = 0.11 * scale, 0.085 * scale
    body = patches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={0.018 * scale}",
        facecolor=color,
        edgecolor=edge,
        linewidth=0.9,
        zorder=4,
    )
    ax.add_patch(body)
    ax.add_line(plt.Line2D([x - 0.030 * scale, x - 0.030 * scale], [y - 0.005 * scale, y + 0.014 * scale], color="white", linewidth=0.8, zorder=5))
    ax.add_line(plt.Line2D([x + 0.030 * scale, x + 0.030 * scale], [y - 0.005 * scale, y + 0.014 * scale], color="white", linewidth=0.8, zorder=5))
    ax.add_patch(patches.Circle((x - 0.030 * scale, y + 0.018 * scale), 0.006 * scale, facecolor="white", edgecolor="none", zorder=6))
    ax.add_patch(patches.Circle((x + 0.030 * scale, y + 0.018 * scale), 0.006 * scale, facecolor="white", edgecolor="none", zorder=6))
    ax.add_line(plt.Line2D([x - 0.020 * scale, x + 0.020 * scale], [y - 0.020 * scale, y - 0.020 * scale], color="white", linewidth=0.7, zorder=5))
    ax.add_line(plt.Line2D([x, x], [y + h / 2, y + h / 2 + 0.026 * scale], color=color, linewidth=0.9, zorder=3))
    ax.add_patch(patches.Circle((x, y + h / 2 + 0.030 * scale), 0.008 * scale, facecolor=color, edgecolor=edge, linewidth=0.5, zorder=4))
    if llm:
        ax.add_patch(patches.Circle((x, y), 0.066 * scale, facecolor="none", edgecolor=DEEP, linewidth=1.0, alpha=0.85, zorder=3))
    if harmful:
        ax.add_patch(patches.RegularPolygon((x + 0.045 * scale, y + 0.033 * scale), numVertices=3, radius=0.020 * scale, orientation=math.pi, facecolor=PINK, edgecolor="white", linewidth=0.5, zorder=7))


def cute_agent(ax: plt.Axes, x: float, y: float, scale: float = 1.0, color: str = TEAL,
               harmful: bool = False, llm: bool = False) -> None:
    """Small vector mascot used in the first-page teaser."""
    ax.add_patch(
        patches.Ellipse(
            (x, y - 0.060 * scale),
            0.118 * scale,
            0.026 * scale,
            facecolor="#EBD7F4",
            edgecolor="none",
            alpha=0.62,
            zorder=1,
        )
    )
    if llm:
        ax.add_patch(
            patches.Circle(
                (x, y + 0.006 * scale),
                0.071 * scale,
                facecolor="none",
                edgecolor=DEEP,
                linewidth=1.0,
                alpha=0.75,
                zorder=2,
            )
        )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x - 0.044 * scale, y - 0.050 * scale),
            0.088 * scale,
            0.090 * scale,
            boxstyle=f"round,pad=0.004,rounding_size={0.030 * scale}",
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x - 0.029 * scale, y - 0.018 * scale),
            0.058 * scale,
            0.039 * scale,
            boxstyle=f"round,pad=0.002,rounding_size={0.011 * scale}",
            facecolor="white",
            edgecolor="none",
            alpha=0.95,
            zorder=5,
        )
    )
    ax.add_patch(patches.Circle((x - 0.014 * scale, y + 0.002 * scale), 0.0048 * scale, facecolor=INK, edgecolor="none", zorder=6))
    ax.add_patch(patches.Circle((x + 0.014 * scale, y + 0.002 * scale), 0.0048 * scale, facecolor=INK, edgecolor="none", zorder=6))
    ax.plot([x - 0.013 * scale, x, x + 0.013 * scale], [y - 0.014 * scale, y - 0.020 * scale, y - 0.014 * scale], color=INK, linewidth=0.65, zorder=6)
    ax.plot([x, x], [y + 0.043 * scale, y + 0.063 * scale], color=color, linewidth=0.9, zorder=3)
    ax.add_patch(patches.Circle((x, y + 0.069 * scale), 0.007 * scale, facecolor=color, edgecolor="white", linewidth=0.4, zorder=4))
    if harmful:
        ax.add_patch(
            patches.RegularPolygon(
                (x + 0.042 * scale, y + 0.032 * scale),
                numVertices=3,
                radius=0.021 * scale,
                orientation=math.pi,
                facecolor=PINK,
                edgecolor="white",
                linewidth=0.55,
                zorder=8,
            )
        )
        ax.text(x + 0.042 * scale, y + 0.030 * scale, "!", fontsize=4.8 * scale, color="white", fontweight="bold", ha="center", va="center", zorder=9)


def curved_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
                 color: str, rad: float = 0.2, lw: float = 1.5) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=2,
    )


def draw_single_column_teaser() -> None:
    """Single-column first-page teaser for the two scaling signatures."""
    setup()
    fig = plt.figure(figsize=(3.35, 1.55))
    ax_curve = fig.add_axes([0.120, 0.245, 0.510, 0.660])
    ax_shift = fig.add_axes([0.725, 0.245, 0.245, 0.660])

    rates_path = OUTPUTS / "real_e1_refined_s1" / "e1_wilson_rates.csv"
    alpha_path = OUTPUTS / "real_e1_refined_s1" / "e1_alpha_c_summary.csv"
    rates = pd.read_csv(rates_path)
    alpha_summary = pd.read_csv(alpha_path)

    from scipy.optimize import curve_fit
    from scipy.special import expit

    def logistic(x: np.ndarray, alpha_c: float, slope: float) -> np.ndarray:
        return expit(slope * (x - alpha_c))

    curve = rates[rates["n_society"].eq(1000)].sort_values("alpha").copy()
    x = curve["alpha"].to_numpy(float)
    y = curve["prob"].to_numpy(float)
    y_fit = np.clip(y, 1e-4, 1 - 1e-4)
    alpha_c0 = float(alpha_summary.loc[alpha_summary["n_society"].eq(1000), "alpha_c"].iloc[0])
    try:
        params, _ = curve_fit(
            logistic,
            x,
            y_fit,
            p0=[alpha_c0, 900.0],
            bounds=([0.0, 10.0], [0.05, 6000.0]),
            maxfev=10000,
        )
        xx = np.linspace(0.0, 0.016, 280)
        yy = logistic(xx, float(params[0]), float(params[1]))
    except Exception:
        xx = np.linspace(float(x.min()), float(x.max()), 280)
        yy = np.interp(xx, x, y)

    ax_curve.set_facecolor("#FFF9FD")
    ax_curve.fill_between([0.0, alpha_c0], 0, 1, color=SAFE, alpha=0.30, linewidth=0)
    ax_curve.fill_between([alpha_c0, 0.016], 0, 1, color="#FCE7F3", alpha=0.26, linewidth=0)
    ax_curve.plot(xx, yy, color=PURPLE, linewidth=2.1, solid_capstyle="round")
    yerr = np.vstack([
        np.maximum(0.0, y - curve["wilson_lo"].to_numpy(float)),
        np.maximum(0.0, curve["wilson_hi"].to_numpy(float) - y),
    ])
    ax_curve.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color=PINK,
        ecolor="#E9B5F2",
        elinewidth=0.75,
        capsize=0,
        markersize=3.0,
        markeredgecolor="white",
        markeredgewidth=0.55,
        zorder=3,
    )
    ax_curve.axhline(0.5, color=MUTED, linewidth=0.75, linestyle=(0, (3, 2)))
    ax_curve.axvline(alpha_c0, color=INK, linewidth=0.75, linestyle=(0, (2, 2)), alpha=0.75)
    ax_curve.text(0.0002, 1.035, "A", fontsize=7.2, fontweight="bold", color=INK, ha="left", va="bottom")
    ax_curve.text(alpha_c0 + 0.00035, 0.08, r"$\alpha_c$", fontsize=5.9, color=INK, ha="left", va="center")
    ax_curve.set_xlim(0.0, 0.016)
    ax_curve.set_ylim(-0.03, 1.05)
    ax_curve.set_xticks([0.0, 0.004, 0.008, 0.012, 0.016])
    ax_curve.set_xticklabels(["0", ".004", ".008", ".012", ".016"])
    ax_curve.set_yticks([0.0, 0.5, 1.0])
    ax_curve.set_xlabel(r"harmful fraction $\alpha$", fontsize=6.2, labelpad=3.2)
    ax_curve.set_ylabel(r"$P(C=1)$", fontsize=6.2, labelpad=1.0)
    ax_curve.grid(True, color="white", linewidth=0.8)
    ax_curve.tick_params(labelsize=5.8, colors=INK, length=2.0, pad=1.0)
    ax_curve.spines[["top", "right"]].set_visible(False)
    ax_curve.spines[["left", "bottom"]].set_color("#BFA3CE")

    main = alpha_summary[
        alpha_summary["n_society"].isin([100, 200, 300, 500, 1000, 2000])
    ].sort_values("n_society")
    ns = main["n_society"].to_numpy(float)
    ac = main["alpha_c"].to_numpy(float)
    lo = main["alpha_c_lo"].to_numpy(float)
    hi = main["alpha_c_hi"].to_numpy(float)
    ax_shift.set_facecolor("#FFF9FD")
    ax_shift.fill_between(ns, lo, hi, color="#E9D5FF", alpha=0.74, linewidth=0)
    ax_shift.plot(ns, ac, color=INK, linewidth=1.7, marker="o", markersize=3.2, markerfacecolor=PURPLE, markeredgecolor="white", markeredgewidth=0.55)
    ax_shift.set_xscale("log")
    ax_shift.set_xlim(82, 2450)
    ax_shift.set_ylim(0.0, 0.060)
    ax_shift.set_xticks([100, 300, 1000, 2000])
    ax_shift.set_xticklabels(["100", "300", "1k", "2k"])
    ax_shift.set_yticks([0.0, 0.03, 0.06])
    ax_shift.set_yticklabels(["0", ".03", ".06"])
    ax_shift.text(88, 0.0622, "B", fontsize=7.2, fontweight="bold", color=INK, ha="left", va="bottom")
    ax_shift.set_xlabel("society size N", fontsize=6.2, labelpad=3.2)
    ax_shift.set_ylabel(r"estimated $\alpha_c$", fontsize=6.2, labelpad=0.5)
    ax_shift.grid(True, color="white", linewidth=0.8)
    ax_shift.tick_params(labelsize=5.8, colors=INK, length=2.0, pad=1.0)
    ax_shift.spines[["top", "right"]].set_visible(False)
    ax_shift.spines[["left", "bottom"]].set_color("#BFA3CE")
    save(fig, "figure1_killer_agent_society")


def draw_story_figure() -> None:
    setup()
    fig, ax = plt.subplots(figsize=(7.25, 3.18))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.95, "WolfBench as a measurement instrument", fontsize=11.4, fontweight="bold", color=INK, ha="left")
    ax.text(0.02, 0.89, "Sweeps over society size, harmful fraction, mechanism, seed, LLM quota, and defense produce critical-regime estimands.", fontsize=7.6, color=MUTED, ha="left")

    modules = [
        (0.025, 0.16, 0.205, 0.64, "1  Inputs"),
        (0.265, 0.13, 0.285, 0.70, "2  Public loop"),
        (0.585, 0.16, 0.175, 0.64, "3  Events"),
        (0.795, 0.16, 0.180, 0.64, "4  Outputs"),
    ]
    for x, y, w, h, title in modules:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.014,rounding_size=0.028", facecolor="#FFF9FD", edgecolor="#D8B4E2", linewidth=0.9))
        ax.text(x + 0.014, y + h - 0.042, title, fontsize=7.9, fontweight="bold", color=INK, ha="left", va="center")
    for i in range(3):
        x0 = modules[i][0] + modules[i][2]
        x1 = modules[i + 1][0]
        ax.annotate("", xy=(x1 - 0.008, 0.48), xytext=(x0 + 0.008, 0.48), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))

    # Module 1: inputs and population composition.
    x, y, w, h, _ = modules[0]
    graph_pts = [(x + 0.045, y + 0.42), (x + 0.095, y + 0.49), (x + 0.150, y + 0.43), (x + 0.070, y + 0.32), (x + 0.140, y + 0.31)]
    for a, b in [(0, 1), (1, 2), (0, 3), (3, 4), (2, 4)]:
        ax.plot([graph_pts[a][0], graph_pts[b][0]], [graph_pts[a][1], graph_pts[b][1]], color="#D8B4E2", linewidth=0.75)
    for i, (px, py) in enumerate(graph_pts):
        color = PINK if i in (2,) else TEAL
        ax.add_patch(patches.Circle((px, py), 0.015, facecolor=color, edgecolor="white", linewidth=0.6, zorder=3))
    ax.add_patch(patches.Circle((graph_pts[1][0], graph_pts[1][1]), 0.024, facecolor="none", edgecolor=BLUE, linewidth=1.0))
    input_lines = [r"$N,\ \alpha=K/N,\ seed$", r"$m\in\{S1,S2,S3,S4\}$", "LLM-control quota", r"optional defense $\pi$"]
    for j, line in enumerate(input_lines):
        ax.text(x + 0.025, y + 0.215 - 0.045 * j, line, fontsize=6.4, color=INK if j == 0 else MUTED, ha="left")
    ax.text(x + 0.030, y + 0.028, "ordinary", fontsize=5.2, color=TEAL, ha="left")
    ax.text(x + 0.095, y + 0.028, "harmful", fontsize=5.2, color=PINK, ha="left")
    ax.text(x + 0.155, y + 0.028, "LLM", fontsize=5.2, color=BLUE, ha="left")

    # Module 2: public loop with defense and hidden state.
    x, y, w, h, _ = modules[1]
    loop_nodes = [
        (x + 0.143, y + 0.465, "public\nsummary", BLUE),
        (x + 0.068, y + 0.340, "agents\nact", TEAL),
        (x + 0.143, y + 0.215, "social\npropagation", PINK),
        (x + 0.218, y + 0.340, "market\nupdate", ORANGE),
    ]
    for px, py, label, color in loop_nodes:
        ax.add_patch(patches.FancyBboxPatch((px - 0.045, py - 0.035), 0.090, 0.070, boxstyle="round,pad=0.006,rounding_size=0.018", facecolor="white", edgecolor=color, linewidth=1.0))
        ax.text(px, py, label, fontsize=5.9, color=INK, ha="center", va="center")
    curved_arrow(ax, (x + 0.105, y + 0.440), (x + 0.075, y + 0.375), TEAL, rad=0.15, lw=0.9)
    curved_arrow(ax, (x + 0.080, y + 0.300), (x + 0.112, y + 0.235), PINK, rad=0.15, lw=0.9)
    curved_arrow(ax, (x + 0.175, y + 0.235), (x + 0.212, y + 0.300), ORANGE, rad=0.15, lw=0.9)
    curved_arrow(ax, (x + 0.220, y + 0.375), (x + 0.180, y + 0.440), BLUE, rad=0.15, lw=0.9)
    ax.add_patch(patches.FancyBboxPatch((x + 0.020, y + 0.565), 0.116, 0.060, boxstyle="round,pad=0.006,rounding_size=0.014", facecolor="#EAF2FF", edgecolor=BLUE, linewidth=0.9))
    ax.text(x + 0.078, y + 0.595, "WolfGuard", fontsize=6.2, color=BLUE, fontweight="bold", ha="center", va="center")
    ax.text(x + 0.078, y + 0.557, "warn / cool / block", fontsize=5.2, color=MUTED, ha="center")
    ax.annotate("", xy=(x + 0.118, y + 0.472), xytext=(x + 0.095, y + 0.565), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9))
    ax.add_patch(patches.FancyBboxPatch((x + 0.170, y + 0.555), 0.095, 0.065, boxstyle="round,pad=0.006,rounding_size=0.012", facecolor="white", edgecolor="#9CA3AF", linewidth=0.8, linestyle="--"))
    ax.text(x + 0.217, y + 0.588, "hidden labels\nnot exposed", fontsize=5.1, color="#6B7280", ha="center", va="center")

    # Module 3: scenario-aligned events.
    x, y, w, h, _ = modules[2]
    events = [("S1", "social +\nprice/liq"), ("S2", "social +\nprice/liq"), ("S3", "spoof /\nliquidity"), ("S4", "fake\nliquidity")]
    for i, (sid, desc) in enumerate(events):
        px = x + 0.030 + (i % 2) * 0.078
        py = y + 0.450 - (i // 2) * 0.155
        color = [PINK, MAGENTA, ORANGE, ROSE][i]
        ax.add_patch(patches.FancyBboxPatch((px, py), 0.064, 0.105, boxstyle="round,pad=0.006,rounding_size=0.014", facecolor="white", edgecolor=color, linewidth=0.95))
        ax.text(px + 0.032, py + 0.070, sid, fontsize=8.0, fontweight="bold", color=color, ha="center", va="center")
        ax.text(px + 0.032, py + 0.030, desc, fontsize=5.2, color=INK, ha="center", va="center")
    ax.add_patch(patches.FancyBboxPatch((x + 0.020, y + 0.100), 0.128, 0.060, boxstyle="round,pad=0.006,rounding_size=0.012", facecolor="#F3F4F6", edgecolor="#9CA3AF", linewidth=0.8))
    ax.text(x + 0.084, y + 0.130, "generic collapse\n= diagnostic", fontsize=5.4, color="#4B5563", ha="center", va="center")

    # Module 4: outputs and secondary defense audit.
    x, y, w, h, _ = modules[3]
    metrics = [(r"$P(C=1)$", "failure"), (r"$\alpha_c(N)$", "boundary"), (r"$W_N$", "width"), (r"$R_{excess}$", "severity"), (r"$\mathrm{CRP}$", "defense"), ("validity", "gates")]
    for i, (big, small) in enumerate(metrics):
        px = x + 0.020 + (i % 2) * 0.076
        py = y + 0.480 - (i // 2) * 0.128
        edge = BLUE if big == r"$\mathrm{CRP}$" else "#D8B4E2"
        ax.add_patch(patches.FancyBboxPatch((px, py), 0.062, 0.085, boxstyle="round,pad=0.006,rounding_size=0.014", facecolor="white", edgecolor=edge, linewidth=0.9))
        ax.text(px + 0.031, py + 0.055, big, fontsize=7.0, color=DEEP, fontweight="bold", ha="center", va="center")
        ax.text(px + 0.031, py + 0.023, small, fontsize=5.2, color=MUTED, ha="center", va="center")
    ax.text(x + 0.090, y + 0.103, "defense audit is secondary", fontsize=5.7, color=BLUE, ha="center")

    save(fig, "figure2_wolfbench_story")


def main() -> None:
    draw_single_column_teaser()
    draw_story_figure()
    print("Wrote story figures")


if __name__ == "__main__":
    main()
