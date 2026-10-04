"""Build the thesis-facing WolfBench Figure 1.

This figure is intentionally different from the experiment diagnostics. It is a
paper-opening synthesis figure: closed-loop mechanism, the main finite-size
transition curve, alpha_c movement, transition-width movement, and the defense
threshold-shift objective.

Usage:
    python -m experiments.paper_killer_figure1

Environment:
    WOLFBENCH_FIG1_SOURCE=exp12_canonical_scaling_refined
    WOLFBENCH_FIG1_OUT=current_figure_board
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from scipy.special import expit

from experiments._common import OUTPUTS_ROOT, SCALING_THEORY_OUTPUTS_ROOT


DEFAULT_SOURCE = "exp12_canonical_scaling_refined"
DEFAULT_OUT = "current_figure_board"

COLORS = {
    "ink": "#25142D",
    "muted": "#6E5A78",
    "grid": "#E9DEEE",
    "rule": "#D6C2DF",
    "panel": "#FFF8FD",
    "panel_alt": "#F9F2FB",
    "safe": "#FDE7F2",
    "critical": "#EFE1FF",
    "collapse": "#E7DCF4",
    "plum": "#3D1B52",
    "purple": "#5E2A84",
    "violet": "#8B5CF6",
    "magenta": "#E84A9B",
    "rose": "#F472B6",
    "pink": "#F7B6D9",
    "deep": "#4C1D95",
    "gray": "#A99BB0",
}

N_RAMP_COLORS = ["#F7B6D9", "#E879B8", "#A855F7", "#4C1D95"]
N_RAMP_LINEWIDTHS = [1.25, 1.65, 2.05, 2.75]
N_RAMP_MARKERSIZES = [2.6, 3.0, 3.4, 4.0]

SCENARIO_COLORS = {
    "s1": "#E84A9B",
    "s2": "#8B5CF6",
    "s3": "#C084FC",
    "s4": "#F472B6",
}

SCENARIO_LABELS = {
    "s1": "S1 social pump",
    "s2": "S2 finfluencer",
    "s3": "S3 spoofing",
    "s4": "S4 wash trading",
}

PRIMARY_LABELS = {
    "generic_collapse": "generic collapse",
    "spoof_liquidity_failure": "spoof/liquidity",
    "fake_liquidity_failure": "fake liquidity",
}


def configure_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": COLORS["rule"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.70,
            "font.family": "DejaVu Sans",
            "font.size": 7.3,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.8,
            "legend.fontsize": 7.0,
            "legend.title_fontsize": 7.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
        },
    )


def source_dir() -> Path:
    name = os.getenv("WOLFBENCH_FIG1_SOURCE", DEFAULT_SOURCE)
    path = SCALING_THEORY_OUTPUTS_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing Figure 1 source directory: {path}")
    return path


def output_dir() -> Path:
    name = os.getenv("WOLFBENCH_FIG1_OUT", DEFAULT_OUT)
    path = OUTPUTS_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Figure 1 artifact: {path}")
    return pd.read_csv(path)


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_artifacts(src: Path) -> dict[str, pd.DataFrame]:
    return {
        "curves": numeric(
            read_csv(src / "failure_curves.csv"),
            [
                "n_society",
                "alpha",
                "primary_failure_mean",
                "primary_failure_ci_low",
                "primary_failure_ci_high",
            ],
        ),
        "thresholds": numeric(
            read_csv(src / "alpha_c_by_scenario_n.csv"),
            [
                "n_society",
                "alpha_c_logistic",
                "alpha_c_ci_low",
                "alpha_c_ci_high",
                "transition_width_10_90",
                "logistic_slope",
            ],
        ),
        "widths": numeric(
            read_csv(src / "width_by_scenario_n.csv"),
            ["n_society", "transition_width_10_90", "alpha_c_logistic"],
        ),
        "summary": read_csv(src / "scenario_law_summary.csv"),
    }


def clean_axis(axis: plt.Axes, minor: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, which="major", color=COLORS["grid"], linewidth=0.70)
    if minor:
        axis.grid(True, which="minor", color=COLORS["grid"], linewidth=0.42, alpha=0.42)


def panel_title(axis: plt.Axes, label: str, title: str) -> None:
    axis.text(
        0.0,
        1.025,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color=COLORS["deep"],
    )
    axis.text(
        0.075,
        1.025,
        title,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.1,
        fontweight="bold",
        color=COLORS["ink"],
    )


def n_style(idx: int, count: int) -> tuple[str, float, float]:
    if count <= 1:
        return N_RAMP_COLORS[-1], N_RAMP_LINEWIDTHS[-1], N_RAMP_MARKERSIZES[-1]
    palette_idx = int(round(idx * (len(N_RAMP_COLORS) - 1) / max(count - 1, 1)))
    return (
        N_RAMP_COLORS[palette_idx],
        N_RAMP_LINEWIDTHS[palette_idx],
        N_RAMP_MARKERSIZES[palette_idx],
    )


def fmt_alpha(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    if value >= 0.01:
        return f"{value:.3g}".replace("0.", ".")
    return f"{value:.4g}".replace("0.", ".")


def draw_mechanism(axis: plt.Axes) -> None:
    axis.axis("off")
    panel_title(axis, "A", "closed-loop amplification")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    def node(text: str, x: float, y: float, face: str, width: float = 0.17) -> None:
        box = patches.FancyBboxPatch(
            (x - width / 2.0, y - 0.058),
            width,
            0.116,
            boxstyle="round,pad=0.012,rounding_size=0.026",
            facecolor=face,
            edgecolor=COLORS["rule"],
            linewidth=0.82,
            zorder=3,
        )
        axis.add_patch(box)
        axis.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=6.25,
            linespacing=0.90,
            color=COLORS["ink"],
            fontweight="bold",
            zorder=4,
        )

    node("harmful\nminority", 0.18, 0.58, COLORS["safe"], 0.18)
    node("shared\nexposure", 0.45, 0.76, "#FBE0F1", 0.18)
    node("benign\namplify", 0.74, 0.58, "#EFE1FF", 0.17)
    node("market\nfeedback", 0.51, 0.33, "#EADCF7", 0.18)
    node("primary\nfailure", 0.78, 0.31, "#E0D2F2", 0.17)

    axis.add_patch(
        patches.FancyBboxPatch(
            (0.335, 0.455),
            0.24,
            0.105,
            boxstyle="round,pad=0.012,rounding_size=0.022",
            facecolor="#FFFFFF",
            edgecolor=COLORS["grid"],
            linewidth=0.65,
            alpha=0.96,
            zorder=2,
        )
    )
    axis.text(
        0.455,
        0.508,
        "coupled response\ncurve",
        ha="center",
        va="center",
        fontsize=6.15,
        linespacing=0.90,
        color=COLORS["muted"],
        zorder=3,
    )

    arrow_specs = [
        ((0.27, 0.61), (0.36, 0.72), 0.02),
        ((0.54, 0.74), (0.65, 0.62), 0.02),
        ((0.70, 0.52), (0.56, 0.39), 0.00),
        ((0.52, 0.40), (0.46, 0.68), -0.34),
        ((0.60, 0.33), (0.69, 0.31), 0.00),
    ]
    for start, end, rad in arrow_specs:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle="-|>",
                color=COLORS["purple"],
                lw=1.0,
                shrinkA=0,
                shrinkB=0,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=2,
        )
    axis.text(0.12, 0.15, "sweep alpha and N", ha="left", va="center", fontsize=6.5, color=COLORS["muted"])
    axis.plot([0.12, 0.35], [0.11, 0.11], color=COLORS["rule"], lw=1.2, solid_capstyle="round")
    axis.plot([0.12, 0.24, 0.35], [0.095, 0.135, 0.095], color=COLORS["magenta"], lw=1.35, solid_capstyle="round")


def draw_hero_transition(axis: plt.Axes, curves: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    panel_title(axis, "B", "finite-size transition curves")
    scenario = "s1"
    rows = curves[curves["scenario"] == scenario].sort_values(["n_society", "alpha"])
    threshold_rows = thresholds[thresholds["scenario"] == scenario].sort_values("n_society")
    n_values = sorted(int(n) for n in rows["n_society"].dropna().unique())
    alpha_max = min(0.04, float(rows["alpha"].max()))
    axis.axvspan(0.0, 0.0085, color=COLORS["safe"], alpha=0.55, lw=0)
    axis.axvspan(0.0085, 0.0175, color=COLORS["critical"], alpha=0.38, lw=0)
    axis.axvspan(0.0175, alpha_max, color=COLORS["collapse"], alpha=0.38, lw=0)
    axis.axhline(0.5, color=COLORS["muted"], linestyle="--", lw=0.85, alpha=0.85)
    for idx, n_society in enumerate(n_values):
        n_rows = rows[rows["n_society"] == n_society].sort_values("alpha")
        if n_rows.empty:
            continue
        color, linewidth, markersize = n_style(idx, len(n_values))
        axis.plot(
            n_rows["alpha"],
            n_rows["primary_failure_mean"],
            color=color,
            linewidth=linewidth,
            marker="o",
            markersize=markersize,
            alpha=0.96,
            label=f"N={n_society}",
        )
        ci_rows = n_rows[n_rows["n_society"] == max(n_values)]
        if not ci_rows.empty and n_society == max(n_values):
            axis.fill_between(
                ci_rows["alpha"].to_numpy(dtype=float),
                ci_rows["primary_failure_ci_low"].to_numpy(dtype=float),
                ci_rows["primary_failure_ci_high"].to_numpy(dtype=float),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
    for idx, (_, t_row) in enumerate(threshold_rows.iterrows()):
        alpha_c = float(t_row["alpha_c_logistic"])
        if not np.isfinite(alpha_c) or alpha_c <= 0 or alpha_c > alpha_max:
            continue
        color, _, _ = n_style(idx, len(threshold_rows))
        axis.plot([alpha_c, alpha_c], [0.02, 0.12], color=color, lw=1.2, solid_capstyle="round")
    axis.text(0.0035, 0.91, "safe", ha="center", va="center", fontsize=7.2, color=COLORS["muted"])
    axis.text(0.013, 0.91, "near-critical", ha="center", va="center", fontsize=7.2, color=COLORS["purple"], fontweight="bold")
    axis.text(0.029, 0.91, "collapsed", ha="center", va="center", fontsize=7.2, color=COLORS["muted"])
    axis.annotate(
        "small change in harmful fraction\nlarge change in collective failure",
        xy=(0.014, 0.55),
        xytext=(0.022, 0.30),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["deep"], lw=0.9, connectionstyle="arc3,rad=-0.18"),
        fontsize=7.1,
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    axis.set_xlim(-0.0008, alpha_max * 1.02)
    axis.set_ylim(-0.04, 1.04)
    ticks = [0.0, 0.01, 0.02, 0.03, 0.04]
    axis.set_xticks(ticks)
    axis.set_xticklabels([fmt_alpha(tick) for tick in ticks])
    axis.set_xlabel("harmful-agent ratio alpha")
    axis.set_ylabel("P(primary failure)")
    clean_axis(axis)


def draw_alpha_c(axis: plt.Axes, thresholds: pd.DataFrame) -> None:
    panel_title(axis, "C", "critical ratio moves with N")
    for scenario in ["s1", "s2", "s3", "s4"]:
        rows = thresholds[(thresholds["scenario"] == scenario) & (thresholds["coverage_status"] != "left_censored_above_threshold")]
        rows = rows.dropna(subset=["alpha_c_logistic"]).sort_values("n_society")
        if rows.empty:
            continue
        color = SCENARIO_COLORS.get(scenario, COLORS["magenta"])
        linewidth = 2.4 if scenario == "s1" else 1.25
        alpha = 0.98 if scenario == "s1" else 0.62
        y = rows["alpha_c_logistic"].to_numpy(dtype=float)
        low = rows["alpha_c_ci_low"].fillna(rows["alpha_c_logistic"]).to_numpy(dtype=float)
        high = rows["alpha_c_ci_high"].fillna(rows["alpha_c_logistic"]).to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(0.0, y - low), np.maximum(0.0, high - y)])
        axis.errorbar(
            rows["n_society"],
            y,
            yerr=yerr,
            marker="o",
            markersize=3.5 if scenario == "s1" else 2.8,
            linewidth=linewidth,
            capsize=2.1,
            color=color,
            alpha=alpha,
            label=scenario.upper(),
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("society size N")
    axis.set_ylabel("alpha_c")
    axis.legend(frameon=False, ncol=2, loc="best", handlelength=1.6)
    clean_axis(axis)


def draw_width(axis: plt.Axes, widths: pd.DataFrame) -> None:
    panel_title(axis, "D", "near-critical width")
    for scenario in ["s1", "s2", "s3", "s4"]:
        rows = widths[(widths["scenario"] == scenario) & (widths["coverage_status"] != "left_censored_above_threshold")]
        rows = rows[pd.to_numeric(rows["transition_width_10_90"], errors="coerce") > 0]
        rows = rows.sort_values("n_society")
        if rows.empty:
            continue
        color = SCENARIO_COLORS.get(scenario, COLORS["magenta"])
        linewidth = 2.4 if scenario == "s1" else 1.25
        alpha = 0.98 if scenario == "s1" else 0.62
        axis.plot(
            rows["n_society"],
            rows["transition_width_10_90"],
            marker="o",
            markersize=3.5 if scenario == "s1" else 2.8,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            label=scenario.upper(),
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("society size N")
    axis.set_ylabel("width W_N")
    clean_axis(axis)


def draw_defense_objective(axis: plt.Axes) -> None:
    panel_title(axis, "E", "defense objective")
    alpha = np.linspace(0.0, 0.045, 300)
    no_guard = expit(260.0 * (alpha - 0.014))
    defended = expit(245.0 * (alpha - 0.022))
    axis.axvspan(0.009, 0.019, color=COLORS["critical"], alpha=0.48, lw=0)
    axis.plot(alpha, no_guard, color=COLORS["gray"], lw=2.0, label="NoGuard")
    axis.plot(alpha, defended, color=COLORS["deep"], lw=2.5, label="useful defense")
    axis.axhline(0.5, color=COLORS["muted"], linestyle="--", lw=0.80)
    axis.plot([0.014, 0.014], [0.0, 0.50], color=COLORS["gray"], lw=1.0)
    axis.plot([0.022, 0.022], [0.0, 0.50], color=COLORS["deep"], lw=1.0)
    axis.annotate(
        "threshold shift",
        xy=(0.022, 0.64),
        xytext=(0.014, 0.64),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["deep"], lw=1.0),
        fontsize=6.9,
        color=COLORS["deep"],
        ha="center",
        va="bottom",
    )
    axis.set_xlim(0.0, 0.045)
    axis.set_ylim(-0.04, 1.04)
    axis.set_xticks([0.0, 0.015, 0.03, 0.045])
    axis.set_xticklabels(["0", ".015", ".030", ".045"])
    axis.set_xlabel("harmful-agent ratio alpha")
    axis.set_ylabel("P(failure)")
    axis.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        loc="lower right",
        handlelength=1.5,
    )
    clean_axis(axis)


def write_manifest_entry(out_dir: Path, source: Path, png: Path, pdf: Path) -> None:
    entry = {
        "figure": "figure_1_killer_scaling",
        "title": "Figure 1: Harmful-agent population scaling",
        "png": str(png.relative_to(OUTPUTS_ROOT.parent)),
        "pdf": str(pdf.relative_to(OUTPUTS_ROOT.parent)),
        "sources": [
            str((source / "failure_curves.csv").relative_to(OUTPUTS_ROOT.parent)),
            str((source / "alpha_c_by_scenario_n.csv").relative_to(OUTPUTS_ROOT.parent)),
            str((source / "width_by_scenario_n.csv").relative_to(OUTPUTS_ROOT.parent)),
            str((source / "scenario_law_summary.csv").relative_to(OUTPUTS_ROOT.parent)),
        ],
        "note": "Thesis-facing overview figure; defense panel is an objective schematic unless replaced by a full TPS rerun.",
    }
    (out_dir / "figure_1_killer_scaling_manifest.json").write_text(json.dumps(entry, indent=2) + "\n")


def build_figure(src: Path, out_dir: Path) -> tuple[Path, Path]:
    artifacts = load_artifacts(src)
    figure = plt.figure(figsize=(9.25, 5.20), constrained_layout=False)
    gs = figure.add_gridspec(
        2,
        6,
        height_ratios=[1.23, 1.0],
        width_ratios=[1.05, 1.30, 1.30, 1.02, 1.02, 1.02],
        left=0.055,
        right=0.985,
        bottom=0.085,
        top=0.795,
        hspace=0.58,
        wspace=0.62,
    )
    mechanism_axis = figure.add_subplot(gs[0, 0:2])
    hero_axis = figure.add_subplot(gs[0, 2:6])
    alpha_axis = figure.add_subplot(gs[1, 0:2])
    width_axis = figure.add_subplot(gs[1, 2:4])
    defense_axis = figure.add_subplot(gs[1, 4:6])

    figure.suptitle(
        "Harmful-agent safety has a population-scaling dimension",
        x=0.055,
        y=0.976,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    figure.text(
        0.055,
        0.922,
        "WolfBench estimates where coupled agent societies tip from safe behavior to collective failure.",
        ha="left",
        va="top",
        fontsize=7.3,
        color=COLORS["muted"],
    )

    draw_mechanism(mechanism_axis)
    draw_hero_transition(hero_axis, artifacts["curves"], artifacts["thresholds"])
    handles, labels = hero_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.902),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.1,
        labelspacing=0.3,
    )
    draw_alpha_c(alpha_axis, artifacts["thresholds"])
    draw_width(width_axis, artifacts["widths"])
    draw_defense_objective(defense_axis)

    png = out_dir / "figure_1_killer_scaling.png"
    pdf = out_dir / "figure_1_killer_scaling.pdf"
    figure.savefig(png, dpi=380, bbox_inches="tight", pad_inches=0.035)
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)
    write_manifest_entry(out_dir, src, png, pdf)
    return png, pdf


def main() -> None:
    configure_style()
    src = source_dir()
    out = output_dir()
    png, pdf = build_figure(src, out)
    print(f"Wrote killer Figure 1 to {png}")
    print(pdf)


if __name__ == "__main__":
    main()