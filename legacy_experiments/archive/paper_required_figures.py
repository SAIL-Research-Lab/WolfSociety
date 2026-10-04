"""Build the five paper-required WolfBench figures.

This script is intentionally separate from ``experiments.paper_figures``. The
older script is an experiment-output overview; this one is a paper-narrative
figure pipeline. It reads existing CSV/JSON artifacts only and writes figures to
``paperoutputs/scaling/paper_required_figures/``.

Usage:
    python -m experiments.paper_required_figures
"""
from __future__ import annotations

import json
import math
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

from experiments._common import (
    DEFENSE_BENCHMARK_OUTPUTS_ROOT,
    OUTPUTS_ROOT,
    SCALING_THEORY_OUTPUTS_ROOT,
)


OUT_DIR = SCALING_THEORY_OUTPUTS_ROOT / "paper_required_figures"

COLORS = {
    "ink": "#25142D",
    "muted": "#6F5F78",
    "grid": "#E8DDEC",
    "panel": "#FFF9FD",
    "lavender": "#7B3FA1",
    "purple": "#5E2A84",
    "violet": "#9B5DE5",
    "magenta": "#F15BB5",
    "rose": "#F284B6",
    "pink": "#FFB3D9",
    "plum": "#3D1B52",
    "gray": "#A99BB0",
    "good": "#B43E8F",
    "bad": "#7756A5",
}

SCENARIO_COLORS = {
    "s1": "#E84A9B",
    "s2": "#8B5CF6",
    "s3": "#C084FC",
    "s4": "#F472B6",
}

SCENARIO_NAMES = {
    "s1": "S1 social pump",
    "s2": "S2 finfluencer",
    "s3": "S3 spoofing",
    "s4": "S4 wash trading",
}

CHANNELS = {
    "s1": "social exposure + momentum",
    "s2": "centrality reach",
    "s3": "liquidity stress",
    "s4": "fake volume signal",
}

FAMILY_LABELS = {
    "asset_liquidity_scale": "Liquidity",
    "social_mean_degree": "Graph degree",
    "placement": "Central placement",
    "retail_wealth_scale": "Retail capital",
    "retail_risk_appetite": "Risk appetite",
    "feedback_strength": "Feedback",
}


def configure_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D8C8DF",
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.7,
            "legend.title_fontsize": 7.7,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.045,
        },
    )


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_path(path))


def read_json(path: Path) -> dict[str, Any]:
    with open(require_path(path)) as handle:
        return json.load(handle)


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, color=COLORS["grid"], linewidth=0.8)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=COLORS["plum"],
    )


def small_note(axis: plt.Axes, text: str, loc: str = "upper left") -> None:
    xy = {
        "upper left": (0.03, 0.96, "left", "top"),
        "upper right": (0.97, 0.96, "right", "top"),
        "lower left": (0.03, 0.04, "left", "bottom"),
        "lower right": (0.97, 0.04, "right", "bottom"),
    }[loc]
    axis.text(
        xy[0],
        xy[1],
        text,
        transform=axis.transAxes,
        ha=xy[2],
        va=xy[3],
        fontsize=7.2,
        color=COLORS["muted"],
        bbox={
            "boxstyle": "round,pad=0.25,rounding_size=0.05",
            "facecolor": COLORS["panel"],
            "edgecolor": "#E3CBEA",
            "linewidth": 0.65,
            "alpha": 0.95,
        },
    )


def save_figure(
    figure: plt.Figure,
    stem: str,
    title: str,
    sources: list[Path],
    note: str,
    manifest: list[dict[str, Any]],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    figure.savefig(png, dpi=360)
    figure.savefig(pdf)
    plt.close(figure)
    manifest.append(
        {
            "figure": stem,
            "title": title,
            "png": str(png.relative_to(OUTPUTS_ROOT.parent)),
            "pdf": str(pdf.relative_to(OUTPUTS_ROOT.parent)),
            "sources": [str(path.relative_to(OUTPUTS_ROOT.parent)) for path in sources if path.exists()],
            "note": note,
        }
    )


def write_manifest(manifest: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figure_manifest.json").write_text(json.dumps({"figures": manifest}, indent=2) + "\n")
    lines = [
        "# Paper-Required WolfBench Figures",
        "",
        "These are the main-paper narrative figures. They use a unified pink/purple palette and read existing experiment artifacts only.",
        "",
        "| Figure | Files | Main sources | Role |",
        "|---|---|---|---|",
    ]
    for entry in manifest:
        files = f"`{entry['png']}`, `{entry['pdf']}`"
        sources = ", ".join(f"`{source}`" for source in entry["sources"])
        lines.append(f"| {entry['title']} | {files} | {sources} | {entry['note']} |")
    lines.append("")
    (OUT_DIR / "figure_manifest.md").write_text("\n".join(lines))


def power_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    pred = np.exp(intercept) * x**slope
    ss_res = float(np.sum((np.log(y) - np.log(pred)) ** 2))
    ss_tot = float(np.sum((np.log(y) - float(np.mean(np.log(y)))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2_log": float(r2)}


def format_alpha(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if abs(value) >= 0.1:
        return f"{value:.2f}"
    if abs(value) >= 0.01:
        return f"{value:.3f}"
    return f"{value:.4f}"


def plot_closed_loop_schematic(axis: plt.Axes) -> None:
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    nodes = [
        (0.18, 0.70, "Harmful\nminority", COLORS["magenta"]),
        (0.50, 0.78, "Social\nexposure", COLORS["violet"]),
        (0.80, 0.55, "Benign\namplify", COLORS["rose"]),
        (0.58, 0.25, "Market\nfeedback", COLORS["purple"]),
        (0.22, 0.32, "Collapse\nevent", COLORS["plum"]),
    ]
    for x, y, label, color in nodes:
        box = patches.FancyBboxPatch(
            (x - 0.122, y - 0.064),
            0.244,
            0.128,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            facecolor=color,
            edgecolor="#FFFFFF",
            linewidth=1.2,
            alpha=0.92,
        )
        axis.add_patch(box)
        axis.text(x, y, label, ha="center", va="center", color="white", fontsize=7.6, fontweight="bold")
    arrows = [
        ((0.29, 0.72), (0.39, 0.77)),
        ((0.61, 0.75), (0.69, 0.60)),
        ((0.76, 0.48), (0.64, 0.32)),
        ((0.47, 0.27), (0.33, 0.31)),
        ((0.20, 0.39), (0.17, 0.62)),
    ]
    for start, end in arrows:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "lw": 1.7, "color": COLORS["plum"], "alpha": 0.78},
        )
    axis.text(
        0.50,
        0.05,
        "Sweep alpha, fit alpha_c and transition width",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )


def figure_1_at_a_glance(manifest: list[dict[str, Any]]) -> None:
    exp1_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp1_alpha_scaling"
    exp2_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp2_society_size_scaling"
    exp1_curve = numeric(
        read_csv(exp1_dir / "collapse_rate_wilson_ci.csv"),
        ["n_society", "alpha", "mean", "ci_low", "ci_high"],
    )
    exp2_curve = numeric(
        read_csv(exp2_dir / "collapse_rate_wilson_ci.csv"),
        ["n_society", "alpha", "mean", "ci_low", "ci_high"],
    )
    thresholds = numeric(
        read_csv(exp2_dir / "alpha_critical_summary.csv"),
        ["n_society", "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high", "logistic_slope", "transition_width_10_90"],
    )

    figure = plt.figure(figsize=(9.8, 6.6), constrained_layout=True)
    gs = figure.add_gridspec(2, 6)
    axis_loop = figure.add_subplot(gs[0, 0:2])
    axis_transition = figure.add_subplot(gs[0, 2:4])
    axis_collapse = figure.add_subplot(gs[0, 4:6])
    axis_alpha = figure.add_subplot(gs[1, 0:3])
    axis_width = figure.add_subplot(gs[1, 3:6])

    plot_closed_loop_schematic(axis_loop)
    panel_label(axis_loop, "A")
    axis_loop.set_title("Closed harmful-agent loop")

    selected_n = [200, 1000, 5000]
    selected_colors = [COLORS["pink"], COLORS["magenta"], COLORS["purple"]]
    for n_value, color in zip(selected_n, selected_colors):
        subset = exp1_curve[exp1_curve["n_society"] == n_value].sort_values("alpha")
        if subset.empty:
            continue
        threshold = thresholds[thresholds["n_society"] == n_value]
        axis_transition.plot(subset["alpha"], subset["mean"], marker="o", lw=1.9, ms=4, color=color, label=f"N={n_value}")
        axis_transition.fill_between(subset["alpha"], subset["ci_low"], subset["ci_high"], color=color, alpha=0.16, lw=0)
        if not threshold.empty and pd.notna(threshold.iloc[0]["alpha_c_logistic"]):
            axis_transition.axvline(float(threshold.iloc[0]["alpha_c_logistic"]), color=color, ls=":", lw=1.3)
    axis_transition.axhline(0.5, color=COLORS["gray"], ls="--", lw=1.0)
    axis_transition.set_xscale("symlog", linthresh=1e-3)
    axis_transition.set_xlim(left=0.0, right=float(exp1_curve["alpha"].max()) * 1.05)
    axis_transition.set_ylim(-0.04, 1.04)
    axis_transition.set_xlabel("Harmful-agent ratio alpha")
    axis_transition.set_ylabel("P(collapse)")
    axis_transition.set_title("Finite-N transition curves")
    axis_transition.legend(frameon=False, loc="lower right")
    clean_axis(axis_transition)
    panel_label(axis_transition, "B")

    for _, row in thresholds.dropna(subset=["alpha_c_logistic", "logistic_slope"]).iterrows():
        n_value = int(row["n_society"])
        subset = exp2_curve[exp2_curve["n_society"] == n_value]
        scaled_x = float(row["logistic_slope"]) * (subset["alpha"].to_numpy(dtype=float) - float(row["alpha_c_logistic"]))
        axis_collapse.scatter(
            scaled_x,
            subset["mean"],
            s=15,
            color=plt.cm.magma(0.18 + 0.68 * (math.log(n_value) - math.log(100)) / (math.log(5000) - math.log(100))),
            alpha=0.72,
            edgecolor="white",
            linewidth=0.3,
        )
    x_grid = np.linspace(-7, 7, 240)
    axis_collapse.plot(x_grid, expit(x_grid), color=COLORS["plum"], lw=2.2, label="sigmoid")
    axis_collapse.set_xlim(-7, 7)
    axis_collapse.set_ylim(-0.04, 1.04)
    axis_collapse.set_xlabel(r"$s_N(\alpha-\alpha_c(N))$")
    axis_collapse.set_ylabel("P(collapse)")
    axis_collapse.set_title("Finite-size data collapse")
    clean_axis(axis_collapse)
    small_note(axis_collapse, "Raw grid rates\nrescaled by fitted midpoint", "lower right")
    panel_label(axis_collapse, "C")

    threshold_subset = thresholds.dropna(subset=["alpha_c_logistic"]).sort_values("n_society")
    x = threshold_subset["n_society"].to_numpy(dtype=float)
    y = threshold_subset["alpha_c_logistic"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            np.maximum(0.0, y - threshold_subset["alpha_c_ci_low"].to_numpy(dtype=float)),
            np.maximum(0.0, threshold_subset["alpha_c_ci_high"].to_numpy(dtype=float) - y),
        ]
    )
    alpha_fit = power_fit(x, y)
    xfit = np.logspace(np.log10(x.min()), np.log10(x.max()), 180)
    yfit = np.exp(alpha_fit["intercept"]) * xfit ** alpha_fit["slope"]
    axis_alpha.errorbar(x, y, yerr=yerr, fmt="o", color=COLORS["magenta"], ecolor=COLORS["pink"], capsize=3, ms=4.8)
    axis_alpha.plot(xfit, yfit, color=COLORS["purple"], ls="--", lw=1.8)
    axis_alpha.set_xscale("log")
    axis_alpha.set_yscale("log")
    axis_alpha.set_xlabel("Society size N")
    axis_alpha.set_ylabel(r"Critical ratio $\alpha_c(N)$")
    axis_alpha.set_title("Critical ratio decreases with N")
    clean_axis(axis_alpha)
    small_note(axis_alpha, f"beta={alpha_fit['slope']:.2f}\nlog-R2={alpha_fit['r2_log']:.2f}", "upper right")
    panel_label(axis_alpha, "D")

    widths = threshold_subset["transition_width_10_90"].to_numpy(dtype=float)
    width_fit = power_fit(x, widths)
    width_yfit = np.exp(width_fit["intercept"]) * xfit ** width_fit["slope"]
    axis_width.plot(x, widths, "o", color=COLORS["violet"], ms=4.8)
    axis_width.plot(xfit, width_yfit, color=COLORS["plum"], ls="--", lw=1.8)
    axis_width.set_xscale("log")
    axis_width.set_yscale("log")
    axis_width.set_xlabel("Society size N")
    axis_width.set_ylabel(r"Width $W_N=\alpha_{.9}-\alpha_{.1}$")
    axis_width.set_title("Transitions sharpen with scale")
    clean_axis(axis_width)
    small_note(axis_width, f"gamma={-width_fit['slope']:.2f}\nlog-R2={width_fit['r2_log']:.2f}", "upper right")
    panel_label(axis_width, "E")

    save_figure(
        figure,
        "figure_1_at_a_glance_scaling",
        "Figure 1: harmful-agent scaling at a glance",
        [
            exp1_dir / "collapse_rate_wilson_ci.csv",
            exp2_dir / "collapse_rate_wilson_ci.csv",
            exp2_dir / "alpha_critical_summary.csv",
        ],
        "Main claim figure: closed loop, finite-N transitions, data collapse, alpha_c(N), and width scaling.",
        manifest,
    )


def figure_2_scaling_law(manifest: list[dict[str, Any]]) -> None:
    exp2_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp2_society_size_scaling"
    thresholds = numeric(
        read_csv(exp2_dir / "alpha_critical_summary.csv"),
        ["n_society", "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high", "transition_width_10_90", "logistic_slope"],
    )
    curve = numeric(read_csv(exp2_dir / "collapse_rate_wilson_ci.csv"), ["n_society", "alpha", "mean"])

    figure = plt.figure(figsize=(8.8, 5.4), constrained_layout=True)
    gs = figure.add_gridspec(2, 3)
    axis_heat = figure.add_subplot(gs[:, 0])
    axis_alpha = figure.add_subplot(gs[0, 1:])
    axis_width = figure.add_subplot(gs[1, 1])
    axis_table = figure.add_subplot(gs[1, 2])

    heat = curve.pivot_table(index="n_society", columns="alpha", values="mean", aggfunc="mean").sort_index()
    sns.heatmap(
        heat,
        ax=axis_heat,
        cmap=sns.blend_palette(["#FFF1F7", "#F15BB5", "#5E2A84", "#25142D"], as_cmap=True),
        vmin=0,
        vmax=1,
        linewidths=0.25,
        linecolor="#FFFFFF",
        cbar_kws={"label": "P(collapse)", "shrink": 0.72},
    )
    axis_heat.set_title("Measured collapse surface")
    axis_heat.set_xlabel("alpha")
    axis_heat.set_ylabel("N")
    axis_heat.set_xticklabels([format_alpha(float(v)) for v in heat.columns], rotation=45, ha="right")
    axis_heat.set_yticklabels([str(int(v)) for v in heat.index], rotation=0)
    panel_label(axis_heat, "A")

    subset = thresholds.dropna(subset=["alpha_c_logistic"]).sort_values("n_society")
    x = subset["n_society"].to_numpy(dtype=float)
    alpha_y = subset["alpha_c_logistic"].to_numpy(dtype=float)
    fit_all = power_fit(x, alpha_y)
    mask_stable = x >= 500
    fit_stable = power_fit(x[mask_stable], alpha_y[mask_stable])
    xfit = np.logspace(np.log10(x.min()), np.log10(x.max()), 180)
    yfit = np.exp(fit_stable["intercept"]) * xfit ** fit_stable["slope"]
    yerr = np.vstack(
        [
            np.maximum(0, alpha_y - subset["alpha_c_ci_low"].to_numpy(dtype=float)),
            np.maximum(0, subset["alpha_c_ci_high"].to_numpy(dtype=float) - alpha_y),
        ]
    )
    axis_alpha.errorbar(x, alpha_y, yerr=yerr, fmt="o", color=COLORS["magenta"], ecolor=COLORS["pink"], capsize=3, ms=4.5)
    axis_alpha.plot(xfit, yfit, color=COLORS["plum"], lw=2.0, ls="--", label="N >= 500 power fit")
    axis_alpha.set_xscale("log")
    axis_alpha.set_yscale("log")
    axis_alpha.set_ylabel(r"$\alpha_c(N)$")
    axis_alpha.set_xlabel("Society size N")
    axis_alpha.set_title("Finite-size critical ratio")
    axis_alpha.legend(frameon=False)
    clean_axis(axis_alpha)
    panel_label(axis_alpha, "B")

    width_y = subset["transition_width_10_90"].to_numpy(dtype=float)
    width_fit = power_fit(x[mask_stable], width_y[mask_stable])
    width_yfit = np.exp(width_fit["intercept"]) * xfit ** width_fit["slope"]
    axis_width.plot(x, width_y, "o", color=COLORS["violet"], ms=4.5)
    axis_width.plot(xfit, width_yfit, color=COLORS["plum"], lw=1.8, ls="--")
    axis_width.set_xscale("log")
    axis_width.set_yscale("log")
    axis_width.set_xlabel("N")
    axis_width.set_ylabel(r"$W_N$")
    axis_width.set_title("Transition width")
    clean_axis(axis_width)
    panel_label(axis_width, "C")

    axis_table.axis("off")
    rows = [
        ("alpha_c range", f"{alpha_y[0]:.4f} -> {alpha_y[-1]:.4f}"),
        ("width range", f"{width_y[0]:.4f} -> {width_y[-1]:.4f}"),
        ("nu, all N", f"{-fit_all['slope']:.2f}"),
        ("nu, N>=500", f"{-fit_stable['slope']:.2f}"),
        ("gamma, N>=500", f"{-width_fit['slope']:.2f}"),
        ("bootstrap", "2000 fits / N"),
    ]
    table = axis_table.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Summary", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.05, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#E4D3EA")
        if row == 0:
            cell.set_facecolor("#F3D8F3")
            cell.set_text_props(weight="bold", color=COLORS["ink"])
        else:
            cell.set_facecolor("#FFF9FD" if row % 2 else "#FBEAF6")
    axis_table.set_title("Scaling summary")
    panel_label(axis_table, "D")

    save_figure(
        figure,
        "figure_2_finite_size_scaling_law",
        "Figure 2: finite-size scaling law evidence",
        [exp2_dir / "collapse_rate_wilson_ci.csv", exp2_dir / "alpha_critical_summary.csv"],
        "Pure RQ1 result figure: heatmap, alpha_c fit, transition-width fit, and numeric summary.",
        manifest,
    )


def figure_3_mechanism_heterogeneity(manifest: list[dict[str, Any]]) -> None:
    exp7_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp7_cross_mechanism_threshold"
    thresholds = numeric(
        read_csv(exp7_dir / "alpha_critical_by_mechanism.csv"),
        ["n_society", "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high", "transition_width_10_90"],
    )
    curves = numeric(read_csv(exp7_dir / "collapse_rate_wilson_ci.csv"), ["n_society", "alpha", "mean"])

    figure = plt.figure(figsize=(9.0, 5.7), constrained_layout=True)
    gs = figure.add_gridspec(2, 3)
    axis_threshold = figure.add_subplot(gs[:, 0])
    axis_curves = figure.add_subplot(gs[0, 1:])
    axis_width = figure.add_subplot(gs[1, 1])
    axis_cards = figure.add_subplot(gs[1, 2])

    for scenario in ["s1", "s2", "s3", "s4"]:
        subset = thresholds[thresholds["scenario"] == scenario].dropna(subset=["alpha_c_logistic"]).sort_values("n_society")
        if subset.empty:
            continue
        color = SCENARIO_COLORS[scenario]
        x = subset["n_society"].to_numpy(dtype=float)
        y = subset["alpha_c_logistic"].to_numpy(dtype=float)
        low = subset["alpha_c_ci_low"].fillna(subset["alpha_c_logistic"]).to_numpy(dtype=float)
        high = subset["alpha_c_ci_high"].fillna(subset["alpha_c_logistic"]).to_numpy(dtype=float)
        axis_threshold.errorbar(x, y, yerr=np.vstack([np.maximum(0, y - low), np.maximum(0, high - y)]), fmt="o-", color=color, capsize=3, lw=1.8, ms=4.5, label=scenario.upper())
    axis_threshold.set_xscale("log")
    axis_threshold.set_yscale("log")
    axis_threshold.set_xlabel("Society size N")
    axis_threshold.set_ylabel(r"Mechanism-specific $\alpha_c^m(N)$")
    axis_threshold.set_title("Same estimand, different thresholds")
    axis_threshold.legend(frameon=False, title="Mechanism")
    clean_axis(axis_threshold)
    small_note(axis_threshold, "S3/S4 curves are\nreported with caveats", "lower left")
    panel_label(axis_threshold, "A")

    n_focus = 1000
    for scenario in ["s1", "s2", "s3", "s4"]:
        subset = curves[(curves["scenario"] == scenario) & (curves["n_society"] == n_focus)].sort_values("alpha")
        if subset.empty:
            continue
        axis_curves.plot(subset["alpha"], subset["mean"], marker="o", lw=1.7, ms=3.8, color=SCENARIO_COLORS[scenario], label=scenario.upper())
    axis_curves.axhline(0.5, color=COLORS["gray"], ls="--", lw=1)
    axis_curves.set_xscale("symlog", linthresh=1e-4)
    axis_curves.set_xlim(left=0.0, right=float(curves["alpha"].max()) * 1.03)
    axis_curves.set_ylim(-0.04, 1.04)
    axis_curves.set_xlabel("alpha")
    axis_curves.set_ylabel("P(collapse)")
    axis_curves.set_title("Response curves at N=1000")
    axis_curves.legend(frameon=False, ncol=4, loc="lower right")
    clean_axis(axis_curves)
    panel_label(axis_curves, "B")

    width_rows = thresholds.dropna(subset=["transition_width_10_90"])
    width_rows = width_rows[width_rows["n_society"] == n_focus].copy()
    width_rows["label"] = width_rows["scenario"].str.upper()
    axis_width.bar(
        width_rows["label"],
        width_rows["transition_width_10_90"],
        color=[SCENARIO_COLORS[s] for s in width_rows["scenario"]],
        alpha=0.86,
    )
    axis_width.set_yscale("log")
    axis_width.set_ylabel("Transition width")
    axis_width.set_title("Sharpness is mechanism dependent")
    clean_axis(axis_width)
    panel_label(axis_width, "C")

    axis_cards.axis("off")
    axis_cards.set_title("Dominant channel")
    y0 = 0.86
    for i, scenario in enumerate(["s1", "s2", "s3", "s4"]):
        y = y0 - i * 0.22
        rect = patches.FancyBboxPatch(
            (0.02, y - 0.075),
            0.96,
            0.15,
            boxstyle="round,pad=0.02,rounding_size=0.035",
            facecolor="#FFF4FB" if i % 2 == 0 else "#F7EAFF",
            edgecolor=SCENARIO_COLORS[scenario],
            linewidth=1.0,
        )
        axis_cards.add_patch(rect)
        axis_cards.text(0.07, y + 0.025, scenario.upper(), color=SCENARIO_COLORS[scenario], fontweight="bold", fontsize=9, ha="left", va="center")
        axis_cards.text(0.22, y + 0.025, SCENARIO_NAMES[scenario][3:], color=COLORS["ink"], fontsize=8, ha="left", va="center")
        axis_cards.text(0.07, y - 0.035, CHANNELS[scenario], color=COLORS["muted"], fontsize=7.5, ha="left", va="center")
    panel_label(axis_cards, "D")

    save_figure(
        figure,
        "figure_3_mechanism_heterogeneity",
        "Figure 3: mechanism-specific critical regimes",
        [exp7_dir / "alpha_critical_by_mechanism.csv", exp7_dir / "collapse_rate_wilson_ci.csv"],
        "RQ2 figure: same alpha_c estimator reveals mechanism-specific thresholds and widths.",
        manifest,
    )


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _alpha_c_lookup(frame: pd.DataFrame, scenario: str, family: str, value: Any) -> float | None:
    rows = frame[(frame["scenario"] == scenario) & (frame["family"] == family) & (frame["value"].astype(str) == str(value))]
    rows = rows.dropna(subset=["alpha_c"])
    if rows.empty:
        return None
    return float(rows.iloc[0]["alpha_c"])


def build_signed_shift_table(alpha_rows: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("s1", "asset_liquidity_scale", "2.0", "0.5", "Liquidity high - low", "+"),
        ("s1", "social_mean_degree", "16", "8", "Degree high - base", "-"),
        ("s1", "placement", "high_degree", "random", "High centrality - random", "-"),
        ("s1", "retail_wealth_scale", "2.0", "0.5", "Retail capital high - low", "-"),
        ("s2", "asset_liquidity_scale", "2.0", "0.5", "Liquidity high - low", "+"),
        ("s2", "retail_risk_appetite", "0.04", "0.01", "Risk appetite high - low", "-"),
        ("s3", "asset_liquidity_scale", "0.75", "0.5", "Liquidity high - low", "+"),
        ("s4", "asset_liquidity_scale", "2.0", "0.5", "Liquidity high - low", "+"),
        ("s4", "feedback_strength", "1.6", "0.0", "Feedback high - low", "-"),
    ]
    rows: list[dict[str, Any]] = []
    for scenario, family, high, low, label, expected in specs:
        high_alpha = _alpha_c_lookup(alpha_rows, scenario, family, high)
        low_alpha = _alpha_c_lookup(alpha_rows, scenario, family, low)
        if high_alpha is None or low_alpha is None:
            continue
        delta = high_alpha - low_alpha
        supports = (delta > 0 and expected == "+") or (delta < 0 and expected == "-") or abs(delta) < 1e-12
        rows.append(
            {
                "scenario": scenario,
                "family": family,
                "label": label,
                "expected": expected,
                "delta_alpha_c": delta,
                "supports": supports,
                "alpha_low": low_alpha,
                "alpha_high": high_alpha,
            }
        )
    return pd.DataFrame(rows)


def figure_4_structural_shifts(manifest: list[dict[str, Any]]) -> None:
    exp10_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp10_comparative_statics_threshold"
    delta_rows = numeric(
        read_csv(exp10_dir / "comparative_statics_summary.csv"),
        [
            "alpha_c_base", "alpha_c_changed", "delta_alpha_c",
            "delta_alpha_c_ci_low", "delta_alpha_c_ci_high",
            "transition_width_base", "transition_width_changed",
        ],
    )
    alpha_rows = numeric(
        read_csv(exp10_dir / "alpha_c_by_variant.csv"),
        ["alpha_c", "alpha_c_ci_low", "alpha_c_ci_high", "transition_width_10_90"],
    )

    figure = plt.figure(figsize=(9.0, 5.6), constrained_layout=True)
    gs = figure.add_gridspec(2, 3)
    axis_sign = figure.add_subplot(gs[:, 0:2])
    axis_range = figure.add_subplot(gs[0, 2])
    axis_feedback = figure.add_subplot(gs[1, 2])

    signed = delta_rows.dropna(subset=["delta_alpha_c"]).copy()
    signed = signed.sort_values("delta_alpha_c", ascending=True).reset_index(drop=True)
    labels = [
        f"{row.scenario.upper()}  {row.lever_label}"
        for row in signed.itertuples()
    ]
    values = signed["delta_alpha_c"].to_numpy(dtype=float)
    ci_low = signed["delta_alpha_c_ci_low"].to_numpy(dtype=float)
    ci_high = signed["delta_alpha_c_ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([
        np.maximum(0.0, values - ci_low),
        np.maximum(0.0, ci_high - values),
    ])
    bar_colors = [
        COLORS["magenta"] if value > 0 else COLORS["purple"] if value < 0 else COLORS["gray"]
        for value in values
    ]
    y_pos = np.arange(len(signed))
    axis_sign.barh(y_pos, values, color=bar_colors, alpha=0.88)
    if np.isfinite(xerr).all():
        axis_sign.errorbar(values, y_pos, xerr=xerr, fmt="none", ecolor=COLORS["ink"], lw=0.75, capsize=2)
    axis_sign.axvline(0.0, color=COLORS["ink"], lw=0.9)
    axis_sign.set_yticks(y_pos)
    axis_sign.set_yticklabels(labels)
    axis_sign.tick_params(axis="y", labelsize=7.3)
    axis_sign.set_xscale("symlog", linthresh=0.002)
    max_abs = max(0.003, float(np.nanmax(np.abs(values))) * 1.25)
    axis_sign.set_xlim(-max_abs, max_abs)
    axis_sign.set_xlabel(r"Signed threshold shift $\Delta\alpha_c$")
    axis_sign.set_title("Several structural levers move the critical boundary")
    clean_axis(axis_sign)
    pass_count = sum(str(value).lower() == "true" for value in signed["sign_pass"])
    small_note(axis_sign, f"Sign pass: {pass_count}/{len(signed)}\nError bars: bootstrap CI", "upper right")
    panel_label(axis_sign, "A")

    strongest = signed.copy()
    strongest["abs_delta"] = strongest["delta_alpha_c"].abs()
    strongest = strongest.sort_values("abs_delta", ascending=False).head(6)
    strongest["name"] = strongest["scenario"].str.upper() + " / " + strongest["lever_label"]
    axis_range.barh(np.arange(len(strongest)), strongest["delta_alpha_c"], color=COLORS["rose"], alpha=0.85)
    axis_range.axvline(0.0, color=COLORS["ink"], lw=0.8)
    axis_range.set_yticks(np.arange(len(strongest)))
    axis_range.set_yticklabels(strongest["name"], fontsize=7)
    axis_range.invert_yaxis()
    axis_range.set_xlabel(r"$\Delta\alpha_c$")
    axis_range.set_title("Largest threshold shifts")
    clean_axis(axis_range)
    panel_label(axis_range, "B")

    fb = alpha_rows[(alpha_rows["scenario"] == "s1") & (alpha_rows["lever"] == "feedback_strength")].dropna(subset=["alpha_c"]).copy()
    fb["value_num"] = fb["lever_value"].map(_to_float)
    fb = fb.sort_values("value_num")
    axis_feedback.plot(fb["value_num"], fb["alpha_c"], marker="o", color=COLORS["violet"], lw=1.8)
    axis_feedback.set_xlabel("Feedback strength")
    axis_feedback.set_ylabel(r"$\alpha_c$")
    axis_feedback.set_title("A useful null: feedback alone")
    clean_axis(axis_feedback)
    small_note(axis_feedback, "Flat response warns\nagainst overclaiming", "upper right")
    panel_label(axis_feedback, "C")

    save_figure(
        figure,
        "figure_4_structural_threshold_shifts",
        "Figure 4: structural threshold shifts",
        [exp10_dir / "comparative_statics_summary.csv", exp10_dir / "alpha_c_by_variant.csv"],
        "RQ3 figure: threshold-shift comparative statics, with sign tests and an explicit null/caveat inset.",
        manifest,
    )


def figure_5_defense_threshold_shift(manifest: list[dict[str, Any]]) -> None:
    defense_dir = DEFENSE_BENCHMARK_OUTPUTS_ROOT / "exp7_threshold_shift_defense"
    curves = numeric(
        read_csv(defense_dir / "alpha_curves.csv"),
        ["n_society", "alpha", "collapse_rate_mean", "intervention_cost_mean", "utility_loss_mean", "false_positive_rate_mean"],
    )
    leaderboard = numeric(
        read_csv(defense_dir / "main_table.csv"),
        [
            "n_society", "alpha_c_noguard", "alpha_c_defense", "delta_alpha_c",
            "delta_alpha_c_raw_or_bound", "defense_score", "clean_utility_cost",
            "mean_intervention_cost", "worst_score",
        ],
    )

    figure = plt.figure(figsize=(9.0, 5.5), constrained_layout=True)
    gs = figure.add_gridspec(2, 3)
    axis_curves = figure.add_subplot(gs[:, 0])
    axis_bars = figure.add_subplot(gs[0, 1:])
    axis_tradeoff = figure.add_subplot(gs[1, 1])
    axis_table = figure.add_subplot(gs[1, 2])

    scenario = "s2"
    n_focus = 1000
    defenses = [
        defense for defense in ["noguard", "topology_aware", "distilled", "calibrated_distilled", "oracle"]
        if defense in set(curves["defense"].astype(str))
    ]
    defense_colors = {
        "noguard": COLORS["gray"],
        "rule": COLORS["violet"],
        "topology_aware": COLORS["rose"],
        "distilled": COLORS["magenta"],
        "calibrated_distilled": COLORS["pink"],
        "oracle": COLORS["plum"],
    }
    for defense in defenses:
        subset = curves[(curves["scenario"] == scenario) & (curves["n_society"] == n_focus) & (curves["defense"] == defense)]
        if subset.empty:
            continue
        summary = subset.sort_values("alpha")
        axis_curves.plot(summary["alpha"], summary["collapse_rate_mean"], marker="o", lw=1.8, ms=4.0, color=defense_colors.get(defense, COLORS["purple"]), label=defense)
        row = leaderboard[(leaderboard["scenario"] == scenario) & (leaderboard["n_society"] == n_focus) & (leaderboard["defense"] == defense)]
        if not row.empty and pd.notna(row.iloc[0].get("alpha_c_defense")):
            axis_curves.axvline(float(row.iloc[0]["alpha_c_defense"]), color=defense_colors.get(defense, COLORS["purple"]), ls=":", lw=1.1)
    axis_curves.axhline(0.5, color=COLORS["gray"], ls="--", lw=1.0)
    axis_curves.set_xscale("symlog", linthresh=1e-4)
    curve_rows = curves[(curves["scenario"] == scenario) & (curves["n_society"] == n_focus)]
    axis_curves.set_xlim(left=0.0, right=float(curve_rows["alpha"].max()) * 1.08)
    axis_curves.set_ylim(-0.04, 1.04)
    axis_curves.set_xlabel("alpha")
    axis_curves.set_ylabel("P(collapse)")
    axis_curves.set_title("Defense shifts the S2 boundary")
    axis_curves.legend(frameon=False, loc="lower right")
    clean_axis(axis_curves)
    small_note(axis_curves, "Positive example:\nthreshold moves right", "upper left")
    panel_label(axis_curves, "A")

    bar_rows = leaderboard[(leaderboard["defense"].isin(["topology_aware", "distilled", "calibrated_distilled", "oracle"]))].copy()
    bar_rows = bar_rows[bar_rows["n_society"] == n_focus]
    pivot = bar_rows.pivot_table(index="scenario", columns="defense", values="delta_alpha_c_raw_or_bound", aggfunc="mean").reindex(["s1", "s2"])
    x_positions = np.arange(len(pivot.index))
    bar_defs = [col for col in ["topology_aware", "distilled", "calibrated_distilled", "oracle"] if col in pivot.columns]
    width = min(0.8 / max(len(bar_defs), 1), 0.2)
    for i, defense in enumerate(bar_defs):
        axis_bars.bar(x_positions + (i - (len(bar_defs) - 1) / 2) * width, pivot[defense], width=width, color=defense_colors.get(defense, COLORS["purple"]), alpha=0.86, label=defense)
    axis_bars.axhline(0.0, color=COLORS["ink"], lw=0.9)
    axis_bars.set_xticks(x_positions)
    axis_bars.set_xticklabels([s.upper() for s in pivot.index])
    axis_bars.set_ylabel(r"Raw/bound $\Delta\alpha_c$ vs NoGuard")
    axis_bars.set_title("Threshold shift reveals hard scenarios")
    axis_bars.legend(frameon=False, ncol=2)
    clean_axis(axis_bars)
    panel_label(axis_bars, "B")

    trade = leaderboard[leaderboard["defense"].isin(["rule", "topology_aware", "distilled", "calibrated_distilled", "oracle"])].copy()
    trade = trade.dropna(subset=["mean_intervention_cost", "delta_alpha_c_raw_or_bound"])
    for defense, group in trade.groupby("defense"):
        axis_tradeoff.scatter(
            group["mean_intervention_cost"],
            group["delta_alpha_c_raw_or_bound"],
            s=42,
            color=defense_colors.get(defense, COLORS["purple"]),
            alpha=0.78,
            edgecolor="white",
            linewidth=0.5,
            label=defense,
        )
    axis_tradeoff.axhline(0.0, color=COLORS["gray"], lw=0.9, ls="--")
    axis_tradeoff.set_xlabel("Intervention cost")
    axis_tradeoff.set_ylabel(r"Raw/bound $\Delta\alpha_c$")
    axis_tradeoff.set_title("Cost-threshold tradeoff")
    clean_axis(axis_tradeoff)
    axis_tradeoff.legend(frameon=False, fontsize=6.8, loc="best")
    panel_label(axis_tradeoff, "C")

    axis_table.axis("off")
    display_rows = []
    for defense in ["topology_aware", "distilled", "calibrated_distilled", "oracle"]:
        rows = leaderboard[leaderboard["defense"] == defense]
        if rows.empty:
            continue
        display_rows.append(
            [
                defense,
                f"{float(rows['defense_score'].mean()):.2f}",
                f"{float(rows['delta_alpha_c_raw_or_bound'].mean()):.4f}",
                f"{float(rows['defense_score'].min()):.2f}",
            ]
        )
    table = axis_table.table(
        cellText=display_rows,
        colLabels=["Defense", "Score", "Mean dA", "Worst"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.05, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#E4D3EA")
        if row == 0:
            cell.set_facecolor("#F3D8F3")
            cell.set_text_props(weight="bold", color=COLORS["ink"])
        else:
            cell.set_facecolor("#FFF9FD" if row % 2 else "#FBEAF6")
    axis_table.set_title("Benchmark message")
    axis_table.text(
        0.0,
        0.08,
        "Non-oracle baselines improve S2\nbut still fail S1 threshold shift.",
        transform=axis_table.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.3,
        color=COLORS["muted"],
    )
    panel_label(axis_table, "D")

    save_figure(
        figure,
        "figure_5_defense_threshold_shift",
        "Figure 5: defense as threshold shift",
        [defense_dir / "alpha_curves.csv", defense_dir / "main_table.csv", defense_dir / "threshold_shift_summary.csv"],
        "RQ4 figure: threshold shift as the defense objective, with honest S1 failure and S2 improvement evidence.",
        manifest,
    )


def build_all() -> None:
    configure_style()
    manifest: list[dict[str, Any]] = []
    figure_1_at_a_glance(manifest)
    figure_2_scaling_law(manifest)
    figure_3_mechanism_heterogeneity(manifest)
    figure_4_structural_shifts(manifest)
    figure_5_defense_threshold_shift(manifest)
    write_manifest(manifest)
    print(f"Wrote {len(manifest)} paper-required figures to {OUT_DIR}")


if __name__ == "__main__":
    build_all()