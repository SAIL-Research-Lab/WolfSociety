"""Build paper-ready WolfBench figures from existing experiment outputs.

The script only reads CSV/JSON artifacts. It does not rerun WolfBench episodes.

Usage:
    python -m experiments.paper_figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import NullFormatter, NullLocator

from experiments._common import (
    DEFENSE_BENCHMARK_OUTPUTS_ROOT,
    OUTPUTS_ROOT,
    SCALING_THEORY_OUTPUTS_ROOT,
)


SCENARIO_LABELS = {"s1": "S1", "s2": "S2", "s3": "S3", "s4": "S4"}
SCENARIO_COLORS = {
    "s1": "#2A9D8F",
    "s2": "#E76F51",
    "s3": "#4E79A7",
    "s4": "#F2A541",
}
DEFENSE_COLORS = {
    "noguard": "#8E9AAF",
    "random": "#6BAED6",
    "rule": "#F4A261",
    "oracle": "#9B7EDE",
    "qwen": "#2A9D8F",
    "qwen_assisted": "#5ABF90",
    "distilled": "#E76F51",
}
CAPACITY_COLORS = {
    "per_agent_capacity": "#2A9D8F",
    "fixed_total_capacity": "#E76F51",
}
PLACEMENT_COLORS = {"random": "#8E9AAF", "high_degree": "#E76F51"}
FAMILY_LABELS = {
    "feedback_strength": "Feedback",
    "asset_liquidity_scale": "Liquidity",
    "retail_wealth_scale": "Retail wealth",
    "retail_risk_appetite": "Risk appetite",
    "social_mean_degree": "Mean degree",
    "placement": "Placement",
}
SIZE_PALETTE = ["#7BC8A4", "#4ECDC4", "#2A9D8F", "#4E79A7", "#7A77B9", "#B07AA1"]
COLLAPSE_CMAP = sns.color_palette("rocket_r", as_cmap=True)
DELTA_CMAP = sns.color_palette("mako_r", as_cmap=True)
DIVERGING_CMAP = sns.diverging_palette(18, 175, s=74, l=56, as_cmap=True)


def configure_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#E7EDF2",
            "grid.linewidth": 0.75,
            "axes.edgecolor": "#C9D3DC",
            "axes.labelcolor": "#243447",
            "xtick.color": "#243447",
            "ytick.color": "#243447",
            "text.color": "#1F2933",
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
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


def numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def run_status(directory: Path) -> str:
    metadata_path = directory / "run_metadata.json"
    if not metadata_path.exists():
        return "missing"
    try:
        return str(read_json(metadata_path).get("status", "unknown"))
    except json.JSONDecodeError:
        return "unknown"


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    manifest: list[dict[str, Any]],
    title: str,
    sources: list[Path],
    note: str,
    evidence: str | None = None,
) -> None:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=340)
    figure.savefig(pdf_path)
    plt.close(figure)
    manifest.append({
        "figure": stem,
        "title": title,
        "png": str(png_path.relative_to(OUTPUTS_ROOT.parent)),
        "pdf": str(pdf_path.relative_to(OUTPUTS_ROOT.parent)),
        "sources": [str(source.relative_to(OUTPUTS_ROOT.parent)) for source in sources],
        "note": note,
        "evidence": evidence or "",
    })


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        color="#1F2933",
        va="top",
        ha="left",
    )


def panel_note(axis: plt.Axes, text: str, location: str = "upper left") -> None:
    positions = {
        "upper left": (0.025, 0.965, "left", "top"),
        "upper right": (0.975, 0.965, "right", "top"),
        "lower left": (0.025, 0.035, "left", "bottom"),
        "lower right": (0.975, 0.035, "right", "bottom"),
        "above left": (0.0, 1.035, "left", "bottom"),
        "above right": (1.0, 1.035, "right", "bottom"),
    }
    x_value, y_value, horizontal, vertical = positions[location]
    axis.text(
        x_value,
        y_value,
        text,
        transform=axis.transAxes,
        ha=horizontal,
        va=vertical,
        fontsize=7.2,
        color="#52616F",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#F8FBFD",
            "edgecolor": "#D7E2EA",
            "linewidth": 0.6,
            "alpha": 0.92,
        },
        clip_on=False,
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, color="#E7EDF2", linewidth=0.75)


def set_log_size_axis(axis: plt.Axes, values: list[int] | np.ndarray | pd.Series) -> None:
    ticks = sorted({int(value) for value in values if pd.notna(value) and int(value) > 0})
    axis.set_xscale("log")
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(tick) for tick in ticks])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_minor_formatter(NullFormatter())


def set_alpha_axis(axis: plt.Axes, max_alpha: float, linthresh: float = 1e-3) -> None:
    axis.set_xscale("symlog", linthresh=linthresh)
    axis.set_xlim(left=0.0, right=float(max_alpha) * 1.06)


def format_alpha(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    if value < 0.001:
        return f"{value:.5f}".rstrip("0")
    if value < 0.01:
        return f"{value:.4f}".rstrip("0")
    return f"{value:.3g}"


def format_count(value: int | float) -> str:
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def format_small_metric(value: Any, digits: int = 4) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric_value):
        return ""
    if abs(numeric_value) < 5e-7:
        return "0"
    return f"{numeric_value:.{digits}f}".rstrip("0").rstrip(".")


def count_unique(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].dropna().nunique())


def episode_count(frame: pd.DataFrame) -> int:
    if "n" in frame.columns:
        return int(pd.to_numeric(frame["n"], errors="coerce").fillna(0).sum())
    return int(len(frame))


def coverage_text(frame: pd.DataFrame, *, dimensions: list[str], count_column: str | None = None) -> str:
    episodes = int(pd.to_numeric(frame[count_column], errors="coerce").fillna(0).sum()) if count_column else episode_count(frame)
    dimension_bits = []
    for dimension in dimensions:
        if dimension in frame.columns:
            dimension_bits.append(f"{count_unique(frame, dimension)} {dimension}")
    return f"{format_count(episodes)} episodes; " + ", ".join(dimension_bits)


def wilson_interval(values: pd.Series) -> tuple[float, float, float, int]:
    clean_values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    count = int(clean_values.size)
    if count == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean_value = float(clean_values.mean())
    z_score = 1.959963984540054
    denominator = 1.0 + z_score**2 / count
    center = (mean_value + z_score**2 / (2.0 * count)) / denominator
    spread = z_score * np.sqrt((mean_value * (1.0 - mean_value) / count) + z_score**2 / (4.0 * count**2)) / denominator
    return mean_value, max(0.0, center - spread), min(1.0, center + spread), count


def collapse_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group_values, group_frame in frame.groupby(group_columns, dropna=False):
        group_tuple = group_values if isinstance(group_values, tuple) else (group_values,)
        record = {column: group_tuple[position] for position, column in enumerate(group_columns)}
        mean_value, ci_low, ci_high, count = wilson_interval(group_frame["collapse_rate"])
        record.update({"mean": mean_value, "ci_low": ci_low, "ci_high": ci_high, "count": count})
        records.append(record)
    return pd.DataFrame(records)


def errorbar_from_ci(frame: pd.DataFrame, value_column: str, low_column: str, high_column: str) -> np.ndarray:
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame[low_column], errors="coerce").fillna(pd.Series(values)).to_numpy(dtype=float)
    highs = pd.to_numeric(frame[high_column], errors="coerce").fillna(pd.Series(values)).to_numpy(dtype=float)
    return np.vstack([np.maximum(0.0, values - lows), np.maximum(0.0, highs - values)])


def figure_scaling_transition(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    exp1_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp1_alpha_scaling"
    exp2_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp2_society_size_scaling"
    exp1_curve = numeric_columns(
        read_csv(exp1_dir / "collapse_rate_wilson_ci.csv"),
        ["n_society", "alpha", "mean", "n", "ci_low", "ci_high"],
    )
    exp1_threshold_path = exp1_dir / "alpha_critical_summary.csv"
    exp1_thresholds = None
    if exp1_threshold_path.exists():
        exp1_thresholds = numeric_columns(
            read_csv(exp1_threshold_path),
            ["n_society", "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high", "transition_width_10_90"],
        )
    exp2_thresholds = numeric_columns(
        read_csv(exp2_dir / "alpha_critical_summary.csv"),
        ["n_society", "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high", "transition_width_10_90"],
    )
    exp2_curve = numeric_columns(
        read_csv(exp2_dir / "collapse_rate_wilson_ci.csv"),
        ["n_society", "alpha", "mean", "n"],
    )
    exp1_episodes = episode_count(exp1_curve)
    exp2_episodes = episode_count(exp2_curve)

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    axis_transition, axis_threshold = axes[0]
    axis_heatmap, axis_width = axes[1]

    n_values = sorted(exp1_curve["n_society"].dropna().astype(int).unique())
    size_colors = {size_value: SIZE_PALETTE[position % len(SIZE_PALETTE)] for position, size_value in enumerate(n_values)}
    for size_value in n_values:
        subset = exp1_curve[exp1_curve["n_society"] == size_value].sort_values("alpha")
        axis_transition.plot(
            subset["alpha"],
            subset["mean"],
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            color=size_colors[size_value],
            label=f"N={size_value}",
        )
        axis_transition.fill_between(
            subset["alpha"].to_numpy(dtype=float),
            subset["ci_low"].to_numpy(dtype=float),
            subset["ci_high"].to_numpy(dtype=float),
            color=size_colors[size_value],
            alpha=0.15,
            linewidth=0,
        )
        if exp1_thresholds is not None:
            threshold_rows = exp1_thresholds[exp1_thresholds["n_society"] == size_value]
            if not threshold_rows.empty:
                alpha_c = threshold_rows.iloc[0].get("alpha_c_logistic")
                if pd.notna(alpha_c):
                    axis_transition.axvline(
                        float(alpha_c),
                        color=size_colors[size_value],
                        linestyle=":",
                        linewidth=1.1,
                        alpha=0.7,
                    )
    axis_transition.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.0)
    set_alpha_axis(axis_transition, float(exp1_curve["alpha"].max()), linthresh=1e-3)
    axis_transition.set_ylim(-0.04, 1.04)
    axis_transition.set_xlabel("Harmful-agent ratio alpha")
    axis_transition.set_ylabel("P(collapse)")
    axis_transition.set_title("Collapse transition at fixed N")
    axis_transition.legend(frameon=False, loc="lower right")
    clean_axis(axis_transition)
    panel_note(
        axis_transition,
        f"{format_count(exp1_episodes)} episodes\n{len(n_values)} N x {count_unique(exp1_curve, 'alpha')} alpha",
        "upper left",
    )
    panel_label(axis_transition, "A")

    threshold_subset = exp2_thresholds.dropna(subset=["alpha_c_logistic"]).sort_values("n_society")
    society_sizes = threshold_subset["n_society"].to_numpy(dtype=float)
    alpha_critical = threshold_subset["alpha_c_logistic"].to_numpy(dtype=float)
    axis_threshold.errorbar(
        society_sizes,
        alpha_critical,
        yerr=errorbar_from_ci(threshold_subset, "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high"),
        fmt="o",
        markersize=5,
        capsize=3,
        color="#2A9D8F",
        ecolor="#93D5C6",
        label="logistic alpha_c",
    )
    if society_sizes.size >= 2:
        slope, intercept = np.polyfit(np.log(society_sizes), np.log(alpha_critical), 1)
        fit_sizes = np.logspace(np.log10(float(society_sizes.min())), np.log10(float(society_sizes.max())), 120)
        fit_values = np.exp(intercept) * fit_sizes**slope
        axis_threshold.plot(fit_sizes, fit_values, linestyle="--", color="#E76F51", linewidth=1.5, label=f"power fit beta={slope:.2f}")
    set_log_size_axis(axis_threshold, society_sizes)
    axis_threshold.set_xlabel("Society size N")
    axis_threshold.set_ylabel("Critical ratio alpha_c")
    axis_threshold.set_title("Finite-size critical point")
    axis_threshold.legend(frameon=False, loc="best")
    clean_axis(axis_threshold)
    panel_note(
        axis_threshold,
        f"{len(threshold_subset)} fitted thresholds\nfrom {format_count(exp2_episodes)} episodes",
        "upper right",
    )
    panel_label(axis_threshold, "B")

    heatmap_frame = exp2_curve.pivot_table(index="n_society", columns="alpha", values="mean", aggfunc="mean").sort_index()
    sns.heatmap(
        heatmap_frame,
        ax=axis_heatmap,
        cmap=COLLAPSE_CMAP,
        vmin=0,
        vmax=1,
        linewidths=0.35,
        linecolor="#FFFFFF",
        cbar_kws={"label": "P(collapse)", "shrink": 0.82},
    )
    axis_heatmap.set_xlabel("alpha")
    axis_heatmap.set_ylabel("Society size N")
    axis_heatmap.set_title("Critical-regime heatmap")
    axis_heatmap.set_xticklabels([format_alpha(float(value)) for value in heatmap_frame.columns], rotation=45, ha="right")
    axis_heatmap.set_yticklabels([str(int(value)) for value in heatmap_frame.index], rotation=0)
    panel_label(axis_heatmap, "C")

    axis_width.plot(
        threshold_subset["n_society"],
        threshold_subset["transition_width_10_90"],
        marker="o",
        linewidth=1.9,
        color="#4E79A7",
    )
    set_log_size_axis(axis_width, threshold_subset["n_society"])
    axis_width.set_xlabel("Society size N")
    axis_width.set_ylabel("Transition width")
    axis_width.set_title("Transition sharpening")
    clean_axis(axis_width)
    panel_note(axis_width, "Derived from\nlogistic fits", "upper right")
    panel_label(axis_width, "D")

    save_figure(
        figure,
        output_dir,
        "figure_1_scaling_transition",
        manifest,
        "Scaling transition and finite-size critical point",
        [
            exp1_dir / "collapse_rate_wilson_ci.csv",
            exp1_threshold_path if exp1_threshold_path.exists() else exp1_dir / "summary.json",
            exp2_dir / "alpha_critical_summary.csv",
            exp2_dir / "collapse_rate_wilson_ci.csv",
        ],
        "Main theory figure: nonlinear collapse transition, estimated alpha_c(N), heatmap, and width scaling.",
        f"Exp1={format_count(exp1_episodes)} episodes; Exp2={format_count(exp2_episodes)} episodes compressed into {len(threshold_subset)} alpha_c estimates.",
    )


def figure_mechanisms_robustness(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    exp3_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp3_centrality_placement"
    exp4_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp4_feedback_ablation"
    exp7_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp7_cross_mechanism_threshold"
    exp8_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp8_sensitivity_audit"
    exp7_thresholds = numeric_columns(
        read_csv(exp7_dir / "alpha_critical_by_mechanism.csv"),
        ["n_society", "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high"],
    )
    exp7_curve = numeric_columns(
        read_csv(exp7_dir / "collapse_rate_wilson_ci.csv"),
        ["scenario", "n_society", "alpha", "n"],
    )
    exp3_rows = numeric_columns(read_csv(exp3_dir / "data.csv"), ["n_society", "collapse_rate"])
    exp4_rows = numeric_columns(read_csv(exp4_dir / "data.csv"), ["feedback_strength", "collapse_rate"])
    exp8_delta = numeric_columns(read_csv(exp8_dir / "sensitivity_delta_summary.csv"), ["delta_alpha_c"])
    exp8_data = read_csv(exp8_dir / "data.csv")

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    axis_mechanisms, axis_centrality = axes[0]
    axis_feedback, axis_sensitivity = axes[1]

    for scenario in ["s1", "s2", "s3", "s4"]:
        scenario_rows = exp7_thresholds[exp7_thresholds["scenario"] == scenario].dropna(subset=["alpha_c_logistic"]).sort_values("n_society")
        if scenario_rows.empty:
            continue
        axis_mechanisms.errorbar(
            scenario_rows["n_society"],
            scenario_rows["alpha_c_logistic"],
            yerr=errorbar_from_ci(scenario_rows, "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high"),
            marker="o",
            linewidth=1.6,
            markersize=4.5,
            capsize=3,
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
        )
    set_log_size_axis(axis_mechanisms, exp7_thresholds["n_society"])
    axis_mechanisms.set_yscale("log")
    axis_mechanisms.set_xlabel("Society size N")
    axis_mechanisms.set_ylabel("Critical ratio alpha_c")
    axis_mechanisms.set_title("Critical regimes across scenarios")
    axis_mechanisms.legend(frameon=False, ncol=2, loc="best")
    clean_axis(axis_mechanisms)
    panel_note(
        axis_mechanisms,
        f"{len(exp7_thresholds)} thresholds\n{format_count(episode_count(exp7_curve))} episodes",
        "upper right",
    )
    panel_label(axis_mechanisms, "A")

    placement_summary = collapse_summary(exp3_rows, ["n_society", "placement"])
    placement_order = [placement for placement in ["random", "high_degree"] if placement in set(placement_summary["placement"])]
    centrality_sizes = sorted(placement_summary["n_society"].dropna().astype(int).unique())
    base_positions = np.arange(len(centrality_sizes))
    offsets = np.linspace(-0.13, 0.13, max(len(placement_order), 1))
    for position, placement in enumerate(placement_order):
        subset = placement_summary[placement_summary["placement"] == placement].set_index("n_society").reindex(centrality_sizes)
        means = subset["mean"].to_numpy(dtype=float)
        ci_low = subset["ci_low"].to_numpy(dtype=float)
        ci_high = subset["ci_high"].to_numpy(dtype=float)
        axis_centrality.errorbar(
            base_positions + offsets[position],
            means,
            yerr=np.vstack([means - ci_low, ci_high - means]),
            fmt="o",
            markersize=5,
            capsize=3,
            color=PLACEMENT_COLORS.get(placement, "#4E79A7"),
            label=placement.replace("_", " "),
        )
    axis_centrality.set_xticks(base_positions)
    axis_centrality.set_xticklabels([f"N={size_value}" for size_value in centrality_sizes])
    axis_centrality.set_ylim(-0.04, 1.04)
    axis_centrality.set_ylabel("P(collapse)")
    axis_centrality.set_title("Centrality placement effect")
    axis_centrality.legend(frameon=False, loc="best")
    clean_axis(axis_centrality)
    panel_note(
        axis_centrality,
        f"{format_count(len(exp3_rows))} episodes\n{len(centrality_sizes)} N x {len(placement_order)} placements",
        "upper left",
    )
    panel_label(axis_centrality, "B")

    feedback_summary = collapse_summary(exp4_rows, ["feedback_strength"]).sort_values("feedback_strength")
    axis_feedback.plot(
        feedback_summary["feedback_strength"],
        feedback_summary["mean"],
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        color="#2A9D8F",
    )
    axis_feedback.fill_between(
        feedback_summary["feedback_strength"].to_numpy(dtype=float),
        feedback_summary["ci_low"].to_numpy(dtype=float),
        feedback_summary["ci_high"].to_numpy(dtype=float),
        color="#2A9D8F",
        alpha=0.16,
        linewidth=0,
    )
    axis_feedback.set_ylim(-0.04, 1.04)
    axis_feedback.set_xlabel("Feedback strength")
    axis_feedback.set_ylabel("P(collapse)")
    axis_feedback.set_title("Near-threshold feedback ablation")
    clean_axis(axis_feedback)
    panel_note(
        axis_feedback,
        f"{format_count(len(exp4_rows))} episodes\n{count_unique(exp4_rows, 'feedback_strength')} feedback values",
        "upper left",
    )
    panel_label(axis_feedback, "C")

    family_order = [family for family in FAMILY_LABELS if family in set(exp8_delta["family"])]
    scenario_order = [scenario for scenario in ["s1", "s2", "s3", "s4"] if scenario in set(exp8_delta["scenario"])]
    sensitivity_matrix = exp8_delta.pivot_table(index="scenario", columns="family", values="delta_alpha_c", aggfunc="mean").reindex(index=scenario_order, columns=family_order)
    sns.heatmap(
        sensitivity_matrix,
        ax=axis_sensitivity,
        cmap=DELTA_CMAP,
        vmin=0,
        vmax=max(0.01, float(np.nanmax(sensitivity_matrix.to_numpy(dtype=float)))),
        annot=True,
        fmt=".2f",
        linewidths=0.35,
        linecolor="#FFFFFF",
        cbar_kws={"label": "Delta alpha_c", "shrink": 0.82},
    )
    axis_sensitivity.set_xlabel("Parameter family")
    axis_sensitivity.set_ylabel("Scenario")
    axis_sensitivity.set_title("Sensitivity audit")
    axis_sensitivity.set_xticklabels([FAMILY_LABELS.get(family, family) for family in family_order], rotation=32, ha="right")
    axis_sensitivity.set_yticklabels([SCENARIO_LABELS.get(scenario, scenario.upper()) for scenario in scenario_order], rotation=0)
    panel_label(axis_sensitivity, "D")

    save_figure(
        figure,
        output_dir,
        "figure_2_mechanisms_robustness",
        manifest,
        "Mechanism breadth and robustness checks",
        [exp7_dir / "alpha_critical_by_mechanism.csv", exp3_dir / "data.csv", exp4_dir / "data.csv", exp8_dir / "sensitivity_delta_summary.csv"],
        "Combines cross-scenario thresholds, centrality placement, feedback ablation, and sensitivity audit.",
        f"Exp7={format_count(episode_count(exp7_curve))} episodes; Exp3={format_count(len(exp3_rows))}; Exp4={format_count(len(exp4_rows))}; Exp8={format_count(len(exp8_data))}.",
    )


def figure_capacity_llm(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    exp2_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp2_society_size_scaling"
    exp5_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp5_capacity_control"
    exp6_dir = SCALING_THEORY_OUTPUTS_ROOT / "exp6_llm_n200_scaling"
    exp2_thresholds = numeric_columns(read_csv(exp2_dir / "alpha_critical_summary.csv"), ["n_society", "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high"])
    capacity_thresholds = numeric_columns(
        read_csv(exp5_dir / "alpha_critical_capacity_summary.csv"),
        ["n_society", "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high"],
    )
    capacity_rows = numeric_columns(read_csv(exp5_dir / "data.csv"), ["n_society", "alpha", "collapse_rate"])
    llm_rows = numeric_columns(read_csv(exp6_dir / "data.csv"), ["alpha", "collapse_rate"])
    llm_threshold = numeric_columns(read_csv(exp6_dir / "alpha_critical_summary.csv"), ["n_society", "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high"])

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    axis_capacity_threshold, axis_capacity_curve = axes[0]
    axis_llm_curve, axis_llm_compare = axes[1]

    for mode in ["per_agent_capacity", "fixed_total_capacity"]:
        subset = capacity_thresholds[capacity_thresholds["capacity_mode"] == mode].dropna(subset=["alpha_c_logistic"]).sort_values("n_society")
        if subset.empty:
            continue
        axis_capacity_threshold.errorbar(
            subset["n_society"],
            subset["alpha_c_logistic"],
            yerr=errorbar_from_ci(subset, "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high"),
            marker="o",
            linewidth=1.7,
            markersize=4.5,
            capsize=3,
            color=CAPACITY_COLORS[mode],
            label=mode.replace("_", " "),
        )
    set_log_size_axis(axis_capacity_threshold, capacity_thresholds["n_society"])
    axis_capacity_threshold.set_xlabel("Society size N")
    axis_capacity_threshold.set_ylabel("Critical ratio alpha_c")
    axis_capacity_threshold.set_title("Capacity-control threshold check")
    axis_capacity_threshold.legend(frameon=False, loc="best")
    clean_axis(axis_capacity_threshold)
    panel_note(
        axis_capacity_threshold,
        f"{format_count(len(capacity_rows))} episodes\n2 capacity modes",
        "upper right",
    )
    panel_label(axis_capacity_threshold, "A")

    capacity_n = 1000 if 1000 in set(capacity_rows["n_society"].dropna().astype(int)) else int(capacity_rows["n_society"].dropna().astype(int).median())
    capacity_curve = collapse_summary(capacity_rows[capacity_rows["n_society"].astype(int) == capacity_n], ["label", "alpha"]).sort_values("alpha")
    for mode in ["per_agent_capacity", "fixed_total_capacity"]:
        subset = capacity_curve[capacity_curve["label"] == mode].sort_values("alpha")
        if subset.empty:
            continue
        axis_capacity_curve.plot(
            subset["alpha"],
            subset["mean"],
            marker="o",
            linewidth=1.7,
            markersize=4.2,
            color=CAPACITY_COLORS[mode],
            label=mode.replace("_", " "),
        )
        axis_capacity_curve.fill_between(
            subset["alpha"].to_numpy(dtype=float),
            subset["ci_low"].to_numpy(dtype=float),
            subset["ci_high"].to_numpy(dtype=float),
            color=CAPACITY_COLORS[mode],
            alpha=0.14,
            linewidth=0,
        )
    axis_capacity_curve.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.0)
    axis_capacity_curve.set_ylim(-0.04, 1.04)
    axis_capacity_curve.set_xlabel("Harmful-agent ratio alpha")
    axis_capacity_curve.set_ylabel("P(collapse)")
    axis_capacity_curve.set_title(f"Capacity curves at N={capacity_n}")
    axis_capacity_curve.legend(frameon=False, loc="lower right")
    clean_axis(axis_capacity_curve)
    panel_note(
        axis_capacity_curve,
        f"{format_count(len(capacity_rows[capacity_rows['n_society'].astype(int) == capacity_n]))} episodes\nshown at N={capacity_n}",
        "upper left",
    )
    panel_label(axis_capacity_curve, "B")

    llm_curve = collapse_summary(llm_rows, ["alpha"]).sort_values("alpha")
    axis_llm_curve.plot(
        llm_curve["alpha"],
        llm_curve["mean"],
        marker="o",
        linewidth=1.8,
        markersize=4.2,
        color="#9B7EDE",
        label="single LLM leader",
    )
    axis_llm_curve.fill_between(
        llm_curve["alpha"].to_numpy(dtype=float),
        llm_curve["ci_low"].to_numpy(dtype=float),
        llm_curve["ci_high"].to_numpy(dtype=float),
        color="#9B7EDE",
        alpha=0.16,
        linewidth=0,
    )
    llm_alpha_critical = float(llm_threshold.iloc[0]["alpha_c_logistic"])
    if np.isfinite(llm_alpha_critical):
        axis_llm_curve.axvline(llm_alpha_critical, color="#9B7EDE", linestyle=":", linewidth=1.4)
    axis_llm_curve.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.0)
    set_alpha_axis(axis_llm_curve, float(llm_curve["alpha"].max()), linthresh=1e-3)
    axis_llm_curve.set_ylim(-0.04, 1.04)
    axis_llm_curve.set_xlabel("Harmful-agent ratio alpha")
    axis_llm_curve.set_ylabel("P(collapse)")
    axis_llm_curve.set_title("LLM strategist collapse curve")
    axis_llm_curve.legend(frameon=False, loc="lower right")
    clean_axis(axis_llm_curve)
    panel_note(
        axis_llm_curve,
        f"{format_count(len(llm_rows))} LLM episodes\n{count_unique(llm_rows, 'alpha')} alpha values",
        "upper left",
    )
    panel_label(axis_llm_curve, "C")

    llm_n_society = int(llm_threshold.iloc[0]["n_society"])
    non_llm_row = exp2_thresholds[exp2_thresholds["n_society"].astype(int) == llm_n_society]
    compare_rows = []
    if not non_llm_row.empty:
        compare_rows.append({"condition": "rule-based", **non_llm_row.iloc[0].to_dict()})
    compare_rows.append({"condition": "single LLM leader", **llm_threshold.iloc[0].to_dict()})
    compare_frame = pd.DataFrame(compare_rows)
    compare_positions = np.arange(len(compare_frame))
    axis_llm_compare.errorbar(
        compare_positions,
        compare_frame["alpha_c_logistic"],
        yerr=errorbar_from_ci(compare_frame, "alpha_c_logistic", "alpha_c_ci_low", "alpha_c_ci_high"),
        fmt="o",
        markersize=7,
        capsize=4,
        color="#2A9D8F",
        ecolor="#93D5C6",
    )
    axis_llm_compare.set_xticks(compare_positions)
    axis_llm_compare.set_xticklabels(compare_frame["condition"], rotation=18, ha="right")
    axis_llm_compare.set_ylabel("Critical ratio alpha_c")
    axis_llm_compare.set_title(f"Threshold comparison at N={llm_n_society}")
    clean_axis(axis_llm_compare)
    panel_note(axis_llm_compare, "CI from bootstrap\nthreshold fits", "upper right")
    panel_label(axis_llm_compare, "D")

    save_figure(
        figure,
        output_dir,
        "figure_3_capacity_llm_checks",
        manifest,
        "Capacity-control and LLM scaling checks",
        [exp5_dir / "alpha_critical_capacity_summary.csv", exp5_dir / "data.csv", exp6_dir / "data.csv", exp6_dir / "alpha_critical_summary.csv"],
        "Shows that capacity controls and a single LLM strategist can be reported without adding separate standalone figures.",
        f"Capacity control={format_count(len(capacity_rows))} episodes; LLM scaling={format_count(len(llm_rows))} episodes.",
    )


def display_defense_label(defense_name: str) -> str:
    return defense_name.replace("_", " ")


def defense_order(overall_frame: pd.DataFrame) -> list[str]:
    ordered_frame = overall_frame.copy()
    ordered_frame["Avg DefenseScore"] = pd.to_numeric(ordered_frame["Avg DefenseScore"], errors="coerce")
    return ordered_frame.sort_values("Avg DefenseScore", ascending=False)["Defense model"].astype(str).tolist()


def figure_defense_leaderboard(output_dir: Path, manifest: list[dict[str, Any]], defense_out_name: str, stem: str, title_suffix: str) -> None:
    defense_dir = DEFENSE_BENCHMARK_OUTPUTS_ROOT / defense_out_name
    scenario_path = defense_dir / "leaderboard_by_scenario.csv"
    if not scenario_path.exists():
        scenario_path = defense_dir / "leaderboard.csv"
    scenario_rows = numeric_columns(
        read_csv(scenario_path),
        ["n_society", "defense_score", "threshold_shift", "def_collapse_rate", "utility_loss", "false_positive_rate", "official_score"],
    )
    overall_rows = numeric_columns(read_csv(defense_dir / "leaderboard_overall.csv"), ["Avg DefenseScore", "Avg ThresholdShift", "Worst Score"])
    summary = read_json(defense_dir / "summary.json")
    data_rows = read_csv(defense_dir / "data.csv")
    overall_metrics = pd.DataFrame(summary.get("overall", []))
    if not overall_metrics.empty:
        numeric_columns(overall_metrics, ["defense_score_mean", "defense_score_ci_low", "defense_score_ci_high"])
    ordered_defenses = defense_order(overall_rows)
    scenario_order = [scenario for scenario in ["s1", "s2", "s3", "s4"] if scenario in set(scenario_rows.get("scenario", pd.Series(dtype=str)).astype(str))]

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    axis_overall, axis_score_heatmap = axes[0]
    axis_shift_heatmap, axis_tradeoff = axes[1]

    bar_frame = overall_rows.set_index("Defense model").reindex(ordered_defenses).reset_index()
    bar_frame = bar_frame.sort_values("Avg DefenseScore", ascending=True)
    bar_colors = [DEFENSE_COLORS.get(str(defense_name), "#4E79A7") for defense_name in bar_frame["Defense model"]]
    axis_overall.barh(
        [display_defense_label(str(defense_name)) for defense_name in bar_frame["Defense model"]],
        bar_frame["Avg DefenseScore"],
        color=bar_colors,
        alpha=0.92,
    )
    if not overall_metrics.empty:
        metrics_by_defense = overall_metrics.set_index("defense")
        for row_position, row in bar_frame.reset_index(drop=True).iterrows():
            defense_name = str(row["Defense model"])
            if defense_name not in metrics_by_defense.index:
                continue
            metric_row = metrics_by_defense.loc[defense_name]
            mean_value = float(row["Avg DefenseScore"])
            low_value = float(metric_row.get("defense_score_ci_low", mean_value))
            high_value = float(metric_row.get("defense_score_ci_high", mean_value))
            axis_overall.errorbar(
                mean_value,
                row_position,
                xerr=np.array([[max(0.0, mean_value - low_value)], [max(0.0, high_value - mean_value)]]),
                fmt="none",
                ecolor="#243447",
                elinewidth=0.8,
                capsize=2,
            )
    axis_overall.axvline(0, color="#374151", linewidth=0.8)
    axis_overall.set_xlabel("Avg DefenseScore")
    axis_overall.set_title("Overall leaderboard")
    clean_axis(axis_overall)
    panel_note(
        axis_overall,
        f"{format_count(len(data_rows))} episodes\n{count_unique(data_rows, 'defense')} defenses",
        "upper left",
    )
    panel_label(axis_overall, "A")

    scenario_mean = scenario_rows.groupby(["defense", "scenario"], as_index=False)["defense_score"].mean()
    score_matrix = scenario_mean.pivot_table(index="defense", columns="scenario", values="defense_score", aggfunc="mean").reindex(index=ordered_defenses, columns=scenario_order)
    sns.heatmap(
        score_matrix,
        ax=axis_score_heatmap,
        cmap=DIVERGING_CMAP,
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.35,
        linecolor="#FFFFFF",
        cbar_kws={"label": "DefenseScore", "shrink": 0.82},
    )
    axis_score_heatmap.set_xlabel("Scenario")
    axis_score_heatmap.set_ylabel("Defense")
    axis_score_heatmap.set_title("Scenario-level score")
    axis_score_heatmap.set_xticklabels([SCENARIO_LABELS.get(scenario, scenario.upper()) for scenario in scenario_order], rotation=0)
    axis_score_heatmap.set_yticklabels([display_defense_label(str(defense_name)) for defense_name in ordered_defenses], rotation=0)
    panel_label(axis_score_heatmap, "B")

    shift_mean = scenario_rows.groupby(["defense", "scenario"], as_index=False)["threshold_shift"].mean()
    shift_matrix = shift_mean.pivot_table(index="defense", columns="scenario", values="threshold_shift", aggfunc="mean").reindex(index=ordered_defenses, columns=scenario_order)
    shift_annotations = shift_matrix.map(format_small_metric)
    sns.heatmap(
        shift_matrix,
        ax=axis_shift_heatmap,
        cmap=DIVERGING_CMAP,
        center=0,
        annot=shift_annotations,
        fmt="",
        linewidths=0.35,
        linecolor="#FFFFFF",
        cbar_kws={"label": "Delta alpha_c", "shrink": 0.82},
    )
    axis_shift_heatmap.set_xlabel("Scenario")
    axis_shift_heatmap.set_ylabel("Defense")
    axis_shift_heatmap.set_title("Critical-threshold shift")
    axis_shift_heatmap.set_xticklabels([SCENARIO_LABELS.get(scenario, scenario.upper()) for scenario in scenario_order], rotation=0)
    axis_shift_heatmap.set_yticklabels([display_defense_label(str(defense_name)) for defense_name in ordered_defenses], rotation=0)
    panel_label(axis_shift_heatmap, "C")

    tradeoff_frame = scenario_rows.groupby("defense", as_index=False).agg(
        defense_score=("defense_score", "mean"),
        def_collapse_rate=("def_collapse_rate", "mean"),
        utility_loss=("utility_loss", "mean"),
        false_positive_rate=("false_positive_rate", "mean"),
    )
    for row in tradeoff_frame.itertuples(index=False):
        defense_name = str(row.defense)
        marker_size = 70.0 + 360.0 * max(0.0, float(row.false_positive_rate))
        axis_tradeoff.scatter(
            float(row.utility_loss),
            float(row.def_collapse_rate),
            s=marker_size,
            color=DEFENSE_COLORS.get(defense_name, "#4E79A7"),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
        )
        axis_tradeoff.annotate(
            display_defense_label(defense_name),
            (float(row.utility_loss), float(row.def_collapse_rate)),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    max_utility = float(tradeoff_frame["utility_loss"].max()) if not tradeoff_frame.empty else 1.0
    min_utility = float(tradeoff_frame["utility_loss"].min()) if not tradeoff_frame.empty else 0.0
    utility_padding = max(0.5, 0.22 * (max_utility - min_utility + 1e-9))
    axis_tradeoff.set_xlim(left=max(0.0, min_utility - utility_padding), right=max_utility + utility_padding)
    axis_tradeoff.set_xlabel("Utility / intervention cost")
    axis_tradeoff.set_ylabel("Post-defense P(collapse)")
    axis_tradeoff.set_title("Safety-cost tradeoff")
    clean_axis(axis_tradeoff)
    panel_note(axis_tradeoff, "Point size tracks\nfalse positives", "upper right")
    panel_label(axis_tradeoff, "D")

    save_figure(
        figure,
        output_dir,
        stem,
        manifest,
        f"Defense leaderboard{title_suffix}",
        [scenario_path, defense_dir / "leaderboard_overall.csv", defense_dir / "summary.json", defense_dir / "data.csv"],
        "Combines overall rank, per-scenario score, threshold shift, and cost/collapse tradeoff.",
        f"Leaderboard aggregates {format_count(len(data_rows))} episodes across {count_unique(data_rows, 'scenario_id')} scenarios, {count_unique(data_rows, 'n_society')} N values, {count_unique(data_rows, 'alpha')} alpha values, and {count_unique(data_rows, 'defense')} defenses.",
    )


def figure_defense_threshold_sweep(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    exp5_dir = DEFENSE_BENCHMARK_OUTPUTS_ROOT / "exp5_wolfguard_defense"
    curve_rows = numeric_columns(
        read_csv(exp5_dir / "alpha_curves.csv"),
        ["alpha", "collapse_rate_mean", "collapse_rate_ci_low", "collapse_rate_ci_high", "n_society"],
    )
    shift_rows = numeric_columns(read_csv(exp5_dir / "threshold_shift_summary.csv"), ["threshold_shift", "defense_score"])
    scenarios = [scenario for scenario in ["s1", "s2", "s3", "s4"] if scenario in set(curve_rows["scenario"])]

    figure = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    curve_axes = [figure.add_subplot(grid[0, column_position]) for column_position in range(2)]
    axis_shift = figure.add_subplot(grid[1, :])

    for axis_position, axis in enumerate(curve_axes):
        if axis_position >= len(scenarios):
            axis.axis("off")
            continue
        scenario = scenarios[axis_position]
        scenario_rows = curve_rows[curve_rows["scenario"] == scenario]
        for defense_name in sorted(scenario_rows["defense"].unique()):
            subset = scenario_rows[scenario_rows["defense"] == defense_name].sort_values("alpha")
            axis.plot(
                subset["alpha"],
                subset["collapse_rate_mean"],
                marker="o",
                linewidth=1.5,
                markersize=3.8,
                color=DEFENSE_COLORS.get(defense_name, "#4E79A7"),
                label=display_defense_label(defense_name),
            )
            axis.fill_between(
                subset["alpha"].to_numpy(dtype=float),
                subset["collapse_rate_ci_low"].to_numpy(dtype=float),
                subset["collapse_rate_ci_high"].to_numpy(dtype=float),
                color=DEFENSE_COLORS.get(defense_name, "#4E79A7"),
                alpha=0.12,
                linewidth=0,
            )
        axis.axhline(0.5, color="#6B7280", linestyle="--", linewidth=0.9)
        set_alpha_axis(axis, float(scenario_rows["alpha"].max()), linthresh=1e-4)
        axis.set_ylim(-0.04, 1.04)
        axis.set_title(f"{SCENARIO_LABELS.get(scenario, scenario.upper())} collapse curves")
        axis.set_xlabel("Harmful-agent ratio alpha")
        axis.set_ylabel("P(collapse)")
        axis.legend(frameon=False, loc="best")
        clean_axis(axis)
        panel_label(axis, chr(ord("A") + axis_position))

    plot_rows = shift_rows[shift_rows["defense"] != "noguard"].copy()
    sns.barplot(
        data=plot_rows,
        ax=axis_shift,
        x="scenario",
        y="threshold_shift",
        hue="defense",
        palette={defense_name: DEFENSE_COLORS.get(defense_name, "#4E79A7") for defense_name in plot_rows["defense"].unique()},
    )
    axis_shift.axhline(0, color="#374151", linewidth=0.8)
    axis_shift.set_xlabel("Scenario")
    axis_shift.set_ylabel("ThresholdShift")
    axis_shift.set_title("Defense threshold-shift sweep")
    axis_shift.set_xticks(axis_shift.get_xticks())
    axis_shift.set_xticklabels([label.get_text().upper() for label in axis_shift.get_xticklabels()])
    axis_shift.legend(frameon=False, title="Defense")
    clean_axis(axis_shift)
    panel_label(axis_shift, "C")

    save_figure(
        figure,
        output_dir,
        "supp_defense_threshold_sweep",
        manifest,
        "Defense threshold-sweep supplement",
        [exp5_dir / "alpha_curves.csv", exp5_dir / "threshold_shift_summary.csv"],
        "Supplementary figure for the older Exp5 defense threshold-shift sweep.",
    )


def figure_alpha_calibration(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    calibration_dir = DEFENSE_BENCHMARK_OUTPUTS_ROOT / "alpha_calibration"
    summary_rows = numeric_columns(read_csv(calibration_dir / "summary.csv"), ["n_society", "alpha", "p_collapse", "p_collapse_ci_low", "p_collapse_ci_high"])
    scenarios = [scenario for scenario in ["s1", "s2", "s3", "s4"] if scenario in set(summary_rows["scenario"])]
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True, sharey=True)
    flat_axes = axes.ravel()
    for axis_position, scenario in enumerate(scenarios):
        axis = flat_axes[axis_position]
        scenario_rows = summary_rows[summary_rows["scenario"] == scenario]
        n_values = sorted(scenario_rows["n_society"].dropna().astype(int).unique())
        size_colors = {size_value: SIZE_PALETTE[position % len(SIZE_PALETTE)] for position, size_value in enumerate(n_values)}
        for size_value in n_values:
            subset = scenario_rows[scenario_rows["n_society"] == size_value].sort_values("alpha")
            axis.plot(
                subset["alpha"],
                subset["p_collapse"],
                marker="o",
                linewidth=1.4,
                markersize=3.4,
                color=size_colors[size_value],
                label=f"N={size_value}",
            )
            axis.fill_between(
                subset["alpha"].to_numpy(dtype=float),
                subset["p_collapse_ci_low"].to_numpy(dtype=float),
                subset["p_collapse_ci_high"].to_numpy(dtype=float),
                color=size_colors[size_value],
                alpha=0.12,
                linewidth=0,
            )
        axis.axhline(0.5, color="#6B7280", linestyle="--", linewidth=0.9)
        set_alpha_axis(axis, float(scenario_rows["alpha"].max()), linthresh=1e-4)
        axis.set_ylim(-0.04, 1.04)
        axis.set_title(f"{SCENARIO_LABELS.get(scenario, scenario.upper())} alpha calibration")
        axis.set_xlabel("Harmful-agent ratio alpha")
        axis.set_ylabel("P(collapse)")
        axis.legend(frameon=False, loc="lower right")
        clean_axis(axis)
        panel_label(axis, chr(ord("A") + axis_position))
    for axis in flat_axes[len(scenarios):]:
        axis.axis("off")

    save_figure(
        figure,
        output_dir,
        "supp_alpha_calibration",
        manifest,
        "Alpha-grid calibration supplement",
        [calibration_dir / "summary.csv", calibration_dir / "recommended_alpha_grid.csv"],
        "Supplementary protocol figure showing calibrated NoGuard collapse curves across scenarios and N.",
    )


def coverage_record(label: str, track: str, path: Path) -> dict[str, Any]:
    frame = read_csv(path)
    scenario_column = "scenario_id" if "scenario_id" in frame.columns else "scenario"
    condition_columns = ["defense", "label", "placement", "feedback_strength", "family", "capacity_mode"]
    condition_count = max([count_unique(frame, column) for column in condition_columns if column in frame.columns] or [0])
    return {
        "experiment": label,
        "track": track,
        "episodes": int(len(frame)),
        "scenarios": count_unique(frame, scenario_column),
        "n_values": count_unique(frame, "n_society"),
        "alphas": count_unique(frame, "alpha"),
        "seeds": count_unique(frame, "seed"),
        "defenses": count_unique(frame, "defense"),
        "conditions": condition_count,
        "source": str(path.relative_to(OUTPUTS_ROOT.parent)),
    }


def figure_data_coverage(output_dir: Path, manifest: list[dict[str, Any]], qwen_out: str) -> None:
    coverage_specs = [
        ("Exp1 alpha sweep", "Scaling", SCALING_THEORY_OUTPUTS_ROOT / "exp1_alpha_scaling" / "data.csv"),
        ("Exp2 alpha_c(N)", "Scaling", SCALING_THEORY_OUTPUTS_ROOT / "exp2_society_size_scaling" / "data.csv"),
        ("Exp3 centrality", "Mechanism", SCALING_THEORY_OUTPUTS_ROOT / "exp3_centrality_placement" / "data.csv"),
        ("Exp4 feedback", "Mechanism", SCALING_THEORY_OUTPUTS_ROOT / "exp4_feedback_ablation" / "data.csv"),
        ("Exp5 capacity", "Scaling", SCALING_THEORY_OUTPUTS_ROOT / "exp5_capacity_control" / "data.csv"),
        ("Exp6 LLM", "LLM", SCALING_THEORY_OUTPUTS_ROOT / "exp6_llm_n200_scaling" / "data.csv"),
        ("Exp7 mechanisms", "Scaling", SCALING_THEORY_OUTPUTS_ROOT / "exp7_cross_mechanism_threshold" / "data.csv"),
        ("Exp8 sensitivity", "Robustness", SCALING_THEORY_OUTPUTS_ROOT / "exp8_sensitivity_audit" / "data.csv"),
        ("Alpha calibration", "Protocol", DEFENSE_BENCHMARK_OUTPUTS_ROOT / "alpha_calibration" / "data.csv"),
        ("Defense Exp5", "Defense", DEFENSE_BENCHMARK_OUTPUTS_ROOT / "exp5_wolfguard_defense" / "data.csv"),
        ("Defense Exp6", "Defense", DEFENSE_BENCHMARK_OUTPUTS_ROOT / "exp6" / "data.csv"),
    ]
    qwen_dir = DEFENSE_BENCHMARK_OUTPUTS_ROOT / qwen_out
    if run_status(qwen_dir) == "summary_written" and (qwen_dir / "data.csv").exists():
        coverage_specs.append(("Qwen Exp6", "LLM defense", qwen_dir / "data.csv"))

    records = [coverage_record(label, track, path) for label, track, path in coverage_specs if path.exists()]
    coverage_frame = pd.DataFrame(records)
    coverage_frame.to_csv(output_dir / "data_coverage_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 4.8), constrained_layout=True)
    axis_counts, axis_design = axes

    ordered = coverage_frame.sort_values("episodes", ascending=True)
    track_palette = {
        "Scaling": "#2A9D8F",
        "Mechanism": "#4E79A7",
        "Robustness": "#F2A541",
        "Protocol": "#8E9AAF",
        "Defense": "#E76F51",
        "LLM": "#9B7EDE",
        "LLM defense": "#5ABF90",
    }
    axis_counts.barh(
        ordered["experiment"],
        ordered["episodes"],
        color=[track_palette.get(track, "#4E79A7") for track in ordered["track"]],
        alpha=0.92,
    )
    for position, row in enumerate(ordered.itertuples(index=False)):
        axis_counts.text(
            float(row.episodes) * 1.05,
            position,
            format_count(int(row.episodes)),
            va="center",
            ha="left",
            fontsize=7.5,
            color="#52616F",
        )
    axis_counts.set_xscale("log")
    axis_counts.set_xlabel("Episodes, log scale")
    axis_counts.set_title("Evidence volume by experiment")
    clean_axis(axis_counts)
    panel_label(axis_counts, "A")

    design_columns = ["scenarios", "n_values", "alphas", "seeds", "defenses", "conditions"]
    design_values = coverage_frame.set_index("experiment")[design_columns].reindex(ordered["experiment"])
    color_values = np.log10(design_values.astype(float) + 1.0)
    annotations = design_values.astype(int).astype(str)
    sns.heatmap(
        color_values,
        ax=axis_design,
        cmap=DELTA_CMAP,
        annot=annotations,
        fmt="",
        linewidths=0.35,
        linecolor="#FFFFFF",
        cbar_kws={"label": "log10(count + 1)", "shrink": 0.82},
    )
    axis_design.set_xlabel("Design dimension")
    axis_design.set_ylabel("")
    axis_design.set_title("Design breadth behind summary points")
    axis_design.set_xticklabels(["scen.", "N", "alpha", "seed", "def.", "cond."], rotation=0)
    axis_design.set_yticklabels(axis_design.get_yticklabels(), rotation=0)
    panel_label(axis_design, "B")

    save_figure(
        figure,
        output_dir,
        "figure_0_data_coverage",
        manifest,
        "Data coverage and design breadth",
        [path for _, _, path in coverage_specs if path.exists()],
        "Diagnostic overview showing why several paper panels look sparse after aggregation: each point compresses large episode grids.",
        f"Aggregates {format_count(int(coverage_frame['episodes'].sum()))} completed episodes across {len(coverage_frame)} experiment outputs.",
    )


def write_manifest(output_dir: Path, manifest: list[dict[str, Any]], skipped: list[dict[str, str]]) -> None:
    json_path = output_dir / "figure_manifest.json"
    json_path.write_text(json.dumps({"figures": manifest, "skipped": skipped}, indent=2) + "\n")
    lines = [
        "# WolfBench Paper Figures",
        "",
        "Unified visual style: white background, light blue-gray grid, teal/coral/blue/amber scenario palette, 340 dpi PNG plus vector PDF.",
        "",
        "| Figure | Files | Main sources | Evidence coverage | Paper role |",
        "|---|---|---|---|---|",
    ]
    for entry in manifest:
        files = f"`{entry['png']}`, `{entry['pdf']}`"
        sources = ", ".join(f"`{source}`" for source in entry["sources"])
        evidence = entry.get("evidence") or "See source artifacts."
        lines.append(f"| {entry['title']} | {files} | {sources} | {evidence} | {entry['note']} |")
    if skipped:
        lines.extend(["", "## Skipped", ""])
        for entry in skipped:
            lines.append(f"- `{entry['name']}`: {entry['reason']}")
    lines.append("")
    (output_dir / "figure_manifest.md").write_text("\n".join(lines))


def build_figures(defense_out: str, qwen_out: str, include_running_qwen: bool) -> None:
    configure_style()
    output_dir = SCALING_THEORY_OUTPUTS_ROOT / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    figure_data_coverage(output_dir, manifest, qwen_out)
    figure_scaling_transition(output_dir, manifest)
    figure_mechanisms_robustness(output_dir, manifest)
    figure_capacity_llm(output_dir, manifest)
    figure_defense_leaderboard(output_dir, manifest, defense_out, "figure_4_defense_leaderboard", "")
    figure_defense_threshold_sweep(output_dir, manifest)
    figure_alpha_calibration(output_dir, manifest)

    qwen_dir = DEFENSE_BENCHMARK_OUTPUTS_ROOT / qwen_out
    qwen_status = run_status(qwen_dir)
    if qwen_dir.exists() and (qwen_status == "summary_written" or include_running_qwen):
        figure_defense_leaderboard(output_dir, manifest, qwen_out, "supp_qwen_defense_leaderboard", " with Qwen")
    else:
        skipped.append({
            "name": qwen_out,
            "reason": f"status={qwen_status}; rerun this script after Qwen leaderboard finishes to generate supp_qwen_defense_leaderboard.",
        })

    write_manifest(output_dir, manifest, skipped)
    print(f"Wrote {len(manifest)} paper figures to {output_dir}")
    if skipped:
        print("Skipped:")
        for entry in skipped:
            print(f"  {entry['name']}: {entry['reason']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defense-out", default="exp6", help="Completed defense leaderboard output folder to use for the main defense figure.")
    parser.add_argument("--qwen-out", default="exp6_qwen", help="Optional Qwen leaderboard output folder.")
    parser.add_argument("--include-running-qwen", action="store_true", help="Force plotting qwen-out even when run_metadata says it is still running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figures(args.defense_out, args.qwen_out, args.include_running_qwen)


if __name__ == "__main__":
    main()