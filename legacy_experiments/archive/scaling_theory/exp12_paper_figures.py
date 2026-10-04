"""Build paper-quality Exp12 canonical scaling figures.

The simulation script writes quick diagnostic PNGs. This module is the polished
paper figure layer: it reads an existing Exp12 output directory and writes
publication-ready PNG/PDF figures with a restrained pink/purple palette.

Usage:
    python -m experiments.scaling_theory.exp12_paper_figures

Environment:
    WOLFBENCH_EXP12_FIG_IN=exp12_canonical_scaling_refined
    WOLFBENCH_EXP12_FIG_OUT=paper_figures
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

from experiments._common import OUTPUTS_ROOT, SCALING_THEORY_OUTPUTS_ROOT


DEFAULT_INPUT = "exp12_canonical_scaling_refined"
DEFAULT_OUTPUT = "paper_figures"

COLORS = {
    "ink": "#25142D",
    "muted": "#6E5A78",
    "grid": "#E9DEEE",
    "panel": "#FFF8FD",
    "rule": "#D6C2DF",
    "purple": "#5E2A84",
    "violet": "#8B5CF6",
    "lavender": "#C084FC",
    "magenta": "#E84A9B",
    "rose": "#F472B6",
    "pink": "#FFB3D9",
    "plum": "#3D1B52",
    "gray": "#A99BB0",
}

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

SCENARIO_ORDER = ["s1", "s2", "s3", "s4"]

N_RAMP_COLORS = [
    "#F7B6D9",  # N=500: light pink
    "#E879B8",  # N=1000: saturated pink
    "#A855F7",  # N=2000: violet
    "#4C1D95",  # N=10000: deep purple
]
N_RAMP_LINEWIDTHS = [1.15, 1.45, 1.80, 2.35]
N_RAMP_MARKERSIZES = [2.4, 2.8, 3.2, 3.8]


def configure_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D9C6E1",
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.75,
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.8,
            "legend.fontsize": 7.0,
            "legend.title_fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.linewidth": 0.75,
        },
    )


def exp12_dir() -> Path:
    name = os.getenv("WOLFBENCH_EXP12_FIG_IN", DEFAULT_INPUT)
    path = SCALING_THEORY_OUTPUTS_ROOT / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Exp12 output directory: {path}. "
            "Run experiments.scaling_theory.exp12_canonical_scaling first, "
            "or set WOLFBENCH_EXP12_FIG_IN to an existing Exp12 output name."
        )
    return path


def output_dir(input_dir: Path) -> Path:
    out_name = os.getenv("WOLFBENCH_EXP12_FIG_OUT", DEFAULT_OUTPUT)
    out = input_dir / out_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Exp12 artifact: {path}")
    return pd.read_csv(path)


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def scenario_order(present: set[str]) -> list[str]:
    ordered = [scenario for scenario in SCENARIO_ORDER if scenario in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    vals = [int(np.clip(channel, 0.0, 1.0) * 255) for channel in rgb]
    return "#" + "".join(f"{value:02x}" for value in vals)


def mix(color: str, other: str = "#FFFFFF", amount: float = 0.35) -> str:
    r1, g1, b1 = hex_to_rgb(color)
    r2, g2, b2 = hex_to_rgb(other)
    rgb = (
        r1 * (1.0 - amount) + r2 * amount,
        g1 * (1.0 - amount) + g2 * amount,
        b1 * (1.0 - amount) + b2 * amount,
    )
    return rgb_to_hex(rgb)


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, which="major", color=COLORS["grid"], linewidth=0.75)
    axis.grid(True, which="minor", color=COLORS["grid"], linewidth=0.45, alpha=0.45)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(out):
        return ""
    if abs(out) >= 0.1:
        return f"{out:.2f}"
    if abs(out) >= 0.01:
        return f"{out:.3f}"
    return f"{out:.{digits}f}"


def n_style(idx: int, count: int) -> tuple[str, float, float]:
    if count <= 1:
        return N_RAMP_COLORS[-1], N_RAMP_LINEWIDTHS[-1], N_RAMP_MARKERSIZES[-1]
    palette_idx = int(round(idx * (len(N_RAMP_COLORS) - 1) / max(count - 1, 1)))
    return (
        N_RAMP_COLORS[palette_idx],
        N_RAMP_LINEWIDTHS[palette_idx],
        N_RAMP_MARKERSIZES[palette_idx],
    )


def plot_alpha_c(axis: plt.Axes, thresholds: pd.DataFrame, scenarios: list[str]) -> None:
    for scenario in scenarios:
        rows = thresholds[thresholds["scenario"] == scenario].dropna(subset=["alpha_c_logistic"])
        rows = rows.sort_values("n_society")
        if rows.empty:
            continue
        y = rows["alpha_c_logistic"].to_numpy(dtype=float)
        low = rows["alpha_c_ci_low"].fillna(rows["alpha_c_logistic"]).to_numpy(dtype=float)
        high = rows["alpha_c_ci_high"].fillna(rows["alpha_c_logistic"]).to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(0.0, y - low), np.maximum(0.0, high - y)])
        axis.errorbar(
            rows["n_society"],
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.8,
            capsize=2.5,
            markersize=4.3,
            color=SCENARIO_COLORS.get(scenario, COLORS["magenta"]),
            label=scenario.upper(),
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("society size N")
    axis.set_ylabel("critical harmful ratio alpha_c")
    axis.set_title("Primary-failure threshold by scale")
    axis.legend(frameon=False, ncol=2, loc="best")
    clean_axis(axis)


def plot_widths(axis: plt.Axes, widths: pd.DataFrame, scenarios: list[str]) -> None:
    any_rows = False
    for scenario in scenarios:
        rows = widths[widths["scenario"] == scenario].copy()
        rows = rows[pd.to_numeric(rows["transition_width_10_90"], errors="coerce") > 0]
        rows = rows.sort_values("n_society")
        if rows.empty:
            continue
        any_rows = True
        axis.plot(
            rows["n_society"],
            rows["transition_width_10_90"],
            marker="o",
            linewidth=1.8,
            markersize=4.3,
            color=SCENARIO_COLORS.get(scenario, COLORS["magenta"]),
            label=scenario.upper(),
        )
    axis.set_xscale("log")
    if any_rows:
        axis.set_yscale("log")
    axis.set_xlabel("society size N")
    axis.set_ylabel("transition width")
    axis.set_title("Transition width across scale")
    axis.legend(frameon=False, ncol=2, loc="best")
    clean_axis(axis)


def plot_summary_table(
    axis: plt.Axes,
    thresholds: pd.DataFrame,
    summary: pd.DataFrame,
    scenarios: list[str],
    input_name: str,
) -> None:
    axis.axis("off")
    rows_out: list[list[str]] = []
    row_colors: list[str] = []
    for scenario in scenarios:
        t_rows = thresholds[thresholds["scenario"] == scenario].dropna(subset=["alpha_c_logistic"])
        if t_rows.empty:
            alpha_at_max = ""
            width_at_max = ""
        else:
            t_row = t_rows.sort_values("n_society").iloc[-1]
            alpha_at_max = fmt(t_row.get("alpha_c_logistic"))
            width_at_max = fmt(t_row.get("transition_width_10_90"))
        s_rows = summary[summary["scenario"] == scenario]
        if s_rows.empty:
            metric = ""
            grade = ""
        else:
            s_row = s_rows.iloc[0]
            metric = PRIMARY_LABELS.get(str(s_row.get("primary_metric", "")), str(s_row.get("primary_metric", "")))
            grade = str(s_row.get("evidence_grade", ""))
        rows_out.append([scenario.upper(), metric, alpha_at_max, width_at_max, grade])
        row_colors.append(mix(SCENARIO_COLORS.get(scenario, COLORS["magenta"]), "#FFFFFF", 0.82))

    table = axis.table(
        cellText=rows_out,
        colLabels=["Scenario", "Primary metric", "alpha_c@maxN", "width@maxN", "Evidence"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.15, 0.30, 0.18, 0.17, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.05, 1.35)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["rule"])
        cell.set_linewidth(0.65)
        if row == 0:
            cell.set_facecolor("#F3D8F3")
            cell.set_text_props(weight="bold", color=COLORS["ink"])
        else:
            cell.set_facecolor(row_colors[row - 1])
            cell.set_text_props(color=COLORS["ink"])
    axis.set_title("Evidence summary", pad=10, color=COLORS["ink"])
    axis.text(
        0.0,
        0.05,
        f"Source: paperoutputs/scaling/{input_name}\nS2 can be a discrete first-finfluencer boundary; report it honestly if the curve is step-like.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.1,
        color=COLORS["muted"],
    )


def load_artifacts(input_dir: Path) -> dict[str, pd.DataFrame]:
    curves = numeric(
        read_csv(input_dir / "failure_curves.csv"),
        [
            "n_society",
            "alpha",
            "primary_failure_mean",
            "primary_failure_ci_low",
            "primary_failure_ci_high",
            "collapse_rate_mean",
        ],
    )
    thresholds = numeric(
        read_csv(input_dir / "alpha_c_by_scenario_n.csv"),
        [
            "n_society",
            "alpha_c_logistic",
            "alpha_c_ci_low",
            "alpha_c_ci_high",
            "transition_width_10_90",
        ],
    )
    widths = numeric(
        read_csv(input_dir / "width_by_scenario_n.csv"),
        ["n_society", "transition_width_10_90", "alpha_c_logistic"],
    )
    summary = read_csv(input_dir / "scenario_law_summary.csv")
    return {
        "curves": curves,
        "thresholds": thresholds,
        "widths": widths,
        "summary": summary,
    }


def alpha_ticks(scenario: str, max_alpha: float) -> list[float]:
    if scenario == "s2":
        candidates = [0.0, 0.0005, 0.001, 0.0015, 0.002]
    elif scenario == "s1":
        candidates = [0.0, 0.01, 0.02, 0.04]
    elif scenario == "s3" and max_alpha <= 0.2:
        candidates = [0.0, 0.05, 0.10, 0.15]
    elif scenario == "s3":
        candidates = [0.0, 0.15, 0.50, 1.00]
    else:
        candidates = [0.0, 0.025, 0.05, 0.075]
    return [tick for tick in candidates if tick <= max_alpha * 1.001]


def alpha_display_max(scenario: str, observed_max: float) -> float:
    if scenario == "s2":
        return min(observed_max, 0.002)
    return observed_max


def alpha_tick_label(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    if value >= 0.1:
        return f"{value:.2g}"
    if value >= 0.01:
        return f"{value:.3g}".replace("0.", ".")
    return f"{value:.4g}".replace("0.", ".")


def save_entry(
    figure: plt.Figure,
    out_dir: Path,
    stem: str,
    title: str,
    input_dir: Path,
    sources: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.035)
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)
    entry: dict[str, Any] = {
        "figure": stem,
        "title": title,
        "png": str(png.relative_to(OUTPUTS_ROOT.parent)),
        "pdf": str(pdf.relative_to(OUTPUTS_ROOT.parent)),
        "sources": [
            str((input_dir / source).relative_to(OUTPUTS_ROOT.parent))
            for source in sources
        ],
    }
    if extra:
        entry.update(extra)
    return entry


def build_primary_curves_figure(
    input_dir: Path,
    out_dir: Path,
    curves: pd.DataFrame,
    scenarios: list[str],
    n_values: list[int],
) -> dict[str, Any]:
    figure, axes = plt.subplots(2, 2, figsize=(7.3, 5.2), constrained_layout=False)
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.86, wspace=0.28, hspace=0.46)
    axes_flat = axes.ravel()
    n_focus = max(n_values)
    for panel_idx, (axis, scenario) in enumerate(zip(axes_flat, scenarios)):
        subset = curves[curves["scenario"] == scenario].sort_values(["n_society", "alpha"])
        for idx, n_society in enumerate(n_values):
            rows = subset[subset["n_society"] == n_society].sort_values("alpha")
            if rows.empty:
                continue
            color, linewidth, markersize = n_style(idx, len(n_values))
            is_focus = n_society == n_focus
            axis.plot(
                rows["alpha"],
                rows["primary_failure_mean"],
                color=color,
                linewidth=linewidth,
                marker="o",
                markersize=markersize,
                alpha=0.96 if is_focus else 0.65,
                label=f"N={n_society}",
            )
            if is_focus:
                axis.fill_between(
                    rows["alpha"].to_numpy(dtype=float),
                    rows["primary_failure_ci_low"].to_numpy(dtype=float),
                    rows["primary_failure_ci_high"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
        observed_max = float(subset["alpha"].max()) if not subset.empty else 1.0
        max_alpha = alpha_display_max(scenario, observed_max)
        ticks = alpha_ticks(scenario, max_alpha)
        axis.set_xlim(-max_alpha * 0.025, max_alpha * 1.025)
        axis.set_xticks(ticks)
        axis.set_xticklabels([alpha_tick_label(tick) for tick in ticks])
        axis.axhline(0.5, color=COLORS["muted"], linestyle="--", linewidth=0.85, alpha=0.76)
        axis.set_ylim(-0.04, 1.04)
        axis.set_title(SCENARIO_LABELS.get(scenario, scenario.upper()), pad=3)
        axis.set_xlabel("harmful-agent ratio alpha" if panel_idx >= 2 else "")
        axis.set_ylabel("P(primary failure)" if panel_idx % 2 == 0 else "")
        clean_axis(axis)
    for axis in axes_flat[len(scenarios) :]:
        axis.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(len(labels), 4),
        frameon=False,
        handlelength=2.0,
    )
    return save_entry(
        figure,
        out_dir,
        "exp12_primary_failure_curves_paper",
        "Exp12 primary failure curves",
        input_dir,
        ["failure_curves.csv"],
        {"scenarios": scenarios, "n_values": n_values},
    )


def build_threshold_figure(
    input_dir: Path,
    out_dir: Path,
    thresholds: pd.DataFrame,
    widths: pd.DataFrame,
    scenarios: list[str],
) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(7.3, 3.0), constrained_layout=True)
    plot_alpha_c(axes[0], thresholds, scenarios)
    plot_widths(axes[1], widths, scenarios)
    axes[0].set_title("Critical ratio alpha_c(N)")
    axes[1].set_title("Transition width W(N)")
    return save_entry(
        figure,
        out_dir,
        "exp12_threshold_scaling_paper",
        "Exp12 threshold scaling",
        input_dir,
        ["alpha_c_by_scenario_n.csv", "width_by_scenario_n.csv"],
        {"scenarios": scenarios},
    )


def build_summary_figure(
    input_dir: Path,
    out_dir: Path,
    thresholds: pd.DataFrame,
    summary: pd.DataFrame,
    scenarios: list[str],
) -> dict[str, Any]:
    figure, axis = plt.subplots(figsize=(7.2, 2.55), constrained_layout=True)
    plot_summary_table(axis, thresholds, summary, scenarios, input_dir.name)
    axis.set_title("Exp12 evidence summary", pad=8, color=COLORS["ink"])
    return save_entry(
        figure,
        out_dir,
        "exp12_evidence_summary_paper",
        "Exp12 evidence summary",
        input_dir,
        ["alpha_c_by_scenario_n.csv", "scenario_law_summary.csv"],
        {"scenarios": scenarios},
    )


def build_paper_figures(input_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    artifacts = load_artifacts(input_dir)
    curves = artifacts["curves"]
    thresholds = artifacts["thresholds"]
    widths = artifacts["widths"]
    summary = artifacts["summary"]
    scenarios = scenario_order(set(curves["scenario"].astype(str)))
    n_values = sorted(int(n) for n in curves["n_society"].dropna().unique())
    if not scenarios or not n_values:
        raise RuntimeError(f"No scenario/N rows found in {input_dir / 'failure_curves.csv'}")
    return [
        build_primary_curves_figure(input_dir, out_dir, curves, scenarios, n_values),
        build_threshold_figure(input_dir, out_dir, thresholds, widths, scenarios),
        build_summary_figure(input_dir, out_dir, thresholds, summary, scenarios),
    ]


def write_manifest(out_dir: Path, entries: list[dict[str, Any]]) -> None:
    manifest = {"figures": entries}
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Exp12 Paper Figures",
        "",
        "| Figure | PNG | PDF | Sources |",
        "|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            f"| {entry['title']} | `{entry['png']}` | `{entry['pdf']}` | "
            + ", ".join(f"`{source}`" for source in entry["sources"])
            + " |"
        )
    if entries:
        lines.extend([
            "",
            f"Scenarios: {', '.join(s.upper() for s in entries[0].get('scenarios', []))}",
            f"N values: {', '.join(str(n) for n in entries[0].get('n_values', []))}",
            "",
        ])
    (out_dir / "figure_manifest.md").write_text("\n".join(lines))


def main() -> None:
    input_path = exp12_dir()
    out_path = output_dir(input_path)
    entries = build_paper_figures(input_path, out_path)
    write_manifest(out_path, entries)
    print(f"Wrote {len(entries)} Exp12 paper figures to {out_path}")
    for entry in entries:
        print(entry["png"])
        print(entry["pdf"])


if __name__ == "__main__":
    main()