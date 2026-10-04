"""Build the two paper-facing scaling figures needed for the main story.

Figure 3 is intentionally S1-only and requires the dense Exp2 society-size
sweep. Figure 4 is a caveat-aware mechanism figure from the current Exp12
canonical scaling audit.

Usage:
    python -m experiments.scaling_theory.paper_oral_figures

Inputs:
    paperoutputs/scaling/exp2_society_size_scaling/
    paperoutputs/scaling/exp12_canonical_scaling_refined/

Outputs:
    paperoutputs/scaling/paper_oral_figures/
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
from scipy.special import expit

from experiments._common import OUTPUTS_ROOT, SCALING_THEORY_OUTPUTS_ROOT


EXP2_NAME = os.getenv("WOLFBENCH_ORAL_EXP2_IN", "exp2_society_size_scaling")
EXP12_NAME = os.getenv("WOLFBENCH_ORAL_EXP12_IN", "exp12_canonical_scaling_refined")
OUT_NAME = os.getenv("WOLFBENCH_ORAL_FIG_OUT", "paper_oral_figures")

COLORS = {
    "ink": "#24172B",
    "muted": "#6F6078",
    "grid": "#E8DEED",
    "rule": "#CDBBD8",
    "magenta": "#D83A8B",
    "rose": "#F06BAA",
    "violet": "#7C4DFF",
    "purple": "#4B1D73",
    "lavender": "#B77AF2",
    "orange": "#F59E0B",
    "green": "#059669",
    "gray": "#9B91A3",
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
SCENARIO_COLORS = {
    "s1": COLORS["magenta"],
    "s2": COLORS["violet"],
    "s3": COLORS["purple"],
    "s4": COLORS["rose"],
}


def configure_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.4,
            "legend.fontsize": 6.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": COLORS["rule"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.65,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
        },
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, which="major", color=COLORS["grid"], linewidth=0.65)
    axis.grid(True, which="minor", color=COLORS["grid"], linewidth=0.35, alpha=0.45)


def require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {what}: {path}\n"
            "Run the required experiment first. For Figure 3:\n"
            "  PYTHONUNBUFFERED=1 python -m experiments.scaling_theory.exp2_society_size_scaling\n"
            "For Figure 4:\n"
            "  WOLFBENCH_EXP12_FIG_IN=exp12_canonical_scaling_refined "
            "python -m experiments.scaling_theory.exp12_paper_figures"
        )
    return path


def read_csv(path: Path, what: str) -> pd.DataFrame:
    return pd.read_csv(require(path, what))


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def fit_power(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return {"coef": float("nan"), "exponent": float("nan"), "r2": float("nan")}
    exponent, log_coef = np.polyfit(np.log(x), np.log(y), 1)
    pred = log_coef + exponent * np.log(x)
    target = np.log(y)
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"coef": float(np.exp(log_coef)), "exponent": float(exponent), "r2": r2}


def alpha_label(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    if value >= 0.01:
        return f"{value:.3g}".replace("0.", ".")
    return f"{value:.4g}".replace("0.", ".")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, sources: list[Path]) -> dict[str, Any]:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=420)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "figure": stem,
        "png": str(png.relative_to(OUTPUTS_ROOT.parent)),
        "pdf": str(pdf.relative_to(OUTPUTS_ROOT.parent)),
        "sources": [str(path.relative_to(OUTPUTS_ROOT.parent)) for path in sources],
    }


def build_s1_scaling_figure(exp2_dir: Path, out_dir: Path) -> dict[str, Any]:
    curve = numeric(
        read_csv(exp2_dir / "collapse_rate_wilson_ci.csv", "Exp2 collapse-rate curves"),
        ["n_society", "alpha", "mean", "ci_low", "ci_high"],
    )
    thresholds = numeric(
        read_csv(exp2_dir / "alpha_critical_summary.csv", "Exp2 threshold summary"),
        [
            "n_society",
            "alpha_c_logistic",
            "alpha_c_ci_low",
            "alpha_c_ci_high",
            "logistic_slope",
            "transition_width_10_90",
        ],
    ).sort_values("n_society")

    fig = plt.figure(figsize=(7.25, 5.35))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_curves = fig.add_subplot(grid[0, :2])
    ax_heat = fig.add_subplot(grid[0, 2])
    ax_alpha = fig.add_subplot(grid[1, 0])
    ax_width = fig.add_subplot(grid[1, 1])
    ax_note = fig.add_subplot(grid[1, 2])

    focus_n = [100, 500, 1000, 5000]
    palette = ["#F4A6CF", COLORS["rose"], COLORS["violet"], COLORS["purple"]]
    alpha_dense = np.linspace(float(curve["alpha"].min()), float(curve["alpha"].max()), 240)
    for color, n_society in zip(palette, focus_n):
        rows = curve[curve["n_society"] == n_society].sort_values("alpha")
        if rows.empty:
            continue
        ax_curves.errorbar(
            rows["alpha"],
            rows["mean"],
            yerr=[
                np.maximum(0, rows["mean"].to_numpy() - rows["ci_low"].to_numpy()),
                np.maximum(0, rows["ci_high"].to_numpy() - rows["mean"].to_numpy()),
            ],
            fmt="o",
            ms=3.2,
            lw=0.9,
            capsize=1.8,
            color=color,
            alpha=0.85,
            label=f"N={n_society}",
        )
        trow = thresholds[thresholds["n_society"] == n_society]
        if not trow.empty and pd.notna(trow.iloc[0].get("logistic_slope")):
            ac = float(trow.iloc[0]["alpha_c_logistic"])
            slope = float(trow.iloc[0]["logistic_slope"])
            ax_curves.plot(alpha_dense, expit(slope * (alpha_dense - ac)), color=color, lw=1.7)
            ax_curves.axvline(ac, color=color, lw=0.8, ls=":", alpha=0.8)
    ax_curves.axhline(0.5, color=COLORS["muted"], lw=0.8, ls="--")
    ax_curves.set_title("A  finite-size collapse transitions")
    ax_curves.set_xlabel("harmful-agent fraction alpha")
    ax_curves.set_ylabel("P(collapse)")
    ax_curves.set_ylim(-0.04, 1.04)
    ax_curves.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
    ax_curves.set_xticklabels([alpha_label(v) for v in [0, 0.01, 0.02, 0.03, 0.04]])
    ax_curves.legend(frameon=False, ncol=2, loc="lower right")
    clean_axis(ax_curves)

    n_values = sorted(curve["n_society"].dropna().unique())
    alpha_values = sorted(curve["alpha"].dropna().unique())
    heat = np.full((len(n_values), len(alpha_values)), np.nan)
    for i, n_society in enumerate(n_values):
        for j, alpha in enumerate(alpha_values):
            rows = curve[(curve["n_society"] == n_society) & (np.isclose(curve["alpha"], alpha))]
            if not rows.empty:
                heat[i, j] = float(rows.iloc[0]["mean"])
    im = ax_heat.imshow(heat, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    ax_heat.set_title("B  response surface")
    ax_heat.set_xlabel("alpha")
    ax_heat.set_ylabel("N")
    ax_heat.set_xticks(np.arange(len(alpha_values))[::3])
    ax_heat.set_xticklabels([alpha_label(alpha_values[i]) for i in range(0, len(alpha_values), 3)], rotation=35)
    ax_heat.set_yticks(np.arange(len(n_values))[::2])
    ax_heat.set_yticklabels([str(int(n_values[i])) for i in range(0, len(n_values), 2)])
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03, label="P")

    x = thresholds["n_society"].to_numpy(dtype=float)
    y = thresholds["alpha_c_logistic"].to_numpy(dtype=float)
    low = thresholds["alpha_c_ci_low"].fillna(thresholds["alpha_c_logistic"]).to_numpy(dtype=float)
    high = thresholds["alpha_c_ci_high"].fillna(thresholds["alpha_c_logistic"]).to_numpy(dtype=float)
    fit_alpha = fit_power(x, y)
    ax_alpha.errorbar(
        x,
        y,
        yerr=[np.maximum(0, y - low), np.maximum(0, high - y)],
        fmt="o",
        color=COLORS["magenta"],
        ms=3.8,
        capsize=2.0,
        lw=1.1,
    )
    xfit = np.logspace(np.log10(np.nanmin(x)), np.log10(np.nanmax(x)), 160)
    ax_alpha.plot(xfit, fit_alpha["coef"] * xfit ** fit_alpha["exponent"], color=COLORS["ink"], lw=1.2, ls="--")
    ax_alpha.set_xscale("log")
    ax_alpha.set_yscale("log")
    ax_alpha.set_title("C  critical fraction moves")
    ax_alpha.set_xlabel("society size N")
    ax_alpha.set_ylabel(r"$\alpha_c(N)$")
    clean_axis(ax_alpha)

    width = thresholds["transition_width_10_90"].to_numpy(dtype=float)
    fit_width = fit_power(x, width)
    ax_width.plot(x, width, "o-", color=COLORS["violet"], ms=3.8, lw=1.4)
    ax_width.plot(xfit, fit_width["coef"] * xfit ** fit_width["exponent"], color=COLORS["ink"], lw=1.2, ls="--")
    ax_width.set_xscale("log")
    ax_width.set_yscale("log")
    ax_width.set_title("D  transition narrows")
    ax_width.set_xlabel("society size N")
    ax_width.set_ylabel(r"$W_N=\alpha_{.9}-\alpha_{.1}$")
    clean_axis(ax_width)

    ax_note.axis("off")
    endpoint = thresholds.iloc[[0, -1]]
    note = (
        "E  finite-size summary\n\n"
        f"alpha_c: {endpoint.iloc[0]['alpha_c_logistic']:.4f} -> "
        f"{endpoint.iloc[1]['alpha_c_logistic']:.4f}\n"
        f"width: {endpoint.iloc[0]['transition_width_10_90']:.4f} -> "
        f"{endpoint.iloc[1]['transition_width_10_90']:.4f}\n\n"
        f"alpha exponent: {fit_alpha['exponent']:.3f}\n"
        f"alpha log-R2: {fit_alpha['r2']:.3f}\n"
        f"width exponent: {fit_width['exponent']:.3f}\n"
        f"width log-R2: {fit_width['r2']:.3f}\n\n"
        "Read as finite-size\nresponse summaries,\nnot universal constants."
    )
    ax_note.text(
        0.02,
        0.98,
        note,
        ha="left",
        va="top",
        transform=ax_note.transAxes,
        fontsize=7.4,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF6FB", edgecolor=COLORS["rule"], linewidth=0.7),
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.945, bottom=0.09, wspace=0.44, hspace=0.46)
    return save_figure(
        fig,
        out_dir,
        "figure3_s1_finite_size_scaling",
        [exp2_dir / "collapse_rate_wilson_ci.csv", exp2_dir / "alpha_critical_summary.csv"],
    )


def scenario_xlim(scenario: str, rows: pd.DataFrame) -> tuple[float, list[float]]:
    max_alpha = float(rows["alpha"].max())
    if scenario == "s2":
        return min(max_alpha, 0.002), [0, 0.0005, 0.001, 0.0015, 0.002]
    if scenario == "s1":
        return min(max_alpha, 0.04), [0, 0.01, 0.02, 0.03, 0.04]
    if scenario == "s3":
        return min(max_alpha, 0.10), [0, 0.025, 0.05, 0.075, 0.10]
    return min(max_alpha, 0.075), [0, 0.025, 0.05, 0.075]


def build_mechanism_figure(exp12_dir: Path, out_dir: Path) -> dict[str, Any]:
    curves = numeric(
        read_csv(exp12_dir / "failure_curves.csv", "Exp12 primary failure curves"),
        [
            "n_society",
            "alpha",
            "primary_failure_mean",
            "primary_failure_ci_low",
            "primary_failure_ci_high",
            "collapse_rate_mean",
            "collapse_rate_ci_low",
            "collapse_rate_ci_high",
        ],
    )
    thresholds = numeric(
        read_csv(exp12_dir / "alpha_c_by_scenario_n.csv", "Exp12 threshold summary"),
        [
            "n_society",
            "alpha_c_logistic",
            "alpha_c_ci_low",
            "alpha_c_ci_high",
            "transition_width_10_90",
        ],
    )
    summary = read_csv(exp12_dir / "scenario_law_summary.csv", "Exp12 scenario law summary")

    n_focus = 1000
    fig = plt.figure(figsize=(7.25, 5.3))
    grid = fig.add_gridspec(3, 4, height_ratios=[0.84, 1.05, 1.05], hspace=0.55, wspace=0.38)
    ax_forest = fig.add_subplot(grid[0, :3])
    ax_table = fig.add_subplot(grid[0, 3])
    axes = [fig.add_subplot(grid[1 + i // 2, (i % 2) * 2 : (i % 2) * 2 + 2]) for i in range(4)]

    forest_rows = thresholds[thresholds["n_society"] == n_focus].copy()
    forest_rows["scenario"] = forest_rows["scenario"].astype(str)
    forest_rows = forest_rows.set_index("scenario").reindex(SCENARIO_ORDER).reset_index()
    y_pos = np.arange(len(forest_rows))[::-1]
    for i, row in forest_rows.iterrows():
        scenario = str(row["scenario"])
        color = SCENARIO_COLORS.get(scenario, COLORS["gray"])
        alpha_c = float(row["alpha_c_logistic"])
        low = float(row["alpha_c_ci_low"]) if pd.notna(row["alpha_c_ci_low"]) else alpha_c
        high = float(row["alpha_c_ci_high"]) if pd.notna(row["alpha_c_ci_high"]) else alpha_c
        ax_forest.errorbar(
            alpha_c,
            y_pos[i],
            xerr=[[max(0, alpha_c - low)], [max(0, high - alpha_c)]],
            fmt="o",
            ms=5.0,
            color=color,
            capsize=2.5,
            lw=1.1,
        )
        ax_forest.text(high * 1.18 if high > 0 else alpha_c + 1e-5, y_pos[i], f"{alpha_c:.4g}", va="center", fontsize=7.0, color=COLORS["ink"])
    ax_forest.set_xscale("log")
    ax_forest.set_xlim(1.7e-4, 9e-2)
    ax_forest.set_yticks(y_pos)
    ax_forest.set_yticklabels([SCENARIO_LABELS.get(s, s.upper()) for s in forest_rows["scenario"]])
    ax_forest.set_xlabel(r"critical harmful-agent fraction $\alpha_c$ at $N=1000$")
    ax_forest.set_title("A  one estimator, mechanism-specific thresholds")
    clean_axis(ax_forest)

    ax_table.axis("off")
    rows_out: list[list[str]] = []
    short_events = {
        "generic_collapse": "generic",
        "spoof_liquidity_failure": "spoof/liq.",
        "fake_liquidity_failure": "fake-liq.",
    }
    short_grades = {
        "strong": "strong",
        "partial": "partial",
        "mechanism-strong": "mech.",
    }
    for scenario in SCENARIO_ORDER:
        srow = summary[summary["scenario"] == scenario]
        if srow.empty:
            rows_out.append([scenario.upper(), "", ""])
            continue
        row = srow.iloc[0]
        rows_out.append([
            scenario.upper(),
            short_events.get(str(row["primary_metric"]), str(row["primary_metric"])),
            short_grades.get(str(row["evidence_grade"]), str(row["evidence_grade"])),
        ])
    table = ax_table.table(
        cellText=rows_out,
        colLabels=["B/S", "event", "claim"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.17, 0.46, 0.37],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.3)
    table.scale(1.00, 1.28)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["rule"])
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#F3D8F3")
            cell.set_text_props(weight="bold", color=COLORS["ink"])
        else:
            cell.set_facecolor("#FFF7FC")
            cell.set_text_props(color=COLORS["ink"])

    panel_letters = ["C", "D", "E", "F"]
    for axis, scenario, letter in zip(axes, SCENARIO_ORDER, panel_letters):
        rows = curves[(curves["scenario"] == scenario) & (curves["n_society"] == n_focus)].sort_values("alpha")
        color = SCENARIO_COLORS.get(scenario, COLORS["gray"])
        xlim, ticks = scenario_xlim(scenario, rows)
        rows = rows[rows["alpha"] <= xlim * 1.001]
        axis.fill_between(
            rows["alpha"],
            rows["primary_failure_ci_low"],
            rows["primary_failure_ci_high"],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        axis.plot(rows["alpha"], rows["primary_failure_mean"], "o-", color=color, ms=3.2, lw=1.5, label="primary")
        if scenario in {"s3", "s4"}:
            axis.plot(rows["alpha"], rows["collapse_rate_mean"], "--", color=COLORS["gray"], lw=1.0, label="generic diagnostic")
        trow = thresholds[(thresholds["scenario"] == scenario) & (thresholds["n_society"] == n_focus)]
        if not trow.empty and pd.notna(trow.iloc[0]["alpha_c_logistic"]):
            axis.axvline(float(trow.iloc[0]["alpha_c_logistic"]), color=color, lw=0.9, ls=":")
        axis.axhline(0.5, color=COLORS["muted"], lw=0.75, ls="--")
        axis.set_title(f"{letter}  {SCENARIO_LABELS[scenario]}", pad=3)
        axis.set_xlim(-xlim * 0.025, xlim * 1.025)
        axis.set_ylim(-0.05, 1.05)
        axis.set_xticks(ticks)
        axis.set_xticklabels([alpha_label(tick) for tick in ticks])
        axis.set_xlabel("harmful-agent fraction alpha")
        axis.set_ylabel("P(primary failure)")
        clean_axis(axis)
        if scenario in {"s3", "s4"}:
            axis.legend(frameon=False, loc="lower right", fontsize=6.2)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.08)
    return save_figure(
        fig,
        out_dir,
        "figure4_mechanism_heterogeneity_caveat",
        [
            exp12_dir / "failure_curves.csv",
            exp12_dir / "alpha_c_by_scenario_n.csv",
            exp12_dir / "scenario_law_summary.csv",
        ],
    )


def write_manifest(out_dir: Path, entries: list[dict[str, Any]]) -> None:
    (out_dir / "figure_manifest.json").write_text(json.dumps({"figures": entries}, indent=2) + "\n")
    lines = [
        "# Oral-Level Paper Figures",
        "",
        "| Figure | PNG | PDF | Sources |",
        "|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            f"| {entry['figure']} | `{entry['png']}` | `{entry['pdf']}` | "
            + ", ".join(f"`{source}`" for source in entry["sources"])
            + " |"
        )
    (out_dir / "figure_manifest.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    configure_style()
    exp2_dir = SCALING_THEORY_OUTPUTS_ROOT / EXP2_NAME
    exp12_dir = SCALING_THEORY_OUTPUTS_ROOT / EXP12_NAME
    out_dir = SCALING_THEORY_OUTPUTS_ROOT / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    strict = os.getenv("WOLFBENCH_ORAL_STRICT", "0") == "1"
    for builder, source_dir, name in [
        (build_s1_scaling_figure, exp2_dir, "figure3_s1_finite_size_scaling"),
        (build_mechanism_figure, exp12_dir, "figure4_mechanism_heterogeneity_caveat"),
    ]:
        try:
            entries.append(builder(source_dir, out_dir))
        except FileNotFoundError as exc:
            if strict:
                raise
            entries.append({
                "figure": name,
                "missing": True,
                "reason": str(exc),
                "png": "",
                "pdf": "",
                "sources": [],
            })
    write_manifest(out_dir, entries)
    print(f"Wrote {len(entries)} figures to {out_dir}")
    for entry in entries:
        print(entry["png"])
        print(entry["pdf"])


if __name__ == "__main__":
    main()
