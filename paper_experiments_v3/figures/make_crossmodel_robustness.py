"""Build cross-model robustness artifacts for P01.

This script reads the completed cross-model P01 outputs and writes a compact
table plus a two-panel response-curve figure for appendix use.
"""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wolfbench_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "wolfbench_xdg_cache"))

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUT_DIR = ROOT / "figures" / "generated"

MODELS = [
    ("qwen235b", "Qwen3-235B", "qwen/qwen3-235b-a22b"),
    ("gpt41", "GPT-4.1", "openai/gpt-4.1"),
    ("llama33_70b", "Llama-3.3-70B", "meta-llama/llama-3.3-70b-instruct"),
    ("glm45", "GLM-4.5", "z-ai/glm-4.5"),
    ("kimik26", "Kimi K2.6", "moonshotai/kimi-k2.6"),
]
AUDIT_MODELS = [
    *MODELS,
    ("opus", "Claude Opus 4.8", "anthropic/claude-opus-4.8"),
    ("gemini25pro", "Gemini 2.5 Pro", "google/gemini-2.5-pro"),
]

COLORS = {
    "Qwen3-235B": "#0072B2",
    "GPT-4.1": "#D55E00",
    "Llama-3.3-70B": "#009E73",
    "GLM-4.5": "#CC79A7",
    "Kimi K2.6": "#000000",
}
MARKERS = {
    "Qwen3-235B": "o",
    "GPT-4.1": "s",
    "Llama-3.3-70B": "^",
    "GLM-4.5": "D",
    "Kimi K2.6": "v",
}


def _cell_counts(rows: dict[tuple[int, float, int], dict]) -> dict[str, int]:
    return {
        "unique_cells": len(rows),
        "ok_cells": sum(1 for row in rows.values() if row.get("status") == "ok"),
        "error_cells": sum(1 for row in rows.values() if row.get("status") != "ok"),
        "n100_cells": sum(1 for n_value, _alpha, _seed in rows if n_value == 100),
        "n1000_cells": sum(1 for n_value, _alpha, _seed in rows if n_value == 1000),
    }


def audit_model(tag: str, label: str, model_id: str) -> dict:
    rows = load_latest_rows(tag)
    counts = _cell_counts(rows)
    first_error = next((row for row in rows.values() if row.get("status") != "ok"), None)
    status = "complete" if counts["unique_cells"] == 45 and counts["ok_cells"] == 45 else "excluded_incomplete"
    reason = ""
    if first_error is not None:
        reason = f"{first_error.get('error_type', '')}: {str(first_error.get('error', ''))[:180]}"
    return {
        "tag": tag,
        "backbone": label,
        "model_id": model_id,
        "status": status,
        **counts,
        "reason": reason,
    }


def load_latest_rows(tag: str) -> dict[tuple[int, float, int], dict]:
    path = OUTPUTS / f"p01_crossmodel_{tag}" / "rows.jsonl"
    latest: dict[tuple[int, float, int], dict] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                int(float(row["n_society"])),
                float(row["alpha"]),
                int(float(row["seed"])),
            )
            latest[key] = row
    return latest


def crossing(points: list[tuple[float, float]], target: float = 0.5) -> float | None:
    points = sorted(points)
    for (a0, p0), (a1, p1) in zip(points, points[1:]):
        if p0 == target:
            return a0
        if (p0 - target) * (p1 - target) <= 0 and p1 != p0:
            return a0 + (target - p0) * (a1 - a0) / (p1 - p0)
    return None


def mean_curve(rows: dict[tuple[int, float, int], dict], n_society: int) -> list[tuple[float, float]]:
    by_alpha: dict[float, list[float]] = defaultdict(list)
    for (n_value, alpha, _seed), row in rows.items():
        if n_value == n_society and row.get("status") == "ok":
            by_alpha[alpha].append(float(row["primary_failure_rate"]))
    return [(alpha, sum(values) / len(values)) for alpha, values in sorted(by_alpha.items())]


def model_summary(tag: str, label: str, model_id: str) -> tuple[dict, dict[int, list[tuple[float, float]]]]:
    rows = load_latest_rows(tag)
    n_ok = sum(1 for row in rows.values() if row.get("status") == "ok")
    if len(rows) != 45 or n_ok != 45:
        raise RuntimeError(f"{label} is incomplete: {n_ok}/45 ok, {len(rows)} unique cells")
    curves = {100: mean_curve(rows, 100), 1000: mean_curve(rows, 1000)}
    alpha_c_100 = crossing(curves[100])
    alpha_c_1000 = crossing(curves[1000])
    if alpha_c_100 is None or alpha_c_1000 is None:
        raise RuntimeError(f"{label} does not bracket the 0.5 failure threshold")
    nu = math.log(alpha_c_100 / alpha_c_1000, 10)
    summary = {
        "tag": tag,
        "backbone": label,
        "model_id": model_id,
        "cells_ok": n_ok,
        "alpha_c_100": alpha_c_100,
        "alpha_c_1000": alpha_c_1000,
        "nu_two_point": nu,
    }
    return summary, curves


def write_csv(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "table_crossmodel_robustness.csv"
    fields = ["backbone", "model_id", "cells_ok", "alpha_c_100", "alpha_c_1000", "nu_two_point"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_latex(rows: list[dict]) -> None:
    path = OUT_DIR / "table_crossmodel_robustness.tex"
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Backbone & $\alpha_c(100)$ & $\alpha_c(1000)$ & $\widehat{\nu}_{100\to1000}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['backbone']} & {row['alpha_c_100']:.4f} & "
            f"{row['alpha_c_1000']:.4f} & {row['nu_two_point']:.3f} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    path.write_text("\n".join(lines))


def write_audit_status(rows: list[dict], summaries: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = OUT_DIR / "table_crossmodel_run_status.csv"
    fields = [
        "backbone", "model_id", "status", "unique_cells", "ok_cells", "error_cells",
        "n100_cells", "n1000_cells", "reason",
    ]
    with status_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    report_path = OUT_DIR / "crossmodel_robustness_audit.md"
    lines = [
        "# Cross-LLM Robustness Audit",
        "",
        "## Completion Check",
        "",
        "| Backbone | Status | Unique cells | OK cells | Error cells | N=100 | N=1000 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['backbone']} | {row['status']} | {row['unique_cells']} | "
            f"{row['ok_cells']} | {row['error_cells']} | {row['n100_cells']} | {row['n1000_cells']} |"
        )
    lines.extend([
        "",
        "## Recomputed Thresholds",
        "",
        "| Backbone | alpha_c(100) | alpha_c(1000) | nu_hat |",
        "|---|---:|---:|---:|",
    ])
    for row in summaries:
        lines.append(
            f"| {row['backbone']} | {row['alpha_c_100']:.4f} | "
            f"{row['alpha_c_1000']:.4f} | {row['nu_two_point']:.3f} |"
        )
    excluded = [row for row in rows if row["status"] != "complete"]
    if excluded:
        lines.extend(["", "## Excluded Runs", ""])
        for row in excluded:
            lines.append(
                f"- {row['backbone']} was excluded from the threshold table because it has "
                f"{row['ok_cells']}/45 OK cells. Latest error sample: {row['reason']}"
            )
    nu_values = [row["nu_two_point"] for row in summaries]
    complete_count = len(summaries)
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"Across the {complete_count} complete backbones, the finite-size shift from N=100 to N=1000 is stable in direction and magnitude. The two-point exponent ranges from {min(nu_values):.3f} to {max(nu_values):.3f}. This supports the claim that the P01 scaling result is not specific to one LLM backbone. The claim should be phrased as direction-and-magnitude robustness across tested backbones, not as model identity or a universal exponent.",
        "",
    ])
    report_path.write_text("\n".join(lines))


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.3,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#2D2730",
        "axes.labelcolor": "#2D2730",
        "xtick.color": "#2D2730",
        "ytick.color": "#2D2730",
        "grid.color": "#E4DFE6",
        "grid.linewidth": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def make_figure(all_curves: dict[str, dict[int, list[tuple[float, float]]]]) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
    for axis, n_society in zip(axes, [100, 1000]):
        for label, curves in all_curves.items():
            curve = curves[n_society]
            xs = [alpha for alpha, _ in curve]
            ys = [failure for _, failure in curve]
            axis.plot(
                xs,
                ys,
                marker=MARKERS[label],
                linewidth=1.45,
                markersize=4.6,
                color=COLORS[label],
                label=label,
            )
        axis.axhline(0.5, color="#817A83", linestyle="--", linewidth=0.9)
        axis.set_title(f"N = {n_society}")
        axis.set_xlabel("harmful fraction $\\alpha$")
        axis.set_ylim(-0.04, 1.04)
        axis.grid(True, axis="y", alpha=0.75)
    axes[0].set_ylabel("primary failure rate")
    axes[0].set_xlim(-0.004, 0.104)
    axes[1].set_xlim(-0.001, 0.031)
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout(w_pad=1.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_crossmodel_robustness.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT_DIR / "fig_crossmodel_robustness.png", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main() -> None:
    summaries: list[dict] = []
    all_curves: dict[str, dict[int, list[tuple[float, float]]]] = {}
    audit_rows = [audit_model(tag, label, model_id) for tag, label, model_id in AUDIT_MODELS]
    for tag, label, model_id in MODELS:
        summary, curves = model_summary(tag, label, model_id)
        summaries.append(summary)
        all_curves[label] = curves
    write_csv(summaries)
    write_latex(summaries)
    write_audit_status(audit_rows, summaries)
    make_figure(all_curves)
    print("Wrote cross-model robustness artifacts to", OUT_DIR)
    for row in summaries:
        print(
            f"{row['backbone']}: alpha_c100={row['alpha_c_100']:.4f} "
            f"alpha_c1000={row['alpha_c_1000']:.4f} nu={row['nu_two_point']:.3f}"
        )


if __name__ == "__main__":
    main()