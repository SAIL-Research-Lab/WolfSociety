"""Publication-style pink/purple figures for final mixed-agent experiments."""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wolfbench-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

from .io_utils import OUTPUTS, ROOT, ensure_dir


INK = "#24112F"
MUTED = "#6B5873"
GRID = "#E9D7F0"
PANEL = "#FFF8FC"
PINK = "#DB2777"
ROSE = "#F472B6"
MAGENTA = "#C026D3"
PURPLE = "#7E22CE"
DEEP = "#4C1D95"
VIOLET = "#A855F7"
LAVENDER = "#E9D5FF"
GRAY = "#9C89A8"

PALETTE = [DEEP, PURPLE, MAGENTA, PINK, ROSE, VIOLET, "#B83280", "#6D28D9", "#E879F9", "#BE185D"]
CMAP = LinearSegmentedColormap.from_list("wolfbench_rose_purple", ["#FFF1F7", "#F9A8D4", "#C026D3", "#4C1D95"])

SCENARIO_LABELS = {
    "s1": "S1 Pump",
    "s1_pump_dump": "S1 Pump",
    "s2": "S2 Finfluencer",
    "s2_finfluencer": "S2 Finfluencer",
    "s3": "S3 Spoofing",
    "s3_spoofing": "S3 Spoofing",
    "s4": "S4 Wash",
    "s4_wash": "S4 Wash",
}

VARIANT_ORDER = ["no_social", "baseline", "high_exposure", "high_reach", "hub_placement", "reflexive"]
QUOTA_ORDER = ["low", "standard", "high"]


def set_style() -> None:
    """Unified paper style shared by all WolfBench paper figures.

    Single source of truth for font sizes, line widths, colours, grid, and
    DPI so every figure in the submission looks like one coherent set.
    """
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=PALETTE,
    )
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "font.family": "DejaVu Sans",
        "font.size": 8.4,
        "axes.labelsize": 8.8,
        "axes.titlesize": 9.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.2,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.6,
        "lines.markeredgewidth": 0.6,
        "patch.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, axis="y")
    axis.grid(False, axis="x")


def read_rows(out_name: str) -> list[dict[str, Any]]:
    path = OUTPUTS / out_name / "data.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_frame(out_name: str) -> pd.DataFrame:
    rows = read_rows(out_name)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in frame.columns:
        if column in {"scenario", "variant", "defense", "status", "quota_variant", "primary_metric", "target_asset", "placement"}:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="ignore")
    return frame


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def label_scenario(value: str) -> str:
    return SCENARIO_LABELS.get(value, value.replace("_", " ").title())


def mean_by(rows: Iterable[dict[str, Any]], keys: list[str], value: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        buckets[tuple(row.get(key, "") for key in keys)].append(f(row, value))
    out = []
    for key, values in buckets.items():
        record = {name: val for name, val in zip(keys, key)}
        record[value] = float(np.mean(values)) if values else 0.0
        record[f"{value}_stderr"] = float(np.std(values) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        record["n"] = len(values)
        out.append(record)
    return out


def save(fig: plt.Figure, out_name: str, stem: str) -> None:
    out_dir = ensure_dir(OUTPUTS / out_name / "figures")
    root_dir = ensure_dir(ROOT / "figures")
    for directory in (out_dir, root_dir):
        fig.savefig(directory / f"{stem}.png", dpi=360)
        fig.savefig(directory / f"{stem}.pdf")
    plt.close(fig)


def plot_curves(
    rows: list[dict[str, Any]],
    out_name: str,
    stem: str,
    group_key: str,
    title: str,
    y_key: str = "primary_failure_rate",
) -> None:
    frame = pd.DataFrame(mean_by(rows, [group_key, "alpha"], y_key))
    if frame.empty:
        return
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    frame[y_key] = pd.to_numeric(frame[y_key], errors="coerce")
    groups = sorted(frame[group_key].dropna().unique(), key=_sort_key)
    fig, ax = plt.subplots(figsize=(4.85, 3.05))
    for idx, group in enumerate(groups):
        selected = frame[frame[group_key] == group].sort_values("alpha")
        xs = selected["alpha"].astype(float).to_numpy()
        ys = selected[y_key].astype(float).to_numpy()
        color = PALETTE[idx % len(PALETTE)]
        label = label_scenario(str(group)) if group_key == "scenario" else str(group)
        sns.lineplot(x=xs, y=ys, ax=ax, color=color, marker="o", markersize=4.2, linewidth=2.0, label=label)
        ax.fill_between(xs, np.maximum(0, ys - 0.025), np.minimum(1, ys + 0.025), color=color, alpha=0.08, linewidth=0)
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    ax.set_xlabel("Harmful-agent ratio alpha")
    ax.set_ylabel("Primary failure rate")
    ax.set_ylim(-0.03, 1.03)
    clean_axis(ax)
    ax.legend(frameon=False, ncol=2 if len(groups) > 3 else 1, loc="upper left")
    save(fig, out_name, stem)


def plot_bars(
    rows: list[dict[str, Any]],
    out_name: str,
    stem: str,
    x_key: str,
    title: str,
    order: list[str] | None = None,
    y_key: str = "primary_failure_rate",
) -> None:
    data = mean_by(rows, [x_key], y_key)
    if order is None:
        labels = sorted([str(row[x_key]) for row in data], key=_sort_key)
    else:
        labels = [item for item in order if any(str(row[x_key]) == item for row in data)]
    values = [next((f(row, y_key) for row in data if str(row[x_key]) == label), 0.0) for label in labels]
    fig, ax = plt.subplots(figsize=(4.85, 3.05))
    frame = pd.DataFrame({x_key: labels, y_key: values})
    sns.barplot(
        data=frame,
        x=x_key,
        y=y_key,
        hue=x_key,
        ax=ax,
        palette=PALETTE[:len(labels)],
        width=0.72,
        edgecolor="white",
        linewidth=0.8,
        legend=False,
    )
    bars = ax.patches
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=7.4, color=INK)
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    ax.set_ylabel("Primary failure rate")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([label.replace("_", "\n") for label in labels])
    ax.set_ylim(0, min(1.05, max(values + [0.2]) + 0.18))
    clean_axis(ax)
    save(fig, out_name, stem)


def plot_quota(rows: list[dict[str, Any]], out_name: str) -> None:
    data = mean_by(rows, ["scenario", "quota_variant"], "primary_failure_rate")
    scenarios = sorted({row["scenario"] for row in data}, key=_sort_key)
    x = np.arange(len(QUOTA_ORDER))
    fig, ax = plt.subplots(figsize=(5.15, 3.05))
    width = 0.18
    for idx, scenario in enumerate(scenarios):
        values = [
            next((f(row, "primary_failure_rate") for row in data if row["scenario"] == scenario and row["quota_variant"] == quota), 0.0)
            for quota in QUOTA_ORDER
        ]
        ax.plot(x, values, color=PALETTE[idx % len(PALETTE)], marker="o", linewidth=1.8, label=label_scenario(str(scenario)))
    ax.set_title("E4 quota robustness", loc="left", fontweight="bold", pad=4)
    ax.set_ylabel("Primary failure rate")
    ax.set_xticks(x)
    ax.set_xticklabels([q.title() for q in QUOTA_ORDER])
    ax.set_ylim(-0.03, 1.03)
    clean_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, out_name, f"{out_name}_quota_robustness")


def plot_e5_heatmap(rows: list[dict[str, Any]], out_name: str) -> None:
    data = mean_by(rows, ["defense", "scenario"], "primary_failure_rate")
    defenses = sorted({str(row["defense"]) for row in data}, key=_defense_sort)
    scenarios = sorted({str(row["scenario"]) for row in data}, key=_sort_key)
    matrix = np.zeros((len(defenses), len(scenarios)), dtype=float)
    for i, defense in enumerate(defenses):
        for j, scenario in enumerate(scenarios):
            matrix[i, j] = next(
                (f(row, "primary_failure_rate") for row in data if row["defense"] == defense and row["scenario"] == scenario),
                0.0,
            )
    fig, ax = plt.subplots(figsize=(4.95, max(2.55, 0.26 * len(defenses) + 0.95)))
    labels = np.array([[f"{matrix[i, j]:.2f}" for j in range(len(scenarios))] for i in range(len(defenses))])
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        annot=labels,
        fmt="",
        cbar=True,
        linewidths=0.6,
        linecolor="white",
        annot_kws={"fontsize": 7.1},
    )
    ax.set_title("E5 WolfGuard benchmark", loc="left", fontweight="bold", pad=4)
    ax.set_xticklabels([label_scenario(s) for s in scenarios], rotation=18, ha="right")
    ax.set_yticklabels([d.replace("_", " ") for d in defenses], rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Primary failure rate", color=INK)
    cbar.ax.tick_params(labelsize=6.8)
    cbar.ax.yaxis.label.set_size(7.4)
    cbar.outline.set_visible(False)
    save(fig, out_name, f"{out_name}_defense_heatmap")


def plot_cost_panel(rows: list[dict[str, Any]], out_name: str) -> None:
    data = mean_by(rows, ["scenario"], "population_llm_episode_calls")
    scenarios = sorted({row["scenario"] for row in data}, key=_sort_key)
    values = [next((f(row, "population_llm_episode_calls") for row in data if row["scenario"] == scenario), 0.0) for scenario in scenarios]
    fig, ax = plt.subplots(figsize=(4.85, 2.55))
    ax.bar(range(len(scenarios)), values, color=[PALETTE[i % len(PALETTE)] for i in range(len(scenarios))], edgecolor="white", linewidth=0.8)
    ax.set_title("LLM calls per episode", loc="left", fontweight="bold", pad=4)
    ax.set_ylabel("Population calls")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([label_scenario(s) for s in scenarios], rotation=12, ha="right")
    clean_axis(ax)
    save(fig, out_name, f"{out_name}_llm_budget")


def plot_out(out_name: str) -> None:
    set_style()
    rows = read_rows(out_name)
    if not rows:
        return
    lower = out_name.lower()
    if "e1" in lower:
        plot_curves(rows, out_name, f"{out_name}_scaling_curves", "n_society", "E1 scaling")
    elif "e2" in lower:
        plot_curves(rows, out_name, f"{out_name}_mechanism_curves", "scenario", "E2 mechanisms")
    elif "e3" in lower:
        plot_bars(rows, out_name, f"{out_name}_social_variants", "variant", "E3 social dynamics", order=VARIANT_ORDER)
    elif "e4" in lower:
        plot_quota(rows, out_name)
    elif "e5" in lower:
        plot_e5_heatmap(rows, out_name)
    else:
        plot_curves(rows, out_name, f"{out_name}_curves", "scenario", "Mixed-agent experiment")
    plot_cost_panel(rows, out_name)


def _sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    try:
        return 0, float(text)
    except ValueError:
        pass
    if text.startswith("s") and len(text) > 1 and text[1].isdigit():
        return 1, int(text[1])
    return 2, text


def _defense_sort(value: str) -> tuple[int, str]:
    priority = {
        "noguard": 0,
        "zscore_guard": 1,
        "rule": 2,
        "topology_aware": 3,
        "oracle": 4,
        "deepseek_v3_risk": 5,
        "deepseek_v4_risk": 6,
        "qwen36_35b_risk": 7,
        "glm52_risk": 8,
        "llama4_maverick_risk": 9,
        "qwen235b_risk": 10,
        "glm45_risk": 11,
        "llama33_70b_risk": 12,
        "gemini25_pro_risk": 13,
        "gpt41_risk": 14,
        "claude_opus_risk": 15,
    }
    return priority.get(value, 50), value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    outs = list(args.out)
    if args.all:
        outs.extend(sorted(path.parent.name for path in OUTPUTS.glob("*/data.csv")))
    if not outs:
        raise SystemExit("Pass --out NAME or --all")
    seen = set()
    for out_name in outs:
        if out_name in seen:
            continue
        seen.add(out_name)
        plot_out(out_name)
        print(f"Plotted {out_name}")


if __name__ == "__main__":
    main()
