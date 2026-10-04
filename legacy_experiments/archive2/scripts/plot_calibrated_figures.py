"""Clean-baseline calibrated figures for final mixed-agent experiments.

The raw WolfBench collapse trigger is intentionally broad. For S1/S2 it can
fire on clean social diffusion, so paper figures should report a
scenario-aligned primary-risk score and its excess over the alpha=0 mixed-agent
baseline. Raw binary failure rates remain in the original data/figures.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wolfbench-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.ticker import AutoMinorLocator

from .io_utils import OUTPUTS, ROOT, ensure_dir
from .plot_final_figures import (
    CMAP,
    INK,
    MUTED,
    PALETTE,
    QUOTA_ORDER,
    SCENARIO_LABELS,
    VARIANT_ORDER,
    clean_axis,
    label_scenario,
    set_style,
)


# Pink-purple ramp with monotonically increasing lightness so the response
# curves stay in the paper's rose/purple family yet remain separable under
# deuteranopia/protanopia (which conflate hue but preserve the luminance channel)
# and in grayscale print. Marker shape and line style add two redundant,
# hue-independent channels so the legend is unambiguous even without colour.
CURVE_COLORS = [
    "#4A1A6B",  # deep violet
    "#8E2A8F",  # purple-magenta
    "#C0399B",  # magenta-pink
    "#E87FB8",  # light orchid pink
    "#5B2C82",  # (spare) muted violet
    "#B23F9A",  # (spare) orchid
    "#D96BB0",  # (spare) rose
    "#F0A6CE",  # (spare) pale pink
]
CURVE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
CURVE_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


# Readable two-line x labels for the E3 social-variant bar chart.
VARIANT_LABELS = {
    "no_social": "No\nsocial",
    "baseline": "Baseline",
    "high_exposure": "High\nexposure",
    "high_reach": "High\nreach",
    "hub_placement": "Hub\nplacement",
    "reflexive": "Reflexive",
}


S1_THRESHOLDS = {
    "social_cascade": 0.55,
    "price_dislocation": 0.21,
    "liquidity_stress": 0.85,
}
S2_THRESHOLDS = {
    "social_cascade": 0.55,
    "price_dislocation": 0.35,
    "liquidity_stress": 1.45,
}
CALIBRATED_MARGIN = 0.20


def _scenario_key(value: Any) -> str:
    text = str(value).lower()
    if text.startswith("s1") or "pump" in text:
        return "s1"
    if text.startswith("s2") or "finfluencer" in text:
        return "s2"
    if text.startswith("s3") or "spoof" in text:
        return "s3"
    if text.startswith("s4") or "wash" in text:
        return "s4"
    return text


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _s1_s2_score(row: pd.Series, thresholds: dict[str, float]) -> float:
    social = _finite(row.get("social_cascade_peak")) / thresholds["social_cascade"]
    price = _finite(row.get("price_dislocation_max")) / thresholds["price_dislocation"]
    liquidity = _finite(row.get("liquidity_stress_max")) / thresholds["liquidity_stress"]
    return float(max(0.0, min(social, max(price, liquidity))))


def scenario_primary_score(row: pd.Series) -> float:
    scenario = _scenario_key(row.get("scenario", ""))
    if scenario == "s1":
        return _s1_s2_score(row, S1_THRESHOLDS)
    if scenario == "s2":
        return _s1_s2_score(row, S2_THRESHOLDS)
    return _finite(row.get("primary_failure_score_max"))


def read_frame(out_name: str) -> pd.DataFrame:
    path = OUTPUTS / out_name / "data.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "status" in frame.columns:
        frame = frame[frame["status"].fillna("ok").eq("ok")].copy()
    for column in frame.columns:
        if column in {"scenario", "variant", "defense", "status", "quota_variant", "primary_metric", "target_asset", "placement"}:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def baseline_keys(frame: pd.DataFrame, out_name: str) -> list[str]:
    keys = ["scenario"]
    for key in ["n_society", "variant", "defense", "quota_variant"]:
        if key in frame.columns and frame[key].nunique(dropna=False) > 1:
            keys.append(key)
    # E2/E5 should compare mechanisms/defenses within the same scenario and
    # population size; E1 additionally needs N-specific clean baselines.
    if "n_society" in frame.columns and "n_society" not in keys:
        keys.append("n_society")
    return keys


def add_calibrated_columns(frame: pd.DataFrame, out_name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["scenario_primary_score"] = frame.apply(scenario_primary_score, axis=1)
    frame["scenario_primary_failure"] = (frame["scenario_primary_score"] >= 1.0).astype(float)
    keys = baseline_keys(frame, out_name)
    clean = frame[np.isclose(frame["alpha"].astype(float), 0.0)]
    if clean.empty:
        frame["clean_baseline_score"] = 0.0
        frame["clean_baseline_std"] = 0.0
    else:
        stats = (
            clean.groupby(keys, dropna=False)["scenario_primary_score"]
            .agg(clean_baseline_score="mean", clean_baseline_std="std")
            .reset_index()
        )
        stats["clean_baseline_std"] = stats["clean_baseline_std"].fillna(0.0)
        frame = frame.merge(stats, on=keys, how="left")
        frame["clean_baseline_score"] = frame["clean_baseline_score"].fillna(0.0)
        frame["clean_baseline_std"] = frame["clean_baseline_std"].fillna(0.0)
    frame["excess_primary_score"] = (
        frame["scenario_primary_score"] - frame["clean_baseline_score"]
    ).clip(lower=0.0)
    clean_alpha = np.isclose(frame["alpha"].astype(float), 0.0)
    frame.loc[clean_alpha, "excess_primary_score"] = 0.0
    threshold = frame["clean_baseline_score"] + np.maximum(
        CALIBRATED_MARGIN,
        2.0 * frame["clean_baseline_std"],
    )
    frame["clean_calibrated_failure"] = (
        (frame["alpha"].astype(float) > 0.0)
        & (frame["scenario_primary_score"] >= threshold)
    ).astype(float)
    frame.loc[clean_alpha, "clean_calibrated_failure"] = 0.0
    return frame


def save(fig: plt.Figure, out_name: str, stem: str, paper_name: str | None = None) -> None:
    directories = [ensure_dir(OUTPUTS / out_name / "figures"), ensure_dir(ROOT / "figures")]
    paper_dir = ROOT.parent / "AuthorKit27" / "Figures"
    if paper_name is not None and paper_dir.exists():
        directories.append(ensure_dir(paper_dir))
    for directory in directories:
        target = paper_name if (paper_name is not None and directory == paper_dir) else stem
        fig.savefig(directory / f"{target}.png", dpi=400)
        fig.savefig(directory / f"{target}.pdf")
    plt.close(fig)


def _sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    try:
        return 0, float(text)
    except ValueError:
        pass
    if text.startswith("s") and len(text) > 1 and text[1].isdigit():
        return 1, int(text[1])
    return 2, text


def aggregate(frame: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(keys, dropna=False)[metric]
        .agg(mean="mean", sem=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0, n="size")
        .reset_index()
    )
    return grouped


def plot_curves(frame: pd.DataFrame, out_name: str, stem: str, group_key: str, title: str, metric: str, paper_name: str | None = None) -> None:
    data = aggregate(frame, [group_key, "alpha"], metric)
    if data.empty:
        return
    groups = sorted(data[group_key].dropna().unique(), key=_sort_key)
    fig, ax = plt.subplots(figsize=(4.6, 3.15))
    for idx, group in enumerate(groups):
        selected = data[data[group_key].eq(group)].sort_values("alpha")
        xs = selected["alpha"].astype(float).to_numpy()
        ys = selected["mean"].astype(float).to_numpy()
        sem = selected["sem"].astype(float).to_numpy()
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]
        marker = CURVE_MARKERS[idx % len(CURVE_MARKERS)]
        linestyle = CURVE_LINESTYLES[idx % len(CURVE_LINESTYLES)]
        label = label_scenario(str(group)) if group_key == "scenario" else str(group)
        # Semi-transparent error bars sit behind the line so overlapping
        # scenarios stay legible.
        ax.errorbar(
            xs, ys, yerr=sem, fmt="none", ecolor=to_rgba(color, 0.45),
            elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=2,
        )
        # Line carries hue + line style; markers add a redundant shape channel
        # so curves and legend remain separable in grayscale and under
        # colour-vision deficiency.
        ax.plot(
            xs, ys, color=color, linestyle=linestyle, linewidth=1.9,
            marker=marker, markersize=5.0, markeredgecolor=INK,
            markeredgewidth=0.5, zorder=3, label=label,
        )
    ax.set_xlabel(r"Harmful-agent ratio $\alpha$")
    ax.set_ylabel(
        "Calibrated collapse probability" if "failure" in metric
        else "Excess primary-risk score\n(threshold-normalized, dimensionless)"
    )
    if "failure" in metric:
        ax.set_ylim(-0.03, 1.16)
    else:
        lo = float((data["mean"] - data["sem"]).min())
        hi = float((data["mean"] + data["sem"]).max())
        span = max(hi - lo, 1e-6)
        ax.set_ylim(min(0.0, lo) - span * 0.06, hi + span * 0.30)
    ax.axhline(0.0, color=MUTED, linewidth=0.8, alpha=0.6)
    clean_axis(ax)
    ncol = 2 if len(groups) > 3 else 1
    ax.legend(
        frameon=False, ncol=ncol, loc="lower center", bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.4, handlelength=2.4, handletextpad=0.5, borderaxespad=0.2,
    )
    fig.subplots_adjust(top=0.86)
    save(fig, out_name, stem, paper_name=paper_name)


def _sigmoid(alpha: np.ndarray, slope: float, midpoint: float) -> np.ndarray:
    z = np.clip(slope * (alpha - midpoint), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_sigmoid_grid(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float] | None:
    if xs.size < 3 or np.allclose(ys, ys[0]):
        return None
    slopes = np.geomspace(20.0, 3000.0, 160)
    midpoints = np.linspace(float(xs.min()), float(xs.max()), 180)
    best: tuple[float, float, float] | None = None
    y = np.clip(ys.astype(float), 0.0, 1.0)
    for slope in slopes:
        preds = _sigmoid(xs[:, None], slope, midpoints[None, :])
        preds = np.clip(preds, 1e-6, 1 - 1e-6)
        losses = -(y[:, None] * np.log(preds) + (1 - y[:, None]) * np.log(1 - preds)).mean(axis=0)
        idx = int(np.argmin(losses))
        loss = float(losses[idx])
        if best is None or loss < best[0]:
            best = (loss, float(slope), float(midpoints[idx]))
    if best is None:
        return None
    return best[1], best[2]


def plot_e1_scurve(frame: pd.DataFrame, out_name: str) -> None:
    if "n_society" not in frame.columns:
        return
    metric = "clean_calibrated_failure"
    data = aggregate(frame, ["n_society", "alpha"], metric)
    if data.empty:
        return
    groups = sorted(data["n_society"].dropna().unique(), key=_sort_key)
    fig, ax = plt.subplots(figsize=(5.05, 3.05))
    for idx, group in enumerate(groups):
        selected = data[data["n_society"].eq(group)].sort_values("alpha")
        xs = selected["alpha"].astype(float).to_numpy()
        ys = selected["mean"].astype(float).to_numpy()
        ns = selected["n"].astype(float).to_numpy()
        ci = 1.96 * np.sqrt(np.maximum(ys * (1.0 - ys), 0.0) / np.maximum(ns, 1.0))
        color = PALETTE[idx % len(PALETTE)]
        ax.errorbar(
            xs,
            ys,
            yerr=ci,
            fmt="o",
            color=color,
            markersize=4.4,
            linewidth=1.2,
            capsize=2.2,
            label=None,
        )
        rows = frame[frame["n_society"].eq(group)].sort_values("alpha")
        fit = _fit_sigmoid_grid(rows["alpha"].astype(float).to_numpy(), rows[metric].astype(float).to_numpy())
        if fit is not None:
            slope, midpoint = fit
            smooth_x = np.linspace(float(xs.min()), float(xs.max()), 240)
            ax.plot(smooth_x, _sigmoid(smooth_x, slope, midpoint), color=color, linewidth=2.2, label=f"N={int(group)}")
            ax.axvline(midpoint, color=color, linewidth=0.9, alpha=0.28)
        else:
            ax.plot(xs, ys, color=color, linewidth=1.6, linestyle="--", alpha=0.8, label=f"N={int(group)}")
    ax.set_title("E1 collapse probability S-curve", loc="left", fontweight="bold", pad=4)
    ax.set_xlabel("Harmful-agent ratio alpha")
    ax.set_ylabel("Calibrated collapse probability")
    ax.set_ylim(-0.03, 1.03)
    clean_axis(ax)
    ax.legend(frameon=False, ncol=1, loc="upper left", title="Society size")
    save(fig, out_name, f"{out_name}_e1_scurve")


def plot_variant_bars(frame: pd.DataFrame, out_name: str, paper_name: str | None = None) -> None:
    data = aggregate(frame, ["variant"], "excess_primary_score")
    if data.empty:
        return
    labels = [item for item in VARIANT_ORDER if item in set(data["variant"])]
    values = [float(data.loc[data["variant"].eq(label), "mean"].iloc[0]) for label in labels]
    sems = [float(data.loc[data["variant"].eq(label), "sem"].iloc[0]) for label in labels]
    fig, ax = plt.subplots(figsize=(4.6, 3.15))
    x = np.arange(len(labels))
    # Colour encodes emphasis, not category (categories are on the x-axis):
    # context bars share one light-pink fill so the single deep-purple bar --
    # hub placement, the key finding -- draws the eye immediately. The social-off
    # control keeps a hatch so it still reads as a reference condition.
    neutral = "#F1B6D3"
    highlight = "#7E22CE"
    bars = ax.bar(
        x, values, yerr=sems, edgecolor="white",
        linewidth=0.8, capsize=2.6, error_kw={"elinewidth": 0.9},
        color=[highlight if label == "hub_placement" else neutral for label in labels],
    )
    for bar, label in zip(bars, labels):
        if label == "no_social":
            bar.set_hatch("////")
    # Every variant perturbs the baseline social loop, so a dashed guide at the
    # baseline value shows which levers rise above it.
    if "baseline" in labels:
        baseline_val = values[labels.index("baseline")]
        ax.axhline(baseline_val, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.85, zorder=1)
        ax.text(0.0, baseline_val, "baseline", ha="left", va="bottom", fontsize=6.8, color=MUTED)
    headroom = max([v + s for v, s in zip(values, sems)] + [0.1])
    # no_social is the social-off control; its excess risk is exactly zero (the
    # primary event needs both the social and market channels), so the bar is
    # invisible -- label it as a control anchor above the axis instead.
    if "no_social" in labels:
        xi_ns = x[labels.index("no_social")]
        ax.text(
            xi_ns, headroom * 0.62, "control\n(social off)", ha="center", va="center",
            fontsize=6.6, color=MUTED, style="italic", linespacing=1.15,
        )
    for xi, value, sem, label in zip(x, values, sems, labels):
        is_hub = label == "hub_placement"
        ax.text(
            xi, value + sem + headroom * 0.03, f"{value:.2f}", ha="center", va="bottom",
            fontsize=7.4, color=INK, fontweight="bold" if is_hub else "normal",
        )
        if is_hub:
            # Star flags that hub placement's bootstrap 95% CI excludes the
            # baseline CI (Table 6); the caption defines the current intervals.
            ax.text(
                xi, value + sem + headroom * 0.13, "*", ha="center", va="bottom",
                fontsize=13, color=highlight, fontweight="bold",
            )
    ax.set_ylim(0, headroom * 1.30)
    ax.set_ylabel("Excess primary-risk score\n(threshold-normalized, dimensionless)")
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS.get(label, label.replace("_", "\n")) for label in labels])
    clean_axis(ax)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=2.5, color=MUTED)
    save(fig, out_name, f"{out_name}_calibrated_social_variants", paper_name=paper_name)


def plot_quota(frame: pd.DataFrame, out_name: str, paper_name: str | None = None) -> None:
    data = aggregate(frame, ["scenario", "quota_variant"], "excess_primary_score")
    if data.empty:
        return
    scenarios = sorted(data["scenario"].dropna().unique(), key=_sort_key)
    x = np.arange(len(QUOTA_ORDER))
    fig, ax = plt.subplots(figsize=(4.9, 3.15))
    ymax = 0.0
    for idx, scenario in enumerate(scenarios):
        values = []
        sems = []
        for quota in QUOTA_ORDER:
            selected = data[data["scenario"].eq(scenario) & data["quota_variant"].eq(quota)]
            values.append(float(selected["mean"].iloc[0]) if not selected.empty else 0.0)
            sems.append(float(selected["sem"].iloc[0]) if not selected.empty else 0.0)
        color = PALETTE[idx % len(PALETTE)]
        offset = (idx - (len(scenarios) - 1) / 2.0) * 0.045
        ax.errorbar(
            x + offset, values, yerr=sems, color=color, marker="o", linewidth=1.9,
            markersize=4.4, markeredgecolor="white", markeredgewidth=0.6,
            elinewidth=0.9, capsize=2.0, capthick=0.9, label=label_scenario(str(scenario)),
        )
        ymax = max(ymax, max(v + s for v, s in zip(values, sems)))
    ax.set_ylabel("Excess primary-risk score")
    ax.set_xticks(x)
    ax.set_xticklabels([quota.title() for quota in QUOTA_ORDER])
    ax.set_xlim(-0.35, len(QUOTA_ORDER) - 0.65)
    ax.set_ylim(top=ymax * 1.24)
    clean_axis(ax)
    ax.legend(
        frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.4, handlelength=1.6, handletextpad=0.5, borderaxespad=0.2,
    )
    fig.subplots_adjust(top=0.86)
    save(fig, out_name, f"{out_name}_calibrated_quota", paper_name=paper_name)


def plot_failure_heatmap(frame: pd.DataFrame, out_name: str) -> None:
    if "defense" not in frame.columns:
        return
    data = aggregate(frame, ["defense", "scenario"], "clean_calibrated_failure")
    if data.empty:
        return
    defenses = sorted(data["defense"].dropna().unique(), key=str)
    scenarios = sorted(data["scenario"].dropna().unique(), key=_sort_key)
    matrix = np.zeros((len(defenses), len(scenarios)))
    for i, defense in enumerate(defenses):
        for j, scenario in enumerate(scenarios):
            selected = data[data["defense"].eq(defense) & data["scenario"].eq(scenario)]
            matrix[i, j] = float(selected["mean"].iloc[0]) if not selected.empty else 0.0
    fig, ax = plt.subplots(figsize=(4.95, max(2.55, 0.28 * len(defenses) + 0.95)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        annot=np.array([[f"{v:.2f}" for v in row] for row in matrix]),
        fmt="",
        linewidths=0.6,
        linecolor="white",
        cbar=True,
        annot_kws={"fontsize": 7.0},
    )
    ax.set_title("Clean-calibrated failure rate", loc="left", fontweight="bold", pad=4)
    ax.set_xticklabels([SCENARIO_LABELS.get(str(s), str(s)) for s in scenarios], rotation=18, ha="right")
    ax.set_yticklabels([str(d).replace("_", " ") for d in defenses], rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Failure rate", color=INK)
    cbar.outline.set_visible(False)
    save(fig, out_name, f"{out_name}_calibrated_failure_heatmap")


def plot_out(out_name: str) -> Path:
    set_style()
    frame = add_calibrated_columns(read_frame(out_name), out_name)
    out_csv = OUTPUTS / out_name / "data_calibrated.csv"
    frame.to_csv(out_csv, index=False)
    lower = out_name.lower()
    if "e1" in lower:
        plot_curves(frame, out_name, f"{out_name}_calibrated_scaling", "n_society", "E1 scaling, clean-normalized", "excess_primary_score")
        plot_curves(frame, out_name, f"{out_name}_calibrated_failure", "n_society", "E1 calibrated failure", "clean_calibrated_failure")
        plot_e1_scurve(frame, out_name)
    elif "e2" in lower:
        plot_curves(frame, out_name, f"{out_name}_calibrated_mechanisms", "scenario", "E2 mechanisms, clean-normalized", "excess_primary_score", paper_name="figure4_e2_mixed_mechanisms")
        plot_curves(frame, out_name, f"{out_name}_calibrated_failure", "scenario", "E2 calibrated failure", "clean_calibrated_failure")
    elif "e3" in lower:
        plot_variant_bars(frame, out_name, paper_name="figure5_e3_mixed_social")
    elif "e4" in lower:
        plot_quota(frame, out_name, paper_name="figure6_e4_quota")
    elif "e5" in lower:
        plot_failure_heatmap(frame, out_name)
    else:
        plot_curves(frame, out_name, f"{out_name}_calibrated_curves", "scenario", "Mixed-agent experiment, clean-normalized", "excess_primary_score")
    return out_csv


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
    seen: set[str] = set()
    for out_name in outs:
        if out_name in seen:
            continue
        seen.add(out_name)
        out_csv = plot_out(out_name)
        print(f"Calibrated {out_name}: {out_csv}")


if __name__ == "__main__":
    main()
