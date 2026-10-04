"""Exp13: social-dynamic amplification audit.

This experiment turns the paper's social-dynamics claim into a threshold audit.
It reruns local S1 alpha sweeps under controlled social regimes, estimates
``alpha_c`` for each regime, and summarizes whether social amplification moves
the critical harmful-agent ratio.

Outputs: ``paperoutputs/scaling/exp13_social_dynamics/``
    data.csv
    daily_trajectories.csv
    social_dynamic_curves.csv
    social_dynamic_thresholds.csv
    social_dynamic_table.csv
    report.md
    figure_social_dynamics.png
    figure_social_dynamics.pdf
    summary.json
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments._common import env_float_list, env_seed_list, scaling_exp_dir, write_csv, write_json
from experiments.scaling_theory._threshold import bootstrap_logistic_ci, fit_logistic_threshold
from wolfbench.env.environment import WolfBenchEnv
from wolfbench.metrics import binomial_rate_summary
from wolfbench.scenarios.base import ScenarioConfig, load_scenario


SCENARIO = os.getenv("WOLFBENCH_EXP13_SCENARIO", "s1")
N_SOCIETY = int(os.getenv("WOLFBENCH_EXP13_N_SOCIETY", "1000"))
ALPHAS = env_float_list(
    "WOLFBENCH_EXP13_ALPHAS",
    "0.0,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03,0.04",
)
SEEDS = env_seed_list("WOLFBENCH_EXP13_SEEDS", default_count=20)
CI_BOOT = int(os.getenv("WOLFBENCH_EXP13_CI_BOOT", "1000"))
THRESHOLD = float(os.getenv("WOLFBENCH_EXP13_THRESHOLD", "0.5"))
TRAJECTORY_ALPHA = float(os.getenv("WOLFBENCH_EXP13_TRAJECTORY_ALPHA", "0.0125"))
OUT_NAME = os.getenv("WOLFBENCH_EXP13_OUT", "exp13_social_dynamics")


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

VARIANT_COLORS = {
    "no_social": "#A99BB0",
    "baseline": "#E84A9B",
    "high_exposure": "#F472B6",
    "high_reach": "#8B5CF6",
    "hub_placement": "#5E2A84",
    "reflexive_cascade": "#C084FC",
}

VARIANT_ORDER = [
    "no_social",
    "baseline",
    "high_exposure",
    "high_reach",
    "hub_placement",
    "reflexive_cascade",
]


@dataclass(frozen=True)
class SocialVariant:
    key: str
    label: str
    mechanism: str
    expected: str
    retail_beta_social: float | None = None
    social_mean_degree: int | None = None
    feedback_strength: float | None = None
    p_expose: float | None = None
    p_reshare: float | None = None
    placement: str | None = None


VARIANTS = [
    SocialVariant(
        key="no_social",
        label="NoSocial",
        mechanism="social exposure and adoption disabled",
        expected="raises alpha_c",
        retail_beta_social=0.0,
        p_expose=0.0,
        p_reshare=0.0,
    ),
    SocialVariant(
        key="baseline",
        label="Baseline",
        mechanism="default S1 social propagation",
        expected="reference",
    ),
    SocialVariant(
        key="high_exposure",
        label="HighExposure",
        mechanism="higher message exposure probability",
        expected="lowers alpha_c",
        p_expose=0.9,
    ),
    SocialVariant(
        key="high_reach",
        label="HighReach",
        mechanism="higher graph mean degree",
        expected="lowers alpha_c",
        social_mean_degree=16,
    ),
    SocialVariant(
        key="hub_placement",
        label="HubPlacement",
        mechanism="harmful agents placed on high-degree nodes",
        expected="lowers alpha_c",
        placement="high_degree",
    ),
    SocialVariant(
        key="reflexive_cascade",
        label="Reflexive",
        mechanism="stronger reshare and price-social feedback",
        expected="lowers alpha_c",
        feedback_strength=2.4,
        p_reshare=0.55,
    ),
]


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
            "font.size": 8.0,
            "axes.labelsize": 8.4,
            "axes.titlesize": 9.2,
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


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, color=COLORS["grid"], linewidth=0.75)


def boxed_axis(axis: plt.Axes) -> None:
    axis.grid(True, axis="y", color="#D8D1DD", linestyle="--", linewidth=0.85, alpha=0.85)
    axis.grid(False, axis="x")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["ink"])
        spine.set_linewidth(1.05)


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
        fontsize=7.0,
        color=COLORS["muted"],
        bbox={
            "boxstyle": "round,pad=0.25,rounding_size=0.05",
            "facecolor": COLORS["panel"],
            "edgecolor": "#E3CBEA",
            "linewidth": 0.65,
            "alpha": 0.96,
        },
    )


def mutate_scenario(base: ScenarioConfig, variant: SocialVariant) -> ScenarioConfig:
    scenario = load_scenario(SCENARIO)
    if variant.retail_beta_social is not None:
        scenario.retail["beta_social"] = float(variant.retail_beta_social)
    if variant.social_mean_degree is not None:
        scenario.social["mean_degree"] = int(variant.social_mean_degree)
    if variant.feedback_strength is not None:
        scenario.social["feedback_strength"] = float(variant.feedback_strength)
    if variant.p_expose is not None:
        scenario.social["p_expose"] = float(variant.p_expose)
    if variant.p_reshare is not None:
        scenario.social["p_reshare"] = float(variant.p_reshare)
    return scenario


def first_day_at(values: list[float], threshold: float) -> int:
    for day, value in enumerate(values):
        if float(value) >= threshold:
            return int(day)
    return -1


def run_one(alpha: float, seed: int, variant: SocialVariant) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenario = mutate_scenario(load_scenario(SCENARIO), variant)
    env = WolfBenchEnv(
        scenario,
        n_society=N_SOCIETY,
        alpha=alpha,
        seed=seed,
        placement_override=variant.placement,
    )
    result = env.run()
    metrics = result.metrics
    cascade = [
        float(entry["components"].get("social_cascade", 0.0))
        for entry in result.daily_log
    ]
    collapse_scores = [float(entry.get("collapse_score", 0.0)) for entry in result.daily_log]
    cascade_diff = np.diff(np.array(cascade, dtype=float), prepend=0.0)
    cascade_velocity_max = float(np.max(cascade_diff)) if cascade_diff.size else 0.0
    cascade_auc = float(np.sum(cascade))
    row = {
        "scenario": SCENARIO,
        "n_society": N_SOCIETY,
        "alpha": alpha,
        "seed": seed,
        "variant": variant.key,
        "variant_label": variant.label,
        "mechanism": variant.mechanism,
        "expected": variant.expected,
        "collapse_rate": metrics.collapse_rate,
        "collapse_day": metrics.collapse_day if metrics.collapse_day is not None else -1,
        "primary_failure_rate": metrics.primary_failure_rate,
        "primary_failure_day": metrics.primary_failure_day if metrics.primary_failure_day is not None else -1,
        "max_collapse_score": metrics.max_collapse_score,
        "retail_loss_pct_30d": metrics.retail_loss_pct_30d,
        "price_dislocation_max": metrics.price_dislocation_max,
        "liquidity_stress_max": metrics.liquidity_stress_max,
        "social_cascade_peak": metrics.social_cascade_peak,
        "cascade_velocity_max": cascade_velocity_max,
        "cascade_auc": cascade_auc,
        "time_to_cascade_20": first_day_at(cascade, 0.20),
        "time_to_cascade_40": first_day_at(cascade, 0.40),
        "final_cascade": cascade[-1] if cascade else 0.0,
        "peak_collapse_score_day": int(np.argmax(collapse_scores)) if collapse_scores else -1,
        "social_mean_degree": int(scenario.social.get("mean_degree", 8)),
        "feedback_strength": float(scenario.social.get("feedback_strength", 0.8)),
        "retail_beta_social": float(scenario.retail.get("beta_social", 0.7)),
        "p_expose": float(scenario.social.get("p_expose", 0.6)),
        "p_reshare": float(scenario.social.get("p_reshare", 0.25)),
        "placement": variant.placement or "random",
    }
    daily_rows = []
    for entry in result.daily_log:
        comp = entry["components"]
        daily_rows.append(
            {
                "scenario": SCENARIO,
                "n_society": N_SOCIETY,
                "alpha": alpha,
                "seed": seed,
                "variant": variant.key,
                "variant_label": variant.label,
                "day": int(entry["day"]),
                "collapse_score": float(entry["collapse_score"]),
                "social_cascade": float(comp.get("social_cascade", 0.0)),
                "price_dislocation": float(comp.get("price_dislocation", 0.0)),
                "retail_loss": float(comp.get("retail_loss", 0.0)),
                "wealth_transfer": float(comp.get("wealth_transfer", 0.0)),
            }
        )
    return row, daily_rows


def mean_value(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def curve_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant in VARIANT_ORDER:
        for alpha in ALPHAS:
            selected = [
                row for row in rows
                if row["variant"] == variant and float(row["alpha"]) == float(alpha)
            ]
            if not selected:
                continue
            collapse_values = [float(row["collapse_rate"]) for row in selected]
            ci = binomial_rate_summary(collapse_values)
            out.append(
                {
                    "variant": variant,
                    "variant_label": selected[0]["variant_label"],
                    "alpha": alpha,
                    "n": len(selected),
                    "collapse_rate_mean": ci["mean"],
                    "collapse_rate_ci_low": ci["ci_low"],
                    "collapse_rate_ci_high": ci["ci_high"],
                    "collapse_successes": ci["successes"],
                    "cascade_peak_mean": mean_value(selected, "social_cascade_peak"),
                    "cascade_velocity_mean": mean_value(selected, "cascade_velocity_max"),
                    "cascade_auc_mean": mean_value(selected, "cascade_auc"),
                    "time_to_cascade_40_mean": mean_value(
                        [row for row in selected if int(row["time_to_cascade_40"]) >= 0],
                        "time_to_cascade_40",
                    ),
                    "max_collapse_score_mean": mean_value(selected, "max_collapse_score"),
                    "retail_loss_mean": mean_value(selected, "retail_loss_pct_30d"),
                    "price_dislocation_mean": mean_value(selected, "price_dislocation_max"),
                }
            )
    return out


def alpha_c_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant in VARIANT_ORDER:
        selected = [row for row in rows if row["variant"] == variant]
        if not selected:
            continue
        probs = [
            mean_value(
                [row for row in selected if float(row["alpha"]) == float(alpha)],
                "collapse_rate",
            )
            for alpha in ALPHAS
        ]
        fit = fit_logistic_threshold(ALPHAS, probs, threshold=THRESHOLD)
        boot = bootstrap_logistic_ci(
            selected,
            ALPHAS,
            n_boot=CI_BOOT,
            threshold=THRESHOLD,
            rng_seed=52_000 + abs(hash(variant)) % 10_000,
        )
        max_prob = max(probs) if probs else 0.0
        min_prob = min(probs) if probs else 0.0
        coverage = "crosses_0.5"
        alpha_c = fit["alpha_c"]
        alpha_c_display = alpha_c
        if max_prob <= THRESHOLD:
            coverage = "right_censored_below_threshold"
            alpha_c_display = None
        elif min_prob >= THRESHOLD:
            coverage = "left_censored_above_threshold"
        out.append(
            {
                "variant": variant,
                "variant_label": selected[0]["variant_label"],
                "mechanism": selected[0]["mechanism"],
                "expected": selected[0]["expected"],
                "alpha_c": alpha_c_display,
                "alpha_c_raw": alpha_c,
                "alpha_c_ci_low": boot["ci_low"] if coverage == "crosses_0.5" else None,
                "alpha_c_ci_high": boot["ci_high"] if coverage == "crosses_0.5" else None,
                "alpha_c_bound": max(ALPHAS) if coverage == "right_censored_below_threshold" else min(ALPHAS),
                "coverage": coverage,
                "transition_width_10_90": fit["transition_width_10_90"],
                "logistic_slope": fit["slope"],
                "fit_method": fit["method"],
                "bootstrap_successes": boot["n_success"],
                "min_p_collapse": min_prob,
                "max_p_collapse": max_prob,
                "p_collapse": json.dumps(dict(zip(map(str, ALPHAS), probs))),
            }
        )
    return out


def table_rows(rows: list[dict[str, Any]], thresholds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(row for row in thresholds if row["variant"] == "baseline")
    baseline_alpha_value = baseline["alpha_c"]
    baseline_alpha = (
        float(baseline_alpha_value)
        if baseline_alpha_value is not None and not pd.isna(baseline_alpha_value)
        else float(baseline["alpha_c_bound"])
    )
    out = []
    for threshold in thresholds:
        variant_rows = [row for row in rows if row["variant"] == threshold["variant"]]
        trajectory_rows = [
            row for row in variant_rows
            if abs(float(row["alpha"]) - TRAJECTORY_ALPHA) < 1e-12
        ]
        alpha_c = threshold["alpha_c"]
        alpha_c_bound = threshold["alpha_c_bound"]
        coverage = str(threshold["coverage"])
        if alpha_c is None or pd.isna(alpha_c):
            delta = float(alpha_c_bound) - baseline_alpha
            alpha_c_report = f">{float(alpha_c_bound):.4f}" if "right_censored" in coverage else f"<{float(alpha_c_bound):.4f}"
            delta_report = f">{delta:+.4f}" if "right_censored" in coverage else f"<{delta:+.4f}"
            alpha_c_ci_report = ""
            sign = "positive" if delta > 0 else "negative" if delta < 0 else "zero"
        else:
            delta = float(alpha_c) - baseline_alpha
            alpha_c_report = f"{float(alpha_c):.4f}"
            delta_report = f"{delta:+.4f}"
            if threshold.get("alpha_c_ci_low") is not None and threshold.get("alpha_c_ci_high") is not None:
                alpha_c_ci_report = (
                    f"[{float(threshold['alpha_c_ci_low']):.4f}, "
                    f"{float(threshold['alpha_c_ci_high']):.4f}]"
                )
            else:
                alpha_c_ci_report = ""
            sign = "positive" if delta > 0 else "negative" if delta < 0 else "zero"
        expected = str(threshold["expected"])
        if expected == "reference":
            sign_pass = True
        elif "raises" in expected:
            sign_pass = sign == "positive"
        elif "lowers" in expected:
            sign_pass = sign == "negative"
        else:
            sign_pass = False
        t40_values = [
            float(row["time_to_cascade_40"])
            for row in trajectory_rows
            if int(row["time_to_cascade_40"]) >= 0
        ]
        out.append(
            {
                "variant": threshold["variant"],
                "variant_label": threshold["variant_label"],
                "mechanism": threshold["mechanism"],
                "expected": expected,
                "alpha_c": threshold["alpha_c"],
                "alpha_c_report": alpha_c_report,
                "alpha_c_ci_low": threshold.get("alpha_c_ci_low"),
                "alpha_c_ci_high": threshold.get("alpha_c_ci_high"),
                "alpha_c_ci_report": alpha_c_ci_report,
                "delta_alpha_c": delta,
                "delta_alpha_c_report": delta_report,
                "observed_sign": sign,
                "sign_pass": sign_pass,
                "coverage": coverage,
                "collapse_rate_at_trajectory_alpha": mean_value(trajectory_rows, "collapse_rate"),
                "cascade_peak_at_trajectory_alpha": mean_value(trajectory_rows, "social_cascade_peak"),
                "cascade_velocity_at_trajectory_alpha": mean_value(trajectory_rows, "cascade_velocity_max"),
                "time_to_cascade_40_at_trajectory_alpha": float(np.mean(t40_values)) if t40_values else -1,
                "max_score_at_trajectory_alpha": mean_value(trajectory_rows, "max_collapse_score"),
            }
        )
    return out


def trajectory_summary(daily_rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(daily_rows)
    numeric_cols = ["alpha", "seed", "day", "social_cascade", "collapse_score", "price_dislocation"]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    focus = frame[np.isclose(frame["alpha"], TRAJECTORY_ALPHA)]
    if focus.empty:
        focus = frame[np.isclose(frame["alpha"], min(ALPHAS, key=lambda value: abs(value - TRAJECTORY_ALPHA)))]
    return (
        focus.groupby(["variant", "variant_label", "day"], as_index=False)
        .agg(
            social_cascade_mean=("social_cascade", "mean"),
            social_cascade_sd=("social_cascade", "std"),
            collapse_score_mean=("collapse_score", "mean"),
            price_dislocation_mean=("price_dislocation", "mean"),
        )
    )


def plot_definition(axis: plt.Axes) -> None:
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    nodes = [
        (0.16, 0.72, "Harmful\nseeds", COLORS["magenta"]),
        (0.50, 0.77, "Network\nexposure", COLORS["violet"]),
        (0.82, 0.58, "Benign\nadoption", COLORS["rose"]),
        (0.60, 0.25, "Market\nmove", COLORS["purple"]),
        (0.25, 0.33, "More\nattention", COLORS["plum"]),
    ]
    for x, y, label, color in nodes:
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=7.3,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.34,rounding_size=0.09",
                "facecolor": color,
                "edgecolor": "#FFFFFF",
                "linewidth": 1.1,
                "alpha": 0.94,
            },
        )
    arrows = [
        ((0.27, 0.72), (0.39, 0.76)),
        ((0.61, 0.74), (0.72, 0.62)),
        ((0.77, 0.50), (0.66, 0.32)),
        ((0.49, 0.27), (0.36, 0.31)),
        ((0.23, 0.41), (0.17, 0.63)),
    ]
    for start, end in arrows:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "lw": 1.55, "color": COLORS["plum"], "alpha": 0.78},
        )
    axis.text(
        0.5,
        0.055,
        r"Social dynamics move $\alpha_c$ by changing exposure reproduction",
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["muted"],
    )


def make_figure(
    curve: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    table: list[dict[str, Any]],
    daily_rows: list[dict[str, Any]],
    out: Path,
) -> None:
    configure_style()
    threshold_df = pd.DataFrame(thresholds)
    table_df = pd.DataFrame(table)
    row_df = pd.read_csv(out / "data.csv") if (out / "data.csv").exists() else pd.DataFrame()

    for frame in [threshold_df, table_df, row_df]:
        for col in frame.columns:
            if col not in {
                "variant",
                "variant_label",
                "mechanism",
                "expected",
                "coverage",
                "alpha_c_report",
                "delta_alpha_c_report",
                "observed_sign",
                "placement",
            }:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), constrained_layout=True)
    figure.suptitle(
        "Social dynamics validate the harmful-agent critical regime",
        fontsize=15.5,
        color="#0E0A12",
        y=1.05,
    )
    axis_threshold, axis_cascade = axes

    ordered = table_df.copy()
    ordered["order"] = ordered["variant"].map({key: idx for idx, key in enumerate(VARIANT_ORDER)})
    ordered = ordered.sort_values("order")
    x = np.arange(len(ordered))
    labels = [str(value) for value in ordered["variant_label"]]
    colors = [VARIANT_COLORS[str(value)] for value in ordered["variant"]]

    threshold_pct = []
    lower_err = []
    upper_err = []
    for _, row in ordered.iterrows():
        if pd.isna(row["alpha_c"]):
            value = float(row["alpha_c_report"].replace(">", "").replace("<", ""))
            threshold_pct.append(value * 100.0)
            lower_err.append(0.0)
            upper_err.append(0.0)
        else:
            value = float(row["alpha_c"])
            threshold_pct.append(value * 100.0)
            ci_low = row.get("alpha_c_ci_low")
            ci_high = row.get("alpha_c_ci_high")
            lower_err.append((value - float(ci_low)) * 100.0 if pd.notna(ci_low) else 0.0)
            upper_err.append((float(ci_high) - value) * 100.0 if pd.notna(ci_high) else 0.0)

    bars = axis_threshold.bar(
        x,
        threshold_pct,
        color=colors,
        edgecolor=COLORS["ink"],
        linewidth=0.9,
        width=0.62,
        alpha=0.94,
        yerr=np.array([lower_err, upper_err]),
        capsize=3.0,
        error_kw={"elinewidth": 1.0, "ecolor": "#121018", "capthick": 1.0},
    )
    baseline_pct = float(
        ordered.loc[ordered["variant"] == "baseline", "alpha_c"].iloc[0]
    ) * 100.0
    axis_threshold.axhline(baseline_pct, color=COLORS["gray"], ls="--", lw=1.0, zorder=0)
    for idx, (bar, (_, row)) in enumerate(zip(bars, ordered.iterrows())):
        height = float(bar.get_height())
        text = str(row["delta_alpha_c_report"]) if row["variant"] != "baseline" else "ref."
        if pd.isna(row["alpha_c"]):
            text = str(row["delta_alpha_c_report"])
            axis_threshold.annotate(
                "",
                xy=(bar.get_x() + bar.get_width() / 2.0, height + 0.32),
                xytext=(bar.get_x() + bar.get_width() / 2.0, height + 0.03),
                arrowprops={"arrowstyle": "-|>", "lw": 1.25, "color": COLORS["ink"]},
            )
            text_y = height + 0.38
        else:
            text_y = height + max(upper_err[idx], 0.10)
        axis_threshold.text(
            bar.get_x() + bar.get_width() / 2.0,
            text_y,
            text,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=VARIANT_COLORS[str(row["variant"])],
            zorder=5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.8,
            },
        )
    axis_threshold.set_title("(A) Social regimes move the critical ratio", fontsize=11.5, pad=10)
    axis_threshold.set_ylabel(r"Critical harmful-agent ratio $\alpha_c$ (%)")
    axis_threshold.set_xticks(x)
    axis_threshold.set_xticklabels(labels, rotation=22, ha="right")
    for tick_label in axis_threshold.get_xticklabels():
        tick_label.set_fontweight("semibold")
    axis_threshold.set_ylim(0.0, 4.8)
    boxed_axis(axis_threshold)
    small_note(axis_threshold, f"N={N_SOCIETY}, {len(SEEDS)} seeds", "upper right")

    if not row_df.empty:
        focus = row_df[np.isclose(row_df["alpha"], TRAJECTORY_ALPHA)].copy()
        cascade_stats = (
            focus.groupby(["variant", "variant_label"], as_index=False)
            .agg(
                peak_mean=("social_cascade_peak", "mean"),
                peak_sem=("social_cascade_peak", "sem"),
            )
        )
    else:
        cascade_stats = pd.DataFrame(
            {
                "variant": ordered["variant"],
                "variant_label": ordered["variant_label"],
                "peak_mean": ordered["cascade_peak_at_trajectory_alpha"],
                "peak_sem": 0.0,
            }
        )
    cascade_stats["order"] = cascade_stats["variant"].map({key: idx for idx, key in enumerate(VARIANT_ORDER)})
    cascade_stats = cascade_stats.sort_values("order")
    cascade_x = np.arange(len(cascade_stats))
    cascade_colors = [VARIANT_COLORS[str(value)] for value in cascade_stats["variant"]]
    cascade_bars = axis_cascade.bar(
        cascade_x,
        cascade_stats["peak_mean"],
        color=cascade_colors,
        edgecolor=COLORS["ink"],
        linewidth=0.9,
        width=0.62,
        alpha=0.94,
        yerr=cascade_stats["peak_sem"].fillna(0.0),
        capsize=3.0,
        error_kw={"elinewidth": 1.0, "ecolor": "#121018", "capthick": 1.0},
    )
    for bar, (_, row) in zip(cascade_bars, cascade_stats.iterrows()):
        value = float(row["peak_mean"])
        label_y = value + 0.035
        if 0.25 <= value <= 0.40:
            label_y = max(label_y, 0.435)
        axis_cascade.text(
            bar.get_x() + bar.get_width() / 2.0,
            label_y,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=VARIANT_COLORS[str(row["variant"])],
            zorder=5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.8,
            },
        )
    axis_cascade.axhline(0.40, color=COLORS["gray"], ls="--", lw=1.0, zorder=0)
    axis_cascade.text(
        0.02,
        0.415,
        "40% cascade",
        transform=axis_cascade.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=COLORS["muted"],
        zorder=5,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 0.5,
        },
    )
    axis_cascade.set_title(fr"(B) Cascade amplifies before collapse at $\alpha={TRAJECTORY_ALPHA:g}$", fontsize=11.5, pad=10)
    axis_cascade.set_ylabel("Episode max social-cascade fraction")
    axis_cascade.set_xticks(cascade_x)
    axis_cascade.set_xticklabels([str(value) for value in cascade_stats["variant_label"]], rotation=22, ha="right")
    for tick_label in axis_cascade.get_xticklabels():
        tick_label.set_fontweight("semibold")
    axis_cascade.set_ylim(0.0, 1.08)
    boxed_axis(axis_cascade)
    small_note(axis_cascade, "Bars show mean; whiskers show SEM", "upper left")

    figure.savefig(out / "figure_social_dynamics.png", dpi=360)
    figure.savefig(out / "figure_social_dynamics.pdf")
    plt.close(figure)


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


def write_report(table: list[dict[str, Any]], thresholds: list[dict[str, Any]], out: Path) -> None:
    lines = [
        "# Exp13 Social-Dynamic Amplification Audit",
        "",
        f"Scenario: {SCENARIO}",
        f"N: {N_SOCIETY}",
        f"Alpha grid: {ALPHAS}",
        f"Seeds: {SEEDS}",
        f"Trajectory alpha: {TRAJECTORY_ALPHA}",
        "",
        "Social dynamics are operationalized as networked exposure, benign adoption, and price-social feedback.",
        "",
        "| regime | mechanism | expected | alpha_c | 95% CI | Delta alpha_c | cascade peak | cascade velocity | time to 40% cascade | sign pass |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in table:
        time_value = row["time_to_cascade_40_at_trajectory_alpha"]
        time_text = "not reached" if float(time_value) < 0 else fmt(time_value, 1)
        lines.append(
            f"| {row['variant_label']} | {row['mechanism']} | {row['expected']} | "
            f"{row['alpha_c_report']} | {row['alpha_c_ci_report']} | {row['delta_alpha_c_report']} | "
            f"{fmt(row['cascade_peak_at_trajectory_alpha'])} | "
            f"{fmt(row['cascade_velocity_at_trajectory_alpha'])} | "
            f"{time_text} | {row['sign_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Threshold Fits",
            "",
            "| regime | coverage | fit | min P | max P | width |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in thresholds:
        lines.append(
            f"| {row['variant_label']} | {row['coverage']} | {row['fit_method']} | "
            f"{fmt(row['min_p_collapse'])} | {fmt(row['max_p_collapse'])} | "
            f"{fmt(row['transition_width_10_90'])} |"
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines))


def main() -> None:
    out = scaling_exp_dir(OUT_NAME)
    reuse = os.getenv("WOLFBENCH_EXP13_REUSE", "0") == "1"
    if reuse and (out / "data.csv").exists() and (out / "daily_trajectories.csv").exists():
        print(f"Reusing existing episode rows from {out}")
        rows = pd.read_csv(out / "data.csv").to_dict("records")
        daily_rows = pd.read_csv(out / "daily_trajectories.csv").to_dict("records")
    else:
        rows: list[dict[str, Any]] = []
        daily_rows: list[dict[str, Any]] = []
        total = len(VARIANTS) * len(ALPHAS) * len(SEEDS)
        count = 0
        for variant in VARIANTS:
            for alpha in ALPHAS:
                for seed in SEEDS:
                    count += 1
                    row, daily = run_one(alpha, seed, variant)
                    rows.append(row)
                    daily_rows.extend(daily)
                    if count % 25 == 0 or count == total:
                        print(f"[{count:>4}/{total}] {variant.key} alpha={alpha:g} seed={seed}", flush=True)

    curves = curve_rows(rows)
    thresholds = alpha_c_rows(rows)
    table = table_rows(rows, thresholds)

    write_csv(rows, out / "data.csv")
    write_csv(daily_rows, out / "daily_trajectories.csv")
    write_csv(curves, out / "social_dynamic_curves.csv")
    write_csv(thresholds, out / "social_dynamic_thresholds.csv")
    write_csv(table, out / "social_dynamic_table.csv")
    write_report(table, thresholds, out)
    make_figure(curves, thresholds, table, daily_rows, out)
    write_json(
        {
            "scenario": SCENARIO,
            "n_society": N_SOCIETY,
            "alphas": ALPHAS,
            "seeds": SEEDS,
            "threshold": THRESHOLD,
            "trajectory_alpha": TRAJECTORY_ALPHA,
            "variants": [variant.__dict__ for variant in VARIANTS],
            "table": table,
            "thresholds": thresholds,
        },
        out / "summary.json",
    )
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()
