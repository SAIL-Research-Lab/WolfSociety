"""Exp12: canonical four-scenario scaling evidence.

This experiment is the paper-facing scaling cleanup pass. It estimates one
scenario-aligned primary failure curve for each canonical scenario and society
size, then fits alpha_c and transition width with the shared logistic helper.

S1/S2 use the generic collapse trigger. S3 uses the spoofing/liquidity primary
failure signal. S4 uses the fake-liquidity primary failure signal; generic
collapse remains exported as a diagnostic field, but is not the main S4 claim.

Outputs: ``paperoutputs/scaling/exp12_canonical_scaling_refined/``
    data.csv
    failure_curves.csv
    alpha_c_by_scenario_n.csv
    width_by_scenario_n.csv
    scenario_law_summary.csv
    report.md
    summary.json
    primary_failure_curves.png
    transition_width_by_n.png
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import (
    RunSpec,
    env_float_list,
    env_int_list,
    env_list,
    env_seed_list,
    run_grid,
    scaling_exp_dir,
    write_csv,
    write_json,
)
from experiments.scaling_theory._threshold import (
    bootstrap_logistic_ci,
    fit_logistic_threshold,
    linear_alpha_c,
)
from wolfbench.metrics import binomial_rate_summary


DEFAULT_ALPHA_GRIDS = {
    "s1": "0.0,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03,0.04",
    "s2": "0.0,0.00004,0.00005,0.00006,0.00007,0.000075,0.00008,0.00009,0.0001,0.000125,0.00015,0.000175,0.0002,0.000225,0.00025,0.000275,0.0003,0.00035,0.0004,0.00045,0.0005,0.00055,0.0006,0.00065,0.0007,0.00075,0.0008,0.00085,0.0009,0.00095,0.001,0.00105,0.0011,0.00115,0.0012,0.00125,0.0013,0.0014,0.0015,0.00175,0.002,0.0025,0.003,0.004,0.005",
    "s3": "0.0,0.0005,0.001,0.0015,0.002,0.0025,0.003,0.004,0.005,0.006,0.0075,0.009,0.01,0.0125,0.015,0.0175,0.02,0.0225,0.025,0.0275,0.03,0.0325,0.035,0.0375,0.04,0.0425,0.045,0.0475,0.05,0.0525,0.055,0.0575,0.06,0.0625,0.065,0.0675,0.07,0.0725,0.075,0.0775,0.08,0.0825,0.085,0.0875,0.09,0.095,0.1,0.11,0.125,0.15",
    "s4": "0.0,0.01,0.0125,0.015,0.0175,0.02,0.0225,0.025,0.0275,0.03,0.035,0.04,0.05,0.075",
}

PRIMARY_METRIC_LABELS = {
    "s1": "generic_collapse",
    "s2": "generic_collapse",
    "s3": "spoof_liquidity_failure",
    "s4": "fake_liquidity_failure",
}

SCENARIOS = env_list("WOLFBENCH_EXP12_SCENARIOS", "s1,s2,s3,s4")
N_GRID = env_int_list("WOLFBENCH_EXP12_N_GRID", "500,1000,2000,10000")
SEEDS = env_seed_list("WOLFBENCH_EXP12_SEEDS", default_count=30)
CI_BOOT = int(os.getenv("WOLFBENCH_EXP12_CI_BOOT", "1000"))
OUT_NAME = os.getenv("WOLFBENCH_EXP12_OUT", "exp12_canonical_scaling_refined")
PROGRESS_EVERY = int(os.getenv("WOLFBENCH_EXP12_PROGRESS_EVERY", "50"))


def _alphas_for(scenario: str) -> list[float]:
    key = f"WOLFBENCH_EXP12_ALPHAS_{scenario.upper()}"
    default = os.getenv("WOLFBENCH_EXP12_ALPHAS") or DEFAULT_ALPHA_GRIDS[scenario]
    return env_float_list(key, default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _stable_seed(scenario: str, n_society: int) -> int:
    scenario_code = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(scenario)))
    return int(120_000 + scenario_code + int(n_society))


def _selected(rows: list[dict[str, Any]], scenario: str, n_society: int,
              alpha: float | None = None) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row["scenario"] == scenario and int(row["n_society"]) == int(n_society)
    ]
    if alpha is not None:
        selected = [row for row in selected if float(row["alpha"]) == float(alpha)]
    return selected


def _coverage_status(probs: list[float]) -> str:
    if not probs:
        return "empty"
    p_min = min(probs)
    p_max = max(probs)
    if p_min < 0.5 <= p_max:
        return "crosses_0.5"
    if p_max < 0.5:
        return "right_censored_below_threshold"
    if p_min >= 0.5:
        return "left_censored_above_threshold"
    return "noisy_or_flat"


def build_failure_curve_rows(rows: list[dict[str, Any]],
                             alpha_grids: dict[str, list[float]],
                             scenarios: list[str] | None = None,
                             n_grid: list[int] | None = None) -> list[dict[str, Any]]:
    curve_rows: list[dict[str, Any]] = []
    scenario_list = scenarios if scenarios is not None else SCENARIOS
    n_values = n_grid if n_grid is not None else N_GRID
    for scenario in scenario_list:
        for n_society in n_values:
            for alpha in alpha_grids[scenario]:
                selected = _selected(rows, scenario, n_society, alpha)
                primary_values = [_float(row.get("primary_failure_rate")) for row in selected]
                collapse_values = [_float(row.get("collapse_rate")) for row in selected]
                primary_ci = binomial_rate_summary(primary_values)
                collapse_ci = binomial_rate_summary(collapse_values)
                curve_rows.append({
                    "scenario": scenario,
                    "n_society": n_society,
                    "alpha": alpha,
                    "primary_metric": PRIMARY_METRIC_LABELS.get(scenario, "generic_collapse"),
                    "primary_failure_mean": primary_ci["mean"],
                    "primary_failure_n": primary_ci["n"],
                    "primary_failure_successes": primary_ci["successes"],
                    "primary_failure_ci_low": primary_ci["ci_low"],
                    "primary_failure_ci_high": primary_ci["ci_high"],
                    "primary_failure_score_mean": _mean([
                        _float(row.get("primary_failure_score_max")) for row in selected
                    ]),
                    "collapse_rate_mean": collapse_ci["mean"],
                    "collapse_rate_ci_low": collapse_ci["ci_low"],
                    "collapse_rate_ci_high": collapse_ci["ci_high"],
                    "max_collapse_score_mean": _mean([_float(row.get("max_collapse_score")) for row in selected]),
                    "retail_loss_pct_mean": _mean([_float(row.get("retail_loss_pct_30d")) for row in selected]),
                    "liquidity_stress_mean": _mean([_float(row.get("liquidity_stress_max")) for row in selected]),
                    "cancel_rate_mean": _mean([_float(row.get("cancel_rate_max")) for row in selected]),
                    "spoof_depth_to_liquidity_mean": _mean([
                        _float(row.get("spoof_depth_to_liquidity_max")) for row in selected
                    ]),
                    "wash_share_mean": _mean([_float(row.get("wash_share_max")) for row in selected]),
                    "volume_distortion_mean": _mean([_float(row.get("volume_distortion_max")) for row in selected]),
                    "withdrawal_loss_mean": _mean([_float(row.get("withdrawal_loss_max")) for row in selected]),
                })
    return curve_rows


def _bootstrap_primary_ci(rows: list[dict[str, Any]], alphas: list[float],
                          seed: int) -> dict[str, Any]:
    boot_rows = []
    for row in rows:
        copied = dict(row)
        copied["collapse_rate"] = _float(row.get("primary_failure_rate"))
        boot_rows.append(copied)
    return bootstrap_logistic_ci(boot_rows, alphas, n_boot=CI_BOOT, rng_seed=seed)


def estimate_threshold_rows(rows: list[dict[str, Any]],
                            alpha_grids: dict[str, list[float]],
                            scenarios: list[str] | None = None,
                            n_grid: list[int] | None = None) -> list[dict[str, Any]]:
    threshold_rows: list[dict[str, Any]] = []
    scenario_list = scenarios if scenarios is not None else SCENARIOS
    n_values = n_grid if n_grid is not None else N_GRID
    for scenario in scenario_list:
        alphas = alpha_grids[scenario]
        for n_society in n_values:
            rows_n = _selected(rows, scenario, n_society)
            probs = []
            for alpha in alphas:
                selected = _selected(rows, scenario, n_society, alpha)
                probs.append(_mean([_float(row.get("primary_failure_rate")) for row in selected]))
            fit = fit_logistic_threshold(alphas, probs)
            ci = _bootstrap_primary_ci(
                rows_n,
                alphas,
                seed=_stable_seed(scenario, n_society),
            )
            threshold_rows.append({
                "scenario": scenario,
                "n_society": n_society,
                "primary_metric": PRIMARY_METRIC_LABELS.get(scenario, "generic_collapse"),
                "alpha_c_logistic": fit["alpha_c"],
                "alpha_c_ci_low": ci["ci_low"],
                "alpha_c_ci_high": ci["ci_high"],
                "alpha_c_linear": linear_alpha_c(alphas, probs),
                "logistic_slope": fit["slope"],
                "transition_width_10_90": fit["transition_width_10_90"],
                "fit_method": fit["method"],
                "bootstrap_successes": ci["n_success"],
                "coverage_status": _coverage_status(probs),
                "p_min": min(probs) if probs else 0.0,
                "p_max": max(probs) if probs else 0.0,
                "p_delta": (max(probs) - min(probs)) if probs else 0.0,
                "alpha_grid": ";".join(str(alpha) for alpha in alphas),
            })
    return threshold_rows


def _log_slope(xs: list[float], ys: list[float]) -> float | None:
    valid = [(float(x), float(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(valid) < 2:
        return None
    x_arr = np.log([x for x, _ in valid])
    y_arr = np.log([y for _, y in valid])
    slope, _ = np.polyfit(x_arr, y_arr, 1)
    return float(slope)


def build_width_rows(threshold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scenario": row["scenario"],
            "n_society": row["n_society"],
            "primary_metric": row["primary_metric"],
            "transition_width_10_90": row["transition_width_10_90"],
            "alpha_c_logistic": row["alpha_c_logistic"],
            "coverage_status": row["coverage_status"],
            "fit_method": row["fit_method"],
        }
        for row in threshold_rows
    ]


def build_scenario_law_summary(threshold_rows: list[dict[str, Any]],
                               curve_rows: list[dict[str, Any]],
                               scenarios: list[str] | None = None,
                               n_grid: list[int] | None = None) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    scenario_list = scenarios if scenarios is not None else SCENARIOS
    n_values = n_grid if n_grid is not None else N_GRID
    for scenario in scenario_list:
        rows = sorted(
            [row for row in threshold_rows if row["scenario"] == scenario],
            key=lambda row: int(row["n_society"]),
        )
        widths = [
            _float(row["transition_width_10_90"], default=float("nan"))
            for row in rows
            if row["transition_width_10_90"] not in {None, ""}
        ]
        ns_for_width = [
            int(row["n_society"])
            for row in rows
            if row["transition_width_10_90"] not in {None, ""}
        ]
        complete_alpha_c = all(row["alpha_c_logistic"] not in {None, ""} for row in rows)
        crossing_count = sum(row["coverage_status"] == "crosses_0.5" for row in rows)
        mean_delta = _mean([_float(row["p_delta"]) for row in rows])
        width_slope = _log_slope(ns_for_width, widths) if len(widths) >= 2 else None
        width_narrows = bool(width_slope is not None and width_slope < 0.0)

        if scenario == "s4":
            if complete_alpha_c and mean_delta >= 0.5:
                grade = "mechanism-strong"
            elif mean_delta >= 0.35:
                grade = "mechanism-partial"
            else:
                grade = "mechanism-inconclusive"
            caveat = "S4 is evaluated with fake-liquidity primary failure; generic collapse width is diagnostic only."
        elif complete_alpha_c and width_narrows and crossing_count == len(rows):
            grade = "strong"
            caveat = "All tested N values cross the primary-failure threshold and fitted width narrows with N."
        elif crossing_count >= max(1, len(rows) - 1) and (width_narrows or scenario == "s3"):
            grade = "partial"
            caveat = "Threshold evidence is present, but one N or width trend needs denser calibration."
        else:
            grade = "inconclusive"
            caveat = "Alpha grid or primary-failure threshold needs calibration before making a strong law claim."

        summary_rows.append({
            "scenario": scenario,
            "primary_metric": PRIMARY_METRIC_LABELS.get(scenario, "generic_collapse"),
            "n_grid": ";".join(str(n) for n in n_values),
            "alpha_c_complete": complete_alpha_c,
            "crossing_count": crossing_count,
            "n_count": len(rows),
            "mean_curve_delta": mean_delta,
            "width_slope_loglog": width_slope,
            "width_narrows": width_narrows,
            "evidence_grade": grade,
            "caveat": caveat,
        })
    return summary_rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(threshold_rows: list[dict[str, Any]],
                 scenario_rows: list[dict[str, Any]], path) -> None:
    lines = [
        "# Exp12 Canonical Scaling Evidence",
        "",
        "Primary-failure convention: S1/S2 use generic collapse, S3 uses spoofing/liquidity failure, and S4 uses fake-liquidity failure.",
        "Generic collapse remains in data.csv and failure_curves.csv as a diagnostic field.",
        "",
        f"Scenarios: {', '.join(SCENARIOS)}",
        f"N grid: {N_GRID}",
        f"Seeds: {SEEDS}",
        f"CI bootstrap draws: {CI_BOOT}",
        "",
        "## Scenario Summary",
        "",
        "| scenario | primary metric | crossings | mean curve delta | width slope | grade | caveat |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in scenario_rows:
        lines.append(
            f"| {row['scenario'].upper()} | {row['primary_metric']} | "
            f"{row['crossing_count']}/{row['n_count']} | {_fmt(row['mean_curve_delta'])} | "
            f"{_fmt(row['width_slope_loglog'])} | {row['evidence_grade']} | {row['caveat']} |"
        )
    lines.extend([
        "",
        "## Alpha-C By Scenario And N",
        "",
        "| scenario | N | alpha_c | 95% CI | width 10-90 | coverage | fit |",
        "|---|---:|---:|---|---:|---|---|",
    ])
    for row in threshold_rows:
        ci = f"[{_fmt(row['alpha_c_ci_low'])}, {_fmt(row['alpha_c_ci_high'])}]"
        lines.append(
            f"| {row['scenario'].upper()} | {row['n_society']} | {_fmt(row['alpha_c_logistic'])} | "
            f"{ci} | {_fmt(row['transition_width_10_90'])} | {row['coverage_status']} | {row['fit_method']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def plot_failure_curves(curve_rows: list[dict[str, Any]], path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    axes_flat = axes.ravel()
    for axis, scenario in zip(axes_flat, SCENARIOS):
        for n_society in N_GRID:
            rows = [
                row for row in curve_rows
                if row["scenario"] == scenario and int(row["n_society"]) == int(n_society)
            ]
            rows.sort(key=lambda row: float(row["alpha"]))
            if not rows:
                continue
            xs = [float(row["alpha"]) for row in rows]
            ys = [float(row["primary_failure_mean"]) for row in rows]
            axis.plot(xs, ys, marker="o", linewidth=1.7, label=f"N={n_society}")
            lows = [float(row["primary_failure_ci_low"]) for row in rows]
            highs = [float(row["primary_failure_ci_high"]) for row in rows]
            axis.fill_between(xs, lows, highs, alpha=0.12)
        axis.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.55)
        axis.set_xscale("symlog", linthresh=1e-4)
        axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel("alpha")
        axis.set_ylabel("P(primary failure)")
        axis.set_title(f"{scenario.upper()} - {PRIMARY_METRIC_LABELS.get(scenario, 'generic_collapse')}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes_flat[len(SCENARIOS):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_widths(width_rows: list[dict[str, Any]], path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    positive_widths: list[float] = []
    for scenario in SCENARIOS:
        rows = [row for row in width_rows if row["scenario"] == scenario]
        rows.sort(key=lambda row: int(row["n_society"]))
        xs = [int(row["n_society"]) for row in rows]
        ys = [
            _float(row["transition_width_10_90"], default=float("nan"))
            for row in rows
        ]
        positive_widths.extend([y for y in ys if np.isfinite(y) and y > 0.0])
        ax.plot(xs, ys, marker="o", linewidth=1.7, label=scenario.upper())
    ax.set_xscale("log")
    if positive_widths:
        ax.set_yscale("log")
    else:
        ax.text(
            0.5,
            0.5,
            "No positive fitted widths in this run",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    ax.set_xlabel("Society size N")
    ax.set_ylabel("Primary-failure transition width")
    ax.set_title("Exp12 primary-failure transition width by N")
    ax.grid(which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    out = scaling_exp_dir(OUT_NAME)
    alpha_grids = {scenario: _alphas_for(scenario) for scenario in SCENARIOS}
    specs = [
        RunSpec(scenario, n_society, alpha, seed)
        for scenario in SCENARIOS
        for n_society in N_GRID
        for alpha in alpha_grids[scenario]
        for seed in SEEDS
    ]
    print(f"Running {len(specs)} episodes for exp12 canonical scaling...")
    rows = run_grid(specs, progress_every=PROGRESS_EVERY)
    curve_rows = build_failure_curve_rows(rows, alpha_grids)
    threshold_rows = estimate_threshold_rows(rows, alpha_grids)
    width_rows = build_width_rows(threshold_rows)
    scenario_rows = build_scenario_law_summary(threshold_rows, curve_rows)

    write_csv(rows, out / "data.csv")
    write_csv(curve_rows, out / "failure_curves.csv")
    write_csv(threshold_rows, out / "alpha_c_by_scenario_n.csv")
    write_csv(width_rows, out / "width_by_scenario_n.csv")
    write_csv(scenario_rows, out / "scenario_law_summary.csv")
    write_report(threshold_rows, scenario_rows, out / "report.md")
    plot_failure_curves(curve_rows, out / "primary_failure_curves.png")
    plot_widths(width_rows, out / "transition_width_by_n.png")
    write_json({
        "scenarios": SCENARIOS,
        "n_grid": N_GRID,
        "alpha_grids": alpha_grids,
        "seeds": SEEDS,
        "ci_boot": CI_BOOT,
        "primary_metrics": PRIMARY_METRIC_LABELS,
        "threshold_summary": threshold_rows,
        "scenario_law_summary": scenario_rows,
        "outputs": {
            "data": "data.csv",
            "failure_curves": "failure_curves.csv",
            "alpha_c_by_scenario_n": "alpha_c_by_scenario_n.csv",
            "width_by_scenario_n": "width_by_scenario_n.csv",
            "scenario_law_summary": "scenario_law_summary.csv",
            "report": "report.md",
            "primary_failure_curves": "primary_failure_curves.png",
            "transition_width_by_n": "transition_width_by_n.png",
        },
    }, out / "summary.json")
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()