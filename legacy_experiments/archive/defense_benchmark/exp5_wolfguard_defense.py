"""Experiment 5: defense threshold-shift sweep.

This experiment estimates alpha_c(defense) - alpha_c(NoGuard) on targeted
critical-regime grids. It is the defense-side counterpart to the scaling-law
experiments: a useful defense should move the estimated critical harmful-agent
ratio to the right while keeping utility, false-positive, and intervention
costs small.

Environment controls:
    WOLFBENCH_EXP5_SCENARIOS=s1,s2
    WOLFBENCH_EXP5_DEFENSES=noguard,rule,random
    WOLFBENCH_EXP5_N_GRID=1000
    WOLFBENCH_EXP5_SEEDS=1,2,3,4,5
    WOLFBENCH_EXP5_ALPHAS_S1=0.0,0.0075,...
    WOLFBENCH_EXP5_ALPHAS_S2=0.0,0.00025,...
    WOLFBENCH_EXP5_OUT=exp5_wolfguard_defense

Outputs: paperoutputs/benchmark/<out>/
    data.csv
    alpha_curves.csv
    threshold_shift_summary.csv
    report.md
    summary.json
    collapse_curves.png
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import (
    RunSpec,
    benchmark_exp_dir,
    env_float_list,
    env_int_list,
    env_list,
    run_grid,
    write_csv,
    write_json,
)
from wolfbench.defense import get_policy, get_track
from wolfbench.metrics import bootstrap_ci, defense_score, threshold_shift
from wolfbench.tracks.runner import calibrate_clean_baseline


DEFAULT_ALPHA_GRIDS = {
    "s1": "0.0,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03",
    "s2": "0.0,0.00025,0.0005,0.00075,0.001,0.0015,0.002,0.0025",
    "s3": "0.0,0.15,0.3,0.4,0.5,0.6",
    "s4": "0.0,0.01,0.015,0.02,0.03,0.05,0.1,0.15,0.2",
}

SCENARIOS = env_list("WOLFBENCH_EXP5_SCENARIOS", "s1,s2")
DEFENSES = env_list("WOLFBENCH_EXP5_DEFENSES", "noguard,rule,random")
N_GRID = env_int_list("WOLFBENCH_EXP5_N_GRID", "1000")
SEEDS = env_int_list("WOLFBENCH_EXP5_SEEDS", "1,2,3,4,5")
CI_BOOT = int(os.getenv("WOLFBENCH_EXP5_CI_BOOT", "2000"))
OUT_NAME = os.getenv("WOLFBENCH_EXP5_OUT", "exp5_wolfguard_defense")


def _alphas_for(scenario: str) -> list[float]:
    key = f"WOLFBENCH_EXP5_ALPHAS_{scenario.upper()}"
    default = os.getenv("WOLFBENCH_EXP5_ALPHAS") or DEFAULT_ALPHA_GRIDS.get(
        scenario,
        "0.0,0.005,0.01,0.02,0.05,0.1",
    )
    return env_float_list(key, default)


def _defense_names() -> list[str]:
    names: list[str] = []
    for name in ["noguard", *DEFENSES]:
        key = name.strip().lower()
        if key and key not in names:
            names.append(key)
    return names


def _build_specs(scenario: str, n_society: int, defense_name: str) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for alpha in _alphas_for(scenario):
        for seed in SEEDS:
            policy = None if defense_name == "noguard" else get_policy(defense_name)
            specs.append(RunSpec(
                scenario=scenario,
                n_society=n_society,
                alpha=alpha,
                seed=seed,
                defense=defense_name != "noguard",
                defense_policy=policy,
                label=defense_name,
            ))
    return specs


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def build_alpha_curve_rows(
    rows: list[dict[str, Any]],
    alpha_grids: dict[str, list[float]],
    scenarios: list[str],
    n_grid: list[int],
    defenses: list[str],
) -> list[dict[str, Any]]:
    """Aggregate per-alpha collapse curves for plotting and diagnostics."""
    out: list[dict[str, Any]] = []
    for scenario in scenarios:
        for n_society in n_grid:
            for defense_name in defenses:
                for alpha in alpha_grids[scenario]:
                    selected = [
                        row for row in rows
                        if row["scenario_id"] == scenario
                        and int(row["n_society"]) == int(n_society)
                        and row["defense"] == defense_name
                        and float(row["alpha"]) == float(alpha)
                    ]
                    if not selected:
                        continue
                    collapse_values = [float(row["collapse_rate"]) for row in selected]
                    ci_low, ci_high = bootstrap_ci(collapse_values, n_boot=CI_BOOT)
                    out.append({
                        "scenario": scenario,
                        "n_society": int(n_society),
                        "defense": defense_name,
                        "track": get_track(defense_name),
                        "alpha": alpha,
                        "n": len(selected),
                        "collapse_rate_mean": float(np.mean(collapse_values)),
                        "collapse_rate_ci_low": ci_low,
                        "collapse_rate_ci_high": ci_high,
                        "retail_loss_mean": _mean(selected, "retail_loss_pct_30d"),
                        "utility_loss_mean": _mean(selected, "utility_loss"),
                        "false_positive_rate_mean": _mean(selected, "false_positive_rate"),
                        "intervention_cost_mean": _mean(selected, "intervention_cost"),
                    })
    return out


def summarize_threshold_shift(
    rows: list[dict[str, Any]],
    defenses: list[str],
    scenarios: list[str],
    n_grid: list[int],
    alpha_grids: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Return one threshold-shift row per scenario, N, and defense."""
    out: list[dict[str, Any]] = []
    for scenario in scenarios:
        alphas = alpha_grids[scenario]
        for n_society in n_grid:
            rows_no = [
                row for row in rows
                if row["scenario_id"] == scenario
                and int(row["n_society"]) == int(n_society)
                and row["defense"] == "noguard"
            ]
            for defense_name in defenses:
                rows_def = [
                    row for row in rows
                    if row["scenario_id"] == scenario
                    and int(row["n_society"]) == int(n_society)
                    and row["defense"] == defense_name
                ]
                if not rows_no or not rows_def:
                    continue
                score = defense_score(rows_no, rows_def, alphas=alphas)
                shift = threshold_shift(rows_no, rows_def, alphas)
                out.append({
                    "scenario": scenario,
                    "n_society": int(n_society),
                    "defense": defense_name,
                    "track": get_track(defense_name),
                    "alpha_c_no_def": shift["alpha_c_no_def"],
                    "alpha_c_def": shift["alpha_c_def"],
                    "threshold_shift": shift["threshold_shift"],
                    "threshold_shift_raw_or_bound": score["threshold_shift_raw"],
                    "threshold_shift_normalized": score["threshold_shift_normalized"],
                    "defense_score": score["defense_score"],
                    "delta_collapse": score["delta_collapse"],
                    "delta_harm_reduction": score["delta_harm_reduction"],
                    "gated_delta_harm_reduction": score["gated_delta_harm_reduction"],
                    "delta_delay_days": score["delta_delay_days"],
                    "no_def_collapse_rate": score["no_def_collapse_rate"],
                    "def_collapse_rate": score["def_collapse_rate"],
                    "no_def_retail_loss": score["no_def_retail_loss"],
                    "def_retail_loss": score["def_retail_loss"],
                    "utility_loss": score["utility_loss"],
                    "false_positive_rate": score["false_positive_rate"],
                    "intervention_cost": score["intervention_cost"],
                })
    return out


def _write_dict_csv(rows: list[dict[str, Any]], path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_report(summary_rows: list[dict[str, Any]], alpha_grids: dict[str, list[float]], path) -> None:
    lines = [
        "# Exp5 Defense Threshold-Shift Sweep",
        "",
        f"Scenarios: {', '.join(SCENARIOS)}",
        f"Defenses: {', '.join(_defense_names())}",
        f"N grid: {N_GRID}",
        f"Seeds: {SEEDS}",
        f"Alpha grids: {alpha_grids}",
        "",
        "| scenario | N | defense | alpha_c NoGuard | alpha_c defense | ThresholdShift | raw/bound shift | DefenseScore | Delta collapse | Utility | FP | Intervention |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['scenario']} | {row['n_society']} | {row['defense']} | "
            f"{_fmt(row['alpha_c_no_def'])} | {_fmt(row['alpha_c_def'])} | "
            f"{_fmt(row['threshold_shift'])} | {_fmt(row['threshold_shift_raw_or_bound'])} | "
            f"{_fmt(row['defense_score'], 2)} | {_fmt(row['delta_collapse'], 3)} | "
            f"{_fmt(row['utility_loss'], 3)} | {_fmt(row['false_positive_rate'], 3)} | "
            f"{_fmt(row['intervention_cost'], 3)} |"
        )
    lines.extend([
        "",
        "Notes:",
        "- ThresholdShift is alpha_c(defense) - alpha_c(NoGuard) when both critical points are observed inside the tested grid.",
        "- raw/bound shift uses the conservative DefenseScore convention when a defense prevents collapse across the whole grid.",
        "- Positive shifts mean the defense moved the critical harmful-agent ratio to the right.",
        "",
    ])
    path.write_text("\n".join(lines))


def _plot_curves(curve_rows: list[dict[str, Any]], defenses: list[str], path) -> None:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        groups[(row["scenario"], int(row["n_society"]))].append(row)
    if not groups:
        return
    n_panels = len(groups)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4), squeeze=False)
    for ax, ((scenario, n_society), rows) in zip(axes[0], sorted(groups.items())):
        for defense_name in defenses:
            selected = [row for row in rows if row["defense"] == defense_name]
            if not selected:
                continue
            selected.sort(key=lambda row: float(row["alpha"]))
            x = [float(row["alpha"]) for row in selected]
            y = [float(row["collapse_rate_mean"]) for row in selected]
            ax.plot(x, y, marker="o", label=defense_name)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{scenario.upper()} N={n_society}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("P(collapse)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    out = benchmark_exp_dir(OUT_NAME)
    defenses = _defense_names()
    alpha_grids = {scenario: _alphas_for(scenario) for scenario in SCENARIOS}
    baseline = calibrate_clean_baseline(n_society=min(max(N_GRID), 1000))

    all_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for n_society in N_GRID:
            for defense_name in defenses:
                print(f"\n=== Exp5 threshold sweep: {scenario} / N={n_society} / {defense_name} ===")
                rows = run_grid(
                    _build_specs(scenario, n_society, defense_name),
                    baseline=baseline,
                    progress_every=20,
                )
                for row in rows:
                    row["scenario_id"] = scenario
                    row["defense"] = defense_name
                    row["track"] = get_track(defense_name)
                all_rows.extend(rows)

    curve_rows = build_alpha_curve_rows(all_rows, alpha_grids, SCENARIOS, N_GRID, defenses)
    summary_rows = summarize_threshold_shift(all_rows, defenses, SCENARIOS, N_GRID, alpha_grids)

    write_csv(all_rows, out / "data.csv")
    _write_dict_csv(curve_rows, out / "alpha_curves.csv")
    _write_dict_csv(summary_rows, out / "threshold_shift_summary.csv")
    _write_report(summary_rows, alpha_grids, out / "report.md")
    _plot_curves(curve_rows, defenses, out / "collapse_curves.png")
    write_json({
        "scenarios": SCENARIOS,
        "defenses": defenses,
        "alpha_grids": alpha_grids,
        "n_grid": N_GRID,
        "seeds": SEEDS,
        "summary": summary_rows,
        "curves": curve_rows,
    }, out / "summary.json")

    print(f"\nDone. Wrote {out}")
    for row in summary_rows:
        print(
            f"  {row['scenario']} N={row['n_society']} {row['defense']:<10} "
            f"alpha_c={row['alpha_c_def']} shift={row['threshold_shift']} "
            f"score={row['defense_score']:.2f}"
        )


if __name__ == "__main__":
    main()
