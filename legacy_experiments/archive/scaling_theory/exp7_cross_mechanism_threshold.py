"""Experiment 7: cross-mechanism critical-regime audit.

Estimate finite-size critical harmful ratios across S1-S4 rather than only S1.
This experiment is intended for the paper evidence package that supports the
claim that WolfBench measures a family of closed-loop failure mechanisms, not a
single hand-tuned pump-and-dump setting.

Output: paperoutputs/scaling/exp7_cross_mechanism_threshold/
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import (
    RunSpec,
    aggregate,
    env_float_list,
    env_int_list,
    env_list,
    env_seed_list,
    run_grid,
    scaling_exp_dir,
    write_csv,
    write_json,
)
from experiments.scaling_theory._threshold import bootstrap_logistic_ci, fit_logistic_threshold
from wolfbench.metrics import binomial_rate_summary


DEFAULT_ALPHA_GRIDS = {
    "s1": "0.0,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03",
    "s2": "0.0,0.00025,0.0005,0.00075,0.001,0.0015,0.0025,0.005",
    "s3": "0.0,0.15,0.25,0.3,0.35,0.4,0.5,0.6",
    "s4": "0.0,0.01,0.015,0.02,0.03,0.05,0.075,0.1,0.15",
}


SCENARIOS = env_list("WOLFBENCH_EXP7_SCENARIOS", "s1,s2,s3,s4")
N_GRID = env_int_list("WOLFBENCH_EXP7_N_GRID", "500,1000,2000")
SEEDS = env_seed_list("WOLFBENCH_EXP7_SEEDS", default_count=50)
CI_BOOT = int(os.getenv("WOLFBENCH_EXP7_CI_BOOT", "2000"))


def _alphas_for(scenario: str) -> list[float]:
    key = f"WOLFBENCH_EXP7_ALPHAS_{scenario.upper()}"
    default = os.getenv("WOLFBENCH_EXP7_ALPHAS") or DEFAULT_ALPHA_GRIDS[scenario]
    return env_float_list(key, default)


def _estimate(rows: list[dict], scenario: str, n_society: int, alphas: list[float]) -> dict:
    probs = []
    for alpha in alphas:
        agg = aggregate(
            [
                r for r in rows
                if r["scenario"] == scenario
                and int(r["n_society"]) == int(n_society)
                and float(r["alpha"]) == float(alpha)
            ],
            ["alpha"],
            "collapse_rate",
        )
        probs.append(agg.get((alpha,), {"mean": 0.0})["mean"])
    fit = fit_logistic_threshold(alphas, probs)
    rows_n = [
        r for r in rows
        if r["scenario"] == scenario and int(r["n_society"]) == int(n_society)
    ]
    ci = bootstrap_logistic_ci(rows_n, alphas, n_boot=CI_BOOT, rng_seed=70_000 + n_society)
    return {
        "scenario": scenario,
        "n_society": n_society,
        "alpha_c_logistic": fit["alpha_c"],
        "alpha_c_ci_low": ci["ci_low"],
        "alpha_c_ci_high": ci["ci_high"],
        "logistic_slope": fit["slope"],
        "transition_width_10_90": fit["transition_width_10_90"],
        "fit_method": fit["method"],
        "bootstrap_successes": ci["n_success"],
    }


def main() -> None:
    out = scaling_exp_dir("exp7_cross_mechanism_threshold")
    alpha_grids = {scenario: _alphas_for(scenario) for scenario in SCENARIOS}
    specs = [
        RunSpec(scenario, n_society, alpha, seed)
        for scenario in SCENARIOS
        for n_society in N_GRID
        for alpha in alpha_grids[scenario]
        for seed in SEEDS
    ]
    print(f"Running {len(specs)} episodes for exp7...")
    rows = run_grid(specs)
    write_csv(rows, out / "data.csv")

    threshold_rows = []
    wilson_rows = []
    for scenario in SCENARIOS:
        alphas = alpha_grids[scenario]
        for n_society in N_GRID:
            threshold_rows.append(_estimate(rows, scenario, n_society, alphas))
            for alpha in alphas:
                vals = [
                    r["collapse_rate"] for r in rows
                    if r["scenario"] == scenario
                    and int(r["n_society"]) == int(n_society)
                    and float(r["alpha"]) == float(alpha)
                ]
                wilson_rows.append({
                    "scenario": scenario,
                    "n_society": n_society,
                    "alpha": alpha,
                    **binomial_rate_summary(vals),
                })
    write_csv(threshold_rows, out / "alpha_critical_by_mechanism.csv")
    write_csv(wilson_rows, out / "collapse_rate_wilson_ci.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario in SCENARIOS:
        xs = []
        ys = []
        yerr_low = []
        yerr_high = []
        for row in threshold_rows:
            if row["scenario"] != scenario or row["alpha_c_logistic"] is None:
                continue
            xs.append(row["n_society"])
            ys.append(row["alpha_c_logistic"])
            low = row["alpha_c_ci_low"] if row["alpha_c_ci_low"] is not None else row["alpha_c_logistic"]
            high = row["alpha_c_ci_high"] if row["alpha_c_ci_high"] is not None else row["alpha_c_logistic"]
            yerr_low.append(max(0.0, row["alpha_c_logistic"] - low))
            yerr_high.append(max(0.0, high - row["alpha_c_logistic"]))
        if xs:
            ax.errorbar(xs, ys, yerr=[yerr_low, yerr_high], marker="o", capsize=4, label=scenario.upper())
    ax.set_xscale("log")
    ax.set_xlabel("Society size N")
    ax.set_ylabel("Estimated critical harmful ratio alpha_c")
    ax.set_title("Cross-mechanism finite-size critical regimes")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "alpha_c_by_mechanism.png", dpi=150)
    plt.close(fig)

    write_json({
        "scenarios": SCENARIOS,
        "n_grid": N_GRID,
        "alpha_grids": alpha_grids,
        "seeds": SEEDS,
        "ci_boot": CI_BOOT,
        "threshold_summary": threshold_rows,
        "wilson_ci_csv": "collapse_rate_wilson_ci.csv",
    }, out / "summary.json")
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()