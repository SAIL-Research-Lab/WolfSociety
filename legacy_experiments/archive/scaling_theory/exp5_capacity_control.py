"""Experiment 5: capacity-controlled scaling check.

Compare the default per-agent-capacity protocol with a fixed-total-capacity
protocol. In the fixed-total-capacity condition, per-agent wealth is scaled by
``BASE_N / N`` and scenario asset liquidity is scaled by ``sqrt(BASE_N / N)``;
after the environment's built-in ``sqrt(N / 1000)`` liquidity scaling, market
capacity is held at the BASE_N level. This tests whether observed or missing
``alpha_c(N)`` scaling is an artifact of total capital/liquidity growth.

Output: paperoutputs/scaling/exp5_capacity_control/
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
from experiments.scaling_theory._threshold import bootstrap_logistic_ci, fit_logistic_threshold, linear_alpha_c


SCENARIO = "s1"
BASE_N = int(os.getenv("WOLFBENCH_EXP5_CAPACITY_BASE_N", "1000"))
N_GRID = env_int_list("WOLFBENCH_EXP5_CAPACITY_N_GRID", "200,1000,5000")
ALPHAS = env_float_list("WOLFBENCH_EXP5_CAPACITY_ALPHAS", "0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.0225,0.025,0.03")
SEEDS = env_seed_list("WOLFBENCH_EXP5_CAPACITY_SEEDS", default_count=20)
CAPACITY_MODES = env_list("WOLFBENCH_EXP5_CAPACITY_MODES", "per_agent_capacity,fixed_total_capacity")
CI_BOOT = int(os.getenv("WOLFBENCH_EXP5_CAPACITY_CI_BOOT", "500"))


def _spec(scenario: str, mode: str, n_society: int, alpha: float, seed: int) -> RunSpec:
    if mode == "fixed_total_capacity":
        return RunSpec(
            scenario,
            n_society,
            alpha,
            seed,
            retail_wealth_scale=BASE_N / n_society,
            asset_liquidity_scale=(BASE_N / n_society) ** 0.5,
            label=mode,
        )
    return RunSpec(scenario, n_society, alpha, seed, label=mode)


def _estimate(rows: list[dict], mode: str, n_society: int) -> tuple[dict, np.ndarray]:
    selected = [r for r in rows if r["label"] == mode and int(r["n_society"]) == n_society]
    probs = []
    for alpha in ALPHAS:
        stats = aggregate([r for r in selected if float(r["alpha"]) == alpha], ["alpha"], "collapse_rate")
        probs.append(stats.get((alpha,), {"mean": 0.0})["mean"])
    fit = fit_logistic_threshold(ALPHAS, probs)
    fit["alpha_c_linear"] = linear_alpha_c(ALPHAS, probs)
    fit.update(bootstrap_logistic_ci(selected, ALPHAS, n_boot=CI_BOOT, rng_seed=20_000 + n_society))
    return fit, np.array(probs)


def main() -> None:
    out = scaling_exp_dir("exp5_capacity_control")
    specs = [
        _spec(SCENARIO, mode, n_society, alpha, seed)
        for mode in CAPACITY_MODES
        for n_society in N_GRID
        for alpha in ALPHAS
        for seed in SEEDS
    ]
    print(f"Running {len(specs)} episodes for exp5...")
    rows = run_grid(specs, progress_every=50)
    write_csv(rows, out / "data.csv")
    write_json({
        "scenario": SCENARIO,
        "base_n": BASE_N,
        "n_grid": N_GRID,
        "alphas": ALPHAS,
        "seeds": SEEDS,
        "capacity_modes": CAPACITY_MODES,
        "ci_boot": CI_BOOT,
        "fixed_total_capacity": {
            "retail_wealth_scale": "BASE_N / N",
            "asset_liquidity_scale": "sqrt(BASE_N / N)",
        },
    }, out / "config.json")

    summary_rows = []
    p_curves: dict[str, dict[str, list[float]]] = {}
    for mode in CAPACITY_MODES:
        p_curves[mode] = {}
        for n_society in N_GRID:
            fit, probs = _estimate(rows, mode, n_society)
            p_curves[mode][str(n_society)] = probs.tolist()
            summary_rows.append({
                "capacity_mode": mode,
                "n_society": n_society,
                "alpha_c_logistic": fit["alpha_c"],
                "alpha_c_ci_low": fit["ci_low"],
                "alpha_c_ci_high": fit["ci_high"],
                "alpha_c_linear": fit["alpha_c_linear"],
                "logistic_slope": fit["slope"],
                "transition_width_10_90": fit["transition_width_10_90"],
                "fit_method": fit["method"],
                "bootstrap_successes": fit["n_success"],
            })
    write_csv(summary_rows, out / "alpha_critical_capacity_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, color in zip(CAPACITY_MODES, ["C0", "C3"]):
        rows_mode = [
            r for r in summary_rows
            if r["capacity_mode"] == mode and r["alpha_c_logistic"] is not None
        ]
        if not rows_mode:
            continue
        ns = np.array([r["n_society"] for r in rows_mode], dtype=float)
        acs = np.array([r["alpha_c_logistic"] for r in rows_mode], dtype=float)
        low = np.array([r["alpha_c_ci_low"] if r["alpha_c_ci_low"] is not None else r["alpha_c_logistic"] for r in rows_mode])
        high = np.array([r["alpha_c_ci_high"] if r["alpha_c_ci_high"] is not None else r["alpha_c_logistic"] for r in rows_mode])
        ax.errorbar(ns, acs, yerr=[acs - low, high - acs], fmt="-o", capsize=4,
                    color=color, label=mode)
    ax.set_xscale("log")
    ax.set_xlabel("Society size N (log)")
    ax.set_ylabel("Logistic α_c")
    ax.set_title("Capacity-controlled α_c(N) check")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "alpha_critical_capacity_compare.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(N_GRID), figsize=(4.2 * len(N_GRID), 4.2), sharey=True)
    for ax, n_society in zip(axes, N_GRID):
        for mode, color in zip(CAPACITY_MODES, ["C0", "C3"]):
            ax.plot(ALPHAS, p_curves[mode][str(n_society)], "-o", color=color, label=mode)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
        ax.set_title(f"N={n_society}")
        ax.set_xlabel("α")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("P(collapse)")
    axes[0].legend()
    fig.suptitle("Capacity control: P(collapse) curves", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "p_collapse_capacity_compare.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    write_json({
        "threshold_summary": summary_rows,
        "p_collapse": {
            mode: {n: dict(zip(map(str, ALPHAS), probs)) for n, probs in curves.items()}
            for mode, curves in p_curves.items()
        },
    }, out / "summary.json")
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()