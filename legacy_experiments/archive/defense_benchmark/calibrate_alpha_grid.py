"""Calibrate per-scenario alpha grids around primary-failure critical regions.

This experiment runs NoGuard only. It estimates
P(primary failure | scenario, N, alpha) and recommends a compact alpha grid
that includes low, near-critical, and high points for each scenario.

Environment overrides:
* WOLFBENCH_CALIB_SCENARIOS=s1,s2,s3,s4
* WOLFBENCH_CALIB_N_GRID=500,1000,2000
* WOLFBENCH_CALIB_ALPHAS=0,0.0025,...
* WOLFBENCH_CALIB_SEEDS=1,2,3,4,5
* WOLFBENCH_CALIB_PROBABILITY_KEY=primary_failure_rate

Outputs: paperoutputs/benchmark/alpha_calibration/
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import RunSpec, aggregate, benchmark_exp_dir, run_grid, write_csv, write_json
from wolfbench.metrics import bootstrap_ci


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


def _env_float_list(name: str, default: str) -> list[float]:
    return [float(x) for x in _env_list(name, default)]


def _env_int_list(name: str, default: str) -> list[int]:
    return [int(x) for x in _env_list(name, default)]


SCENARIOS = _env_list("WOLFBENCH_CALIB_SCENARIOS", "s1,s2,s3,s4")
N_GRID = _env_int_list("WOLFBENCH_CALIB_N_GRID", "500,1000,2000")
ALPHAS = _env_float_list(
    "WOLFBENCH_CALIB_ALPHAS",
    "0.0,0.00025,0.0005,0.00075,0.001,0.0015,0.0025,0.005,0.0075,0.01,0.015,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5",
)
SEEDS = _env_int_list("WOLFBENCH_CALIB_SEEDS", "1,2,3,4,5")
THRESHOLD = float(os.getenv("WOLFBENCH_CALIB_THRESHOLD", "0.5"))
CI_BOOT = int(os.getenv("WOLFBENCH_CALIB_CI_BOOT", "2000"))
PROBABILITY_KEY = os.getenv("WOLFBENCH_CALIB_PROBABILITY_KEY", "primary_failure_rate")


def _estimate_alpha_c(rows: list[dict], scenario: str, n_society: int) -> tuple[float | None, list[float]]:
    means = []
    for alpha in ALPHAS:
        stats = aggregate(
            [r for r in rows if r["scenario"] == scenario
            and int(r["n_society"]) == n_society and float(r["alpha"]) == alpha],
            ["alpha"], PROBABILITY_KEY,
        )
        means.append(stats.get((alpha,), {"mean": 0.0})["mean"])
    for idx, prob in enumerate(means):
        if prob >= THRESHOLD:
            if idx == 0:
                return float(ALPHAS[0]), means
            p0, p1 = means[idx - 1], prob
            a0, a1 = ALPHAS[idx - 1], ALPHAS[idx]
            if p1 == p0:
                return float(a1), means
            alpha_c = a0 + (THRESHOLD - p0) * (a1 - a0) / (p1 - p0)
            return float(alpha_c), means
    return None, means


def _nearest_grid(alpha_c_values: list[float | None]) -> list[float]:
    valid = [a for a in alpha_c_values if a is not None]
    if not valid:
        return sorted({ALPHAS[0], *ALPHAS[::4], ALPHAS[-1]})
    anchors = {min(ALPHAS)}
    for ac in valid:
        anchors.update({ac * 0.5, ac * 0.75, ac, ac * 1.25, ac * 1.5, ac * 2.0})
    recommended = []
    for point in sorted(anchors):
        closest = min(ALPHAS, key=lambda a: abs(a - point))
        if min(ALPHAS) <= closest <= max(ALPHAS):
            recommended.append(float(closest))
    return sorted(set(recommended))


def _write_recommended_env(path: Path, recommended: dict[str, list[float]]) -> None:
    with open(path, "w") as handle:
        for scenario, alphas in recommended.items():
            value = ",".join(f"{a:g}" for a in alphas)
            handle.write(f"export WOLFBENCH_EXP6_ALPHAS_{scenario.upper()}={value}\n")


def main() -> None:
    out = benchmark_exp_dir("alpha_calibration")
    specs = [RunSpec(scenario, n, alpha, seed)
             for scenario in SCENARIOS for n in N_GRID for alpha in ALPHAS for seed in SEEDS]
    print(f"Running {len(specs)} NoGuard episodes for alpha calibration...")
    rows = run_grid(specs, progress_every=50)
    write_csv(rows, out / "data.csv")

    summary_rows = []
    recommended: dict[str, list[float]] = {}
    for scenario in SCENARIOS:
        alpha_cs = []
        for n_society in N_GRID:
            alpha_c, probs = _estimate_alpha_c(rows, scenario, n_society)
            alpha_cs.append(alpha_c)
            for alpha, prob in zip(ALPHAS, probs):
                vals = [r[PROBABILITY_KEY] for r in rows
                        if r["scenario"] == scenario
                        and int(r["n_society"]) == n_society
                        and float(r["alpha"]) == alpha]
                ci_low, ci_high = bootstrap_ci(vals, n_boot=CI_BOOT)
                summary_rows.append({
                    "scenario": scenario,
                    "n_society": n_society,
                    "alpha": alpha,
                    "probability_key": PROBABILITY_KEY,
                    "p_primary_failure": prob,
                    "p_primary_failure_ci_low": ci_low,
                    "p_primary_failure_ci_high": ci_high,
                    "alpha_c": alpha_c,
                })
        recommended[scenario] = _nearest_grid(alpha_cs)

    write_csv(summary_rows, out / "summary.csv")
    write_csv(
        [{"scenario": s, "recommended_alphas": ",".join(f"{a:g}" for a in alphas)}
         for s, alphas in recommended.items()],
        out / "recommended_alpha_grid.csv",
    )
    _write_recommended_env(out / "recommended_env.sh", recommended)
    write_json({
        "scenarios": SCENARIOS,
        "n_grid": N_GRID,
        "alphas": ALPHAS,
        "seeds": SEEDS,
        "threshold": THRESHOLD,
        "probability_key": PROBABILITY_KEY,
        "recommended_alpha_grid": recommended,
    }, out / "summary.json")

    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(8, 3.2 * len(SCENARIOS)), sharex=True)
    if len(SCENARIOS) == 1:
        axes = [axes]
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_GRID)))
    for ax, scenario in zip(axes, SCENARIOS):
        for n_society, colour in zip(N_GRID, colours):
            probs = [r["p_primary_failure"] for r in summary_rows
                     if r["scenario"] == scenario and r["n_society"] == n_society]
            ax.plot(ALPHAS, probs, "-o", color=colour, label=f"N={n_society}")
        ax.axhline(THRESHOLD, color="grey", linestyle="--", linewidth=1)
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"{scenario} P(primary failure)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    axes[-1].set_xlabel("harmful-agent ratio alpha")
    fig.tight_layout()
    fig.savefig(out / "p_primary_failure_calibration.png", dpi=150)
    plt.close(fig)

    print(f"Done. Wrote {out}")
    for scenario, alphas in recommended.items():
        print(f"  {scenario}: {','.join(f'{a:g}' for a in alphas)}")


if __name__ == "__main__":
    main()
