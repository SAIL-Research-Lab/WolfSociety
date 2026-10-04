"""Experiment 1: P_collapse(alpha) at fixed N -- expect S-shaped curves with
shifting critical threshold alpha_c(N). Demonstrates finite-size critical-regime
behavior.

Output: paperoutputs/scaling/exp1_alpha_scaling/
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import (
    RunSpec, env_float_list, env_int_list,
    env_seed_list, run_grid, scaling_exp_dir, write_csv, write_json,
)
from experiments.scaling_theory._threshold import bootstrap_logistic_ci, fit_logistic_threshold, linear_alpha_c
from wolfbench.metrics import binomial_rate_summary


SCENARIO = "s1"
ALPHAS = env_float_list(
    "WOLFBENCH_EXP1_ALPHAS",
    "0.0,0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.0225,0.025,0.03,0.04,0.05,0.10,0.20",
)
N_GRID = env_int_list("WOLFBENCH_EXP1_N_GRID", "200,1000,5000")
SEEDS = env_seed_list("WOLFBENCH_EXP1_SEEDS", default_count=50)
CI_BOOT = int(__import__("os").getenv("WOLFBENCH_EXP1_CI_BOOT", "2000"))


def _metric_summary(rows: list[dict], metrics: list[str]) -> list[dict]:
    summary_rows = []
    for N in N_GRID:
        for alpha in ALPHAS:
            selected = [
                row for row in rows
                if int(row["n_society"]) == int(N) and float(row["alpha"]) == float(alpha)
            ]
            out = {"n_society": N, "alpha": alpha, "n": len(selected)}
            for metric in metrics:
                vals = np.array([float(row[metric]) for row in selected], dtype=float)
                out[f"{metric}_mean"] = float(vals.mean()) if vals.size else 0.0
                out[f"{metric}_std"] = float(vals.std()) if vals.size else 0.0
            summary_rows.append(out)
    return summary_rows


def main():
    out = scaling_exp_dir("exp1_alpha_scaling")
    specs = [RunSpec(SCENARIO, N, a, s) for N in N_GRID for a in ALPHAS for s in SEEDS]
    print(f"Running {len(specs)} episodes for exp1...")
    rows = run_grid(specs)
    write_csv(rows, out / "data.csv")
    write_json({
        "scenario": SCENARIO,
        "alphas": ALPHAS,
        "n_grid": N_GRID,
        "seeds": SEEDS,
        "alpha_c_estimator": "logistic_fit_with_bootstrap_ci",
        "ci_boot": CI_BOOT,
    },
               out / "config.json")

    collapse_ci_rows = []
    threshold_rows = []
    p_curves = {}
    fitted_curves = {}
    for N in N_GRID:
        probs = []
        for a in ALPHAS:
            vals = [
                r["collapse_rate"] for r in rows
                if r["n_society"] == N and float(r["alpha"]) == float(a)
            ]
            ci = binomial_rate_summary(vals)
            probs.append(ci["mean"])
            collapse_ci_rows.append({
                "n_society": N,
                "alpha": a,
                **ci,
            })
        fit = fit_logistic_threshold(ALPHAS, probs)
        rows_n = [r for r in rows if int(r["n_society"]) == int(N)]
        boot = bootstrap_logistic_ci(rows_n, ALPHAS, n_boot=CI_BOOT, rng_seed=11_000 + N)
        threshold_rows.append({
            "n_society": N,
            "alpha_c_logistic": fit["alpha_c"],
            "alpha_c_ci_low": boot["ci_low"],
            "alpha_c_ci_high": boot["ci_high"],
            "alpha_c_linear": linear_alpha_c(ALPHAS, probs),
            "logistic_slope": fit["slope"],
            "transition_width_10_90": fit["transition_width_10_90"],
            "fit_method": fit["method"],
            "bootstrap_successes": boot["n_success"],
        })
        p_curves[N] = probs
        fitted_curves[N] = fit["fitted_probs"]
    write_csv(collapse_ci_rows, out / "collapse_rate_wilson_ci.csv")
    write_csv(threshold_rows, out / "alpha_critical_summary.csv")

    harm_metrics = [
        "max_collapse_score",
        "retail_loss_pct_30d",
        "social_cascade_peak",
        "price_dislocation_max",
        "liquidity_stress_max",
    ]
    metric_summary_rows = _metric_summary(rows, harm_metrics)
    write_csv(metric_summary_rows, out / "metrics_summary.csv")

    # ---------- Figure 1: P_collapse vs alpha, one curve per N ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_GRID)))
    summary = {}
    for N, c in zip(N_GRID, colours):
        curve_rows = [row for row in collapse_ci_rows if int(row["n_society"]) == int(N)]
        means = np.array([float(row["mean"]) for row in curve_rows], dtype=float)
        ci_low = np.array([float(row["ci_low"]) for row in curve_rows], dtype=float)
        ci_high = np.array([float(row["ci_high"]) for row in curve_rows], dtype=float)
        fitted = np.array(fitted_curves[N], dtype=float)
        threshold = next(row for row in threshold_rows if int(row["n_society"]) == int(N))
        alpha_c = threshold["alpha_c_logistic"]
        means = np.array(means)
        summary[str(N)] = {
            "alpha_c_logistic": alpha_c,
            "alpha_c_linear": threshold["alpha_c_linear"],
            "transition_width_10_90": threshold["transition_width_10_90"],
            "p_collapse": dict(zip(map(str, ALPHAS), means.tolist())),
            "p_collapse_logistic_fit": dict(zip(map(str, ALPHAS), fitted.tolist())),
        }
        ax.plot(ALPHAS, means, "o", color=c, label=f"N={N}  alpha_c={alpha_c:.4g}" if alpha_c is not None else f"N={N}")
        ax.plot(ALPHAS, fitted, "-", color=c, alpha=0.85, linewidth=1.7)
        ax.fill_between(ALPHAS,
                        ci_low,
                        ci_high,
                        color=c, alpha=0.15)
        if alpha_c is not None:
            ax.axvline(alpha_c, color=c, linestyle=":", alpha=0.65)
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="P=0.5")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Harmful-agent ratio α")
    ax.set_ylabel("P(collapse)  over 30-day episode")
    ax.set_title("WolfBench S1: nonlinear collapse vs harmful ratio\nat different society sizes")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "p_collapse_vs_alpha.png", dpi=140)
    plt.close(fig)

    # ---------- Figure 2: companion harm metrics ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for N, c in zip(N_GRID, colours):
        for ax, metric, title in zip(
            axes,
            ["max_collapse_score", "retail_loss_pct_30d", "social_cascade_peak"],
            ["max CollapseScore", "RetailLoss@30d (frac)", "SocialCascadePeak"],
        ):
            ms = [
                row[f"{metric}_mean"]
                for row in metric_summary_rows
                if int(row["n_society"]) == int(N)
            ]
            ax.plot(ALPHAS, ms, "-o", color=c, label=f"N={N}")
            ax.set_xscale("symlog", linthresh=1e-3)
            ax.set_xlabel("α")
            ax.set_title(title)
            ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("WolfBench S1: per-α harm metrics across N", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "metrics_vs_alpha.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    write_json(summary, out / "summary.json")
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()
