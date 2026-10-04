"""Experiment 6: N=200 single-LLM strategic leader scaling check.

This is the minimal LLM-based scaling-theory experiment. It keeps society size
fixed at N=200, upgrades at most one harmful strategic leader to an LLM, and
sweeps alpha to test whether the nonlinear collapse transition persists when a
single LLM strategist is embedded in an otherwise rule-based harmful population.

Output: paperoutputs/scaling/exp6_llm_n200_scaling/
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import RunSpec, aggregate, run_grid, scaling_exp_dir, write_csv, write_json
from experiments.scaling_theory._threshold import bootstrap_logistic_ci, fit_logistic_threshold, linear_alpha_c
from wolfbench.agents.llm import make_chat_backend


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


def _env_float_list(name: str, default: str) -> list[float]:
    return [float(x) for x in _env_list(name, default)]


def _env_int_list(name: str, default: str) -> list[int]:
    return [int(x) for x in _env_list(name, default)]


SCENARIO = os.getenv("WOLFBENCH_LLM_N200_SCENARIO", "s1")
N_SOCIETY = int(os.getenv("WOLFBENCH_LLM_N200_N", "200"))
N_LLM_LEADERS = int(os.getenv("WOLFBENCH_LLM_N200_LEADERS", "1"))
ALPHAS = _env_float_list(
    "WOLFBENCH_LLM_N200_ALPHAS",
    "0.0,0.001,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.0225,0.025,0.03,0.05,0.10,0.20",
)
SEEDS = _env_int_list("WOLFBENCH_LLM_N200_SEEDS", ",".join(str(i) for i in range(1, 21)))
LLM_PROVIDER = os.getenv("WOLFBENCH_LLM_N200_PROVIDER", "vllm")
LLM_MODEL = os.getenv("WOLFBENCH_LLM_N200_MODEL", os.getenv("WOLFBENCH_VLLM_MODEL", "qwen3-8b"))
LLM_BASE_URL = os.getenv("WOLFBENCH_LLM_N200_BASE_URL", os.getenv("WOLFBENCH_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
LLM_API_KEY = os.getenv("WOLFBENCH_LLM_N200_API_KEY", os.getenv("WOLFBENCH_VLLM_API_KEY", "EMPTY"))
CI_BOOT = int(os.getenv("WOLFBENCH_LLM_N200_CI_BOOT", "500"))


def _build_backend():
    return make_chat_backend(
        provider=LLM_PROVIDER,
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        strict=True,
    )


def _estimate_threshold(rows: list[dict]) -> tuple[dict, np.ndarray]:
    probs = []
    for alpha in ALPHAS:
        stats = aggregate([r for r in rows if float(r["alpha"]) == alpha], ["alpha"], "collapse_rate")
        probs.append(stats.get((alpha,), {"mean": 0.0})["mean"])
    fit = fit_logistic_threshold(ALPHAS, probs)
    fit["alpha_c_linear"] = linear_alpha_c(ALPHAS, probs)
    fit.update(bootstrap_logistic_ci(rows, ALPHAS, n_boot=CI_BOOT, rng_seed=30_200))
    return fit, np.array(probs)


def main() -> None:
    out = scaling_exp_dir("exp6_llm_n200_scaling")
    backend = _build_backend()
    specs = [
        RunSpec(
            SCENARIO,
            N_SOCIETY,
            alpha,
            seed,
            llm_backend=backend,
            n_llm_leaders=N_LLM_LEADERS,
            llm_provider=LLM_PROVIDER,
            llm_model=LLM_MODEL,
            label="single_llm_leader",
        )
        for alpha in ALPHAS
        for seed in SEEDS
    ]
    print(
        f"Running {len(specs)} LLM episodes for exp6 "
        f"(scenario={SCENARIO}, N={N_SOCIETY}, k_llm={N_LLM_LEADERS}, model={LLM_MODEL})..."
    )
    rows = run_grid(specs, progress_every=max(1, min(10, len(specs))))
    write_csv(rows, out / "data.csv")
    write_json({
        "scenario": SCENARIO,
        "n_society": N_SOCIETY,
        "n_llm_leaders": N_LLM_LEADERS,
        "alphas": ALPHAS,
        "seeds": SEEDS,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "ci_boot": CI_BOOT,
        "note": "Single LLM strategic leader; remaining harmful agents are rule-based bots/traders.",
    }, out / "config.json")

    fit, probs = _estimate_threshold(rows)
    threshold_row = {
        "scenario": SCENARIO,
        "n_society": N_SOCIETY,
        "n_llm_leaders": N_LLM_LEADERS,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "alpha_c_logistic": fit["alpha_c"],
        "alpha_c_ci_low": fit["ci_low"],
        "alpha_c_ci_high": fit["ci_high"],
        "alpha_c_linear": fit["alpha_c_linear"],
        "logistic_slope": fit["slope"],
        "transition_width_10_90": fit["transition_width_10_90"],
        "fit_method": fit["method"],
        "bootstrap_successes": fit["n_success"],
    }
    write_csv([threshold_row], out / "alpha_critical_summary.csv")

    metric_summary = {}
    for metric in [
        "collapse_rate",
        "max_collapse_score",
        "retail_loss_pct_30d",
        "social_cascade_peak",
        "price_dislocation_max",
    ]:
        metric_summary[metric] = {}
        for alpha in ALPHAS:
            stats = aggregate([r for r in rows if float(r["alpha"]) == alpha], ["alpha"], metric)
            stat = stats.get((alpha,), {"mean": 0.0, "std": 0.0, "n": 0})
            metric_summary[metric][str(alpha)] = {
                "mean": stat["mean"],
                "std": stat["std"],
                "n": stat["n"],
            }

    fig, ax = plt.subplots(figsize=(7, 5))
    stds = np.array([metric_summary["collapse_rate"][str(alpha)]["std"] for alpha in ALPHAS])
    ax.plot(ALPHAS, probs, "-o", color="C3", label="single LLM leader")
    ax.fill_between(ALPHAS, np.clip(probs - stds, 0, 1), np.clip(probs + stds, 0, 1), color="C3", alpha=0.18)
    if fit["alpha_c"] is not None:
        ax.axvline(fit["alpha_c"], color="C3", linestyle=":", alpha=0.7, label=f"logistic alpha_c={fit['alpha_c']:.4f}")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Harmful-agent ratio alpha")
    ax.set_ylabel("P(collapse)")
    ax.set_title(f"WolfBench S1: single LLM harmful strategist at N={N_SOCIETY}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "p_collapse_vs_alpha.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, title in zip(
        axes,
        ["max_collapse_score", "retail_loss_pct_30d", "social_cascade_peak"],
        ["max CollapseScore", "RetailLoss@30d", "SocialCascadePeak"],
    ):
        means = [metric_summary[metric][str(alpha)]["mean"] for alpha in ALPHAS]
        ax.plot(ALPHAS, means, "-o", color="C0")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel("alpha")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle(f"WolfBench S1: LLM leader companion metrics at N={N_SOCIETY}", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "metrics_vs_alpha.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    write_json({
        "threshold_summary": threshold_row,
        "p_collapse": dict(zip(map(str, ALPHAS), probs.tolist())),
        "p_collapse_logistic_fit": dict(zip(map(str, ALPHAS), fit["fitted_probs"])),
        "metrics": metric_summary,
        "llm_backend_calls": int(getattr(backend, "calls", 0)),
        "llm_backend_failures": int(getattr(backend, "failures", 0)),
        "llm_backend_last_error": getattr(backend, "last_error", ""),
    }, out / "summary.json")
    print(f"Done. Wrote {out}")
    print(f"LLM backend calls={getattr(backend, 'calls', 0)} failures={getattr(backend, 'failures', 0)}")


if __name__ == "__main__":
    main()