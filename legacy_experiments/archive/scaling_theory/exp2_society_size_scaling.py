"""Experiment 2: alpha_c(N) with a dense near-threshold finite-size grid.

This is the main controlled scaling-protocol experiment. It estimates the
critical harmful-agent ratio with a logistic fit and bootstrap confidence
intervals, rather than only using a grid-crossing interpolation.

Output: paperoutputs/scaling/exp2_society_size_scaling/
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import (
    RunSpec, aggregate, env_float_list, env_int_list, env_seed_list,
    run_episode, scaling_exp_dir, write_csv, write_json,
)
from experiments.scaling_theory._threshold import bootstrap_logistic_ci, fit_logistic_threshold, linear_alpha_c
from wolfbench.metrics import binomial_rate_summary


SCENARIO = "s1"
N_GRID = env_int_list(
    "WOLFBENCH_EXP2_N_GRID",
    "100,150,200,300,500,750,1000,1500,2000,3000,5000",
)
ALPHAS = env_float_list(
    "WOLFBENCH_EXP2_ALPHAS",
    "0.0,0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.0225,0.025,0.03,0.04",
)
SEEDS = env_seed_list("WOLFBENCH_EXP2_SEEDS", default_count=50)
CI_BOOT = int(__import__("os").getenv("WOLFBENCH_EXP2_CI_BOOT", "2000"))
CHECKPOINT_EVERY = int(__import__("os").getenv("WOLFBENCH_EXP2_CHECKPOINT_EVERY", "50"))


def _spec_key(scenario: str, n_society: int, alpha: float, seed: int) -> tuple[str, int, float, int]:
    return (str(scenario), int(n_society), round(float(alpha), 12), int(seed))


def _row_key(row: dict) -> tuple[str, int, float, int]:
    return _spec_key(row["scenario"], int(row["n_society"]), float(row["alpha"]), int(row["seed"]))


def _load_existing_rows(path):
    if not path.exists():
        return []
    import csv

    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            # Keep only fields used downstream strongly typed; unused diagnostics can
            # remain strings without affecting the Exp2 analysis.
            row["scenario"] = str(row["scenario"])
            row["n_society"] = int(float(row["n_society"]))
            row["alpha"] = float(row["alpha"])
            row["seed"] = int(float(row["seed"]))
            row["collapse_rate"] = float(row["collapse_rate"])
            rows.append(row)
    return rows


def run_or_resume(specs: list[RunSpec], out) -> list[dict]:
    data_path = out / "data.csv"
    rows = _load_existing_rows(data_path)
    seen = {_row_key(row) for row in rows}
    missing = [spec for spec in specs if _spec_key(spec.scenario, spec.n_society, spec.alpha, spec.seed) not in seen]
    print(f"Exp2 rows: existing={len(rows)} missing={len(missing)} total={len(specs)}")
    if not missing:
        return rows

    import time

    t0 = time.time()
    for i, spec in enumerate(missing, 1):
        rows.append(run_episode(spec))
        if i % CHECKPOINT_EVERY == 0 or i == len(missing):
            write_csv(rows, data_path)
            dt = time.time() - t0
            print(
                f"[{i:>4}/{len(missing)} missing] {len(rows):>5}/{len(specs)} rows "
                f"{dt:7.1f}s elapsed; checkpointed {data_path}",
                flush=True,
            )
    return rows


def estimate_alpha_c(rows, N, alphas, threshold: float = 0.5):
    """Return logistic and linear alpha_c estimates plus P(collapse) curve."""
    ps = []
    for a in alphas:
        agg = aggregate([r for r in rows if r["n_society"] == N and r["alpha"] == a],
                        ["alpha"], "collapse_rate")
        ps.append(agg.get((a,), {"mean": 0.0})["mean"])
    ps = np.array(ps)
    fit = fit_logistic_threshold(alphas, ps, threshold=threshold)
    fit["alpha_c_linear"] = linear_alpha_c(alphas, ps, threshold=threshold)
    return fit, ps


def main():
    out = scaling_exp_dir("exp2_society_size_scaling")
    specs = [RunSpec(SCENARIO, N, a, s) for N in N_GRID for a in ALPHAS for s in SEEDS]
    print(f"Running {len(specs)} episodes for exp2...")
    rows = run_or_resume(specs, out)
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
    for N in N_GRID:
        for a in ALPHAS:
            vals = [
                r["collapse_rate"] for r in rows
                if r["n_society"] == N and float(r["alpha"]) == float(a)
            ]
            collapse_ci_rows.append({
                "n_society": N,
                "alpha": a,
                **binomial_rate_summary(vals),
            })
    write_csv(collapse_ci_rows, out / "collapse_rate_wilson_ci.csv")

    # ---------- estimate alpha_c per N ----------
    alpha_cs = []
    alpha_cs_linear = []
    p_curves = {}
    fitted_curves = {}
    threshold_rows = []
    for N in N_GRID:
        fit, ps = estimate_alpha_c(rows, N, ALPHAS)
        rows_n = [r for r in rows if r["n_society"] == N]
        ci = bootstrap_logistic_ci(rows_n, ALPHAS, n_boot=CI_BOOT, rng_seed=10_000 + N)
        ac = fit["alpha_c"]
        alpha_cs.append(ac)
        alpha_cs_linear.append(fit["alpha_c_linear"])
        p_curves[N] = ps
        fitted_curves[N] = fit["fitted_probs"]
        threshold_rows.append({
            "n_society": N,
            "alpha_c_logistic": ac,
            "alpha_c_ci_low": ci["ci_low"],
            "alpha_c_ci_high": ci["ci_high"],
            "alpha_c_linear": fit["alpha_c_linear"],
            "logistic_slope": fit["slope"],
            "transition_width_10_90": fit["transition_width_10_90"],
            "fit_method": fit["method"],
            "bootstrap_successes": ci["n_success"],
        })
    write_csv(threshold_rows, out / "alpha_critical_summary.csv")

    # ---------- Figure 1: log-log alpha_c vs N + power-law fit ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    Ns = np.array([N for N, ac in zip(N_GRID, alpha_cs) if ac is not None])
    acs = np.array([ac for ac in alpha_cs if ac is not None])
    yerr_low, yerr_high = [], []
    for row in threshold_rows:
        if row["alpha_c_logistic"] is None:
            continue
        low = row["alpha_c_ci_low"] if row["alpha_c_ci_low"] is not None else row["alpha_c_logistic"]
        high = row["alpha_c_ci_high"] if row["alpha_c_ci_high"] is not None else row["alpha_c_logistic"]
        yerr_low.append(max(0.0, row["alpha_c_logistic"] - low))
        yerr_high.append(max(0.0, high - row["alpha_c_logistic"]))
    ax.errorbar(Ns, acs, yerr=[yerr_low, yerr_high], fmt="o", markersize=8,
                color="C3", capsize=4, label="logistic α_c(N) with bootstrap CI")
    fit_info = {}
    if len(Ns) >= 2:
        # log alpha_c = log A + beta * log N
        beta, logA = np.polyfit(np.log(Ns), np.log(acs), 1)
        A = float(np.exp(logA))
        fit_x = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
        fit_y = A * fit_x ** beta
        ax.loglog(fit_x, fit_y, "--", color="C0",
                  label=f"power fit  α_c ≈ {A:.3g}·N^{{{beta:.2f}}}")
        fit_info = {"A": A, "beta": float(beta)}
    ax.set_xlabel("Society size N (log)")
    ax.set_ylabel("Critical harmful ratio α_c (log)")
    ax.set_title("WolfBench S1: scaling of critical harmful ratio with society size")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "alpha_critical_vs_N.png", dpi=140)
    plt.close(fig)

    # ---------- Figure 2: heatmap of P_collapse(N, alpha) ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    grid = np.zeros((len(N_GRID), len(ALPHAS)))
    for i, N in enumerate(N_GRID):
        grid[i, :] = p_curves[N]
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   extent=[0, len(ALPHAS), 0, len(N_GRID)],
                   cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(ALPHAS)) + 0.5)
    ax.set_xticklabels([f"{a:g}" for a in ALPHAS], rotation=45)
    ax.set_yticks(np.arange(len(N_GRID)) + 0.5)
    ax.set_yticklabels(N_GRID)
    ax.set_xlabel("α")
    ax.set_ylabel("N")
    ax.set_title("P(collapse) heatmap")
    fig.colorbar(im, ax=ax, label="P(collapse)")
    fig.tight_layout()
    fig.savefig(out / "p_collapse_heatmap.png", dpi=140)
    plt.close(fig)

    # ---------- Figure 3: transition width vs N ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    widths = [r["transition_width_10_90"] for r in threshold_rows]
    ax.plot(N_GRID, widths, "-o", color="C4")
    ax.set_xscale("log")
    ax.set_xlabel("Society size N (log)")
    ax.set_ylabel("Logistic transition width α(0.9)-α(0.1)")
    ax.set_title("WolfBench S1: finite-size transition width")
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "transition_width_vs_N.png", dpi=140)
    plt.close(fig)

    write_json({
        "alpha_critical_logistic": {str(N): ac for N, ac in zip(N_GRID, alpha_cs)},
        "alpha_critical_linear": {str(N): ac for N, ac in zip(N_GRID, alpha_cs_linear)},
        "threshold_summary": threshold_rows,
        "p_collapse": {str(N): dict(zip(map(str, ALPHAS), p_curves[N].tolist())) for N in N_GRID},
        "p_collapse_wilson_ci_csv": "collapse_rate_wilson_ci.csv",
        "p_collapse_logistic_fit": {str(N): dict(zip(map(str, ALPHAS), fitted_curves[N])) for N in N_GRID},
        "fit": fit_info,
    }, out / "summary.json")
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()
