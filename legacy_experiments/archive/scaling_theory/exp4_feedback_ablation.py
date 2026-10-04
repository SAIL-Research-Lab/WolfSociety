"""Experiment 4: near-threshold market-feedback ablation.

Hold harmful ratio near the S1 critical region; sweep
``social.feedback_strength`` to test whether the social-financial reflexive
loop changes collapse probability rather than only increasing already-saturated
loss severity.

Output: paperoutputs/scaling/exp4_feedback_ablation/
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import (
    RunSpec, env_float_list, env_seed_list, run_grid, scaling_exp_dir,
    write_csv, write_json,
)
from wolfbench.metrics import binomial_rate_summary


SCENARIO = "s1"
N_SOCIETY = int(__import__("os").getenv("WOLFBENCH_EXP4_N_SOCIETY", "1000"))
ALPHA = float(__import__("os").getenv("WOLFBENCH_EXP4_ALPHA", "0.015"))
FEEDBACKS = env_float_list("WOLFBENCH_EXP4_FEEDBACKS", "0.0,0.2,0.4,0.6,0.8,1.0,1.4,2.0")
SEEDS = env_seed_list("WOLFBENCH_EXP4_SEEDS", default_count=50)


def main():
    out = scaling_exp_dir("exp4_feedback_ablation")
    specs = []
    for f in FEEDBACKS:
        for s in SEEDS:
            specs.append(RunSpec(SCENARIO, N_SOCIETY, ALPHA, s,
                                 feedback_strength=f, label=f"f={f}"))
    print(f"Running {len(specs)} episodes for exp4...")
    rows = run_grid(specs)
    write_csv(rows, out / "data.csv")
    write_json({"scenario": SCENARIO, "alpha": ALPHA, "n_society": N_SOCIETY,
                "feedback_strengths": FEEDBACKS, "seeds": SEEDS},
               out / "config.json")

    metrics = ["collapse_rate", "max_collapse_score",
               "retail_loss_pct_30d", "price_dislocation_max"]
    titles = ["P(collapse)", "max CollapseScore",
              "RetailLoss@30d", "PriceDislocation"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.5))
    summary = {}
    for ax, metric, title in zip(axes, metrics, titles):
        means, stds = [], []
        for f in FEEDBACKS:
            rows_sel = [r for r in rows if r["feedback_strength"] == f]
            vals = np.array([r[metric] for r in rows_sel])
            means.append(float(vals.mean()))
            stds.append(float(vals.std()))
            summary.setdefault(str(f), {})[metric] = {
                "mean": float(vals.mean()), "std": float(vals.std()),
            }
            if metric == "collapse_rate":
                summary[str(f)]["collapse_rate_wilson"] = binomial_rate_summary(vals)
        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(FEEDBACKS, means, yerr=stds, fmt="-o", capsize=4, color="C2")
        ax.set_xlabel("social.feedback_strength")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"WolfBench S1: near-threshold feedback ablation (α={ALPHA}, N={N_SOCIETY})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "feedback_compare.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    write_json(summary, out / "summary.json")
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()
