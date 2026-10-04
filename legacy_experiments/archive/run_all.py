"""Run all WolfBench experiment tracks sequentially.

Usage::
    python -m experiments.run_all
"""
from __future__ import annotations

import importlib
import time

EXPERIMENTS = [
    "experiments.scaling_theory.exp1_alpha_scaling",
    "experiments.scaling_theory.exp2_society_size_scaling",
    "experiments.scaling_theory.exp3_centrality_placement",
    "experiments.scaling_theory.exp4_feedback_ablation",
    "experiments.scaling_theory.exp5_capacity_control",
    "experiments.scaling_theory.exp7_cross_mechanism_threshold",
    "experiments.scaling_theory.exp8_sensitivity_audit",
    "experiments.defense_benchmark.exp5_wolfguard_defense",
    "experiments.defense_benchmark.calibrate_alpha_grid",
    "experiments.defense_benchmark.exp6_defense_leaderboard",
]


def main():
    t_total = time.time()
    for name in EXPERIMENTS:
        print(f"\n=== {name} ===")
        t0 = time.time()
        mod = importlib.import_module(name)
        mod.main()
        print(f"  ({time.time() - t0:.1f}s)")
    print(f"\nAll experiments done in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
