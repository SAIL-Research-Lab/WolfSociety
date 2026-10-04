"""Run the defense benchmark calibration and leaderboard experiments.

Usage::
    python -m experiments.defense_benchmark.run_all
"""
from __future__ import annotations

import importlib
import time


EXPERIMENTS = [
    "experiments.defense_benchmark.exp5_wolfguard_defense",
    "experiments.defense_benchmark.calibrate_alpha_grid",
    "experiments.defense_benchmark.exp6_defense_leaderboard",
]


def main() -> None:
    t_total = time.time()
    for name in EXPERIMENTS:
        print(f"\n=== {name} ===")
        t0 = time.time()
        mod = importlib.import_module(name)
        mod.main()
        print(f"  ({time.time() - t0:.1f}s)")
    print(f"\nDefense benchmark experiments done in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()