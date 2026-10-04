"""Run all final experiment entrypoints in no-network smoke mode."""
from __future__ import annotations

import os
import sys

from . import run_e1_scaling, run_e2_mechanisms, run_e3_social_dynamics, run_e4_quota_robustness, run_e5_wolfguard_benchmark, run_smoke


def main() -> None:
    os.environ.setdefault("WOLFBENCH_FINAL_MOCK", "1")
    os.environ.setdefault("WOLFBENCH_FINAL_PRESET", "smoke")
    os.environ.setdefault("WOLFBENCH_FINAL_SEEDS", "1")
    os.environ.setdefault("WOLFBENCH_FINAL_N_GRID", "40")
    os.environ.setdefault("WOLFBENCH_FINAL_ALPHAS", "0.1")
    sys.argv = [sys.argv[0]]
    run_smoke.main()
    run_e1_scaling.main()
    run_e2_mechanisms.main()
    run_e3_social_dynamics.main()
    run_e4_quota_robustness.main()
    run_e5_wolfguard_benchmark.main()


if __name__ == "__main__":
    main()
