"""Local S2/S3 alpha-grid refinement for Exp12.

This runner only evaluates the additional alpha points needed to make the S2
first-finfluencer boundary and S3 spoof-liquidity transition better resolved.
Merge its ``data.csv`` with a completed canonical Exp12 run via
``exp12_merge_refinements.py`` before making paper figures.
"""
from __future__ import annotations

import os

from experiments._common import (
    RunSpec,
    env_float_list,
    env_int_list,
    env_list,
    env_seed_list,
    run_grid,
    scaling_exp_dir,
    write_csv,
    write_json,
)


EXTRA_ALPHA_GRIDS = {
    "s2": "0.00005,0.000075,0.0001,0.000125,0.00015,0.000175,0.0002,0.0003,0.00035,0.0004,0.00045,0.00055,0.0006,0.00065,0.0007,0.00105,0.0011,0.00115,0.0012",
    "s3": "0.0125,0.0175,0.0225,0.025,0.0275,0.035,0.04,0.045,0.055,0.06,0.065,0.07,0.08,0.085,0.09,0.095,0.11",
}

SCENARIOS = env_list("WOLFBENCH_EXP12_REFINE_SCENARIOS", "s2,s3")
N_GRID = env_int_list("WOLFBENCH_EXP12_REFINE_N_GRID", "500,1000,2000,10000")
SEEDS = env_seed_list("WOLFBENCH_EXP12_REFINE_SEEDS", default_count=30)
OUT_NAME = os.getenv("WOLFBENCH_EXP12_REFINE_OUT", "exp12_s2_s3_smooth_refinement")
PROGRESS_EVERY = int(os.getenv("WOLFBENCH_EXP12_REFINE_PROGRESS_EVERY", "50"))


def _alphas_for(scenario: str) -> list[float]:
    key = f"WOLFBENCH_EXP12_REFINE_ALPHAS_{scenario.upper()}"
    return env_float_list(key, EXTRA_ALPHA_GRIDS[scenario])


def main() -> None:
    out = scaling_exp_dir(OUT_NAME)
    alpha_grids = {scenario: _alphas_for(scenario) for scenario in SCENARIOS}
    specs = [
        RunSpec(scenario, n_society, alpha, seed)
        for scenario in SCENARIOS
        for n_society in N_GRID
        for alpha in alpha_grids[scenario]
        for seed in SEEDS
    ]
    print(f"Running {len(specs)} S2/S3 Exp12 refinement episodes...")
    rows = run_grid(specs, progress_every=PROGRESS_EVERY)
    write_csv(rows, out / "data.csv")
    write_json({
        "scenarios": SCENARIOS,
        "n_grid": N_GRID,
        "alpha_grids": alpha_grids,
        "seeds": SEEDS,
        "episodes": len(specs),
        "merge_command": (
            "PYTHONPATH=. WOLFBENCH_EXP12_MERGE_BASE=exp12_canonical_scaling_refined "
            "WOLFBENCH_EXP12_MERGE_REFINEMENTS=exp12_s2_s3_smooth_refinement "
            "python -m experiments.scaling_theory.exp12_merge_refinements"
        ),
        "outputs": {"data": "data.csv"},
    }, out / "summary.json")
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()