"""P09: depth-regime dense alpha_c(N) sweeps for closure auditing."""
from __future__ import annotations

import argparse

from wolfbench.scenarios.base import ScenarioConfig

from ..runtime.runner import (
    add_run_args,
    alpha_values,
    as_run_args,
    load_protocol,
    profile_values,
    run_factorial,
    write_artifacts,
)


CONDITIONS = {
    "fixed_depth_q0": 0.0,
    "baseline_depth_q05": 0.5,
    "per_capita_depth_q1": 1.0,
}


def mutate(scenario: ScenarioConfig, variant: str) -> tuple[ScenarioConfig, str | None]:
    scenario.retail["controller_mode"] = "mixed_roles"
    scenario.market_makers["liquidity_exponent"] = CONDITIONS[variant]
    return scenario, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_args(parser, "p09_depth_scaling")
    args = as_run_args(parser.parse_args())
    seeds, requested_n = profile_values(args.profile)
    protocol = load_protocol()
    refined = {int(n): values for n, values in protocol["s1_refined_alpha"].items()}
    n_values = [n for n in requested_n if n in refined]
    if not n_values:
        raise SystemExit(f"No refined S1 grids for requested N={requested_n}")

    rows = []
    frozen_grid = {}
    for n_society in n_values:
        grid = alpha_values("s1", refined[n_society])
        frozen_grid[n_society] = grid
        rows.extend(run_factorial(
            experiment_id="P09_DEPTH_SCALING",
            args=args,
            scenarios=["s1"],
            n_values=[n_society],
            alphas={"s1": grid},
            seeds=seeds,
            variants=list(CONDITIONS),
            mutate=mutate,
        ))
    out = write_artifacts(
        args=args,
        experiment_id="P09_DEPTH_SCALING",
        rows=rows,
        config={
            "reviewer_question": "Observed nu under fixed, baseline, and per-capita depth regimes.",
            "conditions": CONDITIONS,
            "n_specific_alpha_grid": frozen_grid,
            "seeds": seeds,
        },
        group_keys=["variant", "n_society", "alpha"],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()