"""P02b: boundary-support gain grid for held-out closure prediction.

This experiment is a targeted follow-up to P02. It keeps the baseline q=0.5
social-enabled condition and expands the harmful-count grid so held-out closure
prediction is supported near the P01 critical counts, especially at large N.
"""
from __future__ import annotations

import argparse
import os

from wolfbench.scenarios.base import ScenarioConfig

from ..runtime.runner import (
    add_run_args,
    as_run_args,
    profile_values,
    run_factorial,
    write_artifacts,
)


DEFAULT_BOUNDARY_SUPPORT_K = [2, 4, 8, 16, 32, 64]
DEFAULT_MAX_ALPHA = 0.32
VARIANT = "baseline_q05_boundary_support"


def harmful_counts() -> list[int]:
    raw = os.getenv("WOLFBENCH_P02_BOUNDARY_K")
    if raw:
        return [int(value.strip()) for value in raw.split(",") if value.strip()]
    return DEFAULT_BOUNDARY_SUPPORT_K[:]


def max_alpha() -> float:
    raw = os.getenv("WOLFBENCH_P02_BOUNDARY_MAX_ALPHA")
    if raw:
        return float(raw)
    return DEFAULT_MAX_ALPHA


def mutate(scenario: ScenarioConfig, _variant: str) -> tuple[ScenarioConfig, str | None]:
    scenario.retail["controller_mode"] = "mixed_roles"
    scenario.market_makers["liquidity_exponent"] = 0.5
    return scenario, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_args(parser, "p02_boundary_support_decomposition")
    args = as_run_args(parser.parse_args())
    seeds, n_values = profile_values(args.profile)
    counts = harmful_counts()
    alpha_cap = max_alpha()
    rows = []
    frozen_grid: dict[int, list[int]] = {}
    for n_society in n_values:
        n_counts = [count for count in counts if 0 < count <= n_society and count / n_society <= alpha_cap]
        frozen_grid[n_society] = n_counts
        for harmful_count in n_counts:
            alpha = harmful_count / n_society
            cell = run_factorial(
                experiment_id="P02_BOUNDARY_SUPPORT_DECOMPOSITION",
                args=args,
                scenarios=["s1"],
                n_values=[n_society],
                alphas={"s1": [alpha]},
                seeds=seeds,
                variants=[VARIANT],
                mutate=mutate,
            )
            for row in cell:
                row["requested_harmful_count"] = harmful_count
                row["boundary_support_grid"] = "1"
            rows.extend(cell)
    out = write_artifacts(
        args=args,
        experiment_id="P02_BOUNDARY_SUPPORT_DECOMPOSITION",
        rows=rows,
        config={
            "claim": "held-out closure boundary-support gain grid",
            "variant": VARIANT,
            "n_values": n_values,
            "harmful_counts_by_n": frozen_grid,
            "default_harmful_counts": DEFAULT_BOUNDARY_SUPPORT_K,
            "max_alpha": alpha_cap,
            "seeds": seeds,
            "preregistered_decision_rule": (
                "Use for positive closure only if leave-one-N-out prediction beats "
                "the leave-one-N-out constant-geomean alpha_c baseline."
            ),
        },
        group_keys=["variant", "n_society", "requested_harmful_count"],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
