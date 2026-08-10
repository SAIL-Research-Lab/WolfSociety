"""P02: fixed-K social/liquidity decomposition of the society-size effect."""
from __future__ import annotations

import argparse
from copy import deepcopy

from wolfbench.scenarios.base import ScenarioConfig

from ..runtime.runner import (
    add_run_args,
    as_run_args,
    load_protocol,
    profile_values,
    run_factorial,
    write_artifacts,
)


CONDITIONS = {
    "baseline_q05": {"q": 0.5, "social": True},
    "no_social_q05": {"q": 0.5, "social": False},
    "fixed_depth_q0": {"q": 0.0, "social": True},
    "per_capita_depth_q1": {"q": 1.0, "social": True},
    "per_capita_no_social_q1": {"q": 1.0, "social": False},
}


def mutate(scenario: ScenarioConfig, variant: str) -> tuple[ScenarioConfig, str | None]:
    settings = CONDITIONS[variant]
    scenario.retail["controller_mode"] = "mixed_roles"
    scenario.market_makers["liquidity_exponent"] = settings["q"]
    if not settings["social"]:
        scenario.social["p_expose"] = 0.0
        scenario.social["p_reshare"] = 0.0
        scenario.retail["conformity_scale"] = 0.0
    return scenario, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_args(parser, "p02_size_decomposition")
    args = as_run_args(parser.parse_args())
    seeds, n_values = profile_values(args.profile)
    harmful_counts = [int(value) for value in load_protocol()["fixed_harmful_counts"]]
    rows = []
    for n_society in n_values:
        for harmful_count in harmful_counts:
            cell = run_factorial(
                experiment_id="P02_SIZE_DECOMPOSITION",
                args=args,
                scenarios=["s1"],
                n_values=[n_society],
                alphas={"s1": [harmful_count / n_society]},
                seeds=seeds,
                variants=list(CONDITIONS),
                mutate=mutate,
            )
            for row in cell:
                row["requested_harmful_count"] = harmful_count
            rows.extend(cell)
    out = write_artifacts(
        args=args,
        experiment_id="P02_SIZE_DECOMPOSITION",
        rows=rows,
        config={
            "claim": "C2 size scaling after separating social and liquidity channels",
            "conditions": CONDITIONS,
            "n_values": n_values,
            "harmful_counts": harmful_counts,
            "seeds": seeds,
        },
        group_keys=["variant", "n_society", "requested_harmful_count"],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
