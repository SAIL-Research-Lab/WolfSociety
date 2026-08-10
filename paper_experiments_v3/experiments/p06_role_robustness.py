"""P06: plain-language retail-role realism and robustness."""
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


POPULATIONS = [
    "legacy_score",
    "mixed_roles",
    "all_risk_averse",
    "all_value",
    "all_trend",
    "all_social",
    "all_aggressive",
]


def mutate(scenario: ScenarioConfig, variant: str) -> tuple[ScenarioConfig, str | None]:
    scenario.retail["controller_mode"] = variant
    return scenario, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_args(parser, "p06_role_robustness")
    args = as_run_args(parser.parse_args())
    seeds, n_values = profile_values(args.profile)
    profile_n_values = n_values
    n_values = [n_values[0]] if args.profile != "paper" else [300, 1000]
    grid = alpha_values("s1", load_protocol()["s1_refined_alpha"][100])
    rows = run_factorial(
        experiment_id="P06_ROLE_ROBUSTNESS",
        args=args,
        scenarios=["s1"],
        n_values=n_values,
        alphas={"s1": grid},
        seeds=seeds,
        variants=POPULATIONS,
        mutate=mutate,
    )
    if args.profile == "paper":
        audited_n = [n for n in profile_n_values if n not in n_values]
        rows.extend(
            run_factorial(
                experiment_id="P06_BEHAVIORAL_DIVERSITY_AUDIT",
                args=args,
                scenarios=["s1"],
                n_values=audited_n,
                alphas={"s1": [0.0]},
                seeds=seeds,
                variants=["mixed_roles"],
                mutate=mutate,
            )
        )
    out = write_artifacts(
        args=args,
        experiment_id="P06_ROLE_ROBUSTNESS",
        rows=rows,
        config={
            "purpose": "reject the identical fully-rational score-agent explanation",
            "populations": POPULATIONS,
            "required_metrics": [
                "role_action_information_bits",
                "role_trade_rate_gap",
                "decision_role_entropy_normalized",
                "pairwise_role_behavior_jsd_bits",
                "hse_like_behavioral_diversity",
                "population_role_entropy_normalized",
                "population_effective_roles",
                "trade_participation_rate",
                "action_entropy_bits",
                "alpha=0 primary failure",
            ],
            "alphas": grid,
            "diversity_audit": (
                "For paper profile, mixed_roles is additionally evaluated at alpha=0 "
                "for every protocol N to test whether behavioral diversity itself "
                "changes mechanically with population size."
            ),
            "n_values": n_values,
            "seeds": seeds,
        },
        group_keys=["variant", "n_society", "alpha"],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
