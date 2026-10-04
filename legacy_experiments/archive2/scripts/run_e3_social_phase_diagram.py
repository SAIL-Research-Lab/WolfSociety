"""E3b: game-theoretic social-coupling phase diagram.

The controlled QRE population matches the assumptions of the mean-field proof.
Interventions separately vary response precision, conformity, network reach,
and attention capacity, then combine them in a strong-coupling condition. The
mixed-role population is retained as the external-validity reference.
"""
from __future__ import annotations

import argparse

from wolfbench.scenarios.base import ScenarioConfig

from .run_common import add_common_args, grid_from_env, run_rows, write_run_outputs


VARIANTS = [
    "qre_weak_coupling",
    "qre_baseline",
    "qre_high_precision",
    "qre_high_conformity",
    "qre_high_reach",
    "qre_high_attention",
    "qre_strong_coupling",
    "mixed_roles_reference",
]


def mutate_phase_condition(scenario: ScenarioConfig, variant: str) -> tuple[ScenarioConfig, str]:
    if variant == "mixed_roles_reference":
        scenario.retail["controller_mode"] = "mixed_roles"
        return scenario, ""
    scenario.retail["controller_mode"] = "all_value"
    scenario.retail["composition"] = {"value_investor": 1.0}
    if variant == "qre_weak_coupling":
        scenario.retail["qre_beta_scale"] = 0.5
        scenario.retail["conformity_scale"] = 0.25
        scenario.retail["attention_capacity_scale"] = 0.5
        scenario.social["mean_degree"] = 4
    elif variant == "qre_baseline":
        pass
    elif variant == "qre_high_precision":
        scenario.retail["qre_beta_scale"] = 2.0
    elif variant == "qre_high_conformity":
        scenario.retail["conformity_scale"] = 2.0
    elif variant == "qre_high_reach":
        scenario.social["mean_degree"] = 16
    elif variant == "qre_high_attention":
        scenario.retail["attention_capacity_scale"] = 2.0
    elif variant == "qre_strong_coupling":
        scenario.retail["qre_beta_scale"] = 2.0
        scenario.retail["conformity_scale"] = 2.0
        scenario.retail["attention_capacity_scale"] = 2.0
        scenario.social["mean_degree"] = 16
    else:
        raise ValueError(f"Unknown phase condition: {variant}")
    return scenario, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "e3_social_phase_diagram")
    args = parser.parse_args()
    grid = grid_from_env(
        args.preset,
        scenarios_default="s1",
        n_default="100" if args.preset == "smoke" else ("300" if args.preset == "pilot" else "300,1000"),
    )
    rows = run_rows(
        experiment_name=args.out,
        grid=grid,
        population_model=args.model,
        mock=args.mock,
        quota_mode=args.quota_mode,
        plan_interval=args.plan_interval,
        continue_on_error=args.continue_on_error,
        variant_names=VARIANTS,
        scenario_mutator=mutate_phase_condition,
    )
    out_dir = write_run_outputs(
        args.out,
        rows,
        {
            "experiment": "E3b game-theoretic social phase diagram",
            "research_question": "Does the observed cascade boundary follow the mean-field coupling condition J>1?",
            "variants": VARIANTS,
            "grid": grid.__dict__,
            "population_model": args.model,
            "mock_openrouter": args.mock,
            "quota_mode": args.quota_mode,
        },
        ["variant", "n_society", "alpha"],
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
