"""E7: reviewer-facing retail-controller realism ablation.

This experiment compares five easy-to-explain population styles under matched
scenarios, graphs, attack fractions, and seeds. Mathematical implementations
(QRE, satisficing, heuristics) stay internal; paper-facing labels describe the
observable trading style.
"""
from __future__ import annotations

import argparse

from wolfbench.scenarios.base import ScenarioConfig

from .run_common import add_common_args, grid_from_env, run_rows, write_run_outputs
from .social_game import ROLE_CATALOG


CONTROLLERS = [
    "legacy_score",
    "mixed_roles",
    "all_risk_averse",
    "all_trend",
    "all_social",
    "all_aggressive",
]


def mutate_controller(scenario: ScenarioConfig, controller: str) -> tuple[ScenarioConfig, str]:
    if controller not in CONTROLLERS:
        raise ValueError(f"Unknown controller: {controller}")
    scenario.retail["controller_mode"] = controller
    return scenario, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "e7_behavioral_realism")
    args = parser.parse_args()
    grid = grid_from_env(
        args.preset,
        scenarios_default="s1,s2",
        n_default="40" if args.preset == "smoke" else ("300" if args.preset == "pilot" else "300,1000"),
    )
    rows = run_rows(
        experiment_name=args.out,
        grid=grid,
        population_model=args.model,
        mock=args.mock,
        quota_mode=args.quota_mode,
        plan_interval=args.plan_interval,
        continue_on_error=args.continue_on_error,
        variant_names=CONTROLLERS,
        scenario_mutator=mutate_controller,
    )
    out_dir = write_run_outputs(
        args.out,
        rows,
        {
            "experiment": "E7 behavioral realism and controller-class ablation",
            "research_question": (
                "Are the benchmark results an artifact of a homogeneous, fully rational, "
                "weighted-risk-score population?"
            ),
            "controllers": CONTROLLERS,
            "role_catalog": ROLE_CATALOG,
            "grid": grid.__dict__,
            "population_model": args.model,
            "mock_openrouter": args.mock,
            "quota_mode": args.quota_mode,
            "plan_interval": args.plan_interval,
        },
        ["scenario", "variant", "n_society", "alpha"],
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
