"""E3: social-dynamic variants under mixed populations."""
from __future__ import annotations

import argparse

from .run_common import (
    add_common_args,
    grid_from_env,
    mutate_social_variant,
    run_rows,
    write_run_outputs,
)


VARIANTS = ["no_social", "baseline", "high_exposure", "high_reach", "hub_placement", "reflexive"]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, "e3_social_dynamics")
    args = parser.parse_args()
    grid = grid_from_env(
        args.preset,
        scenarios_default="s1",
        n_default="300" if args.preset != "paper" else "1000",
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
        scenario_mutator=mutate_social_variant,
    )
    out_dir = write_run_outputs(
        args.out,
        rows,
        {
            "experiment": "E3 social dynamics",
            "variants": VARIANTS,
            "grid": grid.__dict__,
            "population_model": args.model,
            "mock_openrouter": args.mock,
            "quota_mode": args.quota_mode,
            "plan_interval": args.plan_interval,
        },
        ["scenario", "variant", "defense", "n_society", "alpha"],
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
