"""E2: cross-mechanism scaling across S1-S4 under mixed populations."""
from __future__ import annotations

import argparse

from .run_common import add_common_args, grid_from_env, run_rows, write_run_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, "e2_mechanisms")
    args = parser.parse_args()
    n_default = "100,300" if args.preset != "paper" else "1000"
    grid = grid_from_env(args.preset, scenarios_default="s1,s2,s3,s4", n_default=n_default)
    rows = run_rows(
        experiment_name=args.out,
        grid=grid,
        population_model=args.model,
        mock=args.mock,
        quota_mode=args.quota_mode,
        plan_interval=args.plan_interval,
        continue_on_error=args.continue_on_error,
    )
    out_dir = write_run_outputs(
        args.out,
        rows,
        {
            "experiment": "E2 cross-mechanism",
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
