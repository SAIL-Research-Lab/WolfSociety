"""E4: robustness to the number of LLM-controlled agents."""
from __future__ import annotations

import argparse

from .run_common import add_common_args, grid_from_env, run_rows, write_run_outputs


QUOTAS = ["low", "standard", "high"]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, "e4_quota_robustness")
    args = parser.parse_args()
    grid = grid_from_env(
        args.preset,
        scenarios_default="s1,s2,s3,s4",
        n_default="300" if args.preset != "paper" else "1000",
    )
    rows = []
    for quota in QUOTAS:
        print(f"Running quota_variant={quota}", flush=True)
        quota_rows = run_rows(
            experiment_name=args.out,
            grid=grid,
            population_model=args.model,
            mock=args.mock,
            quota_mode=quota,
            plan_interval=args.plan_interval,
            continue_on_error=args.continue_on_error,
        )
        for row in quota_rows:
            row["quota_variant"] = quota
        rows.extend(quota_rows)
    out_dir = write_run_outputs(
        args.out,
        rows,
        {
            "experiment": "E4 LLM quota robustness",
            "quota_variants": QUOTAS,
            "grid": grid.__dict__,
            "population_model": args.model,
            "mock_openrouter": args.mock,
            "plan_interval": args.plan_interval,
        },
        ["scenario", "variant", "defense", "quota_variant", "n_society", "alpha"],
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
