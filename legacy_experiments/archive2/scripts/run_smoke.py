"""No-network smoke test for final mixed-agent experiments."""
from __future__ import annotations

import argparse

from .run_common import Grid, add_common_args, run_rows, write_run_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, "smoke")
    args = parser.parse_args()
    grid = Grid(
        scenarios=["s1", "s2", "s3", "s4"],
        n_grid=[40],
        alphas_by_scenario={scenario: [0.10] for scenario in ["s1", "s2", "s3", "s4"]},
        seeds=[1],
    )
    rows = run_rows(
        experiment_name=args.out,
        grid=grid,
        population_model=args.model,
        mock=True,
        quota_mode=args.quota_mode,
        plan_interval=args.plan_interval,
        continue_on_error=args.continue_on_error,
    )
    for row in rows:
        if row.get("status", "ok") == "ok":
            assert int(float(row["n_llm_total"])) > 0
            assert int(float(row["n_policy_agents"])) > 0
            assert float(row["population_llm_episode_calls"]) > 0
    out_dir = write_run_outputs(
        args.out,
        rows,
        {
            "purpose": "no-network smoke for final mixed-agent pipeline",
            "mock_openrouter": True,
            "quota_mode": args.quota_mode,
            "plan_interval": args.plan_interval,
        },
        ["scenario", "variant", "defense", "n_society", "alpha"],
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
