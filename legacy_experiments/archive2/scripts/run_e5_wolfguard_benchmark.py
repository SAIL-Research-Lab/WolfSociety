"""E5: WolfGuard benchmark on mixed-agent episodes."""
from __future__ import annotations

import argparse
import os

from .io_utils import env_list
from .run_common import (
    add_common_args,
    aliases_for_paper_benchmark,
    grid_from_env,
    run_rows,
    write_run_outputs,
)


def defenses_for_preset(preset: str) -> list[str]:
    explicit = os.getenv("WOLFBENCH_FINAL_DEFENSES")
    if explicit:
        return env_list("WOLFBENCH_FINAL_DEFENSES", explicit)
    if preset == "paper":
        return aliases_for_paper_benchmark()
    if preset == "smoke":
        return ["noguard", "zscore_guard", "deepseek_v3_risk"]
    return ["noguard", "zscore_guard", "topology_aware", "oracle", "deepseek_v3_risk", "qwen235b_risk"]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, "e5_wolfguard_benchmark")
    args = parser.parse_args()
    grid = grid_from_env(
        args.preset,
        scenarios_default="s1,s2,s3,s4",
        n_default="100" if args.preset == "smoke" else ("300" if args.preset == "pilot" else "1000"),
    )
    defenses = defenses_for_preset(args.preset)
    rows = run_rows(
        experiment_name=args.out,
        grid=grid,
        population_model=args.model,
        mock=args.mock,
        quota_mode=args.quota_mode,
        plan_interval=args.plan_interval,
        continue_on_error=args.continue_on_error,
        defense_names=defenses,
    )
    out_dir = write_run_outputs(
        args.out,
        rows,
        {
            "experiment": "E5 WolfGuard benchmark",
            "defenses": defenses,
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
