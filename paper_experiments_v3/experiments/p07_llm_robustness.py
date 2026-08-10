"""P07: LLM-allocation robustness at matched benchmark conditions."""
from __future__ import annotations

import argparse
from dataclasses import replace

from ..runtime.runner import (
    add_run_args,
    alpha_values,
    as_run_args,
    load_protocol,
    profile_values,
    run_factorial,
    write_artifacts,
)


QUOTAS = ["behavioral_only", "low", "standard", "high"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_args(parser, "p07_llm_robustness")
    args = as_run_args(parser.parse_args())
    seeds, n_values = profile_values(args.profile)
    n_values = [n_values[0]] if args.profile != "paper" else [300, 1000]
    grid = alpha_values("s1", load_protocol()["s1_refined_alpha"][100])
    rows = []
    for quota in QUOTAS:
        quota_args = replace(args, quota_mode=quota)
        quota_rows = run_factorial(
            experiment_id="P07_LLM_ROBUSTNESS",
            args=quota_args,
            scenarios=["s1"],
            n_values=n_values,
            alphas={"s1": grid},
            seeds=seeds,
        )
        for row in quota_rows:
            row["quota_variant"] = quota
        rows.extend(quota_rows)
    out = write_artifacts(
        args=args,
        experiment_id="P07_LLM_ROBUSTNESS",
        rows=rows,
        config={
            "purpose": "show that C1/C2 do not arise from an N-dependent LLM count",
            "quota_variants": QUOTAS,
            "alphas": grid,
            "n_values": n_values,
            "seeds": seeds,
        },
        group_keys=["quota_variant", "n_society", "alpha"],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
