"""P10: LLM-fraction dense alpha_c(N) sweeps."""
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


QUOTAS = ["behavioral_only", "standard", "double"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_args(parser, "p10_llm_fraction_scaling")
    args = as_run_args(parser.parse_args())
    seeds, requested_n = profile_values(args.profile)
    protocol = load_protocol()
    refined = {int(n): values for n, values in protocol["s1_refined_alpha"].items()}
    n_values = [n for n in requested_n if n in refined]
    if not n_values:
        raise SystemExit(f"No refined S1 grids for requested N={requested_n}")

    rows = []
    frozen_grid = {}
    for quota in QUOTAS:
        quota_args = replace(args, quota_mode=quota)
        for n_society in n_values:
            grid = alpha_values("s1", refined[n_society])
            frozen_grid[n_society] = grid
            quota_rows = run_factorial(
                experiment_id="P10_LLM_FRACTION_SCALING",
                args=quota_args,
                scenarios=["s1"],
                n_values=[n_society],
                alphas={"s1": grid},
                seeds=seeds,
            )
            for row in quota_rows:
                row["quota_variant"] = quota
            rows.extend(quota_rows)
    out = write_artifacts(
        args=args,
        experiment_id="P10_LLM_FRACTION_SCALING",
        rows=rows,
        config={
            "reviewer_question": "Does alpha_c(N) and nu remain stable under 0x, 1x, and 2x LLM quotas?",
            "quota_variants": QUOTAS,
            "n_specific_alpha_grid": frozen_grid,
            "seeds": seeds,
        },
        group_keys=["quota_variant", "n_society", "alpha"],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()