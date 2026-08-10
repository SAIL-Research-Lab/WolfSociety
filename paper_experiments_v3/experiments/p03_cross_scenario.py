"""P03: scope of the nonlinear harmful-minority effect across S1--S4."""
from __future__ import annotations

import argparse

from ..runtime.runner import (
    add_run_args,
    alpha_values,
    as_run_args,
    load_protocol,
    profile_values,
    run_factorial,
    write_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_args(parser, "p03_cross_scenario")
    args = as_run_args(parser.parse_args())
    seeds, n_values = profile_values(args.profile)
    if args.profile == "paper":
        n_values = [300, 1000]
    scenarios = ["s1", "s2", "s3", "s4"]
    defaults = load_protocol()["cross_scenario_alpha"]
    alphas = {scenario: alpha_values(scenario, defaults[scenario]) for scenario in scenarios}
    rows = run_factorial(
        experiment_id="P03_CROSS_SCENARIO",
        args=args,
        scenarios=scenarios,
        n_values=n_values,
        alphas=alphas,
        seeds=seeds,
    )
    out = write_artifacts(
        args=args,
        experiment_id="P03_CROSS_SCENARIO",
        rows=rows,
        config={
            "purpose": "S1 dense result, S2 directional replication, S3/S4 continuous-risk scope",
            "alphas": alphas,
            "n_values": n_values,
            "seeds": seeds,
            "censoring_rule": "never report alpha_c unless failure probability brackets 0.5",
        },
        group_keys=["scenario", "n_society", "alpha"],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
