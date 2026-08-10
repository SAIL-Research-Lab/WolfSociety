"""P08: compact defense-utility demonstration near recalibrated boundaries."""
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
    add_run_args(parser, "p08_defense_utility")
    args = as_run_args(parser.parse_args())
    seeds, _ = profile_values(args.profile)
    n_values = [100] if args.profile == "smoke" else [1000]
    scenarios = ["s1"] if args.profile == "smoke" else ["s1", "s2", "s3", "s4"]
    defaults = load_protocol()["cross_scenario_alpha"]
    alphas = {scenario: alpha_values(scenario, defaults[scenario]) for scenario in scenarios}
    defenses = (
        ["noguard", "zscore_guard"]
        if args.profile == "smoke"
        else ["noguard", "zscore_guard", "topology_aware", "deepseek_v3_risk", "oracle"]
    )
    rows = run_factorial(
        experiment_id="P08_DEFENSE_UTILITY",
        args=args,
        scenarios=scenarios,
        n_values=n_values,
        alphas=alphas,
        seeds=seeds,
        defenses=defenses,
    )
    out = write_artifacts(
        args=args,
        experiment_id="P08_DEFENSE_UTILITY",
        rows=rows,
        config={
            "purpose": "benchmark utility demonstration; not a central scientific claim",
            "defenses": defenses,
            "alphas": alphas,
            "n_values": n_values,
            "seeds": seeds,
            "prerequisite": "run only after v3 near-critical grids are frozen",
        },
        group_keys=["scenario", "defense", "n_society", "alpha"],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
