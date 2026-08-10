"""P01: dense S1 nonlinear response and finite-size scaling."""
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
    add_run_args(parser, "p01_nonlinear_scaling")
    args = as_run_args(parser.parse_args())
    seeds, requested_n = profile_values(args.profile)
    protocol = load_protocol()
    refined = {int(n): values for n, values in protocol["s1_refined_alpha"].items()}
    n_values = [n for n in requested_n if n in refined]
    if not n_values:
        raise SystemExit(f"No refined S1 grids for requested N={requested_n}")
    rows = []
    frozen_grid = {}
    for n_society in n_values:
        grid = alpha_values("s1", refined[n_society])
        frozen_grid[n_society] = grid
        rows.extend(run_factorial(
            experiment_id="P01_NONLINEAR_SCALING",
            args=args,
            scenarios=["s1"],
            n_values=[n_society],
            alphas={"s1": grid},
            seeds=seeds,
        ))
    out = write_artifacts(
        args=args,
        experiment_id="P01_NONLINEAR_SCALING",
        rows=rows,
        config={
            "claims": ["C1 nonlinear response", "C2 finite-size shift"],
            "n_specific_alpha_grid": frozen_grid,
            "seeds": seeds,
            "preregistered_analyses": [
                "grid-interpolated alpha_c",
                "logistic midpoint sensitivity",
                "10-90 transition width",
                "seed bootstrap",
                "linear-vs-logistic likelihood comparison",
                "critical harmful count K_c=N*alpha_c",
            ],
        },
        group_keys=["n_society", "alpha"],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
