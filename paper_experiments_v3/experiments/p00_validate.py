"""P00: four-scenario integration and alpha=0 sanity validation."""
from __future__ import annotations

import argparse

from ..runtime.runner import add_run_args, as_run_args, run_factorial, write_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_args(parser, "p00_validate")
    args = as_run_args(parser.parse_args())
    scenarios = ["s1", "s2", "s3", "s4"]
    rows = run_factorial(
        experiment_id="P00_VALIDATE",
        args=args,
        scenarios=scenarios,
        n_values=[40],
        alphas={scenario: [0.0, 0.10] for scenario in scenarios},
        seeds=[1],
    )
    alpha0_failures = [
        row for row in rows
        if float(row.get("alpha", -1)) == 0.0
        and float(row.get("primary_failure_rate", 1.0)) != 0.0
    ]
    if alpha0_failures:
        raise RuntimeError(f"alpha=0 sanity failed: {alpha0_failures}")
    out = write_artifacts(
        args=args,
        experiment_id="P00_VALIDATE",
        rows=rows,
        config={"purpose": "integration and alpha=0 sanity; not paper evidence"},
        group_keys=["scenario", "alpha"],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
