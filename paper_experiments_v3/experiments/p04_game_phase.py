"""P04: game-theoretic social-coupling phase diagram."""
from __future__ import annotations

import argparse

from wolfbench.scenarios.base import ScenarioConfig

from ..runtime.runner import (
    add_run_args,
    alpha_values,
    as_run_args,
    load_protocol,
    profile_values,
    run_factorial,
    write_artifacts,
)


CONDITIONS = [
    "qre_weak_coupling",
    "qre_baseline",
    "qre_high_precision",
    "qre_high_conformity",
    "qre_high_reach",
    "qre_high_attention",
    "qre_strong_coupling",
    "mixed_roles_reference",
]


def mutate(scenario: ScenarioConfig, variant: str) -> tuple[ScenarioConfig, str | None]:
    if variant == "mixed_roles_reference":
        scenario.retail["controller_mode"] = "mixed_roles"
        return scenario, None
    scenario.retail["controller_mode"] = "all_value"
    scenario.retail["composition"] = {"value_investor": 1.0}
    if variant == "qre_weak_coupling":
        scenario.retail.update({
            "qre_beta_scale": 0.5,
            "conformity_scale": 0.25,
            "attention_capacity_scale": 0.5,
        })
        scenario.social["mean_degree"] = 4
    elif variant == "qre_high_precision":
        scenario.retail["qre_beta_scale"] = 2.0
    elif variant == "qre_high_conformity":
        scenario.retail["conformity_scale"] = 2.0
    elif variant == "qre_high_reach":
        scenario.social["mean_degree"] = 16
    elif variant == "qre_high_attention":
        scenario.retail["attention_capacity_scale"] = 2.0
    elif variant == "qre_strong_coupling":
        scenario.retail.update({
            "qre_beta_scale": 2.0,
            "conformity_scale": 2.0,
            "attention_capacity_scale": 2.0,
        })
        scenario.social["mean_degree"] = 16
    elif variant != "qre_baseline":
        raise ValueError(variant)
    return scenario, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_args(parser, "p04_game_phase")
    args = as_run_args(parser.parse_args())
    seeds, n_values = profile_values(args.profile)
    n_values = [n_values[0]] if args.profile != "paper" else [300, 1000]
    default_grid = load_protocol()["s1_refined_alpha"][100]
    alphas = {"s1": alpha_values("s1", default_grid)}
    rows = run_factorial(
        experiment_id="P04_GAME_PHASE",
        args=args,
        scenarios=["s1"],
        n_values=n_values,
        alphas=alphas,
        seeds=seeds,
        variants=CONDITIONS,
        mutate=mutate,
    )
    out = write_artifacts(
        args=args,
        experiment_id="P04_GAME_PHASE",
        rows=rows,
        config={
            "claim": "C3 game-theoretic coupling comparative statics",
            "conditions": CONDITIONS,
            "alphas": alphas,
            "n_values": n_values,
            "seeds": seeds,
            "warning": "J is a mean-field regime proxy, not a proof about the heterogeneous simulator",
        },
        group_keys=["variant", "n_society", "alpha"],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
