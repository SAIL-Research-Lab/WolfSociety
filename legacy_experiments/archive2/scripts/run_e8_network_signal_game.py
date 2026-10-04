"""E8: causal ablations for the bounded-rational network signaling game.

The interventions isolate message content, visible popularity, attention,
adaptive source trust, source identity, timing, and network position.  In
addition to episode-level outcomes, this runner writes agent decisions,
originating messages, and recipient exposures for information-theoretic audit.
"""
from __future__ import annotations

import argparse
import time
from copy import deepcopy
from typing import Any

from wolfbench.scenarios.base import ScenarioConfig, load_scenario

from .hybrid_runtime import make_population_backend, run_hybrid_episode_detailed
from .io_utils import write_csv
from .social_game import ROLE_CATALOG
from .run_common import (
    add_common_args,
    backend_delta,
    grid_from_env,
    output_dir,
    write_run_outputs,
)


VARIANTS = [
    "private_only",
    "content_only",
    "proof_only",
    "full_game",
    "low_attention",
    "high_attention",
    "precise_private_signal",
    "noisy_private_signal",
    "static_trust",
    "shuffled_sender",
    "delayed_messages",
    "hub_placement",
]


def mutate_signal_game(scenario: ScenarioConfig, variant: str) -> tuple[ScenarioConfig, str]:
    scenario.retail["controller_mode"] = "mixed_roles"
    placement = ""
    if variant == "private_only":
        scenario.social["p_expose"] = 0.0
        scenario.social["p_reshare"] = 0.0
        scenario.retail["conformity_scale"] = 0.0
    elif variant == "content_only":
        scenario.social["content_visible"] = True
        scenario.social["social_proof_visible"] = False
    elif variant == "proof_only":
        scenario.social["content_visible"] = False
        scenario.social["social_proof_visible"] = True
    elif variant == "full_game":
        pass
    elif variant == "low_attention":
        scenario.retail["attention_capacity_scale"] = 0.35
    elif variant == "high_attention":
        scenario.retail["attention_capacity_scale"] = 2.0
    elif variant == "precise_private_signal":
        scenario.retail["private_noise_scale"] = 0.5
    elif variant == "noisy_private_signal":
        scenario.retail["private_noise_scale"] = 2.0
    elif variant == "static_trust":
        scenario.retail["trust_learning_scale"] = 0.0
    elif variant == "shuffled_sender":
        scenario.social["shuffle_sender"] = True
    elif variant == "delayed_messages":
        scenario.social["message_delay"] = 2
    elif variant == "hub_placement":
        placement = "high_degree"
    else:
        raise ValueError(f"Unknown signal-game variant: {variant}")
    return scenario, placement


def _tag_events(
    events: list[dict[str, Any]], row: dict[str, Any], variant: str
) -> list[dict[str, Any]]:
    keys = {
        "scenario": row["scenario"],
        "variant": variant,
        "n_society": row["n_society"],
        "alpha": row["alpha"],
        "seed": row["seed"],
    }
    return [{**keys, **event} for event in events]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "e8_network_signal_game")
    args = parser.parse_args()
    grid = grid_from_env(
        args.preset,
        scenarios_default="s1",
        n_default="40" if args.preset == "smoke" else ("300" if args.preset == "pilot" else "1000"),
    )
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    exposures: list[dict[str, Any]] = []
    total = sum(
        len(grid.alphas_by_scenario[scenario]) * len(grid.n_grid) * len(grid.seeds) * len(VARIANTS)
        for scenario in grid.scenarios
    )
    index = 0
    for scenario_name in grid.scenarios:
        for variant in VARIANTS:
            scenario, placement = mutate_signal_game(deepcopy(load_scenario(scenario_name)), variant)
            for n_society in grid.n_grid:
                for alpha in grid.alphas_by_scenario[scenario_name]:
                    for seed in grid.seeds:
                        index += 1
                        print(
                            f"[{index}/{total}] {args.out} scenario={scenario_name} variant={variant} "
                            f"N={n_society} alpha={alpha} seed={seed}",
                            flush=True,
                        )
                        started = time.time()
                        backend = make_population_backend(
                            model=args.model,
                            experiment_name=args.out,
                            mock=args.mock,
                            strict=not args.mock,
                        )
                        before = backend.snapshot()
                        try:
                            row, _, episode_decisions, episode_messages, episode_exposures = run_hybrid_episode_detailed(
                                deepcopy(scenario),
                                n_society=n_society,
                                alpha=alpha,
                                seed=seed,
                                population_backend=backend,
                                quota_mode=args.quota_mode,
                                plan_interval=args.plan_interval,
                                placement_override=placement or None,
                            )
                            row.update({"status": "ok", "error_type": "", "error": ""})
                            decisions.extend(_tag_events(episode_decisions, row, variant))
                            messages.extend(_tag_events(episode_messages, row, variant))
                            exposures.extend(_tag_events(episode_exposures, row, variant))
                        except Exception as exc:
                            if not args.continue_on_error:
                                raise
                            row = {
                                "scenario": scenario_name,
                                "n_society": n_society,
                                "alpha": alpha,
                                "seed": seed,
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "error": repr(exc),
                            }
                        row.update({
                            "experiment": args.out,
                            "variant": variant,
                            "defense": "noguard",
                            "mock_openrouter": int(args.mock),
                            "runtime_sec": time.time() - started,
                        })
                        row.update(backend_delta(before, backend.snapshot(), "population_llm_episode"))
                        rows.append(row)

    out_dir = write_run_outputs(
        args.out,
        rows,
        {
            "experiment": "E8 bounded-rational network signaling game",
            "research_question": (
                "When do social content and visible social proof dominate private information "
                "and induce an information cascade?"
            ),
            "variants": VARIANTS,
            "role_catalog": ROLE_CATALOG,
            "grid": grid.__dict__,
            "population_model": args.model,
            "mock_openrouter": args.mock,
            "quota_mode": args.quota_mode,
            "plan_interval": args.plan_interval,
            "event_log_contract": {
                "agent_decisions.csv": "one asset-level decision per retail agent-day",
                "messages.csv": "one originating post, reshare, challenge, or harmful message",
                "exposures.csv": "one delivered message-recipient exposure",
            },
        },
        ["scenario", "variant", "n_society", "alpha"],
    )
    write_csv(decisions, out_dir / "agent_decisions.csv")
    write_csv(messages, out_dir / "messages.csv")
    write_csv(exposures, out_dir / "exposures.csv")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
