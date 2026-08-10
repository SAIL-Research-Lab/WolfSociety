"""P05: information-theoretic private-to-social cascade analysis."""
from __future__ import annotations

import argparse
from copy import deepcopy

from wolfbench.scenarios.base import ScenarioConfig, load_scenario

from ..runtime.io_utils import OUTPUTS, append_jsonl, ensure_dir, write_csv_from_jsonl
from ..runtime.runner import (
    add_run_args,
    alpha_values,
    as_run_args,
    load_existing_ok_rows,
    load_protocol,
    planned_row_key,
    profile_values,
    run_detailed_episode,
    write_artifacts,
)


CONDITIONS = [
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


def mutate(scenario: ScenarioConfig, variant: str) -> tuple[ScenarioConfig, str | None]:
    scenario.retail["controller_mode"] = "mixed_roles"
    placement = None
    if variant == "private_only":
        scenario.social.update({"p_expose": 0.0, "p_reshare": 0.0})
        scenario.retail["conformity_scale"] = 0.0
    elif variant == "content_only":
        scenario.social["social_proof_visible"] = False
    elif variant == "proof_only":
        scenario.social["content_visible"] = False
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
    elif variant != "full_game":
        raise ValueError(variant)
    return scenario, placement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_args(parser, "p05_information_cascade")
    args = as_run_args(parser.parse_args())
    seeds, n_values = profile_values(args.profile)
    n_values = [n_values[0]] if args.profile != "paper" else [300, 1000]
    default_grid = load_protocol()["s1_refined_alpha"][100]
    alphas = alpha_values("s1", default_grid)
    out_dir = ensure_dir(OUTPUTS / args.out)
    rows_jsonl = out_dir / "rows.jsonl"
    decisions_jsonl = out_dir / "agent_decisions.jsonl"
    messages_jsonl = out_dir / "messages.jsonl"
    exposures_jsonl = out_dir / "exposures.jsonl"
    existing_ok = load_existing_ok_rows(out_dir)
    rows = []
    total = len(CONDITIONS) * len(n_values) * len(alphas) * len(seeds)
    index = 0
    for variant in CONDITIONS:
        scenario, placement = mutate(deepcopy(load_scenario("s1")), variant)
        for n_society in n_values:
            for alpha in alphas:
                for seed in seeds:
                    index += 1
                    key = planned_row_key(
                        scenario=scenario.id,
                        variant=variant,
                        n_society=n_society,
                        alpha=alpha,
                        seed=seed,
                        defense="noguard",
                        args=args,
                    )
                    if key in existing_ok:
                        rows.append(existing_ok[key])
                        print(
                            f"[skip {index}/{total}] P05 variant={variant} N={n_society} "
                            f"alpha={alpha} seed={seed}",
                            flush=True,
                        )
                        continue
                    print(
                        f"[{index}/{total}] P05 variant={variant} N={n_society} "
                        f"alpha={alpha} seed={seed}",
                        flush=True,
                    )
                    try:
                        row, d, m, e = run_detailed_episode(
                            scenario,
                            experiment_id="P05_INFORMATION_CASCADE",
                            args=args,
                            variant=variant,
                            n_society=n_society,
                            alpha=alpha,
                            seed=seed,
                            placement=placement,
                        )
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        print(
                            f"[{index}/{total}] P05 ERROR variant={variant} "
                            f"N={n_society} alpha={alpha} seed={seed}: {exc!r}",
                            flush=True,
                        )
                        row = {
                            "experiment_id": "P05_INFORMATION_CASCADE",
                            "scenario": scenario.id,
                            "variant": variant,
                            "n_society": n_society,
                            "alpha": alpha,
                            "seed": seed,
                            "defense": "noguard",
                            "profile": args.profile,
                            "mock_openrouter": int(args.mock),
                            "quota_mode": args.quota_mode,
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": repr(exc),
                        }
                        d, m, e = [], [], []
                    rows.append(row)
                    append_jsonl(row, rows_jsonl)
                    for event in d:
                        append_jsonl(event, decisions_jsonl)
                    for event in m:
                        append_jsonl(event, messages_jsonl)
                    for event in e:
                        append_jsonl(event, exposures_jsonl)
    out = write_artifacts(
        args=args,
        experiment_id="P05_INFORMATION_CASCADE",
        rows=rows,
        config={
            "claim": "C3 social information overtakes private information",
            "conditions": CONDITIONS,
            "alphas": alphas,
            "n_values": n_values,
            "seeds": seeds,
            "independent_unit": "episode/seed; agent-day rows are within-episode diagnostics",
            "primary_estimands": [
                "bias-corrected I(A;M|V,X)",
                "bias-corrected I(A;V|M,X)",
                "private signal quality",
                "social-to-trade transfer entropy",
                "private/social conflict-following rate",
            ],
        },
        group_keys=["variant", "n_society", "alpha"],
    )
    if decisions_jsonl.exists():
        write_csv_from_jsonl(decisions_jsonl, out / "agent_decisions.csv")
    if messages_jsonl.exists():
        write_csv_from_jsonl(messages_jsonl, out / "messages.csv")
    if exposures_jsonl.exists():
        write_csv_from_jsonl(exposures_jsonl, out / "exposures.csv")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
