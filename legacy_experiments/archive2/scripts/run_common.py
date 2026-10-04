"""Shared runner utilities for final mixed-agent experiments."""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from wolfbench.scenarios.base import ScenarioConfig, load_scenario

from .hybrid_runtime import (
    DEFAULT_POPULATION_MODEL,
    DEFENSE_MODEL_ALIASES,
    defense_backend_snapshot,
    make_defense_policy,
    make_population_backend,
    run_hybrid_episode,
)
from .io_utils import OUTPUTS, append_manifest, ensure_dir, env_float_list, env_int_list, env_list, write_csv, write_json


DEFAULT_ALPHA_GRIDS = {
    # Recalibrated for network_signal_game_v2. At N=100 the three-seed pilot
    # transitions between 0.05 (0/3 failures) and 0.075 (3/3 failures).
    "s1": [0.0, 0.02, 0.04, 0.05, 0.06, 0.075, 0.10],
    "s2": [0.0, 0.00025, 0.0005, 0.00075, 0.001, 0.002],
    "s3": [0.0, 0.03, 0.05, 0.075, 0.10, 0.15],
    "s4": [0.0, 0.015, 0.02, 0.03, 0.05, 0.08],
}

PRESET_SEEDS = {
    "smoke": [1],
    "pilot": [1, 2, 3],
    "paper": list(range(1, 11)),
}

PRESET_N = {
    "smoke": [40],
    "pilot": [100, 300],
    "paper": [100, 300, 1000, 3000, 10000],
}


@dataclass(frozen=True)
class Grid:
    scenarios: list[str]
    n_grid: list[int]
    alphas_by_scenario: dict[str, list[float]]
    seeds: list[int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def add_common_args(parser: argparse.ArgumentParser, default_out: str) -> None:
    parser.add_argument("--preset", choices=["smoke", "pilot", "paper"], default=os.getenv("WOLFBENCH_FINAL_PRESET", "pilot"))
    parser.add_argument("--out", default=os.getenv("WOLFBENCH_FINAL_OUT", default_out))
    parser.add_argument("--mock", action="store_true", default=env_bool("WOLFBENCH_FINAL_MOCK", False))
    parser.add_argument("--model", default=os.getenv("WOLFBENCH_FINAL_POPULATION_MODEL", DEFAULT_POPULATION_MODEL))
    parser.add_argument("--quota-mode", choices=["behavioral_only", "low", "standard", "high", "micro_full"], default=os.getenv("WOLFBENCH_FINAL_QUOTA_MODE", "standard"))
    parser.add_argument("--plan-interval", type=int, default=int(os.getenv("WOLFBENCH_FINAL_PLAN_INTERVAL", "5")))
    parser.add_argument("--continue-on-error", action="store_true", default=env_bool("WOLFBENCH_FINAL_CONTINUE_ON_ERROR", False))


def grid_from_env(
    preset: str,
    scenarios_default: str,
    n_default: str | None = None,
    seeds_default: str | None = None,
    alpha_default: str | None = None,
) -> Grid:
    scenarios = env_list("WOLFBENCH_FINAL_SCENARIOS", scenarios_default)
    n_grid = env_int_list(
        "WOLFBENCH_FINAL_N_GRID",
        n_default or ",".join(str(n) for n in PRESET_N[preset]),
    )
    seeds = env_int_list(
        "WOLFBENCH_FINAL_SEEDS",
        seeds_default or ",".join(str(seed) for seed in PRESET_SEEDS[preset]),
    )
    alphas_by_scenario = {}
    for scenario in scenarios:
        key = f"WOLFBENCH_FINAL_ALPHAS_{scenario.upper()}"
        if alpha_default is not None:
            default = alpha_default
        else:
            default = ",".join(str(alpha) for alpha in DEFAULT_ALPHA_GRIDS.get(scenario, [0.0, 0.01, 0.02]))
        alphas_by_scenario[scenario] = env_float_list(key, os.getenv("WOLFBENCH_FINAL_ALPHAS", default))
    return Grid(scenarios=scenarios, n_grid=n_grid, alphas_by_scenario=alphas_by_scenario, seeds=seeds)


def output_dir(name: str) -> Path:
    return ensure_dir(OUTPUTS / name)


def scenario_copy(scenario: str | ScenarioConfig) -> ScenarioConfig:
    base = load_scenario(scenario) if isinstance(scenario, str) else scenario
    return deepcopy(base)


def mutate_social_variant(scenario: ScenarioConfig, variant: str) -> tuple[ScenarioConfig, str]:
    scen = deepcopy(scenario)
    placement = ""
    if variant == "baseline":
        return scen, placement
    if variant == "no_social":
        scen.retail["beta_social"] = 0.0
        scen.social["p_expose"] = 0.0
        scen.social["p_reshare"] = 0.0
        scen.social["feedback_strength"] = 0.0
    elif variant == "high_exposure":
        scen.social["p_expose"] = 0.9
    elif variant == "high_reach":
        scen.social["mean_degree"] = max(16, int(scen.social.get("mean_degree", 8)) * 2)
    elif variant == "hub_placement":
        placement = "high_degree"
    elif variant == "reflexive":
        scen.social["feedback_strength"] = max(2.4, float(scen.social.get("feedback_strength", 0.6)) * 3.0)
        scen.social["p_reshare"] = 0.55
    else:
        raise ValueError(f"Unknown social variant: {variant}")
    return scen, placement


def backend_delta(before: dict[str, Any], after: dict[str, Any], prefix: str) -> dict[str, Any]:
    fields = ["calls", "cache_hits", "failures", "prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd"]
    out = {}
    for field in fields:
        out[f"{prefix}_{field}"] = float(after.get(field, 0.0)) - float(before.get(field, 0.0))
    out[f"{prefix}_backend"] = after.get("backend", "")
    out[f"{prefix}_model"] = after.get("model", "")
    out[f"{prefix}_last_error_type"] = after.get("last_error_type", "")
    return out


def defense_stats(policy: Any) -> dict[str, Any]:
    if policy is None:
        return {
            "defense_llm_decision_calls": 0,
            "defense_llm_expected_asset_decisions": 0,
            "defense_llm_valid_asset_decisions": 0,
            "defense_llm_semantic_fallbacks": 0,
            "defense_llm_fallback_days": 0,
        }
    return {
        "defense_llm_decision_calls": int(getattr(policy, "llm_decision_calls", 0)),
        "defense_llm_expected_asset_decisions": int(getattr(policy, "llm_expected_asset_decisions", 0)),
        "defense_llm_valid_asset_decisions": int(getattr(policy, "llm_valid_asset_decisions", 0)),
        "defense_llm_semantic_fallbacks": int(getattr(policy, "llm_semantic_fallbacks", 0)),
        "defense_llm_fallback_days": int(getattr(policy, "llm_fallback_days", 0)),
    }


def run_rows(
    experiment_name: str,
    grid: Grid,
    population_model: str,
    mock: bool,
    quota_mode: str,
    plan_interval: int,
    continue_on_error: bool = False,
    defense_names: Iterable[str] | None = None,
    variant_names: Iterable[str] | None = None,
    scenario_mutator: Callable[[ScenarioConfig, str], tuple[ScenarioConfig, str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    defenses = list(defense_names or ["noguard"])
    variants = list(variant_names or ["baseline"])
    total = sum(len(grid.alphas_by_scenario[s]) * len(grid.n_grid) * len(grid.seeds) * len(defenses) * len(variants) for s in grid.scenarios)
    idx = 0
    for scenario in grid.scenarios:
        for variant in variants:
            base_scenario = load_scenario(scenario)
            scen_obj, variant_placement = (
                scenario_mutator(base_scenario, variant) if scenario_mutator is not None else (base_scenario, "")
            )
            for n_society in grid.n_grid:
                for alpha in grid.alphas_by_scenario[scenario]:
                    for seed in grid.seeds:
                        for defense_name in defenses:
                            idx += 1
                            started = time.time()
                            print(
                                f"[{idx}/{total}] {experiment_name} scenario={scenario} variant={variant} "
                                f"N={n_society} alpha={alpha} seed={seed} defense={defense_name}",
                                flush=True,
                            )
                            population_backend = make_population_backend(
                                model=population_model,
                                experiment_name=experiment_name,
                                mock=mock,
                                strict=not mock,
                            )
                            defense_policy = None if defense_name == "noguard" else make_defense_policy(
                                defense_name,
                                experiment_name=experiment_name,
                                mock=mock,
                            )
                            pop_before = population_backend.snapshot()
                            def_before = defense_backend_snapshot(defense_policy)
                            try:
                                row, _ = run_hybrid_episode(
                                    deepcopy(scen_obj),
                                    n_society=n_society,
                                    alpha=alpha,
                                    seed=seed,
                                    population_backend=population_backend,
                                    quota_mode=quota_mode,
                                    plan_interval=plan_interval,
                                    defense_policy=defense_policy,
                                    placement_override=variant_placement or None,
                                )
                                row["status"] = "ok"
                                row["error_type"] = ""
                                row["error"] = ""
                            except Exception as exc:
                                if not continue_on_error:
                                    raise
                                row = {
                                    "scenario": scenario,
                                    "n_society": n_society,
                                    "alpha": alpha,
                                    "seed": seed,
                                    "status": "error",
                                    "error_type": type(exc).__name__,
                                    "error": repr(exc),
                                }
                            pop_after = population_backend.snapshot()
                            def_after = defense_backend_snapshot(defense_policy)
                            row.update({
                                "experiment": experiment_name,
                                "variant": variant,
                                "defense": defense_name,
                                "mock_openrouter": int(mock),
                                "runtime_sec": time.time() - started,
                            })
                            row.update(backend_delta(pop_before, pop_after, "population_llm_episode"))
                            row.update(backend_delta(def_before, def_after, "defense_llm_episode"))
                            row.update(defense_stats(defense_policy))
                            rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    numeric_fields = [
        "collapse_rate",
        "primary_failure_rate",
        "primary_failure_score_max",
        "max_collapse_score",
        "retail_loss_pct_30d",
        "harmful_profit",
        "wealth_transfer",
        "price_dislocation_max",
        "liquidity_stress_max",
        "social_cascade_peak",
        "wash_share_max",
        "volume_distortion_max",
        "cancel_rate_max",
        "spoof_depth_to_liquidity_max",
        "withdrawal_loss_max",
        "intervention_cost",
        "utility_loss",
        "false_positive_rate",
        "social_information_bits",
        "social_information_raw_bits",
        "social_information_null_bits",
        "private_information_bits",
        "private_information_raw_bits",
        "private_information_null_bits",
        "private_signal_quality_bits",
        "private_signal_direction_accuracy",
        "social_proof_information_bits",
        "social_proof_information_raw_bits",
        "social_proof_information_null_bits",
        "social_dominance_ratio",
        "action_entropy_bits",
        "trade_participation_rate",
        "role_action_information_bits",
        "role_trade_rate_gap",
        "mean_social_coupling_proxy",
        "mean_choice_entropy_bits",
        "mean_qre_action_entropy_bits",
        "mean_attention_used",
        "mean_sender_trust",
        "transfer_entropy_social_to_trade_bits",
        "transfer_entropy_social_to_trade_raw_bits",
        "transfer_entropy_social_to_trade_null_bits",
        "transfer_entropy_price_to_message_bits",
        "transfer_entropy_price_to_message_raw_bits",
        "transfer_entropy_price_to_message_null_bits",
        "cascade_decision_rate",
        "n_information_conflicts",
        "n_messages",
        "n_benign_messages",
        "n_reshares",
        "n_challenges",
        "max_cascade_reach",
        "mean_cascade_reach",
        "n_agent_decisions",
        "n_exposure_events",
        "n_llm_total",
        "n_policy_agents",
        "population_llm_episode_calls",
        "population_llm_episode_cache_hits",
        "population_llm_episode_failures",
        "population_llm_episode_estimated_cost_usd",
        "defense_llm_episode_calls",
        "defense_llm_episode_failures",
        "defense_llm_episode_estimated_cost_usd",
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, group in groups.items():
        record = {name: value for name, value in zip(group_keys, key)}
        record["n"] = len(group)
        record["n_ok"] = sum(1 for row in group if row.get("status", "ok") == "ok")
        for field in numeric_fields:
            values = []
            for row in group:
                try:
                    values.append(float(row.get(field, 0.0)))
                except (TypeError, ValueError):
                    pass
            if values:
                record[f"{field}_mean"] = sum(values) / len(values)
        out.append(record)
    return out


def write_run_outputs(
    out_name: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    summary_group_keys: list[str],
) -> Path:
    out_dir = output_dir(out_name)
    config = dict(config)
    config["created_at_utc"] = utc_now()
    config["output_dir"] = str(out_dir)
    write_json(config, out_dir / "config.json")
    write_csv(rows, out_dir / "data.csv")
    summary_rows = summarize_rows(rows, summary_group_keys)
    write_csv(summary_rows, out_dir / "summary_table.csv")
    summary = {
        "created_at_utc": utc_now(),
        "output_dir": str(out_dir),
        "n_rows": len(rows),
        "n_ok": sum(1 for row in rows if row.get("status", "ok") == "ok"),
        "n_error": sum(1 for row in rows if row.get("status", "ok") != "ok"),
        "total_population_llm_calls": sum(float(row.get("population_llm_episode_calls", 0.0)) for row in rows),
        "total_population_llm_cache_hits": sum(float(row.get("population_llm_episode_cache_hits", 0.0)) for row in rows),
        "total_population_llm_cost_usd": sum(float(row.get("population_llm_episode_estimated_cost_usd", 0.0)) for row in rows),
        "total_defense_llm_calls": sum(float(row.get("defense_llm_episode_calls", 0.0)) for row in rows),
        "total_defense_llm_cost_usd": sum(float(row.get("defense_llm_episode_estimated_cost_usd", 0.0)) for row in rows),
        "summary_group_keys": summary_group_keys,
    }
    write_json(summary, out_dir / "summary.json")
    append_manifest(out_name, {"config": config, "summary": summary})
    return out_dir


def aliases_for_paper_benchmark() -> list[str]:
    return [
        "noguard",
        "zscore_guard",
        "topology_aware",
        "oracle",
        *DEFENSE_MODEL_ALIASES.keys(),
    ]
