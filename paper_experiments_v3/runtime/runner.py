"""Shared execution and artifact contract for v3 paper experiments."""
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

import yaml

from wolfbench.scenarios.base import ScenarioConfig, load_scenario

from .hybrid_runtime import (
    DEFAULT_POPULATION_MODEL,
    defense_backend_snapshot,
    make_defense_policy,
    make_population_backend,
    run_hybrid_episode,
    run_hybrid_episode_detailed,
)
from .io_utils import OUTPUTS, ensure_dir, write_csv, write_json
from .io_utils import append_jsonl, read_csv_rows, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "protocol.yaml"


@dataclass(frozen=True)
class RunArgs:
    profile: str
    out: str
    mock: bool
    model: str
    quota_mode: str
    plan_interval: int
    continue_on_error: bool


def load_protocol() -> dict[str, Any]:
    with PROTOCOL_PATH.open() as handle:
        return yaml.safe_load(handle)


def add_run_args(parser: argparse.ArgumentParser, default_out: str) -> None:
    parser.add_argument("--profile", choices=["smoke", "pilot", "paper"], default="pilot")
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model", default=os.getenv("WOLFBENCH_PAPER_MODEL", DEFAULT_POPULATION_MODEL))
    parser.add_argument(
        "--quota-mode",
        default=os.getenv("WOLFBENCH_PAPER_QUOTA", "behavioral_only"),
        choices=["behavioral_only", "low", "standard", "high", "double", "micro_full"],
    )
    parser.add_argument("--plan-interval", type=int, default=5)
    parser.add_argument("--continue-on-error", action="store_true")


def as_run_args(namespace: argparse.Namespace) -> RunArgs:
    return RunArgs(
        profile=namespace.profile,
        out=namespace.out,
        mock=bool(namespace.mock),
        model=namespace.model,
        quota_mode=namespace.quota_mode,
        plan_interval=namespace.plan_interval,
        continue_on_error=bool(namespace.continue_on_error),
    )


def profile_values(profile: str) -> tuple[list[int], list[int]]:
    config = load_protocol()["profiles"][profile]
    seeds = _env_ints("WOLFBENCH_PAPER_SEEDS", config["seeds"])
    n_values = _env_ints("WOLFBENCH_PAPER_N", config["n"])
    return seeds, n_values


def alpha_values(scenario: str, defaults: Iterable[float]) -> list[float]:
    key = f"WOLFBENCH_PAPER_ALPHAS_{scenario.upper()}"
    raw = os.getenv(key) or os.getenv("WOLFBENCH_PAPER_ALPHAS")
    if raw:
        return [float(value.strip()) for value in raw.split(",") if value.strip()]
    return [float(value) for value in defaults]


def run_factorial(
    *,
    experiment_id: str,
    args: RunArgs,
    scenarios: list[str],
    n_values: list[int],
    alphas: dict[str, list[float]],
    seeds: list[int],
    variants: list[str] | None = None,
    mutate: Callable[[ScenarioConfig, str], tuple[ScenarioConfig, str | None]] | None = None,
    defenses: list[str] | None = None,
) -> list[dict[str, Any]]:
    variants = variants or ["baseline"]
    defenses = defenses or ["noguard"]
    out_dir = ensure_dir(OUTPUTS / args.out)
    checkpoint_path = out_dir / "rows.jsonl"
    existing_ok = load_existing_ok_rows(out_dir)
    total = sum(
        len(alphas[scenario]) * len(n_values) * len(seeds) * len(variants) * len(defenses)
        for scenario in scenarios
    )
    rows: list[dict[str, Any]] = []
    index = 0
    for scenario_name in scenarios:
        for variant in variants:
            base = deepcopy(load_scenario(scenario_name))
            scenario, placement = mutate(base, variant) if mutate else (base, None)
            scenario_id = scenario.id
            for n_society in n_values:
                for alpha in alphas[scenario_name]:
                    for seed in seeds:
                        for defense_name in defenses:
                            index += 1
                            key = planned_row_key(
                                scenario=scenario_id,
                                variant=variant,
                                n_society=n_society,
                                alpha=alpha,
                                seed=seed,
                                defense=defense_name,
                                args=args,
                            )
                            if key in existing_ok:
                                rows.append(existing_ok[key])
                                print(
                                    f"[skip {index}/{total}] {experiment_id} scenario={scenario_name} "
                                    f"variant={variant} N={n_society} alpha={alpha} "
                                    f"seed={seed} defense={defense_name}",
                                    flush=True,
                                )
                                continue
                            print(
                                f"[{index}/{total}] {experiment_id} scenario={scenario_name} "
                                f"variant={variant} N={n_society} alpha={alpha} "
                                f"seed={seed} defense={defense_name}",
                                flush=True,
                            )
                            backend = make_population_backend(
                                model=args.model,
                                experiment_name=args.out,
                                mock=args.mock,
                                strict=not args.mock,
                            )
                            defense = None if defense_name == "noguard" else make_defense_policy(
                                defense_name, experiment_name=args.out, mock=args.mock
                            )
                            before = backend.snapshot()
                            defense_before = defense_backend_snapshot(defense)
                            started = time.time()
                            try:
                                row, _ = run_hybrid_episode(
                                    deepcopy(scenario),
                                    n_society=n_society,
                                    alpha=alpha,
                                    seed=seed,
                                    population_backend=backend,
                                    quota_mode=args.quota_mode,
                                    plan_interval=args.plan_interval,
                                    defense_policy=defense,
                                    placement_override=placement,
                                )
                                row.update({"status": "ok", "error_type": "", "error": ""})
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
                                "experiment_id": experiment_id,
                                "variant": variant,
                                "defense": defense_name,
                                "profile": args.profile,
                                "mock_openrouter": int(args.mock),
                                "runtime_sec": time.time() - started,
                            })
                            row.update(_snapshot_delta(before, backend.snapshot(), "population_llm"))
                            row.update(_snapshot_delta(
                                defense_before, defense_backend_snapshot(defense), "defense_llm"
                            ))
                            rows.append(row)
                            append_jsonl(row, checkpoint_path)
    return rows


def run_detailed_episode(
    scenario: ScenarioConfig,
    *,
    experiment_id: str,
    args: RunArgs,
    variant: str,
    n_society: int,
    alpha: float,
    seed: int,
    placement: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    backend = make_population_backend(
        model=args.model,
        experiment_name=args.out,
        mock=args.mock,
        strict=not args.mock,
    )
    before = backend.snapshot()
    started = time.time()
    row, _, decisions, messages, exposures = run_hybrid_episode_detailed(
        deepcopy(scenario),
        n_society=n_society,
        alpha=alpha,
        seed=seed,
        population_backend=backend,
        quota_mode=args.quota_mode,
        plan_interval=args.plan_interval,
        placement_override=placement,
    )
    row.update({
        "experiment_id": experiment_id,
        "variant": variant,
        "defense": "noguard",
        "profile": args.profile,
        "mock_openrouter": int(args.mock),
        "runtime_sec": time.time() - started,
        "status": "ok",
        "error_type": "",
        "error": "",
    })
    row.update(_snapshot_delta(before, backend.snapshot(), "population_llm"))
    tags = {
        "experiment_id": experiment_id,
        "scenario": row["scenario"],
        "variant": variant,
        "n_society": n_society,
        "alpha": alpha,
        "seed": seed,
    }
    return (
        row,
        [{**tags, **event} for event in decisions],
        [{**tags, **event} for event in messages],
        [{**tags, **event} for event in exposures],
    )


def write_artifacts(
    *,
    args: RunArgs,
    experiment_id: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    group_keys: list[str],
    extra_csv: dict[str, list[dict[str, Any]]] | None = None,
) -> Path:
    out_dir = ensure_dir(OUTPUTS / args.out)
    metadata = {
        **config,
        "experiment_id": experiment_id,
        "benchmark_version": load_protocol()["benchmark_version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "model": args.model,
        "quota_mode": args.quota_mode,
        "mock_openrouter": args.mock,
        "output_dir": str(out_dir),
    }
    write_json(metadata, out_dir / "config.json")
    write_csv(rows, out_dir / "data.csv")
    write_csv(_summarize(rows, group_keys), out_dir / "summary.csv")
    for filename, values in (extra_csv or {}).items():
        write_csv(values, out_dir / filename)
    write_json({
        "n_rows": len(rows),
        "n_ok": sum(row.get("status", "ok") == "ok" for row in rows),
        "n_error": sum(row.get("status", "ok") != "ok" for row in rows),
    }, out_dir / "run_status.json")
    return out_dir


def planned_row_key(
    *,
    scenario: str,
    variant: str,
    n_society: int,
    alpha: float,
    seed: int,
    defense: str,
    args: RunArgs,
) -> tuple[str, ...]:
    return (
        str(scenario),
        str(variant),
        str(int(n_society)),
        _norm_float(alpha),
        str(int(seed)),
        str(defense),
        str(args.profile),
        str(int(args.mock)),
        str(args.quota_mode),
    )


def row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("scenario", "")),
        str(row.get("variant", "baseline")),
        str(int(float(row.get("n_society", 0) or 0))),
        _norm_float(row.get("alpha", 0.0)),
        str(int(float(row.get("seed", 0) or 0))),
        str(row.get("defense", "noguard")),
        str(row.get("profile", "")),
        str(int(float(row.get("mock_openrouter", 0) or 0))),
        str(row.get("quota_mode", "")),
    )


def load_existing_ok_rows(out_dir: Path) -> dict[tuple[str, ...], dict[str, Any]]:
    rows = read_csv_rows(out_dir / "data.csv") + read_jsonl(out_dir / "rows.jsonl")
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("status", "ok") == "ok":
            out[row_key(row)] = row
    return out


def _norm_float(value: Any) -> str:
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return str(value)


def _snapshot_delta(before: dict[str, Any], after: dict[str, Any], prefix: str) -> dict[str, Any]:
    fields = ("calls", "cache_hits", "failures", "prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd")
    return {
        f"{prefix}_{field}": float(after.get(field, 0.0)) - float(before.get(field, 0.0))
        for field in fields
    }


def _summarize(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in group_keys)].append(row)
    preferred = (
        "primary_failure_rate", "primary_failure_score_max", "collapse_rate",
        "retail_loss_pct_30d", "harmful_profit", "price_dislocation_max",
        "social_cascade_peak", "social_information_bits", "private_information_bits",
        "social_dominance_ratio", "cascade_decision_rate",
        "transfer_entropy_social_to_trade_bits", "role_action_information_bits",
        "trade_participation_rate", "mean_social_coupling_proxy",
    )
    summaries = []
    for key, group in groups.items():
        record = {name: value for name, value in zip(group_keys, key)}
        record["n"] = len(group)
        record["n_ok"] = sum(row.get("status", "ok") == "ok" for row in group)
        for field in preferred:
            values = []
            for row in group:
                try:
                    values.append(float(row[field]))
                except (KeyError, TypeError, ValueError):
                    pass
            if values:
                record[f"{field}_mean"] = sum(values) / len(values)
        summaries.append(record)
    return summaries


def _env_ints(name: str, default: Iterable[int]) -> list[int]:
    raw = os.getenv(name)
    if raw:
        return [int(value.strip()) for value in raw.split(",") if value.strip()]
    return [int(value) for value in default]
