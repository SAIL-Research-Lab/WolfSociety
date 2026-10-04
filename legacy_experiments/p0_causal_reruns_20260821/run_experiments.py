#!/usr/bin/env python3
"""Fresh native-liquidity and closed-loop causal reruns.

The script deliberately writes outside the historical experiment output tree.
Every row is checkpointed immediately and can be resumed safely.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
SIM_ROOT = (HERE / CONFIG["simulator_root"]).resolve()
sys.path.insert(0, str(SIM_ROOT / "src"))
sys.path.insert(0, str(SIM_ROOT))

from wolfbench.agents.social_game import install_network_signal_game, normalize_controller_mode
from wolfbench.env.environment import WolfBenchEnv
from wolfbench.scenarios.base import ScenarioConfig, load_scenario

from paper_experiments_v3.runtime.hybrid_runtime import (
    apply_hybrid_agents,
    episode_row,
    make_population_backend,
)


RAW = HERE / "results" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = [
        CONFIG_PATH,
        SIM_ROOT / "src/wolfbench/env/environment.py",
        SIM_ROOT / "src/wolfbench/env/market.py",
        SIM_ROOT / "src/wolfbench/agents/social_game.py",
        SIM_ROOT / "paper_experiments_v3/runtime/hybrid_runtime.py",
    ]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


SOURCE_HASH = source_fingerprint()


class ObservationRecorderEnv(WolfBenchEnv):
    """Record clean market-facing observations for paired replay."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.observation_replay: dict[int, dict[str, Any]] = {}

    def _build_observation(self, day, prices, recent_ret):
        observation = super()._build_observation(day, prices, recent_ret)
        self.observation_replay[int(day)] = deepcopy({
            "prices": observation["prices"],
            "market": observation["market"],
            "recent_return": observation["recent_return"],
            "volume_z": observation["volume_z"],
        })
        return observation


class CleanMarketReplayEnv(WolfBenchEnv):
    """Execute treated trades but expose a paired alpha=0 market path."""

    def __init__(self, *args: Any, observation_replay: dict[int, dict[str, Any]], **kwargs: Any):
        self._clean_observation_replay = deepcopy(observation_replay)
        super().__init__(*args, **kwargs)

    def _build_observation(self, day, prices, recent_ret):
        actual = super()._build_observation(day, prices, recent_ret)
        clean = deepcopy(self._clean_observation_replay[int(day)])
        clean["day"] = day
        clean["social_env"] = actual["social_env"]
        return clean


def scenario_for(q: float, *, social: bool = True) -> ScenarioConfig:
    scenario = deepcopy(load_scenario(CONFIG["scenario"]))
    scenario.retail["controller_mode"] = "mixed_roles"
    scenario.market_makers["liquidity_exponent"] = float(q)
    if not social:
        scenario.social["p_expose"] = 0.0
        scenario.social["p_reshare"] = 0.0
        scenario.retail["conformity_scale"] = 0.0
    return scenario


def run_episode(
    scenario: ScenarioConfig,
    *,
    n_society: int,
    alpha: float,
    seed: int,
    env_kind: str = "standard",
    replay: dict[int, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]] | None]:
    backend = make_population_backend(
        model="deterministic/mock",
        experiment_name=CONFIG["experiment_version"],
        mock=True,
        strict=False,
    )
    kwargs = dict(scenario=scenario, n_society=n_society, alpha=alpha, seed=seed)
    if env_kind == "record":
        env = ObservationRecorderEnv(**kwargs)
    elif env_kind == "replay":
        if replay is None:
            raise ValueError("replay observations are required")
        env = CleanMarketReplayEnv(**kwargs, observation_replay=replay)
    else:
        env = WolfBenchEnv(**kwargs)

    controller_mode = normalize_controller_mode(str(scenario.retail["controller_mode"]))
    install_network_signal_game(env, controller_mode=controller_mode)
    if env_kind == "replay" and hasattr(env.social, "feedback_strength"):
        # The treated market still evolves, but its return cannot amplify the
        # messages delivered to later agents.
        env.social.feedback_strength = 0.0
    if env_kind == "first_hop":
        # Keep harmful-source messages to immediate neighbors, but prohibit
        # second-hop bot resharing and all benign retransmission/posting.
        original_step = env.social.step

        def harmful_first_hop_step(day, messages, market_returns):
            old_reshare = env.social.p_reshare
            env.social.p_reshare = 0.0
            try:
                original_step(
                    day,
                    [message for message in messages if bool(message.is_harmful)],
                    market_returns,
                )
            finally:
                env.social.p_reshare = old_reshare

        env.social.step = harmful_first_hop_step
    mix = apply_hybrid_agents(
        env,
        backend=backend,
        quota_mode=CONFIG["quota_mode"],
        plan_interval=5,
    )
    result = env.run()
    row = episode_row(result, env, mix, backend)
    recorded = getattr(env, "observation_replay", None)
    return row, recorded


def load_existing(path: Path) -> set[tuple[str, str, int, str, int]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("source_hash") == SOURCE_HASH:
                keys.add(row_key(row))
    return keys


def row_key(row: dict[str, Any]) -> tuple[str, str, int, str, int]:
    return (
        str(row["experiment"]),
        str(row["variant"]),
        int(row["n_society"]),
        f"{float(row['alpha']):.12g}",
        int(row["seed"]),
    )


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def finalize_csv(path: Path) -> None:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def execute_cell(
    path: Path,
    existing: set[tuple[str, str, int, str, int]],
    *,
    experiment: str,
    variant: str,
    n_society: int,
    alpha: float,
    seed: int,
    q: float,
    social: bool = True,
    env_kind: str = "standard",
    replay: dict[int, dict[str, Any]] | None = None,
    requested_harmful_count: int | None = None,
) -> None:
    planned = {
        "experiment": experiment,
        "variant": variant,
        "n_society": n_society,
        "alpha": alpha,
        "seed": seed,
    }
    if row_key(planned) in existing:
        return
    started = time.time()
    try:
        row, _ = run_episode(
            scenario_for(q, social=social),
            n_society=n_society,
            alpha=alpha,
            seed=seed,
            env_kind=env_kind,
            replay=replay,
        )
        row.update({"status": "ok", "error": ""})
    except Exception as exc:
        row = {
            "n_society": n_society,
            "alpha": alpha,
            "seed": seed,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    row.update({
        "experiment": experiment,
        "variant": variant,
        "requested_harmful_count": requested_harmful_count if requested_harmful_count is not None else "",
        "native_liquidity_exponent": q,
        "environment_feedback_visible": int(env_kind != "replay"),
        "social_propagation_enabled": int(social),
        "source_hash": SOURCE_HASH,
        "experiment_version": CONFIG["experiment_version"],
        "fresh_rerun": 1,
        "runtime_sec": time.time() - started,
    })
    append_row(path, row)
    if row["status"] == "ok":
        existing.add(row_key(row))
    print(
        f"{experiment} {variant} N={n_society} alpha={alpha:.6g} "
        f"seed={seed} status={row['status']} t={row['runtime_sec']:.2f}s",
        flush=True,
    )


def run_fixed_k(seeds: list[int]) -> None:
    path = RAW / "liquidity_fixed_k.jsonl"
    existing = load_existing(path)
    cfg = CONFIG["fixed_k"]
    for q in CONFIG["liquidity_exponents"]:
        variant = f"native_q{q:g}"
        for n_society in cfg["n_values"]:
            for harmful_count in cfg["harmful_counts"]:
                for seed in seeds:
                    execute_cell(
                        path, existing,
                        experiment="liquidity_fixed_k",
                        variant=variant,
                        n_society=n_society,
                        alpha=harmful_count / n_society,
                        seed=seed,
                        q=q,
                        requested_harmful_count=harmful_count,
                    )
    finalize_csv(path)


def run_boundary(seeds: list[int]) -> None:
    path = RAW / "liquidity_boundary.jsonl"
    existing = load_existing(path)
    for q in CONFIG["liquidity_exponents"]:
        variant = f"native_q{q:g}"
        for n_text, alphas in CONFIG["boundary"]["alpha_by_n"].items():
            n_society = int(n_text)
            for alpha in alphas:
                for seed in seeds:
                    execute_cell(
                        path, existing,
                        experiment="liquidity_boundary",
                        variant=variant,
                        n_society=n_society,
                        alpha=alpha,
                        seed=seed,
                        q=q,
                    )
    finalize_csv(path)


def clean_replay(n_society: int, seed: int, q: float) -> dict[int, dict[str, Any]]:
    _, replay = run_episode(
        scenario_for(q),
        n_society=n_society,
        alpha=0.0,
        seed=seed,
        env_kind="record",
    )
    if replay is None or len(replay) != CONFIG["horizon_days"]:
        raise RuntimeError("clean replay did not record the full horizon")
    return replay


def run_feedback(seeds: list[int]) -> None:
    path = RAW / "feedback_ablation.jsonl"
    existing = load_existing(path)
    q = float(CONFIG["feedback_ablation"]["liquidity_exponent"])
    for n_text, alphas in CONFIG["feedback_ablation"]["alpha_by_n"].items():
        n_society = int(n_text)
        for seed in seeds:
            replay = clean_replay(n_society, seed, q)
            for alpha in alphas:
                execute_cell(
                    path, existing,
                    experiment="feedback_ablation",
                    variant="closed_loop",
                    n_society=n_society,
                    alpha=alpha,
                    seed=seed,
                    q=q,
                )
                execute_cell(
                    path, existing,
                    experiment="feedback_ablation",
                    variant="no_environment_feedback",
                    n_society=n_society,
                    alpha=alpha,
                    seed=seed,
                    q=q,
                    env_kind="replay",
                    replay=replay,
                )
                execute_cell(
                    path, existing,
                    experiment="feedback_ablation",
                    variant="no_social_propagation",
                    n_society=n_society,
                    alpha=alpha,
                    seed=seed,
                    q=q,
                    social=False,
                )
                execute_cell(
                    path, existing,
                    experiment="feedback_ablation",
                    variant="direct_harmful_first_hop",
                    n_society=n_society,
                    alpha=alpha,
                    seed=seed,
                    q=q,
                    env_kind="first_hop",
                )
    finalize_csv(path)


def parse_seeds(raw: str | None) -> list[int]:
    if raw is None:
        return [int(seed) for seed in CONFIG["seeds"]]
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["fixed_k", "boundary", "feedback", "all"],
        default="all",
    )
    parser.add_argument("--seeds", help="comma-separated override; default is frozen 1..12")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    print(f"source_hash={SOURCE_HASH} seeds={seeds}", flush=True)
    if args.experiment in {"fixed_k", "all"}:
        run_fixed_k(seeds)
    if args.experiment in {"boundary", "all"}:
        run_boundary(seeds)
    if args.experiment in {"feedback", "all"}:
        run_feedback(seeds)


if __name__ == "__main__":
    main()
