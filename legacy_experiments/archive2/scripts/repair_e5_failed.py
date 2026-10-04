"""Repair E5 by rerunning only missing/error leaderboard cells.

Use this after changing defense parsing or model fallbacks. The script reads a
base E5 ``data.csv``, reruns target defenses only for cells without an OK row,
merges the new rows, and rebuilds the paper-facing leaderboard.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from wolfbench.scenarios.base import load_scenario

from .build_e5_leaderboard import scenario_key, write_outputs as write_leaderboard
from .hybrid_runtime import (
    DEFAULT_POPULATION_MODEL,
    defense_backend_snapshot,
    make_defense_policy,
    make_population_backend,
    run_hybrid_episode,
)
from .io_utils import OUTPUTS, env_list
from .run_common import backend_delta, defense_stats, write_run_outputs


DEFAULT_REPAIR_DEFENSES = (
    "zscore_guard",
    "oracle",
    "qwen235b_risk",
    "glm45_risk",
    "gpt41_risk",
    "gemini25_pro_risk",
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_base(base: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    base_dir = OUTPUTS / base
    data_path = base_dir / "data.csv"
    config_path = base_dir / "config.json"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return base_dir, json.loads(config_path.read_text()), _read_csv(data_path)


def _float_key(value: Any) -> str:
    return f"{float(value):.12g}"


def _int_key(value: Any) -> str:
    return str(int(float(value)))


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        scenario_key(row.get("scenario", "")),
        str(row.get("variant") or "baseline"),
        _int_key(row.get("n_society", 0)),
        _float_key(row.get("alpha", 0.0)),
        _int_key(row.get("seed", 0)),
        str(row.get("defense", "")),
    )


def _grid_from_config(config: dict[str, Any]) -> tuple[list[str], list[int], list[int], dict[str, list[float]]]:
    grid = config.get("grid", {}) or {}
    scenarios = [scenario_key(item) for item in grid.get("scenarios", ["s1", "s2", "s3", "s4"])]
    n_grid = [int(x) for x in grid.get("n_grid", [1000])]
    seeds = [int(x) for x in grid.get("seeds", [1, 2])]
    raw_alphas = grid.get("alphas_by_scenario", {}) or {}
    alphas = {
        scenario: [float(x) for x in raw_alphas.get(scenario, raw_alphas.get(scenario_key(scenario), [0.0]))]
        for scenario in scenarios
    }
    return scenarios, n_grid, seeds, alphas


def _target_defenses(value: str | None) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    env_value = os.getenv("WOLFBENCH_E5_REPAIR_DEFENSES")
    if env_value:
        return env_list("WOLFBENCH_E5_REPAIR_DEFENSES", env_value)
    return list(DEFAULT_REPAIR_DEFENSES)


def _jobs(
    existing_rows: list[dict[str, Any]],
    config: dict[str, Any],
    target_defenses: list[str],
    full_targets: bool,
) -> list[dict[str, Any]]:
    scenarios, n_grid, seeds, alphas = _grid_from_config(config)
    ok_keys = {
        row_key(row)
        for row in existing_rows
        if str(row.get("status", "ok")) == "ok"
    }
    jobs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for n_society in n_grid:
            for alpha in alphas[scenario]:
                for seed in seeds:
                    for defense in target_defenses:
                        key = (
                            scenario,
                            "baseline",
                            str(int(n_society)),
                            _float_key(alpha),
                            str(int(seed)),
                            defense,
                        )
                        if full_targets or key not in ok_keys:
                            jobs.append({
                                "scenario": scenario,
                                "variant": "baseline",
                                "n_society": int(n_society),
                                "alpha": float(alpha),
                                "seed": int(seed),
                                "defense": defense,
                            })
    return jobs


def _run_job(
    job: dict[str, Any],
    out_name: str,
    population_model: str,
    mock: bool,
    quota_mode: str,
    plan_interval: int,
    continue_on_error: bool,
) -> dict[str, Any]:
    started = time.time()
    population_backend = make_population_backend(
        model=population_model,
        experiment_name=out_name,
        mock=mock,
        strict=not mock,
    )
    defense_name = str(job["defense"])
    defense_policy = None if defense_name == "noguard" else make_defense_policy(
        defense_name,
        experiment_name=out_name,
        mock=mock,
    )
    pop_before = population_backend.snapshot()
    def_before = defense_backend_snapshot(defense_policy)
    try:
        row, _ = run_hybrid_episode(
            deepcopy(load_scenario(str(job["scenario"]))),
            n_society=int(job["n_society"]),
            alpha=float(job["alpha"]),
            seed=int(job["seed"]),
            population_backend=population_backend,
            quota_mode=quota_mode,
            plan_interval=plan_interval,
            defense_policy=defense_policy,
        )
        row["status"] = "ok"
        row["error_type"] = ""
        row["error"] = ""
    except Exception as exc:
        if not continue_on_error:
            raise
        row = {
            "scenario": job["scenario"],
            "n_society": job["n_society"],
            "alpha": job["alpha"],
            "seed": job["seed"],
            "status": "error",
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }
    pop_after = population_backend.snapshot()
    def_after = defense_backend_snapshot(defense_policy)
    row.update({
        "experiment": out_name,
        "variant": job.get("variant", "baseline"),
        "defense": defense_name,
        "mock_openrouter": int(mock),
        "runtime_sec": time.time() - started,
    })
    row.update(backend_delta(pop_before, pop_after, "population_llm_episode"))
    row.update(backend_delta(def_before, def_after, "defense_llm_episode"))
    row.update(defense_stats(defense_policy))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="real_e5_wolfguard_benchmark")
    parser.add_argument("--out", default=os.getenv("WOLFBENCH_E5_REPAIR_OUT", "real_e5_wolfguard_benchmark_repaired"))
    parser.add_argument("--defenses", default="")
    parser.add_argument("--full-targets", action="store_true", help="rerun every target-defense cell, not only missing/error cells")
    parser.add_argument("--mock", action="store_true", default=os.getenv("WOLFBENCH_FINAL_MOCK", "").lower() in {"1", "true", "yes"})
    parser.add_argument("--quota-mode", default=os.getenv("WOLFBENCH_FINAL_QUOTA_MODE", "standard"))
    parser.add_argument("--plan-interval", type=int, default=int(os.getenv("WOLFBENCH_FINAL_PLAN_INTERVAL", "5")))
    parser.add_argument("--model", default=os.getenv("WOLFBENCH_FINAL_POPULATION_MODEL", DEFAULT_POPULATION_MODEL))
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    _, config, base_rows = _load_base(args.base)
    target_defenses = _target_defenses(args.defenses or None)
    jobs = _jobs(base_rows, config, target_defenses, full_targets=args.full_targets)
    print(f"Repairing {len(jobs)} cells from {args.base} into {args.out}: {', '.join(target_defenses)}", flush=True)

    merged: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str, str, str, str]] = []
    for row in base_rows:
        key = row_key(row)
        if key not in merged:
            order.append(key)
        merged[key] = row

    for idx, job in enumerate(jobs, 1):
        print(
            f"[{idx}/{len(jobs)}] repair_e5 scenario={job['scenario']} N={job['n_society']} "
            f"alpha={job['alpha']} seed={job['seed']} defense={job['defense']}",
            flush=True,
        )
        row = _run_job(
            job,
            out_name=args.out,
            population_model=args.model,
            mock=args.mock,
            quota_mode=args.quota_mode,
            plan_interval=args.plan_interval,
            continue_on_error=not args.stop_on_error,
        )
        key = row_key(row)
        if key not in merged:
            order.append(key)
        merged[key] = row

    rows = [merged[key] for key in order]
    defenses = sorted({str(row.get("defense", "")) for row in rows if row.get("defense")})
    out_dir = write_run_outputs(
        args.out,
        rows,
        {
            "experiment": "E5 WolfGuard benchmark repair",
            "base_output": args.base,
            "repair_defenses": target_defenses,
            "defenses": defenses,
            "grid": config.get("grid", {}),
            "population_model": args.model,
            "mock_openrouter": args.mock,
            "quota_mode": args.quota_mode,
            "plan_interval": args.plan_interval,
        },
        ["scenario", "variant", "defense", "n_society", "alpha"],
    )
    write_leaderboard(args.out)
    print(f"Wrote repaired E5 to {out_dir}")


if __name__ == "__main__":
    main()
