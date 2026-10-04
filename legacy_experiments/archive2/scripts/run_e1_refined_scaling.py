"""E1 refined finite-size scaling with N-specific alpha grids.

This runner is intentionally separate from ``run_e1_scaling``. It supports a
fully LLM-controlled micro-society endpoint at N<=10 via ``micro_full`` quota,
then uses the standard bounded LLM/policy agent society for larger N.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from copy import deepcopy
from typing import Any

from .hybrid_runtime import (
    defense_backend_snapshot,
    make_population_backend,
    run_hybrid_episode,
)
from .run_common import (
    add_common_args,
    backend_delta,
    defense_stats,
    write_run_outputs,
)
from .io_utils import OUTPUTS
from wolfbench.scenarios.base import load_scenario


# v3 exploratory brackets. Only the N=100 S1 interval has been pilot-checked;
# use pilot seeds first and refine each N before starting the paper run.
S1_GRID: dict[int, list[float]] = {
    10: [0.0, 0.10, 0.15, 0.20, 0.30],
    50: [0.0, 0.04, 0.06, 0.08, 0.10, 0.12],
    100: [0.0, 0.03, 0.04, 0.05, 0.06, 0.07, 0.075, 0.10],
    200: [0.0, 0.02, 0.03, 0.04, 0.05, 0.06],
    300: [0.0, 0.015, 0.025, 0.030, 0.035, 0.040, 0.050],
    500: [0.0, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040],
    1000: [0.0, 0.006, 0.010, 0.014, 0.018, 0.022, 0.030],
    2000: [0.0, 0.004, 0.007, 0.010, 0.013, 0.016, 0.022],
}

S2_GRID: dict[int, list[float]] = {
    100: [0.0, 0.005, 0.010, 0.020, 0.040, 0.080],
    300: [0.0, 0.001, 0.003, 0.006, 0.012, 0.025, 0.050],
    1000: [0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016],
    2000: [0.0, 0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008],
}

TRACKS = {
    "s1": ("s1", S1_GRID),
    "s2": ("s2", S2_GRID),
}


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _filter_grid(grid: dict[int, list[float]], n_values: list[int] | None) -> dict[int, list[float]]:
    if not n_values:
        return dict(grid)
    missing = [n for n in n_values if n not in grid]
    if missing:
        raise SystemExit(f"N values not available for refined grid: {missing}")
    return {n: grid[n] for n in n_values}


def _parse_alpha_grid(value: str) -> dict[int, list[float]]:
    """Parse N-specific top-up grids formatted as ``N:a,b,c;N:a,b,c``."""
    out: dict[int, list[float]] = {}
    if not value.strip():
        return out
    for block in value.split(";"):
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            raise SystemExit(f"Invalid alpha-grid block {block!r}; expected N:a,b,c")
        n_text, alpha_text = block.split(":", 1)
        n = int(n_text.strip())
        alphas = [float(item.strip()) for item in alpha_text.split(",") if item.strip()]
        if not alphas:
            raise SystemExit(f"Alpha-grid block {block!r} has no alpha values")
        out[n] = sorted(set(alphas))
    return out


def _scenario_key(value: Any) -> str:
    text = str(value).lower()
    if text.startswith("s1") or "pump" in text:
        return "s1"
    if text.startswith("s2") or "finfluencer" in text:
        return "s2"
    return text


def _dedupe_key(row: dict[str, Any]) -> tuple[str, int, float, int, str, str]:
    return (
        _scenario_key(row.get("scenario", "")),
        int(float(row.get("n_society", 0) or 0)),
        float(row.get("alpha", 0.0) or 0.0),
        int(float(row.get("seed", 0) or 0)),
        str(row.get("variant", "baseline") or "baseline"),
        str(row.get("defense", "noguard") or "noguard"),
    )


def _prefer_row(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    existing_ok = str(existing.get("status", "ok") or "ok") == "ok"
    incoming_ok = str(incoming.get("status", "ok") or "ok") == "ok"
    if incoming_ok and not existing_ok:
        return incoming
    if existing_ok and not incoming_ok:
        return existing
    return incoming


def append_existing_rows(out_name: str, new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = OUTPUTS / out_name / "data.csv"
    if not path.exists():
        return new_rows
    with path.open(newline="") as handle:
        existing = list(csv.DictReader(handle))
    merged: dict[tuple[str, int, float, int, str, str], dict[str, Any]] = {}
    for row in existing:
        key = _dedupe_key(row)
        merged[key] = _prefer_row(merged[key], row) if key in merged else row
    for row in new_rows:
        key = _dedupe_key(row)
        merged[key] = _prefer_row(merged[key], row) if key in merged else row
    return sorted(
        merged.values(),
        key=lambda row: (
            _scenario_key(row.get("scenario", "")),
            int(float(row.get("n_society", 0) or 0)),
            float(row.get("alpha", 0.0) or 0.0),
            int(float(row.get("seed", 0) or 0)),
        ),
    )


def existing_ok_keys(out_name: str) -> set[tuple[str, int, float, int, str, str]]:
    path = OUTPUTS / out_name / "data.csv"
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        _dedupe_key(row)
        for row in rows
        if str(row.get("status", "ok") or "ok") == "ok"
    }


def _default_quota_mode(args: argparse.Namespace) -> str:
    if "--quota-mode" not in sys.argv and "WOLFBENCH_FINAL_QUOTA_MODE" not in os.environ:
        return "micro_full"
    return str(args.quota_mode)


def run_refined_rows(
    *,
    experiment_name: str,
    scenario: str,
    grid: dict[int, list[float]],
    seeds: list[int],
    population_model: str,
    mock: bool,
    quota_mode: str,
    plan_interval: int,
    continue_on_error: bool,
    skip_keys: set[tuple[str, int, float, int, str, str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = sum(len(alphas) * len(seeds) for alphas in grid.values())
    idx = 0
    base_scenario = load_scenario(scenario)
    for n_society, alphas in grid.items():
        for alpha in alphas:
            for seed in seeds:
                idx += 1
                key = (scenario, int(n_society), float(alpha), int(seed), "baseline", "noguard")
                if skip_keys and key in skip_keys:
                    print(
                        f"[{idx}/{total}] skip existing {experiment_name} scenario={scenario} "
                        f"N={n_society} alpha={alpha} seed={seed}",
                        flush=True,
                    )
                    continue
                started = time.time()
                print(
                    f"[{idx}/{total}] {experiment_name} scenario={scenario} "
                    f"N={n_society} alpha={alpha} seed={seed} quota={quota_mode}",
                    flush=True,
                )
                population_backend = make_population_backend(
                    model=population_model,
                    experiment_name=experiment_name,
                    mock=mock,
                    strict=not mock,
                )
                pop_before = population_backend.snapshot()
                try:
                    row, _ = run_hybrid_episode(
                        deepcopy(base_scenario),
                        n_society=n_society,
                        alpha=alpha,
                        seed=seed,
                        population_backend=population_backend,
                        quota_mode=quota_mode,
                        plan_interval=plan_interval,
                        defense_policy=None,
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
                row.update({
                    "experiment": experiment_name,
                    "variant": "baseline",
                    "defense": "noguard",
                    "mock_openrouter": int(mock),
                    "runtime_sec": time.time() - started,
                })
                row.update(backend_delta(pop_before, pop_after, "population_llm_episode"))
                row.update(backend_delta(defense_backend_snapshot(None), defense_backend_snapshot(None), "defense_llm_episode"))
                row.update(defense_stats(None))
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, "real_e1_refined_s1")
    parser.add_argument("--track", choices=sorted(TRACKS), default=os.getenv("WOLFBENCH_E1_REFINED_TRACK", "s1"))
    parser.add_argument("--n-values", default=os.getenv("WOLFBENCH_E1_REFINED_N", ""))
    parser.add_argument("--seeds", default=os.getenv("WOLFBENCH_E1_REFINED_SEEDS", os.getenv("WOLFBENCH_FINAL_SEEDS", "1,2,3,4,5")))
    parser.add_argument("--alpha-grid", default=os.getenv("WOLFBENCH_E1_REFINED_ALPHA_GRID", ""))
    parser.add_argument("--append-existing", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    args = parser.parse_args()

    scenario, base_grid = TRACKS[args.track]
    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    if alpha_grid:
        missing = [n for n in alpha_grid if n not in base_grid]
        if missing:
            raise SystemExit(f"N values not available for refined grid: {missing}")
        grid = alpha_grid
    else:
        n_values = _parse_ints(args.n_values) if args.n_values.strip() else None
        grid = _filter_grid(base_grid, n_values)
    seeds = _parse_ints(args.seeds)
    quota_mode = _default_quota_mode(args)
    skip_keys = existing_ok_keys(args.out) if args.skip_existing else set()

    rows = run_refined_rows(
        experiment_name=args.out,
        scenario=scenario,
        grid=grid,
        seeds=seeds,
        population_model=args.model,
        mock=args.mock,
        quota_mode=quota_mode,
        plan_interval=args.plan_interval,
        continue_on_error=args.continue_on_error,
        skip_keys=skip_keys,
    )
    if args.append_existing:
        rows = append_existing_rows(args.out, rows)
    out_dir = write_run_outputs(
        args.out,
        rows,
        {
            "experiment": "E1 refined scaling",
            "track": args.track,
            "scenario": scenario,
            "n_specific_alpha_grid": grid,
            "seeds": seeds,
            "skip_existing": args.skip_existing,
            "population_model": args.model,
            "mock_openrouter": args.mock,
            "quota_mode": quota_mode,
            "plan_interval": args.plan_interval,
        },
        ["scenario", "variant", "defense", "n_society", "alpha"],
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
