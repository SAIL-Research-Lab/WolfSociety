"""E6: controller-class robustness for the finite-size scaling claim.

This runner tests whether the S1 finite-size ordering survives when the
population controller class changes. It is intentionally small: the goal is a
targeted reviewer-facing ablation, not a replacement for the full E1 sweep.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from copy import deepcopy
from typing import Any

from wolfbench.scenarios.base import load_scenario

from .hybrid_runtime import defense_backend_snapshot, make_population_backend, run_hybrid_episode
from .io_utils import OUTPUTS
from .run_common import add_common_args, backend_delta, defense_stats, write_run_outputs


MAIN_GRID: dict[int, list[float]] = {
    300: [0.0, 0.015, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060],
    1000: [0.0, 0.006, 0.010, 0.014, 0.018, 0.022, 0.030, 0.040],
}

MICRO_GRID: dict[int, list[float]] = {
    10: [0.0, 0.10, 0.20, 0.30],
}

CONTROLLER_VARIANTS = {
    "behavioral_only": "behavioral_only",
    "mixed_default": "standard",
    "higher_llm_quota": "high",
}

MICRO_VARIANTS = {
    "full_llm_micro": "micro_full",
}


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_grid(value: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    if not value.strip():
        return out
    for block in value.split(";"):
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            raise SystemExit(f"Invalid grid block {block!r}; expected N:a,b,c")
        n_text, alpha_text = block.split(":", 1)
        alphas = [float(item.strip()) for item in alpha_text.split(",") if item.strip()]
        if not alphas:
            raise SystemExit(f"Grid block {block!r} has no alpha values")
        out[int(n_text.strip())] = sorted(set(alphas))
    return out


def _filter_grid(grid: dict[int, list[float]], n_values: list[int] | None) -> dict[int, list[float]]:
    if not n_values:
        return dict(grid)
    missing = [n for n in n_values if n not in grid]
    if missing:
        raise SystemExit(f"N values not available for E6 grid: {missing}")
    return {n: grid[n] for n in n_values}


def _filter_variants(value: str, variants: dict[str, str]) -> dict[str, str]:
    if not value.strip():
        return dict(variants)
    names = [item.strip() for item in value.split(",") if item.strip()]
    missing = [
        name for name in names
        if name not in variants and not (name.startswith("fixed_b") and "_h" in name)
    ]
    if missing:
        raise SystemExit(f"Controller variants not available for E6: {missing}")
    return {name: variants.get(name, name) for name in names}


def _dedupe_key(row: dict[str, Any]) -> tuple[str, int, float, int]:
    return (
        str(row.get("controller_variant", "")),
        int(float(row.get("n_society", 0) or 0)),
        float(row.get("alpha", 0.0) or 0.0),
        int(float(row.get("seed", 0) or 0)),
    )


def _prefer_row(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    existing_ok = str(existing.get("status", "ok") or "ok") == "ok"
    incoming_ok = str(incoming.get("status", "ok") or "ok") == "ok"
    if incoming_ok and not existing_ok:
        return incoming
    if existing_ok and not incoming_ok:
        return existing
    return incoming


def existing_ok_keys(out_name: str) -> set[tuple[str, int, float, int]]:
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


def append_existing_rows(out_name: str, new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = OUTPUTS / out_name / "data.csv"
    if not path.exists():
        return new_rows
    with path.open(newline="") as handle:
        existing = list(csv.DictReader(handle))
    merged: dict[tuple[str, int, float, int], dict[str, Any]] = {}
    for row in existing:
        key = _dedupe_key(row)
        merged[key] = _prefer_row(merged[key], row) if key in merged else row
    for row in new_rows:
        key = _dedupe_key(row)
        merged[key] = _prefer_row(merged[key], row) if key in merged else row
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("controller_variant", "")),
            int(float(row.get("n_society", 0) or 0)),
            float(row.get("alpha", 0.0) or 0.0),
            int(float(row.get("seed", 0) or 0)),
        ),
    )


def run_e6_rows(
    *,
    experiment_name: str,
    grid: dict[int, list[float]],
    seeds: list[int],
    variants: dict[str, str],
    population_model: str,
    mock: bool,
    plan_interval: int,
    continue_on_error: bool,
    skip_keys: set[tuple[str, int, float, int]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = sum(len(alphas) * len(seeds) for alphas in grid.values()) * len(variants)
    idx = 0
    scenario = load_scenario("s1")
    for controller_variant, quota_mode in variants.items():
        for n_society, alphas in grid.items():
            for alpha in alphas:
                for seed in seeds:
                    idx += 1
                    key = (controller_variant, int(n_society), float(alpha), int(seed))
                    if skip_keys and key in skip_keys:
                        print(
                            f"[{idx}/{total}] skip existing {experiment_name} variant={controller_variant} "
                            f"N={n_society} alpha={alpha} seed={seed}",
                            flush=True,
                        )
                        continue
                    started = time.time()
                    print(
                        f"[{idx}/{total}] {experiment_name} variant={controller_variant} "
                        f"quota={quota_mode} N={n_society} alpha={alpha} seed={seed}",
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
                            deepcopy(scenario),
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
                            "scenario": "s1",
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
                        "controller_variant": controller_variant,
                        "quota_variant": quota_mode,
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
    add_common_args(parser, "real_e6_controller_ablation")
    parser.add_argument("--seeds", default=os.getenv("WOLFBENCH_E6_SEEDS", os.getenv("WOLFBENCH_FINAL_SEEDS", "1,2")))
    parser.add_argument("--n-values", default=os.getenv("WOLFBENCH_E6_N", "300,1000"))
    parser.add_argument("--alpha-grid", default=os.getenv("WOLFBENCH_E6_ALPHA_GRID", ""))
    parser.add_argument("--controller-variants", default=os.getenv("WOLFBENCH_E6_CONTROLLER_VARIANTS", ""))
    parser.add_argument("--include-micro", action="store_true", default=os.getenv("WOLFBENCH_E6_INCLUDE_MICRO", "").lower() in {"1", "true", "yes"})
    parser.add_argument("--append-existing", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    args = parser.parse_args()

    seeds = _parse_ints(args.seeds)
    n_values = _parse_ints(args.n_values) if args.n_values.strip() else None
    alpha_grid = _parse_grid(args.alpha_grid)
    if alpha_grid:
        grid = alpha_grid
    else:
        grid = _filter_grid(MAIN_GRID, n_values)
    controller_variants = _filter_variants(args.controller_variants, CONTROLLER_VARIANTS)
    skip_keys = existing_ok_keys(args.out) if args.skip_existing else set()
    rows = run_e6_rows(
        experiment_name=args.out,
        grid=grid,
        seeds=seeds,
        variants=controller_variants,
        population_model=args.model,
        mock=args.mock,
        plan_interval=args.plan_interval,
        continue_on_error=args.continue_on_error,
        skip_keys=skip_keys,
    )
    if args.include_micro:
        rows.extend(
            run_e6_rows(
                experiment_name=args.out,
                grid=MICRO_GRID,
                seeds=seeds,
                variants=dict(MICRO_VARIANTS),
                population_model=args.model,
                mock=args.mock,
                plan_interval=args.plan_interval,
                continue_on_error=args.continue_on_error,
                skip_keys=skip_keys,
            )
        )
    if args.append_existing:
        rows = append_existing_rows(args.out, rows)
    out_dir = write_run_outputs(
        args.out,
        rows,
        {
            "experiment": "E6 controller-class robustness",
            "scenario": "s1",
            "n_specific_alpha_grid": grid,
            "micro_alpha_grid": MICRO_GRID if args.include_micro else {},
            "controller_variants": controller_variants,
            "micro_variants": MICRO_VARIANTS if args.include_micro else {},
            "seeds": seeds,
            "population_model": args.model,
            "mock_openrouter": args.mock,
            "plan_interval": args.plan_interval,
            "append_existing": args.append_existing,
            "skip_existing": args.skip_existing,
        },
        ["scenario", "controller_variant", "quota_variant", "n_society", "alpha"],
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
