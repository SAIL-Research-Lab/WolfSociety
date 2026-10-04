"""Merge independently-run E5 shard outputs and rebuild the leaderboard."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .build_e5_leaderboard import scenario_key, write_outputs as write_leaderboard
from .io_utils import OUTPUTS
from .repair_e5_failed import row_key
from .run_common import write_run_outputs


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _merge_grid(configs: list[dict[str, Any]]) -> dict[str, Any]:
    scenarios: list[str] = []
    n_grid: list[int] = []
    seeds: list[int] = []
    alphas: dict[str, list[float]] = {}
    for config in configs:
        grid = config.get("grid", {}) or {}
        for item in grid.get("scenarios", []):
            key = scenario_key(item)
            if key not in scenarios:
                scenarios.append(key)
        for item in grid.get("n_grid", []):
            value = int(float(item))
            if value not in n_grid:
                n_grid.append(value)
        for item in grid.get("seeds", []):
            value = int(float(item))
            if value not in seeds:
                seeds.append(value)
        for scenario, values in (grid.get("alphas_by_scenario", {}) or {}).items():
            key = scenario_key(scenario)
            bucket = alphas.setdefault(key, [])
            for item in values:
                value = float(item)
                if value not in bucket:
                    bucket.append(value)
    return {
        "scenarios": sorted(scenarios),
        "n_grid": sorted(n_grid),
        "seeds": sorted(seeds),
        "alphas_by_scenario": {key: sorted(values) for key, values in sorted(alphas.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", required=True, help="Comma-separated output dirs under final_hybrid_experiments/outputs")
    args = parser.parse_args()

    shard_names = [item.strip() for item in args.shards.split(",") if item.strip()]
    if not shard_names:
        raise SystemExit("--shards is empty")

    configs: list[dict[str, Any]] = []
    merged: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str, str, str, str]] = []
    for name in shard_names:
        directory = OUTPUTS / name
        data_path = directory / "data.csv"
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        configs.append(_read_json(directory / "config.json"))
        for row in _read_csv(data_path):
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
            "experiment": "E5 WolfGuard benchmark, merged open-weight shards",
            "source_shards": shard_names,
            "defenses": defenses,
            "grid": _merge_grid(configs),
            "population_model": configs[0].get("population_model", ""),
            "mock_openrouter": any(bool(config.get("mock_openrouter", False)) for config in configs),
            "quota_mode": configs[0].get("quota_mode", "standard"),
            "plan_interval": configs[0].get("plan_interval", 5),
        },
        ["scenario", "variant", "defense", "n_society", "alpha"],
    )
    write_leaderboard(args.out)
    print(f"Merged {len(shard_names)} shards, {len(rows)} unique rows -> {out_dir}")


if __name__ == "__main__":
    main()
