"""WolfBench trajectory dataset export.

The trajectory dataset stores exactly the public observation handed to a
WolfGuardPolicy, plus train-only oracle supervision and future outcome labels.
It is intentionally JSONL so downstream defense baselines can be trained
without importing the simulator.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from wolfbench.env.environment import EpisodeResult, WolfBenchEnv
from wolfbench.metrics.collapse import collapse_triggered
from wolfbench.scenarios.base import load_scenario


SCHEMA_VERSION = "wolfbench-trajectory-v1"
TRAIN_LABEL_SPLITS = {"public_dev"}


def labels_available_for_split(split: str, labels_policy: str = "auto") -> bool:
    """Return whether oracle/outcome labels should be emitted for a split."""
    if labels_policy == "include":
        return True
    if labels_policy == "hide":
        return False
    if labels_policy != "auto":
        raise ValueError("labels_policy must be one of: auto, include, hide")
    return split in TRAIN_LABEL_SPLITS


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items() if k != "oracle_view"}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else 0.0
    try:
        value = float(obj)
    except (TypeError, ValueError):
        return str(obj)
    return float(value) if math.isfinite(value) else 0.0


def _future_outcomes(daily_log: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    outcomes: list[dict[str, float | int]] = [{} for _ in daily_log]
    future_collapse = 0
    future_max_score = 0.0
    for idx in range(len(daily_log) - 1, -1, -1):
        entry = daily_log[idx]
        components = entry.get("components", {})
        future_collapse = int(future_collapse or collapse_triggered(components))
        future_max_score = max(future_max_score, float(entry.get("collapse_score", 0.0)))
        outcomes[idx] = {
            "future_collapse": future_collapse,
            "future_max_score": float(future_max_score),
        }
    return outcomes


def episode_to_trajectory_records(
    result: EpisodeResult,
    split: str,
    labels_policy: str = "auto",
) -> list[dict[str, Any]]:
    """Convert an EpisodeResult with recorded trajectory logs to JSONL rows."""
    include_labels = labels_available_for_split(split, labels_policy)
    outcomes = _future_outcomes(result.daily_log)
    rows: list[dict[str, Any]] = []

    for entry, outcome in zip(result.daily_log, outcomes):
        observation = _json_safe(entry.get("observation", {}))
        if not observation:
            continue
        oracle_actions = entry.get("oracle_actions", {}) or {}
        collapse_components = _json_safe(entry.get("components", {}))
        day = int(entry.get("day", observation.get("day", 0)))
        scenario_short = result.scenario_id.split("_", 1)[0]
        for asset in sorted(observation.get("market", {})):
            oracle_action = oracle_actions.get(asset, {}) or {}
            row = {
                "schema_version": SCHEMA_VERSION,
                "scenario": scenario_short,
                "scenario_id": result.scenario_id,
                "split": split,
                "seed": result.seed,
                "alpha": result.alpha,
                "n_society": result.n_society,
                "n_harmful": int(round(result.alpha * result.n_society)),
                "day": day,
                "asset": asset,
                "target_asset": result.target_asset,
                "is_target_asset": asset == result.target_asset,
                "observation": observation,
                "label_available": include_labels,
                "oracle_label": oracle_action.get("action", "none") if include_labels else None,
                "oracle_risk": float(oracle_action.get("risk", 0.0)) if include_labels else None,
                "oracle_components": _json_safe(oracle_action.get("components", {})) if include_labels else None,
                "future_collapse": outcome["future_collapse"] if include_labels else None,
                "future_max_score": outcome["future_max_score"] if include_labels else None,
                "collapse_components": collapse_components if include_labels else None,
            }
            rows.append(row)
    return rows


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def export_trajectory_dataset(
    out_path: str | Path,
    scenarios: Iterable[str],
    alphas: Iterable[float],
    n_society_grid: Iterable[int],
    seeds: Iterable[int],
    split: str,
    labels_policy: str = "auto",
) -> dict[str, Any]:
    """Run no-defense episodes and write public-observation trajectory JSONL."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = list(scenarios)
    alphas = [float(a) for a in alphas]
    n_society_grid = [int(n) for n in n_society_grid]
    seeds = [int(s) for s in seeds]
    include_labels = labels_available_for_split(split, labels_policy)

    episode_count = 0
    record_count = 0
    with open(out_path, "w") as handle:
        for scenario_id in scenarios:
            scenario = load_scenario(scenario_id)
            for n_society in n_society_grid:
                for alpha in alphas:
                    for seed in seeds:
                        env = WolfBenchEnv(
                            scenario,
                            n_society=n_society,
                            alpha=alpha,
                            seed=seed,
                            record_trajectory=True,
                        )
                        result = env.run()
                        rows = episode_to_trajectory_records(
                            result, split=split, labels_policy=labels_policy,
                        )
                        for row in rows:
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
                        episode_count += 1
                        record_count += len(rows)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "path": str(out_path),
        "split": split,
        "labels_policy": labels_policy,
        "labels_available": include_labels,
        "scenarios": scenarios,
        "alphas": alphas,
        "n_society": n_society_grid,
        "seeds": seeds,
        "episode_count": episode_count,
        "record_count": record_count,
        "observation_contract": "public WolfGuardPolicy summary only; oracle_view is never serialized",
    }
    metadata_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata