"""Small post-run analysis for the exp6 Qwen/vLLM risk baseline."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from experiments._common import DEFENSE_BENCHMARK_OUTPUTS_ROOT


EXP6_DIR = DEFENSE_BENCHMARK_OUTPUTS_ROOT / os.getenv("WOLFBENCH_EXP6_OUT", "exp6")
TARGET_DEFENSE = os.getenv("WOLFBENCH_QWEN_DEFENSE", "qwen_risk")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    return _read_csv(path) if path.exists() else []


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", "None", "null"} else 0.0


def _first_float(row: dict[str, str], keys: list[str]) -> float:
    for key in keys:
        if key in row:
            return _float(row, key)
    return 0.0


def _defense_name(row: dict[str, str]) -> str:
    return row.get("defense") or row.get("Defense") or row.get("Defense model") or ""


def _leaderboard_score(row: dict[str, str]) -> float:
    return _first_float(row, ["TPS", "tps_official_mean", "tps_mean", "Avg DefenseScore", "defense_score_mean"])


def _threshold_shift(row: dict[str, str]) -> float:
    return _first_float(row, ["DeltaAlphaC/W0", "delta_alpha_c_over_w0_mean", "Avg ThresholdShift", "threshold_shift_mean"])


def _scenario_mean(rows: list[dict[str, str]], defense: str, key: str) -> float:
    vals = [_float(row, key) for row in rows if row.get("defense") == defense]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    scenario_path = EXP6_DIR / "leaderboard_by_scenario.csv"
    leaderboard_path = scenario_path if scenario_path.exists() else EXP6_DIR / "leaderboard.csv"
    overall_path = EXP6_DIR / "leaderboard_overall.csv"
    if not leaderboard_path.exists() or not overall_path.exists():
        raise FileNotFoundError("Run experiments.defense_benchmark.exp6_defense_leaderboard first.")

    leaderboard = _read_csv(leaderboard_path)
    overall = _read_csv(overall_path)
    comparison_rows = (
        overall
        + _read_csv_if_exists(EXP6_DIR / "leaderboard_controls.csv")
        + _read_csv_if_exists(EXP6_DIR / "leaderboard_upper_bounds.csv")
    )
    by_defense = {_defense_name(row): row for row in comparison_rows}
    target = TARGET_DEFENSE if TARGET_DEFENSE in by_defense else "qwen"
    if target not in by_defense:
        raise RuntimeError(f"The exp6 leaderboard does not contain {TARGET_DEFENSE} or qwen.")

    ordered = sorted(overall, key=_leaderboard_score, reverse=True)
    qwen = by_defense[target]
    qwen_rank = next(i for i, row in enumerate(ordered, 1) if _defense_name(row) == target)
    comparisons = {}
    for baseline in ["noguard", "random", "rule", "oracle"]:
        if baseline in by_defense:
            comparisons[baseline] = {
                "tps_delta": (
                    _leaderboard_score(qwen) - _leaderboard_score(by_defense[baseline])
                ),
                "collapse_rate_delta": (
                    _scenario_mean(leaderboard, target, "legacy_def_collapse_rate")
                    - _scenario_mean(leaderboard, baseline, "legacy_def_collapse_rate")
                ),
                "utility_loss_delta": (
                    _scenario_mean(leaderboard, target, "clean_utility_cost")
                    - _scenario_mean(leaderboard, baseline, "clean_utility_cost")
                ),
            }

    scenario_rows = [row for row in leaderboard if row.get("defense") == target]
    qwen_collapse_mean = _scenario_mean(leaderboard, target, "legacy_def_collapse_rate")
    qwen_utility_mean = _scenario_mean(leaderboard, target, "clean_utility_cost")
    analysis = {
        "target_defense": target,
        "qwen_rank": qwen_rank,
        "qwen_overall": qwen,
        "comparisons": comparisons,
        "qwen_by_scenario": scenario_rows,
    }
    (EXP6_DIR / "qwen_analysis.json").write_text(json.dumps(analysis, indent=2))

    lines = [
        f"# {target} vLLM Baseline Analysis",
        "",
        f"Overall rank: {qwen_rank} / {len(ordered)}",
        f"Mean TPS: {_leaderboard_score(qwen):.2f}",
        f"Mean DeltaAlphaC/W0: {_threshold_shift(qwen):.4f}",
        f"Mean CollapseRate: {qwen_collapse_mean:.3f}",
        f"Mean CleanCost: {qwen_utility_mean:.3f}",
        "",
        "## Deltas vs Baselines",
        "",
        "| baseline | Delta TPS | Delta CollapseRate | Delta CleanCost |",
        "|---|---:|---:|---:|",
    ]
    for baseline, vals in comparisons.items():
        lines.append(
            f"| {baseline} | {vals['tps_delta']:.2f} | "
            f"{vals['collapse_rate_delta']:.3f} | {vals['utility_loss_delta']:.3f} |"
        )
    lines.extend([
        "",
        "## Scenario Rows",
        "",
        "| scenario | TPS | RawNet | CollapseRate | RetailLoss | CleanCost | DeltaAlphaC/W0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in scenario_rows:
        lines.append(
            f"| {row['scenario']} | {_float(row, 'tps'):.2f} | "
            f"{_float(row, 'raw_net'):.2f} | "
            f"{_float(row, 'legacy_def_collapse_rate'):.3f} | "
            f"{_float(row, 'legacy_def_retail_loss'):.4f} | "
            f"{_float(row, 'clean_utility_cost'):.3f} | "
            f"{_float(row, 'delta_alpha_c_over_w0'):.4f} |"
        )
    (EXP6_DIR / "qwen_analysis.md").write_text("\n".join(lines) + "\n")
    print("Wrote", EXP6_DIR / "qwen_analysis.md")


if __name__ == "__main__":
    main()