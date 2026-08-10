"""Reviewer-facing scaling robustness and grouped nu audits."""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime.io_utils import OUTPUTS, ensure_dir, write_csv, write_json


SEED = 202707


def read_rows(run: str) -> list[dict[str, str]]:
    shard_paths = sorted(OUTPUTS.glob(f"{run}_s*/data.csv"))
    paths = shard_paths or [OUTPUTS / run / "data.csv"]
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows.extend(row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok")
    return rows


def as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def crossing(points: list[tuple[float, float]], target: float = 0.5) -> float | None:
    points = sorted(points)
    for (a0, p0), (a1, p1) in zip(points, points[1:]):
        if p0 == target:
            return a0
        if (p0 - target) * (p1 - target) <= 0 and p1 != p0:
            return a0 + (target - p0) * (a1 - a0) / (p1 - p0)
    return None


def curve_diagnostics(points: list[tuple[float, float]], target: float = 0.5) -> dict[str, Any]:
    points = sorted(points)
    if not points:
        return {
            "alpha_min": "",
            "alpha_max": "",
            "p_min": "",
            "p_max": "",
            "p_at_alpha_min": "",
            "p_at_alpha_max": "",
            "coverage_status": "empty",
            "censoring_direction": "empty",
        }

    alpha_min, p_at_alpha_min = points[0]
    alpha_max, p_at_alpha_max = points[-1]
    probs = [prob for _, prob in points]
    midpoint = crossing(points, target)
    if midpoint is not None:
        coverage_status = "crosses_target"
        censoring = "resolved"
    elif max(probs) < target:
        coverage_status = "right_censored_below_target"
        censoring = "right"
    elif min(probs) > target:
        coverage_status = "left_censored_above_target"
        censoring = "left"
    else:
        coverage_status = "nonmonotone_unresolved"
        censoring = "ambiguous"

    positive_points = [(alpha, prob) for alpha, prob in points if alpha > 0]
    first_positive_alpha = positive_points[0][0] if positive_points else ""
    first_positive_p = positive_points[0][1] if positive_points else ""
    first_positive_crosses = (
        first_positive_p != "" and float(first_positive_p) >= target and p_at_alpha_min < target
    )
    return {
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "p_min": float(min(probs)),
        "p_max": float(max(probs)),
        "p_at_alpha_min": p_at_alpha_min,
        "p_at_alpha_max": p_at_alpha_max,
        "first_positive_alpha": first_positive_alpha,
        "first_positive_p": first_positive_p,
        "first_positive_crosses_target": int(first_positive_crosses),
        "coverage_status": coverage_status,
        "censoring_direction": censoring,
    }


def aggregate_curve(rows: list[dict[str, str]]) -> list[tuple[float, float]]:
    by_alpha: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        alpha = as_float(row, "alpha")
        value = as_float(row, "primary_failure_rate")
        if math.isfinite(alpha) and math.isfinite(value):
            by_alpha[alpha].append(value)
    return sorted((alpha, float(np.mean(values))) for alpha, values in by_alpha.items())


def logistic_midpoint(rows: list[dict[str, str]]) -> float | None:
    x = np.asarray([as_float(row, "alpha") for row in rows], dtype=float)
    y = np.asarray([as_float(row, "primary_failure_rate") for row in rows], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or len(np.unique(y)) < 2 or float(np.std(x)) <= 0:
        return None
    center = float(np.mean(x))
    scale = float(np.std(x))
    z = (x - center) / scale
    design = np.column_stack([np.ones_like(z), z])
    theta = np.zeros(2, dtype=float)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(design @ theta, -30.0, 30.0)))
        weights = np.clip(p * (1.0 - p), 1e-8, None)
        hessian = design.T @ (weights[:, None] * design) + 1e-7 * np.eye(2)
        step = np.linalg.solve(hessian, design.T @ (y - p))
        theta += step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    if abs(float(theta[1])) < 1e-9:
        return None
    midpoint = center + scale * (-float(theta[0]) / float(theta[1]))
    if midpoint < float(np.min(x)) or midpoint > float(np.max(x)):
        return None
    return float(midpoint)


def summaries(rows: list[dict[str, str]], group_key: str | None = None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = row.get(group_key, "all") if group_key else "all"
        n_society = int(as_float(row, "n_society"))
        groups[(label, n_society)].append(row)
    out: list[dict[str, Any]] = []
    for (label, n_society), group in sorted(groups.items()):
        curve = aggregate_curve(group)
        linear = crossing(curve, 0.5)
        logistic = logistic_midpoint(group)
        out.append({
            "group": label,
            "n_society": n_society,
            "n_seeds": len({row.get("seed", "") for row in group}),
            **curve_diagnostics(curve, 0.5),
            "alpha_c_linear": "" if linear is None else linear,
            "alpha_c_logistic": "" if logistic is None else logistic,
            "alpha_c_abs_diff": "" if linear is None or logistic is None else abs(linear - logistic),
            "K_c_effective": "" if linear is None else n_society * linear,
            "K_c_floor": "" if linear is None else math.floor(n_society * linear),
            "K_c_ceil": "" if linear is None else math.ceil(n_society * linear),
        })
    return out


def _resolved_support(group_rows: list[dict[str, Any]], alpha_key: str) -> set[int]:
    return {int(row["n_society"]) for row in group_rows if row.get(alpha_key, "") != ""}


def common_support_nu_table(
    rows: list[dict[str, str]],
    group_key: str,
    alpha_key: str = "alpha_c_linear",
    n_boot: int = 2000,
) -> list[dict[str, Any]]:
    table = summaries(rows, group_key)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table:
        by_group[str(row["group"])].append(row)
    if not by_group:
        return []

    supports = [_resolved_support(group_rows, alpha_key) for group_rows in by_group.values()]
    common_support = sorted(set.intersection(*supports)) if supports else []
    out = []
    rng = np.random.default_rng(SEED)
    for group, group_rows in sorted(by_group.items()):
        support_rows = [row for row in group_rows if int(row["n_society"]) in common_support]
        nu_value = estimate_nu(support_rows, alpha_key)
        k_values = [float(row["K_c_effective"]) for row in support_rows if row.get("K_c_effective", "") != ""]
        alpha_values = [float(row[alpha_key]) for row in support_rows if row.get(alpha_key, "") != ""]
        qualitative_fraction_decreases = int(len(alpha_values) >= 2 and alpha_values[-1] < alpha_values[0])
        qualitative_count_increases = int(len(k_values) >= 2 and k_values[-1] > k_values[0])

        group_raw = [row for row in rows if str(row.get(group_key, "")) == group]
        seeds = sorted({row.get("seed", "") for row in group_raw if row.get("seed", "") != ""})
        boot_values = []
        if len(seeds) >= 2 and len(common_support) >= 2:
            rows_by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in group_raw:
                if int(as_float(row, "n_society")) in common_support:
                    rows_by_seed[row.get("seed", "")].append(row)
            for _ in range(n_boot):
                sample: list[dict[str, str]] = []
                for seed in rng.choice(seeds, size=len(seeds), replace=True):
                    sample.extend(rows_by_seed[str(seed)])
                value = estimate_nu(
                    [row for row in summaries(sample) if int(row["n_society"]) in common_support],
                    alpha_key,
                )
                if math.isfinite(value):
                    boot_values.append(value)
        if boot_values:
            arr = np.asarray(boot_values, dtype=float)
            ci_lo, ci_hi = np.quantile(arr, [0.025, 0.975])
            sublinear_fraction = float(np.mean((arr > 0.0) & (arr < 1.0)))
        else:
            ci_lo = ci_hi = sublinear_fraction = float("nan")

        out.append({
            "group": group,
            "alpha_key": alpha_key,
            "common_support": ";".join(str(n_value) for n_value in common_support),
            "n_common": len(common_support),
            "nu_common": nu_value,
            "nu_ci_lo": ci_lo,
            "nu_ci_hi": ci_hi,
            "bootstrap_n": len(boot_values),
            "bootstrap_fraction_0_lt_nu_lt_1": sublinear_fraction,
            "alpha_c_first_common": alpha_values[0] if alpha_values else "",
            "alpha_c_last_common": alpha_values[-1] if alpha_values else "",
            "K_c_first_common": k_values[0] if k_values else "",
            "K_c_last_common": k_values[-1] if k_values else "",
            "qualitative_alpha_decreases": qualitative_fraction_decreases,
            "qualitative_K_increases": qualitative_count_increases,
        })
    return out


LLM_AUDIT_FIELDS = [
    "population_llm_calls",
    "population_llm_cache_hits",
    "population_llm_failures",
    "population_llm_prompt_tokens",
    "population_llm_completion_tokens",
    "population_llm_total_tokens",
    "population_llm_estimated_cost_usd",
    "defense_llm_calls",
    "defense_llm_cache_hits",
    "defense_llm_failures",
    "defense_llm_prompt_tokens",
    "defense_llm_completion_tokens",
    "defense_llm_total_tokens",
    "defense_llm_estimated_cost_usd",
]


MATCHED_COMPONENT_FIELDS = [
    "scenario",
    "variant",
    "defense",
    "profile",
    "mock_openrouter",
    "population_model",
    "population_model_policy",
    "population_model_version",
    "retail_population_model",
    "controller_mode",
    "social_game_version",
    "placement",
    "asset",
    "liquidity_exponent",
    "requested_harmful_count",
    "n_harmful",
    "n_population_agents_actual",
]


def control_definition_audit(rows: list[dict[str, str]], group_key: str = "quota_variant") -> list[dict[str, Any]]:
    out = []
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get(group_key, ""))].append(row)
    for group, group_rows in sorted(by_group.items()):
        row_out: dict[str, Any] = {
            "group": group,
            "n_rows": len(group_rows),
            "quota_modes": ";".join(sorted({row.get("quota_mode", "") for row in group_rows})),
        }
        for field in LLM_AUDIT_FIELDS:
            values = [as_float(row, field, 0.0) for row in group_rows]
            row_out[f"{field}_max"] = max(values) if values else 0.0
            row_out[f"{field}_sum"] = float(np.sum(values)) if values else 0.0
        row_out["zero_llm_counts"] = int(
            all(row_out[f"{field}_max"] == 0.0 and row_out[f"{field}_sum"] == 0.0 for field in LLM_AUDIT_FIELDS)
        )
        out.append(row_out)

    signatures: dict[tuple[str, int, float, str], dict[str, tuple[Any, ...]]] = defaultdict(dict)
    present_fields = [field for field in MATCHED_COMPONENT_FIELDS if any(field in row for row in rows)]
    for row in rows:
        key = (
            str(row.get("scenario", "")),
            int(as_float(row, "n_society")),
            as_float(row, "alpha"),
            str(row.get("seed", "")),
        )
        group = str(row.get(group_key, ""))
        signatures[key][group] = tuple(row.get(field, "") for field in present_fields)
    mismatched = 0
    compared = 0
    for group_values in signatures.values():
        if len(group_values) < 2:
            continue
        compared += 1
        if len(set(group_values.values())) > 1:
            mismatched += 1
    out.append({
        "group": "matched_non_controller_components",
        "n_rows": len(rows),
        "quota_modes": ";".join(sorted(by_group)),
        "signature_fields": ";".join(present_fields),
        "matched_cells_compared": compared,
        "matched_cells_with_mismatch": mismatched,
        "matched_non_controller_components": int(mismatched == 0),
    })
    return out


def estimate_nu(summary_rows: list[dict[str, Any]], alpha_key: str = "alpha_c_linear") -> float:
    usable = [row for row in summary_rows if row.get(alpha_key, "") != ""]
    if len(usable) < 2:
        return float("nan")
    nvals = np.asarray([float(row["n_society"]) for row in usable], dtype=float)
    avals = np.asarray([float(row[alpha_key]) for row in usable], dtype=float)
    slope, _ = np.polyfit(np.log(nvals), np.log(avals), 1)
    return -float(slope)


def grouped_nu_table(rows: list[dict[str, str]], group_key: str | None = None) -> list[dict[str, Any]]:
    table = summaries(rows, group_key)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table:
        by_group[str(row["group"])].append(row)
    out = []
    for group, group_rows in sorted(by_group.items()):
        out.append({
            "group": group,
            "n_resolved_linear": sum(row["alpha_c_linear"] != "" for row in group_rows),
            "nu_linear": estimate_nu(group_rows, "alpha_c_linear"),
            "n_resolved_logistic": sum(row["alpha_c_logistic"] != "" for row in group_rows),
            "nu_logistic": estimate_nu(group_rows, "alpha_c_logistic"),
        })
    return out


def seed_subset_nu(rows: list[dict[str, str]], subset_size: int = 6) -> dict[str, Any]:
    seeds = sorted({int(as_float(row, "seed")) for row in rows if math.isfinite(as_float(row, "seed"))})
    values = []
    for combo in itertools.combinations(seeds, min(subset_size, len(seeds))):
        allowed = {str(seed) for seed in combo}
        sample = [row for row in rows if row.get("seed", "") in allowed]
        value = estimate_nu(summaries(sample))
        if math.isfinite(value):
            values.append(value)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"subset_size": subset_size, "n_effective": 0}
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return {
        "subset_size": subset_size,
        "n_combinations": math.comb(len(seeds), min(subset_size, len(seeds))),
        "n_effective": len(arr),
        "nu_mean": float(np.mean(arr)),
        "nu_ci_lo": float(lo),
        "nu_ci_hi": float(hi),
        "nu_min": float(np.min(arr)),
        "nu_max": float(np.max(arr)),
    }


def leave_one_n(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    base = summaries(rows)
    ns = sorted({int(row["n_society"]) for row in base})
    out = [{"dropped_n": "none", "nu_linear": estimate_nu(base)}]
    for n_value in ns:
        value = estimate_nu([row for row in base if int(row["n_society"]) != n_value])
        out.append({"dropped_n": n_value, "nu_linear": value})
    return out


def leave_one_seed(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    seeds = sorted({int(as_float(row, "seed")) for row in rows if math.isfinite(as_float(row, "seed"))})
    out = []
    for seed in seeds:
        value = estimate_nu(summaries([row for row in rows if int(as_float(row, "seed")) != seed]))
        out.append({"dropped_seed": seed, "nu_linear": value})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p01-run", default="p01_nonlinear_scaling_paper")
    parser.add_argument("--p09-run", default="p09_depth_scaling_paper")
    parser.add_argument("--p10-run", default="p10_llm_fraction_scaling_paper")
    parser.add_argument("--p11-run", default="p11_watts_null_paper")
    parser.add_argument("--out", default="reviewer_scaling_audit")
    args = parser.parse_args()

    out_dir = ensure_dir(OUTPUTS / args.out)
    p01_rows = read_rows(args.p01_run)
    if p01_rows:
        write_csv(summaries(p01_rows), out_dir / "p01_linear_vs_logistic_alpha_c.csv")
        write_csv(leave_one_n(p01_rows), out_dir / "p01_drop_one_n_nu.csv")
        write_csv(leave_one_seed(p01_rows), out_dir / "p01_leave_one_seed_nu.csv")
        write_json(seed_subset_nu(p01_rows), out_dir / "p01_seed6_jackknife_nu.json")

    for run, group_key, stem in [
        (args.p09_run, "variant", "p09_depth"),
        (args.p10_run, "quota_variant", "p10_llm_quota"),
        (args.p11_run, None, "p11_watts_null"),
    ]:
        rows = read_rows(run)
        if not rows:
            continue
        write_csv(summaries(rows, group_key), out_dir / f"{stem}_alpha_c.csv")
        write_csv(grouped_nu_table(rows, group_key), out_dir / f"{stem}_nu.csv")
        if stem == "p10_llm_quota" and group_key is not None:
            write_csv(
                common_support_nu_table(rows, group_key, "alpha_c_linear"),
                out_dir / "p10_llm_quota_common_support_nu.csv",
            )
            write_csv(
                control_definition_audit(rows, group_key),
                out_dir / "p10_llm_quota_control_audit.csv",
            )

    write_json({
        "p01_rows": len(p01_rows),
        "p09_rows": len(read_rows(args.p09_run)),
        "p10_rows": len(read_rows(args.p10_run)),
        "p11_rows": len(read_rows(args.p11_run)),
    }, out_dir / "manifest.json")
    print(f"Wrote reviewer audit tables to {out_dir}")


if __name__ == "__main__":
    main()