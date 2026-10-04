"""Exp6 — WolfBench Defense Leaderboard.

Question: when the same harmful population is held fixed, how do different
defense policies compare on Threshold Protection Score (TPS)?

Outputs:
* ``paperoutputs/benchmark/exp6/data.csv``       — per-(defense, scenario, alpha, seed) row
* ``paperoutputs/benchmark/exp6/leaderboard_by_scenario.csv`` — per-scenario/N aggregate
* ``paperoutputs/benchmark/exp6/leaderboard.csv`` — display leaderboard
* ``paperoutputs/benchmark/exp6/leaderboard_overall.csv`` — display leaderboard alias
* ``paperoutputs/benchmark/exp6/summary.json``
* ``paperoutputs/benchmark/exp6/leaderboard.png`` — TPS by defense / scenario
* ``paperoutputs/benchmark/exp6/threshold_shift.png``
* ``paperoutputs/benchmark/exp6/threshold_table.csv``
* ``paperoutputs/benchmark/exp6/collapse_curves.png``
"""
from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from experiments._common import (
    RunSpec, benchmark_exp_dir, run_grid, write_csv, write_json,
)
from wolfbench.defense import get_policy, get_track
from wolfbench.metrics import (
    bootstrap_ci,
    defense_score,
    rank_stability,
    threshold_protection_score,
)
from wolfbench.tracks.runner import calibrate_clean_baseline


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


def _env_float_list(name: str, default: str) -> list[float]:
    return [float(x) for x in _env_list(name, default)]


def _env_int_list(name: str, default: str) -> list[int]:
    return [int(x) for x in _env_list(name, default)]


def _dedupe(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def _competitive_defenses(defenses: list[str], upper_bounds: list[str] | None = None) -> list[str]:
    upper = set(upper_bounds or [])
    return [
        defense_name for defense_name in defenses
        if defense_name not in upper
        and get_track(defense_name) not in {"control", "oracle_upper_bound", "legacy_assisted_rule"}
    ]


def _env_bool(name: str, default: str = "") -> bool | None:
    value = os.getenv(name, default).strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value")


def _llm_policy_kwargs() -> dict:
    return {
        "provider": os.getenv("WOLFBENCH_EXP6_LLM_PROVIDER") or os.getenv("WOLFBENCH_LLM_PROVIDER"),
        "model": os.getenv("WOLFBENCH_EXP6_LLM_MODEL") or os.getenv("WOLFBENCH_LLM_MODEL"),
        "base_url": os.getenv("WOLFBENCH_EXP6_LLM_BASE_URL") or os.getenv("WOLFBENCH_LLM_BASE_URL"),
        "api_key": os.getenv("WOLFBENCH_EXP6_LLM_API_KEY") or os.getenv("WOLFBENCH_LLM_API_KEY"),
        "strict": _env_bool("WOLFBENCH_EXP6_LLM_STRICT"),
    }


REQUESTED_DEFENSES = _dedupe(_env_list("WOLFBENCH_EXP6_DEFENSES", "noguard,random,rule,topology_aware,distilled"))
UPPER_BOUNDS = _dedupe(_env_list("WOLFBENCH_EXP6_UPPER_BOUNDS", "oracle"))
DEFENSES = _dedupe(REQUESTED_DEFENSES + UPPER_BOUNDS)
ELIGIBLE_DEFENSES = _competitive_defenses(REQUESTED_DEFENSES, UPPER_BOUNDS)
CONTROL_DEFENSES = [d for d in DEFENSES if get_track(d) == "control"]
UPPER_BOUND_DEFENSES = [d for d in DEFENSES if d in UPPER_BOUNDS or get_track(d) == "oracle_upper_bound"]
SCENARIOS = _env_list("WOLFBENCH_EXP6_SCENARIOS", "s1,s2,s3,s4")
DEFAULT_ALPHA_GRIDS = {
    "s1": "0.0,0.01,0.015,0.02,0.03",
    "s2": "0.0,0.00025,0.0005,0.00075,0.001",
    "s3": "0.0,0.03,0.05,0.075,0.1",
    "s4": "0.0,0.015,0.02,0.03,0.05",
}
GLOBAL_ALPHAS = os.getenv("WOLFBENCH_EXP6_ALPHAS")
N_GRID = _env_int_list("WOLFBENCH_EXP6_N_GRID", os.getenv("WOLFBENCH_EXP6_N_SOCIETY", "500,1000,2000"))
SEEDS = _env_int_list(
    "WOLFBENCH_EXP6_SEEDS",
    ",".join(str(i) for i in range(1, 31)),
)
CI_BOOT = int(os.getenv("WOLFBENCH_EXP6_CI_BOOT", "2000"))
OUT_NAME = os.getenv("WOLFBENCH_EXP6_OUT", "exp6")
INPUT_DATA = _env_list("WOLFBENCH_EXP6_INPUT_DATA", "")
DISPLAY_SCENARIOS = ("s1", "s2", "s3", "s4")
DISPLAY_FIELDNAMES = [
    "Rank", "Defense", "Track", "TPS", "DeltaAlphaC/W0",
    "CriticalDeltaP", "CleanCost", "FP", "Valid", "FallbackRate",
    "BackendFailures", "RecoveredFailures", "ScenarioCoverage", "Status",
]


def _alphas_for(scenario: str) -> list[float]:
    key = f"WOLFBENCH_EXP6_ALPHAS_{scenario.upper()}"
    default = GLOBAL_ALPHAS or DEFAULT_ALPHA_GRIDS.get(scenario, "0.0,0.005,0.01,0.02,0.05,0.1")
    return _env_float_list(key, default)


def _build_specs(scenario: str, defense_name: str, n_society: int) -> list[RunSpec]:
    specs = []
    for a in _alphas_for(scenario):
        for s in SEEDS:
            policy = get_policy(defense_name, **_llm_policy_kwargs())
            specs.append(RunSpec(
                scenario=scenario, n_society=n_society, alpha=a, seed=s,
                defense=defense_name != "noguard",
                defense_policy=None if defense_name == "noguard" else policy,
                label=defense_name,
            ))
    return specs


def _official_tps(defense_name: str, tps: float, validation_status: str = "valid") -> float:
    if get_track(defense_name) == "control":
        return 0.0
    if validation_status != "valid":
        return 0.0
    return float(max(tps, 0.0))


def _status(defense_name: str, validation_status: str = "valid") -> str:
    if validation_status != "valid":
        return validation_status.replace("_", " ")
    track = get_track(defense_name)
    if defense_name in UPPER_BOUND_DEFENSES or track == "oracle_upper_bound":
        return "upper bound"
    if defense_name == "noguard":
        return "reference"
    if track == "control":
        return "ineligible"
    if track == "rule_baseline":
        return "weak"
    return "eligible"


def _to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.strip() in {"", "None", "none", "null", "nan"}:
        return default
    return float(value)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _sum_int(rows: list[dict], key: str) -> int:
    return int(sum(_to_float(row.get(key), default=0.0) for row in rows))


def _validation_metrics(rows: list[dict]) -> dict:
    calls = _sum_int(rows, "llm_backend_calls")
    failures = _sum_int(rows, "llm_backend_failures")
    recovered_failures = _sum_int(rows, "llm_backend_recovered_failures")
    expected = _sum_int(rows, "llm_expected_asset_decisions")
    semantic_fallbacks = _sum_int(rows, "llm_semantic_fallbacks")
    fallback_days = _sum_int(rows, "llm_fallback_days")
    fallback_rate = float(semantic_fallbacks / max(expected, 1))
    backend_failure_rate = float(failures / max(calls, 1))
    if failures > 0:
        status = "invalid_provider_error"
    elif semantic_fallbacks > 0:
        status = "invalid_semantic_fallback"
    else:
        status = "valid"
    return {
        "llm_backend_calls": calls,
        "llm_backend_failures": failures,
        "llm_backend_recovered_failures": recovered_failures,
        "llm_backend_failure_rate": backend_failure_rate,
        "llm_expected_asset_decisions": expected,
        "llm_semantic_fallbacks": semantic_fallbacks,
        "llm_semantic_fallback_rate": fallback_rate,
        "llm_fallback_days": fallback_days,
        "validation_status": status,
        "valid": status == "valid",
    }


def _overall_validation_status(rows: list[dict]) -> str:
    statuses = {str(row.get("validation_status", "valid")) for row in rows}
    if "invalid_provider_error" in statuses:
        return "invalid_provider_error"
    if "invalid_semantic_fallback" in statuses:
        return "invalid_semantic_fallback"
    return "valid"


def _defenses_in_order(rows: list[dict], defenses: list[str] | None = None) -> list[str]:
    row_defenses = {str(r.get("defense", r.get("Defense model", ""))) for r in rows}
    ordered: list[str] = []
    if defenses is not None:
        for defense_name in defenses:
            if defense_name in row_defenses and defense_name not in ordered:
                ordered.append(defense_name)
        return ordered
    for row in rows:
        defense_name = str(row.get("defense", row.get("Defense model", "")))
        if defense_name and defense_name not in ordered:
            ordered.append(defense_name)
    return ordered


def _build_display_leaderboard(
    scenario_leaderboard: list[dict],
    defenses: list[str] | None = None,
    scenarios: tuple[str, ...] = DISPLAY_SCENARIOS,
    ranked: bool = True,
) -> list[dict]:
    """Build the public S1-S4 TPS leaderboard from per-scenario rows."""
    by_defense_scenario: dict[tuple[str, str], list[dict]] = {}
    for row in scenario_leaderboard:
        defense_name = str(row.get("defense", row.get("Defense model", "")))
        scenario = str(row.get("scenario", "")).lower()
        if defense_name and scenario in scenarios:
            by_defense_scenario.setdefault((defense_name, scenario), []).append(row)

    display_rows: list[dict] = []
    for defense_name in _defenses_in_order(scenario_leaderboard, defenses):
        tps_values: list[float] = []
        shift_values: list[float] = []
        delta_p_values: list[float] = []
        clean_cost_values: list[float] = []
        fp_values: list[float] = []
        fallback_rates: list[float] = []
        backend_failures = 0
        recovered_failures = 0
        scenario_coverage = 0
        for scenario in scenarios:
            rows = by_defense_scenario.get((defense_name, scenario), [])
            if rows:
                tps_values.append(_mean([_to_float(r.get("tps_official")) for r in rows]))
                shift_values.append(_mean([_to_float(r.get("delta_alpha_c_over_w0")) for r in rows]))
                delta_p_values.append(_mean([_to_float(r.get("critical_band_delta_p")) for r in rows]))
                clean_cost_values.append(_mean([_to_float(r.get("clean_cost_index")) for r in rows]))
                fp_values.append(_mean([_to_float(r.get("clean_false_positive_rate")) for r in rows]))
                fallback_rates.append(_mean([_to_float(r.get("llm_semantic_fallback_rate")) for r in rows]))
                backend_failures += _sum_int(rows, "llm_backend_failures")
                recovered_failures += _sum_int(rows, "llm_backend_recovered_failures")
                if _mean([_to_float(r.get("tps_official")) for r in rows]) > 0.0:
                    scenario_coverage += 1

        if not tps_values:
            continue
        selected_rows = [
            row for row in scenario_leaderboard
            if str(row.get("defense", row.get("Defense model", ""))) == defense_name
            and str(row.get("scenario", "")).lower() in scenarios
        ]
        validation_status = _overall_validation_status(selected_rows)
        display_row = {
            "Rank": "",
            "Defense": defense_name,
            "Track": get_track(defense_name),
            "TPS": 0.0 if validation_status != "valid" else _mean(tps_values),
            "DeltaAlphaC/W0": _mean(shift_values),
            "CriticalDeltaP": _mean(delta_p_values),
            "CleanCost": _mean(clean_cost_values),
            "FP": _mean(fp_values),
            "Valid": validation_status == "valid",
            "FallbackRate": _mean(fallback_rates),
            "BackendFailures": backend_failures,
            "RecoveredFailures": recovered_failures,
            "ScenarioCoverage": scenario_coverage,
            "Status": _status(defense_name, validation_status=validation_status),
        }
        display_rows.append(display_row)

    display_rows.sort(key=lambda r: _to_float(r["TPS"]), reverse=True)
    for idx, row in enumerate(display_rows, 1):
        row["Rank"] = idx if ranked else "-"
    return display_rows


def _write_display_csv(rows: list[dict], path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DISPLAY_FIELDNAMES)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_markdown_display_table(handle, rows: list[dict], empty_message: str) -> None:
    if not rows:
        handle.write(f"{empty_message}\n")
        return
    handle.write("| Rank | Defense | Track | TPS ↑ | Δαc/W0 ↑ | Critical ΔP ↑ | CleanCost ↓ | FP ↓ | Valid | FallbackRate ↓ | Status |\n")
    handle.write("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        handle.write(
            f"| {r['Rank']} | {r['Defense']} | {r['Track']} | {_format_score(r['TPS'])} | "
            f"{_format_shift(r['DeltaAlphaC/W0'])} | {_format_shift(r['CriticalDeltaP'])} | "
            f"{_format_shift(r['CleanCost'])} | {_format_shift(r['FP'])} | {r['Valid']} | "
            f"{_format_shift(r['FallbackRate'])} | {r['Status']} |\n"
        )


def _format_score(value) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.2f}"


def _format_shift(value) -> str:
    return f"{_to_float(value):.4f}"


def _scenario_metric_means(
    scenario_leaderboard: list[dict],
    defense_name: str,
    metric: str,
    scenarios: tuple[str, ...] = DISPLAY_SCENARIOS,
    none_as_zero: bool = False,
) -> list[float]:
    vals: list[float] = []
    for scenario in scenarios:
        rows = [
            r for r in scenario_leaderboard
            if str(r.get("defense")) == defense_name and str(r.get("scenario", "")).lower() == scenario
        ]
        default = 0.0 if none_as_zero else float("nan")
        metric_vals = [_to_float(r.get(metric), default=default) for r in rows]
        if metric_vals:
            vals.append(_mean(metric_vals))
        else:
            vals.append(0.0)
    return vals


def _seed_level_rank_rows(all_rows: list[dict]) -> list[dict]:
    """Per-seed TPS rows used to estimate leaderboard rank stability."""
    rank_rows: list[dict] = []
    seeds = sorted({int(r["seed"]) for r in all_rows})
    scenarios = sorted({str(r["scenario_id"]) for r in all_rows})
    n_values = sorted({int(r["n_society"]) for r in all_rows})
    for seed in seeds:
        for defense_name in DEFENSES:
            scores = []
            official_scores = []
            raw_nets = []
            for scenario in scenarios:
                alphas = _alphas_for(scenario)
                for n_society in n_values:
                    rows_no = [
                        r for r in all_rows
                        if int(r["seed"]) == seed
                        and str(r["scenario_id"]) == scenario
                        and int(r["n_society"]) == n_society
                        and r["defense"] == "noguard"
                    ]
                    rows_def = [
                        r for r in all_rows
                        if int(r["seed"]) == seed
                        and str(r["scenario_id"]) == scenario
                        and int(r["n_society"]) == n_society
                        and r["defense"] == defense_name
                    ]
                    if not rows_no or not rows_def:
                        continue
                    score = threshold_protection_score(rows_no, rows_def, alphas=alphas)
                    tps = _to_float(score.get("tps"))
                    validation = _validation_metrics(rows_def)
                    scores.append(tps)
                    official_scores.append(_official_tps(defense_name, tps, validation["validation_status"]))
                    raw_nets.append(_to_float(score.get("raw_net")))
            if scores:
                rank_rows.append({
                    "seed": seed,
                    "defense": defense_name,
                    "tps": float(np.mean(scores)),
                    "raw_net": float(np.mean(raw_nets)),
                    "tps_official": float(np.mean(official_scores)),
                    "official_score": float(np.mean(official_scores)),
                })
    return rank_rows


def _write_threshold_table(rows: list[dict], path) -> list[dict]:
    fields = ["Defense", "S1 alpha_c", "S1 delta_alpha_c", "S2 alpha_c", "S2 delta_alpha_c"]
    out_rows: list[dict] = []
    for defense_name in _defenses_in_order(rows, DEFENSES):
        out = {"Defense": defense_name}
        for scenario in ["s1", "s2"]:
            selected = [r for r in rows if r["defense"] == defense_name and r["scenario"] == scenario]
            alpha_values = [_to_float(r.get("alpha_c_def"), default=float("nan")) for r in selected]
            delta_values = [_to_float(r.get("delta_alpha_c"), default=0.0) for r in selected]
            alpha_values = [value for value in alpha_values if np.isfinite(value)]
            out[f"{scenario.upper()} alpha_c"] = _mean(alpha_values) if alpha_values else ""
            out[f"{scenario.upper()} delta_alpha_c"] = _mean(delta_values) if delta_values else 0.0
        out_rows.append(out)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def _plot_collapse_curves(all_rows: list[dict], path) -> None:
    scenarios = [scenario for scenario in ["s1", "s2"] if scenario in set(SCENARIOS)]
    if not scenarios:
        return
    n_focus = 1000 if 1000 in N_GRID else N_GRID[0]
    priority = [
        "noguard", "rule", "mistral_risk", "llama_risk", "qwen_risk",
        "deepseek_risk", "topology_aware", "oracle",
    ]
    defenses = [d for d in priority if d in DEFENSES]
    defenses.extend([d for d in DEFENSES if d not in defenses])
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 4.4), squeeze=False)
    for axis, scenario in zip(axes[0], scenarios):
        for defense_name in defenses:
            points = []
            for alpha in _alphas_for(scenario):
                selected = [
                    row for row in all_rows
                    if row["scenario_id"] == scenario
                    and int(row["n_society"]) == int(n_focus)
                    and row["defense"] == defense_name
                    and float(row["alpha"]) == float(alpha)
                ]
                if selected:
                    points.append((alpha, _mean([_to_float(row.get("primary_failure_rate")) for row in selected])))
            if points:
                axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=1.7, label=defense_name)
        axis.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.55)
        axis.set_xscale("symlog", linthresh=1e-4)
        axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel("alpha")
        axis.set_ylabel("P(primary failure)")
        axis.set_title(f"{scenario.upper()} primary-failure curves, N={n_focus}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


_STRING_INPUT_FIELDS = {
    "scenario",
    "placement",
    "llm_provider",
    "llm_model",
    "llm_backend",
    "defense",
    "defense_mode",
    "label",
    "primary_metric",
    "llm_backend_last_error_type",
    "llm_backend_last_recovered_error_type",
    "track",
    "scenario_id",
}


def _coerce_input_row(row: dict[str, str]) -> dict:
    out: dict = {}
    for key, value in row.items():
        if key in _STRING_INPUT_FIELDS or value in ("", None):
            out[key] = value
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            out[key] = value
            continue
        if np.isfinite(numeric) and numeric.is_integer():
            out[key] = int(numeric)
        else:
            out[key] = numeric
    return out


def _read_input_rows(paths: list[str]) -> list[dict]:
    keyed_rows: dict[tuple[str, str, str, str, str], dict] = {}
    for raw_path in paths:
        with open(raw_path, newline="") as handle:
            for row in csv.DictReader(handle):
                row = _coerce_input_row(row)
                scenario = str(row.get("scenario_id") or row.get("scenario") or "")
                key = (
                    scenario,
                    str(row.get("n_society", "")),
                    str(float(row.get("alpha", 0.0))),
                    str(row.get("seed", "")),
                    str(row.get("defense", "")),
                )
                keyed_rows[key] = row
    return list(keyed_rows.values())


def main():
    out = benchmark_exp_dir(OUT_NAME)
    alpha_grids = {s: _alphas_for(s) for s in SCENARIOS}
    if INPUT_DATA:
        print(f"Reading precomputed Exp6 data from {', '.join(INPUT_DATA)}")
        all_rows = _read_input_rows(INPUT_DATA)
    else:
        baseline = calibrate_clean_baseline(n_society=min(max(N_GRID), 1000))

        all_rows: list[dict] = []
        for scen in SCENARIOS:
            for n_society in N_GRID:
                for d in DEFENSES:
                    print(f"\n=== {scen} / N={n_society} / {d} ===")
                    specs = _build_specs(scen, d, n_society)
                    rows = run_grid(specs, baseline=baseline, progress_every=20)
                    for r in rows:
                        r["defense"] = d
                        r["track"] = get_track(d)
                        r["scenario_id"] = scen
                    all_rows.extend(rows)

    write_csv(all_rows, out / "data.csv")

    # Aggregate per-scenario leaderboard rows. These preserve the raw metric
    # components used by summary.json and downstream analysis.
    scenario_leaderboard = []
    for scen in SCENARIOS:
        alphas = _alphas_for(scen)
        for n_society in N_GRID:
            rows_no = [r for r in all_rows if r["scenario_id"] == scen
                       and int(r["n_society"]) == n_society and r["defense"] == "noguard"]
            for d in DEFENSES:
                rows_d = [r for r in all_rows if r["scenario_id"] == scen
                          and int(r["n_society"]) == n_society and r["defense"] == d]
                legacy_score = defense_score(rows_no, rows_d, alphas=alphas)
                tps_score = threshold_protection_score(rows_no, rows_d, alphas=alphas)
                tps = _to_float(tps_score.get("tps"))
                validation = _validation_metrics(rows_d)
                legacy_prefixed = {f"legacy_{key}": value for key, value in legacy_score.items()}
                scenario_leaderboard.append({
                    "scenario": scen,
                    "n_society": n_society,
                    "defense": d,
                    "track": get_track(d),
                    "eligible": d in ELIGIBLE_DEFENSES and validation["validation_status"] == "valid",
                    "status": _status(d, validation_status=validation["validation_status"]),
                    "tps_official": _official_tps(d, tps, validation["validation_status"]),
                    "official_score": _official_tps(d, tps, validation["validation_status"]),
                    "legacy_defense_score": legacy_score["defense_score"],
                    "defense_score": legacy_score["defense_score"],
                    "threshold_shift": tps_score.get("delta_alpha_c"),
                    **validation,
                    **legacy_prefixed,
                    **tps_score,
                })

    with open(out / "leaderboard_by_scenario.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scenario_leaderboard[0].keys()))
        w.writeheader()
        for r in scenario_leaderboard:
            w.writerow(r)

    display_leaderboard = _build_display_leaderboard(scenario_leaderboard, ELIGIBLE_DEFENSES, ranked=True)
    control_leaderboard = _build_display_leaderboard(scenario_leaderboard, CONTROL_DEFENSES, ranked=False)
    upper_bound_leaderboard = _build_display_leaderboard(scenario_leaderboard, UPPER_BOUND_DEFENSES, ranked=False)
    _write_display_csv(display_leaderboard, out / "leaderboard.csv")
    _write_display_csv(display_leaderboard, out / "leaderboard_overall.csv")
    _write_display_csv(control_leaderboard, out / "leaderboard_controls.csv")
    _write_display_csv(upper_bound_leaderboard, out / "leaderboard_upper_bounds.csv")

    overall_metrics = []
    seed_rank_rows = _seed_level_rank_rows(all_rows)
    competitive_seed_rank_rows = [r for r in seed_rank_rows if r["defense"] in ELIGIBLE_DEFENSES]
    rank_stability_summary = (
        rank_stability(
            competitive_seed_rank_rows,
            score_key="tps_official",
            item_key="defense",
            sample_key="seed",
            n_boot=CI_BOOT,
            top_k=min(3, len(ELIGIBLE_DEFENSES)),
            seed=17,
        )
        if competitive_seed_rank_rows and ELIGIBLE_DEFENSES else {}
    )
    for d in DEFENSES:
        rows = [r for r in scenario_leaderboard if r["defense"] == d]
        scores = [r["tps"] for r in rows]
        official_scores = [r["tps_official"] for r in rows]
        raw_nets = [r["raw_net"] for r in rows]
        legacy_scores = [r["legacy_defense_score"] for r in rows]
        ci_low, ci_high = bootstrap_ci(scores, n_boot=CI_BOOT)
        official_ci_low, official_ci_high = bootstrap_ci(official_scores, n_boot=CI_BOOT)
        overall_metrics.append({
            "defense": d,
            "track": get_track(d),
            "eligible": d in ELIGIBLE_DEFENSES,
            "status": _status(d),
            "tps_official_mean": float(np.mean(official_scores)),
            "tps_official_ci_low": official_ci_low,
            "tps_official_ci_high": official_ci_high,
            "tps_mean": float(np.mean(scores)),
            "tps_std": float(np.std(scores)),
            "tps_ci_low": ci_low,
            "tps_ci_high": ci_high,
            "raw_net_mean": float(np.mean(raw_nets)),
            "legacy_defense_score_mean": float(np.mean(legacy_scores)),
            "delta_alpha_c_over_w0_mean": float(np.mean([r["delta_alpha_c_over_w0"] for r in rows])),
            "critical_band_delta_p_mean": float(np.mean([r["critical_band_delta_p"] for r in rows])),
            "clean_cost_index_mean": float(np.mean([r["clean_cost_index"] for r in rows])),
            "false_positive_rate_mean": float(np.mean([r["clean_false_positive_rate"] for r in rows])),
            "llm_semantic_fallback_rate_mean": float(np.mean([r["llm_semantic_fallback_rate"] for r in rows])),
            "llm_backend_failures": _sum_int(rows, "llm_backend_failures"),
            "llm_backend_recovered_failures": _sum_int(rows, "llm_backend_recovered_failures"),
            "validation_status": _overall_validation_status(rows),
        })

    threshold_table = _write_threshold_table(scenario_leaderboard, out / "threshold_table.csv")

    with open(out / "leaderboard.md", "w") as f:
        f.write("# WolfBench Exp6 Defense Leaderboard\n\n")
        f.write(f"N={N_GRID}, alpha_grids={alpha_grids}, seeds={SEEDS}\n\n")
        f.write("TPS is the official nonnegative leaderboard score. It uses the primary-failure curve, rewards threshold protection in the NoGuard near-critical band, and applies a clean-market cost gate.\n")
        f.write("LLM rows with provider failures or semantic fallback are marked invalid and receive official TPS=0.\n")
        f.write("RawNet and legacy DefenseScore remain diagnostic fields in CSV/JSON outputs. Competitive rank excludes controls and oracle upper bounds.\n\n")
        f.write("## Eligible Submissions\n\n")
        _write_markdown_display_table(f, display_leaderboard, "No eligible submissions were requested.\n")
        if control_leaderboard:
            f.write("\n## Controls (Not Ranked)\n\n")
            _write_markdown_display_table(f, control_leaderboard, "No controls were requested.\n")
        if upper_bound_leaderboard:
            f.write("\n## Upper Bounds (Not Ranked)\n\n")
            _write_markdown_display_table(f, upper_bound_leaderboard, "No upper bounds were requested.\n")
        f.write("\n## Threshold Table\n\n")
        f.write("| Defense | S1 alpha_c | S1 delta alpha_c | S2 alpha_c | S2 delta alpha_c |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in threshold_table:
            f.write(
                f"| {row['Defense']} | {_format_shift(row['S1 alpha_c'])} | {_format_shift(row['S1 delta_alpha_c'])} | "
                f"{_format_shift(row['S2 alpha_c'])} | {_format_shift(row['S2 delta_alpha_c'])} |\n"
            )

    write_json({
        "requested_defenses": REQUESTED_DEFENSES,
        "eligible_defenses": ELIGIBLE_DEFENSES,
        "control_defenses": CONTROL_DEFENSES,
        "upper_bounds": UPPER_BOUNDS,
        "defenses": DEFENSES, "scenarios": SCENARIOS,
        "alpha_grids": alpha_grids,
        "n_grid": N_GRID, "seeds": SEEDS,
        "leaderboard": scenario_leaderboard,
        "overall": overall_metrics,
        "display_leaderboard": display_leaderboard,
        "display_controls": control_leaderboard,
        "display_upper_bounds": upper_bound_leaderboard,
        "threshold_table": threshold_table,
        "seed_rank_stability": rank_stability_summary,
        "seed_level_rank_rows": seed_rank_rows,
        "competitive_seed_level_rank_rows": competitive_seed_rank_rows,
    }, out / "summary.json")

    # ---- Plot TPS bars ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered_defenses = [r["Defense"] for r in display_leaderboard]
    width = min(0.8 / max(len(ordered_defenses), 1), 0.18)
    xlabels = [s.upper() for s in DISPLAY_SCENARIOS]
    x = np.arange(len(xlabels))
    for i, d in enumerate(ordered_defenses):
        vals = _scenario_metric_means(scenario_leaderboard, d, "tps_official")
        ax.bar(x + (i - (len(ordered_defenses) - 1) / 2) * width, vals, width, label=d)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("TPS")
    ax.set_ylim(bottom=0.0)
    ax.set_title("WolfBench Defense Leaderboard — TPS by scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "leaderboard.png", dpi=150)
    plt.close(fig)

    # ---- Plot Threshold shift ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, d in enumerate(ordered_defenses):
        vals = _scenario_metric_means(scenario_leaderboard, d, "delta_alpha_c_over_w0", none_as_zero=True)
        ax.bar(x + (i - (len(ordered_defenses) - 1) / 2) * width, vals, width, label=d)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Delta alpha_c / W0")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("WolfBench — Critical-α shift relative to NoGuard")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "threshold_shift.png", dpi=150)
    plt.close(fig)

    # ---- Overall leaderboard ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r["Defense"] for r in display_leaderboard]
    vals = [r["TPS"] for r in display_leaderboard]
    ax.bar(names, vals)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Avg TPS across S1-S4")
    ax.set_title("WolfBench Exp6 — Overall defense leaderboard")
    fig.tight_layout()
    fig.savefig(out / "leaderboard_overall.png", dpi=150)
    plt.close(fig)

    _plot_collapse_curves(all_rows, out / "collapse_curves.png")

    print("\nLeaderboard written to", out)
    if control_leaderboard:
        print("Controls not ranked:", ", ".join(r["Defense"] for r in control_leaderboard))
    if upper_bound_leaderboard:
        print("Upper bounds not ranked:", ", ".join(r["Defense"] for r in upper_bound_leaderboard))
    for r in display_leaderboard:
        print(
            f"  {r['Defense']:<18} TPS={r['TPS']:>7.2f}  "
            f"dalpha/W0={r['DeltaAlphaC/W0']:>7.3f}  critical_dP={r['CriticalDeltaP']:>7.3f}"
        )


if __name__ == "__main__":
    main()
