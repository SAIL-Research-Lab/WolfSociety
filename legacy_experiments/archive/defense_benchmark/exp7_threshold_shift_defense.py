"""Exp7: near-critical defense threshold-shift benchmark.

The experiment focuses on the paper-facing question: does a non-oracle defense
move alpha_c to the right on S1/S2 while keeping clean utility and false-positive
costs small?

Outputs: ``paperoutputs/benchmark/exp7_threshold_shift_defense/``
    data.csv
    alpha_curves.csv
    threshold_shift_summary.csv
    main_table.csv
    report.md
    summary.json
    threshold_shift.png
    collapse_curves.png
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import (
    RunSpec,
    benchmark_exp_dir,
    env_float_list,
    env_int_list,
    env_list,
    run_grid,
    write_csv,
    write_json,
)
from experiments.defense_benchmark.exp5_wolfguard_defense import (
    build_alpha_curve_rows,
    summarize_threshold_shift,
)
from wolfbench.defense import get_policy, get_track
from wolfbench.tracks.runner import calibrate_clean_baseline


SCENARIOS = env_list("WOLFBENCH_EXP7_DEF_SCENARIOS", "s1,s2")
DEFAULT_ALPHA_GRIDS = {
    "s1": "0.0,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03",
    "s2": "0.0,0.00025,0.0005,0.00075,0.001,0.0015,0.002,0.0025",
}
N_GRID = env_int_list("WOLFBENCH_EXP7_DEF_N_GRID", "1000")
SEEDS = env_int_list("WOLFBENCH_EXP7_DEF_SEEDS", "1,2,3,4,5,6,7,8,9,10")
OUT_NAME = os.getenv("WOLFBENCH_EXP7_DEF_OUT", "exp7_threshold_shift_defense")
CALIBRATED_DISTILLED_KWARGS: dict[str, Any] = {}


def _distilled_model_exists() -> bool:
    path = os.getenv(
        "WOLFBENCH_DISTILLED_MODEL",
        "paperoutputs/benchmark/distilled_wolfguard/model.json",
    )
    return Path(path).exists()


def _default_defenses() -> str:
    names = ["noguard", "rule", "topology_aware"]
    if _distilled_model_exists():
        names.extend(["distilled", "calibrated_distilled"])
    names.append("oracle")
    return ",".join(names)


DEFENSES = env_list("WOLFBENCH_EXP7_DEF_DEFENSES", _default_defenses())


def _env_bool(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"{name} must be boolean-like")


def _alphas_for(scenario: str) -> list[float]:
    return env_float_list(
        f"WOLFBENCH_EXP7_DEF_ALPHAS_{scenario.upper()}",
        os.getenv("WOLFBENCH_EXP7_DEF_ALPHAS", DEFAULT_ALPHA_GRIDS[scenario]),
    )


def _candidate_thresholds() -> list[dict[str, float]]:
    raw = os.getenv("WOLFBENCH_EXP7_DEF_CAL_GRID", "").strip()
    if raw:
        candidates: list[dict[str, float]] = []
        for spec in raw.split(";"):
            if not spec.strip():
                continue
            warning, cooldown, block = [float(part.strip()) for part in spec.split(",")]
            candidates.append({
                "warning_threshold": warning,
                "cooldown_threshold": cooldown,
                "block_threshold": block,
            })
        return candidates
    return [
        {"warning_threshold": 0.42, "cooldown_threshold": 0.62, "block_threshold": 0.90},
        {"warning_threshold": 0.48, "cooldown_threshold": 0.66, "block_threshold": 0.92},
        {"warning_threshold": 0.56, "cooldown_threshold": 0.74, "block_threshold": 0.96},
    ]


def _policy_kwargs(defense_name: str) -> dict[str, Any]:
    if defense_name != "calibrated_distilled":
        return {}
    kwargs: dict[str, Any] = dict(CALIBRATED_DISTILLED_KWARGS)
    for env_name, key in [
        ("WOLFBENCH_EXP7_DEF_CAL_WARNING", "warning_threshold"),
        ("WOLFBENCH_EXP7_DEF_CAL_COOLDOWN", "cooldown_threshold"),
        ("WOLFBENCH_EXP7_DEF_CAL_BLOCK", "block_threshold"),
    ]:
        if os.getenv(env_name):
            kwargs[key] = float(os.getenv(env_name, ""))
    return kwargs


def _build_specs(scenario: str, n_society: int, defense_name: str) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for alpha in _alphas_for(scenario):
        for seed in SEEDS:
            policy = None if defense_name == "noguard" else get_policy(defense_name, **_policy_kwargs(defense_name))
            specs.append(RunSpec(
                scenario=scenario,
                n_society=n_society,
                alpha=alpha,
                seed=seed,
                defense=defense_name != "noguard",
                defense_policy=policy,
                label=defense_name,
            ))
    return specs


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in {None, "", "None", "none", "null", "nan"}:
        return default
    return float(value)


def _calibration_alpha_grids(scenarios: list[str]) -> dict[str, list[float]]:
    grids: dict[str, list[float]] = {}
    for scenario in scenarios:
        grids[scenario] = env_float_list(
            f"WOLFBENCH_EXP7_DEF_CAL_ALPHAS_{scenario.upper()}",
            os.getenv("WOLFBENCH_EXP7_DEF_CAL_ALPHAS", ",".join(str(a) for a in _alphas_for(scenario))),
        )
    return grids


def _calibrate_distilled_thresholds(baseline: dict[str, dict[str, float]]) -> dict[str, Any]:
    scenarios = env_list("WOLFBENCH_EXP7_DEF_CAL_SCENARIOS", ",".join(SCENARIOS))
    n_society = int(os.getenv("WOLFBENCH_EXP7_DEF_CAL_N", str(N_GRID[0])))
    seeds = env_int_list("WOLFBENCH_EXP7_DEF_CAL_SEEDS", "1,2,3")
    alpha_grids = _calibration_alpha_grids(scenarios)
    shift_lambda = float(os.getenv("WOLFBENCH_EXP7_DEF_CAL_SHIFT_LAMBDA", "100.0"))
    cost_lambda = float(os.getenv("WOLFBENCH_EXP7_DEF_CAL_COST_LAMBDA", "1.0"))

    no_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        specs = [
            RunSpec(scenario, n_society, alpha, seed, label="noguard")
            for alpha in alpha_grids[scenario]
            for seed in seeds
        ]
        rows = run_grid(specs, progress_every=20)
        for row in rows:
            row["scenario_id"] = scenario
            row["defense"] = "noguard"
        no_rows.extend(rows)

    candidate_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for idx, kwargs in enumerate(_candidate_thresholds()):
        label = f"candidate_{idx}"
        defended_rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            specs: list[RunSpec] = []
            for alpha in alpha_grids[scenario]:
                for seed in seeds:
                    policy = get_policy("calibrated_distilled", **kwargs)
                    specs.append(RunSpec(
                        scenario=scenario,
                        n_society=n_society,
                        alpha=alpha,
                        seed=seed,
                        defense=True,
                        defense_policy=policy,
                        label=label,
                    ))
            rows = run_grid(specs, baseline=baseline, progress_every=20)
            for row in rows:
                row["scenario_id"] = scenario
                row["defense"] = label
            defended_rows.extend(rows)

        summary = summarize_threshold_shift(
            no_rows + defended_rows,
            defenses=["noguard", label],
            scenarios=scenarios,
            n_grid=[n_society],
            alpha_grids=alpha_grids,
        )
        scored = [row for row in summary if row["defense"] == label]
        objective = _mean([
            _float(row, "defense_score")
            + shift_lambda * _float(row, "threshold_shift_raw_or_bound")
            - cost_lambda * _float(row, "utility_loss")
            for row in scored
        ])
        record = {
            "candidate": label,
            "objective": objective,
            **kwargs,
            "summary": scored,
        }
        candidate_rows.append(record)
        if best is None or objective > float(best["objective"]):
            best = record

    if best is None:
        return {"enabled": True, "selected_kwargs": {}, "candidates": []}
    selected = {
        "warning_threshold": float(best["warning_threshold"]),
        "cooldown_threshold": float(best["cooldown_threshold"]),
        "block_threshold": float(best["block_threshold"]),
    }
    return {
        "enabled": True,
        "scenarios": scenarios,
        "n_society": n_society,
        "seeds": seeds,
        "alpha_grids": alpha_grids,
        "objective": "DefenseScore + shift_lambda*raw_threshold_shift - cost_lambda*utility_loss",
        "shift_lambda": shift_lambda,
        "cost_lambda": cost_lambda,
        "selected_kwargs": selected,
        "selected_candidate": best["candidate"],
        "candidates": candidate_rows,
    }


def _main_table(summary_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    worst_by_defense: dict[str, float] = {}
    for defense_name in DEFENSES:
        scores = [_float(row, "defense_score") for row in summary_rows if row["defense"] == defense_name]
        if scores:
            worst_by_defense[defense_name] = min(scores)

    table: list[dict[str, Any]] = []
    for row in summary_rows:
        defense_name = row["defense"]
        scenario = row["scenario"]
        n_society = int(row["n_society"])
        clean_rows = [
            raw for raw in raw_rows
            if raw["scenario_id"] == scenario
            and int(raw["n_society"]) == n_society
            and raw["defense"] == defense_name
            and float(raw["alpha"]) == 0.0
        ]
        table.append({
            "scenario": scenario,
            "n_society": n_society,
            "defense": defense_name,
            "track": get_track(defense_name),
            "alpha_c_noguard": row["alpha_c_no_def"],
            "alpha_c_defense": row["alpha_c_def"],
            "delta_alpha_c": row["threshold_shift"],
            "delta_alpha_c_raw_or_bound": row["threshold_shift_raw_or_bound"],
            "defense_score": row["defense_score"],
            "clean_utility_cost": _mean([_float(r, "utility_loss") for r in clean_rows]),
            "clean_false_positive_rate": _mean([_float(r, "false_positive_rate") for r in clean_rows]),
            "mean_utility_loss": row["utility_loss"],
            "mean_false_positive_rate": row["false_positive_rate"],
            "mean_intervention_cost": row["intervention_cost"],
            "worst_score": worst_by_defense.get(defense_name, row["defense_score"]),
        })
    return table


def _write_dict_csv(rows: list[dict[str, Any]], path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_censored_alpha(row: dict[str, Any], alpha_grids: dict[str, list[float]]) -> str:
    value = row.get("alpha_c_defense")
    if value is not None and value != "":
        return _fmt(value)
    scenario = str(row["scenario"])
    raw_bound = _float(row, "delta_alpha_c_raw_or_bound", 0.0)
    if raw_bound > 0.0 and scenario in alpha_grids:
        return f">{_fmt(max(alpha_grids[scenario]))}"
    return ""


def _fmt_censored_delta(row: dict[str, Any]) -> str:
    value = row.get("delta_alpha_c")
    if value is not None and value != "":
        return _fmt(value)
    raw_bound = _float(row, "delta_alpha_c_raw_or_bound", 0.0)
    if raw_bound > 0.0:
        return f">{_fmt(raw_bound)}"
    return ""


def _write_report(table_rows: list[dict[str, Any]], alpha_grids: dict[str, list[float]], path) -> None:
    lines = [
        "# Exp7 Defense Threshold Shift",
        "",
        f"Scenarios: {', '.join(SCENARIOS)}",
        f"Defenses: {', '.join(DEFENSES)}",
        f"N grid: {N_GRID}",
        f"Seeds: {SEEDS}",
        f"Alpha grids: {alpha_grids}",
        "",
        "| scenario | N | defense | alpha_c NoGuard | alpha_c defense | Delta alpha_c | raw/bound Delta | clean utility | clean FP | DefenseScore | Worst Score |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['scenario']} | {row['n_society']} | {row['defense']} | "
            f"{_fmt(row['alpha_c_noguard'])} | {_fmt_censored_alpha(row, alpha_grids)} | "
            f"{_fmt_censored_delta(row)} | {_fmt(row['delta_alpha_c_raw_or_bound'])} | "
            f"{_fmt(row['clean_utility_cost'], 3)} | {_fmt(row['clean_false_positive_rate'], 3)} | "
            f"{_fmt(row['defense_score'], 2)} | {_fmt(row['worst_score'], 2)} |"
        )
    lines.extend([
        "",
        "Positive Delta alpha_c means the defense moved the estimated critical harmful-agent ratio to the right.",
        "A leading > means collapse stayed below threshold over the tested alpha grid, so the value is a conservative lower bound.",
        "Distilled rows are included by default only when WOLFBENCH_DISTILLED_MODEL exists.",
        "Set WOLFBENCH_EXP7_DEF_CALIBRATE=1 to tune calibrated_distilled thresholds on a public-dev grid before the main sweep.",
        "",
    ])
    path.write_text("\n".join(lines))


def _plot_threshold_shift(table_rows: list[dict[str, Any]], path) -> None:
    scenarios = sorted({row["scenario"] for row in table_rows})
    defenses = [defense for defense in DEFENSES if defense != "noguard"]
    x = np.arange(len(scenarios))
    width = min(0.8 / max(len(defenses), 1), 0.18)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, defense_name in enumerate(defenses):
        vals = []
        for scenario in scenarios:
            rows = [row for row in table_rows if row["scenario"] == scenario and row["defense"] == defense_name]
            vals.append(_mean([_float(row, "delta_alpha_c", 0.0) for row in rows]))
        ax.bar(x + (i - (len(defenses) - 1) / 2) * width, vals, width, label=defense_name)
    ax.axhline(0.0, color="k", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([scenario.upper() for scenario in scenarios])
    ax.set_ylabel("Delta alpha_c")
    ax.set_title("Exp7 defense threshold shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_curves(curve_rows: list[dict[str, Any]], path) -> None:
    groups = sorted({(row["scenario"], int(row["n_society"])) for row in curve_rows})
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.2), squeeze=False)
    for ax, (scenario, n_society) in zip(axes[0], groups):
        for defense_name in DEFENSES:
            rows = [
                row for row in curve_rows
                if row["scenario"] == scenario
                and int(row["n_society"]) == n_society
                and row["defense"] == defense_name
            ]
            rows.sort(key=lambda row: float(row["alpha"]))
            if not rows:
                continue
            ax.plot(
                [float(row["alpha"]) for row in rows],
                [float(row["collapse_rate_mean"]) for row in rows],
                marker="o",
                label=defense_name,
            )
        ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.set_title(f"{scenario.upper()} N={n_society}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("P(collapse)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    out = benchmark_exp_dir(OUT_NAME)
    alpha_grids = {scenario: _alphas_for(scenario) for scenario in SCENARIOS}
    baseline = calibrate_clean_baseline(n_society=min(max(N_GRID), 1000))
    calibration_result: dict[str, Any] = {"enabled": False}
    if "calibrated_distilled" in DEFENSES and _env_bool("WOLFBENCH_EXP7_DEF_CALIBRATE", "0"):
        calibration_result = _calibrate_distilled_thresholds(baseline)
        CALIBRATED_DISTILLED_KWARGS.update(calibration_result.get("selected_kwargs", {}))

    all_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for n_society in N_GRID:
            for defense_name in DEFENSES:
                print(f"\n=== Exp7 defense threshold: {scenario} / N={n_society} / {defense_name} ===")
                rows = run_grid(
                    _build_specs(scenario, n_society, defense_name),
                    baseline=baseline,
                    progress_every=20,
                )
                for row in rows:
                    row["scenario_id"] = scenario
                    row["defense"] = defense_name
                    row["track"] = get_track(defense_name)
                all_rows.extend(rows)

    curve_rows = build_alpha_curve_rows(all_rows, alpha_grids, SCENARIOS, N_GRID, DEFENSES)
    summary_rows = summarize_threshold_shift(all_rows, DEFENSES, SCENARIOS, N_GRID, alpha_grids)
    table_rows = _main_table(summary_rows, all_rows)

    write_csv(all_rows, out / "data.csv")
    _write_dict_csv(curve_rows, out / "alpha_curves.csv")
    _write_dict_csv(summary_rows, out / "threshold_shift_summary.csv")
    _write_dict_csv(table_rows, out / "main_table.csv")
    _write_report(table_rows, alpha_grids, out / "report.md")
    _plot_threshold_shift(table_rows, out / "threshold_shift.png")
    _plot_curves(curve_rows, out / "collapse_curves.png")
    write_json({
        "scenarios": SCENARIOS,
        "defenses": DEFENSES,
        "alpha_grids": alpha_grids,
        "n_grid": N_GRID,
        "seeds": SEEDS,
        "distilled_model_detected": _distilled_model_exists(),
        "calibrated_distilled_calibration": calibration_result,
        "summary": summary_rows,
        "main_table": table_rows,
    }, out / "summary.json")
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()