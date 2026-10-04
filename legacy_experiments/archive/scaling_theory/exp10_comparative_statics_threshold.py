"""Exp10: threshold-shift comparative statics.

This is the paper-facing upgrade of Exp8: each mechanism lever is evaluated by
rerunning a local alpha sweep and estimating the change in critical harmful
ratio, ``Delta alpha_c = alpha_c(changed) - alpha_c(base)``.

Outputs: ``paperoutputs/scaling/exp10_comparative_statics_threshold/``
    data.csv
    alpha_curves.csv
    alpha_c_by_variant.csv
    comparative_statics_summary.csv
    report.md
    summary.json
    delta_alpha_c_sign_table.png
    collapse_curves.png
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import (
    RunSpec,
    env_float_list,
    env_list,
    env_seed_list,
    run_grid,
    scaling_exp_dir,
    write_csv,
    write_json,
)
from experiments.scaling_theory._threshold import bootstrap_logistic_ci, fit_logistic_threshold
from wolfbench.metrics import binomial_rate_summary
from wolfbench.scenarios.base import load_scenario


SCENARIOS = env_list("WOLFBENCH_EXP10_SCENARIOS", "s1,s2")
N_SOCIETY = int(os.getenv("WOLFBENCH_EXP10_N_SOCIETY", "1000"))
SEEDS = env_seed_list("WOLFBENCH_EXP10_SEEDS", default_count=30)
CI_BOOT = int(os.getenv("WOLFBENCH_EXP10_CI_BOOT", "1000"))
THRESHOLD = float(os.getenv("WOLFBENCH_EXP10_THRESHOLD", "0.5"))
OUT_NAME = os.getenv("WOLFBENCH_EXP10_OUT", "exp10_comparative_statics_threshold")

DEFAULT_ALPHA_GRIDS = {
    "s1": "0.0,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03",
    "s2": "0.0,0.00025,0.0005,0.00075,0.001,0.0015,0.002,0.0025",
    "s3": "0.0,0.15,0.3,0.4,0.5,0.6,0.7,0.8",
    "s4": "0.0,0.01,0.015,0.02,0.03,0.05,0.1,0.15,0.2",
}

LEVER_LABELS = {
    "asset_liquidity_scale": "Liquidity",
    "social_mean_degree": "Mean degree",
    "retail_risk_appetite": "Risk appetite",
    "placement": "Placement",
    "feedback_strength": "Feedback",
}

EXPECTED_SIGNS = {
    "asset_liquidity_scale": "positive",
    "social_mean_degree": "negative",
    "retail_risk_appetite": "negative",
    "placement": "negative",
    "feedback_strength": "negative",
}


def _alphas_for(scenario: str) -> list[float]:
    return env_float_list(
        f"WOLFBENCH_EXP10_ALPHAS_{scenario.upper()}",
        os.getenv("WOLFBENCH_EXP10_ALPHAS", DEFAULT_ALPHA_GRIDS[scenario]),
    )


def _changed_value(scenario: str, lever: str) -> float | int | str:
    scen = load_scenario(scenario)
    if lever == "asset_liquidity_scale":
        return float(os.getenv("WOLFBENCH_EXP10_LIQUIDITY_CHANGED", "1.5"))
    if lever == "social_mean_degree":
        default_degree = int(scen.social.get("mean_degree", 8))
        return int(os.getenv("WOLFBENCH_EXP10_MEAN_DEGREE_CHANGED", str(max(default_degree + 4, 12))))
    if lever == "retail_risk_appetite":
        return float(os.getenv("WOLFBENCH_EXP10_RISK_APPETITE_CHANGED", "0.04"))
    if lever == "placement":
        return os.getenv("WOLFBENCH_EXP10_PLACEMENT_CHANGED", "high_degree")
    if lever == "feedback_strength":
        default_feedback = float(scen.social.get("feedback_strength", 0.8))
        return float(os.getenv("WOLFBENCH_EXP10_FEEDBACK_CHANGED", str(max(default_feedback + 0.4, 1.2))))
    raise ValueError(f"Unknown lever: {lever}")


def _base_value(scenario: str, lever: str) -> float | int | str:
    scen = load_scenario(scenario)
    if lever == "asset_liquidity_scale":
        return 1.0
    if lever == "social_mean_degree":
        return int(scen.social.get("mean_degree", 8))
    if lever == "retail_risk_appetite":
        return float(scen.retail.get("risk_appetite", 0.02))
    if lever == "placement":
        return "random"
    if lever == "feedback_strength":
        return float(scen.social.get("feedback_strength", 0.8))
    raise ValueError(f"Unknown lever: {lever}")


def _spec_for(scenario: str, alpha: float, seed: int, lever: str,
              variant: str, value: float | int | str) -> RunSpec:
    label = f"{lever}:{variant}:{value}"
    if variant == "base" and lever != "placement":
        return RunSpec(scenario, N_SOCIETY, alpha, seed, label=label)
    if lever == "asset_liquidity_scale":
        return RunSpec(scenario, N_SOCIETY, alpha, seed, asset_liquidity_scale=float(value), label=label)
    if lever == "social_mean_degree":
        return RunSpec(scenario, N_SOCIETY, alpha, seed, social_mean_degree=int(value), label=label)
    if lever == "retail_risk_appetite":
        return RunSpec(scenario, N_SOCIETY, alpha, seed, retail_risk_appetite=float(value), label=label)
    if lever == "placement":
        return RunSpec(scenario, N_SOCIETY, alpha, seed, placement=str(value), label=label)
    if lever == "feedback_strength":
        return RunSpec(scenario, N_SOCIETY, alpha, seed, feedback_strength=float(value), label=label)
    raise ValueError(f"Unknown lever: {lever}")


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _curve_rows(rows: list[dict[str, Any]], alpha_grids: dict[str, list[float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r["scenario"], r["lever"], r["variant"], str(r["lever_value"])) for r in rows})
    for scenario, lever, variant, lever_value in keys:
        for alpha in alpha_grids[scenario]:
            selected = [
                row for row in rows
                if row["scenario"] == scenario
                and row["lever"] == lever
                and row["variant"] == variant
                and str(row["lever_value"]) == lever_value
                and float(row["alpha"]) == float(alpha)
            ]
            if not selected:
                continue
            collapse_values = [float(row["collapse_rate"]) for row in selected]
            ci = binomial_rate_summary(collapse_values)
            out.append({
                "scenario": scenario,
                "n_society": N_SOCIETY,
                "lever": lever,
                "variant": variant,
                "lever_value": lever_value,
                "alpha": alpha,
                "n": len(selected),
                "collapse_rate_mean": ci["mean"],
                "collapse_rate_ci_low": ci["ci_low"],
                "collapse_rate_ci_high": ci["ci_high"],
                "collapse_successes": ci["successes"],
                "max_collapse_score_mean": _mean(selected, "max_collapse_score"),
                "retail_loss_mean": _mean(selected, "retail_loss_pct_30d"),
                "price_dislocation_mean": _mean(selected, "price_dislocation_max"),
                "liquidity_stress_mean": _mean(selected, "liquidity_stress_max"),
                "social_cascade_mean": _mean(selected, "social_cascade_peak"),
            })
    return out


def _alpha_c_rows(rows: list[dict[str, Any]], alpha_grids: dict[str, list[float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r["scenario"], r["lever"], r["variant"], str(r["lever_value"])) for r in rows})
    for scenario, lever, variant, lever_value in keys:
        selected = [
            row for row in rows
            if row["scenario"] == scenario
            and row["lever"] == lever
            and row["variant"] == variant
            and str(row["lever_value"]) == lever_value
        ]
        alphas = alpha_grids[scenario]
        probs = [
            _mean([row for row in selected if float(row["alpha"]) == float(alpha)], "collapse_rate")
            for alpha in alphas
        ]
        fit = fit_logistic_threshold(alphas, probs, threshold=THRESHOLD)
        boot = bootstrap_logistic_ci(
            selected,
            alphas,
            n_boot=CI_BOOT,
            threshold=THRESHOLD,
            rng_seed=31_000 + abs(hash((scenario, lever, variant, lever_value))) % 10_000,
        )
        out.append({
            "scenario": scenario,
            "n_society": N_SOCIETY,
            "lever": lever,
            "variant": variant,
            "lever_value": lever_value,
            "alpha_c": fit["alpha_c"],
            "alpha_c_ci_low": boot["ci_low"],
            "alpha_c_ci_high": boot["ci_high"],
            "transition_width_10_90": fit["transition_width_10_90"],
            "logistic_slope": fit["slope"],
            "fit_method": fit["method"],
            "bootstrap_successes": boot["n_success"],
            "p_collapse": dict(zip(map(str, alphas), probs)),
        })
    return out


def _alpha_estimate_from_rows(rows: list[dict[str, Any]], alphas: list[float]) -> float | None:
    probs = [
        _mean([row for row in rows if float(row["alpha"]) == float(alpha)], "collapse_rate")
        for alpha in alphas
    ]
    estimate = fit_logistic_threshold(alphas, probs, threshold=THRESHOLD)["alpha_c"]
    return float(estimate) if estimate is not None and np.isfinite(float(estimate)) else None


def _bootstrap_delta_ci(base_rows: list[dict[str, Any]], changed_rows: list[dict[str, Any]],
                        alphas: list[float], n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    base_by_alpha = {
        float(alpha): [row for row in base_rows if float(row["alpha"]) == float(alpha)]
        for alpha in alphas
    }
    changed_by_alpha = {
        float(alpha): [row for row in changed_rows if float(row["alpha"]) == float(alpha)]
        for alpha in alphas
    }
    samples: list[float] = []
    for _ in range(n_boot):
        draw_base: list[dict[str, Any]] = []
        draw_changed: list[dict[str, Any]] = []
        valid = True
        for alpha in alphas:
            base_alpha_rows = base_by_alpha[float(alpha)]
            changed_alpha_rows = changed_by_alpha[float(alpha)]
            if not base_alpha_rows or not changed_alpha_rows:
                valid = False
                break
            base_idx = rng.integers(0, len(base_alpha_rows), size=len(base_alpha_rows))
            changed_idx = rng.integers(0, len(changed_alpha_rows), size=len(changed_alpha_rows))
            draw_base.extend(base_alpha_rows[int(i)] for i in base_idx)
            draw_changed.extend(changed_alpha_rows[int(i)] for i in changed_idx)
        if not valid:
            continue
        base_alpha_c = _alpha_estimate_from_rows(draw_base, alphas)
        changed_alpha_c = _alpha_estimate_from_rows(draw_changed, alphas)
        if base_alpha_c is not None and changed_alpha_c is not None:
            samples.append(float(changed_alpha_c - base_alpha_c))
    if not samples:
        return {"delta_alpha_c_ci_low": None, "delta_alpha_c_ci_high": None, "delta_bootstrap_successes": 0}
    arr = np.array(samples, dtype=float)
    return {
        "delta_alpha_c_ci_low": float(np.quantile(arr, 0.025)),
        "delta_alpha_c_ci_high": float(np.quantile(arr, 0.975)),
        "delta_bootstrap_successes": int(arr.size),
    }


def _sign(value: float | None, eps: float = 1e-12) -> str:
    if value is None or not np.isfinite(float(value)):
        return "missing"
    if float(value) > eps:
        return "positive"
    if float(value) < -eps:
        return "negative"
    return "zero"


def _delta_rows(alpha_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]],
                alpha_grids: dict[str, list[float]]) -> list[dict[str, Any]]:
    by_key = {(r["scenario"], r["lever"], r["variant"]): r for r in alpha_rows}
    out: list[dict[str, Any]] = []
    for scenario in sorted({r["scenario"] for r in alpha_rows}):
        for lever in LEVER_LABELS:
            base = by_key.get((scenario, lever, "base"))
            changed = by_key.get((scenario, lever, "changed"))
            if base is None or changed is None:
                continue
            alpha_base = base.get("alpha_c")
            alpha_changed = changed.get("alpha_c")
            delta = None
            if alpha_base is not None and alpha_changed is not None:
                delta = float(alpha_changed - alpha_base)
            base_raw = [
                row for row in raw_rows
                if row["scenario"] == scenario and row["lever"] == lever and row["variant"] == "base"
            ]
            changed_raw = [
                row for row in raw_rows
                if row["scenario"] == scenario and row["lever"] == lever and row["variant"] == "changed"
            ]
            delta_ci = _bootstrap_delta_ci(
                base_raw,
                changed_raw,
                alpha_grids[scenario],
                CI_BOOT,
                seed=41_000 + abs(hash((scenario, lever))) % 10_000,
            )
            expected = EXPECTED_SIGNS[lever]
            observed = _sign(delta)
            ci_low = delta_ci["delta_alpha_c_ci_low"]
            ci_high = delta_ci["delta_alpha_c_ci_high"]
            ci_excludes_zero = (
                ci_low is not None and ci_high is not None
                and (float(ci_low) > 0.0 or float(ci_high) < 0.0)
            )
            out.append({
                "scenario": scenario,
                "n_society": N_SOCIETY,
                "lever": lever,
                "lever_label": LEVER_LABELS[lever],
                "base_value": base["lever_value"],
                "changed_value": changed["lever_value"],
                "alpha_c_base": alpha_base,
                "alpha_c_base_ci_low": base.get("alpha_c_ci_low"),
                "alpha_c_base_ci_high": base.get("alpha_c_ci_high"),
                "alpha_c_changed": alpha_changed,
                "alpha_c_changed_ci_low": changed.get("alpha_c_ci_low"),
                "alpha_c_changed_ci_high": changed.get("alpha_c_ci_high"),
                "delta_alpha_c": delta,
                **delta_ci,
                "expected_sign": expected,
                "observed_sign": observed,
                "sign_pass": observed == expected,
                "ci_excludes_zero": ci_excludes_zero,
                "transition_width_base": base.get("transition_width_10_90"),
                "transition_width_changed": changed.get("transition_width_10_90"),
            })
    return out


def _plot_delta_table(delta_rows: list[dict[str, Any]], out) -> None:
    if not delta_rows:
        return
    scenarios = sorted({row["scenario"] for row in delta_rows})
    levers = [lever for lever in LEVER_LABELS if any(row["lever"] == lever for row in delta_rows)]
    matrix = np.full((len(scenarios), len(levers)), np.nan)
    for i, scenario in enumerate(scenarios):
        for j, lever in enumerate(levers):
            row = next((r for r in delta_rows if r["scenario"] == scenario and r["lever"] == lever), None)
            if row is not None and row["delta_alpha_c"] is not None:
                matrix[i, j] = float(row["delta_alpha_c"])
    finite = np.isfinite(matrix)
    vmax = max(float(np.nanmax(np.abs(matrix))) if finite.any() else 0.01, 1e-6)
    fig, ax = plt.subplots(figsize=(10.5, 3.2 + 0.6 * len(scenarios)))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(levers)))
    ax.set_xticklabels([LEVER_LABELS[lever] for lever in levers], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels([scenario.upper() for scenario in scenarios])
    ax.set_title("Exp10 threshold-shift comparative statics")
    for i in range(len(scenarios)):
        for j in range(len(levers)):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.4f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta alpha_c")
    fig.tight_layout()
    fig.savefig(out / "delta_alpha_c_sign_table.png", dpi=180)
    plt.close(fig)


def _plot_collapse_curves(curve_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]], out) -> None:
    selected = [row for row in delta_rows if row["scenario"] in {"s1", "s2"}]
    if not selected:
        selected = delta_rows[: min(len(delta_rows), 6)]
    if not selected:
        return
    n_cols = 2
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.2 * n_rows), squeeze=False)
    for ax, row in zip(axes.ravel(), selected):
        for variant, color in [("base", "C0"), ("changed", "C3")]:
            points = [
                r for r in curve_rows
                if r["scenario"] == row["scenario"]
                and r["lever"] == row["lever"]
                and r["variant"] == variant
            ]
            points.sort(key=lambda r: float(r["alpha"]))
            ax.plot(
                [float(r["alpha"]) for r in points],
                [float(r["collapse_rate_mean"]) for r in points],
                marker="o",
                color=color,
                label=variant,
            )
        ax.axhline(THRESHOLD, color="k", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.set_title(f"{row['scenario'].upper()} {LEVER_LABELS[row['lever']]}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("P(collapse)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()
    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "collapse_curves.png", dpi=180)
    plt.close(fig)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_report(delta_rows: list[dict[str, Any]], path) -> None:
    sign_passes = sum(1 for row in delta_rows if row["sign_pass"])
    lines = [
        "# Exp10 Comparative Statics Threshold Shift",
        "",
        f"Scenarios: {', '.join(SCENARIOS)}",
        f"N: {N_SOCIETY}",
        f"Seeds: {SEEDS}",
        f"Threshold: P(collapse)={THRESHOLD}",
        f"Sign agreement: {sign_passes}/{len(delta_rows)} rows",
        "",
        "| scenario | lever | base | changed | alpha_c base | alpha_c changed | Delta alpha_c | 95% CI | expected | observed | pass |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in delta_rows:
        ci = f"[{_fmt(row['delta_alpha_c_ci_low'])}, {_fmt(row['delta_alpha_c_ci_high'])}]"
        lines.append(
            f"| {row['scenario']} | {row['lever_label']} | {row['base_value']} | {row['changed_value']} | "
            f"{_fmt(row['alpha_c_base'])} | {_fmt(row['alpha_c_changed'])} | "
            f"{_fmt(row['delta_alpha_c'])} | {ci} | {row['expected_sign']} | "
            f"{row['observed_sign']} | {row['sign_pass']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    out = scaling_exp_dir(OUT_NAME)
    alpha_grids = {scenario: _alphas_for(scenario) for scenario in SCENARIOS}
    levers = env_list("WOLFBENCH_EXP10_LEVERS", ",".join(LEVER_LABELS))

    all_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for lever in levers:
            variants = [("base", _base_value(scenario, lever)), ("changed", _changed_value(scenario, lever))]
            for variant, value in variants:
                specs = [
                    _spec_for(scenario, alpha, seed, lever, variant, value)
                    for alpha in alpha_grids[scenario]
                    for seed in SEEDS
                ]
                print(f"\n=== Exp10 {scenario} / {lever} / {variant}={value} ===")
                rows = run_grid(specs, progress_every=25)
                for row in rows:
                    row["lever"] = lever
                    row["variant"] = variant
                    row["lever_value"] = value
                    row["expected_sign"] = EXPECTED_SIGNS[lever]
                all_rows.extend(rows)

    curve_rows = _curve_rows(all_rows, alpha_grids)
    alpha_rows = _alpha_c_rows(all_rows, alpha_grids)
    delta_rows = _delta_rows(alpha_rows, all_rows, alpha_grids)

    write_csv(all_rows, out / "data.csv")
    write_csv(curve_rows, out / "alpha_curves.csv")
    write_csv(alpha_rows, out / "alpha_c_by_variant.csv")
    write_csv(delta_rows, out / "comparative_statics_summary.csv")
    _write_report(delta_rows, out / "report.md")
    _plot_delta_table(delta_rows, out)
    _plot_collapse_curves(curve_rows, delta_rows, out)
    write_json({
        "scenarios": SCENARIOS,
        "n_society": N_SOCIETY,
        "seeds": SEEDS,
        "alpha_grids": alpha_grids,
        "threshold": THRESHOLD,
        "levers": levers,
        "expected_signs": EXPECTED_SIGNS,
        "sign_pass_count": sum(1 for row in delta_rows if row["sign_pass"]),
        "summary": delta_rows,
        "alpha_c_by_variant": alpha_rows,
    }, out / "summary.json")
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()
