"""Experiment 8: parameter sensitivity audit.

Sweep one parameter family at a time around canonical S1-S4 settings. The
result is a robustness matrix that documents whether collapse statistics are
stable across plausible calibration ranges rather than tuned to one YAML point.

Output: paperoutputs/scaling/exp8_sensitivity_audit/
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import (
    RunSpec,
    env_float_list,
    env_int_list,
    env_list,
    env_seed_list,
    run_grid,
    scaling_exp_dir,
    write_csv,
    write_json,
)
from wolfbench.metrics import binomial_rate_summary


SCENARIOS = env_list("WOLFBENCH_EXP8_SCENARIOS", "s1,s2,s3,s4")
N_SOCIETY = int(os.getenv("WOLFBENCH_EXP8_N_SOCIETY", "1000"))
SEEDS = env_seed_list("WOLFBENCH_EXP8_SEEDS", default_count=50)
DEFAULT_ALPHAS = {
    "s1": 0.015,
    "s2": 0.001,
    "s3": 0.35,
    "s4": 0.03,
}
DEFAULT_ALPHA_GRIDS = {
    "s1": "0.0,0.0075,0.01,0.015,0.02,0.03",
    "s2": "0.0,0.00025,0.0005,0.00075,0.001,0.0015,0.0025",
    "s3": "0.0,0.15,0.3,0.4,0.5",
    "s4": "0.0,0.01,0.015,0.02,0.03,0.05,0.1,0.15,0.2",
}
FAMILY_LABELS = {
    "feedback_strength": "Feedback",
    "asset_liquidity_scale": "Liquidity",
    "retail_wealth_scale": "Retail wealth",
    "retail_risk_appetite": "Risk appetite",
    "social_mean_degree": "Mean degree",
    "placement": "Placement",
}
CATEGORY_ORDER = {
    "placement": {"random": 0, "high_degree": 1},
}


def _row_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in {None, "", "None", "none", "null", "nan"}:
        return default
    return float(value)


def _sort_value(value, family: str | None = None):
    if family in CATEGORY_ORDER:
        order = CATEGORY_ORDER[family]
        if str(value) in order:
            return (0, order[str(value)])
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _alpha_for(scenario: str) -> float:
    return float(os.getenv(f"WOLFBENCH_EXP8_ALPHA_{scenario.upper()}", DEFAULT_ALPHAS[scenario]))


def _alphas_for(scenario: str) -> list[float]:
    return env_float_list(
        f"WOLFBENCH_EXP8_ALPHAS_{scenario.upper()}",
        os.getenv("WOLFBENCH_EXP8_ALPHAS", DEFAULT_ALPHA_GRIDS[scenario]),
    )


def _spec_for(scenario: str, alpha: float, seed: int, family: str, value: float | int | str) -> RunSpec:
    if family == "feedback_strength":
        return RunSpec(scenario, N_SOCIETY, alpha, seed, feedback_strength=float(value), label=f"{family}={value}")
    if family == "asset_liquidity_scale":
        return RunSpec(scenario, N_SOCIETY, alpha, seed, asset_liquidity_scale=float(value), label=f"{family}={value}")
    if family == "retail_wealth_scale":
        return RunSpec(scenario, N_SOCIETY, alpha, seed, retail_wealth_scale=float(value), label=f"{family}={value}")
    if family == "retail_risk_appetite":
        return RunSpec(scenario, N_SOCIETY, alpha, seed, retail_risk_appetite=float(value), label=f"{family}={value}")
    if family == "social_mean_degree":
        return RunSpec(scenario, N_SOCIETY, alpha, seed, social_mean_degree=int(value), label=f"{family}={value}")
    if family == "placement":
        return RunSpec(scenario, N_SOCIETY, alpha, seed, placement=str(value), label=f"{family}={value}")
    raise ValueError(f"Unknown sensitivity family: {family}")


def _parameter_grid() -> dict[str, list[float | int | str]]:
    return {
        "feedback_strength": env_float_list("WOLFBENCH_EXP8_FEEDBACKS", "0.0,0.4,0.8,1.2,1.6"),
        "asset_liquidity_scale": env_float_list("WOLFBENCH_EXP8_LIQUIDITY_SCALES", "0.5,0.75,1.0,1.5,2.0"),
        "retail_wealth_scale": env_float_list("WOLFBENCH_EXP8_WEALTH_SCALES", "0.5,1.0,2.0"),
        "retail_risk_appetite": env_float_list("WOLFBENCH_EXP8_RISK_APPETITES", "0.01,0.02,0.04"),
        "social_mean_degree": env_int_list("WOLFBENCH_EXP8_MEAN_DEGREES", "4,8,12,16"),
        "placement": env_list("WOLFBENCH_EXP8_PLACEMENTS", "random,high_degree"),
    }


def _threshold_alpha(alpha_means: dict[float, float], threshold: float = 0.5) -> float | None:
    points = sorted((float(a), float(p)) for a, p in alpha_means.items())
    if not points:
        return None
    for idx, (alpha, prob) in enumerate(points):
        if prob >= threshold:
            if idx == 0:
                return alpha
            prev_alpha, prev_prob = points[idx - 1]
            if prob == prev_prob:
                return alpha
            frac = (threshold - prev_prob) / (prob - prev_prob)
            return float(prev_alpha + frac * (alpha - prev_alpha))
    return None


def _alpha_c_rows(summary_rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["scenario"], r["family"], str(r["value"])) for r in summary_rows})
    for scenario, family, value in keys:
        rows = [
            r for r in summary_rows
            if r["scenario"] == scenario and r["family"] == family and str(r["value"]) == value
        ]
        alpha_means = {float(r["alpha"]): _row_float(r, "collapse_rate_mean") for r in rows}
        alpha_c = _threshold_alpha(alpha_means, 0.5)
        alpha_25 = _threshold_alpha(alpha_means, 0.25)
        alpha_75 = _threshold_alpha(alpha_means, 0.75)
        transition_width = None
        if alpha_25 is not None and alpha_75 is not None:
            transition_width = float(max(0.0, alpha_75 - alpha_25))
        out.append({
            "scenario": scenario,
            "n_society": N_SOCIETY,
            "family": family,
            "value": value,
            "alpha_c": alpha_c,
            "alpha_25": alpha_25,
            "alpha_75": alpha_75,
            "transition_width": transition_width,
            "alpha_grid": ";".join(str(a) for a in sorted(alpha_means)),
        })
    return out


def _family_delta_rows(alpha_c_rows: list[dict],
                       scenarios: list[str] | None = None,
                       families: list[str] | None = None) -> list[dict]:
    scenarios = scenarios or sorted({str(r["scenario"]) for r in alpha_c_rows})
    families = families or [f for f in FAMILY_LABELS if any(r["family"] == f for r in alpha_c_rows)]
    out = []
    for scenario in scenarios:
        for family in families:
            rows = [
                r for r in alpha_c_rows
                if r["scenario"] == scenario and r["family"] == family and r.get("alpha_c") is not None
            ]
            if not rows:
                continue
            values = [_row_float(r, "alpha_c") for r in rows]
            min_idx = int(np.argmin(values))
            max_idx = int(np.argmax(values))
            widths = [_row_float(r, "transition_width") for r in rows if r.get("transition_width") is not None]
            out.append({
                "scenario": scenario,
                "family": family,
                "min_alpha_c": values[min_idx],
                "max_alpha_c": values[max_idx],
                "delta_alpha_c": float(values[max_idx] - values[min_idx]),
                "mean_transition_width": float(np.mean(widths)) if widths else None,
                "value_at_min": rows[min_idx]["value"],
                "value_at_max": rows[max_idx]["value"],
                "n_values": len(rows),
            })
    return out


def _plot_sensitivity_heatmap(delta_rows: list[dict],
                              scenarios: list[str],
                              families: list[str],
                              out) -> None:
    matrix = np.full((len(scenarios), len(families)), np.nan)
    by_key = {(r["scenario"], r["family"]): r for r in delta_rows}
    for i, scenario in enumerate(scenarios):
        for j, family in enumerate(families):
            row = by_key.get((scenario, family))
            if row is not None:
                matrix[i, j] = _row_float(row, "delta_alpha_c")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(1.0, float(np.nanmax(matrix))))
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels([FAMILY_LABELS.get(f, f) for f in families], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels([s.upper() for s in scenarios])
    ax.set_title("Exp8 sensitivity audit: alpha_c range by parameter family")
    ax.set_xlabel("Parameter family")
    ax.set_ylabel("Scenario")
    for i in range(len(scenarios)):
        for j in range(len(families)):
            value = matrix[i, j]
            if np.isfinite(value):
                color = "white" if value >= 0.45 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("max alpha_c - min alpha_c")
    fig.tight_layout()
    fig.savefig(out / "sensitivity_delta_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_top_sensitivity_curves(summary_rows: list[dict],
                                 delta_rows: list[dict],
                                 out,
                                 top_k: int = 8) -> None:
    ranked = sorted(delta_rows, key=lambda r: _row_float(r, "delta_alpha_c"), reverse=True)
    selected = [r for r in ranked if _row_float(r, "delta_alpha_c") > 0.0][:top_k]
    if not selected:
        return

    n_cols = 2
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.1 * n_rows), squeeze=False)
    for ax, row in zip(axes.ravel(), selected):
        scenario = row["scenario"]
        family = row["family"]
        plot_rows = [
            r for r in summary_rows
            if r["scenario"] == scenario and r["family"] == family
            and r.get("alpha_c") is not None
        ]
        plot_rows.sort(key=lambda r: _sort_value(r["value"], family))
        xs = np.arange(len(plot_rows))
        ys = np.array([_row_float(r, "alpha_c") for r in plot_rows], dtype=float)
        ax.plot(xs, ys, marker="o", lw=1.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(r["value"]) for r in plot_rows], rotation=20, ha="right")
        ax.grid(alpha=0.25)
        ax.set_title(
            f"{scenario.upper()} / {FAMILY_LABELS.get(family, family)} "
            f"(delta alpha_c={_row_float(row, 'delta_alpha_c'):.4f})"
        )
        ax.set_ylabel("alpha_c")
    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")
    fig.suptitle("Exp8 strongest one-at-a-time sensitivities", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "top_sensitivity_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_feedback_sensitivity(summary_rows: list[dict], scenarios: list[str], out) -> None:
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(9, 3.2 * len(scenarios)), sharex=False)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        default_alpha = _alpha_for(scenario)
        plot_rows = [
            r for r in summary_rows
            if r["scenario"] == scenario
            and r["family"] == "feedback_strength"
            and float(r["alpha"]) == default_alpha
        ]
        plot_rows.sort(key=lambda r: _sort_value(r["value"], "feedback_strength"))
        xs = [_row_float(r, "value") for r in plot_rows]
        ys = [_row_float(r, "collapse_rate_mean") for r in plot_rows]
        yerr = [
            [max(0.0, y - _row_float(r, "collapse_rate_ci_low")) for y, r in zip(ys, plot_rows)],
            [max(0.0, _row_float(r, "collapse_rate_ci_high") - y) for y, r in zip(ys, plot_rows)],
        ]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4)
        ax.set_title(f"{scenario.upper()} feedback sensitivity")
        ax.set_ylabel("P(collapse)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("social.feedback_strength")
    fig.tight_layout()
    fig.savefig(out / "feedback_sensitivity.png", dpi=150)
    plt.close(fig)


def write_sensitivity_figures(summary_rows: list[dict], out,
                              scenarios: list[str] | None = None,
                              families: list[str] | None = None) -> list[dict]:
    scenarios = scenarios or [s for s in SCENARIOS if any(r["scenario"] == s for r in summary_rows)]
    families = families or [f for f in FAMILY_LABELS if any(r["family"] == f for r in summary_rows)]
    alpha_rows = _alpha_c_rows(summary_rows)
    write_csv(alpha_rows, out / "sensitivity_alpha_c_summary.csv")
    delta_rows = _family_delta_rows(alpha_rows, scenarios, families)
    write_csv(delta_rows, out / "sensitivity_delta_summary.csv")
    _plot_sensitivity_heatmap(delta_rows, scenarios, families, out)
    _plot_top_sensitivity_curves(alpha_rows, delta_rows, out)
    _plot_feedback_sensitivity(summary_rows, scenarios, out)
    return delta_rows


def main() -> None:
    out = scaling_exp_dir("exp8_sensitivity_audit")
    grid = _parameter_grid()
    specs = []
    for scenario in SCENARIOS:
        for family, values in grid.items():
            for value in values:
                for alpha in _alphas_for(scenario):
                    for seed in SEEDS:
                        specs.append(_spec_for(scenario, alpha, seed, family, value))

    print(f"Running {len(specs)} episodes for exp8...")
    rows = run_grid(specs)
    write_csv(rows, out / "data.csv")

    summary_rows = []
    metrics = ["collapse_rate", "max_collapse_score", "retail_loss_pct_30d", "price_dislocation_max"]
    for scenario in SCENARIOS:
        for family, values in grid.items():
            for value in values:
                for alpha in _alphas_for(scenario):
                    selected = [
                        r for r in rows
                        if r["scenario"] == scenario
                        and r["label"] == f"{family}={value}"
                        and float(r["alpha"]) == float(alpha)
                    ]
                    row = {
                        "scenario": scenario,
                        "alpha": alpha,
                        "is_default_alpha": float(alpha) == _alpha_for(scenario),
                        "n_society": N_SOCIETY,
                        "family": family,
                        "value": value,
                        "n": len(selected),
                    }
                    for metric in metrics:
                        vals = np.array([float(r[metric]) for r in selected], dtype=float)
                        row[f"{metric}_mean"] = float(vals.mean()) if vals.size else 0.0
                        row[f"{metric}_std"] = float(vals.std()) if vals.size else 0.0
                        if metric == "collapse_rate":
                            ci = binomial_rate_summary(vals)
                            row["collapse_rate_ci_low"] = ci["ci_low"]
                            row["collapse_rate_ci_high"] = ci["ci_high"]
                            row["collapse_successes"] = ci["successes"]
                    summary_rows.append(row)
    write_csv(summary_rows, out / "sensitivity_summary.csv")
    delta_rows = write_sensitivity_figures(summary_rows, out, SCENARIOS, list(grid.keys()))

    write_json({
        "scenarios": SCENARIOS,
        "n_society": N_SOCIETY,
        "default_alphas": {scenario: _alpha_for(scenario) for scenario in SCENARIOS},
        "alpha_grids": {scenario: _alphas_for(scenario) for scenario in SCENARIOS},
        "seeds": SEEDS,
        "parameter_grid": grid,
        "summary_csv": "sensitivity_summary.csv",
        "alpha_c_summary_csv": "sensitivity_alpha_c_summary.csv",
        "delta_summary_csv": "sensitivity_delta_summary.csv",
        "figures": {
            "sensitivity_delta_heatmap": "sensitivity_delta_heatmap.png",
            "top_sensitivity_curves": "top_sensitivity_curves.png",
            "feedback_sensitivity": "feedback_sensitivity.png",
        },
        "strongest_sensitivities": sorted(
            delta_rows,
            key=lambda r: _row_float(r, "delta_alpha_c"),
            reverse=True,
        )[:8],
    }, out / "summary.json")
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()