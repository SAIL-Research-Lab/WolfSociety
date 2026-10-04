"""Build the E5 WolfBench leaderboard/audit figure from raw primary metrics.

This is the paper-facing leaderboard builder. It intentionally uses one metric
family only: raw scenario-primary failure curves. Clean-calibrated variants are
kept for scaling sensitivity checks, not for the defense leaderboard.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wolfbench-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from wolfbench.defense import get_track
from wolfbench.metrics import TPSConfig, threshold_protection_score

from .io_utils import OUTPUTS, ROOT, ensure_dir


SCENARIOS = ("s1", "s2", "s3", "s4")
SCENARIO_LABELS = {
    "s1": "S1 Pump",
    "s2": "S2 Finfluencer",
    "s3": "S3 Spoofing",
    "s4": "S4 Wash",
}
DEFENSE_LABELS = {
    "noguard": "NoGuard",
    "random": "Random",
    "rule": "Rule",
    "zscore_guard": "Z-Score",
    "topology_aware": "Topology",
    "oracle": "Oracle",
    "deepseek_v3_risk": "DeepSeek V3",
    "deepseek_v4_risk": "DeepSeek V4 Pro",
    "qwen235b_risk": "Qwen3 235B",
    "qwen36_35b_risk": "Qwen3.6 35B",
    "glm45_risk": "GLM-4.5",
    "glm52_risk": "GLM-5.2",
    "llama33_70b_risk": "Llama 3.3 70B",
    "llama4_maverick_risk": "Llama 4 Maverick",
    "gpt41_risk": "GPT-4.1",
    "gemini25_pro_risk": "Gemini 2.5 Pro",
    "claude_opus_risk": "Claude Opus",
}
PAPER_TPS_CONFIG = TPSConfig(
    shift_weight=0.15,
    critical_collapse_weight=0.55,
    damage_weight=0.30,
    clean_cost_budget=0.18,
    false_positive_budget=0.25,
    intervention_cost_budget=0.30,
)
OVERALL_MEAN_WEIGHT = 0.70
OVERALL_WORST_WEIGHT = 0.30
MIN_OVERALL_COVERAGE = 0.80
MIN_SCENARIO_COVERAGE = 0.75
MAX_SEMANTIC_FALLBACK_RATE = 0.05


def scenario_key(value: Any) -> str:
    text = str(value).lower()
    if text.startswith("s1") or "pump" in text:
        return "s1"
    if text.startswith("s2") or "finfluencer" in text:
        return "s2"
    if text.startswith("s3") or "spoof" in text:
        return "s3"
    if text.startswith("s4") or "wash" in text:
        return "s4"
    return text


def read_data(out_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = OUTPUTS / out_name / "data.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    full = pd.read_csv(path)
    full["scenario_key"] = full["scenario"].map(scenario_key)
    if "status" not in full.columns:
        full["status"] = "ok"
    ok = full[full["status"].fillna("ok").eq("ok")].copy()
    for frame in (full, ok):
        for column in frame.columns:
            if column in {"scenario", "scenario_key", "variant", "defense", "status", "error", "error_type"}:
                continue
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().sum() > 0:
                frame[column] = converted
    return full, ok


def coverage_table(full: pd.DataFrame) -> pd.DataFrame:
    expected = full.groupby(["defense", "scenario_key"], dropna=False).size().rename("expected")
    ok = (
        full[full["status"].fillna("ok").eq("ok")]
        .groupby(["defense", "scenario_key"], dropna=False)
        .size()
        .rename("ok")
    )
    out = pd.concat([expected, ok], axis=1).fillna(0).reset_index()
    out["expected"] = out["expected"].astype(int)
    out["ok"] = out["ok"].astype(int)
    out["coverage"] = out["ok"] / out["expected"].clip(lower=1)
    return out


def error_summary(full: pd.DataFrame) -> pd.DataFrame:
    errors = full[~full["status"].fillna("ok").eq("ok")].copy()
    if errors.empty:
        return pd.DataFrame(columns=["defense", "n_error"])
    errors["error_category"] = errors["error"].map(_error_category)
    summary = pd.crosstab(errors["defense"], errors["error_category"])
    summary["n_error"] = summary.sum(axis=1)
    return summary.sort_values("n_error", ascending=False).reset_index()


def _error_category(value: Any) -> str:
    text = str(value)
    low = text.lower()
    if "403" in low or "not available in your region" in low:
        return "region_403"
    if "invalid risk json" in low:
        return "invalid_risk_json"
    if "valid json but not a json object" in low:
        return "json_not_object"
    if "did not contain a json object" in low:
        return "no_json_object"
    if "unterminated json object" in low:
        return "unterminated_json"
    if "ssl" in low or "remote end closed" in low or "remotedisconnected" in low:
        return "network_disconnect"
    return "other"


def _alphas_for(ok: pd.DataFrame, scenario: str) -> list[float]:
    return sorted(float(x) for x in ok.loc[ok["scenario_key"].eq(scenario), "alpha"].dropna().unique())


def _rows(ok: pd.DataFrame, defense: str, scenario: str) -> list[dict[str, Any]]:
    return ok[ok["defense"].eq(defense) & ok["scenario_key"].eq(scenario)].to_dict("records")


def _mean_column(rows: list[dict[str, Any]], column: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(column, 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def _fallback_rate(rows: list[dict[str, Any]]) -> float:
    expected = sum(float(row.get("defense_llm_expected_asset_decisions", 0.0) or 0.0) for row in rows)
    fallback = sum(float(row.get("defense_llm_semantic_fallbacks", 0.0) or 0.0) for row in rows)
    return float(fallback / expected) if expected > 0 else 0.0


def build_leaderboard(
    out_name: str,
    config: TPSConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tps_config = config or PAPER_TPS_CONFIG
    full, ok = read_data(out_name)
    coverage = coverage_table(full)
    errors = error_summary(full)
    defenses = list(dict.fromkeys(str(x) for x in full["defense"].dropna()))
    scenario_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []

    for defense in defenses:
        defense_ok = ok[ok["defense"].eq(defense)].to_dict("records")
        scores: list[float] = []
        for scenario in SCENARIOS:
            rows_no = _rows(ok, "noguard", scenario)
            rows_def = _rows(ok, defense, scenario)
            alphas = [
                alpha for alpha in _alphas_for(ok, scenario)
                if any(abs(float(row.get("alpha", 0.0)) - alpha) <= 1e-12 for row in rows_no)
                and any(abs(float(row.get("alpha", 0.0)) - alpha) <= 1e-12 for row in rows_def)
            ]
            if defense == "noguard":
                score = _reference_score()
            elif len(alphas) >= 3 and rows_no and rows_def:
                score = threshold_protection_score(rows_no, rows_def, alphas=alphas, config=tps_config)
            else:
                score = _reference_score()
            cov_row = coverage[coverage["defense"].eq(defense) & coverage["scenario_key"].eq(scenario)]
            scenario_coverage = float(cov_row["coverage"].iloc[0]) if not cov_row.empty else 0.0
            scenario_ok = int(cov_row["ok"].iloc[0]) if not cov_row.empty else 0
            scenario_expected = int(cov_row["expected"].iloc[0]) if not cov_row.empty else 0
            tps = float(score.get("tps") or 0.0)
            scores.append(tps)
            scenario_rows.append({
                "defense": defense,
                "label": DEFENSE_LABELS.get(defense, defense),
                "track": get_track(defense),
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "tps": tps,
                "raw_net": float(score.get("raw_net") or 0.0),
                "delta_alpha_c_over_w0": float(score.get("delta_alpha_c_over_w0") or 0.0),
                "critical_delta_p": float(score.get("critical_band_delta_p") or 0.0),
                "damage_reduction": float(score.get("damage_reduction") or 0.0),
                "clean_cost_index": float(score.get("clean_cost_index") or 0.0),
                "false_positive_rate": float(score.get("clean_false_positive_rate") or 0.0),
                "cost_gate": float(score.get("cost_gate") or 0.0),
                "alpha_points": len(alphas),
                "coverage": scenario_coverage,
                "ok": scenario_ok,
                "expected": scenario_expected,
            })

        cov_rows = coverage[coverage["defense"].eq(defense)]
        overall_expected = int(cov_rows["expected"].sum())
        overall_ok = int(cov_rows["ok"].sum())
        overall_coverage = float(overall_ok / max(overall_expected, 1))
        min_scenario_cov = float(cov_rows["coverage"].min()) if not cov_rows.empty else 0.0
        fallback_rate = _fallback_rate(defense_ok)
        track = get_track(defense)
        mean_tps = float(np.mean(scores)) if scores else 0.0
        worst_tps = float(np.min(scores)) if scores else 0.0
        balanced_tps = OVERALL_MEAN_WEIGHT * mean_tps + OVERALL_WORST_WEIGHT * worst_tps
        valid = (
            overall_coverage >= MIN_OVERALL_COVERAGE
            and min_scenario_cov >= MIN_SCENARIO_COVERAGE
            and fallback_rate <= MAX_SEMANTIC_FALLBACK_RATE
        )
        if defense == "noguard":
            status = "reference"
            official_tps = 0.0
            rankable = False
        elif track == "oracle_upper_bound":
            status = "upper bound"
            official_tps = balanced_tps if valid else 0.0
            rankable = False
        elif not valid:
            status = _invalid_status(overall_ok, overall_expected, fallback_rate)
            official_tps = 0.0
            rankable = False
        else:
            status = "ranked"
            official_tps = balanced_tps
            rankable = True
        overall_rows.append({
            "rank": "",
            "defense": defense,
            "label": DEFENSE_LABELS.get(defense, defense),
            "track": track,
            "status": status,
            "rankable": rankable,
            "valid": valid,
            "TPS": official_tps,
            "BalancedTPS": balanced_tps,
            "MeanTPS": mean_tps,
            "WorstScenarioTPS": worst_tps,
            "S1": scores[0] if len(scores) > 0 else 0.0,
            "S2": scores[1] if len(scores) > 1 else 0.0,
            "S3": scores[2] if len(scores) > 2 else 0.0,
            "S4": scores[3] if len(scores) > 3 else 0.0,
            "coverage": overall_coverage,
            "min_scenario_coverage": min_scenario_cov,
            "ok": overall_ok,
            "expected": overall_expected,
            "semantic_fallback_rate": fallback_rate,
            "clean_cost_index": _mean_column(defense_ok, "utility_loss") / 15.0,
            "false_positive_rate": _mean_column(defense_ok, "false_positive_rate"),
            "defense_cost_usd": _mean_column(defense_ok, "defense_llm_episode_estimated_cost_usd"),
        })

    overall = pd.DataFrame(overall_rows)
    ranked = overall[overall["rankable"]].sort_values(["TPS", "WorstScenarioTPS"], ascending=False)
    for rank, index in enumerate(ranked.index, 1):
        overall.loc[index, "rank"] = str(rank)
    status_order = {
        "ranked": 0,
        "reference": 1,
        "upper bound": 2,
    }
    overall["_status_sort"] = overall["status"].map(lambda value: status_order.get(str(value), 3))
    overall["_rank_sort"] = overall["rank"].map(lambda value: int(value) if str(value).isdigit() else 999)
    overall = (
        overall.sort_values(["_status_sort", "_rank_sort", "TPS", "coverage"], ascending=[True, True, False, False])
        .drop(columns=["_status_sort", "_rank_sort"])
        .reset_index(drop=True)
    )
    scenario = pd.DataFrame(scenario_rows)
    return overall, scenario, coverage, errors


def _reference_score() -> dict[str, float]:
    return {
        "tps": 0.0,
        "raw_net": 0.0,
        "delta_alpha_c_over_w0": 0.0,
        "critical_band_delta_p": 0.0,
        "damage_reduction": 0.0,
        "clean_cost_index": 0.0,
        "clean_false_positive_rate": 0.0,
        "cost_gate": 1.0,
    }


def _invalid_status(ok: int, expected: int, fallback_rate: float) -> str:
    if ok <= 0:
        return "invalid: no successful rows"
    if fallback_rate > MAX_SEMANTIC_FALLBACK_RATE:
        return "invalid: semantic fallback"
    return "ineligible: insufficient coverage"


def write_outputs(out_name: str) -> None:
    overall, scenario, coverage, errors = build_leaderboard(out_name)
    out_dir = ensure_dir(OUTPUTS / out_name)
    overall.to_csv(out_dir / "leaderboard_tps.csv", index=False)
    scenario.to_csv(out_dir / "leaderboard_tps_by_scenario.csv", index=False)
    coverage.to_csv(out_dir / "leaderboard_coverage.csv", index=False)
    errors.to_csv(out_dir / "leaderboard_errors.csv", index=False)
    plot_leaderboard(out_name, overall, scenario)
    print(f"Wrote {out_dir / 'leaderboard_tps.csv'}")


def plot_leaderboard(out_name: str, overall: pd.DataFrame, scenario: pd.DataFrame) -> None:
    ink = "#281335"
    grid = "#ead9ef"
    cmap = sns.blend_palette(["#fbfafc", "#eadcf8", "#c084fc", "#6d28d9"], as_cmap=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.labelcolor": ink,
        "xtick.color": ink,
        "ytick.color": ink,
        "text.color": ink,
        "axes.edgecolor": grid,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    eligible_defs = set(
        overall[
            overall["valid"].fillna(False)
            & (
                overall["rankable"].fillna(False)
                | overall["defense"].isin(["noguard", "oracle"])
            )
        ]["defense"]
    )
    available = set(scenario["defense"]) & eligible_defs
    preferred = [
        "noguard",
        "rule",
        "zscore_guard",
        "topology_aware",
        "deepseek_v3_risk",
        "deepseek_v4_risk",
        "qwen36_35b_risk",
        "glm52_risk",
        "llama4_maverick_risk",
        "llama33_70b_risk",
        "gpt41_risk",
        "qwen235b_risk",
        "glm45_risk",
    ]
    heat_defs = [d for d in preferred if d in available]
    rankable_max = overall.loc[overall["rankable"], "TPS"].max() if overall["rankable"].any() else math.nan
    oracle = overall[overall["defense"].eq("oracle")]
    if (
        not oracle.empty
        and "oracle" in available
        and (math.isnan(rankable_max) or float(oracle.iloc[0]["TPS"]) >= rankable_max)
    ):
        heat_defs.append("oracle")

    fig_height = max(1.82, 0.255 * len(heat_defs) + 0.68)
    fig, ax = plt.subplots(figsize=(3.35, fig_height))
    matrix = (
        scenario[scenario["defense"].isin(heat_defs)]
        .pivot_table(index="defense", columns="scenario", values="tps", aggfunc="mean")
        .reindex(heat_defs)
        .reindex(columns=list(SCENARIOS))
    )
    annot = np.where(np.isfinite(matrix.to_numpy(dtype=float)), matrix.to_numpy(dtype=float), np.nan)
    annot_labels = np.array([["" if not np.isfinite(v) else f"{v:.0f}" for v in row] for row in annot])
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=0.0,
        vmax=100.0,
        linewidths=0.45,
        linecolor="white",
        annot=annot_labels,
        fmt="",
        annot_kws={"fontsize": 7.0},
        cbar=True,
        cbar_kws={"label": "CRP", "shrink": 0.66, "pad": 0.02, "aspect": 18},
    )
    # Keep dark-cell numbers legible against the deep-purple end of the map.
    flat = matrix.to_numpy(dtype=float).ravel()
    for text, value in zip(ax.texts, flat):
        if np.isfinite(value) and value >= 62.0:
            text.set_color("white")
    ax.set_xticklabels(["S1\nPump", "S2\nFin.", "S3\nSpoof", "S4\nWash"], rotation=0, fontsize=7.5)
    ylabels = [DEFENSE_LABELS.get(d, d) + ("*" if d == "oracle" else "") for d in matrix.index]
    ax.set_yticklabels(ylabels, rotation=0, fontsize=7.3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if "oracle" in list(matrix.index):
        i = list(matrix.index).index("oracle")
        ax.add_patch(plt.Rectangle((0, i), len(SCENARIOS), 1, fill=False, edgecolor="#7a647f", linewidth=1.0, linestyle="--"))
    cbar = ax.collections[0].colorbar
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7.0, colors=ink, length=0)
    cbar.ax.yaxis.label.set_size(7.4)
    fig.tight_layout(pad=0.35)
    for directory in (
        ensure_dir(OUTPUTS / out_name / "figures"),
        ensure_dir(ROOT / "figures"),
        ensure_dir(ROOT.parent / "AuthorKit27" / "Figures") if (ROOT.parent / "AuthorKit27").exists() else ensure_dir(ROOT / "figures"),
    ):
        fig.savefig(directory / "figure7_e5_leaderboard.pdf", bbox_inches="tight")
        fig.savefig(directory / "figure7_e5_leaderboard.png", dpi=420, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="real_e5_wolfguard_benchmark")
    args = parser.parse_args()
    write_outputs(args.out)


if __name__ == "__main__":
    main()
