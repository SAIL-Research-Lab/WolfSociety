"""CRP (TPS) weight/budget sensitivity for the E5 defense leaderboard.

Re-scores the *cached* E5 leaderboard data under perturbed safety-gain weights
(shift/critical-collapse/damage) and cost budgets (clean/false-positive/
intervention), and reports whether the published ranking is stable. No new
simulation is run: every perturbation reuses the same
``real_e5_openweight_latest_s5`` episode rows that produced the paper table.

Outputs (written under OUTPUTS/<out_name>/sensitivity/):
  * tps_sensitivity_configs.csv  - per-config official TPS for every defense
  * tps_sensitivity_summary.csv  - rank-stability summary per config
  * tps_sensitivity_summary.json - headline stability statistics
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wolfbench.metrics import TPSConfig

from .build_e5_leaderboard import PAPER_TPS_CONFIG, build_leaderboard
from .io_utils import OUTPUTS, ensure_dir

# Ranked defenses in the published order (best -> worst official TPS).
PAPER_ORDER = [
    "topology_aware",
    "qwen36_35b_risk",
    "deepseek_v3_risk",
    "llama4_maverick_risk",
    "zscore_guard",
    "rule",
]
LLM_DEFENSES = ["qwen36_35b_risk", "deepseek_v3_risk", "llama4_maverick_risk"]


def _kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    common = [d for d in order_a if d in order_b]
    rank_b = {d: i for i, d in enumerate(order_b)}
    seq = [rank_b[d] for d in common]
    n = len(seq)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] < seq[j]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return float((concordant - discordant) / total) if total else 1.0


def _ranked_order(overall: pd.DataFrame) -> list[str]:
    ranked = overall[overall["rankable"]].copy()
    ranked = ranked.sort_values(["TPS", "WorstScenarioTPS"], ascending=False)
    return list(ranked["defense"])


def _config_grid(reference: TPSConfig) -> list[tuple[str, TPSConfig]]:
    configs: list[tuple[str, TPSConfig]] = [("paper", reference)]

    # Safety-gain weight simplex perturbations around (0.15, 0.55, 0.30).
    weight_sets = {
        "w_equal": (1 / 3, 1 / 3, 1 / 3),
        "w_shift_heavy": (0.35, 0.45, 0.20),
        "w_collapse_heavy": (0.10, 0.70, 0.20),
        "w_damage_heavy": (0.10, 0.40, 0.50),
        "w_shift_up": (0.25, 0.50, 0.25),
        "w_shift_down": (0.05, 0.60, 0.35),
        "w_collapse_down": (0.25, 0.40, 0.35),
        "w_code_default": (0.55, 0.35, 0.10),
    }
    for name, (ws, wc, wd) in weight_sets.items():
        configs.append((name, replace(
            reference,
            shift_weight=ws,
            critical_collapse_weight=wc,
            damage_weight=wd,
        )))

    # Cost-budget perturbations around (clean=0.18, fp=0.25, int=0.30).
    for scale, tag in [(0.5, "tight"), (0.75, "tighter"), (1.5, "loose"), (2.0, "looser")]:
        configs.append((f"budget_{tag}", replace(
            reference,
            clean_cost_budget=reference.clean_cost_budget * scale,
            false_positive_budget=reference.false_positive_budget * scale,
            intervention_cost_budget=reference.intervention_cost_budget * scale,
        )))
    # One-at-a-time budget stress on the binding clean-cost gate.
    for scale, tag in [(0.5, "clean_tight"), (2.0, "clean_loose")]:
        configs.append((f"budget_{tag}", replace(
            reference,
            clean_cost_budget=reference.clean_cost_budget * scale,
        )))

    return configs


def run(out_name: str) -> dict[str, Any]:
    configs = _config_grid(PAPER_TPS_CONFIG)
    paper_order = _ranked_order(build_leaderboard(out_name, config=PAPER_TPS_CONFIG)[0])

    config_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for name, cfg in configs:
        overall, _, _, _ = build_leaderboard(out_name, config=cfg)
        order = _ranked_order(overall)
        tps_by_defense = dict(zip(overall["defense"], overall["TPS"]))
        for defense, tps in tps_by_defense.items():
            config_rows.append({
                "config": name,
                "defense": defense,
                "label": overall.loc[overall["defense"].eq(defense), "label"].iloc[0],
                "tps": float(tps),
                "rankable": bool(overall.loc[overall["defense"].eq(defense), "rankable"].iloc[0]),
            })

        best_llm = next((d for d in order if d in LLM_DEFENSES), None)
        summary_rows.append({
            "config": name,
            "shift_w": cfg.shift_weight,
            "collapse_w": cfg.critical_collapse_weight,
            "damage_w": cfg.damage_weight,
            "clean_budget": cfg.clean_cost_budget,
            "fp_budget": cfg.false_positive_budget,
            "int_budget": cfg.intervention_cost_budget,
            "top1_is_topology": order[:1] == ["topology_aware"],
            "best_llm_is_qwen": best_llm == "qwen36_35b_risk",
            "order_matches_paper": order == paper_order,
            "kendall_tau_vs_paper": _kendall_tau(paper_order, order),
            "ranked_order": ">".join(order),
        })

    config_df = pd.DataFrame(config_rows)
    summary_df = pd.DataFrame(summary_rows)

    out_dir = ensure_dir(OUTPUTS / out_name / "sensitivity")
    config_df.to_csv(out_dir / "tps_sensitivity_configs.csv", index=False)
    summary_df.to_csv(out_dir / "tps_sensitivity_summary.csv", index=False)

    taus = summary_df["kendall_tau_vs_paper"].to_numpy(dtype=float)
    headline = {
        "n_configs": int(len(summary_df)),
        "paper_ranked_order": paper_order,
        "frac_top1_topology": float(summary_df["top1_is_topology"].mean()),
        "frac_best_llm_qwen": float(summary_df["best_llm_is_qwen"].mean()),
        "frac_exact_order_match": float(summary_df["order_matches_paper"].mean()),
        "min_kendall_tau": float(taus.min()),
        "mean_kendall_tau": float(taus.mean()),
    }
    (out_dir / "tps_sensitivity_summary.json").write_text(json.dumps(headline, indent=2))

    return headline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="real_e5_openweight_latest_s5",
                        help="E5 output folder containing data.csv")
    args = parser.parse_args()
    headline = run(args.out)
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
