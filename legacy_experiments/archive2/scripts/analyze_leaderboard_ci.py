"""Read-only seed-robustness audit of the E5 WolfGuard leaderboard (paper Table 7).

The paper leaderboard (``real_e5_wolfguard_benchmark_sanitized``) is scored on
only TWO seeds per (defense, scenario, alpha) cell. The reported top cluster
(Qwen3 235B 76.2, GLM-4.5 76.1, Topology 75.3) is separated by less than one
TPS point. With two seeds a Gaussian confidence interval would over-state the
evidence, so this script instead runs an EXHAUSTIVE single-seed sensitivity:

  * both seeds  -> reproduces the paper table (official balanced TPS);
  * seed 1 only -> independent single-seed leaderboard;
  * seed 2 only -> independent single-seed leaderboard.

For each ranked policy it reports the official balanced TPS under each subset and
the resulting [min, max] envelope, and it reports how often the identity of the
rank-1 policy changes across the three subsets. A seed-level nonparametric
bootstrap (resampling the two seeds with replacement) is added as a secondary
readout. No new simulation or LLM call is launched; scoring reuses the exact
``threshold_protection_score`` + ``PAPER_TPS_CONFIG`` used to build the table.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from wolfbench.defense import get_track
from wolfbench.metrics import threshold_protection_score

from .build_e5_leaderboard import (
    MAX_SEMANTIC_FALLBACK_RATE,
    MIN_OVERALL_COVERAGE,
    MIN_SCENARIO_COVERAGE,
    OVERALL_MEAN_WEIGHT,
    OVERALL_WORST_WEIGHT,
    PAPER_TPS_CONFIG,
    SCENARIOS,
    _alphas_for,
    _fallback_rate,
    _rows,
    coverage_table,
    read_data,
)

DEFAULT_OUT = "real_e5_wolfguard_benchmark_sanitized"
LABELS = {
    "qwen235b_risk": "Qwen3 235B",
    "glm45_risk": "GLM-4.5",
    "topology_aware": "Topology",
    "llama33_70b_risk": "Llama 3.3 70B",
    "deepseek_v3_risk": "DeepSeek V3",
    "zscore_guard": "Z-Score",
    "rule": "Rule",
    "gpt41_risk": "GPT-4.1",
    "oracle": "Oracle",
    "noguard": "NoGuard",
}


def _balanced_tps(ok: pd.DataFrame, defense: str) -> tuple[float, bool]:
    """Official balanced TPS and rankability for one defense on a data subset."""
    scores: list[float] = []
    for scenario in SCENARIOS:
        rows_no = _rows(ok, "noguard", scenario)
        rows_def = _rows(ok, defense, scenario)
        alphas = [
            alpha
            for alpha in _alphas_for(ok, scenario)
            if any(abs(float(r.get("alpha", 0.0)) - alpha) <= 1e-12 for r in rows_no)
            and any(abs(float(r.get("alpha", 0.0)) - alpha) <= 1e-12 for r in rows_def)
        ]
        if defense == "noguard" or len(alphas) < 3 or not rows_no or not rows_def:
            scores.append(0.0)
            continue
        score = threshold_protection_score(rows_no, rows_def, alphas=alphas, config=PAPER_TPS_CONFIG)
        scores.append(float(score.get("tps") or 0.0))
    mean_tps = float(np.mean(scores)) if scores else 0.0
    worst_tps = float(np.min(scores)) if scores else 0.0
    return OVERALL_MEAN_WEIGHT * mean_tps + OVERALL_WORST_WEIGHT * worst_tps, True


def _leaderboard(full: pd.DataFrame, ok: pd.DataFrame) -> dict[str, float]:
    """Return {defense: official balanced TPS} for ranked policies only."""
    coverage = coverage_table(full)
    out: dict[str, float] = {}
    for defense in dict.fromkeys(str(x) for x in full["defense"].dropna()):
        track = get_track(defense)
        if defense == "noguard" or track == "oracle_upper_bound":
            continue
        cov_rows = coverage[coverage["defense"].eq(defense)]
        overall_expected = int(cov_rows["expected"].sum())
        overall_ok = int(cov_rows["ok"].sum())
        overall_coverage = float(overall_ok / max(overall_expected, 1))
        min_scenario_cov = float(cov_rows["coverage"].min()) if not cov_rows.empty else 0.0
        fallback_rate = _fallback_rate(ok[ok["defense"].eq(defense)].to_dict("records"))
        valid = (
            overall_coverage >= MIN_OVERALL_COVERAGE
            and min_scenario_cov >= MIN_SCENARIO_COVERAGE
            and fallback_rate <= MAX_SEMANTIC_FALLBACK_RATE
        )
        if not valid:
            continue
        tps, _ = _balanced_tps(ok, defense)
        out[defense] = tps
    return out


def _subset(full: pd.DataFrame, seeds: list[int] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = full if seeds is None else full[full["seed"].isin(seeds)].copy()
    ok = sub[sub["status"].fillna("ok").eq("ok")].copy()
    return sub, ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-name", default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20240517)
    args = parser.parse_args()

    full, _ = read_data(args.out_name)
    seeds = sorted(int(s) for s in full["seed"].dropna().unique())
    print(f"E5 leaderboard seed-robustness audit  (out={args.out_name})")
    print(f"seeds present: {seeds}  (n={len(seeds)} per cell)")
    print()

    # --- exhaustive single-seed sensitivity -----------------------------------
    subsets: dict[str, list[int] | None] = {"both": None}
    for s in seeds:
        subsets[f"seed{s}"] = [s]

    boards: dict[str, dict[str, float]] = {}
    for name, sel in subsets.items():
        sub, ok = _subset(full, sel)
        boards[name] = _leaderboard(sub, ok)

    ranked_defs = sorted(boards["both"], key=lambda d: boards["both"][d], reverse=True)
    header = ["policy"] + list(subsets) + ["min", "max", "range"]
    print(f"{'policy':<15}" + "".join(f"{h:>10}" for h in list(subsets) + ["min", "max", "range"]))
    for d in ranked_defs:
        vals = [boards[name].get(d, float("nan")) for name in subsets]
        finite = [v for v in vals if np.isfinite(v)]
        lo, hi = (min(finite), max(finite)) if finite else (float("nan"), float("nan"))
        row = f"{LABELS.get(d, d):<15}"
        for v in vals:
            row += f"{v:>10.2f}"
        row += f"{lo:>10.2f}{hi:>10.2f}{hi - lo:>10.2f}"
        print(row)

    print()
    rank1 = {name: max(b, key=b.get) for name, b in boards.items() if b}
    print("rank-1 policy per subset:")
    for name in subsets:
        d = rank1.get(name)
        print(f"  {name:<7} -> {LABELS.get(d, d)}  (TPS {boards[name].get(d, float('nan')):.2f})")
    distinct = {LABELS.get(d, d) for d in rank1.values()}
    print(f"  distinct rank-1 identities across subsets: {len(distinct)}  {sorted(distinct)}")

    # --- secondary: seed-level bootstrap over the two seeds -------------------
    print()
    print(f"secondary seed-bootstrap ({args.n_boot} draws, resampling {len(seeds)} seeds w/ replacement):")
    rng = np.random.default_rng(args.seed)
    boot: dict[str, list[float]] = {d: [] for d in ranked_defs}
    rank1_counts: dict[str, int] = {}
    for _ in range(args.n_boot):
        draw = list(rng.choice(seeds, size=len(seeds), replace=True))
        sub, ok = _subset(full, draw)
        board = _leaderboard(sub, ok)
        if not board:
            continue
        for d in ranked_defs:
            if d in board:
                boot[d].append(board[d])
        top = max(board, key=board.get)
        rank1_counts[top] = rank1_counts.get(top, 0) + 1
    for d in ranked_defs:
        arr = np.asarray(boot[d], dtype=float)
        if arr.size == 0:
            continue
        lo, hi = np.percentile(arr, 2.5), np.percentile(arr, 97.5)
        print(f"  {LABELS.get(d, d):<15} TPS {boards['both'][d]:6.2f}  95% CI [{lo:6.2f}, {hi:6.2f}]")
    total = sum(rank1_counts.values())
    print("  P(rank-1) by policy:")
    for d, c in sorted(rank1_counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {LABELS.get(d, d):<15} {c / total:6.3f}")


if __name__ == "__main__":
    main()
