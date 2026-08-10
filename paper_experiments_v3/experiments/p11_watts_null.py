"""P11: Watts-style threshold null model on matched graph sizes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np

from ..runtime.runner import add_run_args, alpha_values, as_run_args, load_protocol, profile_values, write_artifacts


def _simulate_watts(n_society: int, alpha: float, seed: int, *, mean_degree: int = 8) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 991)
    graph_seed = int(rng.integers(0, 2**31 - 1))
    graph = nx.barabasi_albert_graph(n_society, max(1, mean_degree // 2), seed=graph_seed)
    n_harmful = int(round(alpha * n_society))
    harmful = set(rng.choice(n_society, size=min(n_harmful, n_society), replace=False).tolist()) if n_harmful else set()
    active = set(harmful)
    thresholds = rng.beta(2.0, 8.0, size=n_society)
    thresholds[list(harmful)] = 0.0
    steps = 0
    for steps in range(1, 31):
        newly_active = set(active)
        for node in graph.nodes:
            if node in active:
                continue
            degree = graph.degree[node]
            if degree <= 0:
                continue
            active_fraction = sum((nbr in active) for nbr in graph.neighbors(node)) / degree
            if active_fraction >= thresholds[node]:
                newly_active.add(node)
        if len(newly_active) == len(active):
            break
        active = newly_active
    final_active_fraction = len(active) / max(n_society, 1)
    benign_active = max(len(active) - len(harmful), 0)
    benign_count = max(n_society - len(harmful), 1)
    benign_active_fraction = benign_active / benign_count
    return {
        "scenario": "watts_threshold_null",
        "n_society": n_society,
        "alpha": alpha,
        "seed": seed,
        "target_asset": "null",
        "n_harmful": n_harmful,
        "n_harmful_nominal": alpha * n_society,
        "placement": "random",
        "liquidity_exponent": "",
        "collapse_rate": float(final_active_fraction >= 0.5),
        "primary_failure_rate": float(final_active_fraction >= 0.5),
        "primary_failure_score_max": final_active_fraction,
        "watts_final_active_fraction": final_active_fraction,
        "watts_benign_active_fraction": benign_active_fraction,
        "watts_steps_to_fixed_point": steps,
        "watts_threshold_mean": float(np.mean(thresholds)),
        "status": "ok",
        "error_type": "",
        "error": "",
    }


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["n_society"]), float(row["alpha"]))].append(row)
    out = []
    for (n_society, alpha), group in sorted(groups.items()):
        values = [float(row["primary_failure_rate"]) for row in group]
        out.append({
            "n_society": n_society,
            "alpha": alpha,
            "n_seeds": len(group),
            "primary_failure_rate_mean": float(np.mean(values)),
            "active_fraction_mean": float(np.mean([row["watts_final_active_fraction"] for row in group])),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_args(parser, "p11_watts_null")
    args = as_run_args(parser.parse_args())
    seeds, requested_n = profile_values(args.profile)
    protocol = load_protocol()
    refined = {int(n): values for n, values in protocol["s1_refined_alpha"].items()}
    n_values = [n for n in requested_n if n in refined]
    rows = []
    frozen_grid = {}
    for n_society in n_values:
        grid = alpha_values("s1", refined[n_society])
        frozen_grid[n_society] = grid
        for alpha in grid:
            for seed in seeds:
                rows.append(_simulate_watts(n_society, alpha, seed))
    out = write_artifacts(
        args=args,
        experiment_id="P11_WATTS_NULL",
        rows=rows,
        config={
            "reviewer_question": "Can a simple Watts threshold null reproduce the observed finite-size boundary?",
            "graph": "barabasi_albert matched mean_degree=8",
            "threshold_distribution": "Beta(2, 8)",
            "collapse_event": "final active fraction >= 0.5",
            "n_specific_alpha_grid": frozen_grid,
            "seeds": seeds,
        },
        group_keys=["n_society", "alpha"],
        extra_csv={"watts_summary.csv": _summaries(rows)},
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()