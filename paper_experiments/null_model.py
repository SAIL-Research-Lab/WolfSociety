"""Explicit Watts-style null, separate from the full-LLM society engine."""
import networkx as nx
import numpy as np


def run_null(config):
    n, alpha, seed = config["n_society"], config["alpha"], config["seed"]
    rng = np.random.default_rng(seed + 991)
    graph = nx.barabasi_albert_graph(n, min(4, n - 1), seed=int(rng.integers(2**31 - 1)))
    k = config.get("harmful_count", round(alpha * n))
    active = set(rng.choice(n, size=k, replace=False).tolist())
    thresholds = rng.beta(2, 8, size=n)
    for day in range(config["horizon_days"]):
        new = {i for i in range(n) if i not in active and
               sum(j in active for j in graph.neighbors(i)) / graph.degree(i) >= thresholds[i]}
        if not new:
            break
        active.update(new)
    fraction = len(active) / n
    return {"collapse": fraction >= .5, "primary_failure_rate": float(fraction >= .5),
            "collapse_rate": float(fraction >= .5), "primary_failure_score_max": fraction / .5,
            "watts_final_active_fraction": fraction, "watts_steps": day + 1,
            "n_harmful": k, "alpha_realized": k / n, "audit": {"engine": "watts_null", "llm_calls": 0}}
