#!/usr/bin/env python3
"""Validate checkpoint completeness and causal-design invariants."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw"
OUT = HERE / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = json.loads((HERE / "config.json").read_text())


def rows_for(stem: str):
    path = RAW / f"{stem}.jsonl"
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def key(row):
    return (
        row.get("experiment"), row.get("variant"), int(row.get("n_society", 0)),
        f"{float(row.get('alpha', 0)):.12g}", int(row.get("seed", 0)),
    )


def audit(stem: str, expected_per_seed: int) -> dict:
    rows = rows_for(stem)
    keys = [key(row) for row in rows]
    by_seed = Counter(int(row["seed"]) for row in rows if row.get("status") == "ok")
    return {
        "rows": len(rows),
        "ok": sum(row.get("status") == "ok" for row in rows),
        "errors": sum(row.get("status") != "ok" for row in rows),
        "duplicate_keys": len(keys) - len(set(keys)),
        "expected_rows_per_seed": expected_per_seed,
        "complete_seeds": sorted(seed for seed, count in by_seed.items() if count == expected_per_seed),
        "partial_seed_counts": {str(seed): count for seed, count in sorted(by_seed.items()) if count != expected_per_seed},
        "source_hashes": sorted({str(row.get("source_hash", "")) for row in rows}),
        "population_llm_calls_sum": sum(float(row.get("population_llm_calls", 0.0)) for row in rows),
    }


def main() -> None:
    fixed_expected = (
        len(CONFIG["liquidity_exponents"])
        * len(CONFIG["fixed_k"]["n_values"])
        * len(CONFIG["fixed_k"]["harmful_counts"])
    )
    boundary_expected = len(CONFIG["liquidity_exponents"]) * sum(
        len(values) for values in CONFIG["boundary"]["alpha_by_n"].values()
    )
    # Includes the additional first-hop-only condition documented in README.
    feedback_expected = 4 * sum(
        len(values) for values in CONFIG["feedback_ablation"]["alpha_by_n"].values()
    )
    payload = {
        "liquidity_fixed_k": audit("liquidity_fixed_k", fixed_expected),
        "liquidity_boundary": audit("liquidity_boundary", boundary_expected),
        "feedback_ablation": audit("feedback_ablation", feedback_expected),
    }
    feedback_rows = rows_for("feedback_ablation")
    no_social = [row for row in feedback_rows if row.get("variant") == "no_social_propagation"]
    payload["metric_audit"] = {
        "no_social_primary_all_zero": bool(no_social) and all(
            float(row.get("primary_failure_score_max", 0.0)) == 0.0 for row in no_social
        ),
        "reason": (
            "S1 primary score is min(social_score, market_score), so a zero-social "
            "ablation cannot be interpreted through the joint primary endpoint."
        ),
    }
    (OUT / "validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
