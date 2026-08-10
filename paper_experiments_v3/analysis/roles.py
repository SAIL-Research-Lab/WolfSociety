"""Behavioral-separation summary for P06 retail-role robustness."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from ..runtime.io_utils import OUTPUTS, write_csv, write_json


METRICS = (
    "trade_participation_rate",
    "action_entropy_bits",
    "role_action_information_bits",
    "role_trade_rate_gap",
    "social_information_bits",
    "private_information_bits",
    "primary_failure_rate",
    "decision_role_entropy_normalized",
    "effective_decision_roles",
    "pairwise_role_behavior_jsd_bits",
    "hse_like_behavioral_diversity",
    "population_role_entropy_normalized",
    "population_effective_roles",
)


def read_rows(run: str) -> list[dict[str, str]]:
    shard_paths = sorted(OUTPUTS.glob(f"{run}_s*/data.csv"))
    paths = shard_paths or [OUTPUTS / run / "data.csv"]
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows.extend(row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok")
    return rows


def as_float(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="p06_role_robustness_paper")
    args = parser.parse_args()
    out_dir = OUTPUTS / args.run
    rows = read_rows(args.run)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["variant"]].append(row)
    output = []
    for variant, group in sorted(groups.items()):
        record = {"population": variant, "n_episodes": len(group)}
        for metric in METRICS:
            values = [value for row in group if (value := as_float(row, metric)) is not None]
            record[f"{metric}_mean"] = mean(values)
        alpha0 = [
            float(row["primary_failure_rate"])
            for row in group
            if as_float(row, "alpha") == 0.0 and as_float(row, "primary_failure_rate") is not None
        ]
        record["alpha0_failure_mean"] = mean(alpha0)
        output.append(record)
    write_csv(output, out_dir / "role_analysis.csv")
    by_size: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("variant") == "mixed_roles" and as_float(row, "alpha") == 0.0:
            n_value = as_float(row, "n_society")
            if n_value is not None:
                by_size[int(n_value)].append(row)
    size_output = []
    diversity_metrics = (
        "decision_role_entropy_normalized",
        "effective_decision_roles",
        "pairwise_role_behavior_jsd_bits",
        "hse_like_behavioral_diversity",
        "population_role_entropy_normalized",
        "population_effective_roles",
        "role_action_information_bits",
        "role_trade_rate_gap",
        "alpha0_failure_mean",
    )
    for n_value, group in sorted(by_size.items()):
        record = {"n_society": n_value, "n_episodes": len(group)}
        for metric in diversity_metrics:
            if metric == "alpha0_failure_mean":
                values = [value for row in group if (value := as_float(row, "primary_failure_rate")) is not None]
            else:
                values = [value for row in group if (value := as_float(row, metric)) is not None]
            record[f"{metric}_mean"] = mean(values)
        size_output.append(record)
    write_csv(size_output, out_dir / "role_diversity_by_size.csv")
    write_json({
        "n_populations": len(output),
        "n_size_audit_cells": len(size_output),
        "reviewer_question": (
            "Are results an artifact of identical rational score agents or "
            "of behavioral diversity increasing mechanically with N?"
        ),
    }, out_dir / "role_analysis.json")
    print(f"Wrote {out_dir / 'role_analysis.csv'}")
    print(f"Wrote {out_dir / 'role_diversity_by_size.csv'}")


if __name__ == "__main__":
    main()
