"""Paper analysis for P04 game-theoretic phase conditions."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from .scaling import crossing
from ..runtime.io_utils import OUTPUTS, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="p04_game_phase")
    args = parser.parse_args()
    out_dir = OUTPUTS / args.run
    with (out_dir / "data.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok"]
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], int(float(row["n_society"])))].append(row)
    summaries = []
    for (variant, n_society), group in sorted(grouped.items()):
        by_alpha: dict[float, list[float]] = defaultdict(list)
        for row in group:
            by_alpha[float(row["alpha"])].append(float(row["primary_failure_rate"]))
        curve = sorted((alpha, sum(values) / len(values)) for alpha, values in by_alpha.items())
        coupling = sum(float(row["mean_social_coupling_proxy"]) for row in group) / len(group)
        alpha_c = crossing(curve, 0.5)
        alpha10, alpha90 = crossing(curve, 0.1), crossing(curve, 0.9)
        summaries.append({
            "variant": variant,
            "n_society": n_society,
            "mean_social_coupling_proxy": coupling,
            "predicted_regime": "multiple-equilibria candidate" if coupling > 1 else "unique-equilibrium candidate",
            "alpha_c": "" if alpha_c is None else alpha_c,
            "transition_width_10_90": "" if alpha10 is None or alpha90 is None else alpha90 - alpha10,
            "max_failure_probability": max((p for _, p in curve), default=0.0),
            "n_seeds": len({row["seed"] for row in group}),
        })
    write_csv(summaries, out_dir / "game_phase_analysis.csv")
    write_json({
        "n_conditions": len(summaries),
        "interpretation": "Test preregistered ordering across coupling interventions; do not treat proxy J as a proof about the full simulator.",
    }, out_dir / "game_phase_analysis.json")
    print(f"Wrote {out_dir / 'game_phase_analysis.csv'}")


if __name__ == "__main__":
    main()
