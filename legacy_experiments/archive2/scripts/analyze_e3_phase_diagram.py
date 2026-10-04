"""Summarize E3b coupling regimes and empirical response crossings."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .io_utils import OUTPUTS, write_csv, write_json


def _crossing(points: list[tuple[float, float]], target: float) -> float | None:
    points = sorted(points)
    for (a0, p0), (a1, p1) in zip(points, points[1:]):
        if p0 == target:
            return a0
        if (p0 - target) * (p1 - target) <= 0 and p1 != p0:
            return a0 + (target - p0) * (a1 - a0) / (p1 - p0)
    if points and points[-1][1] == target:
        return points[-1][0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="e3_social_phase_diagram")
    parser.add_argument("--data", default="")
    args = parser.parse_args()
    out_dir = OUTPUTS / args.out
    data_path = Path(args.data) if args.data else out_dir / "data.csv"
    with data_path.open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if str(row.get("status", "ok") or "ok") == "ok"
        ]
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), int(float(row["n_society"])))].append(row)
    summaries = []
    for (variant, n_society), group in sorted(grouped.items()):
        by_alpha: dict[float, list[float]] = defaultdict(list)
        coupling = []
        for row in group:
            by_alpha[float(row["alpha"])].append(float(row["primary_failure_rate"]))
            coupling.append(float(row.get("mean_social_coupling_proxy", 0.0) or 0.0))
        points = sorted((alpha, sum(values) / len(values)) for alpha, values in by_alpha.items())
        alpha10, alpha50, alpha90 = (_crossing(points, target) for target in (0.1, 0.5, 0.9))
        mean_coupling = sum(coupling) / len(coupling) if coupling else 0.0
        if alpha50 is None:
            crossing_status = "below_grid" if points and points[0][1] >= 0.5 else "above_grid"
        else:
            crossing_status = "resolved"
        summaries.append({
            "variant": variant,
            "n_society": n_society,
            "mean_social_coupling_proxy": mean_coupling,
            "meanfield_regime": "multiple_equilibria_candidate" if mean_coupling > 1.0 else "unique_equilibrium_candidate",
            "alpha_c": "" if alpha50 is None else alpha50,
            "alpha_10": "" if alpha10 is None else alpha10,
            "alpha_90": "" if alpha90 is None else alpha90,
            "transition_width_10_90": "" if alpha10 is None or alpha90 is None else alpha90 - alpha10,
            "crossing_status": crossing_status,
            "min_failure_probability": min((p for _, p in points), default=0.0),
            "max_failure_probability": max((p for _, p in points), default=0.0),
            "n_alpha": len(points),
            "n_episodes": len(group),
        })
    write_csv(summaries, out_dir / "phase_diagram_summary.csv")
    write_json({
        "data": str(data_path),
        "n_conditions": len(summaries),
        "warning": (
            "The coupling value is a simulator-to-mean-field proxy. Treat J>1 as a "
            "preregistered regime prediction, not a proof about the full heterogeneous simulator."
        ),
    }, out_dir / "phase_diagram_summary.json")
    print(f"Wrote {out_dir / 'phase_diagram_summary.csv'}")


if __name__ == "__main__":
    main()
