"""Regenerate Exp8 sensitivity figures from an existing summary CSV.

This avoids rerunning the full sensitivity audit when only paper-facing plots
need to be refreshed.

Usage::
    python -m experiments.scaling_theory.plot_exp8_sensitivity
"""
from __future__ import annotations

import csv
import json
import os

from experiments._common import SCALING_THEORY_OUTPUTS_ROOT
from experiments.scaling_theory.exp8_sensitivity_audit import (
    FAMILY_LABELS,
    write_sensitivity_figures,
)


EXP8_DIR = SCALING_THEORY_OUTPUTS_ROOT / os.getenv(
    "WOLFBENCH_EXP8_OUT",
    "exp8_sensitivity_audit",
)


def _read_rows(path) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    summary_path = EXP8_DIR / "sensitivity_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run exp8_sensitivity_audit first.")
    rows = _read_rows(summary_path)
    scenarios = sorted({str(r["scenario"]) for r in rows})
    families = [family for family in FAMILY_LABELS if any(r["family"] == family for r in rows)]
    delta_rows = write_sensitivity_figures(rows, EXP8_DIR, scenarios, families)

    metadata_path = EXP8_DIR / "plot_summary.json"
    metadata_path.write_text(json.dumps({
        "summary_csv": "sensitivity_summary.csv",
        "delta_summary_csv": "sensitivity_delta_summary.csv",
        "figures": {
            "sensitivity_delta_heatmap": "sensitivity_delta_heatmap.png",
            "top_sensitivity_curves": "top_sensitivity_curves.png",
            "feedback_sensitivity": "feedback_sensitivity.png",
        },
        "strongest_sensitivities": sorted(
            delta_rows,
            key=lambda r: float(r["delta_collapse_rate"]),
            reverse=True,
        )[:8],
    }, indent=2) + "\n")
    print(f"Wrote Exp8 figures to {EXP8_DIR}")


if __name__ == "__main__":
    main()