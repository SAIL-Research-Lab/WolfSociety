"""Merge canonical Exp12 output with optional local refinement runs.

The current paper protocol uses the high-granularity canonical sweep directly.
This helper remains available for future partial refinements: it concatenates
raw episode rows, drops duplicate scenario/N/alpha/seed cells, and rebuilds the
same artifact schema as ``exp12_canonical_scaling.py``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from experiments._common import SCALING_THEORY_OUTPUTS_ROOT, scaling_exp_dir, write_csv, write_json
from experiments.scaling_theory import exp12_canonical_scaling as exp12


BASE_NAME = os.getenv("WOLFBENCH_EXP12_MERGE_BASE", "exp12_canonical_scaling_refined")
REFINEMENT_NAMES = [
    value.strip()
    for value in os.getenv("WOLFBENCH_EXP12_MERGE_REFINEMENTS", "exp12_s2_s3_smooth_refinement").split(",")
    if value.strip()
]
OUT_NAME = os.getenv("WOLFBENCH_EXP12_MERGE_OUT", "exp12_canonical_scaling_refined_smooth")
CI_BOOT = int(os.getenv("WOLFBENCH_EXP12_MERGE_CI_BOOT", str(exp12.CI_BOOT)))
DEDUP_KEY = ["scenario", "n_society", "alpha", "seed"]


def _resolve_output_dir(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return SCALING_THEORY_OUTPUTS_ROOT / value


def _read_data(directory: Path) -> pd.DataFrame:
    path = directory / "data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp12 data file: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in DEDUP_KEY if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def _scenario_order(present: set[str]) -> list[str]:
    ordered = [scenario for scenario in exp12.SCENARIOS if scenario in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.where(pd.notna(frame), "")
    return clean.to_dict(orient="records")


def _alpha_grids(frame: pd.DataFrame, scenarios: list[str]) -> dict[str, list[float]]:
    grids: dict[str, list[float]] = {}
    for scenario in scenarios:
        values = pd.to_numeric(frame.loc[frame["scenario"] == scenario, "alpha"], errors="coerce")
        grids[scenario] = sorted(float(value) for value in values.dropna().unique())
    return grids


def main() -> None:
    base_dir = _resolve_output_dir(BASE_NAME)
    refinement_dirs = [_resolve_output_dir(name) for name in REFINEMENT_NAMES]
    source_dirs = [base_dir, *refinement_dirs]
    frames = [_read_data(directory) for directory in source_dirs]
    merged = pd.concat(frames, ignore_index=True)
    before = int(len(merged))
    merged = merged.drop_duplicates(subset=DEDUP_KEY, keep="last")
    merged = merged.sort_values(DEDUP_KEY, kind="mergesort").reset_index(drop=True)
    after = int(len(merged))

    scenarios = _scenario_order(set(merged["scenario"].astype(str)))
    n_grid = sorted(int(value) for value in pd.to_numeric(merged["n_society"], errors="coerce").dropna().unique())
    seeds = sorted(int(value) for value in pd.to_numeric(merged["seed"], errors="coerce").dropna().unique())
    alpha_grids = _alpha_grids(merged, scenarios)

    exp12.SCENARIOS = scenarios
    exp12.N_GRID = n_grid
    exp12.SEEDS = seeds
    exp12.CI_BOOT = CI_BOOT

    rows = _records(merged)
    curve_rows = exp12.build_failure_curve_rows(rows, alpha_grids, scenarios=scenarios, n_grid=n_grid)
    threshold_rows = exp12.estimate_threshold_rows(rows, alpha_grids, scenarios=scenarios, n_grid=n_grid)
    width_rows = exp12.build_width_rows(threshold_rows)
    scenario_rows = exp12.build_scenario_law_summary(
        threshold_rows,
        curve_rows,
        scenarios=scenarios,
        n_grid=n_grid,
    )

    out = scaling_exp_dir(OUT_NAME)
    write_csv(rows, out / "data.csv")
    write_csv(curve_rows, out / "failure_curves.csv")
    write_csv(threshold_rows, out / "alpha_c_by_scenario_n.csv")
    write_csv(width_rows, out / "width_by_scenario_n.csv")
    write_csv(scenario_rows, out / "scenario_law_summary.csv")
    exp12.write_report(threshold_rows, scenario_rows, out / "report.md")
    exp12.plot_failure_curves(curve_rows, out / "primary_failure_curves.png")
    exp12.plot_widths(width_rows, out / "transition_width_by_n.png")
    write_json({
        "base": str(base_dir),
        "refinements": [str(directory) for directory in refinement_dirs],
        "output": str(out),
        "dedupe_key": DEDUP_KEY,
        "rows_before_dedupe": before,
        "rows_after_dedupe": after,
        "scenarios": scenarios,
        "n_grid": n_grid,
        "alpha_grids": alpha_grids,
        "seeds": seeds,
        "ci_boot": CI_BOOT,
        "primary_metrics": exp12.PRIMARY_METRIC_LABELS,
        "threshold_summary": threshold_rows,
        "scenario_law_summary": scenario_rows,
        "outputs": {
            "data": "data.csv",
            "failure_curves": "failure_curves.csv",
            "alpha_c_by_scenario_n": "alpha_c_by_scenario_n.csv",
            "width_by_scenario_n": "width_by_scenario_n.csv",
            "scenario_law_summary": "scenario_law_summary.csv",
            "report": "report.md",
            "primary_failure_curves": "primary_failure_curves.png",
            "transition_width_by_n": "transition_width_by_n.png",
        },
    }, out / "summary.json")
    print(f"Merged {after} unique episodes from {len(source_dirs)} source directories into {out}")


if __name__ == "__main__":
    main()
