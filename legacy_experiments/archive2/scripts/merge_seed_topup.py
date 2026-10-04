"""Merge an existing run with a seed top-up run.

This is intended for paper-facing top-ups where the original output contains
seeds 1--3 and a new run contains seeds 4--10 on the same grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import OUTPUTS
from .run_common import write_run_outputs


DEDUP_KEYS = ["scenario", "variant", "defense", "n_society", "alpha", "seed"]


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return OUTPUTS / value


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_data(directory: Path) -> pd.DataFrame:
    path = directory / "data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing data.csv: {path}")
    frame = pd.read_csv(path)
    missing = [key for key in DEDUP_KEYS if key not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required merge keys: {missing}")
    return frame


def parse_expected_seeds(value: str) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Existing output directory name or path")
    parser.add_argument("--topup", required=True, help="Top-up output directory name or path")
    parser.add_argument("--out", required=True, help="Merged output directory name")
    parser.add_argument("--expected-seeds", default="", help="Comma-separated seed set to validate")
    args = parser.parse_args()

    base_dir = resolve_run_dir(args.base)
    topup_dir = resolve_run_dir(args.topup)
    base = read_data(base_dir)
    topup = read_data(topup_dir)
    merged = pd.concat([base, topup], ignore_index=True)
    before = int(len(merged))
    merged = merged.drop_duplicates(subset=DEDUP_KEYS, keep="last")
    merged = merged.sort_values(DEDUP_KEYS, kind="mergesort").reset_index(drop=True)
    observed_seeds = sorted(int(seed) for seed in pd.to_numeric(merged["seed"], errors="coerce").dropna().unique())

    expected_seeds = parse_expected_seeds(args.expected_seeds)
    if expected_seeds is not None:
        observed = set(observed_seeds)
        if observed != expected_seeds:
            raise ValueError(f"Observed seeds {sorted(observed)} != expected {sorted(expected_seeds)}")

    base_config = read_json(base_dir / "config.json")
    base_summary = read_json(base_dir / "summary.json")
    topup_summary = read_json(topup_dir / "summary.json")
    summary_group_keys = list(base_summary.get("summary_group_keys", ["scenario", "variant", "defense", "n_society", "alpha"]))
    merged_grid = dict(base_config.get("grid", {}))
    merged_grid["seeds"] = observed_seeds
    config = {
        **base_config,
        "experiment": f"{base_config.get('experiment', args.base)} seed top-up merge",
        "grid": merged_grid,
        "merge_sources": {
            "base": str(base_dir),
            "topup": str(topup_dir),
            "rows_before_dedupe": before,
            "rows_after_dedupe": int(len(merged)),
            "base_n_error": int(base_summary.get("n_error", -1)),
            "topup_n_error": int(topup_summary.get("n_error", -1)),
        },
    }
    rows = merged.where(pd.notna(merged), "").to_dict(orient="records")
    out_dir = write_run_outputs(args.out, rows, config, summary_group_keys)
    print(f"Merged {len(base)} base rows + {len(topup)} top-up rows -> {len(merged)} rows at {out_dir}")


if __name__ == "__main__":
    main()