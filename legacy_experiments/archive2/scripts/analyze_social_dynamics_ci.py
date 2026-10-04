"""Read-only reanalysis of the E3 social-dynamic audit (paper Table 6).

Reproduces the calibrated per-regime point estimates from the authoritative
per-episode ``real_e3_social_dynamics/data_calibrated.csv`` (which already
carries ``excess_primary_score``, ``clean_calibrated_failure`` and the raw
channel metrics) and attaches nonparametric bootstrap confidence intervals.
No new simulation or LLM call is launched.

Aggregation matches the paper: each regime is averaged over the FULL alpha grid
(including alpha=0, where excess and calibrated failure are zero by
construction). "failure" is the baseline-relative ``clean_calibrated_failure``:
a sweep counts only if its primary score clears the regime's own alpha=0
baseline by ``max(0.20, 2*baseline_std)``. Because HighReach's higher graph
degree raises its own alpha=0 baseline and its variance, this detector is
conservative there even though excess risk and cascade both rise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "real_e3_social_dynamics" / "data_calibrated.csv"

VARIANT_ORDER = [
    "no_social",
    "baseline",
    "high_exposure",
    "high_reach",
    "hub_placement",
    "reflexive",
]
LABEL = {
    "no_social": "NoSocial",
    "baseline": "Baseline",
    "high_exposure": "HighExp.",
    "high_reach": "HighReach",
    "hub_placement": "Hub",
    "reflexive": "Reflex.",
}
METRICS = {
    "excess": "excess_primary_score",
    "failure": "clean_calibrated_failure",
    "cascade": "social_cascade_peak",
}


def _bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    draws = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frame = pd.read_csv(args.data)
    frame = frame[frame["scenario"].astype(str).str.startswith("s1")].copy()
    for col in METRICS.values():
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    rng = np.random.default_rng(args.seed)

    n_seed = int(frame["seed"].nunique())
    n_alpha = int(frame["alpha"].nunique())
    print(f"E3 social-dynamic audit (S1, N=300): {n_seed} seeds x {n_alpha} alpha cells/regime")
    print(f"{'regime':<10}{'excess [95% CI]':>24}{'failure [95% CI]':>24}{'cascade [95% CI]':>24}")
    print("-" * 82)
    for var in VARIANT_ORDER:
        sub = frame[frame["variant"] == var]
        if sub.empty:
            continue
        cells = []
        for _, col in METRICS.items():
            vals = sub[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            pt = float(vals.mean())
            lo, hi = _bootstrap_ci(vals, args.n_boot, rng)
            cells.append(f"{pt:.3f} [{lo:.3f},{hi:.3f}]")
        print(f"{LABEL[var]:<10}{cells[0]:>24}{cells[1]:>24}{cells[2]:>24}")


if __name__ == "__main__":
    main()
