"""Read-only bootstrap CIs for the E2 mechanism audit (paper Table 5).

Reproduces the clean-normalized per-scenario excess-risk summaries from a
calibrated E2 per-episode CSV (column ``excess_primary_score``, averaged over
the available seeds at each alpha) and attaches nonparametric bootstrap 95%
CIs. Two statistics are reported per scenario, matching the paper table:

  * ``score there`` -- excess risk at the smallest nonzero alpha;
  * ``max score``   -- maximum seed-mean excess risk over the tested grid.

CIs come from resampling available seeds with replacement inside each alpha cell
(10000 draws by default): the "score there" CI resamples the seeds at the
smallest nonzero alpha, and the "max score" CI recomputes the seed-mean at every
alpha per draw and takes the maximum, so it reflects both seed noise and which
alpha wins. No new simulation or LLM call is launched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "real_e2_mechanisms" / "data_calibrated.csv"

SCEN_ORDER = ["s1_pump_dump", "s2_finfluencer", "s3_spoofing", "s4_wash"]
LABEL = {
    "s1_pump_dump": "S1",
    "s2_finfluencer": "S2",
    "s3_spoofing": "S3",
    "s4_wash": "S4",
}
METRIC = "excess_primary_score"


def _pct(a: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20240517)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    rng = np.random.default_rng(args.seed)

    print(f"E2 mechanism audit bootstrap CIs  (metric={METRIC})")
    n_seed = int(df["seed"].nunique()) if "seed" in df.columns else 0
    n_alpha = int(df["alpha"].nunique()) if "alpha" in df.columns else 0
    print(f"E2 mechanism audit: {n_seed} seeds x {n_alpha} alpha cells/scenario")
    print(f"{'scen':<5}{'a_min':>8}{'score':>8}{'  95% CI':>16}{'max':>8}{'  95% CI':>16}")
    for sc in SCEN_ORDER:
        s = df[df["scenario"].eq(sc)]
        alphas = sorted(float(a) for a in s["alpha"].dropna().unique())
        nonzero = [a for a in alphas if a > 0]
        a_min = min(nonzero)

        # per-alpha arrays of per-seed values
        per_alpha: dict[float, np.ndarray] = {}
        for a in alphas:
            vals = s.loc[np.isclose(s["alpha"], a), METRIC].to_numpy(dtype=float)
            per_alpha[a] = vals

        score_there = float(np.mean(per_alpha[a_min]))
        max_score = float(max(np.mean(per_alpha[a]) for a in alphas))

        # bootstrap
        boot_there = np.empty(args.n_boot)
        boot_max = np.empty(args.n_boot)
        for b in range(args.n_boot):
            # "score there": resample seeds at a_min
            v = per_alpha[a_min]
            boot_there[b] = rng.choice(v, size=len(v), replace=True).mean() if len(v) else np.nan
            # "max score": resample seeds within every alpha, take max of means
            means = []
            for a in alphas:
                va = per_alpha[a]
                if len(va):
                    means.append(rng.choice(va, size=len(va), replace=True).mean())
            boot_max[b] = max(means) if means else np.nan

        lo_t, hi_t = _pct(boot_there)
        lo_m, hi_m = _pct(boot_max)
        print(
            f"{LABEL[sc]:<5}{a_min:>8.3f}{score_there:>8.3f}"
            f"  [{lo_t:.3f},{hi_t:.3f}]{max_score:>8.3f}  [{lo_m:.3f},{hi_m:.3f}]"
        )


if __name__ == "__main__":
    main()
