"""E1 alpha_c estimator robustness and grid-refinement convergence (S1).

Re-analyzes the *cached* refined S1 sweep (``real_e1_refined_s1``) using the
paper's own calibrated failure probability (``clean_calibrated_failure`` via
``add_calibration``) to answer two reviewer questions without new simulation:

1. Estimator robustness: the paper reports a non-parametric linear-interpolation
   midpoint. We recompute alpha_c with a logistic (sigmoid) fit on the *same*
   calibrated probability and check the leftward trend and per-N agreement.
2. Grid-refinement convergence: alpha_c stability when the alpha grid is
   coarsened (drop every other interior point) vs the full refined grid.

It also quantifies the N=500 vs N=1000 non-monotonicity with seed-bootstrap
bands.

Outputs under OUTPUTS/real_e1_refined_s1/estimator_audit/:
  * alpha_c_estimators.csv       - per-N linear vs logistic midpoints (+bands)
  * alpha_c_grid_refinement.csv  - per-N full vs coarsened-grid midpoints
  * estimator_audit_summary.json - headline agreement / monotonicity stats
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .io_utils import OUTPUTS, ensure_dir
from .plot_e1_refined import (
    add_calibration,
    aggregate_prob,
    alpha_c_from_group,
    fit_sigmoid,
)

RESOLVED_N = [100, 200, 300, 500, 1000, 2000]
N_BOOT = 1000
RNG_SEED = 20260710


def _linear_ac(prob_n: pd.DataFrame) -> tuple[float, str]:
    return alpha_c_from_group(prob_n)


def _logistic_ac(prob_n: pd.DataFrame) -> float:
    positive = prob_n[prob_n["alpha"].astype(float) > 0.0].sort_values("alpha")
    xs = positive["alpha"].to_numpy(dtype=float)
    ys = positive["prob"].to_numpy(dtype=float)
    fit = fit_sigmoid(xs, ys)
    if fit is None:
        ac, _ = alpha_c_from_group(prob_n)
        return ac
    return float(fit[1])


def _coarsen_alphas(alphas: np.ndarray) -> np.ndarray:
    if alphas.size <= 3:
        return alphas
    interior = alphas[1:-1]
    return np.unique(np.concatenate([[alphas[0]], interior[::2], [alphas[-1]]]))


def _bootstrap(frame_n: pd.DataFrame, estimator, n_boot: int = N_BOOT) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    seeds = sorted(frame_n["seed"].dropna().unique())
    if len(seeds) < 2:
        return float("nan"), float("nan")
    vals: list[float] = []
    by_seed = {s: frame_n[frame_n["seed"].eq(s)] for s in seeds}
    for _ in range(n_boot):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        sample = pd.concat([by_seed[s] for s in sampled], ignore_index=True)
        prob = aggregate_prob(sample)
        ac = estimator(prob)
        if isinstance(ac, tuple):
            ac = ac[0]
        if ac is not None and np.isfinite(ac):
            vals.append(float(ac))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def run(out_name: str) -> dict:
    df = pd.read_csv(OUTPUTS / out_name / "data.csv")
    if "status" in df.columns:
        df = df[df["status"].fillna("ok").eq("ok")].copy()
    df = add_calibration(df, 1.0)
    prob = aggregate_prob(df)

    est_rows: list[dict] = []
    grid_rows: list[dict] = []
    for n in RESOLVED_N:
        prob_n = prob[prob["n_society"].eq(n)]
        frame_n = df[df["n_society"].eq(n)]
        if prob_n.empty:
            continue
        ac_lin, method = _linear_ac(prob_n)
        ac_log = _logistic_ac(prob_n)
        lin_lo, lin_hi = _bootstrap(frame_n, _linear_ac)
        log_lo, log_hi = _bootstrap(frame_n, _logistic_ac)
        est_rows.append({
            "n_society": n,
            "alpha_c_linear": ac_lin,
            "linear_method": method,
            "linear_lo": lin_lo,
            "linear_hi": lin_hi,
            "alpha_c_logistic": ac_log,
            "logistic_lo": log_lo,
            "logistic_hi": log_hi,
            "abs_rel_diff": abs(ac_log - ac_lin) / ac_lin if ac_lin else float("nan"),
        })

        alphas = np.array(sorted(prob_n["alpha"].astype(float).unique()))
        coarse = _coarsen_alphas(alphas)
        prob_coarse = prob_n[prob_n["alpha"].astype(float).isin(coarse)]
        ac_lin_c, _ = _linear_ac(prob_coarse)
        ac_log_c = _logistic_ac(prob_coarse)
        grid_rows.append({
            "n_society": n,
            "n_alpha_full": int(alphas.size),
            "n_alpha_coarse": int(coarse.size),
            "alpha_c_linear_full": ac_lin,
            "alpha_c_linear_coarse": ac_lin_c,
            "linear_refine_rel_change": abs(ac_lin_c - ac_lin) / ac_lin if ac_lin else float("nan"),
            "alpha_c_logistic_full": ac_log,
            "alpha_c_logistic_coarse": ac_log_c,
        })

    est_df = pd.DataFrame(est_rows)
    grid_df = pd.DataFrame(grid_rows)
    out_dir = ensure_dir(OUTPUTS / out_name / "estimator_audit")
    est_df.to_csv(out_dir / "alpha_c_estimators.csv", index=False)
    grid_df.to_csv(out_dir / "alpha_c_grid_refinement.csv", index=False)

    def _row(n):
        return est_df[est_df["n_society"].eq(n)].iloc[0]

    def _trend_decreasing(col: str, ns=(100, 200, 300, 1000, 2000)) -> bool:
        seq = est_df.set_index("n_society")[col].reindex(list(ns)).to_numpy(dtype=float)
        return bool(np.all(np.diff(seq) < 0))

    r500, r1000 = _row(500), _row(1000)
    overlap = not (r500["linear_hi"] < r1000["linear_lo"] or r1000["linear_hi"] < r500["linear_lo"])
    headline = {
        "n_resolved": int(len(est_df)),
        "max_abs_rel_diff_linear_vs_logistic": float(np.nanmax(est_df["abs_rel_diff"].to_numpy())),
        "mean_abs_rel_diff_linear_vs_logistic": float(np.nanmean(est_df["abs_rel_diff"].to_numpy())),
        "max_grid_refine_rel_change_linear": float(np.nanmax(grid_df["linear_refine_rel_change"].to_numpy())),
        "trend_decreasing_linear_excl500": _trend_decreasing("alpha_c_linear"),
        "trend_decreasing_logistic_excl500": _trend_decreasing("alpha_c_logistic"),
        "n500_vs_n1000": {
            "alpha_c_500": float(r500["alpha_c_linear"]),
            "alpha_c_1000": float(r1000["alpha_c_linear"]),
            "delta": float(r1000["alpha_c_linear"] - r500["alpha_c_linear"]),
            "band_500": [float(r500["linear_lo"]), float(r500["linear_hi"])],
            "band_1000": [float(r1000["linear_lo"]), float(r1000["linear_hi"])],
            "bands_overlap": bool(overlap),
        },
    }
    (out_dir / "estimator_audit_summary.json").write_text(json.dumps(headline, indent=2))
    return headline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="real_e1_refined_s1")
    args = parser.parse_args()
    print(json.dumps(run(args.out), indent=2))


if __name__ == "__main__":
    main()
