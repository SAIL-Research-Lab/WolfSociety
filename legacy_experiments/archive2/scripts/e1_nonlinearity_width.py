"""E1 nonlinearity (linear vs logistic) and transition-width (W_N) audit (S1).

Re-analyzes the *cached* refined S1 sweep (``real_e1_refined_s1``) using the
paper's own calibrated failure probability (``clean_calibrated_failure`` via
``add_calibration``). No new simulation is run; both quantities are read off the
existing per-episode grid.

For every resolved society size N it fits, on the same binomial data
(k collapses out of n episodes per alpha):

1. A linear response  p(alpha) = a + b*alpha  (2 params), clipped to [0,1].
2. A logistic response p(alpha) = sigma(s*(alpha - alpha_c))  (2 params).

Both are scored by binomial log-likelihood, so AIC = -2*LL + 2*k has the same
penalty term and Delta_AIC = AIC_linear - AIC_logistic directly measures how
much better the sharp (logistic) response fits than a straight line. Positive
Delta_AIC favours the nonlinear response.

It also reports the logistic transition width
    W_N = alpha_{0.9} - alpha_{0.1} = 2*ln(9) / s,
with a seed-bootstrap 95% band, quantifying finite-size sharpening.

Outputs under OUTPUTS/<out_name>/estimator_audit/:
  * nonlinearity_width.csv       - per-N Delta_AIC, LLs, slope, W_N (+band)
  * nonlinearity_width_summary.json - headline stats used in the paper text
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from .io_utils import OUTPUTS, ensure_dir
from .plot_e1_refined import add_calibration, aggregate_prob, fit_sigmoid, sigmoid

RESOLVED_N = [100, 200, 300, 500, 1000, 2000]
N_BOOT = 1000
RNG_SEED = 20260710
LOG9 = math.log(9.0)


def _binom_ll(k: np.ndarray, n: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(np.sum(k * np.log(p) + (n - k) * np.log(1.0 - p)))


def _linear_fit(prob_n: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Weighted least-squares line on p(alpha); returns (coef, binom LL)."""
    xs = prob_n["alpha"].astype(float).to_numpy()
    ys = prob_n["prob"].astype(float).to_numpy()
    ns = prob_n["n"].astype(float).to_numpy()
    ks = prob_n["k"].astype(float).to_numpy()
    coef = np.polyfit(xs, ys, 1, w=np.sqrt(np.clip(ns, 1.0, None)))
    pred = np.clip(np.polyval(coef, xs), 0.0, 1.0)
    return coef, _binom_ll(ks, ns, pred)


def _logistic_fit(prob_n: pd.DataFrame) -> tuple[float, float, float]:
    """Logistic fit; returns (slope, midpoint, binom LL)."""
    positive = prob_n[prob_n["alpha"].astype(float) > 0.0].sort_values("alpha")
    xs = positive["alpha"].astype(float).to_numpy()
    ns = positive["n"].astype(float).to_numpy()
    ks = positive["k"].astype(float).to_numpy()
    fit = fit_sigmoid(xs, positive["prob"].astype(float).to_numpy())
    if fit is None:
        return float("nan"), float("nan"), float("nan")
    slope, midpoint = fit
    pred = sigmoid(xs, slope, midpoint)
    return float(slope), float(midpoint), _binom_ll(ks, ns, pred)


def _width_from_slope(slope: float) -> float:
    return float(2.0 * LOG9 / slope) if slope and np.isfinite(slope) and slope > 0 else float("nan")


def _bootstrap_width(frame_n: pd.DataFrame, n_boot: int = N_BOOT) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    seeds = sorted(frame_n["seed"].dropna().unique())
    if len(seeds) < 2:
        return float("nan"), float("nan")
    by_seed = {s: frame_n[frame_n["seed"].eq(s)] for s in seeds}
    vals: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        sample = pd.concat([by_seed[s] for s in sampled], ignore_index=True)
        prob = aggregate_prob(sample)
        slope, _, _ = _logistic_fit(prob[prob["n_society"].eq(frame_n["n_society"].iloc[0])])
        w = _width_from_slope(slope)
        if np.isfinite(w):
            vals.append(w)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def run(out_name: str) -> dict:
    df = pd.read_csv(OUTPUTS / out_name / "data.csv")
    if "status" in df.columns:
        df = df[df["status"].fillna("ok").eq("ok")].copy()
    df = add_calibration(df, 1.0)
    prob = aggregate_prob(df)

    rows: list[dict] = []
    for n in RESOLVED_N:
        prob_n = prob[prob["n_society"].eq(n)]
        frame_n = df[df["n_society"].eq(n)]
        if prob_n.empty or prob_n["alpha"].astype(float).gt(0).sum() < 3:
            continue
        _, ll_lin = _linear_fit(prob_n)
        slope, midpoint, ll_log = _logistic_fit(prob_n)
        aic_lin = -2.0 * ll_lin + 2 * 2
        aic_log = -2.0 * ll_log + 2 * 2
        width = _width_from_slope(slope)
        w_lo, w_hi = _bootstrap_width(frame_n)
        rows.append({
            "n_society": n,
            "n_alpha_points": int(prob_n["alpha"].astype(float).gt(0).sum()),
            "logL_linear": ll_lin,
            "logL_logistic": ll_log,
            "aic_linear": aic_lin,
            "aic_logistic": aic_log,
            "delta_aic": aic_lin - aic_log,
            "logistic_slope": slope,
            "logistic_midpoint": midpoint,
            "width_W_N": width,
            "width_lo": w_lo,
            "width_hi": w_hi,
        })

    res = pd.DataFrame(rows)
    out_dir = ensure_dir(OUTPUTS / out_name / "estimator_audit")
    res.to_csv(out_dir / "nonlinearity_width.csv", index=False)

    resolved = res[res["delta_aic"].notna()]
    strong = resolved[resolved["n_society"] >= 500]
    headline = {
        "n_resolved": int(len(resolved)),
        "delta_aic_by_n": {int(r.n_society): round(float(r.delta_aic), 2) for r in resolved.itertuples()},
        "min_delta_aic_N_ge_500": float(strong["delta_aic"].min()) if not strong.empty else float("nan"),
        "max_delta_aic_N_ge_500": float(strong["delta_aic"].max()) if not strong.empty else float("nan"),
        "n_favoring_logistic": int((resolved["delta_aic"] > 0).sum()),
        "n_favoring_logistic_strong_ge_10": int((resolved["delta_aic"] > 10).sum()),
        "width_by_n": {int(r.n_society): round(float(r.width_W_N), 4) for r in resolved.itertuples()},
        "width_band_by_n": {
            int(r.n_society): [round(float(r.width_lo), 4), round(float(r.width_hi), 4)]
            for r in resolved.itertuples()
        },
        "width_monotone_decreasing": bool(
            np.all(np.diff(resolved.sort_values("n_society")["width_W_N"].to_numpy()) < 0)
        ),
    }
    (out_dir / "nonlinearity_width_summary.json").write_text(json.dumps(headline, indent=2))
    return headline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="real_e1_refined_s1", help="output dir name under OUTPUTS")
    args = parser.parse_args()
    headline = run(args.out)
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
