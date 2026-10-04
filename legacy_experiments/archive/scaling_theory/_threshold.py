"""Threshold estimators for scaling-theory experiments."""
from __future__ import annotations

from typing import Iterable

import numpy as np

_CURVE_FIT = None
_CURVE_FIT_CHECKED = False


def expit(x):
    arr = np.array(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def _get_curve_fit():
    global _CURVE_FIT, _CURVE_FIT_CHECKED
    if _CURVE_FIT_CHECKED:
        return _CURVE_FIT
    _CURVE_FIT_CHECKED = True
    try:  # pragma: no cover - depends on optional local SciPy.
        from scipy.optimize import curve_fit as scipy_curve_fit
    except Exception:
        _CURVE_FIT = None
    else:
        _CURVE_FIT = scipy_curve_fit
    return _CURVE_FIT


def linear_alpha_c(alphas: Iterable[float], probs: Iterable[float], threshold: float = 0.5) -> float | None:
    alpha_arr = np.array(list(alphas), dtype=float)
    prob_arr = np.array(list(probs), dtype=float)
    idx = np.where(prob_arr >= threshold)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(alpha_arr[0])
    p0, p1 = prob_arr[i - 1], prob_arr[i]
    a0, a1 = alpha_arr[i - 1], alpha_arr[i]
    if p1 == p0:
        return float(a1)
    alpha_c = a0 + (threshold - p0) * (a1 - a0) / (p1 - p0)
    return float(alpha_c)


def logistic_curve(alpha: np.ndarray, alpha_c: float, slope: float) -> np.ndarray:
    return expit(slope * (alpha - alpha_c))


def fit_logistic_threshold(alphas: Iterable[float], probs: Iterable[float], threshold: float = 0.5) -> dict:
    alpha_arr = np.array(list(alphas), dtype=float)
    prob_arr = np.clip(np.array(list(probs), dtype=float), 1e-6, 1 - 1e-6)
    linear = linear_alpha_c(alpha_arr, prob_arr, threshold=threshold)
    if alpha_arr.size < 3 or np.ptp(prob_arr) < 1e-6:
        return {
            "alpha_c": linear,
            "slope": None,
            "transition_width_10_90": None,
            "method": "linear_fallback_constant_curve",
            "fitted_probs": prob_arr.tolist(),
        }

    alpha0 = float(linear if linear is not None else np.median(alpha_arr))
    span = max(float(alpha_arr.max() - alpha_arr.min()), 1e-6)
    slope0 = 8.0 / span
    curve_fit = _get_curve_fit()
    if curve_fit is None:
        return {
            "alpha_c": linear,
            "slope": None,
            "transition_width_10_90": None,
            "method": "linear_fallback_no_scipy",
            "fitted_probs": prob_arr.tolist(),
        }

    try:
        params, _ = curve_fit(
            logistic_curve,
            alpha_arr,
            prob_arr,
            p0=[alpha0, slope0],
            bounds=([float(alpha_arr.min()), 1e-6], [float(alpha_arr.max()), 1e6]),
            maxfev=20000,
        )
        alpha_c, slope = float(params[0]), float(params[1])
        width = float(2.0 * np.log(9.0) / slope) if slope > 0 else None
        return {
            "alpha_c": alpha_c,
            "slope": slope,
            "transition_width_10_90": width,
            "method": "logistic_fit",
            "fitted_probs": logistic_curve(alpha_arr, alpha_c, slope).tolist(),
        }
    except Exception:
        return {
            "alpha_c": linear,
            "slope": None,
            "transition_width_10_90": None,
            "method": "linear_fallback_fit_failed",
            "fitted_probs": prob_arr.tolist(),
        }


def bootstrap_logistic_ci(rows: list[dict], alphas: list[float], n_boot: int = 500,
                          threshold: float = 0.5, rng_seed: int = 12345) -> dict:
    rng = np.random.default_rng(rng_seed)
    by_alpha = {
        float(alpha): [float(r["collapse_rate"]) for r in rows if float(r["alpha"]) == float(alpha)]
        for alpha in alphas
    }
    samples = []
    for _ in range(n_boot):
        probs = []
        valid = True
        for alpha in alphas:
            vals = np.array(by_alpha[float(alpha)], dtype=float)
            if vals.size == 0:
                valid = False
                break
            draw = rng.choice(vals, size=vals.size, replace=True)
            probs.append(float(draw.mean()))
        if not valid:
            continue
        estimate = fit_logistic_threshold(alphas, probs, threshold=threshold)["alpha_c"]
        if estimate is not None and np.isfinite(estimate):
            samples.append(float(estimate))
    if not samples:
        return {"ci_low": None, "ci_high": None, "n_success": 0}
    arr = np.array(samples, dtype=float)
    return {
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "n_success": int(arr.size),
    }
