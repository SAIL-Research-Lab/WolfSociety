"""Threshold Protection Score for WolfBench defense leaderboards.

TPS is the paper-facing official score. It focuses on the near-critical alpha
band of the NoGuard primary-failure curve and keeps adverse behavior in
diagnostic fields rather than making the main leaderboard negative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np

_CURVE_FIT = None
_CURVE_FIT_CHECKED = False


def expit(x):
    arr = np.array(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def _get_curve_fit():
    """Import SciPy only when TPS fitting is requested.

    Some local Python environments have an ABI-broken SciPy build. Importing it
    at package load time prints noisy binary errors even when users only run
    episodes. Keeping it lazy preserves the fitter when SciPy is healthy and
    uses a deterministic linear fallback otherwise.
    """
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


DEFAULT_HORIZON_DAYS = 30
DEFAULT_NUM_ASSETS = 5
DEFAULT_BLOCK_COST = 0.10
EPS = 1e-9


@dataclass(frozen=True)
class TPSConfig:
    threshold: float = 0.5
    lower_band: float = 0.2
    upper_band: float = 0.8
    shift_weight: float = 0.55
    critical_collapse_weight: float = 0.35
    damage_weight: float = 0.10
    clean_cost_budget: float = 0.05
    false_positive_budget: float = 0.10
    intervention_cost_budget: float = 0.12
    intervention_cost_weight: float = 0.5
    horizon_days: int = DEFAULT_HORIZON_DAYS
    num_assets: int = DEFAULT_NUM_ASSETS
    block_cost: float = DEFAULT_BLOCK_COST
    probability_key: str = "primary_failure_rate"
    fallback_probability_key: str = "collapse_rate"
    severity_key: str = "primary_failure_score_max"
    secondary_severity_key: str = "max_collapse_score"
    fallback_severity_key: str = "retail_loss_pct_30d"
    soft_probability_weight: float = 0.49


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _mean(rows: Iterable[dict], key: str, default: float = 0.0) -> float:
    vals = [_to_float(row.get(key, default), default=default) for row in rows]
    return float(np.mean(vals)) if vals else 0.0


def _mean_if_present(rows: Iterable[dict], key: str) -> float | None:
    vals = []
    for row in rows:
        if key not in row:
            continue
        value = _to_float(row.get(key), default=float("nan"))
        if np.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else None


def _linear_threshold(alphas: Iterable[float], probs: Iterable[float], threshold: float) -> float | None:
    alpha_arr = np.array(list(alphas), dtype=float)
    prob_arr = np.array(list(probs), dtype=float)
    idx = np.where(prob_arr >= threshold)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(alpha_arr[0])
    p0, p1 = float(prob_arr[i - 1]), float(prob_arr[i])
    a0, a1 = float(alpha_arr[i - 1]), float(alpha_arr[i])
    if abs(p1 - p0) <= EPS:
        return a1
    return float(a0 + (threshold - p0) * (a1 - a0) / (p1 - p0))


def _logistic_curve(alpha: np.ndarray, alpha_c: float, slope: float) -> np.ndarray:
    return expit(slope * (alpha - alpha_c))


def _fit_curve(alphas: list[float], probs: list[float], threshold: float) -> dict[str, Any]:
    alpha_arr = np.array(alphas, dtype=float)
    prob_arr = np.clip(np.array(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    linear = _linear_threshold(alpha_arr, prob_arr, threshold)
    if alpha_arr.size < 3 or np.ptp(prob_arr) < 1e-6:
        return {
            "alpha_c": linear,
            "slope": None,
            "fitted_probs": prob_arr.tolist(),
            "method": "linear_fallback_constant_curve",
        }

    alpha0 = float(linear if linear is not None else np.median(alpha_arr))
    span = max(float(alpha_arr.max() - alpha_arr.min()), EPS)
    curve_fit = _get_curve_fit()
    if curve_fit is None:
        return {
            "alpha_c": linear,
            "slope": None,
            "fitted_probs": prob_arr.tolist(),
            "method": "linear_fallback_no_scipy",
        }
    try:
        params, _ = curve_fit(
            _logistic_curve,
            alpha_arr,
            prob_arr,
            p0=[alpha0, 8.0 / span],
            bounds=([float(alpha_arr.min()), 1e-6], [float(alpha_arr.max()), 1e6]),
            maxfev=20000,
        )
        alpha_c, slope = float(params[0]), float(params[1])
        return {
            "alpha_c": alpha_c,
            "slope": slope,
            "fitted_probs": _logistic_curve(alpha_arr, alpha_c, slope).tolist(),
            "method": "logistic_fit",
        }
    except Exception:
        return {
            "alpha_c": linear,
            "slope": None,
            "fitted_probs": prob_arr.tolist(),
            "method": "linear_fallback_fit_failed",
        }


def _prob_by_alpha(rows: list[dict], alphas: list[float], config: TPSConfig) -> list[float]:
    return [
        _effective_prob_for_alpha([row for row in rows if float(row.get("alpha", 0.0)) == float(alpha)], config)
        for alpha in alphas
    ]


def _prob_for_alpha(rows: list[dict], config: TPSConfig) -> float:
    value = _mean_if_present(rows, config.probability_key)
    if value is None and config.probability_key != config.fallback_probability_key:
        value = _mean_if_present(rows, config.fallback_probability_key)
    if value is None:
        value = 0.0
    return float(np.clip(value, 0.0, 1.0))


def _effective_prob_for_alpha(rows: list[dict], config: TPSConfig) -> float:
    """Return a probability-like primary curve point.

    With short two-seed runs, binary failure rates often saturate at all-zero
    or all-one curves. The continuous primary-failure severity is already part
    of the benchmark output, so use it as a sub-threshold soft signal. Hard
    failures still dominate, but non-triggered near misses can break ties.
    """
    hard = _prob_for_alpha(rows, config)
    if hard >= config.threshold:
        return hard
    severity = _mean_if_present(rows, config.severity_key)
    if severity is None:
        return hard
    soft = float(config.soft_probability_weight) * float(np.clip(severity, 0.0, 1.0))
    soft = min(soft, float(config.threshold) - EPS)
    return float(np.clip(max(hard, soft), 0.0, 1.0))


def _severity_by_alpha(rows: list[dict], alphas: list[float], config: TPSConfig) -> dict[float, float]:
    out: dict[float, float] = {}
    for alpha in alphas:
        selected = [row for row in rows if float(row.get("alpha", 0.0)) == float(alpha)]
        value = _mean_if_present(selected, config.severity_key)
        if value is None:
            value = _mean_if_present(selected, config.secondary_severity_key)
        if value is None:
            value = _mean_if_present(selected, config.fallback_severity_key)
        if value is None:
            value = 0.0
        out[float(alpha)] = float(max(value, 0.0))
    return out


def _effective_alpha_c(
    fit: dict[str, Any],
    alphas: list[float],
    probs: list[float],
    threshold: float,
) -> float | None:
    if probs and max(probs) < threshold:
        span = float(max(alphas) - min(alphas)) if alphas else 0.0
        max_prob = float(max(probs))
        return float(max(alphas) + span * (threshold - max_prob) / max(threshold, EPS))
    if probs and min(probs) >= threshold:
        span = float(max(alphas) - min(alphas)) if alphas else 0.0
        min_prob = float(min(probs))
        return float(min(alphas) - span * (min_prob - threshold) / max(1.0 - threshold, EPS))
    alpha_c = fit.get("alpha_c")
    if alpha_c is not None and np.isfinite(float(alpha_c)):
        return float(alpha_c)
    if not alphas:
        return None
    return None


def _width_20_80(fit: dict[str, Any], alphas: list[float], probs: list[float]) -> float:
    slope = fit.get("slope")
    if slope is not None and float(slope) > EPS:
        return float(2.0 * np.log(4.0) / float(slope))
    a20 = _linear_threshold(alphas, probs, 0.2)
    a80 = _linear_threshold(alphas, probs, 0.8)
    if a20 is not None and a80 is not None and a80 > a20:
        return float(a80 - a20)
    span = float(max(alphas) - min(alphas)) if alphas else 0.0
    return max(span / 3.0, EPS)


def _critical_band(alphas: list[float], fitted_probs: list[float], alpha_c: float | None) -> list[float]:
    band = [
        float(alpha)
        for alpha, prob in zip(alphas, fitted_probs)
        if 0.2 <= float(prob) <= 0.8
    ]
    if len(band) >= 3:
        return band
    if not alphas:
        return []
    center = float(alpha_c) if alpha_c is not None and np.isfinite(float(alpha_c)) else float(np.median(alphas))
    ordered = sorted(alphas, key=lambda alpha: abs(float(alpha) - center))
    return sorted(float(alpha) for alpha in ordered[: min(3, len(ordered))])


def _clean_rows(rows: list[dict], alphas: list[float]) -> list[dict]:
    if not rows:
        return []
    clean_alpha = 0.0 if any(abs(float(alpha)) <= EPS for alpha in alphas) else min(alphas, key=lambda alpha: abs(float(alpha)))
    return [row for row in rows if abs(float(row.get("alpha", 0.0)) - float(clean_alpha)) <= EPS]


def threshold_protection_score(
    rows_no_def: list[dict],
    rows_def: list[dict],
    alphas: list[float] | None = None,
    config: TPSConfig | None = None,
) -> dict[str, float | str | list[float] | None]:
    """Compute TPS and RawNet for one scenario/N/defense comparison.

    ``rows_no_def`` and ``rows_def`` are per-episode rows for the same scenario
    and society size. ``alphas`` should be the calibrated alpha grid used for
    both curves.
    """
    cfg = config or TPSConfig()
    alpha_grid = sorted(float(alpha) for alpha in (alphas or sorted({float(r["alpha"]) for r in rows_no_def + rows_def})))
    if not alpha_grid or not rows_no_def or not rows_def:
        return _empty_score()

    p0 = _prob_by_alpha(rows_no_def, alpha_grid, cfg)
    pd = _prob_by_alpha(rows_def, alpha_grid, cfg)
    fit0 = _fit_curve(alpha_grid, p0, cfg.threshold)
    fitd = _fit_curve(alpha_grid, pd, cfg.threshold)
    alpha_c0 = _effective_alpha_c(fit0, alpha_grid, p0, cfg.threshold)
    alpha_cd = _effective_alpha_c(fitd, alpha_grid, pd, cfg.threshold)
    w0 = _width_20_80(fit0, alpha_grid, p0)
    band = _critical_band(alpha_grid, list(fit0["fitted_probs"]), alpha_c0)

    p0_fit = dict(zip(alpha_grid, [float(v) for v in fit0["fitted_probs"]]))
    pd_fit = dict(zip(alpha_grid, [float(v) for v in fitd["fitted_probs"]]))
    if alpha_c0 is None or alpha_cd is None:
        signed_shift = 0.0
    else:
        signed_shift = float((alpha_cd - alpha_c0) / max(w0, EPS))
    shift_score = float(np.clip(signed_shift, 0.0, 1.0))

    band_delta_p = _mean_delta([p0_fit[a] - pd_fit[a] for a in band])
    signed_critical = float(np.clip(band_delta_p / 0.5, -1.0, 1.0))
    critical_reduction = float(np.clip(signed_critical, 0.0, 1.0))

    severity0 = _severity_by_alpha(rows_no_def, alpha_grid, cfg)
    severityd = _severity_by_alpha(rows_def, alpha_grid, cfg)
    severity_base = float(np.mean([severity0[a] for a in band])) if band else 0.0
    severity_delta = float(np.mean([severity0[a] - severityd[a] for a in band])) if band else 0.0
    signed_damage = float(np.clip(severity_delta / max(severity_base, EPS), -1.0, 1.0)) if severity_base > EPS else 0.0
    damage_reduction = float(np.clip(signed_damage, 0.0, 1.0))

    safety_gain = float(
        cfg.shift_weight * shift_score
        + cfg.critical_collapse_weight * critical_reduction
        + cfg.damage_weight * damage_reduction
    )

    clean = _clean_rows(rows_def, alpha_grid)
    denom = max(float(cfg.horizon_days * cfg.num_assets) * cfg.block_cost, EPS)
    clean_utility = max(0.0, _mean(clean, "utility_loss"))
    clean_intervention = max(0.0, _mean(clean, "intervention_cost"))
    false_positive = float(np.clip(_mean(clean, "false_positive_rate"), 0.0, 1.0))
    clean_cost_index = float(clean_utility / denom)
    intervention_cost_index = float(clean_intervention / denom)
    cost_penalty = float(
        max(0.0, clean_cost_index / cfg.clean_cost_budget - 1.0)
        + max(0.0, false_positive / cfg.false_positive_budget - 1.0)
        + cfg.intervention_cost_weight * max(0.0, intervention_cost_index / cfg.intervention_cost_budget - 1.0)
    )
    cost_gate = float(np.exp(-cost_penalty))
    tps = float(100.0 * safety_gain * cost_gate)
    raw_net = float(100.0 * (
        0.45 * np.clip(signed_shift, -1.0, 1.0)
        + 0.35 * signed_critical
        + 0.20 * signed_damage
        - cost_penalty
    ))

    return {
        "tps": float(np.clip(tps, 0.0, 100.0)),
        "raw_net": raw_net,
        "raw_net_defense_score": raw_net,
        "alpha_c_no_def": alpha_c0,
        "alpha_c_def": alpha_cd,
        "delta_alpha_c": None if alpha_c0 is None or alpha_cd is None else float(alpha_cd - alpha_c0),
        "delta_alpha_c_over_w0": signed_shift,
        "shift_score": shift_score,
        "critical_band_delta_p": band_delta_p,
        "signed_critical_collapse_reduction": signed_critical,
        "critical_collapse_reduction": critical_reduction,
        "damage_reduction": damage_reduction,
        "signed_damage_reduction": signed_damage,
        "safety_gain": safety_gain,
        "clean_utility_cost": clean_utility,
        "clean_cost_index": clean_cost_index,
        "false_positive_rate": false_positive,
        "clean_false_positive_rate": false_positive,
        "intervention_cost_index": intervention_cost_index,
        "clean_intervention_cost": clean_intervention,
        "cost_penalty": cost_penalty,
        "cost_gate": cost_gate,
        "w0": w0,
        "critical_band": band,
        "probability_key": cfg.probability_key,
        "fallback_probability_key": cfg.fallback_probability_key,
        "severity_key": cfg.severity_key,
        "soft_probability_weight": cfg.soft_probability_weight,
        "alpha_c_method_no_def": str(fit0["method"]),
        "alpha_c_method_def": str(fitd["method"]),
    }


def _mean_delta(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _empty_score() -> dict[str, float | str | list[float] | None]:
    return {
        "tps": 0.0,
        "raw_net": 0.0,
        "raw_net_defense_score": 0.0,
        "alpha_c_no_def": None,
        "alpha_c_def": None,
        "delta_alpha_c": None,
        "delta_alpha_c_over_w0": 0.0,
        "shift_score": 0.0,
        "critical_band_delta_p": 0.0,
        "signed_critical_collapse_reduction": 0.0,
        "critical_collapse_reduction": 0.0,
        "damage_reduction": 0.0,
        "signed_damage_reduction": 0.0,
        "safety_gain": 0.0,
        "clean_utility_cost": 0.0,
        "clean_cost_index": 0.0,
        "false_positive_rate": 0.0,
        "clean_false_positive_rate": 0.0,
        "intervention_cost_index": 0.0,
        "clean_intervention_cost": 0.0,
        "cost_penalty": 0.0,
        "cost_gate": 1.0,
        "w0": 0.0,
        "critical_band": [],
        "probability_key": TPSConfig.probability_key,
        "fallback_probability_key": TPSConfig.fallback_probability_key,
        "severity_key": TPSConfig.severity_key,
        "alpha_c_method_no_def": "missing",
        "alpha_c_method_def": "missing",
    }
