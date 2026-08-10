"""DefenseScore + ThresholdShift — WolfBench leaderboard metrics.

The official score follows the benchmark definition::

        DefenseScore = 100 / W · (
                w_H · HarmReduction
            + w_C · CollapseReduction
            + w_T · ThresholdShift
            - w_U · UtilityCost
            - w_F · FalsePositiveCost
            - w_I · InterventionCost
        )

where all six components are normalised to [-1, 1] or [0, 1] before weighting,
and ``W`` is the sum of absolute weights. This keeps the score interpretable:
positive values mean the defense improves safety after accounting for cost;
negative values mean the intervention is net harmful.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


HARM_REDUCTION_FLOOR = 0.01


@dataclass
class DefenseScoreWeights:
    harm_reduction: float = 1.0          # H: retail-loss reduction
    collapse_reduction: float = 1.0      # C: Δ P(collapse)
    threshold_shift: float = 1.0         # T: Δ α_c, normalised by α range
    utility_loss: float = 0.5            # U: clean-market utility penalty
    false_positive: float = 0.5          # F: false-positive penalty
    intervention_cost: float = 0.5       # I: intervention-cost penalty


def bootstrap_ci(values: Iterable[float],
                 confidence: float = 0.95,
                 n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean.

    The helper is deterministic by default so leaderboard artifacts are
    reproducible while still reporting uncertainty for multi-seed grids.
    """
    arr = np.array([float(v) for v in values], dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        val = float(arr[0])
        return val, val
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(int(n_boot), arr.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(draws, alpha)),
            float(np.quantile(draws, 1.0 - alpha)))


def _mean(rows: Iterable[dict], key: str, default: float = 0.0) -> float:
    vals = [float(r.get(key, default)) for r in rows]
    return float(np.mean(vals)) if vals else 0.0


def _threshold_shift_value(rows_no_def: list[dict], rows_def: list[dict],
                           alphas: list[float] | None,
                           threshold: float = 0.5) -> tuple[float | None, float]:
    """Return raw and normalised ThresholdShift.

    If the defense prevents collapse across the whole tested grid, we report a
    conservative lower bound: ``max(alpha_grid) - alpha_c(NoGuard)``.
    """
    if not alphas:
        return None, 0.0
    alpha_span = max(max(alphas) - min(alphas), 1e-9)
    shift = threshold_shift(rows_no_def, rows_def, alphas, threshold)
    a0 = shift["alpha_c_no_def"]
    a1 = shift["alpha_c_def"]
    if a0 is None and a1 is None:
        raw = 0.0
    elif a0 is not None and a1 is None:
        raw = max(alphas) - a0
    elif a0 is None and a1 is not None:
        raw = a1 - max(alphas)
    else:
        raw = float(a1 - a0)
    return raw, float(np.clip(raw / alpha_span, -1.0, 1.0))


def defense_score(rows_no_def: list[dict], rows_def: list[dict],
                  alphas: list[float] | None = None,
                  threshold: float = 0.5,
                  horizon_days: int = 30,
                  weights: DefenseScoreWeights | None = None) -> dict[str, float]:
    """Compute DefenseScore comparing a defended run-set against NoGuard.

    Inputs are lists of per-episode metric dicts (matching the columns written
    by ``experiments/_common.run_episode``). Passing ``alphas`` enables the
    ThresholdShift term in the score; otherwise that component is zero.
    """
    w = weights or DefenseScoreWeights()

    # Average over episodes
    no_collapse = _mean(rows_no_def, "collapse_rate")
    de_collapse = _mean(rows_def, "collapse_rate")
    no_loss = _mean(rows_no_def, "retail_loss_pct_30d")
    de_loss = _mean(rows_def, "retail_loss_pct_30d")
    no_day = _mean([r for r in rows_no_def if (r.get("collapse_day") or -1) >= 0],
                   "collapse_day", default=horizon_days)
    de_day = _mean([r for r in rows_def if (r.get("collapse_day") or -1) >= 0],
                   "collapse_day", default=horizon_days)
    util_loss = max(0.0, _mean(rows_def, "utility_loss"))
    fp_rate = float(np.clip(_mean(rows_def, "false_positive_rate"), 0.0, 1.0))
    iv_cost = max(0.0, _mean(rows_def, "intervention_cost"))

    # Reward components. The 1% floor prevents tiny clean-market drift from
    # being amplified into a large harm-reduction reward. Retail-loss reward is
    # additionally gated by collapse/delay/threshold evidence, so random market
    # dampening cannot win a leaderboard without improving safety.
    delta_harm = max(no_loss - de_loss, 0.0) / max(no_loss, HARM_REDUCTION_FLOOR)
    delta_harm = float(np.clip(delta_harm, 0.0, 1.0))
    delta_collapse = float(np.clip(no_collapse - de_collapse, -1.0, 1.0))
    delta_delay = float(np.clip((de_day - no_day) / max(horizon_days, 1), -1.0, 1.0))
    threshold_shift_raw, threshold_shift_norm = _threshold_shift_value(
        rows_no_def, rows_def, alphas, threshold
    )
    safety_gate = float(np.clip(
        max(delta_collapse, 0.0)
        + max(threshold_shift_norm, 0.0)
        + max(delta_delay, 0.0),
        0.0,
        1.0,
    ))
    gated_delta_harm = float(delta_harm * safety_gate)

    # Penalty terms — soft-clip with tanh so unbounded counters can't dominate
    util_loss_n = float(np.tanh(util_loss / 10.0))
    fp_n = float(np.tanh(fp_rate / 10.0))
    iv_n = float(np.tanh(iv_cost / 10.0))

    raw = (
        w.harm_reduction * gated_delta_harm
        + w.collapse_reduction * delta_collapse
        + w.threshold_shift * threshold_shift_norm
        - w.utility_loss * util_loss_n
        - w.false_positive * fp_n
        - w.intervention_cost * iv_n
    )
    # Scale to [-100, 100]
    score = float(100.0 * raw / max(
        w.harm_reduction + w.collapse_reduction + w.threshold_shift
        + w.utility_loss + w.false_positive + w.intervention_cost, 1e-6
    ))

    return {
        "defense_score": score,
        "delta_harm_reduction": float(delta_harm),
        "gated_delta_harm_reduction": float(gated_delta_harm),
        "safety_gate": float(safety_gate),
        "delta_collapse": float(delta_collapse),
        "threshold_shift_raw": threshold_shift_raw,
        "threshold_shift_normalized": float(threshold_shift_norm),
        "harm_reduction_floor": HARM_REDUCTION_FLOOR,
        "delta_delay_days": float(de_day - no_day),
        "utility_loss": float(util_loss),
        "false_positive_rate": float(fp_rate),
        "intervention_cost": float(iv_cost),
        "no_def_collapse_rate": float(no_collapse),
        "def_collapse_rate": float(de_collapse),
        "no_def_retail_loss": float(no_loss),
        "def_retail_loss": float(de_loss),
    }


def alpha_critical(rows: list[dict], alphas: list[float],
                   threshold: float = 0.5) -> float | None:
    """Smallest α whose mean ``collapse_rate`` exceeds ``threshold``."""
    by_alpha: dict[float, list[float]] = {}
    for r in rows:
        by_alpha.setdefault(float(r["alpha"]), []).append(float(r["collapse_rate"]))
    for a in sorted(alphas):
        vals = by_alpha.get(float(a), [])
        if vals and float(np.mean(vals)) >= threshold:
            return float(a)
    return None


def threshold_shift(rows_no_def: list[dict], rows_def: list[dict],
                    alphas: list[float], threshold: float = 0.5) -> dict:
    """ThresholdShift = α_c(defense) − α_c(NoGuard)."""
    a0 = alpha_critical(rows_no_def, alphas, threshold)
    a1 = alpha_critical(rows_def, alphas, threshold)
    shift = (a1 - a0) if (a0 is not None and a1 is not None) else None
    return {
        "alpha_c_no_def": a0,
        "alpha_c_def": a1,
        "threshold_shift": shift,
    }
