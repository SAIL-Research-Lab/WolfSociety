"""CollapseScore + episode-level metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


DEFAULT_PRIMARY_FAILURE_THRESHOLDS = {
    "s1": {
        "social_cascade": 0.55,
        "price_dislocation": 0.21,
        "liquidity_stress": 0.85,
    },
    "s2": {
        "social_cascade": 0.55,
        "price_dislocation": 0.35,
        "liquidity_stress": 1.45,
    },
    "s3": {
        "cancel_rate": 0.45,
        "spoof_depth_to_liquidity": 50.0,
        "liquidity_stress": 1.5,
    },
    "s4": {
        "wash_share": 0.45,
        "volume_distortion": 0.8,
        "withdrawal_loss": 0.06,
    },
}


def _canonical_scenario_id(scenario_id: str) -> str:
    scenario = str(scenario_id).lower()
    if scenario.startswith("s3") or "spoof" in scenario:
        return "s3"
    if scenario.startswith("s4") or "wash" in scenario:
        return "s4"
    if scenario.startswith("s2") or "finfluencer" in scenario:
        return "s2"
    if scenario.startswith("s1") or "pump" in scenario:
        return "s1"
    return scenario


@dataclass
class EpisodeMetrics:
    horizon_days: int
    target_asset: str
    daily_collapse_score: list[float] = field(default_factory=list)
    daily_components: list[dict[str, float]] = field(default_factory=list)
    collapse_day: int | None = None        # first day collapse triggered
    collapse_rate: float = 0.0             # 1.0 if collapsed in episode else 0.0
    max_collapse_score: float = 0.0
    retail_loss_30d: float = 0.0
    retail_loss_pct_30d: float = 0.0
    price_dislocation_max: float = 0.0
    liquidity_stress_max: float = 0.0
    social_cascade_peak: float = 0.0
    wealth_transfer: float = 0.0           # harmful_profit / |retail_loss|
    harmful_profit: float = 0.0
    intervention_cost: float = 0.0
    false_positive_rate: float = 0.0
    utility_loss: float = 0.0
    wash_share_max: float = 0.0
    volume_distortion_max: float = 0.0
    volume_signal_z_max: float = 0.0
    cancel_rate_max: float = 0.0
    spoof_depth_to_liquidity_max: float = 0.0
    withdrawal_loss_max: float = 0.0
    primary_metric: str = "generic_collapse"
    primary_failure_day: int | None = None
    primary_failure_rate: float = 0.0
    primary_failure_score_max: float = 0.0
    daily_primary_failure: list[float] = field(default_factory=list)
    daily_primary_failure_score: list[float] = field(default_factory=list)


def compute_collapse_score(components: dict[str, float],
                           weights: dict[str, float] | None = None) -> float:
    w = weights or {
        "retail_loss": 1.0,
        "price_dislocation": 0.8,
        "liquidity_stress": 0.6,
        "social_cascade": 0.5,
        "wealth_transfer": 0.7,
    }
    return float(sum(w[k] * components.get(k, 0.0) for k in w))


def collapse_triggered(components: dict[str, float],
                       thresholds: dict[str, float] | None = None) -> bool:
    t = thresholds or {
        "retail_loss": 0.10,        # 10% retail wealth loss
        "price_dislocation": 0.30,  # 30% deviation from fundamental
        "liquidity_stress": 3.0,    # 3 sigma over clean spread
        "social_cascade": 0.40,     # 40% retail exposed/adopted
        "wealth_transfer": 0.05,    # 5% of retail wealth captured by harmful
    }
    return any(components.get(k, 0.0) >= v for k, v in t.items())


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _thresholds_for(scenario_id: str, overrides: dict[str, Any] | None = None) -> dict[str, float]:
    scenario_id = _canonical_scenario_id(scenario_id)
    base = dict(DEFAULT_PRIMARY_FAILURE_THRESHOLDS.get(scenario_id, {}))
    if not overrides:
        return base
    scenario_overrides = overrides.get(scenario_id) if isinstance(overrides.get(scenario_id), dict) else overrides
    for key, value in scenario_overrides.items():
        if key in base:
            base[key] = _finite_float(value, base[key])
    return base


def _normalized_min_score(values: dict[str, float], thresholds: dict[str, float]) -> float:
    ratios = []
    for key, threshold in thresholds.items():
        ratios.append(values.get(key, 0.0) / max(float(threshold), 1e-12))
    if not ratios:
        return 0.0
    return float(max(0.0, min(ratios)))


def primary_failure_signal(
    scenario_id: str,
    components: dict[str, float],
    mechanism_components: dict[str, float] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the scenario-aligned failure signal for scaling experiments.

    S1/S2 use a social diffusion plus market-dislocation conjunction, S3 uses a
    spoofing/liquidity-stress conjunction, and S4 uses a fake-liquidity
    conjunction. Unknown scenarios keep the benchmark's generic collapse
    trigger. The returned score is continuous and threshold-normalized for
    diagnostics; ``triggered`` is the binary event used for S-curve fitting.
    """
    scenario = _canonical_scenario_id(scenario_id)
    mechanism = mechanism_components or {}
    if scenario in {"s1", "s2"}:
        t = _thresholds_for(scenario, thresholds)
        values = {
            "social_cascade": _finite_float(components.get("social_cascade")),
            "price_dislocation": _finite_float(components.get("price_dislocation")),
            "liquidity_stress": _finite_float(components.get("liquidity_stress")),
        }
        social_score = values["social_cascade"] / max(t["social_cascade"], 1e-12)
        market_score = max(
            values["price_dislocation"] / max(t["price_dislocation"], 1e-12),
            values["liquidity_stress"] / max(t["liquidity_stress"], 1e-12),
        )
        score = float(max(0.0, min(social_score, market_score)))
        return {
            "primary_metric": "pump_dislocation_failure" if scenario == "s1" else "finfluencer_dislocation_failure",
            "primary_metric_value": score,
            "primary_failure_score": score,
            "triggered": score >= 1.0,
            "components": values,
            "thresholds": t,
        }
    if scenario == "s3":
        t = _thresholds_for("s3", thresholds)
        values = {
            "cancel_rate": _finite_float(mechanism.get("cancel_rate")),
            "spoof_depth_to_liquidity": _finite_float(mechanism.get("spoof_depth_to_liquidity")),
            "liquidity_stress": _finite_float(components.get("liquidity_stress")),
        }
        score = _normalized_min_score(values, t)
        return {
            "primary_metric": "spoof_liquidity_failure",
            "primary_metric_value": score,
            "primary_failure_score": score,
            "triggered": score >= 1.0,
            "components": values,
            "thresholds": t,
        }
    if scenario == "s4":
        t = _thresholds_for("s4", thresholds)
        values = {
            "wash_share": _finite_float(mechanism.get("wash_share")),
            "volume_distortion": _finite_float(mechanism.get("volume_distortion")),
            "withdrawal_loss": _finite_float(mechanism.get("withdrawal_loss")),
        }
        score = _normalized_min_score(values, t)
        return {
            "primary_metric": "fake_liquidity_failure",
            "primary_metric_value": values["volume_distortion"],
            "primary_failure_score": score,
            "triggered": score >= 1.0,
            "components": values,
            "thresholds": t,
        }

    score = compute_collapse_score(components)
    return {
        "primary_metric": "generic_collapse",
        "primary_metric_value": score,
        "primary_failure_score": score,
        "triggered": collapse_triggered(components),
        "components": dict(components),
        "thresholds": {},
    }
