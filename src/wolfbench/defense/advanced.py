"""Stronger public-signal WolfGuard policies for threshold-shift audits."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wolfbench.agents.wolfguard import WolfGuardAgent, WolfGuardConfig
from wolfbench.defense.distilled import DistilledWolfGuardPolicy
from wolfbench.defense.policy import make_intervention


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _positive_z(agent: WolfGuardAgent, asset: str, key: str, value: float) -> float:
    return max(agent._z(asset, key, value), 0.0)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


@dataclass
class TopologyAwareWolfGuardPolicy(WolfGuardAgent):
    """Public-signal defense tuned for near-critical cascade regimes.

    The policy uses only the official WolfGuard summary. It combines social
    cascade acceleration, volume/price/liquidity z-scores, and recent-return
    feedback. Actions are capped per day and blocks are rare, so the policy is
    designed to test whether targeted warnings/cooldowns can move alpha_c to
    the right without paying the broad cost of the legacy rule baseline.
    """

    name: str = "TopologyAware-WolfGuard"
    config: WolfGuardConfig = field(default_factory=WolfGuardConfig)
    risk_warning: float = 0.56
    risk_cooldown: float = 0.72
    risk_block: float = 0.94
    near_critical_min: float = 0.42
    max_daily_interventions: int = 2
    max_blocks_per_episode: int = 1
    min_day: int = 2
    _blocks_used: int = field(default=0, init=False, repr=False)

    def _components(self, asset: str, summary: dict[str, Any]) -> dict[str, float]:
        market = (summary.get("market", {}) or {}).get(asset, {}) or {}
        social = (summary.get("social", {}) or {}).get(asset, {}) or {}
        recent_return = _float((summary.get("recent_return", {}) or {}).get(asset, 0.0))

        price = _float(market.get("price", 0.0))
        fundamental = max(_float(market.get("fundamental", 1.0), 1.0), 1e-9)
        price_gap = (price - fundamental) / fundamental
        volume = _float(market.get("volume", 0.0))
        spread_bps = _float(market.get("spread_bps", 0.0))
        depth_imbalance = abs(_float(market.get("depth_imbalance", 0.0)))
        cancel_rate = _float(market.get("cancel_rate", 0.0))
        wash_share = _float(market.get("wash_share", 0.0))

        z_msg = _positive_z(self, asset, "msg_volume", _float(social.get("msg_volume", 0.0)))
        z_volume = _positive_z(self, asset, "volume", volume)
        z_price = _positive_z(self, asset, "price_gap", price_gap)
        z_spread = _positive_z(self, asset, "spread_bps", spread_bps)

        social_all = summary.get("social", {}) or {}
        max_cascade = max(
            [_float(s.get("cascade_size", 0.0)) for s in social_all.values()] or [0.0]
        )
        cascade_proxy = _float(social.get("cascade_size", 0.0)) / max(max_cascade, 1.0)
        harmful_share = float(np.clip(_float(social.get("harmful_msg_share", 0.0)), 0.0, 1.0))
        sentiment = max(_float(social.get("sentiment", 0.0)), 0.0)

        social_acceleration = z_msg + 0.85 * cascade_proxy + 1.25 * harmful_share + 0.35 * sentiment
        market_acceleration = z_volume + z_price + 6.0 * max(recent_return, 0.0)
        micro_liquidity = z_spread + depth_imbalance + cancel_rate + wash_share
        feedback_coupling = social_acceleration * (1.0 + 4.0 * max(recent_return, 0.0))
        near_critical_score = (
            0.52 * feedback_coupling
            + 0.32 * market_acceleration
            + 0.22 * micro_liquidity
        )
        risk = _sigmoid(near_critical_score - 2.45)
        return {
            "risk": risk,
            "near_critical_score": float(near_critical_score),
            "social_acceleration": float(social_acceleration),
            "market_acceleration": float(market_acceleration),
            "micro_liquidity": float(micro_liquidity),
            "cascade_proxy": float(cascade_proxy),
            "harmful_msg_share": float(harmful_share),
            "recent_return": float(recent_return),
            "z_msg_volume": float(z_msg),
            "z_volume": float(z_volume),
            "z_price_gap": float(z_price),
            "z_spread_bps": float(z_spread),
        }

    def decide(self, day: int, summary: dict[str, Any]) -> dict[str, dict]:
        if not self.config.enabled or self.config.mode == "off" or day < self.min_day:
            return {}

        candidates: list[tuple[str, str, float, dict[str, float]]] = []
        for asset in (summary.get("market", {}) or {}):
            components = self._components(asset, summary)
            risk = components["risk"]
            if risk < self.near_critical_min:
                continue
            action = "none"
            if risk >= self.risk_warning:
                action = "warning"
            if risk >= self.risk_cooldown:
                action = "cooldown"
            block_signal = (
                risk >= self.risk_block
                and components["social_acceleration"] >= 3.5
                and components["micro_liquidity"] >= 1.0
                and self._blocks_used < self.max_blocks_per_episode
            )
            if block_signal:
                action = "block"
            if action != "none":
                candidates.append((asset, action, risk, components))

        candidates.sort(key=lambda item: item[2], reverse=True)
        out: dict[str, dict] = {}
        for asset, action, risk, components in candidates[: max(self.max_daily_interventions, 0)]:
            if action == "block":
                self._blocks_used += 1
            out[asset] = make_intervention(
                asset,
                action,
                risk=risk,
                reason="topology_aware",
                components=components,
            )
        return out


@dataclass
class CostAwareZScoreWolfGuardPolicy(WolfGuardAgent):
    """Sparse traditional z-score baseline for public leaderboard comparisons.

    This policy intentionally stays simple: it reuses the public WolfGuard
    z-score risk features, delays action until the market has warmed up, caps
    daily interventions, and never reads hidden oracle fields. It is meant as a
    cheap non-LLM baseline that should be hard for generic LLM risk scorers to
    beat on reliability and clean-market cost.
    """

    name: str = "CostAware-ZScore-WolfGuard"
    config: WolfGuardConfig = field(default_factory=WolfGuardConfig)
    risk_warning: float = 0.68
    risk_cooldown: float = 0.84
    max_daily_interventions: int = 1
    min_day: int = 3

    def decide(self, day: int, summary: dict[str, Any]) -> dict[str, dict]:
        if not self.config.enabled or self.config.mode == "off" or day < self.min_day:
            return {}

        candidates: list[tuple[str, str, float, dict[str, float]]] = []
        for asset, market in (summary.get("market", {}) or {}).items():
            social = (summary.get("social", {}) or {}).get(asset, {}) or {}
            components = self.risk_score(asset, market, social)
            risk = float(components["risk"])
            if risk < self.risk_warning:
                continue
            action = "cooldown" if risk >= self.risk_cooldown else "warning"
            candidates.append((asset, action, risk, components))

        candidates.sort(key=lambda item: item[2], reverse=True)
        return {
            asset: make_intervention(
                asset,
                action,
                risk=risk,
                reason="cost_aware_zscore",
                components=components,
            )
            for asset, action, risk, components in candidates[: max(self.max_daily_interventions, 0)]
        }


@dataclass
class CalibratedDistilledWolfGuardPolicy(DistilledWolfGuardPolicy):
    """Distilled-WolfGuard with cost-aware action thresholds.

    The base distilled classifier imitates oracle labels directly. This wrapper
    converts class probabilities into interventions with tunable public-dev
    thresholds, allowing experiments to optimise threshold shift against clean
    utility and false-positive cost.
    """

    name: str = "Calibrated-Distilled-WolfGuard"
    warning_threshold: float = 0.48
    cooldown_threshold: float = 0.66
    block_threshold: float = 0.92
    max_daily_interventions: int = 2

    def _action_from_probs(self, probs: dict[str, float]) -> tuple[str, float]:
        risk = float(np.clip(1.0 - probs.get("none", 0.0), 0.0, 1.0))
        p_warning = probs.get("warning", 0.0)
        p_cooldown = probs.get("cooldown", 0.0)
        p_block = probs.get("block", 0.0)
        if p_block >= self.block_threshold and risk >= self.block_threshold:
            return "block", risk
        if p_cooldown >= self.cooldown_threshold or risk >= self.cooldown_threshold:
            return "cooldown", risk
        if p_warning >= self.warning_threshold or risk >= self.warning_threshold:
            return "warning", risk
        return "none", risk

    def decide(self, day: int, summary: dict[str, Any]) -> dict[str, dict]:
        public_summary = dict(summary)
        public_summary.pop("oracle_view", None)
        candidates: list[tuple[str, str, float, dict[str, float]]] = []
        for asset in public_summary.get("market", {}):
            probs = self._model.predict_proba(public_summary, asset)
            action, risk = self._action_from_probs(probs)
            if action == "none":
                continue
            components = {f"p_{label}": float(prob) for label, prob in probs.items()}
            components["risk"] = risk
            candidates.append((asset, action, risk, components))
        candidates.sort(key=lambda item: item[2], reverse=True)
        return {
            asset: make_intervention(
                asset,
                action,
                risk=risk,
                reason="calibrated_distilled",
                components=components,
            )
            for asset, action, risk, components in candidates[: max(self.max_daily_interventions, 0)]
        }
