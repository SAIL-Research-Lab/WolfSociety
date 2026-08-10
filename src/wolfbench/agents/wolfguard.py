"""WolfGuard — Early Malicious Capture Agent.

Computes per-asset Risk_a(t) from a system summary and issues warning /
cooldown / block actions against high-risk erroneous trades.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WolfGuardConfig:
    enabled: bool = True
    theta_social: float = 1.0
    theta_volume: float = 1.0
    theta_price: float = 1.0
    theta_liquidity: float = 0.7
    theta_coord: float = 1.0
    theta_micro: float = 1.2
    theta_disclosure: float = 1.0
    risk_warning: float = 0.5
    risk_cooldown: float = 0.7
    risk_block: float = 0.85
    err_trade_exposure_threshold: float = 0.3
    intervention_cost_warning: float = 0.01
    intervention_cost_cooldown: float = 0.05
    intervention_cost_block: float = 0.10
    mode: str = "full"          # "warning" / "cooldown" / "block" / "full" / "off"


@dataclass
class WolfGuardAgent:
    agent_id: str = "wolfguard"
    role: str = "defense"
    is_harmful: bool = False
    config: WolfGuardConfig = field(default_factory=WolfGuardConfig)
    clean_baseline: dict[str, dict[str, float]] = field(default_factory=dict)

    def fit_baseline(self, baseline: dict[str, dict[str, float]]) -> None:
        """Take per-asset clean-market means/stds, used to z-score signals."""
        self.clean_baseline = baseline

    def _z(self, asset: str, key: str, value: float) -> float:
        b = self.clean_baseline.get(asset, {})
        mu = b.get(f"{key}_mu", 0.0)
        sd = b.get(f"{key}_sd", 1.0)
        if mu is None or not np.isfinite(mu):
            mu = 0.0
        if sd is None or not np.isfinite(sd) or sd <= 0.0:
            sd = 1.0
        return float((value - mu) / sd)

    def risk_score(self, asset: str, market: dict, social: dict) -> dict[str, float]:
        z_social = self._z(asset, "msg_volume", social.get("msg_volume", 0.0))
        z_volume = self._z(asset, "volume", market.get("volume", 0.0))
        gap = (market["price"] - market["fundamental"]) / max(market["fundamental"], 1e-6)
        z_price = self._z(asset, "price_gap", gap)
        z_liquidity = self._z(asset, "spread_bps", market.get("spread_bps", 0.0))
        z_coord = social.get("harmful_msg_share", 0.0)  # already a [0,1] proxy
        z_micro = abs(market.get("depth_imbalance", 0.0)) * (1.0 + market.get("cancel_rate", 0.0)) \
                  + market.get("wash_share", 0.0)
        # disclosure not implemented; placeholder 0
        disclosure = 0.0
        c = self.config
        s = (
            c.theta_social * max(z_social, 0.0)
            + c.theta_volume * max(z_volume, 0.0)
            + c.theta_price * max(z_price, 0.0)
            + c.theta_liquidity * max(z_liquidity, 0.0)
            + c.theta_coord * z_coord
            + c.theta_micro * z_micro
            - c.theta_disclosure * disclosure
        )
        risk = float(1.0 / (1.0 + np.exp(-s + 2.0)))
        return {
            "risk": risk,
            "z_social": z_social,
            "z_volume": z_volume,
            "z_price": z_price,
            "z_liquidity": z_liquidity,
            "z_coord": z_coord,
            "z_micro": z_micro,
        }

    def decide(self, day: int, system_summary: dict) -> dict[str, dict]:
        """Return per-asset action dict."""
        if not self.config.enabled or self.config.mode == "off":
            return {}
        out: dict[str, dict] = {}
        for asset, market in system_summary["market"].items():
            social = system_summary["social"].get(asset, {})
            r = self.risk_score(asset, market, social)
            risk = r["risk"]
            action = "none"
            if self.config.mode in ("full", "warning") and risk >= self.config.risk_warning:
                action = "warning"
            if self.config.mode in ("full", "cooldown") and risk >= self.config.risk_cooldown:
                action = "cooldown"
            if self.config.mode in ("full", "block") and risk >= self.config.risk_block:
                action = "block"
            out[asset] = {
                "asset": asset,
                "action": action,
                "risk": risk,
                "components": r,
            }
        return out
