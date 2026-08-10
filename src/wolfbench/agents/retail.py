"""Legacy score-controller versions of the five plain-language retail roles.

These are intentionally NOT LLM agents — they scale to N=100k. Their daily
decision combines fundamental gap, recent return, social exposure and a
warning channel that WolfGuard can drive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wolfbench.agents.base import AgentBase, Portfolio
from wolfbench.env.market import Order
from wolfbench.env.social import Message


RETAIL_ROLES = (
    "risk_averse",
    "value_investor",
    "trend_follower",
    "social_follower",
    "aggressive_trader",
)


@dataclass
class RetailAgent(AgentBase):
    sub_role: str = "value_investor"
    beta_social: float = 0.6
    beta_fundamental: float = 0.8
    beta_momentum: float = 0.6
    beta_warning: float = 1.2
    beta_volume: float = 0.4
    beta_imbalance: float = 0.0   # only meaningful in S3
    risk_appetite: float = 0.02   # fraction of wealth per trade
    skepticism: float = 0.0
    momentum_window: int = 3

    # WolfGuard channels (set per-asset by defense each day)
    warning_level: dict[str, float] = field(default_factory=dict)
    cooldown_until: dict[str, int] = field(default_factory=dict)
    blocked_today: dict[str, bool] = field(default_factory=dict)
    # bookkeeping for ErrTrade definition
    last_belief_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)

    def decide(self, day: int, observation: dict) -> tuple[list[Order], list[Message]]:
        orders: list[Order] = []
        messages: list[Message] = []
        equity = self.portfolio.mark_to_market(observation["prices"])
        if equity <= 0:
            return orders, messages

        for asset, market in observation["market"].items():
            if self.cooldown_until.get(asset, -1) >= day:
                continue
            if self.blocked_today.get(asset, False):
                continue

            price = market["price"]
            fundamental = market["fundamental"]
            ret_recent = observation["recent_return"].get(asset, 0.0)
            vol_z = observation["volume_z"].get(asset, 0.0)
            depth_imb = market.get("depth_imbalance", 0.0)
            warning = self.warning_level.get(asset, 0.0)
            social_env = observation.get("social_env")
            if social_env is not None:
                social_signed, harmful_exp = social_env.agent_signal(self.agent_id, asset)
            else:
                social_signed, harmful_exp = 0.0, 0.0

            # role-specific weighting
            w_s, w_f, w_m, w_v, w_imb = self._role_weights()

            social_score = w_s * self.beta_social * np.tanh(social_signed)
            fundamental_score = w_f * self.beta_fundamental * \
                np.tanh((fundamental - price) / max(price, 1e-6))
            momentum_score = w_m * self.beta_momentum * np.tanh(5.0 * ret_recent)
            volume_score = w_v * self.beta_volume * np.tanh(vol_z)
            imbalance_score = w_imb * self.beta_imbalance * depth_imb
            warning_pen = self.beta_warning * warning

            belief = social_score + fundamental_score + momentum_score + volume_score + imbalance_score - warning_pen
            # Risk-averse agents discount unverified social evidence.
            belief -= self.skepticism * abs(social_score)

            # Aggressive traders react more strongly to noisy salient cues.
            if self.sub_role == "aggressive_trader" and self.rng is not None:
                belief += float(self.rng.normal(0.0, 0.65))

            self.last_belief_breakdown[asset] = {
                "social": float(social_score),
                "fundamental": float(fundamental_score),
                "momentum": float(momentum_score),
                "volume": float(volume_score),
                "imbalance": float(imbalance_score),
                "warning": float(warning_pen),
                "harmful_exposure": float(harmful_exp),
                "total": float(belief),
            }

            p_trade = 1.0 / (1.0 + np.exp(-belief))
            assert self.rng is not None
            # skip if too indifferent (sigmoid near 0.5)
            if abs(p_trade - 0.5) < 0.05:
                continue
            if self.rng.random() > min(1.0, abs(p_trade - 0.5) * 3.0):
                continue   # stochastic activation

            qty_cash = self.risk_appetite * equity
            qty = qty_cash / max(price, 1e-6)
            if qty <= 0:
                continue
            side = "buy" if belief > 0 else "sell"
            if side == "sell":
                qty = min(qty, self.portfolio.position(asset))
                if qty <= 1e-6:
                    continue
            orders.append(Order(self.agent_id, asset, side, qty, is_harmful=False))
        # reset per-day flags
        self.blocked_today = {}
        return orders, messages

    def _role_weights(self) -> tuple[float, float, float, float, float]:
        # (social, fundamental, momentum, volume, imbalance)
        if self.sub_role == "risk_averse":
            return (0.25, 1.4, 0.2, 0.15, 0.1)
        if self.sub_role == "value_investor":
            return (0.3, 1.5, 0.3, 0.2, 0.1)
        if self.sub_role == "trend_follower":
            return (0.5, 0.4, 1.6, 0.6, 0.3)
        if self.sub_role == "social_follower":
            return (1.5, 0.2, 1.0, 1.2, 0.2)
        if self.sub_role == "aggressive_trader":
            return (0.8, 0.3, 1.0, 0.8, 0.3)
        return (0.5, 0.5, 0.5, 0.5, 0.1)


def build_retail_agents(n: int, scenario, rng) -> list[RetailAgent]:
    composition = scenario.retail["composition"]
    roles = list(composition.keys())
    weights = np.array([composition[r] for r in roles], dtype=float)
    weights = weights / weights.sum()
    role_assign = rng.choice(roles, size=n, p=weights)
    initial_wealth = float(scenario.retail["initial_wealth"])

    agents: list[RetailAgent] = []
    for i, role in enumerate(role_assign):
        a = RetailAgent(
            agent_id=f"retail_{i:06d}",
            role=f"retail_{role}",
            sub_role=str(role),
            is_harmful=False,
            portfolio=Portfolio(cash=initial_wealth, initial_wealth=initial_wealth),
            beta_social=float(scenario.retail.get("beta_social", 0.6)),
            beta_fundamental=float(scenario.retail.get("beta_fundamental", 0.8)),
            beta_momentum=float(scenario.retail.get("beta_momentum", 0.6)),
            beta_warning=float(scenario.retail.get("beta_warning", 1.2)),
            beta_volume=float(scenario.retail.get("beta_volume", 0.4)),
            beta_imbalance=float(scenario.retail.get("beta_imbalance", 0.0)),
            risk_appetite=float(scenario.retail.get("risk_appetite", 0.02)),
            skepticism=0.75 if role == "risk_averse" else 0.0,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )
        agents.append(a)
    return agents
