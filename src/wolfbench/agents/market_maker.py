"""Stylized market makers: stabilise spread, mean-revert, no LLM."""
from __future__ import annotations

from dataclasses import dataclass

from wolfbench.agents.base import AgentBase, Portfolio
from wolfbench.env.market import Order
from wolfbench.env.social import Message


@dataclass
class MarketMaker(AgentBase):
    inventory_aversion: float = 0.05

    def decide(self, day: int, observation: dict) -> tuple[list[Order], list[Message]]:
        # Market maker fills are handled inside MarketEnv via pooled liquidity.
        # We expose this stub so the loop is uniform.
        return [], []


def build_market_makers(scenario) -> list[MarketMaker]:
    n = int(scenario.market_makers.get("count", 3))
    out: list[MarketMaker] = []
    inv_av = float(scenario.market_makers.get("inventory_aversion", 0.05))
    for i in range(n):
        out.append(MarketMaker(
            agent_id=f"mm_{i}",
            role="market_maker",
            is_harmful=False,
            portfolio=Portfolio(cash=1e9),
            inventory_aversion=inv_av,
        ))
    return out
