"""Base agent abstractions used by every WolfBench scenario."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from wolfbench.env.market import Order
from wolfbench.env.social import Message


@dataclass
class Portfolio:
    cash: float = 0.0
    holdings: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0
    initial_wealth: float = 0.0

    def position(self, asset: str) -> float:
        return self.holdings.get(asset, 0.0)

    def mark_to_market(self, prices: dict[str, float]) -> float:
        eq = self.cash
        for a, q in self.holdings.items():
            eq += q * prices.get(a, 0.0)
        return eq


@dataclass
class AgentBase:
    agent_id: str
    role: str                     # "retail_*", "harmful_*", "market_maker", "bot", ...
    is_harmful: bool = False
    portfolio: Portfolio = field(default_factory=Portfolio)
    rng: np.random.Generator | None = None


class AgentProtocol(Protocol):
    agent_id: str
    role: str
    is_harmful: bool
    portfolio: Portfolio

    def decide(self, day: int, observation: dict) -> tuple[list[Order], list[Message]]:
        ...
