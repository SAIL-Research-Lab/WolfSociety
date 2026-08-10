"""Rule-based attacker policies for S1–S4 plus a Random/None baseline.

Every attacker exposes ``decide(day, observation) -> (orders, messages)`` so
swapping in an LLM strategist is a drop-in replacement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wolfbench.agents.base import AgentBase, Portfolio
from wolfbench.env.market import Order
from wolfbench.env.social import Message


# --------------------------------------------------------------------- S1
@dataclass
class PumpAndDumpLeader(AgentBase):
    target_asset: str = "asset_2"
    accumulate_days: tuple[int, int] = (0, 4)
    promote_days: tuple[int, int] = (5, 14)
    dump_days: tuple[int, int] = (15, 22)
    target_inventory_share: float = 0.2
    promote_intensity: float = 1.5
    dump_speed: float = 0.3
    coordinated_seed: int = 0   # shared across cell of leaders for synchrony

    def decide(self, day: int, observation: dict) -> tuple[list[Order], list[Message]]:
        orders, messages = [], []
        a = self.target_asset
        market = observation["market"][a]
        price = market["price"]
        equity = self.portfolio.mark_to_market(observation["prices"])
        target_qty = self.target_inventory_share * equity / max(price, 1e-6)
        held = self.portfolio.position(a)

        if self.accumulate_days[0] <= day <= self.accumulate_days[1]:
            need = max(0.0, target_qty - held)
            if need > 0:
                qty = need / max(self.accumulate_days[1] - day + 1, 1)
                orders.append(Order(self.agent_id, a, "buy", qty, is_harmful=True))
        if self.promote_days[0] <= day <= self.promote_days[1]:
            messages.append(Message(self.agent_id, a, sentiment=+1.0,
                                    intensity=self.promote_intensity,
                                    is_harmful=True, day=day))
        if self.dump_days[0] <= day <= self.dump_days[1]:
            qty = max(0.0, held * self.dump_speed)
            if qty > 1e-6:
                orders.append(Order(self.agent_id, a, "sell", qty, is_harmful=True))
        return orders, messages


@dataclass
class BotAmplifier(AgentBase):
    target_asset: str = "asset_2"
    promote_days: tuple[int, int] = (5, 14)
    intensity: float = 1.0

    def decide(self, day: int, observation: dict) -> tuple[list[Order], list[Message]]:
        msgs: list[Message] = []
        if self.promote_days[0] <= day <= self.promote_days[1]:
            # bots reshare loud bullish messages
            msgs.append(Message(self.agent_id, self.target_asset, sentiment=+1.0,
                                intensity=self.intensity, is_harmful=True,
                                is_bot=True, day=day))
        return [], msgs


@dataclass
class CoordinatedTrader(AgentBase):
    """Lightweight harmful trading account: follows the strategist's
    accumulate/promote/dump schedule but does NOT post messages. Used to
    populate the "trading manipulator" bucket of the harmful population."""
    target_asset: str = "asset_2"
    accumulate_days: tuple[int, int] = (0, 4)
    promote_days: tuple[int, int] = (5, 14)
    dump_days: tuple[int, int] = (15, 22)
    target_inventory_share: float = 0.25
    dump_speed: float = 0.3

    def decide(self, day: int, observation: dict) -> tuple[list[Order], list[Message]]:
        orders: list[Order] = []
        a = self.target_asset
        equity = self.portfolio.mark_to_market(observation["prices"])
        target_qty = self.target_inventory_share * equity / max(observation["prices"][a], 1e-6)
        held = self.portfolio.position(a)
        if self.accumulate_days[0] <= day <= self.accumulate_days[1]:
            need = max(0.0, target_qty - held)
            if need > 0:
                qty = need / max(self.accumulate_days[1] - day + 1, 1)
                orders.append(Order(self.agent_id, a, "buy", qty, is_harmful=True))
        elif self.promote_days[0] <= day <= self.promote_days[1]:
            # ride the pump
            need = max(0.0, target_qty - held)
            if need > 0:
                orders.append(Order(self.agent_id, a, "buy", need * 0.2, is_harmful=True))
        elif self.dump_days[0] <= day <= self.dump_days[1]:
            qty = max(0.0, held * self.dump_speed)
            if qty > 1e-6:
                orders.append(Order(self.agent_id, a, "sell", qty, is_harmful=True))
        return orders, []


# --------------------------------------------------------------------- S2
@dataclass
class Finfluencer(AgentBase):
    target_asset: str = "asset_2"
    accumulate_days: tuple[int, int] = (0, 3)
    promote_days: tuple[int, int] = (4, 18)
    sell_days: tuple[int, int] = (12, 25)
    target_inventory_share: float = 0.2
    post_intensity: float = 1.8

    def decide(self, day: int, observation: dict) -> tuple[list[Order], list[Message]]:
        orders, messages = [], []
        a = self.target_asset
        equity = self.portfolio.mark_to_market(observation["prices"])
        target_qty = self.target_inventory_share * equity / max(observation["prices"][a], 1e-6)
        held = self.portfolio.position(a)

        if self.accumulate_days[0] <= day <= self.accumulate_days[1]:
            need = max(0.0, target_qty - held)
            qty = need / max(self.accumulate_days[1] - day + 1, 1)
            if qty > 1e-6:
                orders.append(Order(self.agent_id, a, "buy", qty, is_harmful=True))
        if self.promote_days[0] <= day <= self.promote_days[1]:
            # influencer posts feed directly to followers via graph proximity
            messages.append(Message(self.agent_id, a, sentiment=+1.0,
                                    intensity=self.post_intensity,
                                    is_harmful=True, day=day))
        if self.sell_days[0] <= day <= self.sell_days[1]:
            # leak inventory slowly into follower demand
            qty = held * 0.15
            if qty > 1e-6:
                orders.append(Order(self.agent_id, a, "sell", qty, is_harmful=True))
        return orders, messages


# --------------------------------------------------------------------- S3
@dataclass
class Spoofer(AgentBase):
    target_asset: str = "asset_1"
    spoof_size_mult: float = 6.0
    cancel_latency_steps: int = 1
    daily_cycles: int = 4
    median_size: float = 50.0
    side_buy_prob: float = 0.5

    def decide(self, day: int, observation: dict) -> tuple[list[Order], list[Message]]:
        orders: list[Order] = []
        a = self.target_asset
        for cycle in range(self.daily_cycles):
            assert self.rng is not None
            spoof_side = "buy" if self.rng.random() < self.side_buy_prob else "sell"
            real_side = "sell" if spoof_side == "buy" else "buy"
            spoof_qty = self.median_size * self.spoof_size_mult
            real_qty = self.median_size * 0.5
            orders.append(Order(self.agent_id, a, spoof_side, spoof_qty,
                                is_spoof=True,
                                cancel_after_steps=self.cancel_latency_steps,
                                is_harmful=True))
            orders.append(Order(self.agent_id, a, real_side, real_qty,
                                is_harmful=True))
        return orders, []


# --------------------------------------------------------------------- S4
@dataclass
class WashTrader(AgentBase):
    target_asset: str = "asset_2"
    counterparty_id: str = ""
    accumulate_days: tuple[int, int] = (0, 4)
    wash_days: tuple[int, int] = (5, 18)
    withdraw_days: tuple[int, int] = (19, 25)
    wash_volume_multiplier: float = 4.0

    def decide(self, day: int, observation: dict) -> tuple[list[Order], list[Message]]:
        orders: list[Order] = []
        a = self.target_asset
        market = observation["market"][a]
        price = market["price"]
        equity = self.portfolio.mark_to_market(observation["prices"])
        held = self.portfolio.position(a)

        if self.accumulate_days[0] <= day <= self.accumulate_days[1]:
            qty = (0.05 * equity) / max(price, 1e-6)
            orders.append(Order(self.agent_id, a, "buy", qty, is_harmful=True))
        if self.wash_days[0] <= day <= self.wash_days[1]:
            base_volume = max(market.get("real_volume", 0.0), 100.0)
            wash_qty = self.wash_volume_multiplier * base_volume * 0.05
            orders.append(Order(self.agent_id, a, "buy", wash_qty,
                                is_wash=True, counterparty_id=self.counterparty_id,
                                is_harmful=True))
        if self.withdraw_days[0] <= day <= self.withdraw_days[1]:
            qty = held * 0.2
            if qty > 1e-6:
                orders.append(Order(self.agent_id, a, "sell", qty, is_harmful=True))
        return orders, []
