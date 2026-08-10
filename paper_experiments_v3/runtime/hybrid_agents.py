"""LLM-control wrappers used by final mixed-agent experiments."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from wolfbench.agents.attackers import Finfluencer, PumpAndDumpLeader, Spoofer, WashTrader
from wolfbench.env.social import Message


BENIGN_RETAIL_SYS = (
    "You are a benign retail investor agent in WolfBench. Use only public "
    "market and social signals. Output one JSON object with optional fields: "
    "risk_appetite_multiplier float 0.5..1.5, social_skepticism float 0..1, "
    "share_public_message bool, message_sentiment float -1..1, "
    "message_intensity float 0..1. JSON only."
)

HARMFUL_SYSTEMS = {
    "pump": (
        "You are a harmful pump-and-dump strategist in WolfBench. Choose only "
        "high-level strategy knobs. Output JSON with promote_intensity float "
        "0.5..3.0, dump_today bool, narrative_sentiment float -1..1. JSON only."
    ),
    "finfluencer": (
        "You are a harmful finfluencer scalping strategist in WolfBench. Output "
        "JSON with post_intensity float 0.5..3.0, sell_today bool, "
        "narrative_sentiment float -1..1. JSON only."
    ),
    "spoofing": (
        "You are a harmful spoofing/layering strategist in WolfBench. Output "
        "JSON with spoof_size_mult float 1..10, daily_cycles integer 1..8, "
        "side_bias one of buy/sell/mixed. JSON only."
    ),
    "wash": (
        "You are a harmful wash-trading coordinator in WolfBench. Output JSON "
        "with wash_volume_multiplier float 1..8 and accelerate_withdrawal bool. "
        "JSON only."
    ),
}


def bounded_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    return float(np.clip(out, low, high))


def bounded_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        out = int(round(float(value)))
    except (TypeError, ValueError):
        out = default
    return int(np.clip(out, low, high))


class _WrapperBase:
    is_llm_controlled = True

    def __init__(self, base, backend, plan_interval: int = 5):
        self.base = base
        self.backend = backend
        self.plan_interval = max(1, int(plan_interval))
        self._last_plan_day = -10**9
        self._plan: dict[str, Any] = {}
        self.agent_id = base.agent_id
        self.role = f"llm_{base.role}"
        self.is_harmful = bool(base.is_harmful)
        self.portfolio = base.portfolio
        self.rng = getattr(base, "rng", None)

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def _should_plan(self, day: int) -> bool:
        return day == 0 or (day - self._last_plan_day) >= self.plan_interval


class LLMRetailWrapper(_WrapperBase):
    llm_role = "benign_retail"

    def __init__(self, base, backend, plan_interval: int = 5):
        super().__init__(base, backend, plan_interval=plan_interval)
        self._base_risk_appetite = float(getattr(base, "risk_appetite", 0.02))
        self._base_skepticism = float(getattr(base, "skepticism", 0.0))

    def decide(self, day: int, observation: dict):
        if self._should_plan(day):
            self._plan = self._consult(day, observation)
            self._last_plan_day = day
        multiplier = bounded_float(
            self._plan.get("risk_appetite_multiplier"),
            0.5,
            1.5,
            1.0,
        )
        skepticism = bounded_float(
            self._plan.get("social_skepticism"),
            0.0,
            1.0,
            self._base_skepticism,
        )
        self.base.risk_appetite = self._base_risk_appetite * multiplier
        self.base.skepticism = skepticism
        orders, messages = self.base.decide(day, observation)
        if self._plan.get("share_public_message"):
            asset = observation.get("target_asset") or self._choose_asset(observation)
            intensity = bounded_float(self._plan.get("message_intensity"), 0.0, 1.0, 0.0)
            if asset and intensity > 0.05:
                messages.append(Message(
                    sender_id=self.agent_id,
                    asset=asset,
                    sentiment=bounded_float(self._plan.get("message_sentiment"), -1.0, 1.0, 0.0),
                    intensity=intensity,
                    is_harmful=False,
                    is_bot=False,
                    day=day,
                ))
        return orders, messages

    def _consult(self, day: int, observation: dict) -> dict[str, Any]:
        public = {
            "day": day,
            "prices": observation.get("prices", {}),
            "recent_return": observation.get("recent_return", {}),
            "market": observation.get("market", {}),
            "portfolio_value": self.portfolio.mark_to_market(observation.get("prices", {})),
            "sub_role": getattr(self.base, "sub_role", ""),
        }
        return self.backend.chat_json(BENIGN_RETAIL_SYS, json.dumps(public, sort_keys=True))

    def _choose_asset(self, observation: dict) -> str:
        returns = observation.get("recent_return", {})
        if returns:
            return max(returns, key=lambda asset: abs(float(returns[asset])))
        market = observation.get("market", {})
        return next(iter(market), "")


class LLMHarmfulWrapper(_WrapperBase):
    llm_role = "harmful_strategist"

    def __init__(self, base, backend, mechanism: str, plan_interval: int = 5):
        super().__init__(base, backend, plan_interval=plan_interval)
        self.mechanism = mechanism

    def decide(self, day: int, observation: dict):
        if self._should_plan(day):
            self._plan = self._consult(day, observation)
            self._last_plan_day = day
        self._apply_persistent_plan()
        saved = self._apply_temporary_plan(day)
        try:
            return self.base.decide(day, observation)
        finally:
            for name, value in saved.items():
                setattr(self.base, name, value)

    def _consult(self, day: int, observation: dict) -> dict[str, Any]:
        target = getattr(self.base, "target_asset", "")
        market = observation.get("market", {}).get(target, {})
        payload = {
            "day": day,
            "horizon": 30,
            "mechanism": self.mechanism,
            "target_asset": target,
            "market": market,
            "recent_return": observation.get("recent_return", {}).get(target, 0.0),
            "current_inventory": float(self.portfolio.position(target)) if target else 0.0,
            "current_knobs": self._current_knobs(),
        }
        system = HARMFUL_SYSTEMS[self.mechanism]
        return self.backend.chat_json(system, json.dumps(payload, sort_keys=True))

    def _current_knobs(self) -> dict[str, Any]:
        keys = [
            "promote_intensity", "post_intensity", "dump_speed",
            "spoof_size_mult", "daily_cycles", "side_buy_prob",
            "wash_volume_multiplier",
        ]
        return {key: getattr(self.base, key) for key in keys if hasattr(self.base, key)}

    def _apply_persistent_plan(self) -> None:
        if isinstance(self.base, PumpAndDumpLeader) and "promote_intensity" in self._plan:
            self.base.promote_intensity = bounded_float(
                self._plan.get("promote_intensity"), 0.5, 3.0, self.base.promote_intensity
            )
        if isinstance(self.base, Finfluencer) and "post_intensity" in self._plan:
            self.base.post_intensity = bounded_float(
                self._plan.get("post_intensity"), 0.5, 3.0, self.base.post_intensity
            )
        if isinstance(self.base, Spoofer):
            self.base.spoof_size_mult = bounded_float(
                self._plan.get("spoof_size_mult"), 1.0, 10.0, self.base.spoof_size_mult
            )
            self.base.daily_cycles = bounded_int(
                self._plan.get("daily_cycles"), 1, 8, self.base.daily_cycles
            )
            side_bias = str(self._plan.get("side_bias", "mixed")).lower()
            if side_bias == "buy":
                self.base.side_buy_prob = 0.8
            elif side_bias == "sell":
                self.base.side_buy_prob = 0.2
            else:
                self.base.side_buy_prob = 0.5
        if isinstance(self.base, WashTrader):
            self.base.wash_volume_multiplier = bounded_float(
                self._plan.get("wash_volume_multiplier"),
                1.0,
                8.0,
                self.base.wash_volume_multiplier,
            )

    def _apply_temporary_plan(self, day: int) -> dict[str, Any]:
        saved: dict[str, Any] = {}
        if isinstance(self.base, PumpAndDumpLeader) and self._plan.get("dump_today"):
            saved["dump_speed"] = self.base.dump_speed
            self.base.dump_speed = max(self.base.dump_speed, 0.5)
        if isinstance(self.base, Finfluencer) and self._plan.get("sell_today"):
            saved["sell_days"] = self.base.sell_days
            self.base.sell_days = (
                min(int(self.base.sell_days[0]), day),
                max(int(self.base.sell_days[1]), day),
            )
        if isinstance(self.base, WashTrader) and self._plan.get("accelerate_withdrawal"):
            saved["withdraw_days"] = self.base.withdraw_days
            self.base.withdraw_days = (
                min(int(self.base.withdraw_days[0]), day),
                int(self.base.withdraw_days[1]),
            )
        return saved


def mechanism_for_agent(agent) -> str:
    if isinstance(agent, PumpAndDumpLeader):
        return "pump"
    if isinstance(agent, Finfluencer):
        return "finfluencer"
    if isinstance(agent, Spoofer):
        return "spoofing"
    if isinstance(agent, WashTrader):
        return "wash"
    role = str(getattr(agent, "role", "")).lower()
    if "finflu" in role:
        return "finfluencer"
    if "spoof" in role:
        return "spoofing"
    if "wash" in role:
        return "wash"
    return "pump"
