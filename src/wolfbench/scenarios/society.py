"""Build the agent society for a given scenario, harmful ratio and society size."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wolfbench.agents.base import Portfolio
from wolfbench.agents.retail import RetailAgent, build_retail_agents
from wolfbench.agents.market_maker import MarketMaker, build_market_makers
from wolfbench.agents.attackers import (
    PumpAndDumpLeader, BotAmplifier, CoordinatedTrader, Finfluencer,
    Spoofer, WashTrader,
)
from wolfbench.agents.llm import (
    LLMBackend, RuleFallbackBackend, LLMPumpLeader, LLMFinfluencer,
)
from wolfbench.scenarios.base import ScenarioConfig
from wolfbench.utils.seeds import split_rng


@dataclass
class Society:
    retail: list[RetailAgent]
    market_makers: list[MarketMaker]
    attackers: list                                  # heterogeneous attacker objects
    all_agents: list                                 # ordered list for the social graph
    n_total: int
    n_harmful: int
    target_asset: str
    placement: str = "random"


# WolfBench LLM-count schedule (paper §LLM-scaling).
# Strategic-leader count K_LLM grows sublinearly in (N, alpha) and is
# capped at 10. The harmful population that exceeds K_LLM is filled with
# lightweight bots and coordinated trading accounts.
_K_LLM_TABLE = [
    # alpha tier:  <1%  <3%  <10%  >=10%
    [1, 1, 2, 2],   # N <=  200
    [1, 2, 3, 3],   # N <= 2_000
    [2, 3, 5, 5],   # N <= 20_000
    [4, 6, 10, 10], # N >  20_000
]


def strategic_leader_count(n_society: int, alpha: float, n_harmful: int,
                           hard_cap: int = 10) -> int:
    """Sublinear K_LLM schedule used by S1/S2 builders.

    Returns 0 when there is no harmful budget. Otherwise picks a row from
    ``_K_LLM_TABLE`` based on N tier and an alpha tier, and clips by the
    available harmful slots and ``hard_cap``.
    """
    if n_harmful <= 0 or alpha <= 0:
        return 0
    if n_society <= 200:
        ni = 0
    elif n_society <= 2_000:
        ni = 1
    elif n_society <= 20_000:
        ni = 2
    else:
        ni = 3
    if alpha < 0.01:
        ai = 0
    elif alpha < 0.03:
        ai = 1
    elif alpha < 0.10:
        ai = 2
    else:
        ai = 3
    k = _K_LLM_TABLE[ni][ai]
    return int(max(1, min(k, hard_cap, n_harmful)))


def build_society(scenario: ScenarioConfig, n_society: int, alpha: float,
                  rng: np.random.Generator,
                  attacker_overrides: dict | None = None,
                  placement_override: str | None = None,
                  llm_backend: LLMBackend | None = None,
                  n_llm_leaders: int = 0) -> Society:
    """Build retail + harmful + MM agents for one episode.

    The number of LLM-controlled strategic leaders is bounded by
    ``n_llm_leaders`` *and* the scenario's ``leader_count_max``. All other
    harmful agents (bots, wash workers, spoofers) remain rule-based
    regardless of ``alpha`` or ``n_society``.
    """
    n_harmful = max(0, int(round(alpha * n_society)))
    n_retail = max(0, n_society - n_harmful)

    retail_rng, attacker_rng, mm_rng = split_rng(rng, 3)
    retail = build_retail_agents(n_retail, scenario, retail_rng)
    market_makers = build_market_makers(scenario)

    attackers: list = []
    target_asset = scenario.target_asset
    placement = placement_override or "random"
    backend: LLMBackend = llm_backend or RuleFallbackBackend()

    if n_harmful > 0 and scenario.attackers:
        family = next(iter(scenario.attackers))
        cfg = scenario.attackers[family]
        if attacker_overrides:
            cfg = {**cfg, **attacker_overrides}
        target_asset = cfg.get("target_asset", target_asset)
        if family == "pump_and_dump":
            attackers = _build_pump_and_dump(cfg, n_harmful, attacker_rng,
                                             backend, n_llm_leaders,
                                             n_society, alpha)
        elif family == "finfluencer":
            attackers = _build_finfluencer(cfg, n_harmful, attacker_rng,
                                           backend, n_llm_leaders,
                                           n_society, alpha)
            if placement_override is None:
                placement = cfg.get("placement", "high_degree")
        elif family == "spoofing":
            attackers = _build_spoofers(cfg, n_harmful, attacker_rng)
        elif family == "wash_trading":
            attackers = _build_wash_traders(cfg, n_harmful, attacker_rng)

    initial_wealth = float(scenario.retail["initial_wealth"])
    for ag in attackers:
        # harmful agents start with comparable wealth to retail to keep budgets fair
        ag.portfolio = Portfolio(cash=initial_wealth * 5.0,
                                 initial_wealth=initial_wealth * 5.0)

    all_agents = list(retail) + list(attackers) + list(market_makers)
    return Society(
        retail=retail, market_makers=market_makers, attackers=attackers,
        all_agents=all_agents, n_total=n_society, n_harmful=n_harmful,
        target_asset=target_asset, placement=placement,
    )


# ---------------------------------------------------------------- builders

def _build_pump_and_dump(cfg, n_harmful, rng, backend=None, n_llm_leaders=0,
                         n_society=1000, alpha=0.0):
    bot_share = float(cfg.get("bot_amplifier_share", 0.6))
    n_leaders = strategic_leader_count(n_society, alpha, n_harmful)
    remaining = max(0, n_harmful - n_leaders)
    n_bots = int(round(remaining * bot_share))
    n_traders = remaining - n_bots
    n_llm = min(int(n_llm_leaders), n_leaders)
    out: list = []
    for i in range(n_leaders):
        common = dict(
            agent_id=f"pump_leader_{i:04d}",
            role="harmful_pump_leader",
            is_harmful=True,
            target_asset=cfg["target_asset"],
            accumulate_days=tuple(cfg["accumulate_days"]),
            promote_days=tuple(cfg["promote_days"]),
            dump_days=tuple(cfg["dump_days"]),
            target_inventory_share=float(cfg["target_inventory_share"]),
            promote_intensity=float(cfg["promote_intensity"]),
            dump_speed=float(cfg["dump_speed"]),
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )
        if i < n_llm and backend is not None:
            out.append(LLMPumpLeader(backend=backend, **common))
        else:
            out.append(PumpAndDumpLeader(**common))
    for i in range(n_bots):
        out.append(BotAmplifier(
            agent_id=f"bot_amp_{i:04d}",
            role="harmful_bot",
            is_harmful=True,
            target_asset=cfg["target_asset"],
            promote_days=tuple(cfg["promote_days"]),
            intensity=float(cfg.get("bot_intensity", 1.0)),
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        ))
    for i in range(n_traders):
        out.append(CoordinatedTrader(
            agent_id=f"coord_trader_{i:04d}",
            role="harmful_trader",
            is_harmful=True,
            target_asset=cfg["target_asset"],
            accumulate_days=tuple(cfg["accumulate_days"]),
            promote_days=tuple(cfg["promote_days"]),
            dump_days=tuple(cfg["dump_days"]),
            target_inventory_share=float(cfg["target_inventory_share"]) * 0.5,
            dump_speed=float(cfg["dump_speed"]),
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        ))
    return out


def _build_finfluencer(cfg, n_harmful, rng, backend=None, n_llm_leaders=0,
                       n_society=1000, alpha=0.0):
    bot_share = float(cfg.get("bot_amplifier_share", 0.6))
    n_finflu = strategic_leader_count(n_society, alpha, n_harmful)
    remaining = max(0, n_harmful - n_finflu)
    n_bots = int(round(remaining * bot_share))
    n_traders = remaining - n_bots
    n_llm = min(int(n_llm_leaders), n_finflu)
    out: list = []
    for i in range(n_finflu):
        common = dict(
            agent_id=f"finfluencer_{i:04d}",
            role="harmful_finfluencer",
            is_harmful=True,
            target_asset=cfg["target_asset"],
            accumulate_days=tuple(cfg["accumulate_days"]),
            promote_days=tuple(cfg["promote_days"]),
            sell_days=tuple(cfg["sell_days"]),
            target_inventory_share=float(cfg["target_inventory_share"]),
            post_intensity=float(cfg["post_intensity"]),
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )
        if i < n_llm and backend is not None:
            out.append(LLMFinfluencer(backend=backend, **common))
        else:
            out.append(Finfluencer(**common))
    # remaining harmful budget acts as follower-bot amplifiers
    for i in range(n_bots):
        out.append(BotAmplifier(
            agent_id=f"finflu_bot_{i:04d}",
            role="harmful_bot",
            is_harmful=True,
            target_asset=cfg["target_asset"],
            promote_days=tuple(cfg["promote_days"]),
            intensity=0.8,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        ))
    # coordinated early buyers / hidden seller accounts
    for i in range(n_traders):
        out.append(CoordinatedTrader(
            agent_id=f"finflu_trader_{i:04d}",
            role="harmful_trader",
            is_harmful=True,
            target_asset=cfg["target_asset"],
            accumulate_days=tuple(cfg["accumulate_days"]),
            promote_days=tuple(cfg["promote_days"]),
            dump_days=tuple(cfg["sell_days"]),
            target_inventory_share=float(cfg["target_inventory_share"]) * 0.5,
            dump_speed=0.15,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        ))
    return out


def _build_spoofers(cfg, n_harmful, rng):
    out: list = []
    for i in range(n_harmful):
        out.append(Spoofer(
            agent_id=f"spoofer_{i:04d}",
            role="harmful_spoofer",
            is_harmful=True,
            target_asset=cfg["target_asset"],
            spoof_size_mult=float(cfg["spoof_size_mult"]),
            cancel_latency_steps=int(cfg.get("cancel_latency_steps", 1)),
            daily_cycles=int(cfg.get("daily_cycles", 4)),
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        ))
    return out


def _build_wash_traders(cfg, n_harmful, rng):
    cluster = max(2, int(cfg.get("cluster_size", 4)))
    out: list = []
    # only complete clusters: guarantees every counterparty_id resolves
    n_clusters = max(1, n_harmful // cluster)
    for c in range(n_clusters):
        cluster_ids = [f"wash_{c:03d}_{i}" for i in range(cluster)]
        for i, aid in enumerate(cluster_ids):
            partner = cluster_ids[(i + 1) % cluster]
            out.append(WashTrader(
                agent_id=aid,
                role="harmful_wash",
                is_harmful=True,
                target_asset=cfg["target_asset"],
                counterparty_id=partner,
                accumulate_days=tuple(cfg["accumulate_days"]),
                wash_days=tuple(cfg["wash_days"]),
                withdraw_days=tuple(cfg["withdraw_days"]),
                wash_volume_multiplier=float(cfg["wash_volume_multiplier"]),
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            ))
    return out
