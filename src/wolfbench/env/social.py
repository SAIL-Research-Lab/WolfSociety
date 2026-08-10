"""Social layer: scale-free graph, message propagation, cascade tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

import networkx as nx
import numpy as np


@dataclass
class Message:
    sender_id: str
    asset: str
    sentiment: float            # +1 bullish, -1 bearish
    intensity: float            # >0
    is_harmful: bool = False    # came from a harmful source
    is_bot: bool = False
    day: int = 0
    # Optional metadata used by the final bounded-rational network signaling
    # game. Legacy scenarios can ignore these fields.
    message_id: str = ""
    kind: str = "post"          # post / reshare / challenge
    root_sender_id: str = ""
    confidence: float = 0.5
    social_proof: float = 0.0


@dataclass
class SocialState:
    # per-agent exposure to messages this step (sentiment-weighted)
    exposure: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    harmful_exposure: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    # per-asset aggregate signal
    msg_volume: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    sentiment_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    cascade_size: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    history: dict[str, list[dict[str, float]]] = field(default_factory=lambda: defaultdict(list))


class SocialGraph:
    def __init__(self, n_agents: int, mean_degree: int, rng: np.random.Generator,
                 graph_type: str = "scale_free"):
        m = max(1, mean_degree // 2)
        if graph_type == "scale_free" and n_agents > m + 1:
            seed = int(rng.integers(0, 2**31 - 1))
            self.g = nx.barabasi_albert_graph(n_agents, m, seed=seed)
        else:
            seed = int(rng.integers(0, 2**31 - 1))
            p = mean_degree / max(n_agents - 1, 1)
            self.g = nx.fast_gnp_random_graph(n_agents, p, seed=seed)
        self.agent_nodes: list[int] = list(self.g.nodes())
        self.id_to_node: dict[str, int] = {}
        self.node_to_id: dict[int, str] = {}
        self.rng = rng

    def assign_ids(self, agent_ids: list[str]) -> None:
        if len(agent_ids) != len(self.agent_nodes):
            raise ValueError("agent_ids length does not match graph nodes")
        for aid, node in zip(agent_ids, self.agent_nodes):
            self.id_to_node[aid] = node
            self.node_to_id[node] = aid

    def degree(self, agent_id: str) -> int:
        return self.g.degree[self.id_to_node[agent_id]]

    def degrees(self) -> dict[str, int]:
        return {self.node_to_id[n]: int(d) for n, d in self.g.degree()}

    def top_degree_ids(self, k: int) -> list[str]:
        order = sorted(self.g.degree, key=lambda x: x[1], reverse=True)[:k]
        return [self.node_to_id[n] for n, _ in order]

    def neighbours(self, agent_id: str) -> list[str]:
        node = self.id_to_node[agent_id]
        return [self.node_to_id[n] for n in self.g.neighbors(node)]


class SocialEnv:
    """Daily message propagation over a static graph.

    Agents post messages targeting an asset with a sentiment. Each message
    reaches direct neighbours with probability ``p_expose``; bots reshare with
    probability ``p_reshare``. We track cumulative exposure per (agent, asset)
    used by retail belief updates and by WolfGuard signals.
    """

    def __init__(self, graph: SocialGraph, scenario, rng: np.random.Generator):
        self.graph = graph
        self.rng = rng
        self.feedback_strength = float(scenario.social.get("feedback_strength", 0.6))
        self.message_decay = float(scenario.social.get("message_decay", 0.5))
        self.bot_ratio = float(scenario.social.get("bot_ratio", 0.05))
        self.p_expose = float(scenario.social.get("p_expose", 0.6))
        self.p_reshare = float(scenario.social.get("p_reshare", 0.25))
        self.state = SocialState()
        # decayed running aggregates
        self._decayed_msg_vol: dict[str, float] = defaultdict(float)
        self._decayed_sent: dict[str, float] = defaultdict(float)
        self._decayed_harm_share: dict[str, float] = defaultdict(float)

    def step(self, day: int, messages: list[Message],
             market_returns: dict[str, float]) -> None:
        # decay yesterday's exposure
        for aid in list(self.state.exposure.keys()):
            for ast in list(self.state.exposure[aid].keys()):
                self.state.exposure[aid][ast] *= self.message_decay
                self.state.harmful_exposure[aid][ast] *= self.message_decay
        for ast in list(self._decayed_msg_vol.keys()):
            self._decayed_msg_vol[ast] *= self.message_decay
            self._decayed_sent[ast] *= self.message_decay
            self._decayed_harm_share[ast] *= self.message_decay

        # market feedback: amplify intensity on assets with rising prices
        for m in messages:
            r = market_returns.get(m.asset, 0.0)
            feedback = 1.0 + self.feedback_strength * max(r, 0.0) * (5.0 if m.sentiment > 0 else 1.0)
            eff_intensity = m.intensity * feedback

            # base exposure: direct neighbours
            sender_node = self.graph.id_to_node.get(m.sender_id)
            if sender_node is None:
                continue
            neighbours = list(self.graph.g.neighbors(sender_node))

            harmful_contrib = eff_intensity * m.sentiment if m.is_harmful else 0.0
            for nb in neighbours:
                if self.rng.random() < self.p_expose:
                    aid = self.graph.node_to_id[nb]
                    self.state.exposure[aid][m.asset] += eff_intensity * m.sentiment
                    if m.is_harmful:
                        self.state.harmful_exposure[aid][m.asset] += eff_intensity
                    # bots reshare to neighbours of neighbours (weak)
                    if m.is_bot and self.rng.random() < self.p_reshare:
                        for nb2 in self.graph.g.neighbors(nb):
                            aid2 = self.graph.node_to_id[nb2]
                            self.state.exposure[aid2][m.asset] += 0.5 * eff_intensity * m.sentiment
                            if m.is_harmful:
                                self.state.harmful_exposure[aid2][m.asset] += 0.5 * eff_intensity

            self._decayed_msg_vol[m.asset] += eff_intensity
            self._decayed_sent[m.asset] += eff_intensity * m.sentiment
            if m.is_harmful:
                self._decayed_harm_share[m.asset] += eff_intensity

            # cascade tracking: any agent currently exposed to this asset
            for aid in self.state.exposure:
                if abs(self.state.exposure[aid].get(m.asset, 0.0)) > 0.1:
                    self.state.cascade_size[m.asset].add(aid)

        # snapshot per-asset aggregate
        for ast, vol in self._decayed_msg_vol.items():
            sent = self._decayed_sent[ast] / max(vol, 1e-9)
            harm_share = self._decayed_harm_share[ast] / max(vol, 1e-9)
            self.state.history[ast].append({
                "day": float(day),
                "msg_volume": float(vol),
                "sentiment": float(sent),
                "harmful_msg_share": float(harm_share),
                "cascade_size": float(len(self.state.cascade_size[ast])),
            })

    # -------- views

    def asset_signal(self, asset: str) -> dict[str, float]:
        vol = self._decayed_msg_vol.get(asset, 0.0)
        sent = self._decayed_sent.get(asset, 0.0) / max(vol, 1e-9)
        harm = self._decayed_harm_share.get(asset, 0.0) / max(vol, 1e-9)
        return {
            "msg_volume": vol,
            "sentiment": sent,
            "harmful_msg_share": harm,
            "cascade_size": float(len(self.state.cascade_size.get(asset, set()))),
        }

    def agent_signal(self, agent_id: str, asset: str) -> tuple[float, float]:
        """Return (signed exposure, harmful exposure magnitude)."""
        return (
            self.state.exposure.get(agent_id, {}).get(asset, 0.0),
            self.state.harmful_exposure.get(agent_id, {}).get(asset, 0.0),
        )
