"""Bounded-rational retail agents and an endogenous network signaling game.

This module is the benchmark-level v3 retail/social runtime. It turns the
paper-facing population from one weighted-score controller with different
coefficients into a mixture of genuinely different decision processes:

* value-based choice with heterogeneous payoff responsiveness;
* risk-averse and trend-following heuristics;
* social-following behavior;
* aggressive, impulse-prone trading.

Retail agents receive noisy private signals, have finite attention, choose
whether to post/reshare/challenge, and update sender trust from realized market
outcomes.  ``BoundedSocialEnv`` records exposure and decision events so the
final experiments can estimate conditional mutual information and transfer
entropy instead of treating message volume as social dynamics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log2
from typing import Any, Iterable

import numpy as np

from wolfbench.env.market import Order
from wolfbench.env.social import Message, SocialEnv


CONTROLLER_MODES = (
    "legacy_score",
    "mixed_roles",
    "all_risk_averse",
    "all_value",
    "all_trend",
    "all_social",
    "all_aggressive",
)

CONTROLLER_ALIASES = {
    "bounded_mixed": "mixed_roles",
    "qre_population": "all_value",
    "heuristic_population": "all_trend",
    "zero_intelligence": "all_aggressive",
}

ROLE_ALIASES = {
    "fundamental": "value_investor",
    "momentum": "trend_follower",
    "fomo": "social_follower",
    "skeptical": "risk_averse",
    "noise": "aggressive_trader",
}

ROLE_CATALOG = {
    "risk_averse": {
        "paper_name": "Risk-Averse",
        "plain_style": "Trades rarely, requires strong private evidence, uses small positions, and challenges doubtful claims.",
    },
    "value_investor": {
        "paper_name": "Value-Oriented",
        "plain_style": "Compares price with a noisy private value estimate and accepts occasional decision mistakes.",
    },
    "trend_follower": {
        "paper_name": "Trend-Following",
        "plain_style": "Uses recent price direction and inertia instead of solving an expected-utility problem.",
    },
    "social_follower": {
        "paper_name": "Social-Following",
        "plain_style": "Responds to neighbors, visible popularity, and trusted reshares; most prone to cascades.",
    },
    "aggressive_trader": {
        "paper_name": "Aggressive",
        "plain_style": "Uses larger positions, reacts to salient cues, and sometimes trades impulsively under uncertainty.",
    },
}

MODE_ROLE = {
    "all_risk_averse": "risk_averse",
    "all_value": "value_investor",
    "all_trend": "trend_follower",
    "all_social": "social_follower",
    "all_aggressive": "aggressive_trader",
}


def normalize_controller_mode(mode: str) -> str:
    return CONTROLLER_ALIASES.get(str(mode), str(mode))


@dataclass(frozen=True)
class BehaviorProfile:
    process: str
    qre_beta: float
    attention_capacity: int
    private_noise: float
    loss_aversion: float
    conformity: float
    trust_learning_rate: float
    base_trade_fraction: float
    post_probability: float


ROLE_PROFILES: dict[str, BehaviorProfile] = {
    "risk_averse": BehaviorProfile(
        process="risk_averse", qre_beta=3.2, attention_capacity=2,
        private_noise=0.030, loss_aversion=3.2, conformity=-0.15,
        trust_learning_rate=0.08, base_trade_fraction=0.009,
        post_probability=0.035,
    ),
    "value_investor": BehaviorProfile(
        process="value_based", qre_beta=4.0, attention_capacity=4,
        private_noise=0.025, loss_aversion=2.2, conformity=0.10,
        trust_learning_rate=0.10, base_trade_fraction=0.018,
        post_probability=0.045,
    ),
    "trend_follower": BehaviorProfile(
        process="trend_following", qre_beta=2.2, attention_capacity=3,
        private_noise=0.050, loss_aversion=1.4, conformity=0.35,
        trust_learning_rate=0.15, base_trade_fraction=0.026,
        post_probability=0.075,
    ),
    "social_follower": BehaviorProfile(
        process="social_following", qre_beta=1.6, attention_capacity=6,
        private_noise=0.080, loss_aversion=1.1, conformity=1.35,
        trust_learning_rate=0.22, base_trade_fraction=0.034,
        post_probability=0.14,
    ),
    "aggressive_trader": BehaviorProfile(
        process="aggressive", qre_beta=1.2, attention_capacity=4,
        private_noise=0.16, loss_aversion=0.8, conformity=0.45,
        trust_learning_rate=0.05, base_trade_fraction=0.045,
        post_probability=0.10,
    ),
}


def _entropy(values: Iterable[Any]) -> float:
    values = list(values)
    if not values:
        return 0.0
    counts = Counter(values)
    n = float(len(values))
    return float(-sum((count / n) * log2(count / n) for count in counts.values()))


def _joint(*columns: Iterable[Any]) -> list[tuple[Any, ...]]:
    materialized = [list(column) for column in columns]
    if not materialized:
        return []
    return list(zip(*materialized))


def _distribution(values: Iterable[Any], support: Iterable[Any]) -> np.ndarray:
    values = list(values)
    support = list(support)
    if not support:
        return np.asarray([], dtype=float)
    counts = Counter(values)
    total = float(len(values))
    if total <= 0:
        return np.full(len(support), 1.0 / len(support), dtype=float)
    return np.asarray([counts.get(item, 0) / total for item in support], dtype=float)


def _jensen_shannon_bits(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.size == 0 or q.size == 0:
        return 0.0
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)

    def kl_bits(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], 1e-12, None))))

    return 0.5 * kl_bits(p, m) + 0.5 * kl_bits(q, m)


def conditional_mutual_information(x: Iterable[Any], y: Iterable[Any], z: Iterable[Any]) -> float:
    """Plug-in estimate I(X;Y|Z) in bits for small discrete alphabets."""
    x_list, y_list, z_list = list(x), list(y), list(z)
    if not x_list or not (len(x_list) == len(y_list) == len(z_list)):
        return 0.0
    value = (
        _entropy(_joint(x_list, z_list))
        + _entropy(_joint(y_list, z_list))
        - _entropy(z_list)
        - _entropy(_joint(x_list, y_list, z_list))
    )
    return float(max(value, 0.0))


def permutation_corrected_cmi(
    x: Iterable[Any],
    y: Iterable[Any],
    z: Iterable[Any],
    rng: np.random.Generator,
    n_permutations: int = 16,
) -> tuple[float, float, float]:
    """Return (bias-corrected, raw, conditional-permutation null) CMI.

    The permutation is performed within each conditioning stratum, preserving
    P(Y|Z) while breaking residual X--Y dependence.  This reduces the positive
    finite-sample bias of the discrete plug-in estimator.
    """
    x_list, y_list, z_list = list(x), list(y), list(z)
    raw = conditional_mutual_information(x_list, y_list, z_list)
    if not x_list or n_permutations <= 0:
        return raw, raw, 0.0
    strata: dict[Any, list[int]] = defaultdict(list)
    for index, condition in enumerate(z_list):
        strata[condition].append(index)
    null_values = []
    for _ in range(n_permutations):
        shuffled = list(y_list)
        for indices in strata.values():
            values = [y_list[index] for index in indices]
            permuted = rng.permutation(values).tolist()
            for index, value in zip(indices, permuted):
                shuffled[index] = value
        null_values.append(conditional_mutual_information(x_list, shuffled, z_list))
    null = float(np.mean(null_values))
    return float(max(raw - null, 0.0)), float(raw), null


class BoundedSocialEnv(SocialEnv):
    """Social layer with per-recipient inboxes and information-flow logs."""

    game_version = "network_signal_game_v3"

    def __init__(self, graph, scenario, rng: np.random.Generator):
        super().__init__(graph, scenario, rng)
        self.content_visible = bool(scenario.social.get("content_visible", True))
        self.social_proof_visible = bool(scenario.social.get("social_proof_visible", True))
        self.shuffle_sender = bool(scenario.social.get("shuffle_sender", False))
        self.message_delay = int(scenario.social.get("message_delay", 0))
        self.max_inbox = int(scenario.social.get("max_inbox", 96))
        self.inboxes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.exposure_events: list[dict[str, Any]] = []
        self.message_events: list[dict[str, Any]] = []
        self.decision_events: list[dict[str, Any]] = []
        self._root_activity: dict[str, float] = defaultdict(float)
        self._message_counter = 0

    def agent_messages(self, agent_id: str, asset: str, day: int, max_age: int = 3) -> list[dict[str, Any]]:
        lower = day - max_age - self.message_delay
        upper = day - self.message_delay
        return [
            dict(event)
            for event in self.inboxes.get(agent_id, [])
            if event["asset"] == asset and lower <= int(event["day"]) < upper
        ]

    def record_decision(self, event: dict[str, Any]) -> None:
        self.decision_events.append(dict(event))

    def step(self, day: int, messages: list[Message], market_returns: dict[str, float]) -> None:
        for aid in list(self.state.exposure.keys()):
            for asset in list(self.state.exposure[aid].keys()):
                self.state.exposure[aid][asset] *= self.message_decay
                self.state.harmful_exposure[aid][asset] *= self.message_decay
        for asset in list(self._decayed_msg_vol):
            self._decayed_msg_vol[asset] *= self.message_decay
            self._decayed_sent[asset] *= self.message_decay
            self._decayed_harm_share[asset] *= self.message_decay
        for root in list(self._root_activity):
            self._root_activity[root] *= 0.75

        agent_ids = list(self.graph.id_to_node)
        for message in messages:
            if message.sender_id not in self.graph.id_to_node:
                continue
            self._message_counter += 1
            message_id = message.message_id or f"m{day:03d}_{self._message_counter:08d}"
            root = message.root_sender_id or message.sender_id
            proof = float(np.log1p(self._root_activity[root]))
            self._root_activity[root] += 1.0
            self.message_events.append({
                "day": int(day),
                "sender_id": message.sender_id,
                "root_sender_id": root,
                "message_id": message_id,
                "asset": message.asset,
                "kind": message.kind,
                "sentiment": float(message.sentiment),
                "intensity": float(message.intensity),
                "confidence": float(message.confidence),
                "social_proof": float(proof),
                "is_harmful_source": int(message.is_harmful),
                "is_bot": int(message.is_bot),
            })

            market_return = float(market_returns.get(message.asset, 0.0))
            feedback = 1.0 + self.feedback_strength * max(market_return, 0.0) * (
                5.0 if message.sentiment > 0 else 1.0
            )
            intensity = float(max(0.0, message.intensity * feedback))
            sender_node = self.graph.id_to_node[message.sender_id]
            neighbours = list(self.graph.g.neighbors(sender_node))
            for neighbour in neighbours:
                if self.rng.random() >= self.p_expose:
                    continue
                recipient = self.graph.node_to_id[neighbour]
                observed_sender = message.sender_id
                if self.shuffle_sender and agent_ids:
                    observed_sender = str(self.rng.choice(agent_ids))
                event = self._exposure_event(
                    day, recipient, observed_sender, root, message, message_id,
                    intensity, proof, distance=1,
                )
                self._deliver(event, harmful=message.is_harmful)

                if message.is_bot and self.rng.random() < self.p_reshare:
                    for neighbour2 in self.graph.g.neighbors(neighbour):
                        recipient2 = self.graph.node_to_id[neighbour2]
                        event2 = self._exposure_event(
                            day, recipient2, observed_sender, root, message,
                            message_id, 0.5 * intensity, proof + 0.25, distance=2,
                        )
                        self._deliver(event2, harmful=message.is_harmful)

            self._decayed_msg_vol[message.asset] += intensity
            self._decayed_sent[message.asset] += intensity * message.sentiment
            if message.is_harmful:
                self._decayed_harm_share[message.asset] += intensity

        for asset, volume in self._decayed_msg_vol.items():
            sentiment = self._decayed_sent[asset] / max(volume, 1e-9)
            harmful_share = self._decayed_harm_share[asset] / max(volume, 1e-9)
            self.state.history[asset].append({
                "day": float(day),
                "msg_volume": float(volume),
                "sentiment": float(sentiment),
                "harmful_msg_share": float(harmful_share),
                "cascade_size": float(len(self.state.cascade_size[asset])),
            })

        cutoff = day - 5 - self.message_delay
        for recipient in list(self.inboxes):
            kept = [event for event in self.inboxes[recipient] if int(event["day"]) >= cutoff]
            self.inboxes[recipient] = kept[-self.max_inbox:]

    def _exposure_event(
        self,
        day: int,
        recipient: str,
        sender: str,
        root: str,
        message: Message,
        message_id: str,
        intensity: float,
        proof: float,
        distance: int,
    ) -> dict[str, Any]:
        sentiment = float(message.sentiment if self.content_visible else 0.0)
        observed_proof = float(proof if self.social_proof_visible else 0.0)
        return {
            "day": int(day),
            "recipient_id": recipient,
            "sender_id": sender,
            "root_sender_id": root,
            "message_id": message_id,
            "asset": message.asset,
            "sentiment": sentiment,
            "intensity": float(intensity),
            "confidence": float(message.confidence),
            "social_proof": observed_proof,
            "kind": message.kind,
            "distance": int(distance),
            "is_harmful_source": int(message.is_harmful),
        }

    def _deliver(self, event: dict[str, Any], harmful: bool) -> None:
        recipient = str(event["recipient_id"])
        asset = str(event["asset"])
        contribution = float(event["intensity"] * event["sentiment"])
        self.state.exposure[recipient][asset] += contribution
        if harmful:
            self.state.harmful_exposure[recipient][asset] += abs(float(event["intensity"]))
        # The legacy benchmark's social-cascade component is a *harmful*
        # safety metric. Benign cascades remain available through the new root
        # reach/event metrics, but must not make alpha=0 count as an attack.
        if harmful and self.state.harmful_exposure[recipient][asset] > 0.1:
            self.state.cascade_size[asset].add(recipient)
        self.inboxes[recipient].append(dict(event))
        self.exposure_events.append(dict(event))

    def information_metrics(self) -> dict[str, float]:
        rows = self.decision_events
        if not rows:
            return {
                "social_information_bits": 0.0,
                "social_information_raw_bits": 0.0,
                "social_information_null_bits": 0.0,
                "private_information_bits": 0.0,
                "private_information_raw_bits": 0.0,
                "private_information_null_bits": 0.0,
                "private_signal_quality_bits": 0.0,
                "private_signal_direction_accuracy": 0.0,
                "social_proof_information_bits": 0.0,
                "social_proof_information_raw_bits": 0.0,
                "social_proof_information_null_bits": 0.0,
                "social_dominance_ratio": 0.0,
                "action_entropy_bits": 0.0,
                "trade_participation_rate": 0.0,
                "role_action_information_bits": 0.0,
                "role_trade_rate_gap": 0.0,
                "decision_role_entropy_bits": 0.0,
                "decision_role_entropy_normalized": 0.0,
                "effective_decision_roles": 0.0,
                "pairwise_role_behavior_jsd_bits": 0.0,
                "hse_like_behavioral_diversity": 0.0,
                "mean_social_coupling_proxy": 0.0,
                "mean_choice_entropy_bits": 0.0,
                "mean_qre_action_entropy_bits": 0.0,
                "transfer_entropy_social_to_trade_bits": 0.0,
                "transfer_entropy_social_to_trade_raw_bits": 0.0,
                "transfer_entropy_social_to_trade_null_bits": 0.0,
                "transfer_entropy_price_to_message_bits": 0.0,
                "transfer_entropy_price_to_message_raw_bits": 0.0,
                "transfer_entropy_price_to_message_null_bits": 0.0,
                "cascade_decision_rate": 0.0,
                "n_information_conflicts": 0.0,
                "n_messages": float(len(self.message_events)),
                "n_benign_messages": 0.0,
                "n_reshares": 0.0,
                "n_challenges": 0.0,
                "max_cascade_reach": 0.0,
                "mean_cascade_reach": 0.0,
                "n_agent_decisions": 0.0,
                "n_exposure_events": float(len(self.exposure_events)),
            }
        action = [row["action"] for row in rows]
        social = [row["social_bin"] for row in rows]
        private = [row["private_bin"] for row in rows]
        true_private = [row["true_private_bin"] for row in rows]
        market = [row["market_bin"] for row in rows]
        proof = [row["proof_bin"] for row in rows]
        roles = [row["role"] for row in rows]
        metric_rng = np.random.default_rng(918273)
        social_info, social_raw, social_null = permutation_corrected_cmi(
            action, social, _joint(market, private), metric_rng
        )
        private_info, private_raw, private_null = permutation_corrected_cmi(
            action, private, _joint(market, social), metric_rng
        )
        private_quality, _, _ = permutation_corrected_cmi(
            private, true_private, market, metric_rng
        )
        proof_info, proof_raw, proof_null = permutation_corrected_cmi(
            action, proof, _joint(market, private, social), metric_rng
        )
        role_info, _, _ = permutation_corrected_cmi(
            action, roles, _joint(market, private, social), metric_rng
        )
        denominator = social_info + private_info
        directional_private = [
            int(observed == truth)
            for observed, truth in zip(private, true_private)
            if truth != 0
        ]

        by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_agent[str(row["agent_id"])].append(row)
        current_action: list[int] = []
        current_social: list[int] = []
        social_condition: list[tuple[int, int]] = []
        current_message: list[int] = []
        current_market: list[int] = []
        message_condition: list[int] = []
        for history in by_agent.values():
            history.sort(key=lambda row: (int(row["day"]), str(row["asset"])))
            previous_by_asset: dict[str, dict[str, Any]] = {}
            for row in history:
                asset = str(row["asset"])
                previous = previous_by_asset.get(asset)
                if previous is not None:
                    current_action.append(int(row["action"]))
                    current_social.append(int(row["social_bin"]))
                    social_condition.append((int(previous["action"]), int(row["market_bin"])))
                    current_message.append(int(row["message_action"]))
                    current_market.append(int(row["market_bin"]))
                    message_condition.append(int(previous["message_action"]))
                previous_by_asset[asset] = row
        te_social, te_social_raw, te_social_null = permutation_corrected_cmi(
            current_action, current_social, social_condition, metric_rng
        )
        te_price, te_price_raw, te_price_null = permutation_corrected_cmi(
            current_message, current_market, message_condition, metric_rng
        )
        conflicts = [
            row for row in rows
            if int(row["private_bin"]) != 0
            and int(row["social_bin"]) != 0
            and int(row["private_bin"]) != int(row["social_bin"])
        ]
        cascaded = [
            row for row in conflicts
            if int(row["action"]) == int(row["social_bin"])
        ]
        benign_messages = [event for event in self.message_events if not event["is_harmful_source"]]
        root_recipients: dict[str, set[str]] = defaultdict(set)
        for event in self.exposure_events:
            root_recipients[str(event["root_sender_id"])].add(str(event["recipient_id"]))
        cascade_reaches = [len(recipients) for recipients in root_recipients.values()]
        known_choice_entropies = [
            float(row["choice_entropy"])
            for row in rows
            if np.isfinite(float(row["choice_entropy"]))
        ]
        qre_choice_entropies = [
            float(row["choice_entropy"])
            for row in rows
            if row["policy"] == "value_based" and np.isfinite(float(row["choice_entropy"]))
        ]
        role_trade_rates: dict[str, list[int]] = defaultdict(list)
        role_behavior: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in rows:
            role = str(row["role"])
            role_trade_rates[role].append(int(int(row["action"]) != 0))
            role_behavior[role].append((int(row["action"]), int(row["message_action"])))
        trade_rates = [float(np.mean(values)) for values in role_trade_rates.values()]
        role_entropy = _entropy(roles)
        catalog_entropy = log2(max(len(ROLE_CATALOG), 2))
        role_entropy_normalized = role_entropy / catalog_entropy if catalog_entropy > 0 else 0.0
        effective_roles = 2.0 ** role_entropy
        behavior_support = [(a, m) for a in (-1, 0, 1) for m in (-1, 0, 1)]
        behavior_distributions = {
            role: _distribution(values, behavior_support)
            for role, values in role_behavior.items()
            if values
        }
        pairwise_jsd: list[float] = []
        behavior_roles = sorted(behavior_distributions)
        for i, role_i in enumerate(behavior_roles):
            for role_j in behavior_roles[i + 1:]:
                pairwise_jsd.append(
                    _jensen_shannon_bits(
                        behavior_distributions[role_i],
                        behavior_distributions[role_j],
                    )
                )
        mean_role_behavior_jsd = float(np.mean(pairwise_jsd)) if pairwise_jsd else 0.0
        hse_like_behavioral_diversity = float(role_entropy_normalized * mean_role_behavior_jsd)
        mean_degree = float(np.mean(list(self.graph.degrees().values()))) if self.graph.degrees() else 0.0
        coupling_values = []
        for row in rows:
            beta = float(row["qre_beta"])
            conformity = float(row.get("conformity", 0.0))
            if not (np.isfinite(beta) and np.isfinite(conformity)):
                continue
            capacity = max(float(row["attention_capacity"]), 0.0)
            processed_share = capacity / (capacity + 3.0) if capacity > 0 else 0.0
            coupling_values.append(
                0.5
                * max(beta, 0.0)
                * max(conformity, 0.0)
                * mean_degree
                * processed_share
            )
        return {
            "social_information_bits": float(social_info),
            "social_information_raw_bits": float(social_raw),
            "social_information_null_bits": float(social_null),
            "private_information_bits": float(private_info),
            "private_information_raw_bits": float(private_raw),
            "private_information_null_bits": float(private_null),
            "private_signal_quality_bits": float(private_quality),
            "private_signal_direction_accuracy": (
                float(np.mean(directional_private)) if directional_private else 0.0
            ),
            "social_proof_information_bits": float(proof_info),
            "social_proof_information_raw_bits": float(proof_raw),
            "social_proof_information_null_bits": float(proof_null),
            "social_dominance_ratio": float(social_info / denominator) if denominator > 0 else 0.0,
            "action_entropy_bits": float(_entropy(action)),
            "trade_participation_rate": float(np.mean([int(value != 0) for value in action])),
            "role_action_information_bits": float(role_info),
            "role_trade_rate_gap": float(max(trade_rates) - min(trade_rates)) if trade_rates else 0.0,
            "decision_role_entropy_bits": float(role_entropy),
            "decision_role_entropy_normalized": float(role_entropy_normalized),
            "effective_decision_roles": float(effective_roles),
            "pairwise_role_behavior_jsd_bits": float(mean_role_behavior_jsd),
            "hse_like_behavioral_diversity": float(hse_like_behavioral_diversity),
            "mean_social_coupling_proxy": float(np.mean(coupling_values)) if coupling_values else 0.0,
            "mean_choice_entropy_bits": float(np.mean(known_choice_entropies)) if known_choice_entropies else 0.0,
            "mean_qre_action_entropy_bits": float(np.mean(qre_choice_entropies)) if qre_choice_entropies else 0.0,
            "mean_attention_used": float(np.mean([row["attention_used"] for row in rows])),
            "mean_sender_trust": float(np.mean([row["mean_sender_trust"] for row in rows])),
            "transfer_entropy_social_to_trade_bits": float(te_social),
            "transfer_entropy_social_to_trade_raw_bits": float(te_social_raw),
            "transfer_entropy_social_to_trade_null_bits": float(te_social_null),
            "transfer_entropy_price_to_message_bits": float(te_price),
            "transfer_entropy_price_to_message_raw_bits": float(te_price_raw),
            "transfer_entropy_price_to_message_null_bits": float(te_price_null),
            "cascade_decision_rate": float(len(cascaded) / len(conflicts)) if conflicts else 0.0,
            "n_information_conflicts": float(len(conflicts)),
            "n_messages": float(len(self.message_events)),
            "n_benign_messages": float(len(benign_messages)),
            "n_reshares": float(sum(event["kind"] == "reshare" for event in benign_messages)),
            "n_challenges": float(sum(event["kind"] == "challenge" for event in benign_messages)),
            "max_cascade_reach": float(max(cascade_reaches, default=0)),
            "mean_cascade_reach": float(np.mean(cascade_reaches)) if cascade_reaches else 0.0,
            "n_agent_decisions": float(len(rows)),
            "n_exposure_events": float(len(self.exposure_events)),
        }


class BoundedRationalRetailAgent:
    """Final-hybrid retail wrapper with role-specific cognition and messaging."""

    is_llm_controlled = False

    def __init__(self, base: Any, scenario: Any, controller_mode: str = "mixed_roles"):
        controller_mode = normalize_controller_mode(controller_mode)
        if controller_mode not in CONTROLLER_MODES:
            raise ValueError(f"unknown controller mode: {controller_mode}")
        self.base = base
        self.agent_id = base.agent_id
        self.role = base.role
        raw_role = str(getattr(base, "sub_role", "value_investor"))
        self.sub_role = MODE_ROLE.get(
            controller_mode, ROLE_ALIASES.get(raw_role, raw_role)
        )
        self.is_harmful = False
        self.portfolio = base.portfolio
        self.rng = base.rng
        self.warning_level = base.warning_level
        self.cooldown_until = base.cooldown_until
        self.blocked_today = base.blocked_today
        self.last_belief_breakdown = base.last_belief_breakdown
        self.risk_appetite = float(getattr(base, "risk_appetite", 0.02))
        self._nominal_risk_appetite = max(self.risk_appetite, 1e-6)
        self.skepticism = float(getattr(base, "skepticism", 0.0))
        self.controller_mode = controller_mode
        profile = ROLE_PROFILES.get(self.sub_role, ROLE_PROFILES["value_investor"])
        self.profile = self._individualize(profile, scenario)
        self.process = self._process_for_mode(self.profile.process, controller_mode)
        self.private_values: dict[str, float] = {}
        self.sender_trust: dict[str, float] = defaultdict(lambda: 0.5)
        self._attended_last: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def _individualize(self, profile: BehaviorProfile, scenario: Any) -> BehaviorProfile:
        scale = float(scenario.retail.get("attention_capacity_scale", 1.0))
        beta_scale = float(scenario.retail.get("qre_beta_scale", 1.0))
        conformity_scale = float(scenario.retail.get("conformity_scale", 1.0))
        trust_learning_scale = float(scenario.retail.get("trust_learning_scale", 1.0))
        private_noise_scale = float(scenario.retail.get("private_noise_scale", 1.0))
        jitter = float(self.rng.lognormal(mean=0.0, sigma=0.18))
        return BehaviorProfile(
            process=profile.process,
            qre_beta=max(0.0, profile.qre_beta * beta_scale * jitter),
            attention_capacity=max(1, int(round(profile.attention_capacity * scale / np.sqrt(jitter)))),
            private_noise=max(
                1e-4,
                profile.private_noise * private_noise_scale * float(self.rng.lognormal(0.0, 0.15)),
            ),
            loss_aversion=max(0.5, profile.loss_aversion * float(self.rng.lognormal(0.0, 0.12))),
            conformity=profile.conformity * conformity_scale * float(self.rng.lognormal(0.0, 0.16)),
            trust_learning_rate=float(np.clip(
                profile.trust_learning_rate * trust_learning_scale * jitter, 0.0, 0.5
            )),
            base_trade_fraction=float(np.clip(profile.base_trade_fraction * jitter, 0.003, 0.08)),
            post_probability=float(np.clip(profile.post_probability / max(jitter, 0.5), 0.0, 0.35)),
        )

    @staticmethod
    def _process_for_mode(default: str, mode: str) -> str:
        if mode == "all_risk_averse":
            return "risk_averse"
        if mode == "all_value":
            return "value_based"
        if mode == "all_trend":
            return "trend_following"
        if mode == "all_social":
            return "social_following"
        if mode == "all_aggressive":
            return "aggressive"
        return default

    def decide(self, day: int, observation: dict) -> tuple[list[Order], list[Message]]:
        orders: list[Order] = []
        message_candidates: list[Message] = []
        prices = observation["prices"]
        equity = self.portfolio.mark_to_market(prices)
        if equity <= 0:
            return orders, message_candidates
        social_env = observation.get("social_env")

        for asset, market in observation["market"].items():
            if self.cooldown_until.get(asset, -1) >= day or self.blocked_today.get(asset, False):
                continue
            price = float(market["price"])
            recent_return = float(observation["recent_return"].get(asset, 0.0))
            self._update_trust(asset, recent_return)
            private_value = self._private_signal(asset, float(market["fundamental"]))
            private_gap = float((private_value - price) / max(price, 1e-9))
            true_private_gap = float((float(market["fundamental"]) - price) / max(price, 1e-9))
            inbox = social_env.agent_messages(self.agent_id, asset, day) if hasattr(social_env, "agent_messages") else []
            attended = self._attend(inbox)
            social_content, social_proof, mean_trust = self._social_summary(attended)
            social_content *= float(np.clip(1.0 - self.skepticism, 0.0, 1.0))
            action, probabilities, diagnostic = self._choose_action(
                private_gap, recent_return, social_content, social_proof
            )
            order = self._make_order(asset, price, equity, action, probabilities)
            if order is not None:
                orders.append(order)
            message = self._choose_message(
                day, asset, action, private_gap, recent_return,
                social_content, social_proof, attended,
            )
            if message is not None:
                message_candidates.append(message)
            self._attended_last[asset] = attended

            harmful_exposure = 0.0
            if social_env is not None:
                _, harmful_exposure = social_env.agent_signal(self.agent_id, asset)
            self.last_belief_breakdown[asset] = {
                "social": float(social_content),
                "social_proof": float(social_proof),
                "private_signal": float(private_gap),
                "momentum": float(recent_return),
                "warning": float(self.warning_level.get(asset, 0.0)),
                "harmful_exposure": float(harmful_exposure),
                "total": float(diagnostic),
            }
            if hasattr(social_env, "record_decision"):
                social_env.record_decision({
                    "day": int(day),
                    "agent_id": self.agent_id,
                    "asset": asset,
                    "role": self.sub_role,
                    "policy": self.process,
                    "action": int(action),
                    "message_action": int(0 if message is None else (-1 if message.kind == "challenge" else 1)),
                    "private_bin": _signed_bin(private_gap, 0.006),
                    "true_private_bin": _signed_bin(true_private_gap, 0.006),
                    "social_bin": _signed_bin(social_content, 0.08),
                    "market_bin": _signed_bin(recent_return, 0.003),
                    "proof_bin": int(0 if social_proof < 0.25 else (1 if social_proof < 0.8 else 2)),
                    "attention_used": int(len(attended)),
                    "mean_sender_trust": float(mean_trust),
                    # Heuristic thresholds induce randomness, but they do not
                    # expose a calibrated choice distribution.  Recording NaN
                    # avoids falsely labelling them as uniform-QRE choices.
                    "choice_entropy": float("nan") if probabilities is None else _probability_entropy(probabilities),
                    "qre_beta": float(self.profile.qre_beta),
                    "conformity": float(self.profile.conformity),
                    "attention_capacity": int(self.profile.attention_capacity),
                })

        self.blocked_today.clear()
        if len(message_candidates) > 1:
            message_candidates.sort(key=lambda message: message.intensity, reverse=True)
            message_candidates = message_candidates[:1]
        return orders, message_candidates

    def _private_signal(self, asset: str, fundamental: float) -> float:
        noisy = fundamental * float(np.exp(self.rng.normal(0.0, self.profile.private_noise)))
        if asset not in self.private_values:
            self.private_values[asset] = noisy
        else:
            memory = 0.85 if self.sub_role in {"value_investor", "risk_averse"} else 0.94
            self.private_values[asset] = memory * self.private_values[asset] + (1.0 - memory) * noisy
        return float(self.private_values[asset])

    def _attend(self, inbox: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not inbox:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for event in inbox:
            sender = str(event["sender_id"])
            trust = float(self.sender_trust[sender])
            salience = (
                float(event["intensity"])
                * (0.35 + trust)
                * (1.0 + 0.35 * float(event.get("social_proof", 0.0)))
                + float(self.rng.gumbel(0.0, 0.08))
            )
            if self.sub_role == "risk_averse":
                salience *= 0.65 + 0.35 * float(event.get("confidence", 0.5))
            scored.append((salience, event))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [dict(event) for _, event in scored[: self.profile.attention_capacity]]

    def _social_summary(self, attended: list[dict[str, Any]]) -> tuple[float, float, float]:
        if not attended:
            return 0.0, 0.0, 0.5
        weights, signed, proof, trusts = [], [], [], []
        for event in attended:
            trust = float(self.sender_trust[str(event["sender_id"])])
            weight = max(1e-6, float(event["intensity"]) * trust)
            weights.append(weight)
            signed.append(weight * float(event["sentiment"]))
            proof.append(float(event.get("social_proof", 0.0)))
            trusts.append(trust)
        return (
            float(np.tanh(sum(signed) / max(sum(weights), 1e-9))),
            float(np.mean(proof)),
            float(np.mean(trusts)),
        )

    def _update_trust(self, asset: str, recent_return: float) -> None:
        if self.profile.trust_learning_rate <= 0:
            return
        for event in self._attended_last.get(asset, []):
            sender = str(event["sender_id"])
            sentiment = float(event["sentiment"])
            if abs(recent_return) < 5e-4 or abs(sentiment) < 0.05:
                outcome = 0.5
            else:
                outcome = 1.0 if sentiment * recent_return > 0 else 0.0
            old = float(self.sender_trust[sender])
            rate = self.profile.trust_learning_rate * min(1.0, abs(recent_return) * 40.0 + 0.2)
            self.sender_trust[sender] = float(np.clip(old + rate * (outcome - old), 0.05, 0.95))

    def _choose_action(
        self,
        private_gap: float,
        recent_return: float,
        social_content: float,
        social_proof: float,
    ) -> tuple[int, tuple[float, float, float] | None, float]:
        warning = max(self.warning_level.values(), default=0.0)
        if self.process == "aggressive":
            # Aggressive traders react to whichever cue is most salient, use
            # larger positions, and occasionally trade on impulse. They are
            # intentionally bounded and noisy rather than omniscient.
            impulse = (
                0.25 * private_gap
                + 0.70 * social_content
                + 1.8 * recent_return
                + float(self.rng.normal(0.0, 0.18))
            )
            threshold = float(self.rng.uniform(0.12, 0.42))
            if abs(impulse) > threshold:
                action = 1 if impulse > 0 else -1
            elif self.rng.random() < 0.08:
                action = int(self.rng.choice([-1, 1]))
            else:
                action = 0
            if warning > 0 and self.rng.random() < min(0.75, warning * 0.55):
                action = 0
            return action, None, float(impulse)

        if self.process == "trend_following":
            threshold = float(self.rng.uniform(0.006, 0.025))
            if abs(recent_return) >= threshold:
                direction = 1 if recent_return > 0 else -1
            elif abs(social_content) > 0.35 and self.rng.random() < 0.30:
                direction = 1 if social_content > 0 else -1
            else:
                direction = 0
            if direction != 0 and self.rng.random() < 0.18 + 0.35 * warning:
                direction = 0
            return direction, None, float(recent_return)

        if self.process == "social_following":
            popularity = np.tanh(social_proof)
            activation = 0.65 * social_content + 0.55 * popularity + 2.5 * max(recent_return, 0.0)
            if activation > float(self.rng.uniform(0.55, 1.35)):
                action = 1
            elif social_content < -0.30 or recent_return < -0.025:
                action = -1
            else:
                action = 0
            if warning > 0.0 and self.rng.random() < min(0.9, warning * 0.8):
                action = 0
            return action, None, float(activation)

        if self.process == "risk_averse":
            agrees = np.sign(private_gap) == np.sign(social_content) or abs(social_content) < 0.10
            aspiration = 0.012 * self.profile.loss_aversion
            if agrees and private_gap > aspiration:
                action = 1
            elif private_gap < -aspiration and recent_return < 0:
                action = -1
            else:
                action = 0
            return action, None, float(private_gap - 0.25 * social_content)

        # Value-based choice: each action receives a distinct payoff and a
        # quantal response leaves room for mistakes. This is not a scalar-score
        # threshold, although QRE is the mathematical implementation.
        coordination = self.profile.conformity * (
            social_content + 0.25 * np.tanh(social_proof)
        )
        expected = private_gap + 0.20 * recent_return
        utilities = np.array([
            -expected + coordination * (-1.0) - self.profile.loss_aversion * max(expected, 0.0),
            -0.002 * abs(recent_return),
            expected + coordination - self.profile.loss_aversion * max(-expected, 0.0),
        ])
        participation_cost = 0.50 + 0.08 * self.profile.loss_aversion
        utilities[0] -= participation_cost
        utilities[2] -= participation_cost
        utilities[0] -= 0.35 * warning
        utilities[2] -= 0.75 * warning
        logits = max(self.profile.qre_beta, 1.0) * utilities
        logits -= float(np.max(logits))
        probabilities_arr = np.exp(logits)
        probabilities_arr /= probabilities_arr.sum()
        action = int(self.rng.choice([-1, 0, 1], p=probabilities_arr))
        probabilities = tuple(float(value) for value in probabilities_arr)
        return action, probabilities, float(utilities[action + 1])

    def _make_order(
        self,
        asset: str,
        price: float,
        equity: float,
        action: int,
        probabilities: tuple[float, float, float] | None,
    ) -> Order | None:
        if action == 0:
            return None
        confidence = 0.5
        if probabilities is not None:
            confidence = max(probabilities) - min(probabilities)
        fraction = self.profile.base_trade_fraction * float(self.rng.lognormal(0.0, 0.35))
        fraction *= float(np.clip(self.risk_appetite / self._nominal_risk_appetite, 0.25, 3.0))
        fraction *= 0.55 + 1.4 * confidence
        fraction = float(np.clip(fraction, 0.002, 0.09))
        quantity = fraction * equity / max(price, 1e-9)
        side = "buy" if action > 0 else "sell"
        if side == "sell":
            quantity = min(quantity, self.portfolio.position(asset))
        if quantity <= 1e-8:
            return None
        return Order(self.agent_id, asset, side, float(quantity), is_harmful=False)

    def _choose_message(
        self,
        day: int,
        asset: str,
        action: int,
        private_gap: float,
        recent_return: float,
        social_content: float,
        social_proof: float,
        attended: list[dict[str, Any]],
    ) -> Message | None:
        probability = float(self.profile.post_probability)
        root = self.agent_id
        sentiment = float(np.sign(action if action != 0 else private_gap))
        confidence = float(np.clip(abs(private_gap) * 18.0 + abs(recent_return) * 8.0, 0.15, 1.0))
        intensity = float(np.clip(0.15 + confidence * 0.65, 0.05, 0.9))
        if self.process == "aggressive" and self.rng.random() < 0.35:
            if self.rng.random() >= probability:
                return None
            sentiment = float(self.rng.choice([-1.0, 1.0]))
            return Message(
                sender_id=self.agent_id,
                asset=asset,
                sentiment=sentiment,
                intensity=float(self.rng.uniform(0.05, 0.3)),
                is_harmful=False,
                is_bot=False,
                day=day,
                kind="post",
                root_sender_id=root,
                confidence=float(self.rng.uniform(0.15, 0.75)),
                social_proof=float(social_proof),
            )

        # Bounded-rational signaling game.  Silence, original posting,
        # resharing, and challenging are competing actions with attention and
        # reputational/coordination payoffs.  Logit choice leaves room for
        # mistakes and makes messages endogenous rather than an exogenous feed.
        candidates: list[tuple[str, float, float, float, str, float]] = []
        if sentiment != 0.0:
            post_utility = -0.55 + 0.65 * confidence + 0.25 * abs(private_gap) * 20.0
            candidates.append(("post", sentiment, intensity, confidence, root, post_utility))
        if self.process == "social_following" and attended:
            source = max(attended, key=lambda event: float(event["intensity"]) * (1.0 + float(event["social_proof"])))
            source_root = str(source.get("root_sender_id") or source["sender_id"])
            source_sentiment = float(source["sentiment"])
            source_confidence = float(source.get("confidence", confidence))
            reshare_utility = (
                -0.35
                + 0.45 * abs(social_content)
                + 0.50 * np.tanh(social_proof)
                + 0.35 * float(self.sender_trust[str(source["sender_id"])])
            )
            if source_sentiment != 0.0:
                candidates.append((
                    "reshare", source_sentiment, intensity, source_confidence,
                    source_root, float(reshare_utility),
                ))
        disagreement = (
            abs(social_content) > 0.15
            and np.sign(social_content) != np.sign(private_gap)
            and np.sign(private_gap) != 0
        )
        if self.process == "risk_averse" and disagreement:
            challenge_utility = 0.20 + 0.55 * abs(social_content) + 0.35 * confidence
            candidates.append((
                "challenge", float(-np.sign(social_content)), max(intensity, 0.35),
                confidence, root, float(challenge_utility),
            ))
        if not candidates:
            return None

        beta_message = max(0.7, 0.55 * self.profile.qre_beta)
        # The role-specific posting propensity is the prior odds of speaking;
        # game payoffs then redistribute that mass over message acts.
        silence_weight = max(1e-6, 1.0 - probability)
        candidate_prior = max(1e-6, probability / len(candidates))
        weights = [silence_weight] + [
            candidate_prior * float(np.exp(beta_message * candidate[5]))
            for candidate in candidates
        ]
        probabilities = np.asarray(weights, dtype=float)
        probabilities /= probabilities.sum()
        choice = int(self.rng.choice(len(probabilities), p=probabilities))
        if choice == 0:
            return None
        kind, sentiment, intensity, confidence, root, _ = candidates[choice - 1]
        return Message(
            sender_id=self.agent_id,
            asset=asset,
            sentiment=sentiment,
            intensity=intensity,
            is_harmful=False,
            is_bot=False,
            day=day,
            kind=kind,
            root_sender_id=root,
            confidence=confidence,
            social_proof=float(social_proof),
        )


class LegacyScoreLoggingRetailAgent:
    """Instrument the old weighted-score controller without changing its choice."""

    is_llm_controlled = False
    process = "legacy_score"

    def __init__(self, base: Any):
        self.base = base
        self.agent_id = base.agent_id
        self.role = base.role
        raw_role = str(getattr(base, "sub_role", "unknown"))
        self.sub_role = ROLE_ALIASES.get(raw_role, raw_role)
        self.is_harmful = False
        self.portfolio = base.portfolio

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def decide(self, day: int, observation: dict) -> tuple[list[Order], list[Message]]:
        orders, messages = self.base.decide(day, observation)
        order_by_asset = {order.asset: order for order in orders}
        social_env = observation.get("social_env")
        for asset in observation["market"]:
            breakdown = self.base.last_belief_breakdown.get(asset, {})
            order = order_by_asset.get(asset)
            action = 0 if order is None else (1 if order.side == "buy" else -1)
            inbox = (
                social_env.agent_messages(self.agent_id, asset, day)
                if hasattr(social_env, "agent_messages") else []
            )
            if hasattr(social_env, "record_decision"):
                social_env.record_decision({
                    "day": int(day),
                    "agent_id": self.agent_id,
                    "asset": asset,
                    "role": self.sub_role,
                    "policy": self.process,
                    "action": int(action),
                    "message_action": 0,
                    "private_bin": _signed_bin(float(breakdown.get("fundamental", 0.0)), 0.006),
                    "true_private_bin": _signed_bin(
                        (
                            float(observation["market"][asset]["fundamental"])
                            - float(observation["market"][asset]["price"])
                        ) / max(float(observation["market"][asset]["price"]), 1e-9),
                        0.006,
                    ),
                    "social_bin": _signed_bin(float(breakdown.get("social", 0.0)), 0.02),
                    "market_bin": _signed_bin(float(observation["recent_return"].get(asset, 0.0)), 0.003),
                    "proof_bin": 0,
                    "attention_used": int(len(inbox)),
                    "mean_sender_trust": 0.5,
                    "choice_entropy": float("nan"),
                    "qre_beta": float("nan"),
                    "conformity": float("nan"),
                    "attention_capacity": -1,
                })
        return orders, messages


def _signed_bin(value: float, threshold: float) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _probability_entropy(probabilities: Iterable[float]) -> float:
    return float(-sum(p * log2(p) for p in probabilities if p > 0))


def install_network_signal_game(env: Any, controller_mode: str = "mixed_roles") -> None:
    """Install the final social game and replace retail policy agents in-place."""
    controller_mode = normalize_controller_mode(controller_mode)
    if controller_mode not in CONTROLLER_MODES:
        raise ValueError(f"unknown controller mode: {controller_mode}")
    env.social = BoundedSocialEnv(
        env.graph,
        env.scenario,
        np.random.default_rng(int(env.seed) + 23017),
    )
    replacements: dict[int, Any] = {}
    for agent in list(env.society.retail):
        if controller_mode == "legacy_score":
            replacements[id(agent)] = LegacyScoreLoggingRetailAgent(agent)
        else:
            replacements[id(agent)] = BoundedRationalRetailAgent(
                agent, env.scenario, controller_mode=controller_mode
            )
    for attr in ("retail", "all_agents"):
        agents = getattr(env.society, attr)
        for index, agent in enumerate(agents):
            replacement = replacements.get(id(agent))
            if replacement is not None:
                agents[index] = replacement
