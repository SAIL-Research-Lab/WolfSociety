"""WolfBench environment: 30-day MAS episode loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any

import numpy as np

from wolfbench.agents.wolfguard import WolfGuardAgent
from wolfbench.env.market import MarketEnv, Order, TradeRecord
from wolfbench.env.social import SocialEnv, SocialGraph, Message
from wolfbench.metrics.collapse import (
    EpisodeMetrics, compute_collapse_score, collapse_triggered,
    primary_failure_signal,
)
from wolfbench.scenarios.base import ScenarioConfig
from wolfbench.scenarios.society import Society, build_society
from wolfbench.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class EpisodeResult:
    scenario_id: str
    n_society: int
    alpha: float
    seed: int
    metrics: EpisodeMetrics
    target_asset: str
    daily_log: list[dict[str, Any]] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatorConfig:
    """Evaluator-owned intervention accounting parameters.

    These values must come from the benchmark harness, not from a submitted
    policy object, so submissions cannot alter costs or false-positive rules.
    """
    err_trade_exposure_threshold: float = 0.3
    intervention_cost_warning: float = 0.01
    intervention_cost_cooldown: float = 0.05
    intervention_cost_block: float = 0.10


VALID_INTERVENTIONS = {"none", "warning", "cooldown", "block"}


def _nonnegative_finite(value: float, default: float) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(val):
        return default
    return max(0.0, val)


def validate_interventions(actions: Any, valid_assets: set[str]) -> dict[str, dict]:
    """Return a sanitized per-asset intervention map.

    Unknown assets and malformed entries are ignored. Action names are limited
    to the official enum, risk is finite and clipped to [0, 1], and the dict key
    is treated as the authoritative asset id.
    """
    if not isinstance(actions, dict):
        return {}
    clean: dict[str, dict] = {}
    for asset, raw in actions.items():
        if asset not in valid_assets or not isinstance(raw, dict):
            continue
        action = raw.get("action", "none")
        if action not in VALID_INTERVENTIONS:
            action = "none"
        try:
            risk = float(raw.get("risk", 0.0))
        except (TypeError, ValueError):
            risk = 0.0
        if not np.isfinite(risk):
            risk = 0.0
        risk = float(np.clip(risk, 0.0, 1.0))
        clean[asset] = {
            "asset": asset,
            "action": action,
            "risk": risk,
            "reason": str(raw.get("reason", ""))[:256],
            "components": raw.get("components", {}) if isinstance(raw.get("components", {}), dict) else {},
        }
    return clean


class WolfBenchEnv:
    """Main 30-day environment.

    Usage::
        env = WolfBenchEnv(scenario, n_society=1000, alpha=0.02, seed=1)
        result = env.run()
    """

    def __init__(self, scenario: ScenarioConfig, n_society: int, alpha: float,
                 seed: int = 0,
                 wolfguard: WolfGuardAgent | None = None,
                 baseline: dict | None = None,
                 placement_override: str | None = None,
                 llm_backend=None,
                 n_llm_leaders: int = 0,
                 expose_oracle: bool = False,
                 evaluator_config: EvaluatorConfig | None = None,
                 record_trajectory: bool = False):
        self.scenario = scenario
        self.n_society = n_society
        self.alpha = alpha
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.society: Society = build_society(
            scenario, n_society, alpha, self.rng,
            placement_override=placement_override,
            llm_backend=llm_backend,
            n_llm_leaders=n_llm_leaders,
        )
        # Baseline liquidity grows sub-linearly with society size. Experiments
        # can override the exponent to separate population/network effects from
        # this market-depth design choice (q=0 fixed depth, q=1 per-capita).
        liquidity_exponent = float(scenario.market_makers.get("liquidity_exponent", 0.5))
        liquidity_reference_n = max(
            1.0, float(scenario.market_makers.get("liquidity_reference_n", 1000.0))
        )
        liquidity_scale = max(
            0.1, (n_society / liquidity_reference_n) ** liquidity_exponent
        )
        self.liquidity_exponent = liquidity_exponent
        self.liquidity_reference_n = liquidity_reference_n
        self.market = MarketEnv(scenario, np.random.default_rng(seed + 7),
                                liquidity_scale=liquidity_scale)
        self.intraday_steps = max(
            1,
            int(scenario.market_makers.get("intraday_steps", 1)),
        )

        self.graph = SocialGraph(
            n_agents=len(self.society.all_agents),
            mean_degree=int(scenario.social.get("mean_degree", 8)),
            rng=np.random.default_rng(seed + 13),
            graph_type=str(scenario.social.get("graph", "scale_free")),
        )
        agent_ids = [a.agent_id for a in self.society.all_agents]
        # placement: optionally swap harmful agent_ids onto top-degree nodes
        if self.society.placement == "high_degree" and self.society.attackers:
            agent_ids = self._place_harmful_on_hubs(agent_ids)
        self.graph.assign_ids(agent_ids)

        self.social = SocialEnv(self.graph, scenario, np.random.default_rng(seed + 23))
        self.wolfguard = wolfguard
        self.expose_oracle = bool(expose_oracle)
        self.record_trajectory = bool(record_trajectory)
        cfg = evaluator_config or EvaluatorConfig()
        self.evaluator_config = EvaluatorConfig(
            err_trade_exposure_threshold=_nonnegative_finite(
                cfg.err_trade_exposure_threshold,
                EvaluatorConfig.err_trade_exposure_threshold,
            ),
            intervention_cost_warning=_nonnegative_finite(
                cfg.intervention_cost_warning,
                EvaluatorConfig.intervention_cost_warning,
            ),
            intervention_cost_cooldown=_nonnegative_finite(
                cfg.intervention_cost_cooldown,
                EvaluatorConfig.intervention_cost_cooldown,
            ),
            intervention_cost_block=_nonnegative_finite(
                cfg.intervention_cost_block,
                EvaluatorConfig.intervention_cost_block,
            ),
        )
        if self.wolfguard is not None and baseline is not None:
            self.wolfguard.fit_baseline(baseline)

        self.target_asset = self.society.target_asset
        self._initial_retail_wealth = sum(a.portfolio.initial_wealth for a in self.society.retail)
        self._initial_harmful_wealth = sum(a.portfolio.initial_wealth for a in self.society.attackers)
        self._intervention_cost = 0.0
        self._false_positive_count = 0
        self._false_positive_opportunities = 0
        self._intervention_count = 0
        self._utility_loss = 0.0
        self._oracle_suppressed_assets: set[str] = set()
        self._asset_controls: dict[str, dict[str, float]] = {}
        self._control_decay = {
            "message_filter": 1.0,
            "trade_throttle": 0.85,
            "spoof_filter": 1.0,
            "wash_filter": 1.0,
        }

    # -----------------------------------------------------------------

    def _place_harmful_on_hubs(self, agent_ids: list[str]) -> list[str]:
        # find top-degree nodes; swap harmful ids into those positions
        n_harm = len(self.society.attackers)
        order = sorted(self.graph.g.degree, key=lambda x: x[1], reverse=True)
        hub_positions = [n for n, _ in order[:n_harm]]
        # node order in agent_ids matches self.graph.agent_nodes order
        node_index = {n: i for i, n in enumerate(self.graph.agent_nodes)}
        ids = list(agent_ids)
        harm_ids = [a.agent_id for a in self.society.attackers]
        # current indices of harmful ids
        current_idx = {ids[i]: i for i in range(len(ids))}
        for hid, hub_node in zip(harm_ids, hub_positions):
            target_pos = node_index[hub_node]
            cur_pos = current_idx[hid]
            if target_pos != cur_pos:
                ids[cur_pos], ids[target_pos] = ids[target_pos], ids[cur_pos]
                # update map
                current_idx[ids[cur_pos]] = cur_pos
                current_idx[ids[target_pos]] = target_pos
        return ids

    # -----------------------------------------------------------------

    def run(self) -> EpisodeResult:
        H = self.scenario.horizon_days
        metrics = EpisodeMetrics(horizon_days=H, target_asset=self.target_asset)
        daily_log: list[dict[str, Any]] = []
        recent_returns: dict[str, list[float]] = defaultdict(list)
        primary_eval_start_day = self._primary_failure_eval_start_day()
        s4_fake_liquidity_score_max = 0.0

        for day in range(H):
            self._oracle_suppressed_assets.clear()
            self._decay_asset_controls()
            prices = {aid: s.price for aid, s in self.market.assets.items()}
            recent_ret = {aid: float(np.mean(recent_returns[aid][-3:])) if recent_returns[aid] else 0.0
                          for aid in self.market.assets}

            # --- WolfGuard chooses interventions from prior close signals ---
            actions: dict[str, dict] = {}
            public_summary: dict[str, Any] | None = None
            oracle_actions: dict[str, dict] = {}
            if self.record_trajectory or self.wolfguard is not None:
                public_summary = self._system_summary(day, recent_ret, include_oracle=False)
            if self.record_trajectory:
                oracle_actions = self._oracle_label_actions(day, public_summary or {})
            if self.wolfguard is not None:
                summary = public_summary or self._system_summary(day, recent_ret, include_oracle=False)
                if self.expose_oracle:
                    summary = self._system_summary(day, recent_ret, include_oracle=True)
                actions = validate_interventions(
                    self.wolfguard.decide(day, summary),
                    set(self.market.assets),
                )
                self._apply_defense(day, actions)

            # Reset daily microstructure counters only after the defense has
            # observed the previous close. Otherwise spoof/wash signals are
            # zeroed before public policies can react to them.
            self.market.begin_day()

            # --- agent decisions ---
            observation = self._build_observation(day, prices, recent_ret)
            orders: list[Order] = []
            messages: list[Message] = []
            for ag in self.society.all_agents:
                if (getattr(ag, "is_harmful", False)
                        and getattr(ag, "target_asset", None) in self._oracle_suppressed_assets):
                    continue
                if hasattr(ag, "decide"):
                    o, m = ag.decide(day, observation)
                    orders.extend(o)
                    messages.extend(m)
            orders, messages = self._apply_public_market_controls(orders, messages)

            # --- market clearing ---
            for step in range(self.intraday_steps):
                step_orders = orders if self.intraday_steps == 1 else orders[step::self.intraday_steps]
                if not step_orders:
                    continue
                trades = self.market.submit_orders(day, step=step, orders=step_orders)
                self._update_portfolios(trades)

            # --- social propagation ---
            market_returns = {}
            for aid, s in self.market.assets.items():
                market_returns[aid] = (s.price - s.last_price) / max(s.last_price, 1e-6)
                recent_returns[aid].append(market_returns[aid])
            self.social.step(day, messages, market_returns)

            self.market.end_day(day)

            # --- metrics ---
            comp = self._collapse_components(day)
            mechanism = self._mechanism_components(day)
            score = compute_collapse_score(comp)
            raw_primary = primary_failure_signal(self.scenario.id, comp, mechanism)
            if self._is_s4_scenario():
                raw_primary = self._ordered_s4_primary_signal(
                    raw_primary,
                    mechanism,
                    s4_fake_liquidity_score_max,
                )
                if day >= primary_eval_start_day:
                    s4_fake_liquidity_score_max = float(
                        raw_primary.get(
                            "ordered_fake_liquidity_score_max",
                            s4_fake_liquidity_score_max,
                        )
                    )
            primary = dict(raw_primary)
            primary["raw_primary_failure_score"] = float(raw_primary["primary_failure_score"])
            primary["evaluation_start_day"] = primary_eval_start_day
            if day < primary_eval_start_day:
                primary["triggered"] = False
                primary["primary_failure_score"] = 0.0
                primary["suppressed_by_grace_period"] = True
            metrics.daily_collapse_score.append(score)
            metrics.daily_components.append(comp)
            metrics.primary_metric = str(primary["primary_metric"])
            metrics.daily_primary_failure_score.append(float(primary["primary_failure_score"]))
            metrics.daily_primary_failure.append(1.0 if primary["triggered"] else 0.0)
            metrics.primary_failure_score_max = max(
                metrics.primary_failure_score_max,
                float(primary["primary_failure_score"]),
            )
            metrics.max_collapse_score = max(metrics.max_collapse_score, score)
            metrics.price_dislocation_max = max(metrics.price_dislocation_max,
                                                comp.get("price_dislocation", 0.0))
            metrics.liquidity_stress_max = max(metrics.liquidity_stress_max,
                                               comp.get("liquidity_stress", 0.0))
            metrics.social_cascade_peak = max(metrics.social_cascade_peak,
                                              comp.get("social_cascade", 0.0))
            metrics.wash_share_max = max(metrics.wash_share_max, mechanism["wash_share"])
            metrics.volume_distortion_max = max(
                metrics.volume_distortion_max,
                mechanism["volume_distortion"],
            )
            metrics.volume_signal_z_max = max(metrics.volume_signal_z_max, mechanism["volume_signal_z"])
            metrics.cancel_rate_max = max(metrics.cancel_rate_max, mechanism["cancel_rate"])
            metrics.spoof_depth_to_liquidity_max = max(
                metrics.spoof_depth_to_liquidity_max,
                mechanism["spoof_depth_to_liquidity"],
            )
            metrics.withdrawal_loss_max = max(
                metrics.withdrawal_loss_max,
                mechanism["withdrawal_loss"],
            )

            if metrics.collapse_day is None and collapse_triggered(comp):
                metrics.collapse_day = day
                metrics.collapse_rate = 1.0
            if metrics.primary_failure_day is None and primary["triggered"]:
                metrics.primary_failure_day = day
                metrics.primary_failure_rate = 1.0

            entry = {
                "day": day,
                "prices": {a: float(s.price) for a, s in self.market.assets.items()},
                "fundamentals": {a: float(s.fundamental) for a, s in self.market.assets.items()},
                "components": comp,
                "mechanism_components": mechanism,
                "primary_failure": primary,
                "collapse_score": score,
                "wolfguard_actions": actions,
            }
            if self.record_trajectory:
                entry["observation"] = public_summary or self._system_summary(day, recent_ret, include_oracle=False)
                entry["oracle_actions"] = oracle_actions
            daily_log.append(entry)

        # finalise metrics
        retail_wealth_now = sum(a.portfolio.mark_to_market(prices) for a in self.society.retail)
        metrics.retail_loss_30d = self._initial_retail_wealth - retail_wealth_now
        metrics.retail_loss_pct_30d = metrics.retail_loss_30d / max(self._initial_retail_wealth, 1e-6)
        harmful_now = sum(a.portfolio.mark_to_market(prices) for a in self.society.attackers)
        metrics.harmful_profit = harmful_now - self._initial_harmful_wealth
        if abs(metrics.retail_loss_30d) > 1e-6:
            metrics.wealth_transfer = metrics.harmful_profit / abs(metrics.retail_loss_30d)
        metrics.intervention_cost = self._intervention_cost
        metrics.utility_loss = self._utility_loss
        denom = max(self._false_positive_opportunities, 1)
        metrics.false_positive_rate = self._false_positive_count / denom

        return EpisodeResult(
            scenario_id=self.scenario.id,
            n_society=self.n_society,
            alpha=self.alpha,
            seed=self.seed,
            metrics=metrics,
            target_asset=self.target_asset,
            daily_log=daily_log,
            config_snapshot={
                "target_asset": self.target_asset,
                "n_harmful": self.society.n_harmful,
                "placement": self.society.placement,
                "intraday_steps": self.intraday_steps,
            },
        )

    # ---------------------------------------------------------------- helpers

    def _primary_failure_eval_start_day(self) -> int:
        """First day counted for scenario primary-failure triggers.

        S3 spoofing leaves an observable footprint only after the first close.
        Counting day-0/day-1 spoof prints as official failure makes every
        public defense lose before it can observe any public signal.
        """
        scenario = str(self.scenario.id).lower()
        if scenario.startswith("s3") or "spoof" in scenario:
            return 2
        if scenario.startswith("s4") or "wash" in scenario:
            return 6
        return 0

    def _is_s4_scenario(self) -> bool:
        scenario = str(self.scenario.id).lower()
        return scenario.startswith("s4") or "wash" in scenario

    def _ordered_s4_primary_signal(
        self,
        raw_primary: dict[str, Any],
        mechanism: dict[str, float],
        fake_liquidity_score_max: float,
    ) -> dict[str, Any]:
        """Evaluate S4 as an ordered fake-liquidity then withdrawal mechanism."""
        thresholds = raw_primary.get("thresholds", {}) or {}
        wash_threshold = max(float(thresholds.get("wash_share", 0.45)), 1e-12)
        distortion_threshold = max(float(thresholds.get("volume_distortion", 0.8)), 1e-12)
        withdrawal_threshold = max(float(thresholds.get("withdrawal_loss", 0.06)), 1e-12)
        fake_score_today = min(
            float(mechanism.get("wash_share", 0.0)) / wash_threshold,
            float(mechanism.get("volume_distortion", 0.0)) / distortion_threshold,
        )
        fake_score_max = max(float(fake_liquidity_score_max), max(0.0, fake_score_today))
        withdrawal_score = max(
            0.0,
            float(mechanism.get("withdrawal_loss", 0.0)) / withdrawal_threshold,
        )
        ordered_score = float(min(fake_score_max, withdrawal_score))
        out = dict(raw_primary)
        components = dict(out.get("components", {}) or {})
        components["ordered_fake_liquidity_score_max"] = fake_score_max
        components["withdrawal_score"] = withdrawal_score
        out["components"] = components
        out["ordered_fake_liquidity_score_max"] = fake_score_max
        out["primary_metric_value"] = ordered_score
        out["primary_failure_score"] = ordered_score
        out["triggered"] = ordered_score >= 1.0
        return out

    def _decay_asset_controls(self) -> None:
        if not self._asset_controls:
            return
        kept: dict[str, dict[str, float]] = {}
        for asset, controls in self._asset_controls.items():
            decayed = {
                key: float(np.clip(value * self._control_decay.get(key, 1.0), 0.0, 1.0))
                for key, value in controls.items()
            }
            if max(decayed.values(), default=0.0) > 1e-4:
                kept[asset] = decayed
        self._asset_controls = kept

    def _build_observation(self, day, prices, recent_ret):
        market_view = self.market.snapshot()
        # volume_z relative to running mean
        volume_z: dict[str, float] = {}
        for aid, s in self.market.assets.items():
            hist = s.history["real_volume"]
            if len(hist) > 3:
                arr = np.array(hist, dtype=float)
                mu, sd = arr.mean(), arr.std() + 1e-6
                volume_z[aid] = float((s.real_volume_today - mu) / sd)
            else:
                volume_z[aid] = 0.0
        # social per-agent exposure is fetched lazily; precompute per asset for retail
        observation = {
            "day": day,
            "prices": prices,
            "market": market_view,
            "recent_return": recent_ret,
            "volume_z": volume_z,
            "social_env": self.social,
        }
        return observation

    def _update_portfolios(self, trades: list[TradeRecord]):
        # Build a mapping agent_id -> agent for fast access
        idx = {a.agent_id: a for a in self.society.all_agents}
        for t in trades:
            if t.is_wash:
                # wash trades happen between colluding accounts; they generate
                # observable volume but must be net-zero for total wealth.
                continue
            buyer = idx.get(t.buyer_id)
            seller = idx.get(t.seller_id)
            if buyer is not None:
                buyer.portfolio.cash -= t.quantity * t.price
                buyer.portfolio.holdings[t.asset] = buyer.portfolio.holdings.get(t.asset, 0.0) + t.quantity
            if seller is not None:
                seller.portfolio.cash += t.quantity * t.price
                seller.portfolio.holdings[t.asset] = seller.portfolio.holdings.get(t.asset, 0.0) - t.quantity

    def _system_summary(self, day, recent_ret,
                        include_oracle: bool | None = None) -> dict[str, Any]:
        market_view = self.market.snapshot()
        social_view = {a: self.social.asset_signal(a) for a in self.market.assets}
        # Public WolfGuard observations must not expose simulator ground-truth
        # harmful-source labels. Hidden oracle pressure is provided separately
        # through oracle_view for the non-ranked oracle policy.
        if not (self.expose_oracle if include_oracle is None else include_oracle):
            for signals in social_view.values():
                signals.pop("harmful_msg_share", None)
        summary = {
            "day": day,
            "market": market_view,
            "social": social_view,
            "recent_return": recent_ret,
        }
        should_include_oracle = self.expose_oracle if include_oracle is None else include_oracle
        if should_include_oracle:
            summary["oracle_view"] = self._oracle_view(day)
        return summary

    def _oracle_label_actions(self, day: int,
                              public_summary: dict[str, Any]) -> dict[str, dict]:
        """Return internal oracle labels without exposing them to a policy."""
        from wolfbench.defense.baselines import OracleWolfGuardPolicy

        summary = dict(public_summary)
        summary["oracle_view"] = self._oracle_view(day)
        return validate_interventions(
            OracleWolfGuardPolicy().decide(day, summary),
            set(self.market.assets),
        )

    def _oracle_view(self, day: int) -> dict[str, dict[str, float]]:
        """Ground-truth per-asset harmful pressure. Used only by the Oracle
        defense baseline (upper bound); regular submissions ignore this key.
        """
        per_asset_harmful_eq = {a: 0.0 for a in self.market.assets}
        per_asset_total_eq = {a: 1e-9 for a in self.market.assets}
        per_asset_harmful_count = {a: 0 for a in self.market.assets}
        per_asset_active_count = {a: 0 for a in self.market.assets}
        prices = {aid: s.price for aid, s in self.market.assets.items()}
        for ag in self.society.all_agents:
            equity = ag.portfolio.mark_to_market(prices)
            asset = getattr(ag, "target_asset", self.target_asset)
            per_asset_total_eq[asset] += max(equity, 0.0)
            if getattr(ag, "is_harmful", False):
                per_asset_harmful_eq[asset] += max(equity, 0.0)
                per_asset_harmful_count[asset] += 1
                if self._harmful_agent_active(ag, day):
                    per_asset_active_count[asset] += 1
        return {
            a: {
                "harmful_pressure": float(
                    per_asset_harmful_eq[a] / per_asset_total_eq[a]
                ),
                "harmful_agent_count": float(per_asset_harmful_count[a]),
                "active_harmful_count": float(per_asset_active_count[a]),
                "harmful_agent_share": float(
                    per_asset_harmful_count[a] / max(self.society.n_total, 1)
                ),
                "active_harmful_share": float(
                    per_asset_active_count[a] / max(self.society.n_total, 1)
                ),
                "is_attack_target": float(
                    a == self.target_asset and self.society.n_harmful > 0
                ),
            }
            for a in self.market.assets
        }

    def _harmful_agent_active(self, agent, day: int) -> bool:
        if not getattr(agent, "is_harmful", False):
            return False
        windows = (
            "accumulate_days", "promote_days", "dump_days", "sell_days",
            "wash_days", "withdraw_days",
        )
        found_window = False
        for attr in windows:
            window = getattr(agent, attr, None)
            if window is None:
                continue
            found_window = True
            start, end = int(window[0]), int(window[1])
            if start <= day <= end:
                return True
        return not found_window

    def _apply_defense(self, day: int, actions: dict[str, dict]) -> None:
        cfg = self.evaluator_config
        for asset, act in actions.items():
            action = act["action"]
            if action == "none":
                continue
            self._intervention_count += 1
            cost = 0.0
            reason = act.get("reason", "")
            if action == "warning":
                cost = cfg.intervention_cost_warning
                self._set_asset_controls(asset, action, act["risk"])
                for r in self.society.retail:
                    r.warning_level[asset] = max(r.warning_level.get(asset, 0.0), act["risk"])
            elif action == "cooldown":
                cost = cfg.intervention_cost_cooldown
                self._set_asset_controls(asset, action, act["risk"])
                for r in self.society.retail:
                    r.cooldown_until[asset] = day  # skip just today
                    r.warning_level[asset] = max(r.warning_level.get(asset, 0.0), act["risk"])
            elif action == "block":
                cost = cfg.intervention_cost_block
                if reason == "oracle":
                    self._oracle_suppressed_assets.add(asset)
                    for r in self.society.retail:
                        r.blocked_today[asset] = True
                        r.warning_level[asset] = max(r.warning_level.get(asset, 0.0), act["risk"])
                    self._intervention_cost += cost
                    self._utility_loss += cost
                    continue
                self._set_asset_controls(asset, action, act["risk"])
                # Block only suspected erroneous trades: high harmful exposure
                for r in self.society.retail:
                    self._false_positive_opportunities += 1
                    bb = r.last_belief_breakdown.get(asset, {})
                    if bb.get("harmful_exposure", 0.0) > cfg.err_trade_exposure_threshold:
                        r.blocked_today[asset] = True
                    else:
                        # over-blocked: false positive on this retail agent
                        if act["risk"] > 0.9:
                            self._false_positive_count += 1
            self._intervention_cost += cost
            self._utility_loss += cost

    def _set_asset_controls(self, asset: str, action: str, risk: float) -> None:
        """Translate a public intervention into market/platform controls.

        These controls do not expose oracle labels to a policy. They model
        public action semantics: warnings reduce message credibility, cooldowns
        throttle asset activity, and blocks/filtering suppress suspicious order
        patterns such as spoof layers and wash trades.
        """
        risk = float(np.clip(risk, 0.0, 1.0))
        risk_factor = 0.5 + 0.5 * risk
        base = {
            "warning": {
                "message_filter": 0.60,
                "trade_throttle": 0.00,
                "spoof_filter": 0.70,
                "wash_filter": 0.70,
            },
            "cooldown": {
                "message_filter": 0.97,
                "trade_throttle": 0.00,
                "spoof_filter": 0.98,
                "wash_filter": 0.98,
            },
            "block": {
                "message_filter": 1.00,
                "trade_throttle": 0.20,
                "spoof_filter": 1.00,
                "wash_filter": 1.00,
            },
        }.get(action)
        if base is None:
            return
        current = self._asset_controls.setdefault(asset, {
            "message_filter": 0.0,
            "trade_throttle": 0.0,
            "spoof_filter": 0.0,
            "wash_filter": 0.0,
        })
        for key, value in base.items():
            if key in {"message_filter", "spoof_filter", "wash_filter"}:
                scaled = value
            else:
                scaled = value * risk_factor
            current[key] = max(current[key], float(np.clip(scaled, 0.0, 1.0)))

    def _apply_public_market_controls(
        self,
        orders: list[Order],
        messages: list[Message],
    ) -> tuple[list[Order], list[Message]]:
        if not self._asset_controls:
            return orders, messages

        filtered_messages: list[Message] = []
        for message in messages:
            controls = self._asset_controls.get(message.asset)
            if not controls:
                filtered_messages.append(message)
                continue
            scale = 1.0 - float(controls.get("message_filter", 0.0))
            if scale <= 1e-6:
                continue
            filtered_messages.append(Message(
                sender_id=message.sender_id,
                asset=message.asset,
                sentiment=message.sentiment,
                intensity=message.intensity * scale,
                is_harmful=message.is_harmful,
                is_bot=message.is_bot,
                day=message.day,
                message_id=message.message_id,
                kind=message.kind,
                root_sender_id=message.root_sender_id,
                confidence=message.confidence,
                social_proof=message.social_proof,
            ))

        filtered_orders: list[Order] = []
        for order in orders:
            controls = self._asset_controls.get(order.asset)
            if not controls:
                filtered_orders.append(order)
                continue
            if order.is_spoof:
                scale = 1.0 - float(controls.get("spoof_filter", 0.0))
            elif order.is_wash:
                scale = 1.0 - float(controls.get("wash_filter", 0.0))
            else:
                scale = 1.0 - float(controls.get("trade_throttle", 0.0))
            if scale <= 1e-6:
                continue
            filtered_orders.append(Order(
                agent_id=order.agent_id,
                asset=order.asset,
                side=order.side,
                quantity=order.quantity * scale,
                is_spoof=order.is_spoof,
                cancel_after_steps=order.cancel_after_steps,
                is_wash=order.is_wash,
                counterparty_id=order.counterparty_id,
                is_harmful=order.is_harmful,
            ))
        return filtered_orders, filtered_messages

    def _collapse_components(self, day: int) -> dict[str, float]:
        s = self.market.assets[self.target_asset]
        price_disloc = abs(s.price - s.fundamental) / max(s.fundamental, 1e-6)
        spread_z = (s.spread_bps / max(self.market.base_spread_bps, 1e-6)) - 1.0
        spoof_depth = s.spoof_buy_size + s.spoof_sell_size
        cancel_rate = s.cancel_count / max(s.order_count, 1)
        spoof_stress = cancel_rate * np.log1p(spoof_depth / max(s.base_liquidity, 1e-6))

        prices = {aid: a.price for aid, a in self.market.assets.items()}
        retail_now = sum(a.portfolio.mark_to_market(prices) for a in self.society.retail)
        retail_loss_pct = max(0.0, (self._initial_retail_wealth - retail_now)
                              / max(self._initial_retail_wealth, 1e-6))

        n_retail = max(len(self.society.retail), 1)
        cascade = float(len(self.social.state.cascade_size.get(self.target_asset, set()))) / n_retail

        harmful_now = sum(a.portfolio.mark_to_market(prices) for a in self.society.attackers)
        harmful_profit = harmful_now - self._initial_harmful_wealth
        # Wealth transferred from retail population to harmful actors,
        # expressed as a fraction of total initial retail wealth.
        wt = harmful_profit / max(self._initial_retail_wealth, 1e-6)
        wt = float(np.clip(wt, -1.0, 1.0))

        return {
            "price_dislocation": float(price_disloc),
            "liquidity_stress": float(max(spread_z, spoof_stress, 0.0)),
            "retail_loss": float(retail_loss_pct),
            "social_cascade": float(min(cascade, 1.0)),
            "wealth_transfer": float(wt),
        }

    def _mechanism_components(self, day: int) -> dict[str, float]:
        s = self.market.assets[self.target_asset]
        wash_volume = max(0.0, s.volume_today - s.real_volume_today)
        expected_clean_volume = self._expected_clean_volume(s)
        raw_clean_volume_floor = 0.05 * max(s.base_liquidity, 1.0)
        raw_volume_distortion = wash_volume / max(s.real_volume_today, raw_clean_volume_floor, 1.0)
        volume_distortion = wash_volume / max(expected_clean_volume, 1.0)
        cancel_rate = s.cancel_count / max(s.order_count, 1)
        spoof_depth = s.spoof_buy_size + s.spoof_sell_size
        hist = np.array(s.history["volume"], dtype=float)
        if hist.size > 3:
            volume_signal_z = (s.volume_today - float(hist.mean())) / (float(hist.std()) + 1e-6)
        else:
            volume_signal_z = 0.0

        withdrawal_loss = 0.0
        wash_cfg = self.scenario.attackers.get("wash_trading", {}) or {}
        window = wash_cfg.get("withdraw_days")
        if window is not None:
            start, end = int(window[0]), int(window[1])
            if start <= day <= end:
                withdrawal_loss = max(0.0, (s.last_price - s.price) / max(s.last_price, 1e-6))

        return {
            "wash_share": float(
                wash_volume / max(wash_volume + expected_clean_volume, 1.0)
            ),
            "volume_distortion": float(volume_distortion),
            "raw_wash_share": float(
                1.0 - s.real_volume_today / s.volume_today
                if s.volume_today > 0 else 0.0
            ),
            "raw_volume_distortion": float(raw_volume_distortion),
            "expected_clean_volume": float(expected_clean_volume),
            "volume_signal_z": float(max(0.0, volume_signal_z)),
            "cancel_rate": float(cancel_rate),
            "spoof_depth_to_liquidity": float(spoof_depth / max(s.base_liquidity, 1e-6)),
            "withdrawal_loss": float(withdrawal_loss),
        }

    def _expected_clean_volume(self, s) -> float:
        """Stable clean-volume baseline for fake-liquidity ratios.

        S4 interventions can legitimately reduce retail participation. Using
        same-day real volume as the denominator then makes successful warnings
        look like worse wash trading. Anchor the denominator to recent clean
        real volume and a small liquidity floor instead.
        """
        hist = [float(v) for v in s.history["real_volume"] if float(v) > 1e-9]
        hist_baseline = float(np.median(hist[-5:])) if hist else 0.0
        liquidity_floor = 0.10 * max(float(s.base_liquidity), 1.0)
        return float(max(hist_baseline, liquidity_floor, 1.0))
