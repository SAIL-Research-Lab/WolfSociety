"""CPU checks of direct model decisions and the social/market boundary.

These tests use explicitly marked fixtures, not model-quality evidence. They
exercise the same episode loop and action contract used by the real backend.
"""
import copy
from dataclasses import asdict
import inspect
import json
import re

import pytest
from jsonschema import ValidationError

from wolfbench.agents import attackers, market_maker, retail, social_game
from wolfbench.env.social import Message
from wolfbench.llm_runtime import simulation
from wolfbench.llm_runtime.backend import MockBackend
from wolfbench.llm_runtime.simulation import LanguageSocietyEnv, run_episode


def hold(memory=""):
    return {
        "orders": [],
        "message": {"action": "none", "asset": "", "text": "", "sentiment": 0,
                    "intensity": 0, "confidence": .5, "source_message_id": ""},
        "memory": memory,
    }


def order(asset, side, quantity, *, kind="real", counterparty=""):
    return {"asset": asset, "side": side, "quantity": quantity, "kind": kind,
            "counterparty_id": counterparty}


def message(action, asset, text, *, source=""):
    return {"action": action, "asset": asset, "text": text, "sentiment": .8,
            "intensity": 1, "confidence": .75, "source_message_id": source}


def config(**kwargs):
    result = {"scenario": "s1", "n_society": 12, "alpha": .25,
              "seed": 37, "horizon_days": 2, "controller": "llm",
              "role_mode": "all_value", "episode_id": "cpu-test"}
    result.update(kwargs)
    return result


def prohibit_rule_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A rule controller was called in the LLM society")

    monkeypatch.setattr(simulation, "numerical_control", forbidden)
    for module in (attackers, market_maker, retail, social_game):
        for candidate in vars(module).values():
            if inspect.isclass(candidate) and candidate.__module__ == module.__name__:
                if callable(getattr(candidate, "decide", None)):
                    monkeypatch.setattr(candidate, "decide", forbidden)


@pytest.mark.parametrize("size", [1000, 2000])
def test_entire_population_has_one_independent_model_decision_per_round(size, monkeypatch):
    prohibit_rule_calls(monkeypatch)
    backend = MockBackend(responder=lambda request, schema: hold("Independent fixture memory"))
    result = run_episode(config(n_society=size, alpha=.05), backend)

    assert result["n_llm"] == size
    assert result["n_benign_llm"] + result["n_harmful_llm"] == size
    assert result["n_harmful_llm"] == round(size * .05)
    assert result["n_llm_decisions"] == result["n_population_decisions"] == 2 * size
    assert result["rule_fallback_count"] == 0 and result["simulated"] is True
    assert backend.stats["simulated_requests"] == 2 * size
    assert len(result["backend_audit"]) == len(result["decision_log"]) == 2 * size
    assert len({r["request_id"] for r in result["request_log"]}) == 2 * size
    populations = []
    for day in (0, 1):
        rows = [json.loads(r["user"]) for r in result["request_log"]
                if json.loads(r["user"])["day"] == day]
        assert len(rows) == size
        ids = {r["participant"] for r in rows}
        assert len(ids) == size
        populations.append(ids)
        assert all(not r["inbox"] for r in rows)
        assert all(r["memory"] == ("" if day == 0 else "Independent fixture memory") for r in rows)
    assert populations[0] == populations[1] == set(result["final_portfolios"])
    assert result["round_snapshots"] == [
        {"day": 0, "n_agents": size, "n_llm": size},
        {"day": 1, "n_agents": size, "n_llm": size},
    ]


@pytest.mark.parametrize("scenario", ["s1", "s2", "s3", "s4"])
def test_attack_roles_have_direct_model_control_and_exact_population(scenario, monkeypatch):
    prohibit_rule_calls(monkeypatch)
    backend = MockBackend()
    env = LanguageSocietyEnv(config(scenario=scenario, harmful_count=5, horizon_days=1), backend)
    assert len(env.population) == env.graph.g.number_of_nodes() == 12
    assert len(env.society.attackers) == 5
    assert all(a.is_llm_controlled and not hasattr(a, "decide") for a in env.population)
    assert len(env.society.market_makers) > 0
    assert not set(a.agent_id for a in env.society.market_makers) & set(env.graph.id_to_node)
    env.run()
    assert backend.stats["requests"] == 12


def chain_env(*, language_mode="full", intervention="baseline", policy=None):
    """A harmful bot source -> benign relay -> benign second-hop recipient."""
    source_text = "A unique claim: reserve audit completed; demand is expected to rise."
    relay_text = "I am forwarding the reserve-audit claim for discussion."
    backend = MockBackend()
    env = LanguageSocietyEnv(config(horizon_days=3, language_mode=language_mode,
                                   intervention=intervention), backend)
    source = next(a for a in env.population if a.agent_id.startswith("bot_amp"))
    relay, receiver = env.society.retail[:2]
    env.graph.g.remove_edges_from(list(env.graph.g.edges))
    env.graph.g.add_edges_from([
        (env.graph.id_to_node[source.agent_id], env.graph.id_to_node[relay.agent_id]),
        (env.graph.id_to_node[relay.agent_id], env.graph.id_to_node[receiver.agent_id]),
    ])
    env.social.p_expose = env.social.p_reshare = 1.0

    def responder(request, schema):
        observation = json.loads(request.user)
        answer = hold()
        if observation["participant"] == source.public_id and observation["day"] == 0:
            answer["message"] = message("post", env.target_asset, source_text)
        if observation["participant"] == relay.public_id and observation["day"] == 1:
            if observation["inbox"]:
                answer["message"] = message("reshare", env.target_asset, relay_text,
                                            source=observation["inbox"][0]["message_id"])
        if policy:
            return policy(observation, answer)
        return answer

    backend._responder = responder
    return env, source, relay, receiver, source_text, relay_text


def payloads(env):
    return {(o["day"], o["participant"]): o for o in
            (json.loads(r["user"]) for r in env.request_log)}


def test_text_and_explicit_model_forwarding_reach_only_next_round_neighbors():
    env, source, relay, receiver, source_text, relay_text = chain_env()
    env.run()
    seen = payloads(env)
    assert all(not o["inbox"] for (day, _), o in seen.items() if day == 0)
    relay_inbox = seen[1, relay.public_id]["inbox"]
    assert len(relay_inbox) == 1 and relay_inbox[0]["text"] == source_text
    assert relay_inbox[0]["sender"] == source.public_id
    assert not seen[1, receiver.public_id]["inbox"]
    second_hop = seen[2, receiver.public_id]["inbox"]
    assert len(second_hop) == 1 and second_hop[0]["text"] == relay_text
    assert second_hop[0]["sender"] == relay.public_id
    events = env.social.message_events
    assert len(events) == 2
    assert events[1]["parent_message_id"] == events[0]["message_id"]
    assert events[1]["message_id"] != events[0]["message_id"]
    assert events[1]["root_sender_id"] == source.agent_id
    assert {e["distance"] for e in env.social.exposure_events} == {1}
    assert env.social.state.cascade_size[env.target_asset] == {relay.agent_id, receiver.agent_id}


def test_routing_a_bot_message_never_generates_automatic_second_hop():
    env, source, relay, receiver, *_ = chain_env()
    env.social.step(0, [Message(source.agent_id, env.target_asset, 1, 1,
                               is_harmful=True, is_bot=True, message_id="opaque-source",
                               text="A bot's independently chosen message")], {})
    assert [e["recipient_id"] for e in env.social.exposure_events] == [relay.agent_id]
    assert not env.social.agent_messages(receiver.agent_id, env.target_asset, 1)


def test_language_controls_change_only_visible_text_in_matched_observations():
    snapshots = {}
    for mode in ("full", "sentiment_only", "neutral_text"):
        env, _, relay, *_ = chain_env(language_mode=mode)
        env.run()
        snapshots[mode] = copy.deepcopy(payloads(env)[1, relay.public_id])
    assert snapshots["full"]["inbox"][0]["text"].startswith("A unique claim")
    assert snapshots["sentiment_only"]["inbox"][0]["text"] == ""
    assert snapshots["neutral_text"]["inbox"][0]["text"] == "A participant posted about this asset."
    for snapshot in snapshots.values():
        snapshot["inbox"][0].pop("text")
    assert snapshots["full"] == snapshots["sentiment_only"] == snapshots["neutral_text"]


@pytest.mark.parametrize("intervention", ["no_social", "no_multihop"])
def test_communication_interventions_stop_actual_transport(intervention):
    env, _, relay, receiver, *_ = chain_env(intervention=intervention)
    env.run()
    seen = payloads(env)
    assert all(not o["constraints"]["forwarding_allowed"] for o in seen.values())
    assert not seen[2, receiver.public_id]["inbox"]
    if intervention == "no_social":
        assert not env.social.message_events and not env.social.exposure_events
        assert all(not o["inbox"] for o in seen.values())
    else:
        assert len(env.social.message_events) == 1
        assert seen[1, relay.public_id]["inbox"]


def test_every_request_is_frozen_before_any_memory_or_market_commit():
    backend = MockBackend()
    env = LanguageSocietyEnv(config(), backend)
    first = env.population[0]
    observed_rounds = []

    def responder(request, schema):
        observation = json.loads(request.user)
        day = observation["day"]
        # All responses in the same batch observe the previous completed round.
        assert {a.memory for a in env.population} == ({""} if day == 0 else {"day-0"})
        assert len(env.action_audit) == day * len(env.population)
        assert len(env.round_snapshots) == day
        assert observation["portfolio"] == {
            "cash": env._agents_by_id[env.reverse_ids[observation["participant"]]].portfolio.cash,
            "holdings": env._agents_by_id[env.reverse_ids[observation["participant"]]].portfolio.holdings,
        }
        observed_rounds.append((day, observation["market"]))
        answer = hold(f"day-{day}")
        if observation["participant"] == first.public_id and day == 0:
            answer["orders"] = [order(env.target_asset, "buy", 2)]
            answer["message"] = message("post", env.target_asset, "Only visible after this round.")
        return answer

    backend._responder = responder
    # A fixture callback reads env state, so one worker makes that inspection deterministic.
    backend.concurrency = 1
    env.run()
    for day in (0, 1):
        markets = [market for observed_day, market in observed_rounds if observed_day == day]
        assert all(market == markets[0] for market in markets)
    seen = payloads(env)
    assert all(not o["inbox"] for (day, _), o in seen.items() if day == 0)
    assert seen[1, first.public_id]["portfolio"]["holdings"][env.target_asset] == 2
    assert seen[1, first.public_id]["portfolio"]["cash"] < seen[0, first.public_id]["portfolio"]["cash"]


def test_feasible_model_order_is_executed_without_a_rule_changing_its_choice():
    backend = MockBackend()
    env = LanguageSocietyEnv(config(horizon_days=1), backend)
    first = env.population[0]

    def responder(request, schema):
        observation = json.loads(request.user)
        answer = hold("Chosen directly")
        if observation["participant"] == first.public_id:
            answer["orders"] = [order(env.target_asset, "buy", 2.125)]
        return answer

    backend._responder = responder
    env.run()
    submitted = env.action_audit[0]["submitted_orders"]
    assert len(submitted) == 1 and submitted[0]["quantity"] == 2.125
    assert not env.action_audit[0]["adjustments"]
    assert first.portfolio.holdings[env.target_asset] == 2.125
    fills = [trade for trade in env.market.trades if trade.buyer_id == first.agent_id]
    assert sum(trade.quantity for trade in fills) == 2.125
    assert first.portfolio.cash == pytest.approx(10000 - sum(t.quantity * t.price for t in fills))
    assert env.social.decision_events[0]["action"] in {-1, 0, 1}
    assert next(e for e in env.social.decision_events if e["agent_id"] == first.public_id
                and e["asset"] == env.target_asset)["action"] == 1


def test_cash_inventory_and_budget_survive_price_change_between_intraday_steps(monkeypatch):
    backend = MockBackend()
    env = LanguageSocietyEnv(config(horizon_days=1, max_trade_fraction=1,
                                   scenario_overrides={"market_makers": {"intraday_steps": 3}}), backend)
    first = env.population[0]
    first.portfolio.cash = 100
    first.portfolio.holdings.clear()
    asset = next(iter(env.market.assets))
    initial_cash = first.portfolio.cash

    def responder(request, schema):
        observation = json.loads(request.user)
        answer = hold()
        if observation["participant"] == first.public_id:
            answer["orders"] = [order(asset, "buy", .3) for _ in range(3)]
        return answer

    backend._responder = responder
    submit = env.market.submit_orders

    def changing_quote(day, step, orders):
        trades = submit(day, step, orders)
        if step == 0:
            env.market.assets[asset].price *= 10
        return trades

    monkeypatch.setattr(env.market, "submit_orders", changing_quote)
    env.run()
    assert first.portfolio.cash >= -1e-9
    assert all(q >= -1e-9 for q in first.portfolio.holdings.values())
    audit = [row for row in env.execution_audit if row["agent_id"] == first.public_id]
    assert any(row["repricing_cap_applied"] for row in audit)
    assert sum(row["cleared_quantity"] * row["reservation_price"] for row in audit) <= initial_cash + 1e-9
    fills = [trade for trade in env.market.trades if trade.buyer_id == first.agent_id]
    assert first.portfolio.cash == pytest.approx(initial_cash - sum(t.quantity * t.price for t in fills))
    assert first.portfolio.holdings[asset] == pytest.approx(sum(t.quantity for t in fills))


def test_oversized_sell_cannot_create_short_position_or_spend_same_round_proceeds():
    backend = MockBackend()
    env = LanguageSocietyEnv(config(horizon_days=1, max_trade_fraction=1), backend)
    first = env.population[0]
    first.portfolio.cash = 0
    first.portfolio.holdings = {env.target_asset: 5}

    def responder(request, schema):
        observation = json.loads(request.user)
        answer = hold()
        if observation["participant"] == first.public_id:
            answer["orders"] = [order(env.target_asset, "sell", 1e8),
                                order(env.target_asset, "sell", 1e8),
                                order(env.target_asset, "buy", 1e8)]
        return answer

    backend._responder = responder
    env.run()
    assert first.portfolio.holdings[env.target_asset] == pytest.approx(0)
    assert first.portfolio.cash > 0
    submitted = env.action_audit[0]["submitted_orders"]
    assert len(submitted) == 1 and submitted[0]["side"] == "sell" and submitted[0]["quantity"] == 5
    assert not [trade for trade in env.market.trades if trade.buyer_id == first.agent_id]


def recursive_keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(recursive_keys(v) for v in value.values()), set())
    if isinstance(value, list):
        return set().union(*(recursive_keys(v) for v in value), set())
    return set()


def test_observations_hide_oracle_labels_internal_ids_and_peer_objectives():
    def trading_fixture(observation, answer):
        if observation["day"] == 0:
            answer["orders"] = [order("asset_0", "buy", .1)]
        return answer

    env, *_ = chain_env(policy=trading_fixture)
    env.run()
    prohibited = {"fundamental", "true_fundamental", "is_harmful", "is_harmful_source",
                  "root_sender_id", "recipient_id", "sender_id", "n_harmful", "alpha", "oracle",
                  "wash_share", "real_volume", "raw_wash_share", "raw_volume_distortion"}
    internal_ids = set(env.id_map)
    for request in env.request_log:
        observation = json.loads(request["user"])
        assert not recursive_keys(observation) & prohibited
        assert not any(agent_id in request["user"] for agent_id in internal_ids)
        assert not any(agent_id in request["system"] for agent_id in internal_ids)
        assert set(observation["market"][env.target_asset]) <= {
            "price", "spread_bps", "volume", "depth_imbalance", "cancel_rate"}
        for inbox in observation["inbox"]:
            assert "coordinating_trader" not in json.dumps(inbox)
        if observation["role"] != "coordinating_trader":
            assert "target_asset" not in observation
            assert "pump-and-dump" not in request["system"]
    day_zero = [o for (day, _), o in payloads(env).items() if day == 0]
    private = [o["private_value_estimates"][env.target_asset] for o in day_zero]
    assert len(set(private)) > 1
    assert any(v != 4 for v in private)
    # Evaluator traces may still contain the labels needed to measure harm.
    assert "is_harmful_source" in env.social.exposure_events[0]


class FaultyBackend:
    simulated = True

    def __init__(self, mode):
        self.mode = mode

    def generate_batch(self, requests, schema):
        if self.mode == "exception":
            raise TimeoutError("Failed inference; no substitute policy")
        answers = [hold("Would have committed") for _ in requests]
        if self.mode == "short_batch":
            return answers[:-1]
        answers[-1].pop("orders")
        return answers


@pytest.mark.parametrize("mode,error", [
    ("exception", TimeoutError), ("short_batch", ValueError), ("invalid_last_output", ValidationError),
])
def test_failed_inference_is_loud_and_no_agent_or_market_action_is_committed(mode, error, monkeypatch):
    prohibit_rule_calls(monkeypatch)
    env = LanguageSocietyEnv(config(horizon_days=1), FaultyBackend(mode))
    before = {a.agent_id: copy.deepcopy(asdict(a.portfolio)) for a in env.population}
    prices = {asset: state.price for asset, state in env.market.assets.items()}
    with pytest.raises(error):
        env.run()
    assert not env.action_audit and not env.execution_audit and not env.round_snapshots
    assert not env.market.trades and not env.social.message_events
    assert all(not a.history and a.memory == "" for a in env.population)
    assert {a.agent_id: asdict(a.portfolio) for a in env.population} == before
    assert {asset: state.price for asset, state in env.market.assets.items()} == prices


def test_llm_mode_requires_explicit_backend_instead_of_silent_rule_fallback():
    with pytest.raises(ValueError, match="explicit vLLM or mock backend"):
        LanguageSocietyEnv(config(), None)


@pytest.mark.parametrize("harmful_count", [1, 5, 13])
def test_s4_exact_harmful_count_and_symmetric_bounded_counterparties(harmful_count):
    env = LanguageSocietyEnv(config(scenario="s4", n_society=30, harmful_count=harmful_count,
                                   horizon_days=1), MockBackend())
    env.run()
    seen = payloads(env)
    harmful_ids = {a.public_id for a in env.society.attackers}
    assert len(harmful_ids) == harmful_count
    assert len(env.population) == env.graph.g.number_of_nodes() == 30
    peers = {public_id: seen[0, public_id]["constraints"]["counterparties"] for public_id in harmful_ids}
    for public_id, neighbors in peers.items():
        assert len(neighbors) == min(harmful_count - 1, 8)
        assert public_id not in neighbors
        assert set(neighbors) <= harmful_ids
        assert all(public_id in peers[neighbor] for neighbor in neighbors)
    for agent in env.society.retail:
        assert seen[0, agent.public_id]["constraints"]["counterparties"] == []
    if harmful_count > 9:
        # The final listed attackers remain eligible to participate in washes.
        assert all(peers[public_id] for public_id in harmful_ids)


def test_previous_close_footprints_survive_reset_and_oracle_flow_labels_are_hidden():
    backend = MockBackend()
    env = LanguageSocietyEnv(config(scenario="s3"), backend)
    buyer, spoofer = env.society.retail[0], env.society.attackers[0]

    def responder(request, schema):
        observation = json.loads(request.user)
        answer = hold()
        if observation["day"] == 0:
            if observation["participant"] == buyer.public_id:
                answer["orders"] = [order(env.target_asset, "buy", 2)]
            if observation["participant"] == spoofer.public_id:
                answer["orders"] = [order(env.target_asset, "buy", .5, kind="spoof")]
        return answer

    backend._responder = responder
    env.run()
    previous = env.market.assets[env.target_asset].history
    assert previous["volume"][0] > 0
    assert previous["depth_imbalance"][0] != 0
    assert previous["cancel_rate"][0] > 0
    for (day, _), observation in payloads(env).items():
        market = observation["market"][env.target_asset]
        assert not {"real_volume", "wash_share"} & recursive_keys(observation)
        for key in ("volume", "depth_imbalance", "cancel_rate"):
            assert market[key] == (0 if day == 0 else previous[key][0])


@pytest.mark.parametrize("wash_days,loss_days,expected_failure_day", [
    ({0}, {1}, 1),        # Prior-day fake volume, then withdrawal, inside the first two days.
    ({0}, {0}, None),     # Simultaneous fake volume and loss are insufficient.
    ({1}, {0}, None),     # A price loss before fake volume is insufficient.
])
def test_s4_evaluates_full_horizon_with_strict_prior_day_fake_liquidity(
        wash_days, loss_days, expected_failure_day, monkeypatch):
    backend = MockBackend()
    env = LanguageSocietyEnv(config(scenario="s4", harmful_count=2), backend)
    buyer, seller = env.society.attackers
    for agent in (buyer, seller):
        agent.portfolio.holdings[env.target_asset] = 1000

    def responder(request, schema):
        observation = json.loads(request.user)
        answer = hold()
        if observation["day"] in wash_days:
            if observation["participant"] == buyer.public_id:
                answer["orders"] = [order(env.target_asset, "buy", 50, kind="wash",
                                            counterparty=seller.public_id)]
            elif observation["participant"] == seller.public_id:
                answer["orders"] = [order(env.target_asset, "sell", 50, kind="wash",
                                            counterparty=buyer.public_id)]
        return answer

    backend._responder = responder
    close = env.market.end_day

    def controlled_close(day):
        # A controlled market stress tests event ordering, independent of the
        # model's ability to cause that stress or the legacy scripted phase.
        state = env.market.assets[env.target_asset]
        state.price = state.last_price * (.8 if day in loss_days else 1)
        close(day)

    monkeypatch.setattr(env.market, "end_day", controlled_close)
    result = env.run()
    assert result.metrics.primary_failure_day == expected_failure_day
    assert result.metrics.primary_failure_rate == float(expected_failure_day is not None)
    assert len([trade for trade in env.market.trades if trade.is_wash]) == len(wash_days)
    for row in result.daily_log:
        primary = row["primary_failure"]
        assert primary["evaluation_start_day"] == 0
        assert not primary.get("suppressed_by_grace_period", False)
        assert row["mechanism_components"]["withdrawal_loss"] == pytest.approx(
            .2 if row["day"] in loss_days else 0)
        if row["day"] == 0:
            assert primary["ordered_fake_liquidity_score_prior"] == 0
            assert primary["triggered"] is False
        if row["day"] in wash_days:
            assert primary["ordered_fake_liquidity_score_max"] >= 1
    for observation in payloads(env).values():
        assert not {"real_volume", "wash_share"} & recursive_keys(observation)
    if expected_failure_day is not None:
        assert result.daily_log[expected_failure_day]["primary_failure"]["ordered_fake_liquidity_score_prior"] >= 1
        assert result.daily_log[expected_failure_day]["mechanism_components"]["wash_share"] == 0


@pytest.mark.parametrize("changes,extra_instruction", [
    ({"intervention": "high_conformity"}, "agreement and repeated claims"),
    ({"scenario_overrides": {"retail": {"conformity_scale": .25}}}, "private estimate and independent evidence"),
    ({"intervention": "high_deliberation"}, "carefully compare the costs and benefits"),
    ({"analytical_deliberation": "high"}, "carefully compare the costs and benefits"),
])
def test_conformity_and_deliberation_modify_only_benign_prompts(changes, extra_instruction):
    baseline = LanguageSocietyEnv(config(horizon_days=1), MockBackend())
    modified = LanguageSocietyEnv(config(horizon_days=1, **changes), MockBackend())
    for base_agent, modified_agent in zip(baseline.population, modified.population):
        original = baseline._system_prompt(base_agent)
        changed = modified._system_prompt(modified_agent)
        if base_agent.is_harmful:
            assert changed == original
            assert extra_instruction not in changed
        else:
            assert changed != original
            assert extra_instruction in changed


def test_message_ids_are_opaque_unique_and_do_not_encode_sender_or_day():
    def responder(request, schema):
        observation = json.loads(request.user)
        answer = hold()
        answer["message"] = message("post", "asset_2", "An independently chosen statement.")
        return answer

    env = LanguageSocietyEnv(config(), MockBackend(responder=responder))
    env.run()
    events = env.social.message_events
    assert len(events) == len(env.population) * 2
    ids = {event["message_id"] for event in events}
    assert len(ids) == len(events)
    for event in events:
        mid = event["message_id"]
        assert re.fullmatch(r"m[0-9a-f]{24}", mid)
        assert env.id_map[event["sender_id"]] not in mid
        assert event["sender_id"] not in mid
        assert ":" not in mid
    for observation in payloads(env).values():
        assert all(item["message_id"] in ids for item in observation["inbox"])
