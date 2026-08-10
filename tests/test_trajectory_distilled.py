from wolfbench.data import episode_to_trajectory_records
from wolfbench.defense import get_policy, get_track
from wolfbench.defense.distilled import DistilledWolfGuardPolicy, train_distilled_model
from wolfbench.env.environment import WolfBenchEnv
from wolfbench.scenarios.base import load_scenario


def test_trajectory_records_hide_oracle_from_public_observation():
    scen = load_scenario("s1")
    res = WolfBenchEnv(scen, n_society=120, alpha=0.05, seed=3,
                       record_trajectory=True).run()

    records = episode_to_trajectory_records(res, split="public_dev")
    assert records
    assert all("oracle_view" not in r["observation"] for r in records)
    assert all(r["label_available"] for r in records)
    assert {r["oracle_label"] for r in records}.issubset(
        {"none", "warning", "cooldown", "block"}
    )

    heldout = episode_to_trajectory_records(res, split="public_test")
    assert heldout
    assert all(not r["label_available"] for r in heldout)
    assert all(r["oracle_label"] is None for r in heldout)
    assert all(r["future_collapse"] is None for r in heldout)


def test_distilled_wolfguard_trains_loads_and_decides(tmp_path):
    scen = load_scenario("s1")
    res = WolfBenchEnv(scen, n_society=120, alpha=0.05, seed=4,
                       record_trajectory=True).run()
    records = episode_to_trajectory_records(res, split="public_dev")

    model = train_distilled_model(records, epochs=5, lr=0.05, seed=7)
    model_path = tmp_path / "distilled.json"
    model.save(model_path)

    policy = DistilledWolfGuardPolicy(model_path=str(model_path))
    summary = records[0]["observation"]
    actions = policy.decide(day=records[0]["day"], summary=summary)

    assert get_track("distilled") == "simulator_trained_baseline"
    assert set(actions) == set(summary["market"])
    assert all(a["action"] in {"none", "warning", "cooldown", "block"}
               for a in actions.values())


def test_topology_aware_policy_registered_and_uses_public_signals():
    policy = get_policy("topology_aware", risk_warning=0.2, risk_cooldown=0.4)
    policy.fit_baseline({
        "asset_0": {
            "msg_volume_mu": 0.0,
            "msg_volume_sd": 1.0,
            "volume_mu": 10.0,
            "volume_sd": 1.0,
            "price_gap_mu": 0.0,
            "price_gap_sd": 0.01,
            "spread_bps_mu": 10.0,
            "spread_bps_sd": 1.0,
        }
    })
    summary = {
        "day": 5,
        "market": {
            "asset_0": {
                "price": 120.0,
                "fundamental": 100.0,
                "volume": 40.0,
                "spread_bps": 20.0,
                "depth_imbalance": 0.5,
                "cancel_rate": 0.2,
                "wash_share": 0.1,
            }
        },
        "social": {
            "asset_0": {
                "msg_volume": 8.0,
                "sentiment": 0.9,
                "harmful_msg_share": 0.7,
                "cascade_size": 50.0,
            }
        },
        "recent_return": {"asset_0": 0.08},
        "oracle_view": {"asset_0": {"harmful_pressure": 1.0}},
    }

    actions = policy.decide(5, summary)

    assert get_track("topology_aware") == "submission"
    assert actions["asset_0"]["action"] in {"warning", "cooldown", "block"}
    assert actions["asset_0"]["reason"] == "topology_aware"
    assert "harmful_pressure" not in actions["asset_0"]["components"]


def test_calibrated_distilled_policy_registered(tmp_path):
    scen = load_scenario("s1")
    res = WolfBenchEnv(scen, n_society=120, alpha=0.05, seed=5,
                       record_trajectory=True).run()
    records = episode_to_trajectory_records(res, split="public_dev")
    model = train_distilled_model(records, epochs=3, lr=0.05, seed=8)
    model_path = tmp_path / "distilled.json"
    model.save(model_path)

    policy = get_policy("calibrated_distilled", model_path=str(model_path))
    actions = policy.decide(day=records[0]["day"], summary=records[0]["observation"])

    assert get_track("calibrated_distilled") == "simulator_trained_calibrated"
    assert all(a["action"] in {"warning", "cooldown", "block"} for a in actions.values())


def test_oracle_blocks_active_attack_target_at_low_pressure():
    policy = get_policy("oracle")
    actions = policy.decide(0, {
        "oracle_view": {
            "asset_0": {
                "active_harmful_count": 1.0,
                "active_harmful_share": 0.0075,
                "harmful_agent_share": 0.0075,
                "harmful_pressure": 0.0075,
                "is_attack_target": 1.0,
            }
        }
    })

    assert actions["asset_0"]["action"] == "block"
    assert actions["asset_0"]["risk"] == 1.0
