"""Smoke tests — run a tiny S0 and S1 episode end-to-end."""
import numpy as np
import pytest

from wolfbench.agents.wolfguard import WolfGuardAgent, WolfGuardConfig
from wolfbench.env.environment import EvaluatorConfig, WolfBenchEnv, validate_interventions
from wolfbench.scenarios.base import load_scenario
from wolfbench.tracks.runner import calibrate_clean_baseline


@pytest.mark.parametrize("scenario", ["s0", "s1"])
def test_episode_runs(scenario):
    scen = load_scenario(scenario)
    env = WolfBenchEnv(scen, n_society=200, alpha=0.0 if scenario == "s0" else 0.05, seed=1)
    res = env.run()
    assert len(res.metrics.daily_collapse_score) == scen.horizon_days
    assert res.metrics.max_collapse_score >= 0.0


def test_s3_uses_intraday_clearing_steps():
    scen = load_scenario("s3")
    env = WolfBenchEnv(scen, n_society=200, alpha=0.05, seed=1)
    assert env.intraday_steps == 12
    res = env.run()
    assert res.config_snapshot["intraday_steps"] == 12
    assert res.metrics.primary_metric == "spoof_liquidity_failure"
    assert res.metrics.cancel_rate_max >= 0.0
    assert res.metrics.spoof_depth_to_liquidity_max >= 0.0
    assert res.metrics.primary_failure_score_max >= 0.0


def test_s4_reports_mechanism_metrics():
    scen = load_scenario("s4")
    res = WolfBenchEnv(scen, n_society=200, alpha=0.05, seed=2).run()
    assert res.metrics.primary_metric == "fake_liquidity_failure"
    assert res.metrics.wash_share_max >= 0.0
    assert res.metrics.volume_distortion_max >= 0.0
    assert res.metrics.withdrawal_loss_max >= 0.0
    assert res.metrics.primary_failure_score_max >= 0.0


def test_pump_dump_collapse_more_likely_with_more_harmful():
    scen = load_scenario("s1")
    low = WolfBenchEnv(scen, n_society=500, alpha=0.0, seed=42).run()
    high = WolfBenchEnv(scen, n_society=500, alpha=0.10, seed=42).run()
    assert high.metrics.max_collapse_score >= low.metrics.max_collapse_score


def test_wolfguard_runs():
    scen = load_scenario("s1")
    base = calibrate_clean_baseline(n_society=200, seeds=(1, 2))
    wg = WolfGuardAgent(config=WolfGuardConfig())
    env = WolfBenchEnv(scen, n_society=300, alpha=0.05, seed=1,
                       wolfguard=wg, baseline=base)
    res = env.run()
    assert res.metrics.utility_loss >= 0.0


class CapturePolicy:
    name = "CapturePolicy"

    def __init__(self):
        self.summaries = []

    def fit_baseline(self, baseline):
        pass

    def decide(self, day, summary):
        self.summaries.append(summary)
        return {}


def test_oracle_view_is_hidden_by_default():
    scen = load_scenario("s1")
    policy = CapturePolicy()
    env = WolfBenchEnv(scen, n_society=100, alpha=0.05, seed=1, wolfguard=policy)
    env.run()
    assert policy.summaries
    assert all("oracle_view" not in summary for summary in policy.summaries)


def test_oracle_view_requires_explicit_exposure():
    scen = load_scenario("s1")
    policy = CapturePolicy()
    env = WolfBenchEnv(scen, n_society=100, alpha=0.05, seed=1,
                       wolfguard=policy, expose_oracle=True)
    env.run()
    assert any("oracle_view" in summary for summary in policy.summaries)


def test_validate_interventions_drops_bad_assets_and_clamps_risk():
    clean = validate_interventions({
        "asset_1": {"action": "block", "risk": float("inf")},
        "asset_2": {"action": "invalid", "risk": 2.5},
        "asset_x": {"action": "block", "risk": 1.0},
    }, {"asset_1", "asset_2"})
    assert set(clean) == {"asset_1", "asset_2"}
    assert clean["asset_1"]["risk"] == 0.0
    assert clean["asset_2"]["action"] == "none"
    assert clean["asset_2"]["risk"] == 1.0


class MaliciousConfigPolicy:
    name = "MaliciousConfigPolicy"
    config = WolfGuardConfig(intervention_cost_block=-100.0,
                             err_trade_exposure_threshold=-1.0)

    def fit_baseline(self, baseline):
        pass

    def decide(self, day, summary):
        return {asset: {"action": "block", "risk": 1.0} for asset in summary["market"]}


def test_evaluator_config_ignores_policy_costs_and_thresholds():
    scen = load_scenario("s1")
    env = WolfBenchEnv(
        scen, n_society=100, alpha=0.0, seed=1,
        wolfguard=MaliciousConfigPolicy(),
        evaluator_config=EvaluatorConfig(intervention_cost_block=0.25,
                                         err_trade_exposure_threshold=1.0),
    )
    res = env.run()
    assert res.metrics.intervention_cost >= 0.25
    assert res.metrics.utility_loss >= 0.25
