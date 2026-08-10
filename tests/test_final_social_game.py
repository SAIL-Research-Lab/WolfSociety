from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pytest

from wolfbench.env.environment import WolfBenchEnv
from wolfbench.scenarios.base import load_scenario

from paper_experiments_v3.theory.meanfield import (
    critical_external_pressure,
    effective_coupling,
    finite_horizon_susceptibility,
    finite_size_critical_fraction,
    information_crossover,
    network_jacobian,
    spectral_radius,
)
from wolfbench.agents.social_game import (
    BoundedRationalRetailAgent,
    BoundedSocialEnv,
    LegacyScoreLoggingRetailAgent,
    conditional_mutual_information,
    install_network_signal_game,
)


def _environment(*, social: bool = True, seed: int = 7) -> WolfBenchEnv:
    scenario = deepcopy(load_scenario("s1"))
    scenario.retail["controller_mode"] = "mixed_roles"
    if not social:
        scenario.social["p_expose"] = 0.0
        scenario.social["p_reshare"] = 0.0
        scenario.retail["conformity_scale"] = 0.0
    env = WolfBenchEnv(scenario, n_society=80, alpha=0.025, seed=seed)
    install_network_signal_game(env, controller_mode="mixed_roles")
    return env


def test_conditional_mutual_information_known_cases() -> None:
    x = [0, 0, 1, 1] * 20
    y_same = list(x)
    y_independent = [0, 1, 0, 1] * 20
    z = [0] * len(x)
    assert conditional_mutual_information(x, y_same, z) == pytest.approx(1.0)
    assert conditional_mutual_information(x, y_independent, z) == pytest.approx(0.0)


def test_population_uses_distinct_decision_processes() -> None:
    env = _environment()
    assert isinstance(env.social, BoundedSocialEnv)
    assert all(isinstance(agent, BoundedRationalRetailAgent) for agent in env.society.retail)
    processes = {agent.process for agent in env.society.retail}
    assert {
        "risk_averse",
        "value_based",
        "trend_following",
        "social_following",
        "aggressive",
    }.issubset(processes)


def test_legacy_controller_is_logged_without_replacing_its_choice_rule() -> None:
    scenario = deepcopy(load_scenario("s1"))
    scenario.retail["controller_mode"] = "legacy_score"
    env = WolfBenchEnv(scenario, n_society=40, alpha=0.025, seed=4)
    install_network_signal_game(env, controller_mode="legacy_score")
    assert all(isinstance(agent, LegacyScoreLoggingRetailAgent) for agent in env.society.retail)
    env.run()
    metrics = env.social.information_metrics()
    assert metrics["n_agent_decisions"] > 0
    assert {row["policy"] for row in env.social.decision_events} == {"legacy_score"}


def test_network_game_records_auditable_information_flow() -> None:
    env = _environment()
    env.run()
    metrics = env.social.information_metrics()
    assert metrics["n_agent_decisions"] > 0
    assert metrics["n_messages"] > 0
    assert metrics["n_benign_messages"] > 0
    assert metrics["n_exposure_events"] > 0
    assert metrics["max_cascade_reach"] > 0
    assert 0.0 <= metrics["social_dominance_ratio"] <= 1.0
    assert 0.0 <= metrics["cascade_decision_rate"] <= 1.0
    assert all(
        math.isfinite(value)
        for key, value in metrics.items()
        if key != "mean_qre_action_entropy_bits"
    )


def test_private_only_removes_social_information_channel() -> None:
    env = _environment(social=False)
    env.run()
    metrics = env.social.information_metrics()
    assert metrics["n_exposure_events"] == 0
    assert metrics["social_information_bits"] == pytest.approx(0.0)
    assert metrics["social_proof_information_bits"] == pytest.approx(0.0)
    assert metrics["transfer_entropy_social_to_trade_bits"] == pytest.approx(0.0)


def test_benign_diffusion_is_not_counted_as_harmful_safety_cascade() -> None:
    scenario = deepcopy(load_scenario("s1"))
    scenario.retail["controller_mode"] = "mixed_roles"
    env = WolfBenchEnv(scenario, n_society=50, alpha=0.0, seed=9)
    install_network_signal_game(env, controller_mode="mixed_roles")
    result = env.run()
    information = env.social.information_metrics()
    assert information["n_benign_messages"] > 0
    assert information["max_cascade_reach"] > 0
    assert result.metrics.social_cascade_peak == pytest.approx(0.0)
    assert result.metrics.primary_failure_rate == pytest.approx(0.0)


def test_mean_field_multiplicity_boundary() -> None:
    below = effective_coupling(beta=1.0, conformity=0.1, mean_degree=4, capacity=1)
    above = effective_coupling(beta=4.0, conformity=1.0, mean_degree=8, capacity=6)
    assert below < 1.0
    assert above > 1.0
    assert critical_external_pressure(beta=1.0, network_feedback=2 * below) is None
    pressure = critical_external_pressure(beta=4.0, network_feedback=2 * above / 4.0)
    assert pressure is not None and pressure > 0.0


def test_network_loop_gain_reduces_to_homogeneous_regular_case() -> None:
    # Four-node cycle: every row has unnormalized degree two.
    influence = np.asarray([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    jacobian = network_jacobian(
        influence,
        beta=2.0,
        conformity=1.0,
        social_capacity=3.0,
    )
    scalar = effective_coupling(
        beta=2.0,
        conformity=1.0,
        mean_degree=2.0,
        capacity=3.0,
    )
    assert spectral_radius(jacobian) == pytest.approx(scalar)


def test_finite_horizon_gain_captures_cumulative_response() -> None:
    jacobian = 0.5 * np.eye(2)
    gain = finite_horizon_susceptibility(
        jacobian,
        horizon=3,
        attack_direction=np.ones(2),
        readout=np.full(2, 0.5),
    )
    assert gain == pytest.approx(1.0 + 0.5 + 0.25)

    negative_direction = -np.ones(2)
    assert finite_horizon_susceptibility(
        jacobian,
        horizon=3,
        attack_direction=negative_direction,
        readout=np.full(2, 0.5),
        two_sided=False,
    ) == pytest.approx(0.0)
    assert finite_horizon_susceptibility(
        jacobian,
        horizon=3,
        attack_direction=negative_direction,
        readout=np.full(2, 0.5),
        two_sided=True,
    ) == pytest.approx(1.0 + 0.5 + 0.25)


def test_finite_size_law_and_information_crossover_are_distinct() -> None:
    alpha_100 = finite_size_critical_fraction(
        collapse_margin=1.0,
        attack_effect=1.0,
        n_society=100,
        attack_aggregation_exponent=0.5,
        susceptibility=1.0,
    )
    alpha_400 = finite_size_critical_fraction(
        collapse_margin=1.0,
        attack_effect=1.0,
        n_society=400,
        attack_aggregation_exponent=0.5,
        susceptibility=1.0,
    )
    assert alpha_400 == pytest.approx(alpha_100 / 2.0)

    omega, dominance = information_crossover(
        social_coefficient=2.0,
        private_coefficient=1.0,
        social_residual_variance=1.0,
        private_residual_variance=1.0,
    )
    assert omega == pytest.approx(4.0)
    assert dominance == pytest.approx(0.8)
