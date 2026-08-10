from __future__ import annotations

import pytest

from wolfbench.metrics import (
    bootstrap_ci,
    defense_score,
    threshold_protection_score,
    threshold_shift,
)


def _row(
    alpha: float,
    collapse: float,
    loss: float,
    *,
    utility: float = 0.0,
    false_positive_rate: float = 0.0,
    cost: float = 0.0,
) -> dict[str, float]:
    return {
        "alpha": alpha,
        "collapse_rate": collapse,
        "collapse_day": -1.0,
        "retail_loss_pct_30d": loss,
        "utility_loss": utility,
        "false_positive_rate": false_positive_rate,
        "intervention_cost": cost,
    }


def test_harm_reduction_is_gated_without_safety_signal() -> None:
    no_defense = [_row(0.01, 1.0, 0.10), _row(0.02, 1.0, 0.10)]
    defended = [
        _row(0.01, 1.0, 0.01, utility=1.0, cost=1.0),
        _row(0.02, 1.0, 0.01, utility=1.0, cost=1.0),
    ]

    score = defense_score(no_defense, defended, alphas=[0.01, 0.02])

    assert score["delta_harm_reduction"] > 0.0
    assert score["safety_gate"] == 0.0
    assert score["gated_delta_harm_reduction"] == 0.0
    assert score["defense_score"] < 0.0


def test_bootstrap_ci_singleton_is_degenerate() -> None:
    assert bootstrap_ci([3.0]) == (3.0, 3.0)


def test_negative_costs_are_not_rewards() -> None:
    no_defense = [_row(0.01, 1.0, 0.10)]
    defended = [
        _row(
            0.01,
            1.0,
            0.10,
            utility=-100.0,
            false_positive_rate=2.0,
            cost=-100.0,
        )
    ]

    score = defense_score(no_defense, defended, alphas=[0.01])

    assert score["utility_loss"] == 0.0
    assert score["intervention_cost"] == 0.0
    assert score["false_positive_rate"] == 1.0
    assert score["defense_score"] <= 0.0


def test_tps_rewards_a_rightward_threshold_shift() -> None:
    no_defense = []
    defended = []
    for alpha, no_collapse, defended_collapse in [
        (0.00, 0.0, 0.0),
        (0.01, 0.0, 0.0),
        (0.02, 1.0, 0.0),
        (0.03, 1.0, 1.0),
    ]:
        no_defense.append(_row(alpha, no_collapse, 0.10))
        defended.append(_row(alpha, defended_collapse, 0.05))

    score = threshold_protection_score(
        no_defense,
        defended,
        alphas=[0.00, 0.01, 0.02, 0.03],
    )
    shift = threshold_shift(
        no_defense,
        defended,
        alphas=[0.00, 0.01, 0.02, 0.03],
    )

    assert score["tps"] > 0.0
    assert score["shift_score"] > 0.0
    assert score["critical_band_delta_p"] > 0.0
    assert score["cost_gate"] == pytest.approx(1.0)
    assert shift["threshold_shift"] > 0.0


def test_tps_caps_bad_defense_at_zero_but_keeps_signed_diagnostic() -> None:
    no_defense = [
        _row(0.00, 0.0, 0.01),
        _row(0.01, 0.0, 0.01),
        _row(0.02, 1.0, 0.10),
    ]
    defended = [
        _row(0.00, 1.0, 0.20, utility=10.0, false_positive_rate=1.0, cost=10.0),
        _row(0.01, 1.0, 0.20, utility=10.0, false_positive_rate=1.0, cost=10.0),
        _row(0.02, 1.0, 0.20, utility=10.0, false_positive_rate=1.0, cost=10.0),
    ]

    score = threshold_protection_score(
        no_defense,
        defended,
        alphas=[0.00, 0.01, 0.02],
    )

    assert score["tps"] == 0.0
    assert score["raw_net"] < 0.0


def test_tps_prefers_primary_failure_rate_when_available() -> None:
    no_defense = [
        {
            **_row(0.0, 0.0, 0.01),
            "primary_failure_rate": 0.0,
            "primary_failure_score_max": 0.01,
        },
        {
            **_row(0.1, 0.0, 0.10),
            "primary_failure_rate": 1.0,
            "primary_failure_score_max": 0.10,
        },
    ]
    defended = [
        {
            **_row(0.0, 0.0, 0.01),
            "primary_failure_rate": 0.0,
            "primary_failure_score_max": 0.01,
        },
        {
            **_row(0.1, 0.0, 0.05),
            "primary_failure_rate": 0.0,
            "primary_failure_score_max": 0.05,
        },
    ]

    score = threshold_protection_score(
        no_defense,
        defended,
        alphas=[0.0, 0.1],
    )

    assert score["probability_key"] == "primary_failure_rate"
    assert score["tps"] > 0.0
    assert score["critical_band_delta_p"] > 0.0