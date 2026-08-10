"""Theory utilities for Agent Society Dynamics.

The formal model distinguishes:

* social feedback gain, governed by a network Jacobian;
* private and social information capacities, which are separate channels;
* finite-horizon susceptibility; and
* attack aggregation with society size.

The scalar tanh model remains as a homogeneous corollary.  Its spinodal field
must not be identified with the benchmark's empirical response midpoint.
This module contains no simulator calls and consumes no experimental output.
"""
from __future__ import annotations

import argparse
from math import atanh, sqrt
from typing import Iterable

import numpy as np

from ..runtime.io_utils import OUTPUTS, ensure_dir, write_csv, write_json


def social_attention_share(capacity: float, half_saturation: float = 3.0) -> float:
    """Saturating share of socially available information that is processed."""
    capacity = max(float(capacity), 0.0)
    return capacity / (capacity + max(float(half_saturation), 1e-12))


def attention_share(capacity: float, half_saturation: float = 3.0) -> float:
    """Backward-compatible alias; ``capacity`` here is social capacity C_s."""
    return social_attention_share(capacity, half_saturation)


def effective_coupling(
    beta: float,
    conformity: float,
    mean_degree: float,
    capacity: float,
    half_saturation: float = 3.0,
) -> float:
    """Homogeneous loop gain Lambda = beta*gamma*d*q_s(C_s)/2.

    This is exact only for the homogeneous, unnormalized, d-regular reduction.
    It is not a valid network statistic for heterogeneous or hub-dominated
    societies; use :func:`network_jacobian` and :func:`spectral_radius` there.
    """
    network_feedback = (
        float(conformity)
        * float(mean_degree)
        * social_attention_share(capacity, half_saturation)
    )
    return 0.5 * float(beta) * network_feedback


def _agent_vector(value: float | Iterable[float], n: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(n, float(array), dtype=float)
    array = np.ravel(array)
    if len(array) != n:
        raise ValueError(f"{name} must be scalar or have length {n}")
    return array


def network_jacobian(
    influence_matrix: np.ndarray,
    *,
    beta: float | Iterable[float],
    conformity: float | Iterable[float],
    social_capacity: float | Iterable[float],
    equilibrium: float | Iterable[float] = 0.0,
    half_saturation: float = 3.0,
) -> np.ndarray:
    """Return the heterogeneous expected-action Jacobian at an equilibrium.

    Rows of ``influence_matrix`` map neighbors' expected actions into each
    recipient's payoff difference.  The matrix may be directed and need not be
    row normalized.
    """
    influence = np.asarray(influence_matrix, dtype=float)
    if influence.ndim != 2 or influence.shape[0] != influence.shape[1]:
        raise ValueError("influence_matrix must be square")
    if np.any(influence < 0):
        raise ValueError("influence_matrix must be nonnegative")
    n = influence.shape[0]
    beta_i = _agent_vector(beta, n, "beta")
    gamma_i = _agent_vector(conformity, n, "conformity")
    capacity_i = _agent_vector(social_capacity, n, "social_capacity")
    equilibrium_i = _agent_vector(equilibrium, n, "equilibrium")
    if np.any(np.abs(equilibrium_i) > 1):
        raise ValueError("equilibrium entries must lie in [-1,1]")
    processed = np.asarray(
        [social_attention_share(value, half_saturation) for value in capacity_i],
        dtype=float,
    )
    local_slope = (1.0 - equilibrium_i**2) * 0.5 * beta_i * gamma_i * processed
    return np.diag(local_slope) @ influence


def spectral_radius(matrix: np.ndarray) -> float:
    """Largest eigenvalue magnitude of a square matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def finite_horizon_susceptibility(
    jacobian: np.ndarray,
    *,
    horizon: int,
    external_response: np.ndarray | None = None,
    attack_direction: np.ndarray | None = None,
    readout: np.ndarray | None = None,
    two_sided: bool = True,
) -> float:
    """Maximum constant-shock response over a finite horizon.

    With ``two_sided=True``, computes
    ``max_t |c' sum_{k=0}^{t-1} J^k E v|``, appropriate when an adversary may
    choose either sign.  Otherwise it computes the positive part
    ``max_t [c' sum J^k E v]_+`` for a fixed signed attack direction.  Both
    capture transient amplification in directed/non-normal systems that a
    spectral radius alone can miss.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
        raise ValueError("jacobian must be square")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    n = jacobian.shape[0]
    response = np.eye(n) if external_response is None else np.asarray(external_response, dtype=float)
    if response.ndim != 2 or response.shape[0] != n:
        raise ValueError("external_response must have one row per agent")
    n_inputs = response.shape[1]
    direction = (
        np.full(n_inputs, 1.0 / max(n_inputs, 1), dtype=float)
        if attack_direction is None
        else np.asarray(attack_direction, dtype=float).reshape(-1)
    )
    if len(direction) != n_inputs:
        raise ValueError("attack_direction has incompatible length")
    output = (
        np.full(n, 1.0 / max(n, 1), dtype=float)
        if readout is None
        else np.asarray(readout, dtype=float).reshape(-1)
    )
    if len(output) != n:
        raise ValueError("readout has incompatible length")

    power = np.eye(n)
    cumulative = np.zeros(n, dtype=float)
    maximum = 0.0
    shock = response @ direction
    for _ in range(horizon):
        cumulative = cumulative + power @ shock
        value = float(output @ cumulative)
        response_value = abs(value) if two_sided else max(value, 0.0)
        maximum = max(maximum, response_value)
        power = jacobian @ power
    return maximum


def critical_external_pressure(beta: float, network_feedback: float) -> float | None:
    """Positive spinodal field destroying the negative metastable branch.

    The opposite spinodal is its negative. ``None`` means no metastable branch.
    This is not the empirical WolfBench ``alpha_c`` midpoint.
    """
    if beta <= 0 or network_feedback <= 0:
        return None
    coupling = 0.5 * beta * network_feedback
    if coupling <= 1.0:
        return None
    u = sqrt(1.0 - 1.0 / coupling)
    return network_feedback * u - (2.0 / beta) * atanh(u)


def spinodal_harmful_fraction(
    beta: float,
    conformity: float,
    mean_degree: float,
    social_capacity: float,
    attack_effect: float = 1.0,
    private_pressure: float = 0.0,
    half_saturation: float = 3.0,
) -> float | None:
    """Homogeneous harmful fraction that destroys the lower branch."""
    network_feedback = (
        conformity
        * mean_degree
        * social_attention_share(social_capacity, half_saturation)
    )
    pressure = critical_external_pressure(beta, network_feedback)
    if pressure is None:
        return None
    return max(0.0, (pressure - private_pressure) / max(attack_effect, 1e-12))


def critical_harmful_fraction(
    beta: float,
    conformity: float,
    mean_degree: float,
    capacity: float,
    attack_effect: float = 1.0,
    private_pressure: float = 0.0,
    half_saturation: float = 3.0,
) -> float | None:
    """Backward-compatible alias for :func:`spinodal_harmful_fraction`.

    The old name is theoretically ambiguous and should not be used in paper
    text because the benchmark response midpoint is a different estimand.
    """
    return spinodal_harmful_fraction(
        beta=beta,
        conformity=conformity,
        mean_degree=mean_degree,
        social_capacity=capacity,
        attack_effect=attack_effect,
        private_pressure=private_pressure,
        half_saturation=half_saturation,
    )


def finite_size_critical_fraction(
    *,
    collapse_margin: float,
    attack_effect: float,
    n_society: int,
    attack_aggregation_exponent: float,
    susceptibility: float,
) -> float:
    """Linear-response alpha_c = b/(eta*N**delta*chi), for a chosen gain."""
    if collapse_margin < 0:
        raise ValueError("collapse_margin must be nonnegative")
    if attack_effect <= 0 or n_society <= 0 or susceptibility <= 0:
        raise ValueError("attack_effect, n_society, and susceptibility must be positive")
    denominator = (
        attack_effect
        * float(n_society) ** float(attack_aggregation_exponent)
        * susceptibility
    )
    return float(collapse_margin / denominator)


def information_crossover(
    *,
    social_coefficient: float,
    private_coefficient: float,
    social_residual_variance: float,
    private_residual_variance: float,
) -> tuple[float, float | None]:
    """Weak-signal crossover ``Omega`` and approximate dominance ``D``.

    ``D`` is undefined when both channels carry zero local signal.
    """
    social_strength = float(social_coefficient) ** 2 * max(float(social_residual_variance), 0.0)
    private_strength = float(private_coefficient) ** 2 * max(float(private_residual_variance), 0.0)
    total = social_strength + private_strength
    if total == 0:
        return 0.0, None
    omega = float("inf") if private_strength == 0 else social_strength / private_strength
    return omega, social_strength / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="network_game_theory")
    parser.add_argument("--betas", default="0.5,1,2,3,4")
    parser.add_argument("--conformities", default="0.1,0.35,0.7,1.35")
    parser.add_argument("--degrees", default="4,8,16")
    parser.add_argument("--social-capacities", default="1,2,4,6")
    parser.add_argument("--attack-effect", type=float, default=4.0)
    args = parser.parse_args()
    parse = lambda value: [float(item) for item in value.split(",") if item.strip()]
    rows = []
    for beta in parse(args.betas):
        for conformity in parse(args.conformities):
            for degree in parse(args.degrees):
                for capacity in parse(args.social_capacities):
                    share = social_attention_share(capacity)
                    network_feedback = conformity * degree * share
                    coupling = effective_coupling(beta, conformity, degree, capacity)
                    pressure = critical_external_pressure(beta, network_feedback)
                    rows.append({
                        "beta": beta,
                        "conformity": conformity,
                        "mean_degree": degree,
                        "social_attention_capacity": capacity,
                        "social_attention_share": share,
                        "network_feedback": network_feedback,
                        "homogeneous_loop_gain_Lambda": coupling,
                        "multiple_equilibria_in_homogeneous_model": int(coupling > 1.0),
                        "spinodal_external_pressure": "" if pressure is None else pressure,
                        "spinodal_attack_fraction": (
                            "" if pressure is None else max(0.0, pressure / args.attack_effect)
                        ),
                    })
    out_dir = ensure_dir(OUTPUTS / args.out)
    write_csv(rows, out_dir / "meanfield_predictions.csv")
    write_json({
        "model": "homogeneous corollary of the heterogeneous network theory",
        "multiplicity_condition": "beta*gamma*d*q_s(C_s)/2 > 1",
        "attack_effect_eta": args.attack_effect,
        "n_predictions": len(rows),
        "warning": (
            "spinodal_attack_fraction is not the empirical alpha_c midpoint; "
            "the finite-horizon benchmark requires a network susceptibility"
        ),
    }, out_dir / "config.json")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
