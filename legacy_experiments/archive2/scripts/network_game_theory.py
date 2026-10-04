"""Analytical predictions for the bounded-rational network signaling game.

The binary mean-field reduction is

    m = tanh[ beta/2 * (eta*alpha + theta*s + K*m) ],

where K = conformity * mean_degree * attention_share(capacity).  The module
contains no simulator calls; it can therefore be used to preregister the
direction and location of tipping predictions before E8 is run.
"""
from __future__ import annotations

import argparse
from math import atanh, sqrt

from .io_utils import OUTPUTS, ensure_dir, write_csv, write_json


def attention_share(capacity: float, half_saturation: float = 3.0) -> float:
    """Saturating share of socially available information that is processed."""
    capacity = max(float(capacity), 0.0)
    return capacity / (capacity + max(float(half_saturation), 1e-12))


def effective_coupling(
    beta: float,
    conformity: float,
    mean_degree: float,
    capacity: float,
    half_saturation: float = 3.0,
) -> float:
    """Dimensionless local slope J = beta*K/2 at the symmetric equilibrium."""
    network_feedback = conformity * mean_degree * attention_share(capacity, half_saturation)
    return 0.5 * beta * network_feedback


def critical_external_pressure(beta: float, network_feedback: float) -> float | None:
    """Positive spinodal pressure; None means the fixed point is unique."""
    if beta <= 0 or network_feedback <= 0:
        return None
    coupling = 0.5 * beta * network_feedback
    if coupling <= 1.0:
        return None
    u = sqrt(1.0 - 1.0 / coupling)
    return network_feedback * u - (2.0 / beta) * atanh(u)


def critical_harmful_fraction(
    beta: float,
    conformity: float,
    mean_degree: float,
    capacity: float,
    attack_effect: float = 1.0,
    private_pressure: float = 0.0,
    half_saturation: float = 3.0,
) -> float | None:
    """Mean-field alpha_c for leaving the lower branch, clipped at zero."""
    network_feedback = conformity * mean_degree * attention_share(capacity, half_saturation)
    pressure = critical_external_pressure(beta, network_feedback)
    if pressure is None:
        return None
    return max(0.0, (pressure - private_pressure) / max(attack_effect, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="network_game_theory")
    parser.add_argument("--betas", default="0.5,1,2,3,4")
    parser.add_argument("--conformities", default="0.1,0.35,0.7,1.35")
    parser.add_argument("--degrees", default="4,8,16")
    parser.add_argument("--capacities", default="1,2,4,6")
    parser.add_argument("--attack-effect", type=float, default=4.0)
    args = parser.parse_args()
    parse = lambda text: [float(value) for value in text.split(",") if value.strip()]
    rows = []
    for beta in parse(args.betas):
        for conformity in parse(args.conformities):
            for degree in parse(args.degrees):
                for capacity in parse(args.capacities):
                    network_feedback = conformity * degree * attention_share(capacity)
                    coupling = effective_coupling(beta, conformity, degree, capacity)
                    pressure = critical_external_pressure(beta, network_feedback)
                    rows.append({
                        "beta": beta,
                        "conformity": conformity,
                        "mean_degree": degree,
                        "attention_capacity": capacity,
                        "attention_share": attention_share(capacity),
                        "network_feedback": network_feedback,
                        "dimensionless_coupling_J": coupling,
                        "multiple_equilibria_predicted": int(coupling > 1.0),
                        "critical_external_pressure": "" if pressure is None else pressure,
                        "alpha_c_predicted": "" if pressure is None else max(0.0, pressure / args.attack_effect),
                    })
    out_dir = ensure_dir(OUTPUTS / args.out)
    write_csv(rows, out_dir / "meanfield_predictions.csv")
    write_json({
        "model": "m=tanh(beta/2*(eta*alpha+theta*s+K*m))",
        "multiplicity_condition": "beta*K/2 > 1",
        "attack_effect_eta": args.attack_effect,
        "n_predictions": len(rows),
    }, out_dir / "config.json")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
