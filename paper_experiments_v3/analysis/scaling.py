"""Paper analysis for P01 nonlinear response and finite-size scaling."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..runtime.io_utils import OUTPUTS, write_csv, write_json


def crossing(points: list[tuple[float, float]], target: float) -> float | None:
    points = sorted(points)
    for (a0, p0), (a1, p1) in zip(points, points[1:]):
        if p0 == target:
            return a0
        if (p0 - target) * (p1 - target) <= 0 and p1 != p0:
            return a0 + (target - p0) * (a1 - a0) / (p1 - p0)
    return points[-1][0] if points and points[-1][1] == target else None


def seed_curve(rows: list[dict[str, str]], sampled_seeds: list[int] | None = None) -> list[tuple[float, float]]:
    allowed = None if sampled_seeds is None else sampled_seeds
    by_alpha: dict[float, list[float]] = defaultdict(list)
    rows_by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_seed[int(float(row["seed"]))].append(row)
    source = rows if allowed is None else [row for seed in allowed for row in rows_by_seed[seed]]
    for row in source:
        by_alpha[float(row["alpha"])].append(float(row["primary_failure_rate"]))
    return sorted((alpha, sum(values) / len(values)) for alpha, values in by_alpha.items())


def model_comparison(rows: list[dict[str, str]]) -> tuple[float, float, float]:
    x = np.asarray([float(row["alpha"]) for row in rows], dtype=float)
    y = np.asarray([float(row["primary_failure_rate"]) for row in rows], dtype=float)
    eps = 1e-9

    design = np.column_stack([np.ones_like(x), x])
    theta = np.zeros(2, dtype=float)
    # Ridge-stabilized IRLS avoids a SciPy dependency and handles near-separable
    # smoke grids without divergent coefficients.
    for _ in range(100):
        probability = np.clip(
            1.0 / (1.0 + np.exp(-np.clip(design @ theta, -30.0, 30.0))),
            eps,
            1 - eps,
        )
        weights = probability * (1.0 - probability)
        hessian = design.T @ (weights[:, None] * design) + 1e-7 * np.eye(2)
        step = np.linalg.solve(hessian, design.T @ (y - probability))
        theta += step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    logit_probability = np.clip(
        1.0 / (1.0 + np.exp(-np.clip(design @ theta, -30.0, 30.0))),
        eps,
        1 - eps,
    )
    logit_nll = float(-np.sum(
        y * np.log(logit_probability) + (1 - y) * np.log(1 - logit_probability)
    ))

    # A constrained linear-probability likelihood is parameterized by its
    # probabilities at the smallest and largest tested alpha. A deterministic
    # coarse-to-fine search finds the two-parameter maximum likelihood fit.
    x_min, x_max = float(np.min(x)), float(np.max(x))
    fraction = np.zeros_like(x) if x_max == x_min else (x - x_min) / (x_max - x_min)
    best_nll, best_pair = float("inf"), (0.5, 0.5)
    low0, high0, low1, high1 = 0.001, 0.999, 0.001, 0.999
    for resolution in (51, 41, 41):
        for p0 in np.linspace(low0, high0, resolution):
            for p1 in np.linspace(low1, high1, resolution):
                probability = np.clip(p0 + (p1 - p0) * fraction, eps, 1 - eps)
                nll = float(-np.sum(
                    y * np.log(probability) + (1 - y) * np.log(1 - probability)
                ))
                if nll < best_nll:
                    best_nll, best_pair = nll, (float(p0), float(p1))
        span0 = max((high0 - low0) / max(resolution - 1, 1) * 3, 1e-4)
        span1 = max((high1 - low1) / max(resolution - 1, 1) * 3, 1e-4)
        low0, high0 = max(eps, best_pair[0] - span0), min(1 - eps, best_pair[0] + span0)
        low1, high1 = max(eps, best_pair[1] - span1), min(1 - eps, best_pair[1] + span1)

    logit_aic = 2 * 2 + 2 * logit_nll
    linear_aic = 2 * 2 + 2 * best_nll
    return logit_aic, linear_aic, linear_aic - logit_aic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="p01_nonlinear_scaling")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    out_dir = OUTPUTS / args.run
    with (out_dir / "data.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok"]
    by_n: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_n[int(float(row["n_society"]))].append(row)
    rng = np.random.default_rng(202707)
    summaries = []
    for n_society, group in sorted(by_n.items()):
        curve = seed_curve(group)
        alpha10, alpha50, alpha90 = (crossing(curve, target) for target in (0.1, 0.5, 0.9))
        seeds = sorted({int(float(row["seed"])) for row in group})
        boot = []
        if alpha50 is not None and len(seeds) > 1:
            for _ in range(args.bootstrap):
                sampled = rng.choice(seeds, size=len(seeds), replace=True).tolist()
                value = crossing(seed_curve(group, sampled), 0.5)
                if value is not None:
                    boot.append(value)
        logit_aic, linear_aic, delta_aic = model_comparison(group)
        summaries.append({
            "n_society": n_society,
            "n_seeds": len(seeds),
            "alpha_c": "" if alpha50 is None else alpha50,
            "alpha_c_lo": "" if not boot else float(np.quantile(boot, 0.025)),
            "alpha_c_hi": "" if not boot else float(np.quantile(boot, 0.975)),
            "critical_harmful_count": "" if alpha50 is None else n_society * alpha50,
            "transition_width_10_90": "" if alpha10 is None or alpha90 is None else alpha90 - alpha10,
            "crossing_status": "resolved" if alpha50 is not None else "right_censored",
            "logistic_aic": logit_aic,
            "linear_aic": linear_aic,
            "delta_aic_linear_minus_logistic": delta_aic,
        })
    write_csv(summaries, out_dir / "scaling_analysis.csv")
    resolved = [row for row in summaries if row["alpha_c"] != ""]
    exponent = ""
    if len(resolved) >= 2:
        slope = np.polyfit(
            np.log([row["n_society"] for row in resolved]),
            np.log([float(row["alpha_c"]) for row in resolved]),
            1,
        )[0]
        exponent = float(-slope)
    write_json({
        "n_sizes": len(summaries),
        "n_resolved": len(resolved),
        "descriptive_scaling_exponent": exponent,
        "bootstrap_resamples": args.bootstrap,
        "warning": "The exponent is descriptive; causal channel attribution comes from P02.",
    }, out_dir / "scaling_analysis.json")
    print(f"Wrote {out_dir / 'scaling_analysis.csv'}")


if __name__ == "__main__":
    main()
