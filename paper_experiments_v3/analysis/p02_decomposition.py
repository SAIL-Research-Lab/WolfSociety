"""P02 finite-size decomposition analysis for the paper figures.

This script deliberately reports proxy quantities unless the simulator writes a
direct attack field and finite-horizon perturbation response. The paper figure
should keep the same labels: attack-magnitude proxy and finite-horizon gain
proxy, not exact Jacobian objects.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from ..runtime.io_utils import OUTPUTS, write_csv, write_json


EPS = 1e-9


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _variant_settings(variant: str) -> tuple[float, bool]:
    if variant.endswith("_q0"):
        return 0.0, "no_social" not in variant
    if variant.endswith("_q1"):
        return 1.0, "no_social" not in variant
    return 0.5, "no_social" not in variant


def _attack_proxy(row: dict[str, str]) -> float:
    if row.get("attack_magnitude_proxy", "") != "":
        return max(_float(row, "attack_magnitude_proxy"), EPS)
    n = max(_float(row, "n_society"), 1.0)
    k = max(_float(row, "requested_harmful_count", _float(row, "n_harmful")), 0.0)
    q = _float(row, "liquidity_exponent", _variant_settings(row.get("variant", ""))[0])
    return max(k / max(n**q, EPS), EPS)


def _gain_proxy(row: dict[str, str]) -> float:
    if row.get("failure_gain_proxy", "") != "":
        return max(_float(row, "failure_gain_proxy"), EPS)
    score = max(_float(row, "primary_failure_score_max"), 0.0)
    return max(score / _attack_proxy(row), EPS)


def _fit_log_exponent(rows: list[dict[str, Any]], y_key: str) -> dict[str, float]:
    """Fit log(y) ~ intercept + beta_N log(N) + beta_K log(K)."""
    usable = []
    for row in rows:
        y = float(row[y_key])
        n = float(row["n_society"])
        k = float(row["requested_harmful_count"])
        if y > 0 and n > 0 and k > 0:
            usable.append((math.log(n), math.log(k), math.log(y)))
    if len(usable) < 3:
        return {"slope_n": float("nan"), "slope_k": float("nan"), "n_points": len(usable)}
    x = np.asarray([[1.0, item[0], item[1]] for item in usable], dtype=float)
    y = np.asarray([item[2] for item in usable], dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {"slope_n": float(beta[1]), "slope_k": float(beta[2]), "n_points": len(usable)}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok"]


def _read_nu(p01_run: str) -> float:
    path = OUTPUTS / p01_run / "scaling_analysis.json"
    if not path.exists():
        return float("nan")
    import json

    payload = json.loads(path.read_text())
    value = payload.get("descriptive_scaling_exponent", "")
    return float(value) if value != "" else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="p02_size_decomposition")
    parser.add_argument("--p01-run", default="p01_nonlinear_scaling")
    args = parser.parse_args()

    out_dir = OUTPUTS / args.run
    rows = _read_rows(out_dir / "data.csv")
    enriched: list[dict[str, Any]] = []
    for row in rows:
        q, social_on = _variant_settings(row.get("variant", ""))
        k = _float(row, "requested_harmful_count", _float(row, "n_harmful"))
        enriched.append({
            "variant": row.get("variant", ""),
            "n_society": int(_float(row, "n_society")),
            "requested_harmful_count": int(k),
            "seed": int(_float(row, "seed")),
            "liquidity_exponent": q,
            "social_on": int(social_on),
            "attack_magnitude_proxy": _attack_proxy(row),
            "failure_gain_proxy": _gain_proxy(row),
            "primary_failure_score_max": _float(row, "primary_failure_score_max"),
            "primary_failure_rate": _float(row, "primary_failure_rate"),
        })

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[(row["variant"], row["n_society"], row["requested_harmful_count"])].append(row)

    cell_rows = []
    for (variant, n_society, harmful_count), group in sorted(grouped.items()):
        cell_rows.append({
            "variant": variant,
            "n_society": n_society,
            "requested_harmful_count": harmful_count,
            "n_seeds": len({row["seed"] for row in group}),
            "liquidity_exponent": group[0]["liquidity_exponent"],
            "social_on": group[0]["social_on"],
            "attack_magnitude_proxy_mean": mean(row["attack_magnitude_proxy"] for row in group),
            "failure_gain_proxy_mean": mean(row["failure_gain_proxy"] for row in group),
            "primary_failure_score_max_mean": mean(row["primary_failure_score_max"] for row in group),
            "primary_failure_rate_mean": mean(row["primary_failure_rate"] for row in group),
        })

    write_csv(cell_rows, out_dir / "p02_decomposition_cells.csv")

    exponent_rows = []
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_variant[row["variant"]].append(row)
    for variant, group in sorted(by_variant.items()):
        attack_fit = _fit_log_exponent(group, "attack_magnitude_proxy")
        gain_fit = _fit_log_exponent(group, "failure_gain_proxy")
        # For fixed K, attack proxy slope is delta - 1.
        delta_hat = 1.0 + attack_fit["slope_n"]
        zeta_hat = gain_fit["slope_n"]
        exponent_rows.append({
            "variant": variant,
            "liquidity_exponent": group[0]["liquidity_exponent"],
            "social_on": group[0]["social_on"],
            "attack_slope_fixed_k": attack_fit["slope_n"],
            "delta_hat_proxy": delta_hat,
            "gain_slope_fixed_k": zeta_hat,
            "zeta_hat_proxy": zeta_hat,
            "n_points": min(attack_fit["n_points"], gain_fit["n_points"]),
        })
    write_csv(exponent_rows, out_dir / "p02_exponent_proxy_fits.csv")

    nu_hat = _read_nu(args.p01_run)
    baseline = next((row for row in exponent_rows if row["variant"] == "baseline_q05"), None)
    closure = {}
    if baseline is not None and not math.isnan(nu_hat):
        predicted = float(baseline["delta_hat_proxy"]) + float(baseline["zeta_hat_proxy"])
        closure = {
            "nu_hat_from_p01": nu_hat,
            "baseline_delta_hat_proxy": float(baseline["delta_hat_proxy"]),
            "baseline_zeta_hat_proxy": float(baseline["zeta_hat_proxy"]),
            "delta_plus_zeta_proxy": predicted,
            "nu_minus_delta_plus_zeta_proxy": nu_hat - predicted,
        }
    write_json({
        "warning": (
            "Uses attack_magnitude_proxy = K / N^liquidity_exponent and "
            "failure_gain_proxy = primary_failure_score_max / attack_magnitude_proxy. "
            "Do not label these as exact h_N or chi_T,N unless the simulator writes "
            "direct perturbation-response measurements."
        ),
        "p01_run": args.p01_run,
        "closure": closure,
    }, out_dir / "p02_decomposition_summary.json")
    print(f"Wrote {out_dir / 'p02_decomposition_cells.csv'}")


if __name__ == "__main__":
    main()
