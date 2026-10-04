#!/usr/bin/env python3
"""Analyze only fresh P0 rerun rows and emit conservative claim audits."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw"
OUT = HERE / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = json.loads((HERE / "config.json").read_text())
RNG = np.random.default_rng(20260821)


def read_rows(stem: str) -> list[dict[str, Any]]:
    path = RAW / f"{stem}.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok" and int(row.get("fresh_rerun", 0)) == 1:
                    rows.append(row)
    if not rows:
        return []
    # Never mix simulator fingerprints in one analysis.
    hashes = defaultdict(int)
    for row in rows:
        hashes[str(row["source_hash"])] += 1
    selected = max(hashes, key=hashes.get)
    return [row for row in rows if str(row["source_hash"]) == selected]


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def q_label(row: dict[str, Any]) -> str:
    return f"q={float(row['native_liquidity_exponent']):g}"


def fit_surface(rows: list[dict[str, Any]]) -> dict[str, float]:
    usable = [
        row for row in rows
        if float(row.get("primary_failure_score_max", 0.0)) > 0
        and float(row.get("requested_harmful_count", 0.0) or 0.0) > 0
    ]
    if len(usable) < 4:
        return {"b_n": math.nan, "b_k": math.nan, "nu_response": math.nan, "n_rows": len(usable)}
    x = np.asarray([
        [1.0, math.log(float(row["n_society"])), math.log(float(row["requested_harmful_count"]))]
        for row in usable
    ])
    y = np.log(np.asarray([float(row["primary_failure_score_max"]) for row in usable]))
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    b_n, b_k = float(beta[1]), float(beta[2])
    nu = 1.0 + b_n / b_k if abs(b_k) > 1e-12 else math.nan
    return {"b_n": b_n, "b_k": b_k, "nu_response": nu, "n_rows": len(usable)}


def cluster_bootstrap_surface(rows: list[dict[str, Any]], n_boot: int = 1000) -> dict[str, float]:
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) < 2:
        return {}
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    values = []
    for _ in range(n_boot):
        sample = []
        for seed in RNG.choice(seeds, size=len(seeds), replace=True):
            sample.extend(by_seed[int(seed)])
        fit = fit_surface(sample)
        if all(math.isfinite(fit[key]) for key in ("b_n", "b_k", "nu_response")):
            values.append([fit["b_n"], fit["b_k"], fit["nu_response"]])
    if not values:
        return {}
    arr = np.asarray(values)
    out = {}
    for index, key in enumerate(("b_n", "b_k", "nu_response")):
        out[f"{key}_ci_lo"], out[f"{key}_ci_hi"] = [float(v) for v in np.quantile(arr[:, index], [0.025, 0.975])]
    out["bootstrap_n"] = len(values)
    return out


def aggregate_curve(rows: list[dict[str, Any]], variant_key: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(variant_key(row), int(row["n_society"]), float(row["alpha"]))].append(row)
    out = []
    for (variant, n_society, alpha), group in sorted(groups.items()):
        out.append({
            "variant": variant,
            "n_society": n_society,
            "alpha": alpha,
            "n_seeds": len({int(row["seed"]) for row in group}),
            "failure_rate": mean(float(row["primary_failure_rate"]) for row in group),
            "response_mean": mean(float(row["primary_failure_score_max"]) for row in group),
            "response_sd": float(np.std([float(row["primary_failure_score_max"]) for row in group], ddof=1)) if len(group) > 1 else 0.0,
        })
    return out


def crossing(group: list[dict[str, Any]], target: float = 0.5) -> tuple[float | None, str]:
    curve = sorted(group, key=lambda row: float(row["alpha"]))
    values = [float(row["failure_rate"]) for row in curve]
    if not curve or min(values) > target or max(values) < target:
        return None, "not_bracketed"
    for left, right in zip(curve, curve[1:]):
        y0, y1 = float(left["failure_rate"]), float(right["failure_rate"])
        if y0 == target:
            return float(left["alpha"]), "exact"
        if (y0 - target) * (y1 - target) <= 0 and y1 != y0:
            x0, x1 = float(left["alpha"]), float(right["alpha"])
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0), "interpolated"
    if values[-1] == target:
        return float(curve[-1]["alpha"]), "exact"
    return None, "nonmonotone_unresolved"


def boundary_table(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in curves:
        groups[(row["variant"], int(row["n_society"]))].append(row)
    out = []
    for (variant, n_society), group in sorted(groups.items()):
        alpha_c, status = crossing(group)
        out.append({
            "variant": variant,
            "n_society": n_society,
            "alpha_c": "" if alpha_c is None else alpha_c,
            "boundary_status": status,
            "n_seeds": max(int(row["n_seeds"]) for row in group),
            "min_failure_rate": min(float(row["failure_rate"]) for row in group),
            "max_failure_rate": max(float(row["failure_rate"]) for row in group),
        })
    return out


def estimate_two_size_nu(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        if row["alpha_c"] != "":
            groups[row["variant"]].append(row)
    out = []
    for variant, group in sorted(groups.items()):
        group = sorted(group, key=lambda row: int(row["n_society"]))
        if len(group) < 2:
            continue
        n0, n1 = float(group[0]["n_society"]), float(group[-1]["n_society"])
        a0, a1 = float(group[0]["alpha_c"]), float(group[-1]["alpha_c"])
        nu = -math.log(a1 / a0) / math.log(n1 / n0) if a0 > 0 and a1 > 0 else math.nan
        out.append({"variant": variant, "n_low": int(n0), "n_high": int(n1), "nu_two_size": nu})
    return out


def bootstrap_boundary_nu(
    raw_rows: list[dict[str, Any]],
    variant_key: Callable[[dict[str, Any]], str],
    n_boot: int = 1000,
) -> dict[str, dict[str, float]]:
    seeds = sorted({int(row["seed"]) for row in raw_rows})
    variants = sorted({variant_key(row) for row in raw_rows})
    if len(seeds) < 2:
        return {}
    by_seed = defaultdict(list)
    for row in raw_rows:
        by_seed[int(row["seed"])].append(row)
    values = defaultdict(list)
    for _ in range(n_boot):
        sample = []
        for seed in RNG.choice(seeds, size=len(seeds), replace=True):
            sample.extend(by_seed[int(seed)])
        boundaries = boundary_table(aggregate_curve(sample, variant_key))
        for row in estimate_two_size_nu(boundaries):
            if math.isfinite(float(row["nu_two_size"])):
                values[str(row["variant"])].append(float(row["nu_two_size"]))
    out = {}
    for variant in variants:
        arr = np.asarray(values.get(variant, []), dtype=float)
        if len(arr):
            lo, hi = np.quantile(arr, [0.025, 0.975])
            out[variant] = {
                "nu_ci_lo": float(lo),
                "nu_ci_hi": float(hi),
                "bootstrap_n": len(arr),
            }
    return out


def add_boundary_bootstrap(
    table: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    variant_key: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    boot = bootstrap_boundary_nu(raw_rows, variant_key)
    for row in table:
        row.update(boot.get(str(row["variant"]), {}))
    return table


def analyze_liquidity_fixed(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fits = []
    for q in sorted({float(row["native_liquidity_exponent"]) for row in rows}):
        group = [row for row in rows if float(row["native_liquidity_exponent"]) == q]
        for regime, predicate in [
            ("all_positive", lambda r: float(r["primary_failure_score_max"]) > 0),
            ("subcritical_R_lt_1", lambda r: 0 < float(r["primary_failure_score_max"]) < 1),
        ]:
            selected = [row for row in group if predicate(row)]
            fit = fit_surface(selected)
            fit.update(cluster_bootstrap_surface(selected))
            fit.update({"liquidity_exponent": q, "regime": regime, "n_seeds": len({int(r["seed"]) for r in selected})})
            fits.append(fit)

    cells = aggregate_curve(rows, q_label)
    plt.figure(figsize=(7.4, 4.8))
    for q in sorted({float(row["native_liquidity_exponent"]) for row in rows}):
        q_rows = [row for row in rows if float(row["native_liquidity_exponent"]) == q]
        for k in sorted({int(row["requested_harmful_count"]) for row in q_rows}):
            selected = [row for row in q_rows if int(row["requested_harmful_count"]) == k]
            groups = defaultdict(list)
            for row in selected:
                groups[int(row["n_society"])].append(float(row["primary_failure_score_max"]))
            ns = sorted(groups)
            ys = [mean(groups[n]) for n in ns]
            plt.plot(ns, ys, marker="o", label=f"q={q:g}, K={k}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Society size N")
    plt.ylabel("Mean primary response R")
    plt.title("Fresh native-liquidity fixed-count reruns")
    plt.legend(ncol=3, fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "liquidity_fixed_k.png", dpi=180)
    plt.close()
    return fits, cells


def analyze_severity(rows: list[dict[str, Any]], observed_nu: float | None) -> list[dict[str, Any]]:
    baseline = [row for row in rows if abs(float(row["native_liquidity_exponent"]) - 0.5) < 1e-9]
    bins = [
        ("R_lt_0.3", 0.0, 0.3),
        ("R_0.3_to_0.6", 0.3, 0.6),
        ("R_0.6_to_1", 0.6, 1.0),
        ("R_ge_1", 1.0, math.inf),
        ("all_subcritical_R_lt_1", 0.0, 1.0),
    ]
    out = []
    for label, lo, hi in bins:
        selected = [row for row in baseline if lo < float(row["primary_failure_score_max"]) <= hi]
        fit = fit_surface(selected)
        fit.update(cluster_bootstrap_surface(selected))
        discrepancy = None
        kappa = None
        if observed_nu is not None and math.isfinite(fit["nu_response"]):
            discrepancy = fit["nu_response"] - observed_nu
            kappa = fit["b_k"] * discrepancy
        fit.update({
            "stratification": "realized_severity",
            "severity_regime": label,
            "n_seeds": len({int(row["seed"]) for row in selected}),
            "n_unique_cells": len({(int(row["n_society"]), int(row["requested_harmful_count"])) for row in selected}),
            "observed_boundary_nu": "" if observed_nu is None else observed_nu,
            "nu_response_minus_boundary": "" if discrepancy is None else discrepancy,
            "kappa_residual": "" if kappa is None else kappa,
        })
        out.append(fit)
    return out


def analyze_boundary_distance(
    fixed_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    boundary_estimates: list[dict[str, Any]],
    observed_nu: float | None,
) -> list[dict[str, Any]]:
    """Fit response surfaces by pre-defined distance to the fresh boundary."""
    alpha_c = {
        int(row["n_society"]): float(row["alpha_c"])
        for row in boundary_estimates
        if row["variant"] == "q=0.5" and row["alpha_c"] != ""
    }
    candidates = []
    for original in fixed_rows + boundary_rows:
        if abs(float(original["native_liquidity_exponent"]) - 0.5) > 1e-9:
            continue
        n_society = int(original["n_society"])
        if n_society not in alpha_c or float(original["alpha"]) <= 0:
            continue
        row = dict(original)
        if not row.get("requested_harmful_count", ""):
            row["requested_harmful_count"] = int(row.get("n_harmful", round(float(row["alpha"]) * n_society)))
        row["boundary_distance"] = float(row["alpha"]) / alpha_c[n_society]
        candidates.append(row)
    # A fixed-K cell can also appear in the boundary sweep; keep one copy.
    deduped = {}
    for row in candidates:
        key = (int(row["n_society"]), float(row["alpha"]), int(row["seed"]))
        deduped[key] = row
    candidates = list(deduped.values())
    bins = [
        ("alpha_over_alpha_c_lt_0.5", 0.0, 0.5),
        ("alpha_over_alpha_c_0.5_to_0.9", 0.5, 0.9),
        ("alpha_over_alpha_c_0.9_to_1.3", 0.9, 1.3),
        ("alpha_over_alpha_c_ge_1.3", 1.3, math.inf),
    ]
    out = []
    for label, lo, hi in bins:
        selected = [row for row in candidates if lo < float(row["boundary_distance"]) <= hi]
        fit = fit_surface(selected)
        fit.update(cluster_bootstrap_surface(selected))
        discrepancy = None
        kappa = None
        if observed_nu is not None and math.isfinite(fit["nu_response"]):
            discrepancy = fit["nu_response"] - observed_nu
            kappa = fit["b_k"] * discrepancy
        fit.update({
            "stratification": "distance_to_fresh_boundary",
            "severity_regime": label,
            "n_seeds": len({int(row["seed"]) for row in selected}),
            "n_unique_cells": len({(int(row["n_society"]), float(row["alpha"])) for row in selected}),
            "observed_boundary_nu": "" if observed_nu is None else observed_nu,
            "nu_response_minus_boundary": "" if discrepancy is None else discrepancy,
            "kappa_residual": "" if kappa is None else kappa,
        })
        out.append(fit)
    return out


def plot_curves(curves: list[dict[str, Any]], name: str, title: str) -> None:
    if not curves:
        return
    plt.figure(figsize=(7.2, 4.6))
    for variant in sorted({row["variant"] for row in curves}):
        for n_society in sorted({int(row["n_society"]) for row in curves if row["variant"] == variant}):
            group = sorted(
                [row for row in curves if row["variant"] == variant and int(row["n_society"]) == n_society],
                key=lambda row: float(row["alpha"]),
            )
            plt.plot(
                [float(row["alpha"]) for row in group],
                [float(row["failure_rate"]) for row in group],
                marker="o",
                label=f"{variant}, N={n_society}",
            )
    plt.axhline(0.5, color="black", lw=0.8, ls="--")
    plt.xlabel("Harmful fraction alpha")
    plt.ylabel("Primary failure probability")
    plt.ylim(-0.03, 1.03)
    plt.title(title)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180)
    plt.close()


def paired_feedback_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells = defaultdict(dict)
    for row in rows:
        key = (int(row["n_society"]), float(row["alpha"]), int(row["seed"]))
        cells[key][row["variant"]] = row
    groups = defaultdict(list)
    for (n_society, alpha, _seed), variants in cells.items():
        baseline = variants.get("closed_loop")
        if baseline is None:
            continue
        for variant in ("no_environment_feedback", "no_social_propagation", "direct_harmful_first_hop"):
            treated = variants.get(variant)
            if treated is None:
                continue
            groups[(variant, n_society, alpha)].append(
                float(treated["primary_failure_score_max"]) - float(baseline["primary_failure_score_max"])
            )
    out = []
    for (variant, n_society, alpha), effects in sorted(groups.items()):
        arr = np.asarray(effects)
        out.append({
            "variant": variant,
            "n_society": n_society,
            "alpha": alpha,
            "n_pairs": len(arr),
            "paired_delta_response_mean": float(arr.mean()),
            "paired_delta_response_sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "fraction_reduced": float(np.mean(arr < 0)),
        })
    return out


def paired_endpoint_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired ablation effects for joint, social, and market endpoints."""
    endpoints = (
        "primary_failure_score_max",
        "social_cascade_peak",
        "price_dislocation_max",
        "liquidity_stress_max",
        "retail_loss_pct_30d",
    )
    cells = defaultdict(dict)
    for row in rows:
        key = (int(row["n_society"]), float(row["alpha"]), int(row["seed"]))
        cells[key][str(row["variant"])] = row
    effects = defaultdict(list)
    for (n_society, alpha, _seed), variants in cells.items():
        baseline = variants.get("closed_loop")
        if baseline is None:
            continue
        for variant in ("no_environment_feedback", "no_social_propagation", "direct_harmful_first_hop"):
            treated = variants.get(variant)
            if treated is None:
                continue
            for endpoint in endpoints:
                effects[(variant, endpoint, n_society, alpha)].append(
                    float(treated[endpoint]) - float(baseline[endpoint])
                )
    out = []
    for (variant, endpoint, n_society, alpha), values in sorted(effects.items()):
        arr = np.asarray(values, dtype=float)
        out.append({
            "variant": variant,
            "endpoint": endpoint,
            "n_society": n_society,
            "alpha": alpha,
            "n_pairs": len(arr),
            "treated_minus_closed_mean": float(arr.mean()),
            "treated_minus_closed_sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "fraction_treated_lower": float(np.mean(arr < 0)),
        })
    return out


def clean_normalized_endpoints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Subtract the matched alpha=0 path before interpreting market outcomes.

    The absolute S1 market thresholds are already exceeded in many alpha=0
    episodes, so raw market threshold crossings are not a valid causal endpoint.
    """
    endpoints = (
        "social_cascade_peak",
        "price_dislocation_max",
        "liquidity_stress_max",
        "retail_loss_pct_30d",
    )
    clean = {}
    for row in rows:
        if abs(float(row["alpha"])) < 1e-12:
            clean[(str(row["variant"]), int(row["n_society"]), int(row["seed"]))] = row
    groups = defaultdict(list)
    for row in rows:
        key = (str(row["variant"]), int(row["n_society"]), int(row["seed"]))
        reference = clean.get(key)
        if reference is None:
            continue
        for endpoint in endpoints:
            groups[(str(row["variant"]), endpoint, int(row["n_society"]), float(row["alpha"]))].append(
                float(row[endpoint]) - float(reference[endpoint])
            )
    out = []
    for (variant, endpoint, n_society, alpha), values in sorted(groups.items()):
        arr = np.asarray(values, dtype=float)
        out.append({
            "variant": variant,
            "endpoint": endpoint,
            "n_society": n_society,
            "alpha": alpha,
            "n_pairs": len(arr),
            "alpha_minus_paired_clean_mean": float(arr.mean()),
            "alpha_minus_paired_clean_sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        })
    return out


def near_boundary_contrasts(
    rows: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    n_boot: int = 2000,
) -> list[dict[str, Any]]:
    """Seed-clustered paired effects in 0.5 <= alpha/alpha_c <= 1.5."""
    alpha_c = {
        int(row["n_society"]): float(row["alpha_c"])
        for row in boundaries
        if row["variant"] == "closed_loop" and row["alpha_c"] != ""
    }
    cells = defaultdict(dict)
    for row in rows:
        n_society = int(row["n_society"])
        if n_society not in alpha_c:
            continue
        ratio = float(row["alpha"]) / alpha_c[n_society]
        if 0.5 <= ratio <= 1.5:
            cells[(n_society, float(row["alpha"]), int(row["seed"]))][str(row["variant"])] = row
    endpoints = (
        "primary_failure_score_max", "social_cascade_peak", "price_dislocation_max",
        "liquidity_stress_max", "retail_loss_pct_30d",
    )
    per_seed = defaultdict(list)
    for (_n, _alpha, seed), variants in cells.items():
        baseline = variants.get("closed_loop")
        if baseline is None:
            continue
        for variant in ("no_environment_feedback", "direct_harmful_first_hop", "no_social_propagation"):
            treated = variants.get(variant)
            if treated is None:
                continue
            for endpoint in endpoints:
                per_seed[(variant, endpoint, seed)].append(float(treated[endpoint]) - float(baseline[endpoint]))
    grouped = defaultdict(dict)
    for (variant, endpoint, seed), values in per_seed.items():
        grouped[(variant, endpoint)][seed] = float(np.mean(values))
    out = []
    for (variant, endpoint), seed_values in sorted(grouped.items()):
        seeds = sorted(seed_values)
        arr = np.asarray([seed_values[seed] for seed in seeds], dtype=float)
        boot = np.asarray([
            np.mean(RNG.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)
        ]) if len(arr) >= 2 else np.asarray([])
        out.append({
            "variant": variant,
            "endpoint": endpoint,
            "n_seeds": len(arr),
            "treated_minus_closed_mean": float(arr.mean()),
            "ci_lo": "" if not len(boot) else float(np.quantile(boot, 0.025)),
            "ci_hi": "" if not len(boot) else float(np.quantile(boot, 0.975)),
            "fraction_seed_means_lower": float(np.mean(arr < 0)),
            "window": "0.5 <= alpha/closed_loop_alpha_c <= 1.5",
        })
    return out


def make_report(
    fixed_rows: list[dict[str, Any]],
    fixed_fits: list[dict[str, Any]],
    liquidity_boundaries: list[dict[str, Any]],
    feedback_boundaries: list[dict[str, Any]],
    severity: list[dict[str, Any]],
) -> None:
    seeds = sorted({int(row["seed"]) for row in fixed_rows})
    status = "frozen 12-seed" if len(seeds) >= 12 else f"pilot ({len(seeds)} seeds)"
    lines = [
        "# Fresh P0 causal experiment audit",
        "",
        f"Analysis status: **{status}**. All analyzed rows have `fresh_rerun=1`; historical rows are excluded.",
        "",
        "## 1. Native liquidity rerun",
        "",
    ]
    for row in fixed_fits:
        if row["regime"] != "subcritical_R_lt_1":
            continue
        lines.append(
            f"- q={row['liquidity_exponent']:g}: b_N={row['b_n']:.3f}, "
            f"b_K={row['b_k']:.3f}, nu_response={row['nu_response']:.3f} "
            f"({int(row['n_rows'])} episode rows)."
        )
    resolved_liq = [row for row in liquidity_boundaries if row["alpha_c"] != ""]
    lines.extend([
        "",
        f"Resolved fresh liquidity boundaries: {len(resolved_liq)}/{len(liquidity_boundaries)}. "
        "An unresolved cell is not extrapolated.",
        "",
        "## 2. Closed-loop ablation",
        "",
        "Important metric audit: the S1 primary score is the minimum of a social-threshold score and a market-threshold score. "
        "It is therefore structurally zero when social propagation is disabled; that fact is not causal evidence by itself.",
        "",
    ])
    for row in feedback_boundaries:
        value = "unresolved" if row["alpha_c"] == "" else f"alpha_c={float(row['alpha_c']):.4f}"
        lines.append(f"- {row['variant']}, N={row['n_society']}: {value} ({row['boundary_status']}).")
    lines.extend([
        "",
        "Interpret boundary movement only after all three paired conditions are bracketed. "
        "The no-environment intervention changes information flow during simulation; it is not post-hoc normalization.",
        "Absolute market thresholds are also audited against paired alpha=0 runs. If alpha=0 already exceeds a market threshold, "
        "only clean-normalized market/retail contrasts are interpreted.",
        "",
        "## 3. Response-regime crossover",
        "",
    ])
    for row in severity:
        lines.append(
            f"- {row['stratification']} / {row['severity_regime']}: nu_response={row['nu_response']:.3f}, "
            f"b_N={row['b_n']:.3f}, b_K={row['b_k']:.3f}, "
            f"rows={int(row['n_rows'])}, unique cells={int(row['n_unique_cells'])}."
        )
    baseline_fit = next((
        row for row in fixed_fits
        if abs(float(row["liquidity_exponent"]) - 0.5) < 1e-9
        and row["regime"] == "subcritical_R_lt_1"
    ), None)
    baseline_boundary = [
        row for row in liquidity_boundaries
        if row["variant"] == "q=0.5" and row["alpha_c"] != ""
    ]
    if baseline_fit is not None and len(baseline_boundary) >= 2:
        ordered = sorted(baseline_boundary, key=lambda row: int(row["n_society"]))
        n0, n1 = float(ordered[0]["n_society"]), float(ordered[-1]["n_society"])
        a0, a1 = float(ordered[0]["alpha_c"]), float(ordered[-1]["alpha_c"])
        nu_b = -math.log(a1 / a0) / math.log(n1 / n0)
        gap = float(baseline_fit["nu_response"]) - nu_b
        kappa = float(baseline_fit["b_k"]) * gap
        lines.extend([
            "",
            "### Fresh discrepancy decomposition",
            "",
            f"For the current pilot, nu_response={float(baseline_fit['nu_response']):.3f} and the "
            f"two-size boundary exponent is nu_b={nu_b:.3f}, leaving a gap of {gap:.3f}.",
            f"Under a near-boundary correction R ~ N^(b_N-kappa) K^b_K, the algebraic residual is "
            f"kappa=b_K(nu_response-nu_b)={kappa:.3f}.",
            "This kappa is a target for the stratified tests above, not yet an independent theoretical finding.",
        ])
    lines.extend([
        "",
        "Severity-conditioned fits are diagnostic rather than automatically causal: conditioning on realized R can change cell support. "
        "A crossover claim requires ordered movement with adequate N/K support and seed-bootstrap intervals, not just point estimates.",
        "",
        "## Reporting guardrails",
        "",
        "- Do not call this real-LLM evidence; the fresh mechanism run is behavioral-only.",
        "- Do not report a critical fraction when the evaluated grid fails to bracket 50% failure.",
        "- Do not claim a universal scaling law from the two-size boundary check.",
        "- Preserve the source hash and frozen config when extending the run.",
    ])
    (OUT / "CLAIM_AUDIT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    fixed_rows = read_rows("liquidity_fixed_k")
    boundary_rows = read_rows("liquidity_boundary")
    feedback_rows = read_rows("feedback_ablation")

    fixed_fits, fixed_cells = analyze_liquidity_fixed(fixed_rows) if fixed_rows else ([], [])
    liquidity_curves = aggregate_curve(boundary_rows, q_label) if boundary_rows else []
    liquidity_boundaries = boundary_table(liquidity_curves)
    liquidity_nu = add_boundary_bootstrap(
        estimate_two_size_nu(liquidity_boundaries), boundary_rows, q_label
    )

    feedback_curves = aggregate_curve(feedback_rows, lambda row: str(row["variant"])) if feedback_rows else []
    feedback_boundaries = boundary_table(feedback_curves)
    feedback_nu = add_boundary_bootstrap(
        estimate_two_size_nu(feedback_boundaries), feedback_rows, lambda row: str(row["variant"])
    )
    feedback_effects = paired_feedback_effects(feedback_rows)
    endpoint_effects = paired_endpoint_effects(feedback_rows)
    clean_effects = clean_normalized_endpoints(feedback_rows)
    near_effects = near_boundary_contrasts(feedback_rows, feedback_boundaries)

    baseline_nu = next(
        (float(row["nu_two_size"]) for row in liquidity_nu if row["variant"] == "q=0.5"),
        None,
    )
    severity = analyze_severity(fixed_rows, baseline_nu) if fixed_rows else []
    severity.extend(analyze_boundary_distance(
        fixed_rows, boundary_rows, liquidity_boundaries, baseline_nu
    ))

    write_csv("liquidity_fixed_k_cells.csv", fixed_cells)
    write_csv("liquidity_surface_fits.csv", fixed_fits)
    write_csv("liquidity_boundary_curves.csv", liquidity_curves)
    write_csv("liquidity_boundaries.csv", liquidity_boundaries)
    write_csv("liquidity_two_size_nu.csv", liquidity_nu)
    write_csv("feedback_curves.csv", feedback_curves)
    write_csv("feedback_boundaries.csv", feedback_boundaries)
    write_csv("feedback_two_size_nu.csv", feedback_nu)
    write_csv("feedback_paired_effects.csv", feedback_effects)
    write_csv("feedback_all_endpoint_effects.csv", endpoint_effects)
    write_csv("feedback_clean_normalized_endpoints.csv", clean_effects)
    write_csv("feedback_near_boundary_contrasts.csv", near_effects)
    write_csv("severity_crossover_fits.csv", severity)
    plot_curves(liquidity_curves, "liquidity_boundary.png", "Native liquidity boundary reruns")
    plot_curves(feedback_curves, "feedback_ablation.png", "Closed-loop causal ablation")
    make_report(fixed_rows, fixed_fits, liquidity_boundaries, feedback_boundaries, severity)
    print(f"Wrote fresh analysis to {OUT}")


if __name__ == "__main__":
    main()
