"""Audit scaling-law evidence from existing scaling-theory outputs.

This script does not rerun WolfBench episodes. It reads Exp2/Exp5/Exp7/Exp8
summaries, fits simple finite-size scaling laws, and writes a paper-facing
experiment-alignment report.

Usage::
    python -m experiments.scaling_theory.analyze_scaling_law
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from experiments._common import SCALING_THEORY_OUTPUTS_ROOT, scaling_exp_dir, write_csv, write_json


EXP2_DIR = SCALING_THEORY_OUTPUTS_ROOT / "exp2_society_size_scaling"
EXP5_DIR = SCALING_THEORY_OUTPUTS_ROOT / "exp5_capacity_control"
EXP7_DIR = SCALING_THEORY_OUTPUTS_ROOT / "exp7_cross_mechanism_threshold"
EXP8_DIR = SCALING_THEORY_OUTPUTS_ROOT / "exp8_sensitivity_audit"
STABLE_N_MIN = 500


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value, default: float | None = None) -> float | None:
    if value in {None, "", "None", "none", "nan", "NaN", "null"}:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _r2(y: np.ndarray, pred: np.ndarray) -> float | None:
    if y.size < 2:
        return None
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 0:
        return None
    return 1.0 - ss_res / ss_tot


def _log_r2(y: np.ndarray, pred: np.ndarray) -> float | None:
    if np.any(y <= 0) or np.any(pred <= 0):
        return None
    return _r2(np.log(y), np.log(pred))


def _power_pred(x: np.ndarray, amplitude: float, beta: float) -> np.ndarray:
    return amplitude * np.power(x, beta)


def _finite_pred(x: np.ndarray, alpha_inf: float, amplitude: float, nu: float) -> np.ndarray:
    return alpha_inf + amplitude * np.power(x, -nu)


def _fit_power(dataset: str, metric: str, x: np.ndarray, y: np.ndarray, note: str = "") -> dict:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    row = {
        "dataset": dataset,
        "metric": metric,
        "model": "power",
        "n_points": int(x.size),
        "amplitude": None,
        "beta": None,
        "nu_or_gamma": None,
        "alpha_inf": None,
        "r2": None,
        "log_r2": None,
        "nu_ci_low": None,
        "nu_ci_high": None,
        "loo_nu_min": None,
        "loo_nu_max": None,
        "loo_nu_span": None,
        "loo_log_r2_min": None,
        "evidence_grade": None,
        "note": note,
    }
    if x.size < 2:
        row["note"] = f"{note}; insufficient positive points".strip("; ")
        return row
    beta, log_amplitude = np.polyfit(np.log(x), np.log(y), 1)
    amplitude = float(np.exp(log_amplitude))
    pred = _power_pred(x, amplitude, float(beta))
    row.update({
        "amplitude": amplitude,
        "beta": float(beta),
        "nu_or_gamma": float(-beta),
        "r2": _r2(y, pred),
        "log_r2": _log_r2(y, pred),
    })
    return row


def _fit_finite_asymptote(dataset: str, metric: str, x: np.ndarray, y: np.ndarray,
                          note: str = "") -> dict:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    row = {
        "dataset": dataset,
        "metric": metric,
        "model": "alpha_inf_plus_power",
        "n_points": int(x.size),
        "amplitude": None,
        "beta": None,
        "nu_or_gamma": None,
        "alpha_inf": None,
        "r2": None,
        "log_r2": None,
        "nu_ci_low": None,
        "nu_ci_high": None,
        "loo_nu_min": None,
        "loo_nu_max": None,
        "loo_nu_span": None,
        "loo_log_r2_min": None,
        "evidence_grade": "diagnostic_only",
        "note": note,
    }
    if x.size < 4:
        row["note"] = f"{note}; needs at least 4 points".strip("; ")
        return row
    try:
        alpha_inf0 = max(min(y) * 0.5, 1e-8)
        amplitude0 = max(y.max() - y.min(), 1e-8) * np.power(float(x.min()), 0.2)
        upper_alpha_inf = max(min(y) * 0.999, 1e-8)
        params, _ = curve_fit(
            _finite_pred,
            x,
            y,
            p0=[alpha_inf0, amplitude0, 0.2],
            bounds=([0.0, 0.0, 1e-5], [upper_alpha_inf, np.inf, 5.0]),
            maxfev=20000,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        row["note"] = f"{note}; fit failed: {exc}".strip("; ")
        return row
    alpha_inf, amplitude, nu = map(float, params)
    pred = _finite_pred(x, alpha_inf, amplitude, nu)
    row.update({
        "amplitude": amplitude,
        "beta": float(-nu),
        "nu_or_gamma": nu,
        "alpha_inf": alpha_inf,
        "r2": _r2(y, pred),
        "log_r2": _log_r2(y, pred),
    })
    return row


def _xy(rows: list[dict], y_key: str) -> tuple[np.ndarray, np.ndarray]:
    x_vals = []
    y_vals = []
    for row in rows:
        x = _float(row.get("n_society"))
        y = _float(row.get(y_key))
        if x is None or y is None:
            continue
        x_vals.append(x)
        y_vals.append(y)
    return np.array(x_vals, dtype=float), np.array(y_vals, dtype=float)


def _fmt(value, digits: int = 4) -> str:
    value = _float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}g}"


def _power_exponent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float] | None:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.unique(x).size < 2:
        return None
    beta, log_amplitude = np.polyfit(np.log(x), np.log(y), 1)
    amplitude = float(np.exp(log_amplitude))
    pred = _power_pred(x, amplitude, float(beta))
    return float(beta), float(-beta), _r2(y, pred), _log_r2(y, pred)


def _bootstrap_nu_ci(x: np.ndarray, y: np.ndarray, seed: int, n_boot: int = 4000) -> tuple[float | None, float | None]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 4:
        return None, None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, size=x.size)
        if np.unique(x[idx]).size < 2:
            continue
        fit = _power_exponent(x[idx], y[idx])
        if fit is not None:
            values.append(fit[1])
    if len(values) < 50:
        return None, None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _leave_one_out_diagnostics(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 4:
        return {
            "loo_nu_min": None,
            "loo_nu_max": None,
            "loo_nu_span": None,
            "loo_log_r2_min": None,
        }
    nu_values = []
    log_r2_values = []
    for leave_out in range(x.size):
        keep = np.ones(x.size, dtype=bool)
        keep[leave_out] = False
        fit = _power_exponent(x[keep], y[keep])
        if fit is None:
            continue
        _, nu, _, log_r2 = fit
        nu_values.append(nu)
        if log_r2 is not None:
            log_r2_values.append(log_r2)
    if not nu_values:
        return {
            "loo_nu_min": None,
            "loo_nu_max": None,
            "loo_nu_span": None,
            "loo_log_r2_min": None,
        }
    return {
        "loo_nu_min": float(np.min(nu_values)),
        "loo_nu_max": float(np.max(nu_values)),
        "loo_nu_span": float(np.max(nu_values) - np.min(nu_values)),
        "loo_log_r2_min": float(np.min(log_r2_values)) if log_r2_values else None,
    }


def _evidence_grade(row: dict) -> str:
    n_points = int(row.get("n_points") or 0)
    log_r2 = _float(row.get("log_r2"), -1.0) or -1.0
    loo_span = _float(row.get("loo_nu_span"), None)
    ci_low = _float(row.get("nu_ci_low"), None)
    ci_high = _float(row.get("nu_ci_high"), None)
    ci_width = None if ci_low is None or ci_high is None else ci_high - ci_low
    if n_points >= 10 and log_r2 >= 0.7 and loo_span is not None and loo_span <= 0.12 and ci_width is not None and ci_width <= 0.25:
        return "law_candidate"
    if n_points >= 6 and log_r2 >= 0.45:
        return "finite_size_trend"
    return "diagnostic_only"


def _enrich_power_fit(row: dict, x: np.ndarray, y: np.ndarray, seed: int) -> dict:
    if row.get("model") != "power" or row.get("amplitude") is None:
        return row
    ci_low, ci_high = _bootstrap_nu_ci(x, y, seed=seed)
    row["nu_ci_low"] = ci_low
    row["nu_ci_high"] = ci_high
    row.update(_leave_one_out_diagnostics(x, y))
    row["evidence_grade"] = _evidence_grade(row)
    return row


def _fit_rows(exp2_rows: list[dict], exp5_rows: list[dict], exp7_rows: list[dict]) -> list[dict]:
    fits = []
    n_exp2, ac_exp2 = _xy(exp2_rows, "alpha_c_logistic")
    _, width_exp2 = _xy(exp2_rows, "transition_width_10_90")
    fits.append(_enrich_power_fit(_fit_power("exp2_s1", "alpha_c", n_exp2, ac_exp2, "primary S1 scaling"), n_exp2, ac_exp2, 21_001))
    fits.append(_fit_finite_asymptote("exp2_s1", "alpha_c", n_exp2, ac_exp2,
                                      "diagnostic finite-size asymptote model; use only if alpha_inf is identifiable"))
    fits.append(_enrich_power_fit(_fit_power("exp2_s1", "transition_width_10_90", n_exp2, width_exp2,
                                            "finite-size transition sharpening"), n_exp2, width_exp2, 21_002))

    stable_rows = [row for row in exp2_rows if (_float(row.get("n_society")) or 0) >= STABLE_N_MIN]
    n_stable, ac_stable = _xy(stable_rows, "alpha_c_logistic")
    _, width_stable = _xy(stable_rows, "transition_width_10_90")
    fits.append(_enrich_power_fit(_fit_power("exp2_s1_n_ge_500", "alpha_c", n_stable, ac_stable,
                                            "exclude small-N nonmonotonic finite-size noise"), n_stable, ac_stable, 21_003))
    fits.append(_enrich_power_fit(_fit_power("exp2_s1_n_ge_500", "transition_width_10_90", n_stable, width_stable,
                                            "stable-regime transition sharpening"), n_stable, width_stable, 21_004))

    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in exp5_rows:
        by_mode[str(row.get("capacity_mode"))].append(row)
    for mode, rows in by_mode.items():
        n_vals, ac_vals = _xy(rows, "alpha_c_logistic")
        _, width_vals = _xy(rows, "transition_width_10_90")
        fits.append(_enrich_power_fit(_fit_power(f"exp5_{mode}", "alpha_c", n_vals, ac_vals,
                            "capacity-control robustness check"), n_vals, ac_vals, 22_000 + len(fits)))
        fits.append(_enrich_power_fit(_fit_power(f"exp5_{mode}", "transition_width_10_90", n_vals, width_vals,
                            "capacity-control width check"), n_vals, width_vals, 22_000 + len(fits)))

    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for row in exp7_rows:
        by_scenario[str(row.get("scenario"))].append(row)
    for scenario, rows in sorted(by_scenario.items()):
        n_vals, ac_vals = _xy(rows, "alpha_c_logistic")
        _, width_vals = _xy(rows, "transition_width_10_90")
        fits.append(_enrich_power_fit(_fit_power(f"exp7_{scenario}", "alpha_c", n_vals, ac_vals,
                            "cross-mechanism threshold audit"), n_vals, ac_vals, 23_000 + len(fits)))
        fits.append(_enrich_power_fit(_fit_power(f"exp7_{scenario}", "transition_width_10_90", n_vals, width_vals,
                            "cross-mechanism transition width"), n_vals, width_vals, 23_000 + len(fits)))
    return fits


def _plot_exp2_scaling(exp2_rows: list[dict], fits: list[dict], out: Path) -> None:
    x, y = _xy(exp2_rows, "alpha_c_logistic")
    lows = np.array([_float(row.get("alpha_c_ci_low"), _float(row.get("alpha_c_logistic"))) for row in exp2_rows], dtype=float)
    highs = np.array([_float(row.get("alpha_c_ci_high"), _float(row.get("alpha_c_logistic"))) for row in exp2_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(x, y, yerr=[y - lows, highs - y], fmt="o", capsize=4, label="Exp2 logistic alpha_c")
    fit_x = np.logspace(np.log10(float(x.min())), np.log10(float(x.max())), 100)
    for fit in fits:
        if fit["dataset"] != "exp2_s1" or fit["metric"] != "alpha_c" or fit["amplitude"] is None:
            continue
        if fit["model"] == "power":
            fit_y = _power_pred(fit_x, float(fit["amplitude"]), float(fit["beta"]))
            label = f"power: nu={_fmt(fit['nu_or_gamma'], 3)}, logR2={_fmt(fit['log_r2'], 3)}"
        else:
            fit_y = _finite_pred(fit_x, float(fit["alpha_inf"]), float(fit["amplitude"]), float(fit["nu_or_gamma"]))
            label = f"alpha_inf+power: alpha_inf={_fmt(fit['alpha_inf'], 3)}"
        ax.plot(fit_x, fit_y, "--", label=label)
    stable_fit = _fit_lookup(fits, "exp2_s1_n_ge_500", "alpha_c")
    if stable_fit and stable_fit["amplitude"] is not None:
        stable_x = np.logspace(np.log10(500.0), np.log10(float(x.max())), 100)
        stable_y = _power_pred(stable_x, float(stable_fit["amplitude"]), float(stable_fit["beta"]))
        ax.plot(stable_x, stable_y, ":", color="C3",
                label=f"N>=500 power: nu={_fmt(stable_fit['nu_or_gamma'], 3)}")
    ax.set_xscale("log")
    ax.set_xlabel("Society size N")
    ax.set_ylabel("Estimated critical harmful ratio alpha_c")
    ax.set_title("S1 finite-size scaling fits")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "s1_alpha_scaling_fits.png", dpi=170)
    plt.close(fig)


def _plot_width(exp2_rows: list[dict], fits: list[dict], out: Path) -> None:
    x, y = _xy(exp2_rows, "transition_width_10_90")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, y, "o", label="Exp2 width")
    fit = next((row for row in fits if row["dataset"] == "exp2_s1" and row["metric"] == "transition_width_10_90"), None)
    if fit and fit["amplitude"] is not None:
        fit_x = np.logspace(np.log10(float(x.min())), np.log10(float(x.max())), 100)
        fit_y = _power_pred(fit_x, float(fit["amplitude"]), float(fit["beta"]))
        ax.plot(fit_x, fit_y, "--", label=f"power: gamma={_fmt(fit['nu_or_gamma'], 3)}")
    stable_fit = _fit_lookup(fits, "exp2_s1_n_ge_500", "transition_width_10_90")
    if stable_fit and stable_fit["amplitude"] is not None:
        stable_x = np.logspace(np.log10(500.0), np.log10(float(x.max())), 100)
        stable_y = _power_pred(stable_x, float(stable_fit["amplitude"]), float(stable_fit["beta"]))
        ax.plot(stable_x, stable_y, ":", color="C3",
                label=f"N>=500 power: gamma={_fmt(stable_fit['nu_or_gamma'], 3)}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Society size N")
    ax.set_ylabel("Transition width alpha(0.9)-alpha(0.1)")
    ax.set_title("S1 finite-size transition sharpening")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "s1_width_scaling_fit.png", dpi=170)
    plt.close(fig)


def _plot_cross_mechanism(exp7_rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for row in exp7_rows:
        by_scenario[str(row.get("scenario"))].append(row)
    for scenario, rows in sorted(by_scenario.items()):
        x, y = _xy(rows, "alpha_c_logistic")
        if x.size == 0:
            continue
        order = np.argsort(x)
        ax.plot(x[order], y[order], "-o", label=scenario.upper())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Society size N")
    ax.set_ylabel("Estimated critical harmful ratio alpha_c")
    ax.set_title("Mechanism-specific alpha_c(N)")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "cross_mechanism_alpha_scaling.png", dpi=170)
    plt.close(fig)


def _plot_capacity(exp5_rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in exp5_rows:
        by_mode[str(row.get("capacity_mode"))].append(row)
    for mode, rows in sorted(by_mode.items()):
        x, y = _xy(rows, "alpha_c_logistic")
        if x.size == 0:
            continue
        order = np.argsort(x)
        ax.plot(x[order], y[order], "-o", label=mode)
    ax.set_xscale("log")
    ax.set_xlabel("Society size N")
    ax.set_ylabel("Estimated critical harmful ratio alpha_c")
    ax.set_title("Capacity-control scaling check")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "capacity_control_alpha_scaling.png", dpi=170)
    plt.close(fig)


def _strongest_sensitivities(exp8_rows: list[dict], limit: int = 8) -> list[dict]:
    rows = sorted(
        exp8_rows,
        key=lambda row: _float(row.get("delta_collapse_rate"), 0.0) or 0.0,
        reverse=True,
    )
    return rows[:limit]


def _fit_lookup(fits: list[dict], dataset: str, metric: str, model: str = "power") -> dict | None:
    return next(
        (row for row in fits if row["dataset"] == dataset and row["metric"] == metric and row["model"] == model),
        None,
    )


def _interval(row: dict | None, low_key: str, high_key: str, digits: int = 3) -> str:
    if row is None:
        return "NA"
    low = _fmt(row.get(low_key), digits)
    high = _fmt(row.get(high_key), digits)
    if low == "NA" or high == "NA":
        return "NA"
    return f"[{low}, {high}]"


def _exp2_law_rows(fits: list[dict]) -> list[dict]:
    rows = []
    for row in fits:
        if not str(row.get("dataset", "")).startswith("exp2_s1"):
            continue
        if row.get("model") != "power":
            continue
        rows.append({
            "dataset": row.get("dataset"),
            "metric": row.get("metric"),
            "n_points": row.get("n_points"),
            "nu_or_gamma": row.get("nu_or_gamma"),
            "nu_ci_low": row.get("nu_ci_low"),
            "nu_ci_high": row.get("nu_ci_high"),
            "log_r2": row.get("log_r2"),
            "loo_nu_min": row.get("loo_nu_min"),
            "loo_nu_max": row.get("loo_nu_max"),
            "loo_nu_span": row.get("loo_nu_span"),
            "loo_log_r2_min": row.get("loo_log_r2_min"),
            "evidence_grade": row.get("evidence_grade"),
            "note": row.get("note"),
        })
    return rows


def _write_markdown(out: Path, fits: list[dict], exp8_rows: list[dict]) -> None:
    exp2_power = _fit_lookup(fits, "exp2_s1", "alpha_c")
    exp2_finite = _fit_lookup(fits, "exp2_s1", "alpha_c", "alpha_inf_plus_power")
    exp2_width = _fit_lookup(fits, "exp2_s1", "transition_width_10_90")
    stable_power = _fit_lookup(fits, "exp2_s1_n_ge_500", "alpha_c")
    stable_width = _fit_lookup(fits, "exp2_s1_n_ge_500", "transition_width_10_90")
    s2_power = _fit_lookup(fits, "exp7_s2", "alpha_c")
    s4_power = _fit_lookup(fits, "exp7_s4", "alpha_c")
    exp2_n_points = int(exp2_power.get("n_points") or 0) if exp2_power else 0
    exp2_status = "dense" if exp2_n_points >= 10 else "sparse"

    top = _strongest_sensitivities(exp8_rows)
    top_lines = []
    for row in top:
        top_lines.append(
            f"- {row['scenario'].upper()} / {row['family']}: delta={_fmt(row['delta_collapse_rate'], 3)} "
            f"({row['value_at_min']} -> {row['value_at_max']})"
        )

    text = f"""# Scaling-law audit

## Verdict

WolfBench should state the theory as a finite-size scaling ansatz, not as a thermodynamic phase-transition theorem. The main law claim is protocol-specific: under the fixed S1 Exp2 protocol, collapse curves define an estimated midpoint alpha_c(N) and transition width w_N, and these estimands can be fitted by empirical power laws. Current Exp2 status is **{exp2_status}** with {exp2_n_points} positive N points; use the evidence grade below to decide whether to write "empirical finite-size scaling law" or the weaker "finite-size scaling trend".

## Current fitted laws

- Exp2 S1 alpha_c power ansatz: alpha_c ~= A * N^beta, with beta={_fmt(exp2_power and exp2_power['beta'], 4)} and nu={_fmt(exp2_power and exp2_power['nu_or_gamma'], 4)}. Bootstrap nu CI={_interval(exp2_power, 'nu_ci_low', 'nu_ci_high')}; leave-one-N-out nu range={_interval(exp2_power, 'loo_nu_min', 'loo_nu_max')}; log-space R2={_fmt(exp2_power and exp2_power['log_r2'], 4)}; grade={exp2_power.get('evidence_grade') if exp2_power else 'NA'}.
- Exp2 S1 finite-asymptote diagnostic: alpha_inf={_fmt(exp2_finite and exp2_finite['alpha_inf'], 4)}, nu={_fmt(exp2_finite and exp2_finite['nu_or_gamma'], 4)}. Report this only as a diagnostic unless dense data identify a positive alpha_inf with stable uncertainty.
- Exp2 S1 transition-width ansatz: width ~= B * N^(-gamma), with gamma={_fmt(exp2_width and exp2_width['nu_or_gamma'], 4)}. Bootstrap gamma CI={_interval(exp2_width, 'nu_ci_low', 'nu_ci_high')}; leave-one-N-out gamma range={_interval(exp2_width, 'loo_nu_min', 'loo_nu_max')}; log-space R2={_fmt(exp2_width and exp2_width['log_r2'], 4)}; grade={exp2_width.get('evidence_grade') if exp2_width else 'NA'}.
- Stable-regime S1 alpha_c fit using N>={STABLE_N_MIN}: nu={_fmt(stable_power and stable_power['nu_or_gamma'], 4)}, CI={_interval(stable_power, 'nu_ci_low', 'nu_ci_high')}. This row protects the main claim from small-N finite-size noise.
- Stable-regime S1 width fit using N>={STABLE_N_MIN}: gamma={_fmt(stable_width and stable_width['nu_or_gamma'], 4)}, CI={_interval(stable_width, 'nu_ci_low', 'nu_ci_high')}.
- Exp7 S2 alpha_c fit: nu={_fmt(s2_power and s2_power['nu_or_gamma'], 4)}. This is the strongest cross-mechanism scaling signal.
- Exp7 S4 alpha_c fit: nu={_fmt(s4_power and s4_power['nu_or_gamma'], 4)}. The sign is not expected to match S1 because S4's dominant channel is diffuse under the current collapse metric.

## Proposition alignment

| Theory claim | Current support | Weak point | Best experiment fix |
| --- | --- | --- | --- |
| Proposition 1: finite-size critical harmful ratio exists | Exp2 logistic alpha_c and width estimates; Exp7 S1/S2 thresholds | Exp2 summary has the power fit but not a dedicated scaling-law table; S3/S4 are not yet clean threshold examples | Keep Exp2 as main evidence; report both alpha_c and width exponents; make Exp7 explicitly mechanism-specific rather than universal |
| Proposition 2: comparative statics | Exp8 shows sensitivity ranges; Exp5 checks capacity confounding | Exp8 currently measures delta P(collapse), not delta alpha_c; derivative signs are therefore indirect | Add targeted threshold sweeps varying L, G, beta_social/risk, and placement, then report delta alpha_c and sign tests |
| Proposition 3: defense objective | Existing defense results can show loss reduction | There is no alpha_c(defense)-alpha_c(NoGuard) table yet | Add a defense threshold-shift experiment around S1/S2 critical grids and report ThresholdShift plus utility cost |
| Mechanism heterogeneity | Exp7 shows S1/S2/S3/S4 have different alpha_c scales; Exp8 shows different dominant sensitivities | S3 N=500 has no threshold in the current grid; S4 has a broad/flat transition | Give each mechanism its own dominant-channel prediction and alpha grid; do not force one universal exponent |

## Strongest Exp8 sensitivity evidence

{chr(10).join(top_lines) if top_lines else '- No Exp8 delta rows found.'}

## Recommended experiment optimization

1. Make Exp2 the canonical finite-size scaling ansatz. Report alpha_c(N)=A*N^(-nu) as the conservative main fit. Mention alpha_c(N)=alpha_inf+a*N^(-nu) only as a robustness diagnostic unless dense data identify a positive alpha_inf. Use N>={STABLE_N_MIN} as a robustness row because small-N points can have visible finite-size noise.
2. Add one plot and one CSV for transition width scaling. The width exponent is as important as alpha_c: it directly supports the finite-size critical-regime claim.
3. Reframe Exp5 as a confound check, not a main theorem test. It asks whether the S1 scaling survives capacity normalization. To make it cleaner, add N=500 and N=2000 or increase seeds at N=200 where the CI is very wide.
4. Upgrade Exp8 into a threshold-shift comparative-statics experiment. For each lever, run alpha grids around S1/S2 critical values and report delta alpha_c instead of only delta P(collapse). Minimal levers: asset_liquidity_scale, social_mean_degree, retail_risk_appetite, and placement.
5. Keep mechanism heterogeneity explicit. S1 should be social exposure plus price momentum; S2 high-centrality reach; S3 microstructure/liquidity stress; S4 volume-signal or wash-liquidity illusion. Each mechanism should get its own alpha grid and its own dominant-channel paragraph.
6. For S3, expand the N=500 alpha grid above 0.6 and use tighter grids around the observed N=1000 and N=2000 thresholds. The current N=500 constant/fallback row is a grid-coverage problem, not a theory result.
7. For S4, do not sell the current alpha_c slope as a failure or success. Treat it as a broad transition. Add component-level threshold metrics or a wider alpha grid up to roughly 0.25-0.30 to see whether the composite collapse metric saturates late.
8. Add a defense threshold-shift experiment: NoGuard, WolfGuard, and one simple baseline across alpha near the NoGuard alpha_c. The table should be alpha_c(defense), alpha_c(NoGuard), ThresholdShift, utility cost, and false-positive cost.

## Paper-facing interpretation

The strongest theory story is not a universal phase transition. It is a protocol-specific empirical law: estimate P(collapse | N, alpha), summarize its midpoint alpha_c and width, fit finite-size scaling exponents with uncertainty and leave-one-N-out stability, then show how mechanisms and controls shift those estimands. This turns the experiments into tests of the theory rather than illustrations after the fact.
"""
    (out / "scaling_law_audit.md").write_text(text)


def main() -> None:
    out = scaling_exp_dir("scaling_law_audit")
    exp2_rows = _read_csv(EXP2_DIR / "alpha_critical_summary.csv")
    exp5_rows = _read_csv(EXP5_DIR / "alpha_critical_capacity_summary.csv")
    exp7_rows = _read_csv(EXP7_DIR / "alpha_critical_by_mechanism.csv")
    exp8_rows = _read_csv(EXP8_DIR / "sensitivity_delta_summary.csv")

    fits = _fit_rows(exp2_rows, exp5_rows, exp7_rows)
    write_csv(fits, out / "scaling_law_fits.csv")
    write_csv(_exp2_law_rows(fits), out / "exp2_dense_law_summary.csv")
    write_json({
        "inputs": {
            "exp2": str(EXP2_DIR / "alpha_critical_summary.csv"),
            "exp5": str(EXP5_DIR / "alpha_critical_capacity_summary.csv"),
            "exp7": str(EXP7_DIR / "alpha_critical_by_mechanism.csv"),
            "exp8": str(EXP8_DIR / "sensitivity_delta_summary.csv"),
        },
        "figures": {
            "s1_alpha_scaling_fits": "s1_alpha_scaling_fits.png",
            "s1_width_scaling_fit": "s1_width_scaling_fit.png",
            "cross_mechanism_alpha_scaling": "cross_mechanism_alpha_scaling.png",
            "capacity_control_alpha_scaling": "capacity_control_alpha_scaling.png",
        },
        "tables": {
            "all_fits": "scaling_law_fits.csv",
            "exp2_law_summary": "exp2_dense_law_summary.csv",
        },
        "fits": fits,
    }, out / "summary.json")

    if exp2_rows:
        _plot_exp2_scaling(exp2_rows, fits, out)
        _plot_width(exp2_rows, fits, out)
    if exp7_rows:
        _plot_cross_mechanism(exp7_rows, out)
    if exp5_rows:
        _plot_capacity(exp5_rows, out)
    _write_markdown(out, fits, exp8_rows)
    print(f"Wrote scaling-law audit to {out}")


if __name__ == "__main__":
    main()