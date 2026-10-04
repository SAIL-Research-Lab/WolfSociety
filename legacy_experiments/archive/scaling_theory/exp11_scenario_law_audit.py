"""Exp11: scenario-specific law audit for S1-S4.

This script does not run new simulations. It reads the formal mechanism sweep
artifacts and writes a compact paper-facing table answering a narrower question:
does each canonical scenario show the expected alpha-scaling law under the
metric that matches its mechanism?

S1/S2/S3 use collapse-transition evidence. S4 uses fake-liquidity mechanism
evidence because its generic collapse trigger is intentionally broad and sits on
a plateau once wash trading is active.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import SCALING_THEORY_OUTPUTS_ROOT, scaling_exp_dir, write_csv, write_json


OUT_NAME = "exp11_scenario_law_audit"
MECHANISM_DIR = SCALING_THEORY_OUTPUTS_ROOT / "exp7_cross_mechanism_threshold"


SCENARIO_SPECS = {
    "s1": {
        "label": "S1 pump-and-dump",
        "n_society": 1000,
        "metric": "collapse_rate",
        "law": "collapse transition",
        "pass_delta": 0.50,
        "pass_spearman": 0.75,
        "requires_alpha_c": True,
        "caveat": "sharp social pump transition",
    },
    "s2": {
        "label": "S2 finfluencer scalping",
        "n_society": 1000,
        "metric": "collapse_rate",
        "law": "collapse transition",
        "pass_delta": 0.50,
        "pass_spearman": 0.75,
        "requires_alpha_c": True,
        "caveat": "central placement produces low alpha_c",
    },
    "s3": {
        "label": "S3 spoofing/layering",
        "n_society": 1000,
        "metric": "collapse_rate",
        "law": "liquidity-stress collapse transition",
        "pass_delta": 0.50,
        "pass_spearman": 0.70,
        "requires_alpha_c": True,
        "caveat": "N=500 is below tested critical grid; N>=1000 shows transition",
    },
    "s4": {
        "label": "S4 wash trading/fake liquidity",
        "n_society": 1000,
        "metric": "volume_distortion_max",
        "law": "fake-liquidity mechanism scaling",
        "pass_delta": 3.0,
        "pass_spearman": 0.90,
        "requires_alpha_c": False,
        "caveat": "generic collapse trigger is broad; mechanism metric is the primary law evidence",
    },
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _rank(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty_like(arr, dtype=float)
    ranks[order] = np.arange(arr.size, dtype=float)
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(ys) < 3:
        return 0.0
    rx = _rank(xs)
    ry = _rank(ys)
    if float(rx.std()) <= 1e-12 or float(ry.std()) <= 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _group_metric(rows: list[dict[str, Any]], scenario: str, n_society: int,
                  metric: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    alphas = sorted({
        _float(row["alpha"])
        for row in rows
        if row["scenario"] == scenario and int(row["n_society"]) == int(n_society)
    })
    for alpha in alphas:
        selected = [
            row for row in rows
            if row["scenario"] == scenario
            and int(row["n_society"]) == int(n_society)
            and _float(row["alpha"]) == alpha
        ]
        values = [_float(row[metric]) for row in selected]
        out.append({
            "scenario": scenario,
            "n_society": n_society,
            "alpha": alpha,
            "metric": metric,
            "mean": float(np.mean(values)) if values else 0.0,
            "n": len(values),
        })
    return out


def _threshold_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        out[(row["scenario"], int(row["n_society"]))] = row
    return out


def build_law_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = _read_csv(MECHANISM_DIR / "data.csv")
    threshold_rows = _read_csv(MECHANISM_DIR / "alpha_critical_by_mechanism.csv")
    thresholds = _threshold_lookup(threshold_rows)

    curve_rows: list[dict[str, Any]] = []
    law_rows: list[dict[str, Any]] = []
    for scenario, spec in SCENARIO_SPECS.items():
        n_society = int(spec["n_society"])
        metric = str(spec["metric"])
        grouped = _group_metric(raw_rows, scenario, n_society, metric)
        curve_rows.extend(grouped)
        xs = [_float(row["alpha"]) for row in grouped]
        ys = [_float(row["mean"]) for row in grouped]
        delta = float(ys[-1] - ys[0]) if ys else 0.0
        spearman = _spearman(xs, ys)
        threshold = thresholds.get((scenario, n_society), {})
        alpha_c = threshold.get("alpha_c_logistic") or ""
        width = threshold.get("transition_width_10_90") or ""
        alpha_c_ok = bool(alpha_c) or not bool(spec["requires_alpha_c"])
        law_pass = (
            alpha_c_ok
            and delta >= float(spec["pass_delta"])
            and spearman >= float(spec["pass_spearman"])
        )
        law_rows.append({
            "scenario": scenario,
            "scenario_label": spec["label"],
            "n_society": n_society,
            "law": spec["law"],
            "primary_metric": metric,
            "alpha_min": xs[0] if xs else None,
            "alpha_max": xs[-1] if xs else None,
            "metric_at_alpha_min": ys[0] if ys else None,
            "metric_at_alpha_max": ys[-1] if ys else None,
            "metric_delta": delta,
            "spearman_alpha_metric": spearman,
            "alpha_c_logistic": alpha_c,
            "transition_width_10_90": width,
            "law_pass": law_pass,
            "caveat": spec["caveat"],
        })
    return law_rows, curve_rows


def _write_report(rows: list[dict[str, Any]], path: Path) -> None:
    pass_count = sum(1 for row in rows if row["law_pass"])
    lines = [
        "# Exp11 Scenario-Specific Law Audit",
        "",
        f"Scenario-specific law pass: {pass_count}/{len(rows)}",
        "",
        "| scenario | law | primary metric | metric delta | Spearman | alpha_c | width | pass | caveat |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario'].upper()} | {row['law']} | {row['primary_metric']} | "
            f"{_float(row['metric_delta']):.4g} | {_float(row['spearman_alpha_metric']):.3f} | "
            f"{row['alpha_c_logistic']} | {row['transition_width_10_90']} | "
            f"{row['law_pass']} | {row['caveat']} |"
        )
    lines.extend([
        "",
        "Interpretation: this table is not a universal comparative-statics claim.",
        "It verifies that each canonical scenario has a mechanism-aligned alpha-scaling law.",
        "S4 should be reported with fake-liquidity diagnostics because its generic collapse transition is broad.",
        "",
    ])
    path.write_text("\n".join(lines))


def _plot(rows: list[dict[str, Any]], curve_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), squeeze=False)
    for ax, row in zip(axes.ravel(), rows):
        points = [r for r in curve_rows if r["scenario"] == row["scenario"]]
        points.sort(key=lambda item: _float(item["alpha"]))
        ax.plot(
            [_float(point["alpha"]) for point in points],
            [_float(point["mean"]) for point in points],
            marker="o",
            linewidth=1.8,
        )
        if row["primary_metric"] == "collapse_rate":
            ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{row['scenario'].upper()}: {row['law']}")
        ax.set_xlabel("alpha")
        ax.set_ylabel(row["primary_metric"])
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    out = scaling_exp_dir(OUT_NAME)
    law_rows, curve_rows = build_law_rows()
    write_csv(law_rows, out / "scenario_law_summary.csv")
    write_csv(curve_rows, out / "scenario_law_curves.csv")
    _write_report(law_rows, out / "report.md")
    _plot(law_rows, curve_rows, out / "scenario_law_audit.png")
    write_json({"scenario_laws": law_rows}, out / "summary.json")
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()