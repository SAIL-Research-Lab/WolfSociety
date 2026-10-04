#!/usr/bin/env python3
"""Generate reviewer-facing reinforcement tables from v3 artifacts.

This script does not run new simulations. It consolidates existing v3 outputs
into auditable tables for rebuttal and supplement drafting.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file() and (parent / "src" / "wolfbench").is_dir()
)
OUT = Path(__file__).resolve().parents[1] / "generated"

P01_OUTPUTS = ROOT / "paper_experiments_v3" / "outputs"
P01_DATA_GLOB = "p01_nonlinear_scaling_paper_s*/data.csv"
P01_PILOT_GLOB = "p01_nonlinear_scaling_pilot_s*/data.csv"
P02_DATA_GLOB = "p02_size_decomposition_paper_s*/data.csv"
P02_BOUNDARY_DATA_GLOB = "p02_boundary_support_decomposition*/data.csv"
BOOTSTRAP_JSON = ROOT / "paper_experiments_v3" / "figures" / "generated" / "scaling_exponent_bootstrap.json"
SCALING_RESULTS = ROOT / "paper_experiments_v3" / "figures" / "generated" / "table2_scaling_results.csv"
REVIEWER_AUDIT = ROOT / "paper_experiments_v3" / "outputs" / "reviewer_scaling_audit"
P09_DEPTH_NU = REVIEWER_AUDIT / "p09_depth_nu.csv"
P09_DEPTH_ALPHA_C = REVIEWER_AUDIT / "p09_depth_alpha_c.csv"
P10_LLM_NU = REVIEWER_AUDIT / "p10_llm_quota_nu.csv"
P10_LLM_ALPHA_C = REVIEWER_AUDIT / "p10_llm_quota_alpha_c.csv"
P11_WATTS_NU = REVIEWER_AUDIT / "p11_watts_null_nu.csv"
P11_WATTS_ALPHA_C = REVIEWER_AUDIT / "p11_watts_null_alpha_c.csv"

CROSS_MODELS = {
    "qwen235b": ROOT / "paper_experiments_v3" / "outputs" / "p01_crossmodel_qwen235b" / "rows.jsonl",
    "gpt41": ROOT / "paper_experiments_v3" / "outputs" / "p01_crossmodel_gpt41" / "rows.jsonl",
}

S1_THRESHOLDS = {
    "social": 0.55,
    "price": 0.21,
    "liquidity": 0.85,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_many(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        for row in read_csv(path):
            row = dict(row)
            row["_source_file"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, title: str, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows available.")
    else:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def as_float(value, default=math.nan) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def fmt(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}g}"


def crossing(points: list[tuple[float, float]], target: float = 0.5) -> tuple[float | None, tuple[float, float] | None]:
    points = sorted(points)
    for (a0, p0), (a1, p1) in zip(points, points[1:]):
        if p0 == target:
            return a0, (a0, a0)
        if (p0 - target) * (p1 - target) <= 0 and p1 != p0:
            alpha_c = a0 + (target - p0) * (a1 - a0) / (p1 - p0)
            return alpha_c, (a0, a1)
    return None, None


def fit_log_slope(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 2:
        return None, None
    lx = [math.log(x) for x, _ in pairs]
    ly = [math.log(y) for _, y in pairs]
    mx = mean(lx)
    my = mean(ly)
    denom = sum((x - mx) ** 2 for x in lx)
    if denom == 0:
        return None, None
    slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / denom
    intercept = my - slope * mx
    return slope, intercept


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def s1_score(row: dict, scale: float = 1.0) -> float:
    q = as_float(row.get("social_cascade_peak"), 0.0)
    d = as_float(row.get("price_dislocation_max"), 0.0)
    ell = as_float(row.get("liquidity_stress_max"), 0.0)
    social = q / (S1_THRESHOLDS["social"] * scale)
    market = max(
        d / (S1_THRESHOLDS["price"] * scale),
        ell / (S1_THRESHOLDS["liquidity"] * scale),
    )
    return max(0.0, min(social, market))


def load_published_alpha_c() -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in read_csv(SCALING_RESULTS):
        n = as_int(row.get("N"))
        if n:
            out[n] = row
    return out


def summarize_grid_counts(rows: list[dict], published: dict[int, dict[str, str]]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        n = as_int(row.get("n_society"))
        alpha = as_float(row.get("alpha"))
        grouped[(n, alpha)].append(row)

    point_rows = []
    curves: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for (n, alpha), cell_rows in sorted(grouped.items()):
        values = [as_float(row.get("primary_failure_rate"), 0.0) for row in cell_rows]
        collapse_count = sum(1 for value in values if value >= 0.5)
        n_harmful_values = sorted({as_int(row.get("n_harmful")) for row in cell_rows})
        mean_failure = mean(values) if values else math.nan
        curves[n].append((alpha, mean_failure))
        point_rows.append(
            {
                "n_society": n,
                "alpha": fmt(alpha, 6),
                "k_round_alpha_n": as_int(round(alpha * n)),
                "observed_n_harmful": ";".join(str(value) for value in n_harmful_values),
                "n_seeds": len(cell_rows),
                "primary_failure_count": collapse_count,
                "primary_failure_rate_mean": fmt(mean_failure, 4),
                "llm_total_minmax": f"{min(as_int(r.get('n_llm_total')) for r in cell_rows)}-{max(as_int(r.get('n_llm_total')) for r in cell_rows)}",
                "source_files": len({r.get("_source_file", "") for r in cell_rows}),
            }
        )

    boundary_rows = []
    for n, points in sorted(curves.items()):
        alpha_c, bracket = crossing(points)
        alphas = [alpha for alpha, _ in sorted(points)]
        grid_steps = [b - a for a, b in zip(alphas, alphas[1:])]
        boundary_rows.append(
            {
                "n_society": n,
                "n_alpha_points": len(alphas),
                "alpha_min": fmt(min(alphas), 6),
                "alpha_max": fmt(max(alphas), 6),
                "min_grid_step": fmt(min(grid_steps) if grid_steps else math.nan, 6),
                "max_grid_step": fmt(max(grid_steps) if grid_steps else math.nan, 6),
                "alpha_c_linear": fmt(alpha_c, 6),
                "effective_k_c": fmt(alpha_c * n if alpha_c is not None else math.nan, 6),
                "published_alpha_c": published.get(n, {}).get("alpha_c", ""),
                "published_k_c": published.get(n, {}).get("K_c", ""),
                "raw_minus_published_alpha_c": fmt(
                    alpha_c - as_float(published.get(n, {}).get("alpha_c"))
                    if alpha_c is not None and n in published
                    else math.nan,
                    6,
                ),
                "crossing_bracket_alpha": "" if bracket is None else f"{fmt(bracket[0], 6)}-{fmt(bracket[1], 6)}",
                "crossing_bracket_k": "" if bracket is None else f"{as_int(round(bracket[0] * n))}-{as_int(round(bracket[1] * n))}",
                "coverage_status": "crosses_0.5" if alpha_c is not None else "unresolved",
            }
        )
    return point_rows, boundary_rows


def threshold_and_component_sensitivity(rows: list[dict]) -> list[dict]:
    outcome_specs = []
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        outcome_specs.append((f"s1_gate_scale_{scale:g}", lambda row, scale=scale: s1_score(row, scale) >= 1.0))
    outcome_specs.extend(
        [
            ("social_component", lambda row: as_float(row.get("social_cascade_peak"), 0.0) >= S1_THRESHOLDS["social"]),
            ("price_component", lambda row: as_float(row.get("price_dislocation_max"), 0.0) >= S1_THRESHOLDS["price"]),
            ("liquidity_component", lambda row: as_float(row.get("liquidity_stress_max"), 0.0) >= S1_THRESHOLDS["liquidity"]),
        ]
    )

    out_rows = []
    for name, predicate in outcome_specs:
        grouped: dict[tuple[int, float], list[float]] = defaultdict(list)
        for row in rows:
            if row.get("status") != "ok":
                continue
            n = as_int(row.get("n_society"))
            alpha = as_float(row.get("alpha"))
            grouped[(n, alpha)].append(1.0 if predicate(row) else 0.0)

        alpha_cs = []
        for n in sorted({key[0] for key in grouped}):
            points = []
            for (nn, alpha), values in grouped.items():
                if nn == n:
                    points.append((alpha, mean(values)))
            alpha_c, bracket = crossing(points)
            out_rows.append(
                {
                    "outcome": name,
                    "n_society": n,
                    "alpha_c": fmt(alpha_c, 6),
                    "effective_k_c": fmt(alpha_c * n if alpha_c is not None else math.nan, 6),
                    "crossing_bracket_alpha": "" if bracket is None else f"{fmt(bracket[0], 6)}-{fmt(bracket[1], 6)}",
                    "coverage_status": "crosses_0.5" if alpha_c is not None else "unresolved",
                }
            )
            if alpha_c is not None:
                alpha_cs.append((n, alpha_c))
        slope, _ = fit_log_slope([n for n, _ in alpha_cs], [alpha for _, alpha in alpha_cs])
        nu = -slope if slope is not None else None
        out_rows.append(
            {
                "outcome": name,
                "n_society": "ALL_RESOLVED",
                "alpha_c": "",
                "effective_k_c": "",
                "crossing_bracket_alpha": "",
                "coverage_status": f"n_resolved={len(alpha_cs)}; nu={fmt(nu, 4)}",
            }
        )
    return out_rows


def bootstrap_summary() -> list[dict]:
    if not BOOTSTRAP_JSON.exists():
        return []
    data = json.loads(BOOTSTRAP_JSON.read_text())
    return [
        {
            "estimate": "all_size_resolved",
            "nu_hat": fmt(data.get("nu_hat"), 4),
            "boot_mean": fmt(data.get("nu_boot_mean"), 4),
            "ci_lo": fmt(data.get("nu_ci_lo"), 4),
            "ci_hi": fmt(data.get("nu_ci_hi"), 4),
            "n_boot_effective": data.get("n_boot_effective", ""),
            "notes": f"resolved_count_hist={data.get('resolved_count_hist', {})}",
        },
        {
            "estimate": "partial_ge2_resolved",
            "nu_hat": "",
            "boot_mean": fmt(data.get("partial_ge2_boot_mean"), 4),
            "ci_lo": fmt(data.get("partial_ge2_ci_lo"), 4),
            "ci_hi": fmt(data.get("partial_ge2_ci_hi"), 4),
            "n_boot_effective": data.get("partial_ge2_n_boot_effective", ""),
            "notes": f"missing_by_n={data.get('missing_by_n', {})}",
        },
    ]


def drop_n_sensitivity(boundary_rows: list[dict]) -> list[dict]:
    parsed = []
    for row in boundary_rows:
        alpha_c = as_float(row.get("alpha_c_linear"))
        n = as_int(row.get("n_society"))
        if n and alpha_c > 0 and not math.isnan(alpha_c):
            parsed.append((n, alpha_c))
    specs = {
        "all_resolved": lambda n: True,
        "drop_n100": lambda n: n != 100,
        "drop_n100_n200": lambda n: n not in {100, 200},
    }
    out = []
    for name, keep in specs.items():
        points = [(n, alpha) for n, alpha in parsed if keep(n)]
        slope, _ = fit_log_slope([n for n, _ in points], [alpha for _, alpha in points])
        out.append(
            {
                "fit": name,
                "n_points": len(points),
                "n_values": ";".join(str(n) for n, _ in points),
                "nu": fmt(-slope if slope is not None else None, 4),
            }
        )
    return out


def _fit_gain_surface(rows: list[dict]) -> dict[str, float]:
    usable = []
    for row in rows:
        n = as_float(row.get("n_society"))
        k = max(as_float(row.get("requested_harmful_count"), as_float(row.get("n_harmful"), 1.0)), 1.0)
        gain = max(as_float(row.get("failure_gain_proxy")), 1e-9)
        if n > 0 and gain > 0 and not math.isnan(gain):
            usable.append((math.log(n), math.log(k), math.log(gain)))
    if len(usable) < 3:
        return {
            "intercept": math.nan,
            "slope_n": math.nan,
            "slope_k": math.nan,
            "n_points": len(usable),
        }
    x_rows = [[1.0, log_n, log_k] for log_n, log_k, _ in usable]
    y_vals = [log_gain for _, _, log_gain in usable]

    # Solve the small 3-parameter normal equation directly to keep this script
    # dependency-free. The design is well-conditioned for the fixed P02 grid.
    xtx = [[sum(row[i] * row[j] for row in x_rows) for j in range(3)] for i in range(3)]
    xty = [sum(row[i] * y for row, y in zip(x_rows, y_vals)) for i in range(3)]
    beta = _solve_3x3(xtx, xty)
    if beta is None:
        return {
            "intercept": math.nan,
            "slope_n": math.nan,
            "slope_k": math.nan,
            "n_points": len(usable),
        }
    return {
        "intercept": beta[0],
        "slope_n": beta[1],
        "slope_k": beta[2],
        "n_points": len(usable),
    }


def _solve_3x3(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    a = [row[:] + [value] for row, value in zip(matrix, rhs)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for row in range(3):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [value - factor * base for value, base in zip(a[row], a[col])]
    return [a[row][3] for row in range(3)]


def _closure_score(fit: dict[str, float], n: float, k: float, q: float = 0.5) -> float:
    intercept = fit["intercept"]
    slope_n = fit["slope_n"]
    slope_k = fit["slope_k"]
    if any(math.isnan(value) for value in [intercept, slope_n, slope_k]) or n <= 0 or k <= 0:
        return math.nan
    return math.exp(intercept) * (n ** (slope_n - q)) * (k ** (1.0 + slope_k))


def _predict_alpha_c_from_closure(fit: dict[str, float], n: float, threshold: float, q: float = 0.5) -> float | None:
    intercept = fit["intercept"]
    slope_n = fit["slope_n"]
    slope_k = fit["slope_k"]
    denom = 1.0 + slope_k
    if (
        any(math.isnan(value) for value in [intercept, slope_n, slope_k])
        or n <= 0
        or threshold <= 0
        or abs(denom) < 1e-9
    ):
        return None
    log_k = (math.log(threshold) - intercept - (slope_n - q) * math.log(n)) / denom
    return math.exp(log_k) / n


def _geomean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(mean(math.log(value) for value in positive))


def _loo_constant_alpha(observed: list[tuple[int, float]], holdout_n: int) -> float | None:
    return _geomean([alpha for n, alpha in observed if n != holdout_n])


def _loo_powerlaw_alpha(observed: list[tuple[int, float]], holdout_n: int) -> float | None:
    train = [(n, alpha) for n, alpha in observed if n != holdout_n]
    slope, intercept = fit_log_slope([n for n, _ in train], [alpha for _, alpha in train])
    if slope is None or intercept is None:
        return None
    return math.exp(intercept + slope * math.log(holdout_n))


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None

    def ranks(values: list[float]) -> list[float]:
        ordered = sorted(enumerate(values), key=lambda item: item[1])
        out = [0.0] * len(values)
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][1] == ordered[index][1]:
                end += 1
            rank = 0.5 * (index + end - 1) + 1.0
            for original, _ in ordered[index:end]:
                out[original] = rank
            index = end
        return out

    rx = ranks([x for x, _ in pairs])
    ry = ranks([y for _, y in pairs])
    mx = mean(rx)
    my = mean(ry)
    cov = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def heldout_closure_test(
    p02_rows: list[dict],
    boundary_rows: list[dict],
    source_label: str,
) -> tuple[list[dict], list[dict]]:
    """Train proxy closure on P02 cells and predict held-out P01 alpha_c(N).

    The tested model is intentionally simple and auditable:
    score_hat(N,K) = attack_proxy(N,K) * gain_hat(N,K), where P02 provides
    gain_hat via log(gain_proxy) ~ log(N) + log(K). For each held-out N, the
    gain surface is fit without any P02 rows at that N. The calibrated variant
    also estimates one scalar threshold from the remaining P01 boundary rows,
    then predicts the held-out P01 boundary.
    """
    baseline_rows = [
        row for row in p02_rows
        if row.get("status") == "ok" and str(row.get("variant", "")).startswith("baseline_q05")
    ]
    observed = []
    for row in boundary_rows:
        n = as_int(row.get("n_society"))
        alpha_c = as_float(row.get("alpha_c_linear"))
        if n and alpha_c > 0 and not math.isnan(alpha_c):
            observed.append((n, alpha_c))
    observed_by_n = dict(observed)
    p02_ns = {as_int(row.get("n_society")) for row in baseline_rows}
    holdout_ns = sorted(n for n, _ in observed if n in p02_ns)

    detail_rows: list[dict] = []
    for holdout_n in holdout_ns:
        observed_alpha = observed_by_n[holdout_n]
        observed_k = observed_alpha * holdout_n
        baseline_methods = [
            ("loo_constant_geomean", _loo_constant_alpha(observed, holdout_n)),
            ("loo_p01_powerlaw", _loo_powerlaw_alpha(observed, holdout_n)),
        ]
        for method, predicted_alpha in baseline_methods:
            predicted_k = predicted_alpha * holdout_n if predicted_alpha is not None else None
            log_error = math.log(predicted_alpha / observed_alpha) if predicted_alpha and observed_alpha > 0 else math.nan
            detail_rows.append(
                {
                    "method": method,
                    "closure_data_source": "p01_boundary_only",
                    "heldout_n": holdout_n,
                    "train_n_values": ";".join(str(n) for n, _ in observed if n != holdout_n),
                    "train_p02_rows": 0,
                    "gain_intercept": "",
                    "gain_slope_n": "",
                    "gain_slope_k": "",
                    "closure_threshold": "",
                    "observed_alpha_c": fmt(observed_alpha, 6),
                    "predicted_alpha_c": fmt(predicted_alpha, 6),
                    "alpha_ratio_pred_obs": fmt(predicted_alpha / observed_alpha if predicted_alpha else math.nan, 6),
                    "abs_log_alpha_error": fmt(abs(log_error) if not math.isnan(log_error) else math.nan, 6),
                    "signed_log_alpha_error": fmt(log_error, 6),
                    "observed_k_c": fmt(observed_k, 6),
                    "predicted_k_c": fmt(predicted_k, 6),
                    "protocol": "leave-one-N-out baseline; no P02 gain surface used",
                }
            )
        train_rows = [row for row in baseline_rows if as_int(row.get("n_society")) != holdout_n]
        train_ns = sorted({as_int(row.get("n_society")) for row in train_rows})
        fit = _fit_gain_surface(train_rows)
        strict_threshold = 1.0
        calibration_scores = [
            _closure_score(fit, n, observed_alpha * n)
            for n, observed_alpha in observed
            if n != holdout_n and n in train_ns
        ]
        calibrated_threshold = _geomean(calibration_scores)
        methods = [
            ("strict_proxy_threshold", strict_threshold),
            ("train_boundary_calibrated", calibrated_threshold),
        ]
        for method, threshold in methods:
            predicted_alpha = _predict_alpha_c_from_closure(fit, holdout_n, threshold) if threshold else None
            predicted_k = predicted_alpha * holdout_n if predicted_alpha is not None else None
            log_error = math.log(predicted_alpha / observed_alpha) if predicted_alpha and observed_alpha > 0 else math.nan
            detail_rows.append(
                {
                    "method": method,
                    "closure_data_source": source_label,
                    "heldout_n": holdout_n,
                    "train_n_values": ";".join(str(n) for n in train_ns),
                    "train_p02_rows": fit["n_points"],
                    "gain_intercept": fmt(fit["intercept"], 6),
                    "gain_slope_n": fmt(fit["slope_n"], 6),
                    "gain_slope_k": fmt(fit["slope_k"], 6),
                    "closure_threshold": fmt(threshold, 6),
                    "observed_alpha_c": fmt(observed_alpha, 6),
                    "predicted_alpha_c": fmt(predicted_alpha, 6),
                    "alpha_ratio_pred_obs": fmt(predicted_alpha / observed_alpha if predicted_alpha else math.nan, 6),
                    "abs_log_alpha_error": fmt(abs(log_error) if not math.isnan(log_error) else math.nan, 6),
                    "signed_log_alpha_error": fmt(log_error, 6),
                    "observed_k_c": fmt(observed_k, 6),
                    "predicted_k_c": fmt(predicted_k, 6),
                    "protocol": "leave-one-N-out; train P02 gain surface without held-out N; calibrated threshold uses only non-held-out P01 boundaries",
                }
            )

    summary_rows: list[dict] = []
    mean_error_by_method: dict[str, float] = {}
    for method in sorted({row["method"] for row in detail_rows}):
        rows = [row for row in detail_rows if row["method"] == method]
        abs_errors = [as_float(row.get("abs_log_alpha_error")) for row in rows]
        abs_errors = [value for value in abs_errors if math.isfinite(value)]
        mean_error_by_method[method] = mean(abs_errors) if abs_errors else math.nan
    constant_error = mean_error_by_method.get("loo_constant_geomean", math.nan)
    for method in sorted({row["method"] for row in detail_rows}):
        rows = [row for row in detail_rows if row["method"] == method]
        observed_alpha = [as_float(row.get("observed_alpha_c")) for row in rows]
        predicted_alpha = [as_float(row.get("predicted_alpha_c")) for row in rows]
        abs_errors = [as_float(row.get("abs_log_alpha_error")) for row in rows]
        abs_errors = [value for value in abs_errors if math.isfinite(value)]
        factors = [max(pred / obs, obs / pred) for pred, obs in zip(predicted_alpha, observed_alpha) if pred > 0 and obs > 0]
        obs_slope, _ = fit_log_slope([as_int(row.get("heldout_n")) for row in rows], observed_alpha)
        pred_slope, _ = fit_log_slope([as_int(row.get("heldout_n")) for row in rows], predicted_alpha)
        skill = 1.0 - (mean_error_by_method[method] / constant_error) if constant_error > 0 else math.nan
        spearman = _spearman(observed_alpha, predicted_alpha)
        is_proxy_method = method in {"strict_proxy_threshold", "train_boundary_calibrated"}
        beats_constant = mean_error_by_method[method] < constant_error if is_proxy_method and math.isfinite(constant_error) else ""
        summary_rows.append(
            {
                "method": method,
                "closure_data_source": rows[0].get("closure_data_source", ""),
                "n_holdouts": len(rows),
                "mean_abs_log_alpha_error": fmt(mean(abs_errors) if abs_errors else math.nan, 6),
                "median_abs_log_alpha_error": fmt(median(abs_errors), 6),
                "max_abs_log_alpha_error": fmt(max(abs_errors) if abs_errors else math.nan, 6),
                "skill_vs_loo_constant": fmt(skill, 6),
                "beats_loo_constant": beats_constant,
                "within_factor_1p5": f"{sum(1 for value in factors if value <= 1.5)}/{len(factors)}",
                "within_factor_2": f"{sum(1 for value in factors if value <= 2.0)}/{len(factors)}",
                "observed_nu": fmt(-obs_slope if obs_slope is not None else math.nan, 6),
                "predicted_nu": fmt(-pred_slope if pred_slope is not None else math.nan, 6),
                "nu_error_pred_minus_obs": fmt(
                    (-pred_slope) - (-obs_slope)
                    if pred_slope is not None and obs_slope is not None
                    else math.nan,
                    6,
                ),
                "spearman_observed_predicted_alpha": fmt(spearman, 6),
                "interpretation": (
                    "positive closure claim allowed only if proxy method beats the leave-one-N-out constant baseline"
                ),
            }
        )
    return detail_rows, summary_rows


def llm_ablation_summary() -> list[dict]:
    rows = []
    nu_by_group = {row.get("group", ""): row for row in read_csv(P10_LLM_NU)}
    for row in read_csv(P10_LLM_ALPHA_C):
        group = row.get("group", "")
        nu_row = nu_by_group.get(group, {})
        rows.append(
            {
                "source": "p10_llm_fraction_scaling",
                "quota_variant": group,
                "n_society": row.get("n_society", ""),
                "n_seeds": row.get("n_seeds", ""),
                "alpha_c_linear": row.get("alpha_c_linear", ""),
                "alpha_c_logistic": row.get("alpha_c_logistic", ""),
                "K_c_effective": row.get("K_c_effective", ""),
                "nu_linear_by_quota": nu_row.get("nu_linear", ""),
                "nu_logistic_by_quota": nu_row.get("nu_logistic", ""),
                "n_resolved_linear": nu_row.get("n_resolved_linear", ""),
            }
        )
    return rows


def copy_audit_table(path: Path, source: str) -> list[dict]:
    rows = []
    for row in read_csv(path):
        row = dict(row)
        row["source"] = source
        rows.append(row)
    return rows


def cross_model_summary() -> tuple[list[dict], list[dict]]:
    curve_rows = []
    summary_rows = []
    for tag, path in CROSS_MODELS.items():
        raw_rows = read_jsonl(path)
        if not raw_rows:
            continue
        # Keep the last row for repeated runs of the same (N, alpha, seed).
        latest = {}
        for row in raw_rows:
            key = (as_int(row.get("n_society")), as_float(row.get("alpha")), as_int(row.get("seed")))
            latest[key] = row
        grouped: dict[tuple[int, float], list[dict]] = defaultdict(list)
        for (n, alpha, _seed), row in latest.items():
            if row.get("status") == "ok":
                grouped[(n, alpha)].append(row)
        alpha_cs = []
        model_id = ""
        for (n, alpha), rows in sorted(grouped.items()):
            values = [as_float(row.get("primary_failure_rate"), 0.0) for row in rows]
            model_ids = sorted({str(row.get("population_model", "")) for row in rows})
            if model_ids:
                model_id = model_ids[0]
            curve_rows.append(
                {
                    "model_tag": tag,
                    "model_id": ";".join(model_ids),
                    "n_society": n,
                    "alpha": fmt(alpha, 6),
                    "n_seeds": len(rows),
                    "primary_failure_count": sum(1 for value in values if value >= 0.5),
                    "primary_failure_rate_mean": fmt(mean(values), 4),
                }
            )
        for n in sorted({key[0] for key in grouped}):
            points = []
            for (nn, alpha), rows in grouped.items():
                if nn == n:
                    points.append((alpha, mean(as_float(row.get("primary_failure_rate"), 0.0) for row in rows)))
            alpha_c, bracket = crossing(points)
            if alpha_c is not None:
                alpha_cs.append((n, alpha_c))
            summary_rows.append(
                {
                    "model_tag": tag,
                    "model_id": model_id,
                    "n_society": n,
                    "alpha_c": fmt(alpha_c, 6),
                    "effective_k_c": fmt(alpha_c * n if alpha_c is not None else math.nan, 6),
                    "crossing_bracket_alpha": "" if bracket is None else f"{fmt(bracket[0], 6)}-{fmt(bracket[1], 6)}",
                    "coverage_status": "crosses_0.5" if alpha_c is not None else "unresolved",
                }
            )
        slope, _ = fit_log_slope([n for n, _ in alpha_cs], [alpha for _, alpha in alpha_cs])
        summary_rows.append(
            {
                "model_tag": tag,
                "model_id": model_id,
                "n_society": "ALL_RESOLVED",
                "alpha_c": "",
                "effective_k_c": "",
                "crossing_bracket_alpha": "",
                "coverage_status": f"n_resolved={len(alpha_cs)}; two_or_multi_point_nu={fmt(-slope if slope is not None else None, 4)}",
            }
        )
    return curve_rows, summary_rows


def provenance_rows() -> list[dict]:
    specs = [
        ("S1 grid/count/boundary audit", P01_OUTPUTS, f"latest v3 raw P01 rows from glob {P01_DATA_GLOB}"),
        ("Held-out closure audit", P01_OUTPUTS, f"v3 P02 decomposition rows from glob {P02_DATA_GLOB}"),
        ("Boundary-support closure audit", P01_OUTPUTS, f"v3 P02 boundary-support rows from glob {P02_BOUNDARY_DATA_GLOB}"),
        ("Pilot grid comparison", P01_OUTPUTS, f"pilot rows from glob {P01_PILOT_GLOB}"),
        ("Bootstrap censoring summary", BOOTSTRAP_JSON, "generated v3 bootstrap JSON"),
        ("Main scaling alpha_c table", SCALING_RESULTS, "generated v3 scaling table"),
        ("Depth scaling audit", P09_DEPTH_NU, "v3 p-series reviewer audit"),
        ("LLM quota scaling audit", P10_LLM_NU, "v3 p-series reviewer audit"),
        ("Watts null audit", P11_WATTS_NU, "v3 p-series reviewer audit"),
    ]
    for tag, path in CROSS_MODELS.items():
        specs.append((f"Cross-model pilot {tag}", path, "newly run cross-model rows"))
    rows = []
    for claim, path, note in specs:
        rows.append(
            {
                "claim_or_table": claim,
                "path": str(path.relative_to(ROOT)),
                "exists": "yes" if path.exists() else "no",
                "note": note,
            }
        )
    return rows


def pilot_grid_comparison(main_rows: list[dict], pilot_rows: list[dict]) -> list[dict]:
    def grid_map(rows: list[dict]) -> dict[int, set[float]]:
        out: dict[int, set[float]] = defaultdict(set)
        for row in rows:
            if row.get("status") == "ok":
                out[as_int(row.get("n_society"))].add(as_float(row.get("alpha")))
        return out

    main_grid = grid_map(main_rows)
    pilot_grid = grid_map(pilot_rows)
    all_n = sorted(set(main_grid) | set(pilot_grid))
    rows = []
    for n in all_n:
        main_alphas = sorted(main_grid.get(n, set()))
        pilot_alphas = sorted(pilot_grid.get(n, set()))
        rows.append(
            {
                "n_society": n,
                "pilot_alpha_grid": ";".join(fmt(alpha, 6) for alpha in pilot_alphas),
                "main_alpha_grid": ";".join(fmt(alpha, 6) for alpha in main_alphas),
                "pilot_only": ";".join(fmt(alpha, 6) for alpha in sorted(set(pilot_alphas) - set(main_alphas))),
                "main_only": ";".join(fmt(alpha, 6) for alpha in sorted(set(main_alphas) - set(pilot_alphas))),
            }
        )
    return rows


def write_readme(paths: dict[str, Path], notes: list[str]) -> None:
    lines = ["# Reinforcement Experiment Tables", ""]
    lines.append("Generated from existing v3 artifacts; no new simulator episodes are run by this script.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for name, path in paths.items():
        lines.append(f"- `{path.relative_to(ROOT)}`: {name}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_paths = sorted(P01_OUTPUTS.glob(P01_DATA_GLOB))
    pilot_paths = sorted(P01_OUTPUTS.glob(P01_PILOT_GLOB))
    p02_paths = sorted(P01_OUTPUTS.glob(P02_DATA_GLOB))
    p02_boundary_paths = sorted(P01_OUTPUTS.glob(P02_BOUNDARY_DATA_GLOB))
    main_rows = read_csv_many(main_paths)
    pilot_rows = read_csv_many(pilot_paths)
    p02_rows = read_csv_many(p02_paths)
    p02_boundary_rows = read_csv_many(p02_boundary_paths)
    p02_boundary_real_rows = [row for row in p02_boundary_rows if as_int(row.get("mock_openrouter"), 0) == 0]
    published = load_published_alpha_c()

    point_rows, boundary_rows = summarize_grid_counts(main_rows, published)
    required_boundary_ns = {
        as_int(row.get("n_society"))
        for row in boundary_rows
        if as_float(row.get("alpha_c_linear")) > 0
    }
    p02_boundary_ns = {as_int(row.get("n_society")) for row in p02_boundary_real_rows}
    use_boundary_support = bool(required_boundary_ns) and required_boundary_ns.issubset(p02_boundary_ns)
    closure_rows = p02_boundary_real_rows if use_boundary_support else p02_rows
    closure_source = "p02_boundary_support_decomposition" if use_boundary_support else "p02_size_decomposition"
    threshold_rows = threshold_and_component_sensitivity(main_rows)
    drop_rows = drop_n_sensitivity(boundary_rows)
    closure_detail_rows, closure_summary_rows = heldout_closure_test(closure_rows, boundary_rows, closure_source)
    bootstrap_rows = bootstrap_summary()
    llm_rows = llm_ablation_summary()
    cross_curve_rows, cross_summary_rows = cross_model_summary()
    pilot_rows_out = pilot_grid_comparison(main_rows, pilot_rows)
    provenance = provenance_rows()
    depth_nu_rows = copy_audit_table(P09_DEPTH_NU, "p09_depth_scaling")
    depth_alpha_rows = copy_audit_table(P09_DEPTH_ALPHA_C, "p09_depth_scaling")
    watts_nu_rows = copy_audit_table(P11_WATTS_NU, "p11_watts_null")
    watts_alpha_rows = copy_audit_table(P11_WATTS_ALPHA_C, "p11_watts_null")

    outputs = {
        "Per-(N, alpha) seed counts and primary failure counts": OUT / "p01_grid_counts.csv",
        "Per-N boundary brackets and integer-K granularity": OUT / "p01_boundary_integer_k_audit.csv",
        "S1 gate threshold and component outcome sensitivity": OUT / "p01_threshold_component_sensitivity.csv",
        "Drop-small-N exponent sensitivity": OUT / "p01_drop_n_sensitivity.csv",
        "P02 held-out closure predictions": OUT / "p02_heldout_closure_predictions.csv",
        "P02 held-out closure summary": OUT / "p02_heldout_closure_summary.csv",
        "Bootstrap censoring summary": OUT / "p01_bootstrap_censoring_summary.csv",
        "P10 LLM quota scaling audit": OUT / "p10_llm_quota_scaling_summary.csv",
        "P09 depth scaling exponent audit": OUT / "p09_depth_nu.csv",
        "P09 depth alpha_c audit": OUT / "p09_depth_alpha_c.csv",
        "P11 Watts null exponent audit": OUT / "p11_watts_null_nu.csv",
        "P11 Watts null alpha_c audit": OUT / "p11_watts_null_alpha_c.csv",
        "Cross-model pilot curves": OUT / "cross_model_curves.csv",
        "Cross-model pilot alpha_c summary": OUT / "cross_model_summary.csv",
        "Pilot-vs-main grid comparison": OUT / "pilot_vs_main_grid_comparison.csv",
        "Evidence provenance table": OUT / "provenance.csv",
    }

    write_csv(outputs["Per-(N, alpha) seed counts and primary failure counts"], point_rows)
    write_csv(outputs["Per-N boundary brackets and integer-K granularity"], boundary_rows)
    write_csv(outputs["S1 gate threshold and component outcome sensitivity"], threshold_rows)
    write_csv(outputs["Drop-small-N exponent sensitivity"], drop_rows)
    write_csv(outputs["P02 held-out closure predictions"], closure_detail_rows)
    write_csv(outputs["P02 held-out closure summary"], closure_summary_rows)
    write_csv(outputs["Bootstrap censoring summary"], bootstrap_rows)
    write_csv(outputs["P10 LLM quota scaling audit"], llm_rows)
    write_csv(outputs["P09 depth scaling exponent audit"], depth_nu_rows)
    write_csv(outputs["P09 depth alpha_c audit"], depth_alpha_rows)
    write_csv(outputs["P11 Watts null exponent audit"], watts_nu_rows)
    write_csv(outputs["P11 Watts null alpha_c audit"], watts_alpha_rows)
    write_csv(outputs["Cross-model pilot curves"], cross_curve_rows)
    write_csv(outputs["Cross-model pilot alpha_c summary"], cross_summary_rows)
    write_csv(outputs["Pilot-vs-main grid comparison"], pilot_rows_out)
    write_csv(outputs["Evidence provenance table"], provenance)

    write_markdown_table(
        OUT / "p01_boundary_integer_k_audit.md",
        "P01 Boundary and Integer-K Audit",
        boundary_rows,
        [
            "n_society",
            "n_alpha_points",
            "alpha_c_linear",
            "effective_k_c",
            "published_alpha_c",
            "raw_minus_published_alpha_c",
            "crossing_bracket_alpha",
            "crossing_bracket_k",
            "coverage_status",
        ],
    )
    write_markdown_table(
        OUT / "p01_drop_n_sensitivity.md",
        "Drop-small-N Scaling Sensitivity",
        drop_rows,
        ["fit", "n_points", "n_values", "nu"],
    )
    write_markdown_table(
        OUT / "p02_heldout_closure_summary.md",
        "P02 Held-out Closure Summary",
        closure_summary_rows,
        [
            "method",
            "closure_data_source",
            "n_holdouts",
            "mean_abs_log_alpha_error",
            "median_abs_log_alpha_error",
            "skill_vs_loo_constant",
            "beats_loo_constant",
            "within_factor_1p5",
            "within_factor_2",
            "observed_nu",
            "predicted_nu",
            "spearman_observed_predicted_alpha",
        ],
    )
    write_markdown_table(
        OUT / "cross_model_summary.md",
        "Cross-model Pilot Summary",
        cross_summary_rows,
        ["model_tag", "model_id", "n_society", "alpha_c", "effective_k_c", "coverage_status"],
    )
    write_markdown_table(
        OUT / "p10_llm_quota_scaling_summary.md",
        "P10 LLM Quota Scaling Summary",
        llm_rows,
        [
            "source",
            "quota_variant",
            "n_society",
            "n_seeds",
            "alpha_c_linear",
            "alpha_c_logistic",
            "K_c_effective",
            "nu_linear_by_quota",
            "n_resolved_linear",
        ],
    )

    notes = [
        "Threshold sensitivity is recomputed from raw v3 P01 components using the S1 logical gate at scales 0.8, 0.9, 1.0, 1.1, and 1.2.",
        "Component outcomes are diagnostic and do not use the composite S1 social-and-market gate.",
        "Held-out closure uses P02 baseline_q05 gain rows, leaves out each N, and predicts the matching P01 alpha_c; if boundary-support P02b rows exist, they are used instead of the original narrow K grid.",
        "Positive closure is pre-specified to require beating the leave-one-N-out constant-geomean alpha_c baseline on mean absolute log error.",
        "Cross-model summaries are pilot checks using the currently available rows in paper_experiments_v3/outputs/p01_crossmodel_*.",
        "LLM allocation robustness uses v3 p-series P10 reviewer audit artifacts, not legacy archive/e-series outputs.",
        "P07 smoke is intentionally not used as main evidence because it is N=40, one seed, and mock OpenRouter.",
    ]
    write_readme(outputs, notes)

    print(f"Wrote {len(outputs)} CSV outputs and 5 markdown summaries to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
