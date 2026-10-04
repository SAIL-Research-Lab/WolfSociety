"""Regenerate main and appendix tables from NEW complete episode records only.

Every output is tagged simulated/real. Missing or censored results remain
missing; this module never fills them with numbers from the hybrid manuscript.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .runner import atomic_json, digest, load_complete, validate_manifest


def read_runs(paths, allow_mock=False):
    rows, manifests, coverage = [], [], []
    seen = set()
    for path in map(Path, paths):
        manifest = json.loads((path / "manifest.json").read_text())
        validate_manifest(manifest, check_source=False)
        if manifest["backend"]["kind"] == "mock" and not allow_mock:
            raise ValueError("Mock data are not scientific results. Pass --allow-mock only for pipeline QA")
        n_complete = 0
        for job in manifest["jobs"]:
            row = load_complete(path / "episodes" / f"{job['job_id']}.json", job, manifest)
            if row is None:
                continue
            n_complete += 1
            if row["job_id"] in seen:
                continue
            seen.add(row["job_id"])
            row["run_dir"] = str(path)
            row["backend"] = manifest["backend"]
            row["stage"] = manifest["stage"]
            rows.append(row)
        manifests.append(manifest)
        coverage.append({"run_dir": str(path), "expected": len(manifest["jobs"]), "complete": n_complete,
                         "missing": len(manifest["jobs"]) - n_complete})
    if len({m["source_hash"] for m in manifests}) > 1:
        raise ValueError("Cannot pool different source/prompt versions; analyze each protocol separately")
    settings_by_model = defaultdict(set)
    for m in manifests:
        b = m["backend"]
        settings_by_model[(b.get("model"), b.get("model_revision"))].add(digest(b))
    if any(len(values) > 1 for values in settings_by_model.values()):
        raise ValueError("Same model has different backend settings; analyze protocols separately rather than pooling them")
    return rows, manifests, coverage


def metric(row, name):
    value = row["result"].get(name, row["result"].get("information_metrics", {}).get(name))
    if isinstance(value, (int, float, bool)) and math.isfinite(float(value)):
        return float(value)
    return None


def group_key(row, include_size=True):
    c = row["config"]
    values = [row["family"], row["variant"], c["scenario"], row["backend"]["model"], row["backend"].get("model_revision", ""), row["stage"], row["simulated"]]
    if include_size:
        values.append(c["n_society"])
    return tuple(values)


GROUP_COLUMNS = ["family", "variant", "scenario", "model", "model_revision", "stage", "simulated"]


def labels(key):
    return dict(zip(GROUP_COLUMNS + (["n_society"] if len(key) == 8 else []), key))


def groups(rows, include_size=True):
    out = defaultdict(list)
    for row in rows:
        out[group_key(row, include_size)].append(row)
    return out


def primary(row):
    for name in ("primary_failure_rate", "collapse_rate", "collapse"):
        value = metric(row, name)
        if value is not None:
            return value
    raise ValueError(f"Missing binary failure indicator: {row['job_id']}")


def curve(rows, alpha_mode="target", seed_weights=None, event_fn=primary):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        c = row["config"]
        alpha = c["alpha"] if alpha_mode == "target" else metric(row, "alpha_realized")
        if alpha is None:
            alpha = c.get("harmful_count", round(c["alpha"] * c["n_society"])) / c["n_society"]
        event = event_fn(row)
        if event is not None:
            grouped[float(alpha)][c["seed"]].append(float(event))
    points = []
    for alpha, seed_values in sorted(grouped.items()):
        # Integer rounding may make different target alphas share one realized
        # K/N. They do not create additional independent seeds.
        values = [(float(np.mean(events)), 1 if seed_weights is None else seed_weights.get(seed, 0))
                  for seed, events in seed_values.items()]
        total = sum(weight for _, weight in values)
        if total:
            points.append((alpha, sum(value * weight for value, weight in values) / total, total))
    return points


def crossing(points, probability=.5):
    if not points:
        return {"status": "missing", "alpha_c": None}
    if points[0][0] != 0:
        return {"status": "missing_clean_baseline", "alpha_c": None}
    if points[0][1] >= probability:
        return {"status": "left_censored", "alpha_c": None, "bound": points[0][0]}
    upward = [(a, b) for a, b in zip(points, points[1:]) if a[1] < probability <= b[1]]
    downward = sum(a[1] >= probability > b[1] for a, b in zip(points, points[1:]))
    if not upward:
        return {"status": "right_censored", "alpha_c": None, "bound": points[-1][0]}
    if len(upward) != 1 or downward:
        return {"status": "multiple_crossings", "alpha_c": None, "upcrossings": len(upward), "downcrossings": downward}
    left, right = upward[0]
    value = left[0] + (probability - left[1]) * (right[0] - left[0]) / (right[1] - left[1])
    return {"status": "resolved", "alpha_c": value, "bracket_low": left[0], "bracket_high": right[0],
            "nonmonotone_steps": sum(b[1] < a[1] for a, b in zip(points, points[1:]))}


def logistic_midpoint(points):
    if crossing(points)["status"] != "resolved":
        return None
    x = np.array([p[0] for p in points])
    scale = max(float(x.max()), 1e-9)
    y, counts = np.array([p[1] for p in points]), np.array([p[2] for p in points])
    def objective(theta):
        z = theta[0] + math.exp(theta[1]) * x / scale
        return float(np.sum(counts * (np.logaddexp(0, z) - y * z)) + 1e-6 * np.dot(theta, theta))
    fit = minimize(objective, [-2., 1.5], method="L-BFGS-B", bounds=[(-50, 50), (-10, 10)])
    if not fit.success:
        return None
    midpoint = -fit.x[0] / math.exp(fit.x[1]) * scale
    return float(midpoint) if x.min() <= midpoint <= x.max() else None


def boundary_rows(rows):
    out = []
    for key, members in groups(rows).items():
        if key[0] in {"p02", "p02b"}:
            continue
        for mode in ("target", "realized"):
            points = curve(members, mode)
            midpoint = crossing(points)
            low, high = crossing(points, .1), crossing(points, .9)
            n = key[-1]
            out.append({**labels(key), "alpha_mode": mode, **midpoint,
                        "k_c": midpoint["alpha_c"] * n if midpoint["alpha_c"] is not None else None,
                        "logistic_alpha_c": logistic_midpoint(points),
                        "transition_width_10_90": high["alpha_c"] - low["alpha_c"] if low["alpha_c"] is not None and high["alpha_c"] is not None else None,
                        "n_seeds_min": min((p[2] for p in points), default=0),
                        "clean_failure_rate": points[0][1] if points and points[0][0] == 0 else None})
    return out


def fit_exponent(midpoints):
    valid = [(n, value) for n, value in midpoints.items() if value is not None and value > 0]
    if len(valid) < 3:
        return None
    x, y = np.log(np.array(valid, dtype=float)).T
    slope, intercept = np.polyfit(x, y, 1)
    return {"nu": -float(slope), "log_intercept": float(intercept), "n_sizes": len(valid),
            "sizes": [n for n, _ in valid]}


def bootstrap_scaling(rows, draws=1000, seed=271828):
    rng = np.random.default_rng(seed)
    output, jackknife = [], []
    for key, members in groups(rows, False).items():
        by_n = defaultdict(list)
        for row in members:
            by_n[row["config"]["n_society"]].append(row)
        if len(by_n) < 3 or key[0] in {"p02", "p02b"}:
            continue
        # Paired bootstrap uses only seeds present at every observed grid cell.
        cell_seeds = defaultdict(set)
        for row in members:
            c = row["config"]
            cell_seeds[(c["n_society"], c["alpha"])].add(c["seed"])
        common = sorted(set.intersection(*cell_seeds.values()))
        for mode in ("target", "realized"):
            paired = {n: [r for r in values if r["config"]["seed"] in common] for n, values in by_n.items()}
            mids = {n: crossing(curve(values, mode))["alpha_c"] for n, values in paired.items()}
            estimate = fit_exponent(mids)
            if estimate is None:
                output.append({**labels(key), "alpha_mode": mode, "status": "fewer_than_three_resolved_sizes", "paired_seeds": len(common)})
                continue
            strict, resolved = [], []
            for _ in range(draws):
                weights = defaultdict(int)
                for s in rng.choice(common, size=len(common), replace=True):
                    weights[int(s)] += 1
                sampled = {n: crossing(curve(values, mode, weights))["alpha_c"] for n, values in paired.items()}
                fit = fit_exponent(sampled)
                if fit is not None:
                    resolved.append(fit["nu"])
                    if len(fit["sizes"]) == len(by_n):
                        strict.append(fit["nu"])
            for method, values in [("all_resolved_at_least_three", resolved), ("strict_all_requested_sizes", strict)]:
                interval = np.percentile(values, [2.5, 97.5]).tolist() if values else [None, None]
                output.append({**labels(key), "alpha_mode": mode, "bootstrap_method": method,
                               "status": "resolved" if values else "no_valid_bootstrap_draws", **estimate,
                               "paired_seeds": len(common), "requested_sizes": sorted(by_n),
                               "ci_low": interval[0], "ci_high": interval[1], "valid_draws": len(values), "requested_draws": draws})
            for dropped in sorted(by_n):
                fit = fit_exponent({n: a for n, a in mids.items() if n != dropped})
                jackknife.append({**labels(key), "alpha_mode": mode, "omitted_n": dropped,
                                  **(fit or {"nu": None, "status": "insufficient_sizes"})})
    return output, jackknife


def summary_rows(rows):
    bucket = defaultdict(list)
    for row in rows:
        bucket[(group_key(row), row["config"]["alpha"])].append(row)
    output = []
    for (key, alpha), members in bucket.items():
        values = set()
        for row in members:
            values.update(row["result"])
            values.update(row["result"].get("information_metrics", {}))
        record = {**labels(key), "alpha": alpha, "n_episodes": len(members),
                  "primary_failure_mean": float(np.mean([primary(r) for r in members]))}
        for name in sorted(values):
            numbers = [v for r in members if (v := metric(r, name)) is not None]
            if numbers:
                record[f"{name}_mean"] = float(np.mean(numbers))
                record[f"{name}_se"] = float(np.std(numbers, ddof=1) / np.sqrt(len(numbers))) if len(numbers) > 1 else None
        output.append(record)
    return output


def read_trace(row, name):
    record = row.get("traces", {}).get(name)
    if not record:
        return []
    with gzip.open(Path(row["run_dir"]) / record["path"], "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sensitivity_rows(rows):
    """Uniform threshold scaling uses stored eligible-day joint scores.

    Independent component sensitivity uses actual component/threshold pairs;
    no metric-name guesses are substituted for unavailable observations.
    """
    derived, diagnostics = [], []
    for row in rows:
        if row["family"] != "p01":
            continue
        daily = read_trace(row, "daily_log")
        normalized = defaultdict(list)
        for day in daily:
            failure = day.get("primary_failure", {})
            if day.get("day", 0) < failure.get("evaluation_start_day", 0):
                continue
            score = failure.get("primary_failure_score")
            if isinstance(score, (float, int)):
                normalized["joint"].append(float(score))
            for name, value in failure.get("components", {}).items():
                threshold = failure.get("thresholds", {}).get(name)
                if isinstance(value, (int, float)) and isinstance(threshold, (int, float)) and threshold > 0:
                    normalized[name].append(float(value) / threshold)
            components, thresholds = failure.get("components", {}), failure.get("thresholds", {})
            if all(isinstance(components.get(k), (int, float)) and thresholds.get(k, 0) > 0 for k in ("price_dislocation", "liquidity_stress")):
                normalized["market_any"].append(max(components[k] / thresholds[k] for k in ("price_dislocation", "liquidity_stress")))
        if not normalized:
            score = metric(row, "primary_failure_score_max")
            if score is not None:
                normalized["joint"] = [score]
            diagnostics.append({"job_id": row["job_id"], "status": "daily_components_unavailable"})
        for component, scores in normalized.items():
            for scale in [.8, .9, 1., 1.1, 1.2]:
                copy = dict(row)
                copy["variant"] = f"{row['variant']}:{component}:threshold_{scale:g}"
                copy["result"] = {"primary_failure_rate": float(max(scores) >= scale), "alpha_realized": metric(row, "alpha_realized")}
                derived.append(copy)
    return boundary_rows(derived), diagnostics


def _regression(data):
    if len(data) < 3:
        return None
    x = np.array([[1., math.log(n), math.log(k)] for n, k, value, _ in data])
    y = np.log([value for _, _, value, _ in data])
    if np.linalg.matrix_rank(x) < 3:
        return None
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return {"intercept": float(beta[0]), "b_n": float(beta[1]), "b_k": float(beta[2]),
            "implied_nu": float(1 + beta[1] / beta[2]) if abs(beta[2]) > 1e-8 else None,
            "n_rows": len(data)}


def fixed_count_tables(rows, draws=1000):
    output, per_k, heldout = [], [], []
    rng = np.random.default_rng(314159)
    for key, members in groups([r for r in rows if r["family"] in {"p02", "p02b"}], False).items():
        for outcome in ["primary_failure_score_max", "social_cascade_peak", "price_dislocation_max", "liquidity_stress_max"]:
            cell_p = defaultdict(list)
            clean = {}
            for row in members:
                c = row["config"]
                k = c.get("harmful_count", round(c["alpha"] * c["n_society"]))
                cell_p[(c["n_society"], k)].append(primary(row))
                if k == 0:
                    clean[(c["n_society"], c["seed"])] = metric(row, outcome)
            for subset in ["full", "subcritical", "zero_failure"]:
                data = []
                for row in members:
                    c = row["config"]
                    n, k, s = c["n_society"], c.get("harmful_count", round(c["alpha"] * c["n_society"])), c["seed"]
                    value, baseline = metric(row, outcome), clean.get((n, s))
                    rate = np.mean(cell_p[(n, k)])
                    if subset == "subcritical" and rate >= .5 or subset == "zero_failure" and rate != 0:
                        continue
                    if value is not None and baseline is not None and k > 0 and value > baseline:
                        data.append((n, k, value - baseline, s))
                fit = _regression(data)
                record = {**labels(key), "outcome": outcome, "response": "positive_paired_clean_subtracted", "subset": subset,
                          "status": "resolved" if fit else "insufficient_positive_full_rank_data", **(fit or {})}
                if fit:
                    seeds = sorted({d[3] for d in data})
                    samples = []
                    for _ in range(draws):
                        resampled = [item for seed in rng.choice(seeds, len(seeds), replace=True) for item in data if item[3] == seed]
                        fitted = _regression(resampled)
                        if fitted:
                            samples.append([fitted["b_n"], fitted["b_k"]])
                    if samples:
                        ci = np.percentile(samples, [2.5, 97.5], axis=0)
                        record.update(b_n_ci_low=float(ci[0, 0]), b_n_ci_high=float(ci[1, 0]), b_k_ci_low=float(ci[0, 1]), b_k_ci_high=float(ci[1, 1]), valid_bootstraps=len(samples))
                output.append(record)
                if subset != "full":
                    continue
                for k in sorted({item[1] for item in data}):
                    matches = [item for item in data if item[1] == k]
                    sizes = {item[0] for item in matches}
                    slope = float(np.polyfit(np.log([d[0] for d in matches]), np.log([d[2] for d in matches]), 1)[0]) if len(sizes) >= 2 else None
                    per_k.append({**labels(key), "outcome": outcome, "harmful_count": k, "size_log_slope": slope, "n_sizes": len(sizes)})
                if outcome != "primary_failure_score_max":
                    continue
                # No midpoint used to train this continuous-score response model.
                for omitted in sorted({item[0] for item in data}):
                    fitted = _regression([item for item in data if item[0] != omitted])
                    predicted = None
                    if fitted and fitted["b_k"] > 1e-8:
                        exponent = -(fitted["intercept"] + fitted["b_n"] * math.log(omitted)) / fitted["b_k"]
                        if -700 < exponent < 700:
                            predicted = math.exp(exponent) / omitted
                    # Clean subtraction changes the target: retain explicit score-1
                    # interpretation rather than claiming a calibrated failure midpoint.
                    heldout.append({**labels(key), "omitted_n": omitted, "predicted_alpha_score_increment_1": predicted,
                                    **({"response_intercept": fitted["intercept"], "response_b_n": fitted["b_n"], "response_b_k": fitted["b_k"]} if fitted else {}),
                                    "status": "prediction" if predicted else "unresolved", "calibration": "none; score-increment boundary is not a p=.5 midpoint"})
    return output, per_k, heldout


def heldout_boundary_comparisons(boundaries, predictions):
    output = []
    primary_groups = defaultdict(dict)
    for row in boundaries:
        if row["family"] == "p01" and row["alpha_mode"] == "target" and row["alpha_c"] is not None:
            primary_groups[(row["model"], row["model_revision"], row["stage"], row["simulated"])][row["n_society"]] = row["alpha_c"]
    for pred in predictions:
        n = pred["omitted_n"]
        mids = primary_groups.get((pred["model"], pred["model_revision"], pred["stage"], pred["simulated"]), {})
        training = {size: alpha for size, alpha in mids.items() if size != n}
        observed = mids.get(n)
        constant = float(np.exp(np.mean(np.log(list(training.values()))))) if training else None
        power_fit = fit_exponent(training)
        power = math.exp(power_fit["log_intercept"]) * n ** -power_fit["nu"] if power_fit else None
        calibration_ratios = []
        if pred.get("response_b_k", 0) > 1e-8:
            for size, alpha in training.items():
                exponent = -(pred["response_intercept"] + pred["response_b_n"] * math.log(size)) / pred["response_b_k"]
                if -700 < exponent < 700:
                    predicted_training = math.exp(exponent) / size
                    if predicted_training > 0:
                        calibration_ratios.append(math.log(alpha / predicted_training))
        raw = pred["predicted_alpha_score_increment_1"]
        calibrated = raw * math.exp(float(np.mean(calibration_ratios))) if raw and calibration_ratios else None
        for method, prediction in [("continuous_response_uncalibrated", raw), ("continuous_response_calibrated_on_training_sizes_only", calibrated), ("heldout_constant_geomean", constant), ("heldout_power_law", power)]:
            output.append({**pred, "prediction_method": method, "observed_alpha_c": observed, "predicted_alpha": prediction,
                           "absolute_log_error": abs(math.log(prediction / observed)) if prediction and observed and prediction > 0 else None})
    return output


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def analyze(paths, output, *, allow_mock=False, draws=1000):
    rows, manifests, coverage = read_runs(paths, allow_mock)
    if not rows:
        raise ValueError("No validated complete episode records")
    if len({r["simulated"] for r in rows}) > 1 or len({r["stage"] for r in rows}) > 1:
        raise ValueError("Analyze mock/real and pilot/main separately")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    boundaries = boundary_rows(rows)
    scaling, jackknife = bootstrap_scaling(rows, draws)
    sensitivity, diagnostics = sensitivity_rows(rows)
    fixed, per_k, response = fixed_count_tables(rows, draws)
    tables = {"episode_summary": summary_rows(rows), "boundaries": boundaries,
              "scaling_bootstrap": scaling, "drop_one_size": jackknife,
              "threshold_components": sensitivity, "fixed_count_response": fixed,
              "fixed_count_per_k": per_k, "heldout_response": response,
              "heldout_comparisons": heldout_boundary_comparisons(boundaries, response),
              "call_token_audit": [{**labels(group_key(r)), "job_id": r["job_id"], "seed": r["config"]["seed"], "alpha": r["config"]["alpha"],
                                    **{name: r["result"].get(name) for name in ("n_llm", "n_benign_llm", "n_harmful_llm", "n_llm_decisions", "n_population_decisions", "rule_fallback_count")},
                                    **r["result"].get("backend_usage", {}), **r["result"].get("audit", {})} for r in rows]}
    for name, table in tables.items():
        write_csv(output / f"{name}.csv", table)
    report = {"status": "SIMULATED_PIPELINE_QA_NOT_SCIENTIFIC" if rows[0]["simulated"] else "observed_results",
              "stage": rows[0]["stage"], "source_manifests": [m["manifest_hash"] for m in manifests],
              "coverage": coverage, "complete_dataset": all(c["missing"] == 0 for c in coverage),
              "bootstrap_draws": draws, "independent_unit": "episode seed; paired resampling across N and alpha",
              "censoring": "No extrapolated alpha_c; multiple crossings and missing baseline remain unresolved",
              "diagnostics": diagnostics, "table_rows": {name: len(value) for name, value in tables.items()},
              "limitations": ["Changed full-LLM controllers require new results and revised manuscript statements.",
                              "Response-surface score-increment predictions are exploratory; not independently validated midpoint theory.",
                              "Role and information estimates depend on available simulator trace diagnostics; unavailable values are not backfilled."]}
    atomic_json(output / "analysis_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive")
    print(json.dumps(analyze(args.runs, args.output, allow_mock=args.allow_mock, draws=args.bootstrap_draws), indent=2))


if __name__ == "__main__":
    main()
