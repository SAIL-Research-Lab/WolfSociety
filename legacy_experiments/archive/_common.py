"""Shared helpers for WolfBench experiments.

Paper-facing scaling experiments write to ``paperoutputs/scaling/<exp_name>/``.
Paper-facing benchmark experiments write to ``paperoutputs/benchmark/<exp_name>/``.
The legacy ``outputs/`` tree is kept as an archive for older runs.
Each experiment directory contains at least:
* ``config.json``  -- experiment configuration
* ``data.csv``     -- raw per-episode metrics
* one or more ``*.png`` figures

Use :func:`run_grid` for a uniform episode sweep.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from wolfbench.agents.wolfguard import WolfGuardAgent, WolfGuardConfig
from wolfbench.env.environment import WolfBenchEnv
from wolfbench.scenarios.base import load_scenario
from wolfbench.tracks.runner import calibrate_clean_baseline


REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_OUTPUTS_ROOT = REPO_ROOT / "outputs"
OUTPUTS_ROOT = REPO_ROOT / "paperoutputs"
SCALING_THEORY_OUTPUTS_ROOT = OUTPUTS_ROOT / "scaling"
DEFENSE_BENCHMARK_OUTPUTS_ROOT = OUTPUTS_ROOT / "benchmark"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _record_run_event(directory: Path, event: str) -> None:
    now = _utc_now()
    metadata_path = directory / "run_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            metadata = {}
    metadata["output_dir"] = _rel_path(directory)
    metadata["last_event"] = event
    metadata["last_event_at_utc"] = now
    if event == "started":
        metadata["started_at_utc"] = now
        metadata["status"] = "running"
    elif event == "summary_written":
        metadata["summary_written_at_utc"] = now
        metadata["status"] = "summary_written"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    (directory / "last_run.txt").write_text(
        f"{metadata.get('status', event)} at {now}\n"
        f"output_dir={metadata['output_dir']}\n"
    )
    with open(directory / "run_history.jsonl", "a") as handle:
        handle.write(json.dumps({"event": event, "at_utc": now, "output_dir": metadata["output_dir"]}) + "\n")


def exp_dir(name: str, track: str | None = None) -> Path:
    root = OUTPUTS_ROOT if track is None else OUTPUTS_ROOT / track
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    _record_run_event(d, "started")
    return d


def scaling_exp_dir(name: str) -> Path:
    return exp_dir(name, "scaling")


def benchmark_exp_dir(name: str) -> Path:
    return exp_dir(name, "benchmark")


@dataclass
class RunSpec:
    scenario: str
    n_society: int
    alpha: float
    seed: int
    placement: str | None = None
    feedback_strength: float | None = None   # mutate scenario.social
    social_mean_degree: int | None = None
    retail_wealth_scale: float | None = None
    retail_risk_appetite: float | None = None
    asset_liquidity_scale: float | None = None
    llm_backend: Any = None
    n_llm_leaders: int = 0
    llm_provider: str = ""
    llm_model: str = ""
    defense: bool = False
    defense_mode: str = "full"
    defense_policy: Any = None                # WolfGuardPolicy override
    label: str = ""                          # tag for plotting


def run_episode(spec: RunSpec, baseline=None) -> dict[str, Any]:
    scen = load_scenario(spec.scenario)
    if spec.feedback_strength is not None:
        scen.social["feedback_strength"] = float(spec.feedback_strength)
    if spec.social_mean_degree is not None:
        scen.social["mean_degree"] = int(spec.social_mean_degree)
    if spec.retail_wealth_scale is not None:
        scen.retail["initial_wealth"] = float(scen.retail["initial_wealth"]) * float(spec.retail_wealth_scale)
    if spec.retail_risk_appetite is not None:
        scen.retail["risk_appetite"] = float(spec.retail_risk_appetite)
    if spec.asset_liquidity_scale is not None:
        for asset in scen.assets:
            asset.initial_liquidity *= float(spec.asset_liquidity_scale)

    wg = None
    if spec.defense_policy is not None:
        wg = spec.defense_policy
        if hasattr(wg, "fit_baseline") and baseline is not None:
            wg.fit_baseline(baseline)
    elif spec.defense:
        wg = WolfGuardAgent(config=WolfGuardConfig(mode=spec.defense_mode))

    env = WolfBenchEnv(
        scen, n_society=spec.n_society, alpha=spec.alpha, seed=spec.seed,
        wolfguard=wg, baseline=baseline,
        placement_override=spec.placement,
        llm_backend=spec.llm_backend,
        n_llm_leaders=spec.n_llm_leaders,
        expose_oracle=(spec.label == "oracle" or wg.__class__.__name__ == "OracleWolfGuardPolicy" if wg is not None else False),
    )
    res = env.run()
    m = res.metrics
    backend = getattr(wg, "backend", None) if wg is not None else None
    expected_asset_decisions = int(getattr(wg, "llm_expected_asset_decisions", 0)) if wg is not None else 0
    semantic_fallbacks = int(getattr(wg, "llm_semantic_fallbacks", 0)) if wg is not None else 0
    backend_calls = int(getattr(backend, "calls", 0)) if backend is not None else 0
    backend_failures = int(getattr(backend, "failures", 0)) if backend is not None else 0
    backend_recovered_failures = int(getattr(backend, "recovered_failures", 0)) if backend is not None else 0
    fallback_denominator = max(expected_asset_decisions, 1)
    return {
        "scenario": spec.scenario,
        "n_society": spec.n_society,
        "alpha": spec.alpha,
        "seed": spec.seed,
        "n_harmful": int(round(spec.alpha * spec.n_society)),
        "placement": spec.placement or "",
        "feedback_strength": (
            float(spec.feedback_strength) if spec.feedback_strength is not None else float("nan")
        ),
        "social_mean_degree": (
            int(spec.social_mean_degree) if spec.social_mean_degree is not None else -1
        ),
        "retail_wealth_scale": (
            float(spec.retail_wealth_scale) if spec.retail_wealth_scale is not None else float("nan")
        ),
        "retail_risk_appetite": (
            float(spec.retail_risk_appetite) if spec.retail_risk_appetite is not None else float("nan")
        ),
        "asset_liquidity_scale": (
            float(spec.asset_liquidity_scale) if spec.asset_liquidity_scale is not None else float("nan")
        ),
        "n_llm_leaders": int(spec.n_llm_leaders),
        "llm_provider": spec.llm_provider,
        "llm_model": spec.llm_model,
        "llm_backend": getattr(spec.llm_backend, "name", "") if spec.llm_backend is not None else "",
        "defense": int(spec.defense),
        "defense_mode": spec.defense_mode if spec.defense else "",
        "label": spec.label,
        "collapse_rate": m.collapse_rate,
        "collapse_day": m.collapse_day if m.collapse_day is not None else -1,
        "primary_metric": m.primary_metric,
        "primary_failure_rate": m.primary_failure_rate,
        "primary_failure_day": m.primary_failure_day if m.primary_failure_day is not None else -1,
        "primary_failure_score_max": m.primary_failure_score_max,
        "max_collapse_score": m.max_collapse_score,
        "retail_loss_pct_30d": m.retail_loss_pct_30d,
        "harmful_profit": m.harmful_profit,
        "wealth_transfer": m.wealth_transfer,
        "price_dislocation_max": m.price_dislocation_max,
        "liquidity_stress_max": m.liquidity_stress_max,
        "social_cascade_peak": m.social_cascade_peak,
        "intervention_cost": m.intervention_cost,
        "utility_loss": m.utility_loss,
        "false_positive_rate": m.false_positive_rate,
        "wash_share_max": m.wash_share_max,
        "volume_distortion_max": m.volume_distortion_max,
        "volume_signal_z_max": m.volume_signal_z_max,
        "cancel_rate_max": m.cancel_rate_max,
        "spoof_depth_to_liquidity_max": m.spoof_depth_to_liquidity_max,
        "withdrawal_loss_max": m.withdrawal_loss_max,
        "llm_backend_calls": backend_calls,
        "llm_backend_failures": backend_failures,
        "llm_backend_last_error_type": getattr(backend, "last_error_type", "") if backend is not None else "",
        "llm_backend_recovered_failures": backend_recovered_failures,
        "llm_backend_last_recovered_error_type": (
            getattr(backend, "last_recovered_error_type", "") if backend is not None else ""
        ),
        "llm_decision_calls": int(getattr(wg, "llm_decision_calls", 0)) if wg is not None else 0,
        "llm_expected_asset_decisions": expected_asset_decisions,
        "llm_valid_asset_decisions": int(getattr(wg, "llm_valid_asset_decisions", 0)) if wg is not None else 0,
        "llm_semantic_fallbacks": semantic_fallbacks,
        "llm_fallback_days": int(getattr(wg, "llm_fallback_days", 0)) if wg is not None else 0,
        "llm_semantic_fallback_rate": float(semantic_fallbacks / fallback_denominator),
    }


def run_grid(specs: Iterable[RunSpec], baseline=None,
             progress_every: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = list(specs)
    t0 = time.time()
    for i, s in enumerate(specs, 1):
        rows.append(run_episode(s, baseline=baseline))
        if i % progress_every == 0 or i == len(specs):
            dt = time.time() - t0
            print(f"[{i:>4}/{len(specs)}] {dt:6.1f}s elapsed", flush=True)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(obj: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    if path.name == "summary.json":
        _record_run_event(path.parent, "summary_written")


def aggregate(rows: list[dict[str, Any]], group_by: list[str], metric: str
              ) -> dict[tuple, dict[str, float]]:
    """Return mean/std/n per (group key tuple)."""
    buckets: dict[tuple, list[float]] = {}
    for r in rows:
        key = tuple(r[g] for g in group_by)
        buckets.setdefault(key, []).append(float(r[metric]))
    out: dict[tuple, dict[str, float]] = {}
    for k, vals in buckets.items():
        a = np.array(vals)
        out[k] = {"mean": float(a.mean()), "std": float(a.std()),
                  "n": int(a.size), "values": vals}
    return out


def env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


def env_float_list(name: str, default: str) -> list[float]:
    return [float(x) for x in env_list(name, default)]


def env_int_list(name: str, default: str) -> list[int]:
    return [int(x) for x in env_list(name, default)]


def env_seed_list(name: str, default_count: int, start: int = 1) -> list[int]:
    raw = os.getenv(name)
    if raw:
        return env_int_list(name, raw)
    return list(range(start, start + int(default_count)))


def alpha_critical(rows: list[dict[str, Any]], alphas: list[float],
                   n_society: int, threshold: float = 0.5) -> float | None:
    by_alpha = aggregate([r for r in rows if r["n_society"] == n_society],
                         ["alpha"], "collapse_rate")
    for a in sorted(alphas):
        m = by_alpha.get((a,), {}).get("mean", 0.0)
        if m >= threshold:
            return float(a)
    return None
