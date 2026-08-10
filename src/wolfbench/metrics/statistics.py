"""Statistical helpers for high-confidence WolfBench analyses."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class _TauResult:
    statistic: float


class _NormFallback:
    @staticmethod
    def ppf(value: float) -> float:
        return float(NormalDist().inv_cdf(float(value)))


def _kendalltau_fallback(a: Sequence[float], b: Sequence[float]) -> _TauResult:
    concordant = 0
    discordant = 0
    n = min(len(a), len(b))
    for i in range(n):
        for j in range(i + 1, n):
            da = float(a[i]) - float(a[j])
            db = float(b[i]) - float(b[j])
            prod = da * db
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = concordant + discordant
    value = 0.0 if denom == 0 else (concordant - discordant) / denom
    return _TauResult(float(value))


kendalltau = _kendalltau_fallback
norm = _NormFallback()


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    alpha = (1.0 - confidence) / 2.0
    z = float(norm.ppf(1.0 - alpha))
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    radius = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def binomial_rate_summary(values: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    """Summarize binary collapse outcomes with Wilson uncertainty."""
    arr = np.array([float(v) for v in values], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "n": 0, "successes": 0, "ci_low": 0.0, "ci_high": 0.0}
    successes = int(np.rint(arr.sum()))
    low, high = wilson_interval(successes, int(arr.size), confidence=confidence)
    return {
        "mean": float(arr.mean()),
        "n": int(arr.size),
        "successes": successes,
        "ci_low": low,
        "ci_high": high,
    }


def top_k_overlap(reference: Sequence[str], candidate: Sequence[str], k: int) -> float:
    """Fractional overlap between the top-k items of two rankings."""
    if k <= 0:
        return 0.0
    ref = set(reference[:k])
    cand = set(candidate[:k])
    if not ref and not cand:
        return 1.0
    return float(len(ref & cand) / max(len(ref | cand), 1))


def rank_stability(
    rows: list[dict],
    score_key: str,
    item_key: str = "defense",
    sample_key: str = "seed",
    n_boot: int = 2000,
    top_k: int = 3,
    seed: int = 0,
) -> dict[str, float | list[str]]:
    """Bootstrap rank stability for leaderboard-like tables.

    The function resamples ``sample_key`` groups, recomputes mean scores per
    item, and compares each bootstrap ranking with the full-data ranking.
    """
    if not rows:
        return {
            "kendall_tau_mean": 0.0,
            "kendall_tau_ci_low": 0.0,
            "kendall_tau_ci_high": 0.0,
            "top_k_overlap_mean": 0.0,
            "reference_ranking": [],
            "n_boot": 0,
        }

    by_sample: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        by_sample[row.get(sample_key, len(by_sample))].append(row)
    sample_ids = list(by_sample)

    def mean_scores(selected_rows: list[dict]) -> dict[str, float]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for row in selected_rows:
            buckets[str(row[item_key])].append(float(row[score_key]))
        return {name: float(np.mean(vals)) for name, vals in buckets.items() if vals}

    full_scores = mean_scores(rows)
    reference = sorted(full_scores, key=lambda name: full_scores[name], reverse=True)
    reference_pos = {name: i for i, name in enumerate(reference)}
    rng = np.random.default_rng(seed)
    taus: list[float] = []
    overlaps: list[float] = []

    for _ in range(int(n_boot)):
        draw_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
        selected: list[dict] = []
        for sid in draw_ids:
            selected.extend(by_sample[sid])
        scores = mean_scores(selected)
        ranking = sorted(scores, key=lambda name: scores[name], reverse=True)
        shared = [name for name in reference if name in scores]
        if len(shared) >= 2:
            ref_rank = [reference_pos[name] for name in shared]
            cand_pos = {name: i for i, name in enumerate(ranking)}
            cand_rank = [cand_pos[name] for name in shared]
            tau = kendalltau(ref_rank, cand_rank).statistic
            if np.isfinite(tau):
                taus.append(float(tau))
        overlaps.append(top_k_overlap(reference, ranking, min(top_k, len(reference))))

    tau_arr = np.array(taus, dtype=float) if taus else np.array([0.0])
    overlap_arr = np.array(overlaps, dtype=float) if overlaps else np.array([0.0])
    return {
        "kendall_tau_mean": float(tau_arr.mean()),
        "kendall_tau_ci_low": float(np.quantile(tau_arr, 0.025)),
        "kendall_tau_ci_high": float(np.quantile(tau_arr, 0.975)),
        "top_k_overlap_mean": float(overlap_arr.mean()),
        "top_k_overlap_ci_low": float(np.quantile(overlap_arr, 0.025)),
        "top_k_overlap_ci_high": float(np.quantile(overlap_arr, 0.975)),
        "reference_ranking": reference,
        "n_boot": int(n_boot),
    }
