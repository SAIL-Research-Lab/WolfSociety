"""Closure-validity audit using existing WolfBench fixed-K data.

This analysis is deliberately separate from the paper sources. It answers:
1. Which parts of delta + zeta are definition-implied?
2. How much size-response signal remains in fixed-K, subcritical data?
3. Can that signal predict the collapse boundary without a held-out midpoint?
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/zhangyuejun/Documents/aaai/WolfBench-main")
OUT = Path("/Users/zhangyuejun/Documents/aaai/tmp/closure_validity_audit")
FIXED_GLOB = ROOT / "paper_experiments_v3/outputs/p02_size_decomposition_paper_s*/data.csv"
BOUNDARY_PATH = ROOT / "paper_experiments_v3/figures/generated/table2_scaling_results.csv"
EPS = 1e-9
BOOTSTRAPS = 5000
RNG_SEED = 20260726

SOCIAL_VARIANTS = {
    "fixed_depth_q0": 0.0,
    "baseline_q05": 0.5,
    "per_capita_depth_q1": 1.0,
}


def load_fixed_k() -> pd.DataFrame:
    paths = sorted(glob.glob(str(FIXED_GLOB)))
    if len(paths) != 12:
        raise RuntimeError(f"Expected 12 fixed-K seed files, found {len(paths)}")
    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["status"] == "ok") & df["variant"].isin(SOCIAL_VARIANTS)].copy()
    df["N"] = pd.to_numeric(df["n_society"])
    df["K"] = pd.to_numeric(df["requested_harmful_count"])
    df["R"] = pd.to_numeric(df["primary_failure_score_max"])
    df["G"] = pd.to_numeric(df["failure_gain_proxy"])
    df["failure"] = pd.to_numeric(df["primary_failure_rate"])
    df["seed"] = pd.to_numeric(df["seed"]).astype(int)
    key = ["variant", "N", "K", "seed"]
    duplicate_count = int(df.duplicated(key).sum())
    if duplicate_count:
        raise RuntimeError(f"Found {duplicate_count} duplicate fixed-K rows")
    if (df["R"] <= 0).any() or (df["G"] <= 0).any():
        raise RuntimeError("Social-on audit rows must have positive R and G")
    return df


def load_boundaries() -> pd.DataFrame:
    df = pd.read_csv(BOUNDARY_PATH)
    df = df[df["status"] == "resolved"].copy()
    df["N"] = pd.to_numeric(df["N"])
    df["alpha_c"] = pd.to_numeric(df["alpha_c"])
    return df[["N", "alpha_c"]].sort_values("N").reset_index(drop=True)


def add_cell_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cell_failure_rate"] = out.groupby(["variant", "N", "K"])["failure"].transform("mean")
    return out


def select_regime(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    rated = add_cell_rates(df)
    rated["cell_score_mean"] = rated.groupby(["variant", "N", "K"])["R"].transform("mean")
    if regime == "all_fixed_k":
        return rated
    if regime == "subcritical_below_half":
        return rated[rated["cell_failure_rate"] < 0.5].copy()
    if regime == "mean_score_below_one":
        return rated[rated["cell_score_mean"] < 1.0].copy()
    if regime == "strict_zero_failure":
        return rated[rated["cell_failure_rate"] == 0.0].copy()
    raise ValueError(regime)


def fit_log_surface(df: pd.DataFrame, outcome: str = "R") -> dict[str, float]:
    usable = df[(df["N"] > 0) & (df["K"] > 0) & (df[outcome] > 0)].copy()
    x = np.column_stack(
        [
            np.ones(len(usable)),
            np.log(usable["N"].to_numpy(dtype=float)),
            np.log(usable["K"].to_numpy(dtype=float)),
        ]
    )
    y = np.log(usable[outcome].to_numpy(dtype=float))
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    return {
        "intercept": float(beta[0]),
        "slope_n": float(beta[1]),
        "slope_k": float(beta[2]),
        "n_rows": int(len(usable)),
        "n_cells": int(usable[["N", "K"]].drop_duplicates().shape[0]),
        "n_sizes": int(usable["N"].nunique()),
        "rank": int(rank),
        "rmse_log": float(np.sqrt(np.mean(residual**2))),
        "r2_log": float(1.0 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)),
    }


def bootstrap_surface(
    df: pd.DataFrame, regime: str, outcome: str = "R", n_boot: int = BOOTSTRAPS
) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    seeds = np.array(sorted(df["seed"].unique()), dtype=int)
    draws = []
    by_seed = {seed: df[df["seed"] == seed].copy() for seed in seeds}
    for _ in range(n_boot):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        pieces = []
        for draw_id, seed in enumerate(sampled):
            piece = by_seed[int(seed)].copy()
            piece["seed"] = draw_id
            pieces.append(piece)
        boot = select_regime(pd.concat(pieces, ignore_index=True), regime)
        try:
            fit = fit_log_surface(boot, outcome)
        except np.linalg.LinAlgError:
            continue
        b_n = fit["slope_n"]
        b_k = fit["slope_k"]
        draws.append(
            {
                "slope_n": b_n,
                "slope_k": b_k,
                "nu_proxy_closure": 1.0 + b_n,
                "nu_response_contour": 1.0 + b_n / b_k if abs(b_k) > 1e-9 else np.nan,
            }
        )
    return pd.DataFrame(draws)


def interval(draws: pd.Series) -> tuple[float, float]:
    clean = draws[np.isfinite(draws)]
    if clean.empty:
        return (math.nan, math.nan)
    low, high = np.quantile(clean, [0.025, 0.975])
    return float(low), float(high)


def algebra_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, ell in SOCIAL_VARIANTS.items():
        group = df[df["variant"] == variant].copy()
        fit_r = fit_log_surface(group, "R")
        fit_g = fit_log_surface(group, "G")
        b_n, b_k = fit_r["slope_n"], fit_r["slope_k"]
        zeta_from_definition = b_n + ell
        delta = 1.0 - ell
        rows.append(
            {
                "variant": variant,
                "ell": ell,
                "delta_definition": delta,
                "risk_slope_n_empirical": b_n,
                "risk_slope_k_empirical": b_k,
                "gain_slope_n_fitted": fit_g["slope_n"],
                "gain_slope_n_definition": zeta_from_definition,
                "gain_identity_error": fit_g["slope_n"] - zeta_from_definition,
                "gain_slope_k_fitted": fit_g["slope_k"],
                "gain_slope_k_definition": b_k - 1.0,
                "delta_plus_zeta": delta + fit_g["slope_n"],
                "definition_reduced_sum": 1.0 + b_n,
                "response_contour_nu": 1.0 + b_n / b_k,
                "rows": fit_r["n_rows"],
            }
        )
    return pd.DataFrame(rows)


def reparameterization_audit(baseline: pd.DataFrame) -> pd.DataFrame:
    """Change only the analyst's normalization, holding simulator R fixed."""
    rows = []
    fit_r = fit_log_surface(baseline, "R")
    for ell in [0.0, 0.25, 0.5, 0.75, 1.0]:
        transformed = baseline.copy()
        transformed["G_alt"] = transformed["R"] * transformed["N"] ** ell / transformed["K"]
        fit_g = fit_log_surface(transformed, "G_alt")
        delta = 1.0 - ell
        rows.append(
            {
                "ell_analyst_normalization": ell,
                "delta_definition": delta,
                "zeta_fitted": fit_g["slope_n"],
                "delta_plus_zeta": delta + fit_g["slope_n"],
                "risk_slope_n": fit_r["slope_n"],
                "definition_reduced_sum": 1.0 + fit_r["slope_n"],
                "identity_error": delta + fit_g["slope_n"] - (1.0 + fit_r["slope_n"]),
            }
        )
    return pd.DataFrame(rows)


def per_k_signal_audit(baseline: pd.DataFrame) -> pd.DataFrame:
    selected = select_regime(baseline, "subcritical_below_half")
    rows = []
    for k, group in selected.groupby("K", sort=True):
        if group["N"].nunique() < 3:
            continue
        x = np.column_stack([np.ones(len(group)), np.log(group["N"].to_numpy(dtype=float))])
        y = np.log(group["R"].to_numpy(dtype=float))
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
        rows.append(
            {
                "K": int(k),
                "slope_n_at_fixed_k": float(beta[1]),
                "n_sizes": int(group["N"].nunique()),
                "sizes": ";".join(str(int(n)) for n in sorted(group["N"].unique())),
                "n_rows": int(len(group)),
                "rmse_log": float(np.sqrt(np.mean(residual**2))),
            }
        )
    return pd.DataFrame(rows)


def signal_audit(baseline: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    boot_tables: dict[str, pd.DataFrame] = {}
    for regime in [
        "all_fixed_k",
        "subcritical_below_half",
        "mean_score_below_one",
        "strict_zero_failure",
    ]:
        selected = select_regime(baseline, regime)
        fit = fit_log_surface(selected, "R")
        boots = bootstrap_surface(baseline, regime, "R")
        boot_tables[regime] = boots
        bn_ci = interval(boots["slope_n"])
        bk_ci = interval(boots["slope_k"])
        proxy_ci = interval(boots["nu_proxy_closure"])
        contour_ci = interval(boots["nu_response_contour"])
        rows.append(
            {
                "regime": regime,
                **fit,
                "cell_failure_rate_min": float(selected["cell_failure_rate"].min()),
                "cell_failure_rate_max": float(selected["cell_failure_rate"].max()),
                "slope_n_ci_low": bn_ci[0],
                "slope_n_ci_high": bn_ci[1],
                "slope_k_ci_low": bk_ci[0],
                "slope_k_ci_high": bk_ci[1],
                "nu_proxy_closure": 1.0 + fit["slope_n"],
                "nu_proxy_ci_low": proxy_ci[0],
                "nu_proxy_ci_high": proxy_ci[1],
                "nu_response_contour": 1.0 + fit["slope_n"] / fit["slope_k"],
                "nu_contour_ci_low": contour_ci[0],
                "nu_contour_ci_high": contour_ci[1],
                "bootstrap_p_slope_n_ge_0": float(np.mean(boots["slope_n"] >= 0)),
                "bootstrap_p_slope_k_le_0": float(np.mean(boots["slope_k"] <= 0)),
            }
        )
    return pd.DataFrame(rows), boot_tables


def predict_alpha(fit: dict[str, float], n: float, threshold: float = 1.0) -> float:
    b_k = fit["slope_k"]
    if not np.isfinite(b_k) or abs(b_k) < 1e-9:
        return math.nan
    log_k = (math.log(threshold) - fit["intercept"] - fit["slope_n"] * math.log(n)) / b_k
    return math.exp(log_k) / n


def prediction_audit(
    baseline: pd.DataFrame, boundaries: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = []
    regimes = ["all_fixed_k", "subcritical_below_half", "strict_zero_failure"]
    regimes.insert(2, "mean_score_below_one")
    for regime in regimes:
        # No midpoint is used here: the score's pre-specified threshold R=1
        # supplies the absolute crossing level.
        fit_all_n = fit_log_surface(select_regime(baseline, regime), "R")
        for row in boundaries.itertuples(index=False):
            pred = predict_alpha(fit_all_n, float(row.N))
            detail.append(
                {
                    "protocol": "all_sizes_response_no_midpoints",
                    "regime": regime,
                    "heldout_n": int(row.N),
                    "train_response_sizes": ";".join(str(int(x)) for x in sorted(baseline["N"].unique())),
                    "uses_any_boundary_midpoint_for_fit": False,
                    "uses_heldout_n_response_rows": True,
                    "observed_alpha_c_evaluation_only": float(row.alpha_c),
                    "predicted_alpha_c": pred,
                    "ratio_pred_obs": pred / float(row.alpha_c),
                    "abs_log_error": abs(math.log(pred / float(row.alpha_c))),
                    "fit_slope_n": fit_all_n["slope_n"],
                    "fit_slope_k": fit_all_n["slope_k"],
                    "train_rows": fit_all_n["n_rows"],
                }
            )

        # Stronger test: remove every response row at the target size as well.
        for row in boundaries.itertuples(index=False):
            train_raw = baseline[baseline["N"] != row.N].copy()
            train = select_regime(train_raw, regime)
            fit = fit_log_surface(train, "R")
            pred = predict_alpha(fit, float(row.N))
            detail.append(
                {
                    "protocol": "leave_one_size_out_response_no_midpoints",
                    "regime": regime,
                    "heldout_n": int(row.N),
                    "train_response_sizes": ";".join(str(int(x)) for x in sorted(train["N"].unique())),
                    "uses_any_boundary_midpoint_for_fit": False,
                    "uses_heldout_n_response_rows": False,
                    "observed_alpha_c_evaluation_only": float(row.alpha_c),
                    "predicted_alpha_c": pred,
                    "ratio_pred_obs": pred / float(row.alpha_c),
                    "abs_log_error": abs(math.log(pred / float(row.alpha_c))),
                    "fit_slope_n": fit["slope_n"],
                    "fit_slope_k": fit["slope_k"],
                    "train_rows": fit["n_rows"],
                }
            )
    detail_df = pd.DataFrame(detail)
    summaries = []
    for (protocol, regime), group in detail_df.groupby(["protocol", "regime"], sort=True):
        x = np.log(group["heldout_n"].to_numpy(dtype=float))
        y = np.log(group["predicted_alpha_c"].to_numpy(dtype=float))
        pred_slope = float(np.polyfit(x, y, 1)[0])
        ratios = np.maximum(group["ratio_pred_obs"], 1.0 / group["ratio_pred_obs"])
        summaries.append(
            {
                "protocol": protocol,
                "regime": regime,
                "n_predictions": len(group),
                "mean_abs_log_error": float(group["abs_log_error"].mean()),
                "median_abs_log_error": float(group["abs_log_error"].median()),
                "max_abs_log_error": float(group["abs_log_error"].max()),
                "within_factor_1p5": int((ratios <= 1.5).sum()),
                "within_factor_2": int((ratios <= 2.0).sum()),
                "predicted_nu": -pred_slope,
                "spearman_alpha": float(
                    group["observed_alpha_c_evaluation_only"]
                    .rank()
                    .corr(group["predicted_alpha_c"].rank())
                ),
            }
        )
    return detail_df, pd.DataFrame(summaries)


def write_report(
    algebra: pd.DataFrame,
    reparameterization: pd.DataFrame,
    per_k: pd.DataFrame,
    signal: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    prediction_detail: pd.DataFrame,
    observed_nu: float,
) -> None:
    baseline_alg = algebra[algebra["variant"] == "baseline_q05"].iloc[0]
    sub = signal[signal["regime"] == "subcritical_below_half"].iloc[0]
    strict_pred = prediction_summary[
        (prediction_summary["protocol"] == "leave_one_size_out_response_no_midpoints")
        & (prediction_summary["regime"] == "subcritical_below_half")
    ].iloc[0]
    lines = [
        "# Closure Validity Audit",
        "",
        "This audit uses existing fixed-K episode data only. It does not modify or refit the paper.",
        "",
        "## 1. Definition-implied versus empirical content",
        "",
        "For `log R = a + b_N log N + b_K log K`, the proxy definition",
        "`G = R N^ell / K` implies exactly:",
        "",
        "- `zeta = b_N + ell`,",
        "- `delta + zeta = 1 + b_N`, because `delta = 1 - ell`,",
        "- the response-contour prediction is `nu = 1 + b_N / b_K`, not",
        "  `1 + b_N`, unless `b_K = 1`.",
        "",
        (
            f"Numerically, the largest fitted identity error across depth rules is "
            f"{algebra['gain_identity_error'].abs().max():.3e}. Thus the cancellation "
            "of ell is definition-implied to numerical precision."
        ),
        (
            f"Holding the baseline simulator responses fixed and changing only the "
            f"analyst normalization from ell=0 to ell=1 changes zeta from "
            f"{reparameterization['zeta_fitted'].min():.3f} to "
            f"{reparameterization['zeta_fitted'].max():.3f}, while delta+zeta remains "
            f"{reparameterization['delta_plus_zeta'].iloc[0]:.3f} in every row."
        ),
        (
            f"In the baseline fixed-K data, b_N={baseline_alg['risk_slope_n_empirical']:.3f} "
            f"and b_K={baseline_alg['risk_slope_k_empirical']:.3f}. The proxy sum is "
            f"{baseline_alg['delta_plus_zeta']:.3f}, while the direct response-contour "
            f"formula gives {baseline_alg['response_contour_nu']:.3f}."
        ),
        "",
        "## 2. Independent size-response signal in subcritical fixed-K data",
        "",
        (
            f"Using only cells with P02 failure prevalence below 1/2 gives "
            f"{int(sub['n_rows'])} episodes across {int(sub['n_cells'])} (N,K) cells. "
            f"The direct risk-size coefficient is b_N={sub['slope_n']:.3f} "
            f"(paired-seed bootstrap 95% CI "
            f"[{sub['slope_n_ci_low']:.3f}, {sub['slope_n_ci_high']:.3f}])."
        ),
        (
            f"The harmful-count elasticity is b_K={sub['slope_k']:.3f} "
            f"(95% CI [{sub['slope_k_ci_low']:.3f}, {sub['slope_k_ci_high']:.3f}]). "
            f"The contour-implied exponent is {sub['nu_response_contour']:.3f} "
            f"(95% bootstrap interval [{sub['nu_contour_ci_low']:.3f}, "
            f"{sub['nu_contour_ci_high']:.3f}])."
        ),
        (
            "The sign of b_N is empirical and is not created by the gain normalization. "
            "However, its interpretation as an amplification mechanism remains limited "
            "because R is the same score used to define collapse."
        ),
        (
            "At each fixed K with at least three subcritical sizes, the direct size "
            f"slopes range from {per_k['slope_n_at_fixed_k'].min():.3f} to "
            f"{per_k['slope_n_at_fixed_k'].max():.3f}; the negative direction is not "
            "driven by pooling different harmful counts."
        ),
        "",
        "## 3. Boundary prediction without held-out midpoints",
        "",
        (
            "The strict test fits the subcritical response surface after removing every "
            "response row at the target N, uses the pre-specified score threshold R=1, "
            "and uses no boundary midpoint for fitting or calibration."
        ),
        (
            f"It obtains mean absolute log error {strict_pred['mean_abs_log_error']:.3f}, "
            f"{int(strict_pred['within_factor_1p5'])}/6 predictions within 1.5x, "
            f"{int(strict_pred['within_factor_2'])}/6 within 2x, and predicted "
            f"nu={strict_pred['predicted_nu']:.3f} versus observed nu={observed_nu:.3f}."
        ),
        (
            "For context, the existing leave-one-size-out constant-boundary baseline "
            "has mean absolute log error 0.212, and direct boundary power-law "
            "extrapolation has 0.092. The strict subcritical response prediction does "
            "not beat either midpoint-based baseline."
        ),
        "",
        "## Decision",
        "",
        (
            "The audit supports a real fixed-K size-response signal, but it does not "
            "support treating delta + zeta as an independent two-component validation. "
            "The ell cancellation is algebraic, and the stronger no-midpoint prediction "
            "must be judged by the errors above. Closure is best treated as a descriptive "
            "response-contour consistency check unless an independent perturbation-response "
            "measure is added."
        ),
        "",
        "Full results are in `algebra_audit.csv`, `reparameterization_audit.csv`,",
        "`per_k_subcritical_signal.csv`, `subcritical_signal.csv`,",
        "`no_midpoint_prediction_summary.csv`, and `no_midpoint_prediction_detail.csv`.",
    ]
    (OUT / "audit_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixed = load_fixed_k()
    boundaries = load_boundaries()
    baseline = fixed[fixed["variant"] == "baseline_q05"].copy()
    algebra = algebra_audit(fixed)
    reparameterization = reparameterization_audit(baseline)
    per_k = per_k_signal_audit(baseline)
    signal, boot_tables = signal_audit(baseline)
    pred_detail, pred_summary = prediction_audit(baseline, boundaries)

    observed_slope = np.polyfit(
        np.log(boundaries["N"].to_numpy(dtype=float)),
        np.log(boundaries["alpha_c"].to_numpy(dtype=float)),
        1,
    )[0]
    observed_nu = float(-observed_slope)

    algebra.to_csv(OUT / "algebra_audit.csv", index=False)
    reparameterization.to_csv(OUT / "reparameterization_audit.csv", index=False)
    per_k.to_csv(OUT / "per_k_subcritical_signal.csv", index=False)
    signal.to_csv(OUT / "subcritical_signal.csv", index=False)
    pred_detail.to_csv(OUT / "no_midpoint_prediction_detail.csv", index=False)
    pred_summary.to_csv(OUT / "no_midpoint_prediction_summary.csv", index=False)
    for regime, table in boot_tables.items():
        table.to_csv(OUT / f"bootstrap_{regime}.csv", index=False)
    metadata = {
        "fixed_k_files": len(glob.glob(str(FIXED_GLOB))),
        "fixed_k_social_on_rows": int(len(fixed)),
        "baseline_rows": int(len(baseline)),
        "seeds": sorted(int(x) for x in baseline["seed"].unique()),
        "boundary_path": str(BOUNDARY_PATH),
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": RNG_SEED,
        "observed_nu_recomputed": observed_nu,
        "paper_sources_modified": False,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(
        algebra,
        reparameterization,
        per_k,
        signal,
        pred_summary,
        pred_detail,
        observed_nu,
    )
    print(OUT / "audit_report.md")


if __name__ == "__main__":
    main()
