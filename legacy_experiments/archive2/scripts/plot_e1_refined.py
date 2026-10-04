"""Paper-facing plots for the refined E1 scaling experiment."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wolfbench-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .io_utils import OUTPUTS, ROOT, ensure_dir
from .plot_calibrated_figures import (
    CALIBRATED_MARGIN,
    S1_THRESHOLDS,
    S2_THRESHOLDS,
    _finite,
    _scenario_key,
)
from .plot_final_figures import INK, MUTED, PALETTE, clean_axis, set_style


PINK = "#DB2777"
MAGENTA = "#C026D3"
PURPLE = "#7E22CE"
TEAL = "#14B8A6"
GRID = "#E9D7F0"


def read_ok(out_name: str) -> pd.DataFrame:
    path = OUTPUTS / out_name / "data.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "status" in frame.columns:
        frame = frame[frame["status"].fillna("ok").eq("ok")].copy()
    for column in frame.columns:
        if column in {"scenario", "variant", "defense", "status", "primary_metric", "target_asset", "placement"}:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _s1_s2_score(row: pd.Series, thresholds: dict[str, float], scale: float) -> float:
    t = {key: max(value * scale, 1e-12) for key, value in thresholds.items()}
    social = _finite(row.get("social_cascade_peak")) / t["social_cascade"]
    price = _finite(row.get("price_dislocation_max")) / t["price_dislocation"]
    liquidity = _finite(row.get("liquidity_stress_max")) / t["liquidity_stress"]
    return float(max(0.0, min(social, max(price, liquidity))))


def primary_score(row: pd.Series, scale: float = 1.0) -> float:
    scenario = _scenario_key(row.get("scenario", ""))
    if scenario == "s1":
        return _s1_s2_score(row, S1_THRESHOLDS, scale)
    if scenario == "s2":
        return _s1_s2_score(row, S2_THRESHOLDS, scale)
    return _finite(row.get("primary_failure_score_max"))


def add_calibration(frame: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    frame = frame.copy()
    frame["scenario_primary_score"] = frame.apply(lambda row: primary_score(row, scale), axis=1)
    keys = ["scenario", "n_society"]
    clean = frame[np.isclose(frame["alpha"].astype(float), 0.0)]
    if clean.empty:
        frame["clean_baseline_score"] = 0.0
        frame["clean_baseline_std"] = 0.0
    else:
        stats = (
            clean.groupby(keys, dropna=False)["scenario_primary_score"]
            .agg(clean_baseline_score="mean", clean_baseline_std="std")
            .reset_index()
        )
        stats["clean_baseline_std"] = stats["clean_baseline_std"].fillna(0.0)
        frame = frame.merge(stats, on=keys, how="left")
        frame["clean_baseline_score"] = frame["clean_baseline_score"].fillna(0.0)
        frame["clean_baseline_std"] = frame["clean_baseline_std"].fillna(0.0)
    clean_alpha = np.isclose(frame["alpha"].astype(float), 0.0)
    threshold = frame["clean_baseline_score"] + np.maximum(
        CALIBRATED_MARGIN,
        2.0 * frame["clean_baseline_std"],
    )
    frame["clean_calibrated_failure"] = (
        (frame["alpha"].astype(float) > 0.0)
        & (frame["scenario_primary_score"] >= threshold)
    ).astype(float)
    frame.loc[clean_alpha, "clean_calibrated_failure"] = 0.0
    frame["excess_primary_score"] = (
        frame["scenario_primary_score"] - frame["clean_baseline_score"]
    ).clip(lower=0.0)
    frame.loc[clean_alpha, "excess_primary_score"] = 0.0
    return frame


def sigmoid(xs: np.ndarray, slope: float, midpoint: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(slope * (xs - midpoint), -60.0, 60.0)))


def fit_sigmoid(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float] | None:
    if xs.size < 3 or np.allclose(ys, ys[0]):
        return None
    slopes = np.geomspace(20.0, 4000.0, 140)
    mids = np.linspace(float(xs.min()), float(xs.max()), 180)
    y = np.clip(ys.astype(float), 0.0, 1.0)
    best: tuple[float, float, float] | None = None
    for slope in slopes:
        pred = sigmoid(xs[:, None], slope, mids[None, :])
        pred = np.clip(pred, 1e-6, 1 - 1e-6)
        losses = -(y[:, None] * np.log(pred) + (1.0 - y[:, None]) * np.log(1.0 - pred)).mean(axis=0)
        idx = int(np.argmin(losses))
        loss = float(losses[idx])
        if best is None or loss < best[0]:
            best = (loss, float(slope), float(mids[idx]))
    return None if best is None else (best[1], best[2])


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = float(k) / float(n)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def aggregate_prob(frame: pd.DataFrame) -> pd.DataFrame:
    out = (
        frame.groupby(["n_society", "alpha"], dropna=False)["clean_calibrated_failure"]
        .agg(k="sum", n="size")
        .reset_index()
        .sort_values(["n_society", "alpha"])
    )
    out["prob"] = out["k"] / out["n"].clip(lower=1)
    intervals = [wilson_interval(row.k, row.n) for row in out.itertuples(index=False)]
    out["wilson_lo"] = [lo for lo, _ in intervals]
    out["wilson_hi"] = [hi for _, hi in intervals]
    return out[["n_society", "alpha", "k", "n", "prob", "wilson_lo", "wilson_hi"]]


def alpha_c_from_group(group: pd.DataFrame) -> tuple[float, str]:
    points = group.sort_values("alpha")
    positive = points[points["alpha"].astype(float) > 0.0]
    if positive.empty:
        return float("nan"), "missing"
    xs = positive["alpha"].astype(float).to_numpy()
    ys = positive["prob"].astype(float).to_numpy()
    if np.nanmax(ys) < 0.5:
        return float(xs.max()), "right_censored"
    if np.nanmin(ys) >= 0.5:
        return float(xs.min()), "left_censored"
    for i in range(1, len(xs)):
        y0, y1 = ys[i - 1], ys[i]
        if (y0 < 0.5 <= y1) or (y1 < 0.5 <= y0):
            if abs(y1 - y0) < 1e-9:
                return float(xs[i]), "interpolated"
            frac = (0.5 - y0) / (y1 - y0)
            return float(xs[i - 1] + frac * (xs[i] - xs[i - 1])), "interpolated"
    fit = fit_sigmoid(positive["alpha"].to_numpy(float), positive["prob"].to_numpy(float))
    if fit is not None:
        return float(fit[1]), "fit"
    return float(xs[np.argmin(np.abs(ys - 0.5))]), "nearest"


def alpha_c_table(frame: pd.DataFrame) -> pd.DataFrame:
    prob = aggregate_prob(frame)
    rows: list[dict[str, Any]] = []
    for n, group in prob.groupby("n_society", dropna=False):
        ac, method = alpha_c_from_group(group)
        rows.append({"n_society": int(n), "alpha_c": ac, "alpha_c_method": method})
    return pd.DataFrame(rows).sort_values("n_society")


def bootstrap_alpha_c(frame: pd.DataFrame, n_boot: int = 1000, seed: int = 7, ci: float = 0.95) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q_lo = (1.0 - ci) / 2.0
    q_hi = 1.0 - q_lo
    rows: list[dict[str, Any]] = []
    for n, group in frame.groupby("n_society", dropna=False):
        seeds = sorted(group["seed"].dropna().unique())
        if len(seeds) < 2:
            continue
        values = []
        for _ in range(n_boot):
            sampled = rng.choice(seeds, size=len(seeds), replace=True)
            sample = pd.concat([group[group["seed"].eq(s)] for s in sampled], ignore_index=True)
            prob = aggregate_prob(sample)
            ac, method = alpha_c_from_group(prob[prob["n_society"].eq(n)])
            if math.isfinite(ac):
                values.append(ac)
        if values:
            rows.append({
                "n_society": int(n),
                "alpha_c_lo": float(np.quantile(values, q_lo)),
                "alpha_c_hi": float(np.quantile(values, q_hi)),
                "alpha_c_ci": ci,
                "n_seeds": len(seeds),
            })
    return pd.DataFrame(rows)


def save(fig: plt.Figure, stem: str) -> None:
    for directory in (ensure_dir(ROOT / "figures"), ensure_dir(ROOT.parent / "AuthorKit27" / "Figures")):
        fig.savefig(directory / f"{stem}.pdf")
        fig.savefig(directory / f"{stem}.png", dpi=420)
    plt.close(fig)


def draw_s1_main(frame: pd.DataFrame, out_name: str) -> None:
    frame = add_calibration(frame, 1.0)
    ensure_dir(OUTPUTS / out_name)
    frame.to_csv(OUTPUTS / out_name / "data_calibrated.csv", index=False)
    prob = aggregate_prob(frame)
    ac = alpha_c_table(frame)
    ci = bootstrap_alpha_c(frame)
    if not ci.empty and "n_society" in ci.columns:
        ac = ac.merge(ci, on="n_society", how="left")
    prob.to_csv(OUTPUTS / out_name / "e1_wilson_rates.csv", index=False)
    ac.to_csv(OUTPUTS / out_name / "e1_alpha_c_summary.csv", index=False)

    fig = plt.figure(figsize=(9.79, 3.27))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.05, 1.24], wspace=0.62)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel A: collapse-probability phase map on the N-specific alpha grid.
    # Each society size is a categorical row whose cells span the midpoints
    # between measured alpha values, so the map reads as a continuous heatmap
    # instead of floating markers. The alpha_c boundary is quantified in
    # Panel B and is intentionally not redrawn here to avoid duplication.
    ns_rows = [n for n in [50, 100, 200, 300, 500, 1000, 2000] if prob["n_society"].eq(n).any()]
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("RdPu")
    for yi, n in enumerate(ns_rows):
        row = prob[(prob["n_society"].eq(n)) & (prob["alpha"] <= 0.065)].sort_values("alpha")
        xs = row["alpha"].astype(float).to_numpy()
        ps = row["prob"].astype(float).to_numpy()
        if xs.size == 0:
            continue
        if xs.size == 1:
            edges = np.array([xs[0] - 0.004, xs[0] + 0.004])
        else:
            mids = (xs[:-1] + xs[1:]) / 2.0
            first = xs[0] - (mids[0] - xs[0])
            last = xs[-1] + (xs[-1] - mids[-1])
            edges = np.concatenate([[first], mids, [last]])
        edges = np.clip(edges, -0.001, 0.066)
        ax0.pcolormesh(
            edges,
            [yi - 0.46, yi + 0.46],
            ps[None, :],
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="white",
            linewidth=0.3,
        )
    ax0.set_ylim(-0.6, len(ns_rows) - 0.4)
    ax0.set_yticks(range(len(ns_rows)))
    ax0.set_yticklabels([str(n) for n in ns_rows])
    ax0.set_xlim(-0.001, 0.065)
    ax0.set_xlabel(r"harmful fraction $\alpha$")
    ax0.set_ylabel("society size N")
    ax0.grid(False)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax0, fraction=0.038, pad=0.03)
    cb.ax.set_title("P", fontsize=7.0, color=INK, pad=3)
    cb.ax.tick_params(labelsize=6.2, length=2)
    cb.outline.set_visible(False)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # Panel B: critical alpha versus N. Resolved estimates form a connected
    # purple line; right-censored anchors (alpha_c beyond the tested range,
    # e.g. N=50) use an open upward triangle plus an arrow to denote a lower
    # bound rather than a point estimate -- replacing the old bare ">" glyph.
    main = ac[ac["n_society"] >= 50].copy()
    is_rc = main["alpha_c_method"].eq("right_censored")
    is_lc = main["alpha_c_method"].eq("left_censored")
    resolved = main[~(is_rc | is_lc)]
    if {"alpha_c_lo", "alpha_c_hi"}.issubset(main.columns):
        valid = main["alpha_c_lo"].notna() & main["alpha_c_hi"].notna()
        ax1.fill_between(
            main.loc[valid, "n_society"].astype(float),
            main.loc[valid, "alpha_c_lo"].astype(float),
            main.loc[valid, "alpha_c_hi"].astype(float),
            color=PURPLE,
            alpha=0.12,
            linewidth=0,
        )
    ax1.plot(
        resolved["n_society"], resolved["alpha_c"],
        color=PURPLE, marker="o", linewidth=2.0, markersize=4.4,
        label=r"resolved $\alpha_c$", zorder=4,
    )
    for _, row in main[is_rc].iterrows():
        x = float(row["n_society"]); y = float(row["alpha_c"])
        ax1.plot(
            [x], [y], marker="^", markersize=6.5, linestyle="none",
            markerfacecolor="none", markeredgecolor=MUTED, markeredgewidth=1.1,
            zorder=5, label=r"censored ($\alpha_c\!\geq$)",
        )
        ax1.annotate(
            "", xy=(x, min(y + 0.011, 0.111)), xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0), zorder=5,
        )
    for _, row in main[is_lc].iterrows():
        x = float(row["n_society"]); y = float(row["alpha_c"])
        ax1.plot(
            [x], [y], marker="v", markersize=6.5, linestyle="none",
            markerfacecolor="none", markeredgecolor=MUTED, markeredgewidth=1.1,
            zorder=5, label=r"censored ($\alpha_c\!\leq$)",
        )
    ax1.set_xscale("log")
    ax1.set_ylim(-0.004, 0.114)
    ax1.set_xlabel("society size N")
    ax1.set_ylabel(r"estimated $\alpha_c$")
    ax1.grid(True, axis="both", color=GRID, linewidth=0.6)
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        ax1.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=6.2, loc="upper right", handlelength=1.4, borderaxespad=0.3)
    clean_axis(ax1)

    # Panel C: representative S-curves in the resolved transition region.
    # N=100 is kept in Panel B, but omitted here because its larger alpha range
    # compresses the small-alpha transition for N>=300. Each series gets a
    # colourblind-safe colour plus a distinct marker and line style so the
    # curves stay separable in grayscale and for colourblind readers.
    # Pink-purple ramp with monotonically increasing lightness, shared with the
    # Figure 4 scenario curves so the whole paper uses one colour family; each
    # series still carries a distinct marker and line style for grayscale and
    # colourblind separability.
    series_colors = ["#4A1A6B", "#8E2A8F", "#C0399B", "#E87FB8"]
    markers = ["o", "s", "^", "D"]
    lstyles = ["-", "--", "-.", ":"]
    legend_handles: list[Line2D] = []
    for idx, n in enumerate([300, 500, 1000, 2000]):
        group = frame[frame["n_society"].eq(n)].copy()
        if group.empty:
            continue
        agg = aggregate_prob(group)
        xs = agg["alpha"].astype(float).to_numpy()
        ys = agg["prob"].astype(float).to_numpy()
        color = series_colors[idx]
        marker = markers[idx]
        lstyle = lstyles[idx]
        yerr = np.vstack([
            np.maximum(0.0, ys - agg["wilson_lo"].astype(float).to_numpy()),
            np.maximum(0.0, agg["wilson_hi"].astype(float).to_numpy() - ys),
        ])
        ax2.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt=marker,
            color=color,
            ecolor=to_rgba(color, 0.32),
            elinewidth=0.7,
            capsize=1.2,
            capthick=0.6,
            markersize=3.6,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=3,
        )
        fit = fit_sigmoid(group["alpha"].to_numpy(float), group["clean_calibrated_failure"].to_numpy(float))
        if fit is not None:
            slope, midpoint = fit
            smooth = np.linspace(float(xs.min()), float(xs.max()), 240)
            ax2.plot(smooth, sigmoid(smooth, slope, midpoint), color=color, linewidth=1.9, linestyle=lstyle, zorder=2)
        else:
            ax2.plot(xs, ys, color=color, linewidth=1.5, linestyle=lstyle, zorder=2)
        legend_handles.append(
            Line2D([0], [0], color=color, marker=marker, linestyle=lstyle, linewidth=1.9, markersize=5.0, markerfacecolor=color, markeredgecolor="white", label=f"N={n}")
        )
    ax2.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.9, zorder=1)
    ax2.set_xlim(-0.001, 0.031)
    ax2.set_ylim(-0.04, 1.04)
    ax2.set_xlabel(r"harmful fraction $\alpha$")
    ax2.set_ylabel("collapse probability")
    ax2.grid(True, axis="y", color=GRID, linewidth=0.6)
    if legend_handles:
        ax2.legend(handles=legend_handles, frameon=False, fontsize=6.8, loc="lower right", handlelength=2.0, borderaxespad=0.4)
    clean_axis(ax2)

    fig.subplots_adjust(top=0.94, bottom=0.22)
    save(fig, "figure3_e1_refined_scaling")


def draw_s2_replication(frame: pd.DataFrame, out_name: str) -> None:
    frame = add_calibration(frame, 1.0)
    ensure_dir(OUTPUTS / out_name)
    frame.to_csv(OUTPUTS / out_name / "data_calibrated.csv", index=False)
    prob = aggregate_prob(frame)
    ac = alpha_c_table(frame)
    ci = bootstrap_alpha_c(frame)
    if not ci.empty and "n_society" in ci.columns:
        ac = ac.merge(ci, on="n_society", how="left")
    prob.to_csv(OUTPUTS / out_name / "e1_wilson_rates.csv", index=False)
    ac.to_csv(OUTPUTS / out_name / "e1_alpha_c_summary.csv", index=False)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5.75, 2.25), gridspec_kw={"wspace": 0.50})
    sc = ax0.scatter(
        prob["alpha"],
        prob["n_society"],
        c=prob["prob"],
        s=86,
        marker="s",
        cmap="RdPu",
        vmin=0,
        vmax=1,
        edgecolor="white",
        linewidth=0.45,
    )
    ax0.plot(ac["alpha_c"], ac["n_society"], color=INK, linewidth=1.3, marker="o", markersize=3.1)
    ax0.set_yscale("log")
    ax0.set_title("A  S2 phase check", loc="left", fontweight="bold", color=INK, pad=3)
    ax0.set_xlabel(r"harmful fraction $\alpha$")
    ax0.set_ylabel("society size N")
    ax0.grid(True, which="major", axis="y", color=GRID, linewidth=0.6)
    cb = fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.02)
    cb.set_label("P", fontsize=7, color=INK, labelpad=-1)
    cb.ax.tick_params(labelsize=6.2, length=2)
    cb.outline.set_visible(False)
    clean_axis(ax0)

    ax1.plot(ac["n_society"], ac["alpha_c"], color=MAGENTA, marker="o", linewidth=2.0, markersize=4.2)
    if {"alpha_c_lo", "alpha_c_hi"}.issubset(ac.columns):
        valid = ac["alpha_c_lo"].notna() & ac["alpha_c_hi"].notna()
        ax1.fill_between(
            ac.loc[valid, "n_society"].astype(float),
            ac.loc[valid, "alpha_c_lo"].astype(float),
            ac.loc[valid, "alpha_c_hi"].astype(float),
            color=MAGENTA,
            alpha=0.12,
            linewidth=0,
        )
    ax1.set_xscale("log")
    ax1.set_title("B  replicated finite-size shift", loc="left", fontweight="bold", color=INK, pad=3)
    ax1.set_xlabel("society size N")
    ax1.set_ylabel(r"estimated $\alpha_c$")
    ax1.grid(True, axis="both", color=GRID, linewidth=0.6)
    clean_axis(ax1)
    fig.subplots_adjust(top=0.86, bottom=0.22)
    save(fig, "figure3b_e1_s2_replication")


def draw_threshold_sensitivity(frame: pd.DataFrame, out_name: str) -> None:
    configs = [("loose", 0.90, TEAL), ("nominal", 1.00, PURPLE), ("strict", 1.10, PINK)]
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    rows: list[dict[str, Any]] = []
    for label, scale, color in configs:
        calibrated = add_calibration(frame, scale)
        ac = alpha_c_table(calibrated)
        ac["threshold_setting"] = label
        ac["threshold_scale"] = scale
        rows.extend(ac.to_dict("records"))
        main = ac[ac["n_society"] >= 50]
        ax.plot(main["n_society"], main["alpha_c"], color=color, marker="o", linewidth=1.8, markersize=3.6, label=label)
    out = pd.DataFrame(rows)
    ensure_dir(OUTPUTS / out_name)
    out.to_csv(OUTPUTS / out_name / "threshold_sensitivity.csv", index=False)
    ax.set_xscale("log")
    ax.set_title("Threshold sensitivity", loc="left", fontweight="bold", color=INK, pad=3)
    ax.set_xlabel("society size N")
    ax.set_ylabel(r"estimated $\alpha_c$")
    ax.grid(True, axis="both", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=7.0)
    clean_axis(ax)
    save(fig, "figure3c_e1_threshold_sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-out", default="real_e1_refined_s1")
    parser.add_argument("--s2-out", default="real_e1_refined_s2")
    args = parser.parse_args()
    set_style()

    s1 = read_ok(args.s1_out)
    if not s1.empty:
        draw_s1_main(s1, args.s1_out)
        draw_threshold_sensitivity(s1, args.s1_out)
        print(f"Plotted S1 refined E1 from {args.s1_out}")
    else:
        print(f"No S1 data found for {args.s1_out}")

    s2 = read_ok(args.s2_out)
    if not s2.empty:
        draw_s2_replication(s2, args.s2_out)
        print(f"Plotted S2 replication from {args.s2_out}")
    else:
        print(f"No S2 data found for {args.s2_out}; skipping replication plot")


if __name__ == "__main__":
    main()
