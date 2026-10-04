"""Plot and summarize E6 controller-class robustness."""
from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wolfbench-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import OUTPUTS, ROOT, ensure_dir
from .plot_e1_refined import add_calibration, alpha_c_table, bootstrap_alpha_c
from .plot_final_figures import INK, MUTED, clean_axis, set_style


ORDER = ["behavioral_only", "fixed_b4_h1", "mixed_default", "higher_llm_quota"]
LABELS = {
    "behavioral_only": "Behavioral",
    "fixed_b4_h1": "Matched\n4B+1H",
    "mixed_default": "Mixed",
    "higher_llm_quota": "Higher LLM",
    "full_llm_micro": "Full LLM\nN=10",
}
COLORS = {
    300: "#DB2777",
    1000: "#7E22CE",
}


def read_ok(out_name: str) -> pd.DataFrame:
    path = OUTPUTS / out_name / "data.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "status" in frame.columns:
        frame = frame[frame["status"].fillna("ok").eq("ok")].copy()
    for column in frame.columns:
        if column in {"scenario", "variant", "defense", "status", "primary_metric", "target_asset", "placement", "controller_variant", "quota_variant"}:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def calibrated_by_controller(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby("controller_variant", dropna=False):
        parts.append(add_calibration(group, 1.0))
    return pd.concat(parts, ignore_index=True) if parts else frame.copy()


def summarize_alpha_c(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for controller, group in frame.groupby("controller_variant", dropna=False):
        ac = alpha_c_table(group)
        ci = bootstrap_alpha_c(group)
        if not ci.empty and "n_society" in ci.columns:
            ac = ac.merge(ci, on="n_society", how="left")
        ac["controller_variant"] = controller
        rows.append(ac)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    cols = ["controller_variant", "n_society", "alpha_c", "alpha_c_method", "alpha_c_lo", "alpha_c_hi", "alpha_c_ci", "n_seeds"]
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols].sort_values(["controller_variant", "n_society"])


def ordering_summary(alpha_c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    present = list(alpha_c["controller_variant"].dropna().astype(str).unique())
    controllers = [name for name in ORDER if name in present]
    controllers.extend([name for name in present if name not in controllers])
    for controller in controllers:
        group = alpha_c[alpha_c["controller_variant"].eq(controller)]
        by_n = {int(row.n_society): float(row.alpha_c) for row in group.itertuples(index=False) if math.isfinite(float(row.alpha_c))}
        ac300 = by_n.get(300)
        ac1000 = by_n.get(1000)
        rows.append({
            "controller_variant": controller,
            "alpha_c_300": ac300,
            "alpha_c_1000": ac1000,
            "ordering_300_gt_1000": bool(ac300 is not None and ac1000 is not None and ac300 > ac1000),
            "ratio_300_over_1000": (ac300 / ac1000) if ac300 is not None and ac1000 not in {None, 0.0} else np.nan,
        })
    return pd.DataFrame(rows)


def plot(alpha_c: pd.DataFrame, out_name: str) -> None:
    data = alpha_c[alpha_c["controller_variant"].isin(ORDER) & alpha_c["n_society"].isin([300, 1000])].copy()
    if data.empty:
        return
    order = [name for name in ORDER if name in set(data["controller_variant"])]
    x = np.arange(len(order))
    offsets = {300: -0.11, 1000: 0.11}
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    ax.set_facecolor("#FFF9FD")
    for n in [300, 1000]:
        subset = data[data["n_society"].eq(n)].set_index("controller_variant").reindex(order)
        y = subset["alpha_c"].astype(float).to_numpy()
        lo = subset["alpha_c_lo"].astype(float).to_numpy()
        hi = subset["alpha_c_hi"].astype(float).to_numpy()
        finite = np.isfinite(y)
        yerr = np.vstack([
            np.where(np.isfinite(lo), np.maximum(0.0, y - lo), 0.0),
            np.where(np.isfinite(hi), np.maximum(0.0, hi - y), 0.0),
        ])
        ax.errorbar(
            x[finite] + offsets[n],
            y[finite],
            yerr=yerr[:, finite],
            fmt="o-",
            color=COLORS[n],
            linewidth=1.8,
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.7,
            capsize=0,
            label=f"N={n}",
            zorder=3,
        )
    for i, controller in enumerate(order):
        y300 = data[(data["controller_variant"].eq(controller)) & (data["n_society"].eq(300))]["alpha_c"]
        y1000 = data[(data["controller_variant"].eq(controller)) & (data["n_society"].eq(1000))]["alpha_c"]
        if not y300.empty and not y1000.empty:
            ax.plot([x[i] + offsets[300], x[i] + offsets[1000]], [float(y300.iloc[0]), float(y1000.iloc[0])], color="#C4A5D5", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[v] for v in order], fontsize=7.0)
    ax.set_ylabel(r"estimated $\alpha_c$", fontsize=7.5)
    ax.set_title(
        "Controller-class robustness",
        loc="left",
        fontweight="bold",
        color=INK,
        pad=3,
        fontsize=8.6,
    )
    ax.grid(True, axis="y", color="#E9D7F0", linewidth=0.7)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.tick_params(axis="y", labelsize=7.0)
    ax.text(
        0.01,
        0.03,
        "S1 near-critical grids; lower is more fragile",
        transform=ax.transAxes,
        fontsize=6.2,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    clean_axis(ax)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.26, top=0.88)
    for directory in (ensure_dir(ROOT / "figures"), ensure_dir(ROOT.parent / "AuthorKit27" / "Figures"), ensure_dir(OUTPUTS / out_name / "figures")):
        fig.savefig(directory / "figure6b_e6_controller_ablation.pdf")
        fig.savefig(directory / "figure6b_e6_controller_ablation.png", dpi=420)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="real_e6_controller_ablation")
    args = parser.parse_args()
    set_style()
    frame = read_ok(args.out)
    if frame.empty:
        raise SystemExit(f"No ok E6 rows found for {args.out}")
    calibrated = calibrated_by_controller(frame)
    ensure_dir(OUTPUTS / args.out)
    calibrated.to_csv(OUTPUTS / args.out / "data_calibrated.csv", index=False)
    alpha_c = summarize_alpha_c(calibrated)
    alpha_c.to_csv(OUTPUTS / args.out / "e6_alpha_c_summary.csv", index=False)
    order = ordering_summary(alpha_c)
    order.to_csv(OUTPUTS / args.out / "e6_ordering_summary.csv", index=False)
    plot(alpha_c, args.out)
    print(f"Plotted E6 controller ablation from {args.out}")


if __name__ == "__main__":
    main()
