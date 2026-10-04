"""Offline robustness checks for the refined E1 S1 sweep.

The main purpose is to answer a reviewer concern: whether the finite-size
ordering depends on one particular five-seed draw. The script repeatedly
subsamples five seeds from completed E1 cells and recomputes alpha_c.
"""
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
from .plot_e1_refined import add_calibration, alpha_c_table
from .plot_final_figures import INK, MUTED, clean_axis, set_style


PAIR_TESTS = [
    (200, 500),
    (200, 1000),
    (200, 2000),
    (300, 1000),
    (300, 2000),
    (500, 2000),
]


def read_ok(out_name: str) -> pd.DataFrame:
    path = OUTPUTS / out_name / "data.csv"
    if not path.exists():
        raise SystemExit(f"Missing E1 data: {path}")
    frame = pd.read_csv(path)
    if "status" in frame.columns:
        frame = frame[frame["status"].fillna("ok").eq("ok")].copy()
    for column in frame.columns:
        if column in {"scenario", "variant", "defense", "status", "primary_metric", "target_asset", "placement"}:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def sample_alpha_c(frame: pd.DataFrame, rng: np.random.Generator, sample_size: int) -> dict[int, float]:
    pieces = []
    for n, group in frame.groupby("n_society", dropna=False):
        seeds = np.array(sorted(group["seed"].dropna().unique()), dtype=int)
        if len(seeds) <= sample_size:
            chosen = seeds
        else:
            chosen = rng.choice(seeds, size=sample_size, replace=False)
        pieces.append(group[group["seed"].isin(chosen)])
    sampled = pd.concat(pieces, ignore_index=True)
    ac = alpha_c_table(sampled)
    return {
        int(row.n_society): float(row.alpha_c)
        for row in ac.itertuples(index=False)
        if math.isfinite(float(row.alpha_c))
    }


def run_subsampling(frame: pd.DataFrame, draws: int, sample_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    records = []
    for draw in range(draws):
        values = sample_alpha_c(frame, rng, sample_size)
        row = {"draw": draw}
        for n, value in values.items():
            row[f"alpha_c_{n}"] = value
        row["coarse_order_200_300_gt_1000_2000"] = (
            min(values.get(200, -np.inf), values.get(300, -np.inf))
            > max(values.get(1000, np.inf), values.get(2000, np.inf))
        )
        records.append(row)
    draws_df = pd.DataFrame(records)
    rows = []
    for small, large in PAIR_TESTS:
        s = draws_df.get(f"alpha_c_{small}")
        l = draws_df.get(f"alpha_c_{large}")
        if s is None or l is None:
            continue
        ok = (s > l).dropna()
        rows.append({
            "comparison": f"alpha_c({small}) > alpha_c({large})",
            "small_n": small,
            "large_n": large,
            "success_rate": float(ok.mean()) if len(ok) else np.nan,
            "n_draws": int(len(ok)),
        })
    coarse = draws_df["coarse_order_200_300_gt_1000_2000"].dropna()
    rows.append({
        "comparison": "min alpha_c(200,300) > max alpha_c(1000,2000)",
        "small_n": "200/300",
        "large_n": "1000/2000",
        "success_rate": float(coarse.mean()) if len(coarse) else np.nan,
        "n_draws": int(len(coarse)),
    })
    summary = pd.DataFrame(rows)
    return draws_df, summary


def plot(summary: pd.DataFrame, out_name: str) -> None:
    data = summary[summary["comparison"].str.contains("> alpha_c", regex=False)].copy()
    if data.empty:
        return
    labels = [f"{int(row.small_n)}>{int(row.large_n)}" for row in data.itertuples(index=False)]
    values = data["success_rate"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(3.35, 1.85))
    ax.bar(np.arange(len(values)), values, color="#C026D3", width=0.64)
    ax.axhline(0.95, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=6.3)
    ax.set_ylabel("ordering stability")
    ax.set_title("Seed-subsampling robustness", loc="left", fontweight="bold", color=INK, pad=4, fontsize=8.5)
    ax.text(0.01, 0.05, "5-seed draws from completed E1 cells", transform=ax.transAxes, fontsize=5.8, color=MUTED, ha="left")
    ax.grid(True, axis="y", color="#E9D7F0", linewidth=0.7)
    clean_axis(ax)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.84, bottom=0.32)
    for directory in (ensure_dir(ROOT / "figures"), ensure_dir(ROOT.parent / "AuthorKit27" / "Figures"), ensure_dir(OUTPUTS / out_name / "figures")):
        fig.savefig(directory / "figure_appendix_e1_seed_subsampling.pdf")
        fig.savefig(directory / "figure_appendix_e1_seed_subsampling.png", dpi=420)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="real_e1_refined_s1")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2028)
    args = parser.parse_args()
    set_style()
    frame = add_calibration(read_ok(args.out), 1.0)
    main = frame[frame["n_society"].isin([200, 300, 500, 1000, 2000])].copy()
    draws, summary = run_subsampling(main, args.draws, args.sample_size, args.seed)
    out_dir = ensure_dir(OUTPUTS / args.out)
    draws.to_csv(out_dir / "e1_seed_subsampling_draws.csv", index=False)
    summary.to_csv(out_dir / "e1_seed_subsampling_summary.csv", index=False)
    plot(summary, args.out)
    print(f"Wrote seed-subsampling robustness to {out_dir}")


if __name__ == "__main__":
    main()
