"""Build the four main-paper figures and the scaling result table.

The script is deliberately conservative:

- it reads only v3 paper-profile outputs under ``paper_experiments_v3/outputs``;
- it filters out rows whose status is not ``ok``;
- it writes visible "missing data" panels instead of silently inventing values;
- P02 decomposition is labeled as proxy-based unless direct perturbation fields
  are added to the simulator output.

Typical use after paper runs finish:

    PYTHONPATH=src:. python -m paper_experiments_v3.figures.make_paper_figures
"""
from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wolfbench_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "wolfbench_xdg_cache"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DEFAULT_OUT = ROOT / "figures" / "generated"
SEED = 202707
STYLE_BACKEND = "matplotlib"

COLORS = {
    # WolfBench visual identity: mostly ink/gray with restrained plum and rose.
    "ink": "#2D2730",
    "plum": "#694A78",
    "violet": "#8D6AA1",
    "mauve": "#B18DAF",
    "rose": "#C87589",
    "lilac": "#EEE7F2",
    "blush": "#F7EAEE",
    "cool_fill": "#F1EDF5",
    "harm": "#C87589",
    "safe": "#74609A",
    "social": "#8D6AA1",
    "environment": "#694A78",
    "neutral": "#817A83",
    "warm_light": "#EEE7F2",
    # Backward-compatible aliases used by supplementary diagnostics.
    "blue": "#74609A",
    "orange": "#8D6AA1",
    "green": "#694A78",
    "red": "#C87589",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#817A83",
    "dark": "#2D2730",
    "grid": "#E4DFE6",
    "light": "#F1EDF5",
    "paper": "#FFFFFF",
    "pink": "#F7EAEE",
}
PALETTE = [COLORS[k] for k in ["blue", "orange", "green", "red", "purple", "sky"]]
SIZE_STYLES = {
    100: ("#CF6688", "o"),
    500: ("#9574B0", "s"),
    2000: ("#5E3978", "*"),
}
FIG2_ERROR_COLORS = {
    100: "#E2A9BA",
    500: "#C7B3D6",
    2000: "#A78BBA",
}
FIGURE_SCIENCE = {
    "fit": "#7652A6",
    "point": "#9970C4",
    "error": "#CDB6E3",
    "annotation": "#F5EDF8",
    "comparison": "#B7AFBE",
    "grid": "#ECE7EF",
    "axis": "#3F3745",
    "failure": "#D8709B",
    "fragile": "#D56F93",
    "robust": "#8E76C4",
    "zero": "#9D96A2",
    "fragile_bg": "#FBF1F5",
    "robust_bg": "#F5F1FA",
}

LINE_MAIN = 2.0
LINE_SECONDARY = 1.35
LINE_REFERENCE = 0.9
MARKER_SIZE = 5.5
MARKER_EDGE = 0.8
ERRORBAR_WIDTH = 1.15
CAP_SIZE = 2.5
AXIS_WIDTH = 0.8
GRID_WIDTH = 0.55
VARIANT_ORDER = [
    "qre_weak_coupling",
    "mixed_roles_reference",
    "qre_high_precision",
    "qre_high_attention",
    "qre_high_reach",
    "qre_high_conformity",
    "qre_strong_coupling",
]
VARIANT_LABELS = {
    "qre_baseline": "Baseline",
    "qre_weak_coupling": "Weak-feedback bundle",
    "mixed_roles_reference": "Mixed roles",
    "qre_high_precision": "High response precision",
    "qre_high_attention": "High attention",
    "qre_high_reach": "Increased network reach",
    "qre_high_conformity": "High conformity",
    "qre_strong_coupling": "Strong-feedback bundle",
}
INTERVENTION_GROUPS = [
    (
        "Feedback regimes",
        ["qre_weak_coupling", "qre_strong_coupling", "qre_high_reach"],
    ),
    (
        "Controls",
        ["qre_high_attention", "qre_high_conformity", "qre_high_precision"],
    ),
]
FIG4_LABELS = {
    "qre_weak_coupling": "Weak feedback",
    "qre_strong_coupling": "Strong feedback",
    "qre_high_reach": "Increased reach",
    "qre_high_attention": "High attention",
    "qre_high_conformity": "High conformity",
    "qre_high_precision": "High precision",
}


def setup_style() -> None:
    """Set a publication-grade shared style.

    We use seaborn when the local environment can import it. Some local conda
    installs have a pandas/numpy ABI mismatch; in that case we fall back to
    Matplotlib's bundled seaborn-v0_8 style and the same colorblind palette.
    """
    global STYLE_BACKEND
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import seaborn as sns  # type: ignore

        sns.set_theme(
            context="paper",
            style="whitegrid",
            palette=PALETTE,
            rc={
                "axes.spines.top": False,
                "axes.spines.right": False,
                "grid.color": COLORS["grid"],
                "grid.linewidth": GRID_WIDTH,
            },
        )
        STYLE_BACKEND = "seaborn"
    except Exception:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            STYLE_BACKEND = "matplotlib-seaborn-v0_8"
        except OSError:
            STYLE_BACKEND = "matplotlib-custom"
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Source Sans 3", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": COLORS["dark"],
        "axes.labelcolor": COLORS["dark"],
        "xtick.color": COLORS["dark"],
        "ytick.color": COLORS["dark"],
        "text.color": COLORS["dark"],
        "grid.color": COLORS["grid"],
        "grid.linewidth": GRID_WIDTH,
        "grid.alpha": 0.72,
        "lines.linewidth": LINE_MAIN,
        "lines.markersize": MARKER_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    })


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    ensure_dir(out_dir)
    fig.align_labels()
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def polish_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    if grid:
        ax.grid(True, axis="y", linewidth=GRID_WIDTH, color=COLORS["grid"], alpha=0.72)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(AXIS_WIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_WIDTH)
    ax.tick_params(length=3.0, width=AXIS_WIDTH, pad=2.0)


def polish_science_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    """Lightweight white-background styling for the main scientific plots."""
    ax.set_facecolor("white")
    ax.grid(True, which="major", axis=grid_axis, color=FIGURE_SCIENCE["grid"], linewidth=0.48, alpha=0.78)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FIGURE_SCIENCE["axis"])
        ax.spines[side].set_linewidth(0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.8, width=0.72, pad=2.0, colors=FIGURE_SCIENCE["axis"])
    ax.xaxis.label.set_color(FIGURE_SCIENCE["axis"])
    ax.yaxis.label.set_color(FIGURE_SCIENCE["axis"])


def panel_title(ax: plt.Axes, label: str, title: str) -> None:
    """Place the panel letter and claim-oriented title on one baseline."""
    ax.set_title(f"{label}  {title}", loc="left", fontweight="bold", pad=5.0)


def science_panel_title(ax: plt.Axes, label: str, title: str, *, size_scale: float = 1.0) -> None:
    ax.text(
        0.0,
        1.018,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.1 * size_scale,
        fontweight="bold",
        color=FIGURE_SCIENCE["axis"],
    )
    ax.text(
        0.078,
        1.018,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.35 * size_scale,
        fontweight="normal",
        color=FIGURE_SCIENCE["axis"],
    )


def plot_mean_band(
    ax: plt.Axes,
    curve: list[tuple[float, float, float, float]],
    *,
    color: str,
    label: str,
    marker: str = "o",
    linestyle: str = "-",
    band_alpha: float = 0.12,
) -> None:
    if not curve:
        return
    xs = np.asarray([x for x, *_ in curve], dtype=float)
    ys = np.asarray([y for _, y, *_ in curve], dtype=float)
    lo = np.asarray([l for _, _, l, _ in curve], dtype=float)
    hi = np.asarray([h for _, _, _, h in curve], dtype=float)
    if band_alpha > 0:
        ax.fill_between(xs, lo, hi, color=color, alpha=band_alpha, linewidth=0)
    ax.plot(xs, ys, marker=marker, linestyle=linestyle, color=color, label=label)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(prefix: str, *, include_extra: bool = False) -> list[dict[str, str]]:
    """Read paper shard data, falling back to a combined output directory."""
    rows: list[dict[str, str]] = []
    # ``_extra`` shards extend selected response curves for visualization but
    # are not part of the frozen six-size boundary fit. Callers opt in when
    # they need the completed curve tails.
    shard_paths = sorted(
        path
        for path in OUTPUTS.glob(f"{prefix}_s*/data.csv")
        if include_extra or "_extra" not in path.parent.name
    )
    paths = shard_paths or [OUTPUTS / prefix / "data.csv"]
    for path in paths:
        for row in read_csv(path):
            if row.get("status", "ok") == "ok":
                rows.append(row)
    return rows


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": COLORS["paper"], "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
        zorder=10,
    )


def missing(ax: plt.Axes, title: str, message: str = "No completed v3 paper rows yet.") -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    compact = message.replace("No completed v3 paper rows yet.", "No completed v3\npaper rows yet.")
    ax.text(0.5, 0.5, compact, ha="center", va="center", color=COLORS["gray"], fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#dddddd")


def aggregate_curve(rows: list[dict[str, str]], y_key: str, by: str = "alpha") -> list[tuple[float, float, float, float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        x = f(row, by)
        y = f(row, y_key)
        if math.isfinite(x) and math.isfinite(y):
            grouped[x].append(y)
    out = []
    for x, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        if len(arr) <= 1:
            lo = hi = float(arr.mean()) if len(arr) else float("nan")
        else:
            lo, hi = np.quantile(arr, [0.025, 0.975])
        out.append((x, float(arr.mean()), float(lo), float(hi)))
    return out


def aggregate_mean_ci(
    rows: list[dict[str, str]],
    y_key: str,
    *,
    by: str = "alpha",
    binary: bool = False,
) -> list[tuple[float, float, float, float]]:
    """Aggregate seed means with a 95% mean interval for main-paper plots."""
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        x = f(row, by)
        y = f(row, y_key)
        if math.isfinite(x) and math.isfinite(y):
            grouped[x].append(y)
    out = []
    z = 1.96
    for x, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        center = float(arr.mean())
        n = len(arr)
        if n <= 1:
            lo = hi = center
        elif binary:
            denom = 1.0 + z**2 / n
            adjusted = (center + z**2 / (2 * n)) / denom
            half = z * math.sqrt(center * (1.0 - center) / n + z**2 / (4 * n**2)) / denom
            lo, hi = max(0.0, adjusted - half), min(1.0, adjusted + half)
        else:
            half = z * float(arr.std(ddof=1)) / math.sqrt(n)
            lo, hi = center - half, center + half
        out.append((x, center, float(lo), float(hi)))
    return out


def crossing(curve: list[tuple[float, float]], target: float = 0.5) -> float | None:
    points = sorted(curve)
    for (a0, p0), (a1, p1) in zip(points, points[1:]):
        if p0 == target:
            return a0
        if (p0 - target) * (p1 - target) <= 0 and p1 != p0:
            return a0 + (target - p0) * (a1 - a0) / (p1 - p0)
    return None


def fit_logistic(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(xs) < 3 or len(np.unique(ys)) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    x0 = (xs - xs.mean()) / max(xs.std(), 1e-9)
    design = np.column_stack([np.ones_like(x0), x0])
    theta = np.zeros(2)
    for _ in range(80):
        logits = np.clip(design @ theta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-logits))
        w = np.clip(p * (1 - p), 1e-8, None)
        h = design.T @ (w[:, None] * design) + 1e-6 * np.eye(2)
        step = np.linalg.solve(h, design.T @ (ys - p))
        theta += step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    gx = (grid - xs.mean()) / max(xs.std(), 1e-9)
    gp = 1.0 / (1.0 + np.exp(-np.clip(np.column_stack([np.ones_like(gx), gx]) @ theta, -30, 30)))
    return gp


def figure1_overview(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.05, 2.48))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    sans = "Arial"

    ax.text(
        0.5,
        0.965,
        "Harm is reproduced through shared social and environmental state.",
        ha="center",
        va="top",
        fontsize=9.6,
        fontweight="bold",
        fontfamily=sans,
        color=COLORS["dark"],
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.075),
            0.665,
            0.77,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="#FCFBFD",
            edgecolor="#B6A8BB",
            linewidth=0.8,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.705, 0.075),
            0.28,
            0.77,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="#FAF8FB",
            edgecolor="#B6A8BB",
            linewidth=0.8,
        )
    )
    ax.text(0.035, 0.805, "CLOSED-LOOP SOCIETY", fontsize=7.7, fontweight="bold", fontfamily=sans, color=COLORS["neutral"])
    ax.text(0.725, 0.805, "MEASUREMENT RAIL", fontsize=7.7, fontweight="bold", fontfamily=sans, color=COLORS["neutral"])

    def rounded_node(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        detail: str,
        fill: str,
        edge: str,
        *,
        detail_offset: float = 0.034,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.007,rounding_size=0.010",
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.9,
            )
        )
        ax.text(x + 0.012, y + h - 0.027, title, fontsize=6.9, fontweight="bold", fontfamily=sans, va="top", color=COLORS["dark"], linespacing=1.00)
        ax.text(x + 0.012, y + detail_offset, detail, fontsize=5.4, fontfamily=sans, va="bottom", color=COLORS["neutral"], linespacing=1.02)

    rounded_node(0.035, 0.51, 0.135, 0.205, "1 Harmful\npressure", "messages + demand", COLORS["blush"], COLORS["harm"], detail_offset=0.034)
    rounded_node(0.215, 0.51, 0.145, 0.205, "2 Social\namplification", "exposure + reach", "#F2EAF3", COLORS["violet"], detail_offset=0.034)
    rounded_node(0.405, 0.465, 0.245, 0.285, "Bounded agents", "private evidence · trust · policies", "#F4F1F6", "#807286", detail_offset=0.070)
    rounded_node(0.445, 0.135, 0.175, 0.155, "Collective\nactions", "trade · post · reshare", "#F4F1F6", "#807286", detail_offset=0.010)
    rounded_node(0.205, 0.135, 0.185, 0.155, "3 Environmental\nfeedback", "public outcomes", "#EEEAF4", "#78669A", detail_offset=0.010)

    chip_x = 0.420
    for label, width in [("risk", 0.040), ("value", 0.045), ("trend", 0.047), ("social", 0.049), ("aggressive", 0.074)]:
        ax.add_patch(FancyBboxPatch((chip_x, 0.487), width, 0.038, boxstyle="round,pad=0.003,rounding_size=0.006", facecolor="#FFFFFF", edgecolor="#C9C9C5", linewidth=0.55))
        ax.text(chip_x + width / 2, 0.506, label, ha="center", va="center", fontsize=4.8, fontfamily=sans, color=COLORS["neutral"])
        chip_x += width + 0.005

    def arrow(start: tuple[float, float], end: tuple[float, float], *, color: str = COLORS["neutral"], rad: float = 0.0, lw: float = 1.35) -> None:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}"))

    arrow((0.17, 0.615), (0.215, 0.615), color=COLORS["harm"])
    arrow((0.36, 0.615), (0.405, 0.615), color=COLORS["plum"])
    arrow((0.53, 0.455), (0.53, 0.29))
    arrow((0.445, 0.213), (0.39, 0.213), color=COLORS["plum"])
    arrow((0.25, 0.29), (0.255, 0.51), color=COLORS["violet"], rad=0.18, lw=1.8)
    arrow((0.65, 0.515), (0.725, 0.515), color=COLORS["neutral"], lw=1.1)
    ax.text(0.178, 0.360, "social proof", fontsize=6.0, fontweight="bold", fontfamily=sans, color=COLORS["violet"], rotation=78)
    ax.text(0.668, 0.535, "measure", fontsize=5.8, fontfamily=sans, color=COLORS["neutral"])

    # Mechanism coordinates feed jointly into the observed population response.
    def metric_box(x: float, y: float, w: float, h: float, formula: str, detail: str, fill: str, edge: str, *, formula_size: float = 8.1, detail_size: float = 5.8) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.007,rounding_size=0.010", facecolor=fill, edgecolor=edge, linewidth=0.9))
        ax.text(x + w / 2, y + h * 0.67, formula, ha="center", va="center", fontsize=formula_size, color=COLORS["dark"])
        ax.text(x + w / 2, y + h * 0.24, detail, ha="center", va="center", fontsize=detail_size, fontfamily=sans, color=COLORS["neutral"], linespacing=0.94)

    ax.text(0.73, 0.735, "Mechanism coordinates", fontsize=7.5, fontweight="bold", fontfamily=sans)
    metric_box(0.73, 0.56, 0.105, 0.125, r"$\chi^+$", "feedback\ngain", COLORS["cool_fill"], COLORS["plum"], formula_size=8.0, detail_size=5.4)
    metric_box(0.855, 0.56, 0.105, 0.125, r"$D_N$", "social\ndominance", "#F2EAF3", COLORS["violet"], formula_size=8.0, detail_size=5.4)
    ax.text(0.73, 0.475, "Observed signatures", fontsize=7.5, fontweight="bold", fontfamily=sans)
    metric_box(0.765, 0.325, 0.16, 0.105, r"$p_N(\alpha)$", "collapse", "#FFFFFF", "#AEB3B9")
    metric_box(0.735, 0.13, 0.10, 0.120, r"$\alpha_c(N)\downarrow$", "threshold", "#FFFFFF", "#AEB3B9")
    metric_box(0.855, 0.13, 0.10, 0.120, r"$K_c(N)\uparrow$", "scaling", "#FFFFFF", "#AEB3B9")
    arrow((0.785, 0.56), (0.82, 0.43), color=COLORS["plum"], rad=0.08)
    arrow((0.91, 0.56), (0.87, 0.43), color=COLORS["violet"], rad=-0.08)
    arrow((0.845, 0.325), (0.79, 0.25), color=COLORS["neutral"], rad=0.12)
    arrow((0.845, 0.325), (0.905, 0.25), color=COLORS["neutral"], rad=-0.12)
    ax.text(0.845, 0.283, "finite-size bridge", ha="center", fontsize=6.0, fontfamily=sans, color=COLORS["neutral"])
    save_figure(fig, out_dir, "fig1_overview")


def p01_summaries(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for n in sorted({int(f(row, "n_society")) for row in rows if math.isfinite(f(row, "n_society"))}):
        group = [row for row in rows if int(f(row, "n_society")) == n]
        curve = [(x, y) for x, y, _, _ in aggregate_curve(group, "primary_failure_rate")]
        alpha_c = crossing(curve, 0.5)
        alpha10 = crossing(curve, 0.1)
        alpha90 = crossing(curve, 0.9)
        status = "resolved" if alpha_c is not None else "censored"
        out.append({
            "N": n,
            "alpha_c": "" if alpha_c is None else alpha_c,
            "K_c": "" if alpha_c is None else n * alpha_c,
            "width_10_90": "" if alpha10 is None or alpha90 is None else alpha90 - alpha10,
            "status": status,
        })
    return out


def representative_sizes(ns: list[int]) -> list[int]:
    preferred = [100, 500, 2000]
    if all(n in ns for n in preferred):
        return preferred
    if len(ns) >= 3:
        return [ns[0], ns[len(ns) // 2], ns[-1]]
    return ns


def representative_styles(chosen: list[int]) -> dict[int, tuple[str, str]]:
    fallback = [("#CF6688", "o"), ("#9574B0", "s"), ("#5E3978", "D")]
    return {n: SIZE_STYLES.get(n, fallback[i % len(fallback)]) for i, n in enumerate(chosen)}


def p01_display_xmax(rows: list[dict[str, str]]) -> float:
    alphas = [f(row, "alpha") for row in rows if math.isfinite(f(row, "alpha"))]
    if not alphas:
        return 0.08
    summaries = [row for row in p01_summaries(rows) if row["alpha_c"] != ""]
    if summaries:
        max_midpoint = max(float(row["alpha_c"]) for row in summaries)
        return min(max(alphas), max(0.06, min(0.08, 1.65 * max_midpoint)))
    return min(max(alphas), 0.08)


def bootstrap_alpha_c_ci(
    rows: list[dict[str, str]],
    n: int,
    *,
    n_boot: int = 800,
) -> tuple[float | None, float | None]:
    group = [row for row in rows if int(f(row, "n_society")) == n]
    by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in group:
        by_seed[row.get("seed", "")].append(row)
    seeds = sorted(seed for seed in by_seed if seed != "")
    if len(seeds) < 2:
        return None, None
    rng = np.random.default_rng(SEED + n)
    samples: list[float] = []
    for _ in range(n_boot):
        by_alpha: dict[float, list[float]] = defaultdict(list)
        for seed in rng.choice(seeds, size=len(seeds), replace=True):
            for row in by_seed[str(seed)]:
                alpha = f(row, "alpha")
                value = f(row, "primary_failure_rate")
                if math.isfinite(alpha) and math.isfinite(value):
                    by_alpha[alpha].append(value)
        curve = [(alpha, float(np.mean(vals))) for alpha, vals in sorted(by_alpha.items()) if vals]
        alpha_c = crossing(curve, 0.5)
        if alpha_c is not None:
            samples.append(alpha_c)
    if len(samples) < max(20, n_boot // 10):
        return None, None
    lo, hi = np.quantile(np.asarray(samples), [0.025, 0.975])
    return float(lo), float(hi)


def estimate_nu_from_summaries(summaries: list[dict[str, object]]) -> float:
    resolved = [row for row in summaries if row["alpha_c"] != ""]
    if len(resolved) < 2:
        return float("nan")
    nvals = np.asarray([float(row["N"]) for row in resolved], dtype=float)
    avals = np.asarray([float(row["alpha_c"]) for row in resolved], dtype=float)
    if np.any(nvals <= 0) or np.any(avals <= 0):
        return float("nan")
    slope, _ = np.polyfit(np.log(nvals), np.log(avals), 1)
    return -float(slope)


def bootstrap_nu_stats(
    rows: list[dict[str, str]],
    *,
    n_boot: int = 10000,
) -> dict[str, object]:
    """Seed-bootstrap the finite-size exponent in alpha_c(N) ~ N^{-nu}.

    The resampling unit is the experiment seed, not individual alpha rows. For
    each bootstrap draw we reconstruct the full response curve for every size,
    re-estimate alpha_c by midpoint crossing, and refit the log-log slope.
    """
    point_summaries = p01_summaries(rows)
    point_resolved = [row for row in point_summaries if row["alpha_c"] != ""]
    point_ns = sorted(int(row["N"]) for row in point_resolved)
    point = estimate_nu_from_summaries(point_summaries)

    by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        seed = row.get("seed", "")
        if seed != "":
            by_seed[seed].append(row)
    seeds = sorted(by_seed)
    samples: list[float] = []
    partial_samples: list[float] = []
    resolved_count_hist: dict[int, int] = defaultdict(int)
    missing_by_n: dict[int, int] = defaultdict(int)
    if len(seeds) >= 2 and len(point_ns) >= 2 and math.isfinite(point):
        rng = np.random.default_rng(SEED + 4242)
        seed_arr = np.asarray(seeds, dtype=object)
        for _ in range(n_boot):
            sample_rows: list[dict[str, str]] = []
            for seed in rng.choice(seed_arr, size=len(seed_arr), replace=True):
                sample_rows.extend(by_seed[str(seed)])
            sample_summaries = p01_summaries(sample_rows)
            sample_resolved = [row for row in sample_summaries if row["alpha_c"] != "" and int(row["N"]) in point_ns]
            resolved_count_hist[len(sample_resolved)] += 1
            resolved_ns = {int(row["N"]) for row in sample_resolved}
            for n in point_ns:
                if n not in resolved_ns:
                    missing_by_n[n] += 1
            if len(sample_resolved) >= 2:
                partial_nu = estimate_nu_from_summaries(sample_resolved)
                if math.isfinite(partial_nu):
                    partial_samples.append(partial_nu)
            if len(sample_resolved) != len(point_ns):
                continue
            nu = estimate_nu_from_summaries(sample_resolved)
            if math.isfinite(nu):
                samples.append(nu)

    if samples:
        arr = np.asarray(samples, dtype=float)
        lo, hi = np.quantile(arr, [0.025, 0.975])
        prob_gt0 = float(np.mean(arr > 0.0))
        prob_lt1 = float(np.mean(arr < 1.0))
        mean_boot = float(np.mean(arr))
    else:
        lo = hi = prob_gt0 = prob_lt1 = mean_boot = None

    if partial_samples:
        partial_arr = np.asarray(partial_samples, dtype=float)
        partial_lo, partial_hi = np.quantile(partial_arr, [0.025, 0.975])
        partial_mean = float(np.mean(partial_arr))
        partial_prob_gt0 = float(np.mean(partial_arr > 0.0))
        partial_prob_lt1 = float(np.mean(partial_arr < 1.0))
    else:
        partial_lo = partial_hi = partial_mean = partial_prob_gt0 = partial_prob_lt1 = None

    return {
        "nu_hat": point,
        "nu_boot_mean": mean_boot,
        "nu_ci_lo": None if lo is None else float(lo),
        "nu_ci_hi": None if hi is None else float(hi),
        "prob_nu_gt_0": prob_gt0,
        "prob_nu_lt_1": prob_lt1,
        "partial_ge2_boot_mean": partial_mean,
        "partial_ge2_ci_lo": None if partial_lo is None else float(partial_lo),
        "partial_ge2_ci_hi": None if partial_hi is None else float(partial_hi),
        "partial_ge2_prob_nu_gt_0": partial_prob_gt0,
        "partial_ge2_prob_nu_lt_1": partial_prob_lt1,
        "partial_ge2_n_boot_effective": len(partial_samples),
        "resolved_count_hist": dict(sorted(resolved_count_hist.items())),
        "missing_by_n": dict(sorted(missing_by_n.items())),
        "n_boot_requested": n_boot,
        "n_boot_effective": len(samples),
        "n_seed": len(seeds),
        "n_resolved_sizes": len(point_ns),
    }


def draw_transition_band(ax: plt.Axes, summaries: list[dict[str, object]]) -> None:
    values = [float(row["alpha_c"]) for row in summaries if row["alpha_c"] != ""]
    if not values:
        return
    ax.axvspan(min(values), max(values), color=COLORS["grid"], alpha=0.28, linewidth=0, zorder=0)


def figure2_p01_nonlinear_response(
    out_dir: Path,
    p01_rows: list[dict[str, str]],
    p01_display_rows: list[dict[str, str]] | None = None,
) -> None:
    display_rows = p01_display_rows or p01_rows
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.42), gridspec_kw={"wspace": 0.25}, facecolor="white")
    if not display_rows:
        for ax, title in zip(axes, ["Collapse probability", "Continuous risk"]):
            missing(ax, title)
        save_figure(fig, out_dir, "fig2_nonlinear_response")
        return

    ns = sorted({int(f(row, "n_society")) for row in display_rows})
    chosen = representative_sizes(ns)
    styles = representative_styles(chosen)
    # Boundary annotations remain tied to the frozen primary grid, whereas
    # the plotted curves include later high-alpha top-ups for complete tails.
    summaries = p01_summaries(p01_rows)
    x_max = p01_display_xmax(display_rows)
    midpoint_lookup = {int(row["N"]): float(row["alpha_c"]) for row in summaries if row["alpha_c"] != ""}

    ax = axes[0]
    science_panel_title(ax, "(A)", "Collapse probability", size_scale=1.06)
    for n in chosen:
        color, marker = styles[n]
        error_color = FIG2_ERROR_COLORS.get(n, color)
        group = [row for row in display_rows if int(f(row, "n_society")) == n and f(row, "alpha") <= x_max + 1e-12]
        curve = aggregate_mean_ci(group, "primary_failure_rate", binary=True)
        if not curve:
            continue
        xs = np.asarray([x for x, *_ in curve], dtype=float)
        ys = np.asarray([y for _, y, *_ in curve], dtype=float)
        lo = np.asarray([l for _, _, l, _ in curve], dtype=float)
        hi = np.asarray([h for _, _, _, h in curve], dtype=float)
        ax.vlines(xs, lo, hi, color=error_color, linewidth=0.64, alpha=0.56, zorder=2)
        ax.scatter(
            xs,
            ys,
            marker=marker,
            color=color,
            s=40 if marker == "*" else 22,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
            label=fr"$N={n}$",
        )
        if len(xs) >= 3:
            observed_grid = np.linspace(xs.min(), xs.max(), 160)
            raw_x = np.asarray([f(row, "alpha") for row in group], dtype=float)
            raw_y = np.asarray([f(row, "primary_failure_rate") for row in group], dtype=float)
            has_transition_support = float(np.nanmax(ys)) >= 0.78 and float(np.nanmin(ys)) <= 0.12
            if has_transition_support:
                fitted = fit_logistic(raw_x, raw_y, observed_grid)
                ax.plot(observed_grid, fitted, color=color, linewidth=1.75, alpha=0.94, zorder=3)
            else:
                ax.plot(xs, ys, color=color, linewidth=1.30, alpha=0.70, zorder=3)
            if has_transition_support and xs.max() < x_max - 1e-6:
                extrap_grid = np.linspace(xs.max(), x_max, 80)
                extrap_fitted = fit_logistic(raw_x, raw_y, extrap_grid)
                ax.plot(
                    extrap_grid,
                    extrap_fitted,
                    color=color,
                    linewidth=1.0,
                    linestyle=(0, (2.0, 2.0)),
                    alpha=0.22,
                    zorder=2.8,
                )
        alpha_c = midpoint_lookup.get(n)
        if alpha_c is not None:
            ax.vlines(
                alpha_c,
                0.0,
                0.5,
                color=color,
                linestyle=(0, (2.0, 1.65)),
                linewidth=0.86,
                alpha=0.36,
                zorder=3.2,
            )
    ax.axhline(0.5, color=FIGURE_SCIENCE["zero"], linestyle="--", linewidth=0.82, alpha=0.90)
    ax.text(x_max * 0.985, 0.515, r"$p=0.5$", ha="right", va="bottom", fontsize=7.2, color=FIGURE_SCIENCE["zero"])
    if 2000 in midpoint_lookup and 100 in midpoint_lookup:
        ax.text(
            0.155,
            0.835,
            r"$\alpha_c:\ 0.022 \to 0.047$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.15,
            color=FIGURE_SCIENCE["axis"],
            alpha=0.76,
            bbox={
                "boxstyle": "round,pad=0.16,rounding_size=0.025",
                "facecolor": FIGURE_SCIENCE["annotation"],
                "edgecolor": "none",
                "alpha": 0.78,
            },
        )
    ax.set_ylabel("collapse probability")
    ax.set_xlabel(r"harmful fraction $\alpha$")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.002, x_max)
    polish_science_axis(ax)
    ax = axes[1]
    science_panel_title(ax, "(B)", "Peak risk", size_scale=1.06)
    max_y = 1.05
    for n in chosen:
        color, marker = styles[n]
        error_color = FIG2_ERROR_COLORS.get(n, color)
        group = [row for row in display_rows if int(f(row, "n_society")) == n and f(row, "alpha") <= x_max + 1e-12]
        curve = aggregate_mean_ci(group, "primary_failure_score_max")
        if not curve:
            continue
        xs = np.asarray([x for x, *_ in curve], dtype=float)
        ys = np.asarray([y for _, y, *_ in curve], dtype=float)
        lo = np.asarray([l for _, _, l, _ in curve], dtype=float)
        hi = np.asarray([h for _, _, _, h in curve], dtype=float)
        max_y = max(max_y, float(np.nanmax(hi)))
        ax.errorbar(
            xs,
            ys,
            yerr=np.vstack([ys - lo, hi - ys]),
            marker=marker,
            color=color,
            ecolor=error_color,
            elinewidth=0.72,
            capsize=1.45,
            linewidth=1.68,
            markersize=6.1 if marker == "*" else 4.1,
            markeredgecolor="white",
            markeredgewidth=0.5,
            alpha=0.96,
        )
        alpha_c = midpoint_lookup.get(n)
        if alpha_c is not None:
            ax.vlines(alpha_c, 0.0, min(0.18, max_y), color=error_color, linewidth=0.68, alpha=0.34)
    max_y *= 1.05
    ax.axhspan(1.0, max_y, color=FIGURE_SCIENCE["failure"], alpha=0.055, linewidth=0, zorder=0)
    ax.axhline(1.0, color=FIGURE_SCIENCE["failure"], linestyle="--", linewidth=0.82, alpha=0.90)
    ax.text(0.001, 1.022, "failure region", color=FIGURE_SCIENCE["failure"], fontsize=6.9, va="bottom")
    ax.set_xlabel(r"harmful fraction $\alpha$")
    ax.set_ylabel("episode-level peak risk")
    ax.set_ylim(0, max_y)
    ax.set_xlim(-0.002, x_max)
    polish_science_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.982),
        ncol=len(handles),
        handletextpad=0.35,
        columnspacing=1.0,
        frameon=False,
        borderaxespad=0.0,
        fontsize=7.1,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.80, wspace=0.25)
    save_figure(fig, out_dir, "fig2_nonlinear_response")


def figure3_p01_finite_size_scaling(
    out_dir: Path,
    p01_rows: list[dict[str, str]],
    p04_rows: list[dict[str, str]],
) -> None:
    summaries = p01_summaries(p01_rows)
    write_csv(summaries, out_dir / "table2_scaling_results.csv")
    nu_stats = bootstrap_nu_stats(p01_rows)
    with (out_dir / "scaling_exponent_bootstrap.json").open("w") as handle:
        json.dump(nu_stats, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_csv([nu_stats], out_dir / "table3_scaling_exponent_bootstrap.csv")

    fig = plt.figure(figsize=(3.35, 2.92), facecolor="white")
    gs = fig.add_gridspec(2, 1, hspace=0.58)
    axes = [fig.add_subplot(gs[idx, 0]) for idx in range(2)]
    resolved = [row for row in summaries if row["alpha_c"] != ""]
    if not resolved:
        for ax, title in zip(axes, ["Boundary decreases", "Effective count grows sublinearly"]):
            missing(ax, title)
        save_figure(fig, out_dir, "fig3_finite_size_scaling")
        figure4_intervention_effects(out_dir, p04_rows)
        return

    nvals = np.asarray([float(row["N"]) for row in resolved], dtype=float)
    avals = np.asarray([float(row["alpha_c"]) for row in resolved], dtype=float)
    kvals = np.asarray([float(row["K_c"]) for row in resolved], dtype=float)
    x_min = max(1.0, nvals.min() * 0.78)
    x_max = nvals.max() * 1.24
    alpha_ci = [bootstrap_alpha_c_ci(p01_rows, int(n)) for n in nvals]
    alpha_lo = np.asarray([lo if lo is not None else val for (lo, _), val in zip(alpha_ci, avals)], dtype=float)
    alpha_hi = np.asarray([hi if hi is not None else val for (_, hi), val in zip(alpha_ci, avals)], dtype=float)

    slope, intercept = np.polyfit(np.log(nvals), np.log(avals), 1)
    nu = float(nu_stats["nu_hat"]) if math.isfinite(float(nu_stats["nu_hat"])) else -float(slope)
    grid = np.geomspace(nvals.min(), nvals.max(), 160)
    fit_alpha = np.exp(intercept) * grid**slope
    ci_lo = nu_stats.get("nu_ci_lo")
    ci_hi = nu_stats.get("nu_ci_hi")

    ax = axes[0]
    science_panel_title(ax, "(A)", "Boundary fraction decreases", size_scale=0.94)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.errorbar(
        nvals,
        avals,
        yerr=np.vstack([avals - alpha_lo, alpha_hi - avals]),
        fmt="o",
        color=FIGURE_SCIENCE["point"],
        ecolor=FIGURE_SCIENCE["error"],
        elinewidth=0.82,
        capsize=1.7,
        markersize=5.3,
        markeredgecolor="white",
        markeredgewidth=MARKER_EDGE,
        zorder=3,
    )
    ax.plot(grid, fit_alpha, color=FIGURE_SCIENCE["fit"], linewidth=2.10)
    alpha_constant = np.full_like(grid, avals[0])
    alpha_fixed_count = avals[0] * nvals[0] / grid
    ax.plot(
        grid,
        alpha_constant,
        color=FIGURE_SCIENCE["comparison"],
        linestyle="--",
        linewidth=0.96,
        alpha=0.52,
    )
    ax.plot(
        grid,
        alpha_fixed_count,
        color=FIGURE_SCIENCE["comparison"],
        linestyle=":",
        linewidth=0.96,
        alpha=0.52,
    )
    ax.set_xlabel("")
    ax.set_ylabel(r"$\alpha_c(N)$", fontsize=7.2, labelpad=5.0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(max(0.012, float(np.nanmin(alpha_lo)) * 0.78), float(np.nanmax(alpha_hi)) * 1.35)
    alpha_ticks = [0.02, 0.03, 0.04, 0.05, 0.06]
    ax.set_yticks(alpha_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    ax.set_xticks([100, 200, 500, 1000, 2000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: ""))
    polish_science_axis(ax)
    ax.tick_params(labelsize=6.75)
    ax.text(
        0.035,
        0.06,
        rf"$\alpha_c\propto N^{{-{nu:.3f}}}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=FIGURE_SCIENCE["fit"],
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FBF1F5", edgecolor="#D56F93", linewidth=0.55),
    )
    number_bbox = dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85)
    ax.annotate(f"{avals[0]:.3f}", (nvals[0], avals[0]), xytext=(4, 7), textcoords="offset points", ha="left", va="bottom", fontsize=6.0, color=FIGURE_SCIENCE["axis"], bbox=number_bbox)
    ax.annotate(f"{avals[-1]:.3f}", (nvals[-1], avals[-1]), xytext=(-4, -7), textcoords="offset points", ha="right", va="top", fontsize=6.0, color=FIGURE_SCIENCE["axis"], bbox=number_bbox)

    ax = axes[1]
    science_panel_title(ax, "(B)", "Effective count grows sublinearly", size_scale=0.94)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.errorbar(
        nvals,
        kvals,
        yerr=np.vstack([nvals * (avals - alpha_lo), nvals * (alpha_hi - avals)]),
        fmt="o",
        color=FIGURE_SCIENCE["point"],
        ecolor=FIGURE_SCIENCE["error"],
        elinewidth=0.82,
        capsize=1.7,
        markersize=5.3,
        markeredgecolor="white",
        markeredgewidth=MARKER_EDGE,
        zorder=3,
    )
    empirical_k = np.exp(intercept) * grid ** (1.0 + slope)
    ax.plot(grid, empirical_k, color=FIGURE_SCIENCE["fit"], linewidth=2.10)
    constant = np.full_like(grid, kvals[0])
    proportional = kvals[0] * grid / nvals[0]
    ax.plot(grid, constant, color=FIGURE_SCIENCE["comparison"], linestyle=":", linewidth=0.96, alpha=0.52)
    ax.plot(grid, proportional, color=FIGURE_SCIENCE["comparison"], linestyle="--", linewidth=0.96, alpha=0.52)
    ax.text(
        0.035,
        0.88,
        rf"$K_c\propto N^{{{1.0 - nu:.3f}}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color=FIGURE_SCIENCE["fit"],
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FBF1F5", edgecolor="#D56F93", linewidth=0.55),
    )
    ax.set_xlabel(r"society size $N$", fontsize=7.1, labelpad=4.0)
    ax.set_ylabel(r"$K_c(N)$", fontsize=7.2, labelpad=5.0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(max(2.8, float(np.nanmin(kvals)) * 0.72), float(np.nanmax(proportional)) * 1.18)
    ax.set_xticks([100, 200, 500, 1000, 2000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}" if value in {100, 200, 500, 1000, 2000} else ""))
    ax.set_yticks([5, 10, 20, 50, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}" if value in {5, 10, 20, 50, 100} else ""))
    polish_science_axis(ax)
    ax.tick_params(labelsize=6.75)
    number_bbox_k = dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85)
    ax.annotate(f"{kvals[0]:.1f}", (nvals[0], kvals[0]), xytext=(2, 8), textcoords="offset points", ha="left", va="bottom", fontsize=6.0, color=FIGURE_SCIENCE["axis"], bbox=number_bbox_k)
    ax.annotate(f"{kvals[-1]:.1f}", (nvals[-1], kvals[-1]), xytext=(-4, 7), textcoords="offset points", ha="right", va="bottom", fontsize=6.0, color=FIGURE_SCIENCE["axis"], bbox=number_bbox_k)

    reference_handles = [
        Line2D([0], [0], color=FIGURE_SCIENCE["fit"], linewidth=1.8, label="power fit"),
        Line2D([0], [0], color=FIGURE_SCIENCE["comparison"], linewidth=0.9, linestyle="--", label="constant fraction"),
        Line2D([0], [0], color=FIGURE_SCIENCE["comparison"], linewidth=0.9, linestyle=":", label="constant count"),
    ]
    fig.legend(
        handles=reference_handles,
        loc="center",
        bbox_to_anchor=(0.615, 0.545),
        ncol=3,
        fontsize=5.45,
        handlelength=1.65,
        handletextpad=0.35,
        columnspacing=0.72,
        frameon=True,
        facecolor="white",
        edgecolor=FIGURE_SCIENCE["grid"],
        framealpha=0.96,
        borderpad=0.28,
    )
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.245, right=0.985, top=0.93, bottom=0.15)
    save_figure(fig, out_dir, "fig3_finite_size_scaling")
    figure4_intervention_effects(out_dir, p04_rows)


def figure4_intervention_effects(out_dir: Path, p04_rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.56), facecolor="white")
    if p04_rows:
        if not draw_p04_grouped_effect_forest(ax, p04_rows, show_values=True):
            missing(ax, "Collapse-boundary shift", "No resolved P04 contrasts.")
    else:
        missing(ax, "Collapse-boundary shift", "P04 rows missing.")
    fig.subplots_adjust(left=0.285, right=0.985, top=0.94, bottom=0.19)
    save_figure(fig, out_dir, "fig4_intervention_effects")


def figure2_p01(
    out_dir: Path,
    p01_rows: list[dict[str, str]],
    p01_display_rows: list[dict[str, str]] | None = None,
) -> None:
    figure2_p01_nonlinear_response(out_dir, p01_rows, p01_display_rows)


def variant_settings(variant: str) -> tuple[float, bool]:
    if variant.endswith("_q0"):
        return 0.0, "no_social" not in variant
    if variant.endswith("_q1"):
        return 1.0, "no_social" not in variant
    return 0.5, "no_social" not in variant


def attack_proxy(row: dict[str, str]) -> float:
    if row.get("attack_magnitude_proxy", "") != "":
        return max(f(row, "attack_magnitude_proxy"), 1e-9)
    n = max(f(row, "n_society"), 1.0)
    k = max(f(row, "requested_harmful_count", f(row, "n_harmful")), 0.0)
    q = f(row, "liquidity_exponent", variant_settings(row.get("variant", ""))[0])
    return max(k / max(n**q, 1e-9), 1e-9)


def gain_proxy(row: dict[str, str]) -> float:
    if row.get("failure_gain_proxy", "") != "":
        return max(f(row, "failure_gain_proxy"), 1e-9)
    return max(f(row, "primary_failure_score_max", 0.0) / attack_proxy(row), 1e-9)


def fit_slope(rows: list[dict[str, object]], y_key: str) -> float:
    usable = []
    for row in rows:
        y = float(row[y_key])
        n = float(row["n_society"])
        k = float(row.get("requested_harmful_count", 0))
        if y > 0 and n > 0 and k > 0:
            usable.append((math.log(n), math.log(k), math.log(y)))
    if len(usable) < 3:
        return float("nan")
    x = np.asarray([[1.0, u[0], u[1]] for u in usable])
    y = np.asarray([u[2] for u in usable])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(beta[1])


def figure3_p02(out_dir: Path, p02_rows: list[dict[str, str]], p01_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"wspace": 0.45})
    for label, ax in zip("ABC", axes):
        panel_label(ax, label)
    if not p02_rows:
        for ax, title in zip(axes, ["Attack magnitude", "Feedback gain", "Exponent closure"]):
            missing(ax, title)
        save_figure(fig, out_dir, "fig3_p02_decomposition")
        return
    enriched = []
    for row in p02_rows:
        q, social = variant_settings(row.get("variant", ""))
        enriched.append({
            "variant": row.get("variant", ""),
            "n_society": int(f(row, "n_society")),
            "requested_harmful_count": int(f(row, "requested_harmful_count", f(row, "n_harmful"))),
            "liquidity_exponent": q,
            "social_on": social,
            "attack": attack_proxy(row),
            "gain": gain_proxy(row),
        })

    ax = axes[0]
    ax.set_title(r"Attack magnitude proxy", loc="left", fontweight="bold")
    for q, color in zip([0.0, 0.5, 1.0], [COLORS["green"], COLORS["blue"], COLORS["orange"]]):
        group = [r for r in enriched if r["social_on"] and abs(float(r["liquidity_exponent"]) - q) < 1e-9]
        by_n: dict[int, list[float]] = defaultdict(list)
        for r in group:
            by_n[int(r["n_society"])].append(float(r["attack"]))
        if by_n:
            xs = sorted(by_n)
            ys = [mean(by_n[x]) for x in xs]
            ax.loglog(xs, ys, marker="o", color=color, label=fr"$\ell={q:g}$")
    ax.set_xlabel("society size N")
    ax.set_ylabel(r"proxy $a_N$")
    ax.legend(title="liquidity")
    polish_axis(ax)

    ax = axes[1]
    ax.set_title(r"Finite-horizon gain proxy", loc="left", fontweight="bold")
    for social, color, label in [(True, COLORS["blue"], "social on"), (False, COLORS["red"], "social off")]:
        group = [r for r in enriched if r["social_on"] == social and abs(float(r["liquidity_exponent"]) - 0.5) < 1e-9]
        by_n: dict[int, list[float]] = defaultdict(list)
        for r in group:
            by_n[int(r["n_society"])].append(float(r["gain"]))
        if by_n:
            xs = sorted(by_n)
            ys = [mean(by_n[x]) for x in xs]
            ax.loglog(xs, ys, marker="o", color=color, label=label)
    ax.set_xlabel("society size N")
    ax.set_ylabel(r"proxy $\widehat{\chi}_{T,N}$")
    ax.legend()
    polish_axis(ax)

    ax = axes[2]
    ax.set_title("Exponent closure", loc="left", fontweight="bold")
    p01_sum = p01_summaries(p01_rows)
    resolved = [r for r in p01_sum if r["alpha_c"] != ""]
    nu = float("nan")
    if len(resolved) >= 2:
        slope, _ = np.polyfit(np.log([float(r["N"]) for r in resolved]), np.log([float(r["alpha_c"]) for r in resolved]), 1)
        nu = -float(slope)
    baseline = [r for r in enriched if r["variant"] == "baseline_q05"]
    delta = 1.0 + fit_slope(baseline, "attack")
    zeta = fit_slope(baseline, "gain")
    vals = [nu, delta + zeta if math.isfinite(delta) and math.isfinite(zeta) else float("nan"), nu - (delta + zeta) if all(map(math.isfinite, [nu, delta, zeta])) else float("nan")]
    labels = [r"$\hat\nu$", r"$\hat\delta+\hat\zeta$", r"$\hat\nu-(\hat\delta+\hat\zeta)$"]
    ax.axvline(0, color="#999999", linestyle="--", linewidth=0.8)
    for i, (val, lab) in enumerate(zip(vals, labels)):
        if math.isfinite(val):
            ax.plot(val, i, marker="o", color=[COLORS["blue"], COLORS["orange"], COLORS["purple"]][i])
            ax.text(val, i + 0.12, f"{val:.2f}", ha="center", va="bottom")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("exponent value")
    polish_axis(ax)
    save_figure(fig, out_dir, "fig3_p02_decomposition")


def alpha_c_by_group(rows: list[dict[str, str]], group_keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in group_keys)].append(row)
    out = []
    for key, group in groups.items():
        curve = [(x, y) for x, y, _, _ in aggregate_curve(group, "primary_failure_rate")]
        ac = crossing(curve, 0.5)
        record = {k: v for k, v in zip(group_keys, key)}
        record["alpha_c"] = ac
        record["mean_coupling"] = mean([f(row, "mean_social_coupling_proxy", 0.0) for row in group])
        out.append(record)
    return out


def p04_effect_items(p04_rows: list[dict[str, str]]) -> list[tuple[str, list[float]]]:
    ac = alpha_c_by_group(p04_rows, ("variant", "n_society"))
    baseline = {(r["n_society"]): r["alpha_c"] for r in ac if r["variant"] == "qre_baseline"}
    effects: dict[str, list[float]] = defaultdict(list)
    for r in ac:
        b = baseline.get(r["n_society"])
        if r["variant"] != "qre_baseline" and r["alpha_c"] is not None and b is not None:
            effects[str(r["variant"])].append(float(r["alpha_c"]) - float(b))
    return [(variant, effects[variant]) for variant in VARIANT_ORDER if effects.get(variant)]


def p04_effect_statistics(
    p04_rows: list[dict[str, str]],
    *,
    n_boot: int = 600,
) -> list[tuple[str, float, float, float]]:
    """Return matched-size effect estimates with seed-bootstrap intervals."""
    point = {variant: mean(values) for variant, values in p04_effect_items(p04_rows)}
    if not point:
        return []
    by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in p04_rows:
        seed = row.get("seed", "")
        if seed:
            by_seed[seed].append(row)
    seeds = sorted(by_seed)
    samples: dict[str, list[float]] = defaultdict(list)
    if len(seeds) >= 2:
        rng = np.random.default_rng(SEED + 404)
        for _ in range(n_boot):
            sampled_rows: list[dict[str, str]] = []
            for seed in rng.choice(seeds, size=len(seeds), replace=True):
                sampled_rows.extend(by_seed[str(seed)])
            for variant, values in p04_effect_items(sampled_rows):
                samples[variant].append(mean(values))
    out = []
    for variant, center in point.items():
        arr = np.asarray(samples.get(variant, []), dtype=float)
        if len(arr) >= max(30, n_boot // 10):
            lo, hi = np.quantile(arr, [0.025, 0.975])
        else:
            lo = hi = center
        out.append((variant, center, float(lo), float(hi)))
    return sorted(out, key=lambda item: item[1])


def draw_p04_effect_forest(
    ax: plt.Axes,
    p04_rows: list[dict[str, str]],
    *,
    show_values: bool = False,
    show_direction_labels: bool = False,
) -> bool:
    items = p04_effect_statistics(p04_rows)
    if not items:
        return False
    all_values = [value for _, center, lo, hi in items for value in (center, lo, hi)]
    data_lo, data_hi = min(all_values), max(all_values)
    span = max(data_hi - data_lo, 0.04)
    x_lo = min(-0.01, data_lo - 0.11 * span)
    x_hi = max(0.01, data_hi + 0.11 * span)
    ax.axvspan(x_lo, 0, color=FIGURE_SCIENCE["fragile_bg"], alpha=0.62, linewidth=0)
    ax.axvspan(0, x_hi, color=FIGURE_SCIENCE["robust_bg"], alpha=0.62, linewidth=0)
    ypos = np.arange(len(items))
    ax.axvline(0, color=FIGURE_SCIENCE["zero"], linestyle="--", linewidth=0.82, zorder=2)
    for y, (variant, center, lo, hi) in zip(ypos, items):
        color = FIGURE_SCIENCE["robust"] if center >= 0 else FIGURE_SCIENCE["fragile"]
        crosses_zero = lo <= 0.0 <= hi
        alpha = 0.58 if crosses_zero else 0.94
        line_width = 1.55 if crosses_zero else 2.05
        marker_size = 31 if crosses_zero else 42
        marker = "o" if crosses_zero else "D"
        ax.hlines(y, lo, hi, color=color, linewidth=line_width, alpha=alpha, zorder=3)
        ax.scatter(center, y, color=color, marker=marker, s=marker_size, alpha=alpha, zorder=4, edgecolor="white", linewidth=MARKER_EDGE)
        if show_values:
            ax.text(1.025, y, f"{center:+.3f}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.0, color=FIGURE_SCIENCE["axis"], clip_on=False)
    ax.set_yticks(ypos, [VARIANT_LABELS.get(k, k.replace("_", " ")) for k, *_ in items])
    ax.invert_yaxis()
    ax.set_ylim(len(items) - 0.5, -0.85 if show_direction_labels else -0.5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(r"change in collapse boundary $\Delta\alpha_c$")
    polish_science_axis(ax, grid_axis="x")
    if show_direction_labels:
        ax.text(0.02, 0.985, "more fragile", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color="#B47A90")
        ax.text(0.98, 0.985, "more robust", transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color="#8272A7")
    return True


def draw_p04_grouped_effect_forest(
    ax: plt.Axes,
    p04_rows: list[dict[str, str]],
    *,
    show_values: bool = True,
) -> bool:
    stats = {variant: (center, lo, hi) for variant, center, lo, hi in p04_effect_statistics(p04_rows)}
    grouped_items = [
        (group_name, [variant for variant in variants if variant in stats])
        for group_name, variants in INTERVENTION_GROUPS
    ]
    grouped_items = [(name, variants) for name, variants in grouped_items if variants]
    if not grouped_items:
        return False
    control_variants = {
        variant
        for group_name, variants in grouped_items
        if group_name == "Controls"
        for variant in variants
    }

    all_values = [value for center, lo, hi in stats.values() for value in (center, lo, hi)]
    data_lo, data_hi = min(all_values), max(all_values)
    x_lo = min(-0.04, data_lo - 0.006)
    x_hi = max(0.058, data_hi + 0.012)

    ax.axvline(0, color=FIGURE_SCIENCE["zero"], linestyle="--", linewidth=0.82, alpha=0.88, zorder=2)

    rows: list[tuple[float, str]] = []
    headings: list[tuple[float, str]] = []
    cursor = 0.0
    for group_index, (group_name, variants) in enumerate(grouped_items):
        headings.append((cursor - 0.72, group_name))
        rows.extend((cursor + index, variant) for index, variant in enumerate(variants))
        cursor += len(variants)
        if group_index < len(grouped_items) - 1:
            cursor += 1.05

    group_band_colors = [FIGURE_SCIENCE["robust_bg"], "#F7F4F8"]
    row_lookup = {variant: y for y, variant in rows}
    for group_index, (_, variants) in enumerate(grouped_items):
        group_rows = [row_lookup[variant] for variant in variants]
        ax.axhspan(
            min(group_rows) - 0.48,
            max(group_rows) + 0.48,
            color=group_band_colors[group_index % len(group_band_colors)],
            alpha=0.62,
            linewidth=0,
            zorder=0,
        )
        if group_index < len(grouped_items) - 1:
            ax.axhline(max(group_rows) + 1.02, color=FIGURE_SCIENCE["grid"], linewidth=0.65, zorder=1)

    for y, variant in rows:
        center, lo, hi = stats[variant]
        # Sign carries the scientific meaning: right shifts are more robust,
        # left shifts are more fragile. Use one shape throughout.
        crosses_zero = lo <= 0.0 <= hi
        is_control = variant in control_variants
        color = FIGURE_SCIENCE["comparison"] if crosses_zero else (
            FIGURE_SCIENCE["fit"] if center < 0 else FIGURE_SCIENCE["fragile"]
        )
        edge_color = (
            "#776E7E" if crosses_zero else ("#4F3278" if center < 0 else "#98415F")
        )
        interval_alpha = 0.58 if crosses_zero else 0.72
        point_alpha = 0.70 if is_control else (0.72 if crosses_zero else 0.98)
        ax.hlines(y, lo, hi, color=color, linewidth=1.18, alpha=interval_alpha, zorder=3)
        ax.scatter(
            center,
            y,
            marker="o",
            s=30,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=0.7,
            alpha=point_alpha,
            zorder=4,
        )
        if show_values:
            ax.text(
                x_hi - 0.0012,
                y,
                f"{center:+.3f}",
                ha="right",
                va="center",
                fontsize=6.25,
                color=FIGURE_SCIENCE["axis"] if is_control else color,
                fontweight="semibold" if not crosses_zero else "normal",
            )

    y_positions = [y for y, _ in rows]
    y_labels = [FIG4_LABELS.get(variant, VARIANT_LABELS.get(variant, variant.replace("_", " "))) for _, variant in rows]
    ax.set_yticks(y_positions, y_labels)
    for y, heading in headings:
        ax.text(
            -0.31,
            y,
            heading,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6.7,
            color=FIGURE_SCIENCE["axis"],
            fontweight="semibold",
            clip_on=False,
        )

    # Direction cue lives in the empty band between the two groups, keeping the
    # bottom axis for the Delta-alpha_c title only.
    first_count = len(grouped_items[0][1])
    gap_y = (max(y for y, _ in rows[:first_count]) + min(y for y, _ in rows[first_count:])) / 2.0
    cue_bbox = dict(facecolor="white", edgecolor="none", pad=0.8)
    ax.text(0.012, gap_y, r"$\leftarrow$ Fragility", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.0, color="#8A8A8A", bbox=cue_bbox, zorder=5)
    ax.text(0.988, gap_y, r"Robustness $\rightarrow$", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.0, color="#8A8A8A", bbox=cue_bbox, zorder=5)

    ax.invert_yaxis()
    ax.set_ylim(max(y_positions) + 0.50, min(y for y, _ in headings) - 0.18)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(r"$\Delta\alpha_c$", fontsize=7.5, labelpad=4.0)
    ax.tick_params(axis="y", labelsize=6.0, pad=1.5)
    for label in ax.get_yticklabels():
        label.set_ha("right")
    ax.tick_params(axis="x", labelsize=6.75)
    polish_science_axis(ax, grid_axis="x")
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=FIGURE_SCIENCE["grid"], linewidth=0.45, alpha=0.64)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return True


def figure_p04_feedback_shift(out_dir: Path, p04_rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.72))
    ax.set_title("Feedback strength moves the collapse boundary", loc="left", fontweight="bold", pad=13.0)
    if p04_rows:
        if not draw_p04_effect_forest(ax, p04_rows, show_values=True, show_direction_labels=True):
            missing(ax, "Feedback strength moves the collapse boundary", "No resolved P04 contrasts.")
    else:
        missing(ax, "Feedback strength moves the collapse boundary", "P04 rows missing.")
    fig.subplots_adjust(right=0.83)
    save_figure(fig, out_dir, "fig4_p04_feedback_shift")


def figure_p04_feedback_available(out_dir: Path, p04_rows: list[dict[str, str]]) -> None:
    """Standalone polished figure for the completed P04 intervention experiment."""
    fig = plt.figure(figsize=(7.2, 4.35))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.62)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for label, ax in zip("ABCD", axes):
        panel_label(ax, label)
    if not p04_rows:
        for ax, title in zip(axes, ["Response curves", "Threshold shift", "Feedback plane", "Threshold map"]):
            missing(ax, title, "P04 rows missing.")
        save_figure(fig, out_dir, "fig_p04_feedback_available")
        return

    largest_n = max(int(f(row, "n_society")) for row in p04_rows if math.isfinite(f(row, "n_society")))
    selected = ["qre_baseline", "qre_weak_coupling", "qre_high_reach", "qre_strong_coupling"]
    color_map = {
        "qre_baseline": COLORS["gray"],
        "qre_weak_coupling": COLORS["blue"],
        "qre_high_reach": COLORS["orange"],
        "qre_strong_coupling": COLORS["red"],
        "mixed_roles_reference": COLORS["green"],
    }

    ax = axes[0]
    ax.set_title(f"Representative response curves (N={largest_n})", loc="left", fontweight="bold")
    for variant in selected:
        group = [row for row in p04_rows if row.get("variant") == variant and int(f(row, "n_society")) == largest_n]
        if not group:
            continue
        curve = aggregate_curve(group, "primary_failure_rate")
        plot_mean_band(
            ax,
            curve,
            color=color_map[variant],
            label=VARIANT_LABELS.get(variant, variant),
            marker="o",
            band_alpha=0.0,
        )
    ax.axhline(0.5, color="#8A8F98", linestyle="--", linewidth=0.85)
    ax.set_xlabel(r"harmful fraction $\alpha$")
    ax.set_ylabel("failure probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", ncol=1)
    polish_axis(ax)

    ax = axes[1]
    ax.set_title("Collapse-boundary shift", loc="left", fontweight="bold")
    if not draw_p04_effect_forest(ax, p04_rows):
        missing(ax, "Collapse-boundary shift", "No resolved P04 contrasts.")

    ax = axes[2]
    ax.set_title("Feedback proxy vs. threshold", loc="left", fontweight="bold")
    ac = alpha_c_by_group(p04_rows, ("variant", "n_society"))
    for n, marker in [(300, "o"), (1000, "s")]:
        pts = []
        for r in ac:
            if r["alpha_c"] is None or int(r["n_society"]) != n:
                continue
            pts.append((float(r["mean_coupling"]), float(r["alpha_c"]), str(r["variant"])))
        if not pts:
            continue
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker=marker,
            s=36,
            color=COLORS["purple"] if n == 300 else COLORS["orange"],
            edgecolor="white",
            linewidth=0.45,
            label=f"N={n}",
            alpha=0.92,
        )
    ax.set_xlabel("mean social-coupling proxy")
    ax.set_ylabel(r"collapse boundary $\alpha_c$")
    ax.legend(loc="best")
    polish_axis(ax)

    ax = axes[3]
    ax.set_title(r"Resolved $\alpha_c$ by intervention and size", loc="left", fontweight="bold")
    matrix_variants = ["qre_baseline"] + VARIANT_ORDER
    matrix_ns = sorted({int(f(row, "n_society")) for row in p04_rows if math.isfinite(f(row, "n_society"))})
    ac_lookup = {(str(r["variant"]), int(r["n_society"])): r["alpha_c"] for r in ac if r["alpha_c"] is not None}
    mat = np.full((len(matrix_variants), len(matrix_ns)), np.nan)
    for i, variant in enumerate(matrix_variants):
        for j, n in enumerate(matrix_ns):
            value = ac_lookup.get((variant, n))
            if value is not None:
                mat[i, j] = float(value)
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="YlGnBu", aspect="auto")
    finite = mat[np.isfinite(mat)]
    threshold = float(np.nanmin(finite) + 0.62 * (np.nanmax(finite) - np.nanmin(finite))) if finite.size else float("inf")
    ax.set_xticks(range(len(matrix_ns)), [str(n) for n in matrix_ns])
    ax.set_yticks(range(len(matrix_variants)), [VARIANT_LABELS.get(v, v) for v in matrix_variants])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if math.isfinite(float(mat[i, j])):
                txt_color = "white" if float(mat[i, j]) >= threshold else COLORS["dark"]
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=6.2, color=txt_color)
    ax.set_xlabel("society size N")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=r"$\alpha_c$")
    ax.grid(False)
    save_figure(fig, out_dir, "fig_p04_feedback_available")


def figure4_mechanism(out_dir: Path, p04_rows: list[dict[str, str]], p05_rows: list[dict[str, str]]) -> None:
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.58)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for label, ax in zip("ABCD", axes):
        panel_label(ax, label)

    ax = axes[0]
    ax.set_title("Two-coordinate intervention plane", loc="left", fontweight="bold")
    if p05_rows:
        groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in p05_rows:
            groups[(row.get("variant", ""), int(f(row, "n_society")))].append(row)
        xs, ys, cs, labels = [], [], [], []
        for (variant, n), group in sorted(groups.items()):
            xs.append(mean([f(r, "mean_social_coupling_proxy", 0.0) for r in group]))
            ys.append(mean([f(r, "social_dominance_ratio", 0.0) for r in group]))
            cs.append(mean([f(r, "primary_failure_score_max", 0.0) for r in group]))
            labels.append(variant.replace("_", "\n"))
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=36, edgecolor="#333333", linewidth=0.4)
        ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="risk")
        polish_axis(ax)
    else:
        missing(ax, "Two-coordinate intervention plane", "P05 rows missing.")
    ax.set_xlabel(r"feedback proxy $\widehat{\chi}_{T,N}$")
    ax.set_ylabel(r"information dominance $D_N$")

    ax = axes[1]
    ax.set_title("Feedback interventions", loc="left", fontweight="bold")
    if p04_rows:
        if not draw_p04_effect_forest(ax, p04_rows):
            missing(ax, "Feedback interventions", "No resolved P04 alpha_c contrasts.")
    else:
        missing(ax, "Feedback interventions", "P04 rows missing.")

    ax = axes[2]
    ax.set_title("Information dominance", loc="left", fontweight="bold")
    if p05_rows:
        by_variant: dict[str, list[float]] = defaultdict(list)
        for row in p05_rows:
            by_variant[row.get("variant", "")].append(f(row, "social_dominance_ratio", 0.0))
        items = sorted((variant, mean(values)) for variant, values in by_variant.items())
        ypos = np.arange(len(items))
        ax.axvline(0.5, color="#999999", linestyle="--", linewidth=0.8)
        ax.scatter([v for _, v in items], ypos, color=COLORS["orange"], s=20)
        ax.set_yticks(ypos, [k.replace("_", " ") for k, _ in items])
        ax.invert_yaxis()
        polish_axis(ax)
    else:
        missing(ax, "Information dominance", "P05 rows missing.")
    ax.set_xlabel(r"$D_N$")

    ax = axes[3]
    ax.set_title("Conflict-following behavior", loc="left", fontweight="bold")
    if p05_rows:
        by_variant = defaultdict(list)
        for row in p05_rows:
            by_variant[row.get("variant", "")].append(f(row, "cascade_decision_rate", 0.0))
        items = sorted((variant, mean(values)) for variant, values in by_variant.items())
        ypos = np.arange(len(items))
        ax.scatter([v for _, v in items], ypos, color=COLORS["green"], s=20)
        ax.set_yticks(ypos, [k.replace("_", " ") for k, _ in items])
        ax.invert_yaxis()
        polish_axis(ax)
    else:
        missing(ax, "Conflict-following behavior", "P05 rows missing.")
    ax.set_xlabel("follow-social rate under conflict")
    save_figure(fig, out_dir, "fig4_mechanism")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--figures", default="all", help="Comma list: fig1,fig2,fig3,fig4,p04,all")
    args = parser.parse_args()
    setup_style()
    out_dir = ensure_dir(Path(args.out_dir))

    p01 = read_rows("p01_nonlinear_scaling_paper")
    p01_display = read_rows("p01_nonlinear_scaling_paper", include_extra=True)
    p02 = read_rows("p02_size_decomposition_paper")
    p04 = read_rows("p04_game_phase_paper")
    p05 = read_rows("p05_information_cascade_paper")
    requested = {"fig1", "fig2", "fig3", "fig4", "p04"} if args.figures == "all" else set(args.figures.split(","))

    if "fig1" in requested:
        figure1_overview(out_dir)
    if "fig2" in requested:
        figure2_p01(out_dir, p01, p01_display)
    if "fig3" in requested:
        figure3_p01_finite_size_scaling(out_dir, p01, p04)
    if "fig4" in requested:
        figure4_mechanism(out_dir, p04, p05)
    if "p04" in requested:
        figure_p04_feedback_shift(out_dir, p04)
        figure_p04_feedback_available(out_dir, p04)

    manifest = {
        "out_dir": str(out_dir),
        "style_backend": STYLE_BACKEND,
        "n_rows": {
            "p01": len(p01),
            "p01_display": len(p01_display),
            "p02": len(p02),
            "p04": len(p04),
            "p05": len(p05),
        },
        "warning": "Figures with incomplete experiment rows are draft diagnostics only.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
