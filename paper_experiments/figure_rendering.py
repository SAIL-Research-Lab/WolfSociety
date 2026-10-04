"""Render publication figures exclusively from supplied, validated episode rows.

This module never discovers data files or reads legacy CSV/figure locations.
The publication exporter owns evidence validation and provenance. Synthetic
fixture inputs remain visibly watermarked for visual regression checks.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np

from .analysis import boundary_rows, crossing, curve, primary, write_csv

BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20261004
MIN_RESOLVED_BOOTSTRAP_SHARE = .80
COLORS = {"ink": "#26212B", "muted": "#67616D", "plum": "#715282",
          "teal": "#247F85", "rose": "#BF6881", "grid": "#E7E2EB", "soft": "#F6F3F8"}
SIZE_COLORS = ["#CC809A", "#B276A1", "#946C9F", "#7D6397", "#64568A", "#493D6D"]
INTERVENTIONS = {
    "weak_feedback": "Weak feedback",
    "strong_feedback": "Strong feedback",
    "high_conformity": "High conformity",
    "high_reach": "High reach",
    "high_attention": "High attention",
    "high_deliberation": "High deliberation",
    "no_feedback": "Frozen market observations",
    "no_multihop": "Source-only communication",
}
INTERVAL_NOTE = (
    "95% percentile intervals use paired seed bootstrap (1,000 draws), conditional on resolved draws. "
    "Intervals are omitted when fewer than 80% of draws resolve; source CSV records all resolution counts."
)


def _style():
    return {"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
            "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
            "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
            "axes.edgecolor": "#ABA4B0", "axes.linewidth": .7,
            "text.color": COLORS["ink"], "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
            "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 240,
            "axes.axisbelow": True}


def _decorate(ax, *, ygrid=True):
    if ygrid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=.6)
    ax.tick_params(length=3, width=.6, pad=3)


def _footer(fig, model, fixture, text):
    if fixture:
        text = "SYNTHETIC FIXTURE — NOT SCIENTIFIC RESULTS\n" + text
    short_model = model if len(model) <= 80 else model[:77] + "…"
    width = max(35, int((fig.get_figwidth() - .18) * 72 / (6.1 * .53)))
    wrapped = "\n".join(textwrap.fill(line, width=width, break_long_words=True, break_on_hyphens=False)
                        for line in f"{text}\nModel: {short_model}".splitlines())
    fig.text(.02, .022, wrapped, fontsize=6.1,
             color=COLORS["rose"] if fixture else COLORS["muted"], va="bottom")


def _save(fig, outdir, figure_id):
    files = []
    for suffix in ("pdf", "png"):
        filename = f"{figure_id}.{suffix}"
        fig.savefig(outdir / filename, facecolor="white", bbox_inches="tight", pad_inches=.055)
        files.append(filename)
    plt.close(fig)
    return files


def _source(outdir, figure_id, rows):
    name = f"{figure_id}.source.csv"
    write_csv(outdir / name, rows)
    return name


def _interval(samples):
    values = samples[np.isfinite(samples)]
    valid = len(values)
    if valid < math.ceil(MIN_RESOLVED_BOOTSTRAP_SHARE * len(samples)) or not valid:
        return None, None, valid, "insufficient_resolved_bootstrap_draws"
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi), valid, "resolved_conditional_interval"


def _boundary_statistics(rows):
    """Analysis midpoints plus bootstrap draws shared across sizes/conditions."""
    grouped = defaultdict(list)
    cell_seeds = defaultdict(set)
    for row in rows:
        c = row["config"]
        key = (row["variant"], c["n_society"])
        grouped[key].append(row)
        cell_seeds[(key, c["alpha"])].add(c["seed"])
    common = sorted(set.intersection(*cell_seeds.values())) if cell_seeds else []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(common), np.full(len(common), 1 / len(common)), size=BOOTSTRAP_DRAWS) if common else None
    observed = {(r["variant"], r["n_society"]): r for r in boundary_rows(rows) if r["alpha_mode"] == "target"}
    samples = {}
    for key, members in grouped.items():
        draws = np.full(BOOTSTRAP_DRAWS, np.nan)
        if common:
            by_cell = defaultdict(list)
            for row in members:
                by_cell[(row["config"]["alpha"], row["config"]["seed"])].append(primary(row))
            alphas = sorted({r["config"]["alpha"] for r in members})
            matrix = np.array([[np.mean(by_cell[(a, seed)]) for seed in common] for a in alphas])
            probabilities = weights @ matrix.T / len(common)
            for index, values in enumerate(probabilities):
                estimate = crossing([(a, float(p), len(common)) for a, p in zip(alphas, values)])["alpha_c"]
                if estimate is not None:
                    draws[index] = estimate
        samples[key] = draws
        lo, hi, valid, status = _interval(draws)
        observed[key].update(alpha_c_ci_low=lo, alpha_c_ci_high=hi,
                             bootstrap_valid_draws=valid, bootstrap_total_draws=BOOTSTRAP_DRAWS,
                             bootstrap_paired_seeds=len(common), bootstrap_ci_status=status,
                             bootstrap_seed=BOOTSTRAP_SEED)
    return observed, samples


def _wilson(probability, count):
    if count <= 0:
        return None, None
    z = 1.959963984540054
    denom = 1 + z * z / count
    center = (probability + z * z / (2 * count)) / denom
    half = z * math.sqrt(probability * (1 - probability) / count + z * z / (4 * count * count)) / denom
    return max(0., center - half), min(1., center + half)


def _teaser(outdir, model, revision, fixture):
    figure_id = "teaser"
    fig, ax = plt.subplots(figsize=(7.25, 3.05))
    ax.set(xlim=(0, 10), ylim=(0, 4.3))
    ax.axis("off")
    fig.subplots_adjust(left=.02, right=.98, bottom=.18, top=.98)
    ax.text(.05, 4.13, "A full-LLM society, advanced one round at a time", fontsize=12, weight="bold")
    ax.text(.05, 3.76, "Shared model weights · separate identities, portfolios and memory", fontsize=8, color=COLORS["muted"])
    boxes = [
        (.10, 1.42, 2.78, 1.83, "1  Freeze observations", "Each agent reads its own\nmarket state, private signal,\nreceived text and memory.", COLORS["soft"]),
        (3.58, 1.42, 2.78, 1.83, "2  Generate decisions", "One shared vLLM service\nbatches independent requests.\nLLMs choose orders and text.", "#EAF4F4"),
        (7.08, 1.42, 2.78, 1.83, "3  Validate and settle", "Wait for the whole round.\nEnforce budgets, settle trades,\nand deliver messages.", "#FAEFF3"),
    ]
    source = []
    for x, y, width, height, heading, body, fill in boxes:
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.035,rounding_size=.12", facecolor=fill, edgecolor="#D9D1DF", linewidth=.85))
        ax.text(x + .16, y + height - .39, heading, fontsize=9, weight="bold")
        ax.text(x + .16, y + height - .77, body, fontsize=7.9, linespacing=1.6, va="top")
        source.append({"element": heading, "description": body.replace("\n", " "), "kind": "architecture_schematic", "model": model, "model_revision": revision})
    for start, end in [(2.97, 3.47), (6.46, 6.96)]:
        ax.add_patch(FancyArrowPatch((start, 2.36), (end, 2.36), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=COLORS["plum"]))
    ax.plot([8.48, 8.48, 1.49, 1.49], [1.30, .52, .52, 1.13], color=COLORS["plum"], linewidth=1)
    ax.add_patch(FancyArrowPatch((1.49, 1.0), (1.49, 1.34), arrowstyle="-|>", mutation_scale=11, linewidth=1, color=COLORS["plum"]))
    ax.text(4.98, .70, "Updated portfolios, delivered messages and private memories → next round", ha="center", fontsize=7.7, color=COLORS["plum"])
    ax.text(4.98, .02, "Agent policies make decisions; deterministic market and network rules execute them.", ha="center", fontsize=7.2, color=COLORS["muted"])
    _footer(fig, model, fixture, "Architecture schematic. No empirical values or effect directions are encoded.")
    files = _save(fig, outdir, figure_id)
    files.append(_source(outdir, figure_id, source))
    return {"figure_id": figure_id, "families": [], "files": files, "status": "schematic", "notes": ["No experimental result is represented in this diagram."]}


def _response(rows, outdir, model, revision, fixture):
    figure_id = "fig2_nonlinear_response"
    sizes = sorted({r["config"]["n_society"] for r in rows})
    panels = max(1, len(sizes))
    columns = min(3, panels)
    nrows = math.ceil(panels / columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(7.25, 2.02 * nrows + .67), squeeze=False)
    fig.subplots_adjust(left=.09, right=.985, top=.88, bottom=.18, wspace=.27, hspace=.57)
    fig.suptitle("Collapse response at each population size", x=.09, y=.977, ha="left", fontsize=12, weight="bold")
    source, notes = [], []
    if not sizes:
        axes[0, 0].text(.5, .5, "No validated P01 observations", ha="center", transform=axes[0, 0].transAxes)
    for index, n in enumerate(sizes):
        ax = axes.flat[index]
        members = [r for r in rows if r["config"]["n_society"] == n]
        points = curve(members)
        color = SIZE_COLORS[min(index, len(SIZE_COLORS) - 1)]
        x, y, low, high = [], [], [], []
        for alpha, probability, count in points:
            lo, hi = _wilson(probability, count)
            x.append(alpha * 100)
            y.append(probability)
            low.append(lo)
            high.append(hi)
            source.append({"model": model, "model_revision": revision, "family": "p01", "n_society": n,
                           "alpha": alpha, "failure_probability": probability, "independent_seeds": count,
                           "wilson_95_low": lo, "wilson_95_high": hi})
        if points:
            ax.fill_between(x, low, high, color=color, alpha=.15, linewidth=0)
            ax.plot(x, y, marker="o", markersize=3.6, color=color, linewidth=1.35, markeredgecolor="white", markeredgewidth=.35)
            xmax = max(x)
            ax.set_xlim(-.035 * max(xmax, 1), max(xmax, 1) * 1.055)
        midpoint = crossing(points)
        label = midpoint["status"].replace("_", " ")
        if midpoint["status"] != "resolved":
            ax.text(.98, .08, f"Boundary: {label}", ha="right", fontsize=6.2, color=COLORS["muted"], transform=ax.transAxes)
            notes.append(f"N={n}: {label}; no midpoint imputed")
        ax.axhline(.5, color="#AAA1B0", linestyle=(0, (3, 3)), linewidth=.8)
        ax.set(ylim=(-.04, 1.04), yticks=[0, .5, 1], title=f"N = {n:,}")
        ax.set_xlabel("Harmful share α (%)")
        if index % columns == 0:
            ax.set_ylabel("Collapse probability")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        _decorate(ax)
    for ax in list(axes.flat)[panels:]:
        ax.axis("off")
    _footer(fig, model, fixture, "Dots: observed seed fractions. Bands: 95% Wilson intervals. Dashed line: probability 0.5.")
    files = _save(fig, outdir, figure_id)
    files.append(_source(outdir, figure_id, source))
    return {"figure_id": figure_id, "families": ["p01"], "files": files,
            "status": "rendered" if source else "missing", "notes": notes}


def _finite_size(rows, outdir, model, revision, fixture):
    figure_id = "fig3_finite_size_scaling"
    records, _ = _boundary_statistics(rows)
    data = [dict(row) for _, row in sorted(records.items(), key=lambda item: item[0][1])]
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.4), sharex=True)
    fig.subplots_adjust(left=.20, right=.965, top=.79, bottom=.25, hspace=.45)
    fig.suptitle("Observed failure boundary\nacross population sizes", x=.04, y=.978, ha="left", fontsize=10, weight="bold")
    for row in data:
        n = row["n_society"]
        row["k_c_ci_low"] = row["alpha_c_ci_low"] * n if row["alpha_c_ci_low"] is not None else None
        row["k_c_ci_high"] = row["alpha_c_ci_high"] * n if row["alpha_c_ci_high"] is not None else None
        row["k_c_bound"] = row.get("bound", 0) * n if "bound" in row else None
    notes = [INTERVAL_NOTE, "No power-law fit or effect direction is assumed."]
    for panel, (axis, multiplier, title, ylabel, color) in enumerate([
        (axes[0], lambda n: 100., "A  Critical harmful share", "50% boundary αc (%)", COLORS["plum"]),
        (axes[1], lambda n: float(n), "B  Corresponding harmful count", "Kc = N × αc", COLORS["teal"]),
    ]):
        visible = []
        for row in data:
            scale = multiplier(row["n_society"])
            if row["alpha_c"] is not None:
                visible.append(row["alpha_c"] * scale)
            for name in ("alpha_c_ci_low", "alpha_c_ci_high", "bound"):
                value = row.get(name)
                if value is not None and value > 0:
                    visible.append(value * scale)
        positive = [value for value in visible if value > 0]
        ymin, ymax = (min(positive) / 1.9, max(positive) * 2) if positive else (.01, 1.)
        axis.set_yscale("log")
        axis.set_xscale("log")
        axis.set_ylim(ymin, ymax)
        if data:
            ns = [r["n_society"] for r in data]
            axis.set_xlim(min(ns) / 1.22, max(ns) * 1.22)
            axis.set_xticks(ns, [f"{n:,}" for n in ns], rotation=30, ha="right")
        for row in data:
            n, estimate, scale = row["n_society"], row["alpha_c"], multiplier(row["n_society"])
            if row["status"] == "resolved" and estimate is not None and estimate > 0:
                value = estimate * scale
                lo, hi = row["alpha_c_ci_low"], row["alpha_c_ci_high"]
                if lo is not None and hi is not None:
                    axis.vlines(n, lo * scale, hi * scale, color=color, linewidth=1.2)
                    axis.plot([n, n], [lo * scale, hi * scale], marker="_", linestyle="none", color=color, markersize=5)
                axis.plot(n, value, marker="o", markersize=5.2, markerfacecolor=color if lo is not None else "white", markeredgecolor=color, markeredgewidth=1.1)
            elif row["status"] == "right_censored" and row.get("bound", 0) > 0:
                value = row["bound"] * scale
                axis.plot(n, value, marker="_", color=color, markersize=7)
                axis.annotate("", xy=(n, value * 1.65), xytext=(n, value), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.0})
            else:
                axis.text(n, ymin * 1.18, "unresolved", fontsize=5.7, rotation=90, ha="center", va="bottom", color=COLORS["muted"])
            if panel == 0 and row["status"] != "resolved":
                notes.append(f"N={n}: {row['status']}; estimate left missing")
        axis.set(title=title, ylabel=ylabel)
        if panel == 1:
            axis.set_xlabel("Population size N")
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.)))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_minor_formatter(NullFormatter())
        _decorate(axis)
    if not data:
        for ax in axes:
            ax.text(.5, .5, "No validated P01 observations", ha="center", transform=ax.transAxes)
    handles = [Line2D([], [], marker="o", color=COLORS["plum"], linestyle="none", markersize=4, label="Midpoint"),
               Line2D([], [], marker="o", markerfacecolor="white", color=COLORS["plum"], linestyle="none", markersize=4, label="CI omitted"),
               Line2D([], [], marker=r"$\uparrow$", color=COLORS["plum"], linestyle="none", markersize=7, label="Above tested grid")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.57, .90), ncol=2, frameon=False, columnspacing=1.2, fontsize=6.8)
    _footer(fig, model, fixture, "95% intervals: paired-seed bootstrap, conditional on resolved draws. Resolution counts are in the source CSV.\nArrows show right censoring; no fitted values replace unresolved boundaries.")
    files = _save(fig, outdir, figure_id)
    files.append(_source(outdir, figure_id, data))
    return {"figure_id": figure_id, "families": ["p01"], "files": files,
            "status": "rendered" if data else "missing", "notes": notes}


def _interventions(rows, outdir, model, revision, fixture):
    figure_id = "fig4_intervention_effects"
    records, samples = _boundary_statistics(rows)
    sizes = sorted({n for _, n in records})
    found = {variant for variant, _ in records if variant != "baseline"}
    variants = [variant for variant in INTERVENTIONS if variant in found] + sorted(found - INTERVENTIONS.keys())
    source = []
    for variant in variants:
        for n in sizes:
            observed, baseline = records.get((variant, n)), records.get(("baseline", n))
            record = {"family": "p04", "model": model, "model_revision": revision, "variant": variant,
                      "n_society": n, "alpha_c": observed.get("alpha_c") if observed else None,
                      "baseline_alpha_c": baseline.get("alpha_c") if baseline else None,
                      "variant_status": observed.get("status", "missing") if observed else "missing",
                      "baseline_status": baseline.get("status", "missing") if baseline else "missing",
                      "delta_alpha_c": None, "delta_ci_low": None, "delta_ci_high": None,
                      "bootstrap_valid_draws": 0, "bootstrap_total_draws": BOOTSTRAP_DRAWS,
                      "bootstrap_paired_seeds": observed.get("bootstrap_paired_seeds", 0) if observed else 0,
                      "status": "unresolved"}
            if observed and baseline and observed["status"] == baseline["status"] == "resolved":
                record["delta_alpha_c"] = observed["alpha_c"] - baseline["alpha_c"]
                lo, hi, valid, ci_status = _interval(samples[(variant, n)] - samples[("baseline", n)])
                record.update(delta_ci_low=lo, delta_ci_high=hi, bootstrap_valid_draws=valid,
                              bootstrap_ci_status=ci_status, status="resolved")
            source.append(record)
    fig, ax = plt.subplots(figsize=(3.5, max(4.5, .48 * len(variants) + 1.8)))
    fig.subplots_adjust(left=.42, right=.965, top=.84, bottom=.28)
    fig.suptitle("Intervention effects on the\nobserved failure boundary", x=.04, y=.978, ha="left", fontsize=10, weight="bold")
    palette = [COLORS["plum"], COLORS["teal"], COLORS["rose"]]
    offsets = np.linspace(-.15, .15, max(1, len(sizes))) if len(sizes) > 1 else [0.]
    lookup = {(r["variant"], r["n_society"]): r for r in source}
    limits = [0.]
    for record in source:
        limits.extend(record[name] * 100 for name in ("delta_alpha_c", "delta_ci_low", "delta_ci_high") if record[name] is not None)
    extent = max(max(abs(v) for v in limits) * 1.3, .1)
    ax.set_xlim(-extent, extent)
    ax.axvline(0, color="#AFA6B8", linewidth=.9, linestyle=(0, (3, 3)))
    for index, variant in enumerate(variants):
        for size_index, n in enumerate(sizes):
            record = lookup[(variant, n)]
            color, y = palette[size_index % len(palette)], index + offsets[size_index]
            if record["status"] == "resolved":
                point = record["delta_alpha_c"] * 100
                lo, hi = record["delta_ci_low"], record["delta_ci_high"]
                if lo is not None and hi is not None:
                    ax.hlines(y, lo * 100, hi * 100, color=color, linewidth=1.25)
                    ax.plot([lo * 100, hi * 100], [y, y], marker="|", linestyle="none", color=color, markersize=5)
                ax.plot(point, y, marker="o", markersize=4.7, markerfacecolor=color if lo is not None else "white", markeredgecolor=color)
            else:
                label = "unresolved"
                if record["variant_status"] in {"left_censored", "right_censored"} or record["baseline_status"] in {"left_censored", "right_censored"}:
                    label = "censored"
                label = "cens." if label == "censored" else "unresolved"
                ax.text(.985, y, f"{n}: {label}", color=color, fontsize=5.8, transform=ax.get_yaxis_transform(), va="center", ha="right")
    if not variants:
        ax.text(.5, .5, "No validated P04 contrasts", transform=ax.transAxes, ha="center")
    ax.set_yticks(range(len(variants)), [textwrap.fill(INTERVENTIONS.get(v, v.replace("_", " ")), width=18, break_long_words=False) for v in variants])
    ax.set_ylim(max(len(variants) - .45, .55), -.6)
    ax.set_xlabel("Change in αc vs. baseline\n(percentage points)", labelpad=8)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.grid(axis="y", color=COLORS["grid"], linewidth=.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7, labelsize=7)
    handles = [Line2D([], [], marker="o", color=palette[index % len(palette)], linestyle="none", markersize=4.5, label=f"N = {n:,}") for index, n in enumerate(sizes)]
    if handles:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.61, .91), ncol=max(1, len(handles)), frameon=False, columnspacing=1.2, fontsize=6.8)
    _footer(fig, model, fixture, "Differences in the empirical 50% collapse boundary. 95% intervals use paired seed resampling against baseline.\nCensored or missing contrasts are labeled; no effect or interval is imputed.")
    files = _save(fig, outdir, figure_id)
    files.append(_source(outdir, figure_id, source))
    return {"figure_id": figure_id, "families": ["p04"], "files": files,
            "status": "rendered" if source else "missing", "notes": [INTERVAL_NOTE]}


def render(rows, outdir, model, revision):
    """Return file/dependency metadata; never read data beyond supplied rows.

    The caller must enforce real-main manifest and complete-publication coverage
    requirements. This renderer intentionally allows marked fixtures for QA.
    """
    selected = [row for row in rows if row["backend"].get("model") == model and row["backend"].get("model_revision", "") == revision]
    if not selected:
        raise ValueError("No supplied rows match the requested model and revision")
    fixture = any(row.get("simulated", False) or row.get("stage") != "main" for row in selected)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p01 = [row for row in selected if row["family"] == "p01" and row["variant"] == "baseline" and row["config"]["scenario"] == "s1"]
    p04 = [row for row in selected if row["family"] == "p04" and row["config"]["scenario"] == "s1"]
    with plt.rc_context(_style()):
        return [_teaser(outdir, model, revision, fixture),
                _response(p01, outdir, model, revision, fixture),
                _finite_size(p01, outdir, model, revision, fixture),
                _interventions(p04, outdir, model, revision, fixture)]
