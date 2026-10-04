"""Feed verified new episode rows into the ORIGINAL publication plot functions.

Only schema adaptation, source exports and teaser composition live here. Styles,
layouts, curves, fits and forest plots come from the existing v3 figure code.
The historical CSV reader is never called. Production validation is owned by
figure_bundle; explicitly marked synthetic fixtures are only for rendering QA.
"""
from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_experiments_v3.figures import make_paper_figures as original
from .analysis import write_csv

TEMPLATE = Path(original.__file__).parent / "assets" / "teaser_layout.pdf"


def adapt_rows(rows):
    """Translate schema names, never load cached values or substitute defaults."""
    adapted = []
    for row in rows:
        c, result = row["config"], row["result"]
        values = {"job_id": row["job_id"], "family": row["family"], "variant": row["variant"],
                  "n_society": c["n_society"], "alpha": c["alpha"], "seed": c["seed"],
                  "primary_failure_rate": result["primary_failure_rate"],
                  "primary_failure_score_max": result["primary_failure_score_max"]}
        for name in ("alpha", "primary_failure_rate", "primary_failure_score_max"):
            if not math.isfinite(float(values[name])):
                raise ValueError(f"Missing/nonfinite new result: {row['job_id']}:{name}")
        adapted.append({key: str(value) for key, value in values.items()})
    return adapted


def _embed_figure(page, rect, fig):
    stream = BytesIO()
    fig.savefig(stream, format="pdf", facecolor="white")
    with fitz.open(stream=stream.getvalue(), filetype="pdf") as source:
        page.show_pdf_page(fitz.Rect(rect), source, 0)


def _teaser(outdir, response, scaling):
    """Reuse the original teaser artwork, replacing its two empirical panels.

    The tracked template was sanitized once: all old plot paths/text in A/B
    were removed, not merely covered by a white overlay. No results PDF is read.
    Curves below are the same Matplotlib artists used for the main figures.
    """
    document = fitz.open(TEMPLATE)
    page = document[0]
    response.delaxes(response.axes[1])
    for legend in list(response.legends):
        legend.remove()
    response.set_size_inches(5.6, 1.3)
    ax = response.axes[0]
    ax.set_position([.105, .29, .87, .65])
    for text in list(ax.texts)[:2]:  # Existing panel letter/title, already in artwork.
        text.remove()
    ax.set_ylabel(r"$p_N(\alpha)$", fontsize=7)
    ax.set_xlabel(r"harmful fraction $\alpha$", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6, pad=1)
    ax.legend(loc="upper left", fontsize=5.6, handletextpad=.15, labelspacing=.1)
    _embed_figure(page, (237, 30, 523, 95), response)

    for legend in list(scaling.legends):
        legend.remove()
    unresolved_notes = [text.get_text() for text in scaling.texts]
    for text in list(scaling.texts):
        text.remove()
    scaling.set_size_inches(5.6, .8)
    for index, ax in enumerate(scaling.axes):
        ax.set_position([.09 + index * .50, .34, .39, .60])
        # Main figure labels/fit annotations remain in Fig3/source tables;
        # these small panels retain the original teaser's compact layout.
        for text in list(ax.texts):
            text.remove()
        ax.set_xlabel(r"society size $N$", fontsize=6, labelpad=1)
        ax.set_ylabel(r"$\alpha_c(N)$" if index == 0 else r"$K_c(N)$", fontsize=6, labelpad=1)
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.tick_params(labelsize=5, pad=1, length=2)
        if not ax.lines:
            ax.text(.5, .5, "unresolved", transform=ax.transAxes, ha="center", fontsize=7)
    _embed_figure(page, (237, 115, 523, 150 if unresolved_notes else 154), scaling)
    if unresolved_notes:
        page.insert_text((239, 153), "; ".join(unresolved_notes), fontsize=3.5, color=(.5, .47, .53))
    document.save(outdir / "teaser.pdf", garbage=4, deflate=True)
    document.close()


def _fixture_watermark(path):
    with fitz.open(path) as source, fitz.open() as document:
        bounds = source[0].rect
        page = document.new_page(width=bounds.width, height=bounds.height + 14)
        page.show_pdf_page(fitz.Rect(0, 14, bounds.width, bounds.height + 14), source, 0)
        page.insert_text((8, 9), "SYNTHETIC FIXTURE - NOT SCIENTIFIC RESULTS", fontsize=5, color=(.75, .15, .25))
        contents = document.tobytes(garbage=4, deflate=True)
    path.write_bytes(contents)


def render(rows, outdir, model, revision):
    selected = [r for r in rows if r["backend"].get("model") == model
                and r["backend"].get("model_revision", "") == revision]
    if not selected:
        raise ValueError("No supplied rows match the requested model and revision")
    fixture = any(r.get("simulated", False) or r.get("stage") != "main" for r in selected)
    flat = adapt_rows(selected)
    p01 = [r for r in flat if r["family"] == "p01" and r["variant"] == "baseline"]
    p04 = [r for r in flat if r["family"] == "p04"]
    if not p01 or not p04:
        raise ValueError("Original paper plots require both new P01 and P04 rows")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        original.setup_style()
        response = original.figure2_p01_nonlinear_response(outdir, p01, p01)
        scaling = original.figure3_p01_finite_size_scaling(outdir, p01, p04, render_intervention=False)
        original.figure4_intervention_effects(outdir, p04)
        _teaser(outdir, response, scaling)
    write_csv(outdir / "fig2_nonlinear_response.source.csv", p01)
    write_csv(outdir / "fig3_finite_size_scaling.source.csv", original.p01_summaries(p01))
    write_csv(outdir / "fig4_intervention_effects.source.csv", p04)
    write_csv(outdir / "teaser.source.csv", p01)
    metadata = []
    for figure_id, family in [("teaser", "p01"), ("fig2_nonlinear_response", "p01"),
                              ("fig3_finite_size_scaling", "p01"), ("fig4_intervention_effects", "p04")]:
        pdf = outdir / f"{figure_id}.pdf"
        if fixture:
            _fixture_watermark(pdf)
        with fitz.open(pdf) as document:
            document[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72)).save(outdir / f"{figure_id}.png")
        files = [f"{figure_id}.{extension}" for extension in ("pdf", "png", "source.csv")]
        if figure_id == "fig3_finite_size_scaling":
            files += ["table2_scaling_results.csv", "table3_scaling_exponent_bootstrap.csv", "scaling_exponent_bootstrap.json"]
        metadata.append({"figure_id": figure_id, "families": [family], "files": files,
                         "status": "rendered", "notes": [
                             "Original publication plotting functions; values are from the selected new episodes only.",
                             "Unresolved boundaries/contrasts are labeled; no historical value is substituted."]})
    return metadata
