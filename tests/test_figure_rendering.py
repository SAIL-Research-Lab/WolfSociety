"""Rendering fixtures only: these are never scientific experiment evidence."""
import csv

import pytest

import fitz
from paper_experiments.figure_rendering import TEMPLATE, adapt_rows, original, render


def figure_fixture():
    rows = []
    alphas = [0., .05, .1, .2, .4]
    base = {100: .12, 200: .08, 300: .10, 500: .06, 1000: .07, 2000: .8}
    effects = {"baseline": 1., "weak_feedback": 1.5, "strong_feedback": .6,
               "high_conformity": 1.1, "high_reach": .75, "high_attention": 1.25,
               "high_deliberation": 1., "no_feedback": .85, "no_multihop": 1.4}
    for family, sizes, variants in [("p01", list(base), {"baseline": 1.}),
                                    ("p04", [300, 1000], effects)]:
        for n in sizes:
            for variant, factor in variants.items():
                for alpha in alphas:
                    for seed in range(1, 13):
                        cutoff = base[n] * factor * (.6 + .8 * (seed - 1) / 11)
                        failure = float(alpha >= cutoff)
                        if family == "p04" and variant == "no_feedback" and n == 1000:
                            failure = 1.  # explicit clean-baseline failure => left censoring
                        rows.append({"family": family, "variant": variant, "job_id": f"fixture:{family}:{n}:{variant}:{alpha}:{seed}",
                                     "config": {"scenario": "s1", "n_society": n, "alpha": alpha, "seed": seed},
                                     "backend": {"model": "SYNTHETIC-FIXTURE", "model_revision": "fixture-v1"},
                                     "stage": "fixture", "simulated": True,
                                     "result": {"primary_failure_rate": failure, "primary_failure_score_max": alpha / max(cutoff, .001), "alpha_realized": round(alpha * n) / n}})
    return rows


@pytest.fixture
def quick_bootstrap(monkeypatch):
    # CPU visual fixtures only; production retains the original bootstrap budgets.
    for name in ("bootstrap_nu_stats", "bootstrap_alpha_c_ci", "p04_effect_statistics"):
        function = getattr(original, name)
        def limited(*args, _function=function, **kwargs):
            return _function(*args, **{**kwargs, "n_boot": 40})
        monkeypatch.setattr(original, name, limited)


def test_original_plot_functions_receive_only_new_rows(tmp_path, monkeypatch, quick_bootstrap):
    calls = []
    for name in ("figure2_p01_nonlinear_response", "figure3_p01_finite_size_scaling", "figure4_intervention_effects"):
        function = getattr(original, name)
        def record(*args, _name=name, _function=function, **kwargs):
            calls.append((_name, args[1]))
            return _function(*args, **kwargs)
        monkeypatch.setattr(original, name, record)
    def no_historical_reads(*args, **kwargs):
        raise AssertionError("Historical data reader must never be called")
    monkeypatch.setattr(original, "read_csv", no_historical_reads)
    monkeypatch.setattr(original, "read_rows", no_historical_reads)
    metadata = render(figure_fixture(), tmp_path, "SYNTHETIC-FIXTURE", "fixture-v1")
    assert len(calls) == 3
    assert all(row["job_id"].startswith("fixture:") for _, rows in calls for row in rows)
    assert {item["figure_id"] for item in metadata} == {
        "teaser", "fig2_nonlinear_response", "fig3_finite_size_scaling", "fig4_intervention_effects"}
    for item in metadata:
        assert all((tmp_path / name).stat().st_size for name in item["files"])
        with fitz.open(tmp_path / (item["figure_id"] + ".pdf")) as document:
            assert "SYNTHETIC FIXTURE" in document[0].get_text()
    with (tmp_path / "fig2_nonlinear_response.source.csv").open() as handle:
        sources = list(csv.DictReader(handle))
    assert max(float(row["alpha"]) for row in sources) == .4
    assert all("primary_failure_score_max" in row for row in sources)
    with (tmp_path / "fig3_finite_size_scaling.source.csv").open() as handle:
        scaling = list(csv.DictReader(handle))
    censored = next(row for row in scaling if row["N"] == "2000")
    assert censored["status"] == "censored" and censored["alpha_c"] == ""
    with fitz.open(tmp_path / "fig2_nonlinear_response.pdf") as document:
        text = document[0].get_text()
        assert "Episode severity" in text and "Collapse probability" in text
    with fitz.open(tmp_path / "teaser.pdf") as document:
        text = document[0].get_text()
        assert "Closed-loop society" in text and "Core signatures" in text


def test_missing_new_severity_cannot_be_filled_from_old_values():
    rows = figure_fixture()[:1]
    rows[0]["result"].pop("primary_failure_score_max")
    with pytest.raises(KeyError):
        adapt_rows(rows)


def test_old_threshold_annotation_and_display_window_are_data_driven(tmp_path, quick_bootstrap):
    rows = adapt_rows([r for r in figure_fixture() if r["family"] == "p01"])
    # Move the full transition away from the historical range.
    for row in rows:
        row["primary_failure_rate"] = str(float(float(row["alpha"]) >= .2))
    assert original.p01_display_xmax(rows) == .4
    fig = original.figure2_p01_nonlinear_response(tmp_path, rows, rows)
    annotation = " ".join(text.get_text() for text in fig.axes[0].texts)
    assert "0.022" not in annotation and "0.047" not in annotation
    assert "0.150" in annotation
    assert len(fig.axes) == 2 and fig.axes[0].get_xlim()[1] == .4


def test_template_contains_no_historical_chart_text():
    with fitz.open(TEMPLATE) as document:
        for rect in [(237, 30, 523, 95), (237, 115, 523, 154)]:
            assert not document[0].get_text(clip=fitz.Rect(rect)).strip()


def test_unresolved_p04_contrast_is_not_averaged_over_subset():
    rows = adapt_rows([r for r in figure_fixture() if r["family"] == "p04"])
    effects = dict(original.p04_effect_items(rows))
    assert "no_feedback" not in effects  # N=1000 has clean-baseline failure.
    assert "high_deliberation" in effects and effects["high_deliberation"] == [0., 0.]


def test_renderer_requires_explicit_matching_model(tmp_path):
    with pytest.raises(ValueError, match="No supplied rows"):
        render(figure_fixture(), tmp_path, "wrong-model", "fixture-v1")
