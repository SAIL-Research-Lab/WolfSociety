"""Rendering fixtures only: these are never scientific experiment evidence."""
import csv

import pytest

from paper_experiments.figure_rendering import _boundary_statistics, render


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
                                     "result": {"primary_failure_rate": failure, "alpha_realized": round(alpha * n) / n}})
    return rows


def test_renderer_outputs_all_four_figures_and_records_censoring(tmp_path):
    metadata = render(figure_fixture(), tmp_path, "SYNTHETIC-FIXTURE", "fixture-v1")
    assert {item["figure_id"] for item in metadata} == {
        "teaser", "fig2_nonlinear_response", "fig3_finite_size_scaling", "fig4_intervention_effects"}
    for item in metadata:
        assert len(item["files"]) == 3
        assert all((tmp_path / filename).is_file() for filename in item["files"])
        assert all((tmp_path / filename).stat().st_size > 0 for filename in item["files"])
    with (tmp_path / "fig3_finite_size_scaling.source.csv").open() as handle:
        scaling = list(csv.DictReader(handle))
    censored = next(row for row in scaling if row["n_society"] == "2000")
    assert censored["status"] == "right_censored"
    assert censored["alpha_c"] == "" and censored["k_c"] == ""
    with (tmp_path / "fig4_intervention_effects.source.csv").open() as handle:
        effects = list(csv.DictReader(handle))
    unresolved = next(row for row in effects if row["n_society"] == "1000" and row["variant"] == "no_feedback")
    assert unresolved["status"] == "unresolved" and unresolved["delta_alpha_c"] == ""
    values = [float(row["delta_alpha_c"]) for row in effects if row["delta_alpha_c"]]
    assert min(values) < 0 < max(values)
    paired_zero = [row for row in effects if row["variant"] == "high_deliberation"]
    assert all(float(row["delta_ci_low"]) == float(row["delta_ci_high"]) == 0 for row in paired_zero)


def test_paired_bootstrap_shares_seed_draws_between_identical_conditions():
    rows = [row for row in figure_fixture() if row["family"] == "p04" and row["variant"] in {"baseline", "high_deliberation"}]
    records, samples = _boundary_statistics(rows)
    for n in (300, 1000):
        assert (samples[("baseline", n)] == samples[("high_deliberation", n)]).all()
        assert records[("baseline", n)]["bootstrap_paired_seeds"] == 12


def test_renderer_requires_explicit_matching_model(tmp_path):
    with pytest.raises(ValueError, match="No supplied rows"):
        render(figure_fixture(), tmp_path, "wrong-model", "fixture-v1")
