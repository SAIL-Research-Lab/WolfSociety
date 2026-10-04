"""Protocol, checkpoint and statistical-boundary tests; no GPU/network needed."""
from copy import deepcopy
import json

import pytest

from paper_experiments.analysis import analyze, crossing, curve
from paper_experiments.grids import derive_grid
from paper_experiments.registry import FAMILIES, build_cells, grid_key
from paper_experiments.runner import build_manifest, budget, create_backend, expected_llm_decisions, json_object, load_complete, run_manifest, validate_frozen_backend, validate_manifest


class FakeBackend:
    simulated = True


def fake_episode(config, backend):
    return {"primary_failure_rate": float(config["alpha"] >= .1),
            "primary_failure_score_max": config["alpha"] * 10,
            "alpha_realized": round(config["alpha"] * config["n_society"]) / config["n_society"],
            "daily_log": [{"day": 1, "primary_failure": {"primary_failure_score": config["alpha"] * 10,
              "components": {"social_cascade": config["alpha"]}, "thresholds": {"social_cascade": .1}, "evaluation_start_day": 0}}]}


def make_manifest(families=("p01",), sizes=(20,), seeds=(9001,)):
    return build_manifest(build_cells(families, stage="smoke", sizes=sizes, seeds=seeds, horizon=2),
                          stage="smoke", backend={"kind": "mock", "model": "fixture", "max_tokens": 128}, source_hash="unit-test-source")


def test_registry_final_families_clean_baselines_and_full_llm():
    assert "p08" not in FAMILIES
    assert {"p01", "p02", "p02b", "p03", "p04", "p05", "p06", "p07", "p09", "p10", "p11", "language", "controller_factorial"} == set(FAMILIES)
    cells = build_cells(["all"], stage="smoke", sizes=[20], seeds=[9001], horizon=2)
    groups = {}
    for cell in cells:
        c = cell["config"]
        groups.setdefault((cell["family"], cell["variant"], c["scenario"], c["n_society"]), []).append(c)
        if cell["family"] in {"p01", "p02", "p02b", "p03", "p04", "p05", "p07", "p09", "language"}:
            assert c["controller"] == "llm"
    assert all(any(c["alpha"] == 0 for c in group) for group in groups.values())
    exact = [cell["config"] for cell in cells if cell["family"] == "p02"]
    assert all(c["alpha"] == c["harmful_count"] / c["n_society"] for c in exact)


def test_main_requires_frozen_disjoint_grid():
    with pytest.raises(ValueError, match="frozen"):
        build_cells(["p01"], stage="main")
    grid = {"frozen": True, "pilot_seeds": [1001], "grids": {grid_key("p01", "s1", "baseline", 100): [0, .1, .2]}}
    with pytest.raises(ValueError, match="disjoint"):
        build_cells(["p01"], stage="main", sizes=[100], seeds=[1001], grids=grid)
    cells = build_cells(["p01"], stage="main", sizes=[100], seeds=[1], grids=grid)
    assert len(cells) == 3
    grid["protocols"] = {grid_key("p01", "s1", "baseline", 100): {}}
    with pytest.raises(ValueError, match="protocol differs"):
        build_cells(["p01"], stage="main", sizes=[100], seeds=[1], grids=grid)


def test_manifest_identity_covers_backend_source_seed_and_config():
    manifest = make_manifest()
    validate_manifest(manifest, check_source=False)
    altered = deepcopy(manifest)
    altered["jobs"][0]["config"]["seed"] += 1
    with pytest.raises(ValueError, match="modified"):
        validate_manifest(altered, check_source=False)
    cells = [{k: v for k, v in job.items() if k != "job_id"} for job in manifest["jobs"]]
    changed_backend = build_manifest(cells, stage="smoke", backend={"kind": "mock", "model": "another"}, source_hash="unit-test-source")
    changed_source = build_manifest(cells, stage="smoke", backend=manifest["backend"], source_hash="new-source")
    assert manifest["jobs"][0]["job_id"] != changed_backend["jobs"][0]["job_id"]
    assert manifest["jobs"][0]["job_id"] != changed_source["jobs"][0]["job_id"]


def test_resume_requires_complete_config_and_trace_checksums(tmp_path):
    manifest = make_manifest()
    counter = []
    def episode(config, backend):
        counter.append(config["alpha"])
        return fake_episode(config, backend)
    first = run_manifest(manifest, tmp_path, backend=FakeBackend(), episode_fn=episode, check_source=False)
    assert first["completed_now"] == 2
    second = run_manifest(manifest, tmp_path, backend=FakeBackend(), episode_fn=episode, check_source=False)
    assert second["resumed_complete"] == 2 and len(counter) == 2
    run_dir = tmp_path / "mock" / manifest["manifest_hash"][:16]
    job = manifest["jobs"][0]
    path = run_dir / "episodes" / f"{job['job_id']}.json"
    row = json.loads(path.read_text())
    trace = run_dir / row["traces"]["daily_log"]["path"]
    trace.write_bytes(b"corrupted")
    assert load_complete(path, job, manifest) is None
    third = run_manifest(manifest, tmp_path, backend=FakeBackend(), episode_fn=episode, check_source=False)
    assert third["completed_now"] == 1 and len(counter) == 3


def test_failure_never_marks_complete_and_resume_reruns(tmp_path):
    manifest = make_manifest()
    def fail(config, backend):
        raise RuntimeError("expected synthetic transport failure")
    with pytest.raises(RuntimeError):
        run_manifest(manifest, tmp_path, backend=FakeBackend(), episode_fn=fail, check_source=False)
    run_dir = tmp_path / "mock" / manifest["manifest_hash"][:16]
    assert not list((run_dir / "episodes").glob("*.json"))
    assert list((run_dir / "errors").glob("*.json"))
    assert not list((run_dir / "episodes").glob("*.lock"))
    result = run_manifest(manifest, tmp_path, backend=FakeBackend(), episode_fn=fake_episode, check_source=False)
    assert result["completed_now"] == 2


def test_rule_null_budget_and_mock_separation(tmp_path):
    manifest = make_manifest(("p01", "p11", "controller_factorial"))
    assert budget(manifest)["planned_llm_generations"] > 0
    null = [j for j in manifest["jobs"] if j["family"] == "p11"]
    null_only = build_manifest([{k: v for k, v in j.items() if k != "job_id"} for j in null], stage="smoke", backend=manifest["backend"], source_hash="unit-test-source")
    assert budget(null_only)["planned_llm_generations"] == 0
    real_backend = FakeBackend()
    real_backend.simulated = False
    with pytest.raises(ValueError, match="simulation marker"):
        run_manifest(manifest, tmp_path, backend=real_backend, episode_fn=fake_episode, check_source=False)


def test_mixed_budget_rounds_each_population_and_respects_group_overrides():
    job = {"engine": "society", "config": {"controller": "mixed", "n_society": 6, "alpha": .5, "horizon_days": 2, "llm_fraction": .5}}
    # Three benign and three harmful: Python round(1.5) gives 2 in each group.
    assert expected_llm_decisions(job) == 8
    job["config"].update(controller="rule", benign_controller="llm", harmful_controller="mixed")
    assert expected_llm_decisions(job) == 10


def test_real_backend_constructor_preserves_model_revision_without_network_calls():
    backend = create_backend({"kind": "vllm", "model": "fixture-pinned", "model_revision": "weights-sha-123",
                              "base_url": "http://127.0.0.1:8000/v1", "timeout": 60, "concurrency": 2,
                              "max_tokens": 128, "temperature": .7, "retries": 2, "api_key_env": "WOLFBENCH_TEST_UNUSED_KEY",
                              "chat_template_kwargs": json_object('{"enable_thinking":false}')})
    assert backend.provenance["model_revision"] == "weights-sha-123"
    assert backend.provenance["chat_template_kwargs"] == {"enable_thinking": False}
    assert backend.stats["remote_attempts"] == 0
    import argparse
    for invalid in ["[]", "false", "not-json"]:
        with pytest.raises(argparse.ArgumentTypeError):
            json_object(invalid)


def test_crossing_censoring_and_multiple_crossings():
    assert crossing([(0, 0, 12), (.1, 1, 12)])["alpha_c"] == pytest.approx(.05)
    assert crossing([(0, 0, 12), (.1, .2, 12)])["status"] == "right_censored"
    assert crossing([(0, .5, 12), (.1, 1, 12)])["status"] == "left_censored"
    assert crossing([(.1, 0, 12), (.2, 1, 12)])["status"] == "missing_clean_baseline"
    assert crossing([(0, 0, 12), (.1, 1, 12), (.2, 0, 12), (.3, 1, 12)])["status"] == "multiple_crossings"


def test_realized_fraction_rounding_does_not_inflate_independent_seed_count():
    rows = [{"config": {"alpha": alpha, "n_society": 20, "seed": seed},
             "result": {"alpha_realized": 0, "primary_failure_rate": 0}}
            for alpha in [0, .01] for seed in [1, 2]]
    assert curve(rows, "realized") == [(0, 0, 2)]


def test_mock_analysis_freeze_is_explicit_and_reports_coverage(tmp_path):
    manifest = make_manifest(sizes=[20, 40, 80], seeds=[9001, 9002])
    status = run_manifest(manifest, tmp_path, backend=FakeBackend(), episode_fn=fake_episode, check_source=False)
    with pytest.raises(ValueError, match="Mock data"):
        analyze([status["run_dir"]], tmp_path / "bad")
    report = analyze([status["run_dir"]], tmp_path / "analysis", allow_mock=True, draws=10)
    assert report["status"] == "SIMULATED_PIPELINE_QA_NOT_SCIENTIFIC"
    assert report["complete_dataset"]
    grid = derive_grid([status["run_dir"]], freeze=True, allow_mock=True)
    assert grid["simulated"] and grid["frozen"] and len(grid["grids"]) == 3
    assert set(grid["pilot_seeds"]) == {9001, 9002}
    assert grid["pilot_backend_settings"] == [manifest["backend"]]
    assert (tmp_path / "analysis" / "boundaries.csv").exists()


def test_stable_shards_are_disjoint_and_cover_manifest(tmp_path):
    manifest = make_manifest(sizes=[20, 40], seeds=[9001, 9002])
    outputs = []
    for shard in range(3):
        outputs.append(run_manifest(manifest, tmp_path, backend=FakeBackend(), episode_fn=fake_episode,
                                    shard_index=shard, num_shards=3, check_source=False))
    assert sum(r["completed_now"] for r in outputs) == len(manifest["jobs"])
    assert sum(r["resumed_complete"] for r in outputs) == 0


def test_main_appendix_analysis_tables_are_derived_from_seeded_fixture(tmp_path):
    manifest = make_manifest(families=["p01", "p02", "p02b"], sizes=[20, 40, 80], seeds=[9001, 9002])
    status = run_manifest(manifest, tmp_path, backend=FakeBackend(), episode_fn=fake_episode, check_source=False)
    report = analyze([status["run_dir"]], tmp_path / "tables", allow_mock=True, draws=10)
    for table in ("boundaries", "scaling_bootstrap", "drop_one_size", "threshold_components", "fixed_count_response",
                  "fixed_count_per_k", "heldout_response", "heldout_comparisons", "call_token_audit"):
        assert report["table_rows"][table] > 0, table


def test_threshold_sensitivity_uses_recorded_first_round_evaluation_not_legacy_grace(monkeypatch):
    from paper_experiments import analysis
    rows = [{"family": "p01", "variant": "baseline", "backend": {"model": "fixture"},
             "stage": "smoke", "simulated": True, "job_id": str(alpha),
             "config": {"scenario": "s1", "n_society": 20, "alpha": alpha, "seed": 1},
             "result": {"primary_failure_rate": int(alpha > 0), "alpha_realized": alpha},
             "fixture_score": score} for alpha, score in [(0., 0.), (.2, 1.2)]]
    monkeypatch.setattr(analysis, "read_trace", lambda row, _: [
        {"day": 0, "primary_failure": {"evaluation_start_day": 0, "primary_failure_score": row["fixture_score"], "components": {}, "thresholds": {}}},
        {"day": 1, "primary_failure": {"evaluation_start_day": 0, "primary_failure_score": 0., "components": {}, "thresholds": {}}},
    ])
    boundaries, diagnostics = analysis.sensitivity_rows(rows)
    joint = [r for r in boundaries if r["variant"] == "baseline:joint:threshold_1" and r["alpha_mode"] == "target"]
    assert len(joint) == 1
    assert joint[0]["status"] == "resolved"
    assert joint[0]["alpha_c"] == pytest.approx(.1)
    assert diagnostics == []


def test_real_main_preserves_frozen_sampling_and_requires_evidence():
    settings = {"kind": "vllm", "model": "model-a", "model_revision": "rev-a", "max_tokens": 512,
                "temperature": .7, "chat_template_kwargs": {"enable_thinking": False},
                "base_url": "http://first-host/v1", "concurrency": 32}
    grid = {"pilot_backend_settings": [settings]}
    validate_frozen_backend(grid, settings, {"p01", "p05"})
    changed_transport = {**settings, "base_url": "http://second-host/v1", "concurrency": 64}
    validate_frozen_backend(grid, changed_transport, {"p01"})
    for name, changed in [("model", "model-b"), ("model_revision", "rev-b"), ("max_tokens", 1024),
                          ("temperature", .8), ("chat_template_kwargs", {"enable_thinking": True})]:
        with pytest.raises(ValueError, match="differ from frozen"):
            validate_frozen_backend(grid, {**settings, name: changed}, {"p01"})
    for old_grid in [{}, {"pilot_backend_settings": []}, {"pilot_backend_settings": [{"kind": "vllm", "model": "model-a"}]}]:
        with pytest.raises(ValueError, match="lacks complete"):
            validate_frozen_backend(old_grid, settings, {"p01"})
    with pytest.raises(ValueError, match="lacks complete"):
        build_manifest([], stage="main", backend=settings, source_hash="fixture", grid={"frozen": True})


def test_only_p07_can_change_backbone_under_matched_frozen_sampling():
    settings = {"kind": "vllm", "model": "model-a", "model_revision": "rev-a", "max_tokens": 512,
                "temperature": .7, "chat_template_kwargs": None}
    grid = {"pilot_backend_settings": [settings]}
    second = {**settings, "model": "model-b", "model_revision": "rev-b"}
    validate_frozen_backend(grid, second, {"p07"})
    with pytest.raises(ValueError, match="differ from frozen"):
        validate_frozen_backend(grid, second, {"p07", "p01"})
    with pytest.raises(ValueError, match="differ from frozen"):
        validate_frozen_backend(grid, {**second, "temperature": .9}, {"p07"})
