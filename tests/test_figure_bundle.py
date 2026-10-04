"""Publication provenance gates, using explicitly TEST_ONLY CPU fixtures.

The injected HTTP transport synthesizes completions. Passing a structural gate
here is not evidence of real model behavior or GPU execution. All artifacts
exist only in pytest's temporary directories.
"""
from copy import deepcopy
import gzip
import importlib
import json
from pathlib import Path
import shutil

import pytest

from paper_experiments.registry import build_cells, grid_key
from paper_experiments.runner import atomic_json, build_manifest, canonical, digest, run_manifest
from wolfbench.llm_runtime.backend import HTTPResult, VLLMBackend


TEST_MODEL = "TEST_ONLY_CPU_GATE_FIXTURE"
TEST_REVISION = "b" * 40
TEST_SOURCE = "e" * 64


def fixture_transport(url, body, headers, timeout):
    request = json.loads(body)
    decision = {"orders": [], "message": {"action": "none", "asset": "", "text": "",
                "sentiment": 0, "intensity": 0, "confidence": .5, "source_message_id": ""},
                "memory": "TEST_ONLY synthetic completion"}
    response = {"id": "chatcmpl-" + headers["X-Request-Id"], "model": request["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": canonical(decision)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    return HTTPResult(200, canonical(response).encode(), {"X-Request-Id": headers["X-Request-Id"]})


@pytest.fixture(scope="session")
def source_template(tmp_path_factory):
    """Run the current real-format episode loop through a fake HTTP transport."""
    settings = {"kind": "vllm", "model": TEST_MODEL, "model_revision": TEST_REVISION,
                "base_url": "http://127.0.0.1:8000/v1", "concurrency": 2,
                "max_tokens": 512, "temperature": .7, "timeout": 10., "retries": 0,
                "api_key_env": "WOLFBENCH_UNUSED_TEST_KEY", "chat_template_kwargs": None}
    cells = build_cells(["p01"], stage="smoke", sizes=[4, 6], seeds=[1, 2], horizon=2)
    cells += build_cells(["p04"], stage="smoke", sizes=[4], seeds=[1, 2], horizon=2)
    grid = {"schema": "full-llm-pilot-grid-v1", "frozen": True, "simulated": False,
            "pilot_seeds": [1001], "pilot_source_hashes": [TEST_SOURCE],
            "pilot_manifest_hashes": ["d" * 64], "pilot_backend_settings": [settings],
            "grids": {}, "protocols": {}}
    for cell in cells:
        c = cell["config"]
        key = grid_key(cell["family"], c["scenario"], cell["variant"], c["n_society"])
        grid["grids"][key] = [0., .2]
        grid["protocols"][key] = {k: v for k, v in c.items() if k not in {"alpha", "seed", "harmful_count"}}
    manifest = build_manifest(cells, stage="main", backend=settings, source_hash=TEST_SOURCE, grid=grid)
    backend = VLLMBackend(TEST_MODEL, model_revision=TEST_REVISION, concurrency=2,
                          max_tokens=512, temperature=.7, request_timeout=10., retries=0,
                          transport=fixture_transport)
    status = run_manifest(manifest, tmp_path_factory.mktemp("test_only_bundle_fixture"),
                          backend=backend, check_source=False)
    backend.close()
    return Path(status["run_dir"])


@pytest.fixture
def source(source_template, tmp_path):
    target = tmp_path / "TEST_ONLY_input"
    shutil.copytree(source_template, target)
    return target


@pytest.fixture
def gate(monkeypatch):
    module = importlib.import_module("paper_experiments.figure_bundle")
    monkeypatch.setattr(module, "PAPER_SIZES", [4, 6])
    monkeypatch.setattr(module, "INTERVENTION_SIZES", [4])
    monkeypatch.setattr(module, "MIN_MAIN_SEEDS", 2)
    monkeypatch.setattr(module, "EXPECTED_HORIZON", 2)
    monkeypatch.setattr(module, "code_fingerprint", lambda: TEST_SOURCE)
    return module


def manifest(source):
    return json.loads((source / "manifest.json").read_text())


def episode(source):
    job = next(j for j in manifest(source)["jobs"] if j["family"] == "p01")
    path = source / "episodes" / f"{job['job_id']}.json"
    return path, json.loads(path.read_text())


def trace(source, row, name):
    with gzip.open(source / row["traces"][name]["path"], "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def replace_trace(source, path, row, name, records):
    target = source / row["traces"][name]["path"]
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical(record) + "\n")
    import hashlib
    row["traces"][name]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    row["traces"][name]["rows"] = len(records)
    atomic_json(path, row)


def reseal_manifest(value):
    """Make mutations checksum-valid, so semantic gate tests remain decisive."""
    provenance = {k: value[k] for k in ("schema", "source_hash", "backend", "stage", "grid_hash")}
    value["run_fingerprint"] = digest(provenance)
    for job in value["jobs"]:
        cell = {k: v for k, v in job.items() if k != "job_id"}
        job["job_id"] = digest({"run_fingerprint": value["run_fingerprint"], "cell": cell})
    value["manifest_hash"] = digest({"run_fingerprint": value["run_fingerprint"], "jobs": value["jobs"]})
    return value


def validate(gate, source):
    return gate.validate_sources([source], TEST_MODEL, TEST_REVISION)


def test_current_complete_pinned_main_fixture_validates_and_bundle_verifies(gate, source, tmp_path):
    rows, sources, coverage = validate(gate, source)
    assert len(rows) == len(manifest(source)["jobs"])
    assert sources and coverage
    bundle_path = gate.build_bundle([source], tmp_path / "TEST_ONLY_bundle", TEST_MODEL, TEST_REVISION)
    assert isinstance(bundle_path, Path) and bundle_path.is_file()
    verified = gate.verify_bundle(bundle_path)
    assert isinstance(verified, dict) and verified


@pytest.mark.parametrize("mutation", ["mock", "pilot", "smoke", "source", "schema", "model", "revision"])
def test_checksum_valid_ineligible_protocols_cannot_enter_figures(gate, source, mutation):
    value = manifest(source)
    if mutation == "mock":
        value["backend"]["kind"] = "mock"
    elif mutation in {"pilot", "smoke"}:
        value["stage"] = mutation
    elif mutation == "source":
        value["source_hash"] = "a" * 64
    elif mutation == "schema":
        value["schema"] = "historical-hybrid-v3"
    elif mutation == "model":
        value["backend"]["model"] = "other-model"
    else:
        value["backend"]["model_revision"] = "a" * 40
    atomic_json(source / "manifest.json", reseal_manifest(value))
    with pytest.raises(ValueError):
        validate(gate, source)


def test_partial_run_is_rejected_instead_of_silently_dropping_missing_episode(gate, source):
    path, _ = episode(source)
    path.unlink()
    with pytest.raises(ValueError):
        validate(gate, source)


def test_missing_intervention_family_is_rejected_even_when_remaining_manifest_is_complete(gate, source):
    value = manifest(source)
    value["jobs"] = [j for j in value["jobs"] if j["family"] == "p01"]
    atomic_json(source / "manifest.json", reseal_manifest(value))
    with pytest.raises(ValueError):
        validate(gate, source)


def test_empty_traces_cannot_forge_episode_completeness(gate, source):
    path, row = episode(source)
    row["traces"] = {}
    atomic_json(path, row)
    with pytest.raises(ValueError):
        validate(gate, source)


@pytest.mark.parametrize("name", ["request_log", "backend_audit", "decision_log", "round_snapshots", "daily_log",
                                  "exposure_log", "message_log", "information_events", "execution_audit", "trade_log"])
def test_missing_required_trace_is_rejected(gate, source, name):
    path, row = episode(source)
    row["traces"].pop(name)
    atomic_json(path, row)
    with pytest.raises(ValueError):
        validate(gate, source)


def test_actual_trace_record_count_is_checked(gate, source):
    path, row = episode(source)
    row["traces"]["request_log"]["rows"] += 1
    atomic_json(path, row)
    with pytest.raises(ValueError):
        validate(gate, source)


def test_duplicate_requests_cannot_substitute_for_missing_agent_round(gate, source):
    path, row = episode(source)
    records = trace(source, row, "request_log")
    records[-1] = deepcopy(records[0])
    replace_trace(source, path, row, "request_log", records)
    with pytest.raises(ValueError):
        validate(gate, source)


@pytest.mark.parametrize("mutation", ["participant", "day", "seed", "system"])
def test_request_identity_and_original_input_are_bound_to_backend_audit(gate, source, mutation):
    path, row = episode(source)
    records = trace(source, row, "request_log")
    if mutation in {"participant", "day"}:
        user = json.loads(records[0]["user"])
        user[mutation] = "p999999" if mutation == "participant" else 999
        records[0]["user"] = canonical(user)
    elif mutation == "seed":
        records[0]["seed"] += 1
    else:
        records[0]["system"] += " Modified after completion."
    replace_trace(source, path, row, "request_log", records)
    with pytest.raises(ValueError):
        validate(gate, source)


@pytest.mark.parametrize("mutation", ["failed", "simulated", "model", "missing_id", "wrong_id", "truncated", "content", "payload_hash"])
def test_invalid_or_substitute_completion_cannot_be_marked_complete(gate, source, mutation):
    path, row = episode(source)
    records = trace(source, row, "backend_audit")
    if mutation == "failed":
        records[0]["status"] = "failure"
    elif mutation == "simulated":
        records[0]["simulated"] = True
    elif mutation == "payload_hash":
        records[0]["payload_sha256"] = "a" * 64
    else:
        raw = json.loads(records[0]["attempts"][-1]["raw_response"])
        if mutation == "model":
            raw["model"] = "historical-model"
        elif mutation == "missing_id":
            raw.pop("id")
        elif mutation == "wrong_id":
            raw["id"] = "chatcmpl-substituted-completion"
        elif mutation == "truncated":
            raw["choices"][0]["finish_reason"] = "length"
        else:
            content = json.loads(raw["choices"][0]["message"]["content"])
            content["memory"] = "Different from the audited direct decision"
            raw["choices"][0]["message"]["content"] = canonical(content)
        records[0]["attempts"][-1]["raw_response"] = canonical(raw)
    replace_trace(source, path, row, "backend_audit", records)
    with pytest.raises(ValueError):
        validate(gate, source)


def test_applied_decision_must_match_successful_raw_completion(gate, source):
    path, row = episode(source)
    records = trace(source, row, "decision_log")
    records[0]["raw_decision"]["memory"] = "Substituted policy decision"
    replace_trace(source, path, row, "decision_log", records)
    with pytest.raises(ValueError):
        validate(gate, source)


def test_runtime_version_inside_result_cannot_be_relabelled_with_a_new_hash(gate, source):
    path, row = episode(source)
    row["result"]["runtime_version"] = "historical-hybrid-v3"
    row["result_hash"] = digest(row["result"])
    atomic_json(path, row)
    with pytest.raises(ValueError):
        validate(gate, source)


@pytest.mark.parametrize("mutation", ["declared_calls", "fallback_count", "primary_indicator"])
def test_result_counters_and_failure_indicator_are_bound_to_traces(gate, source, mutation):
    path, row = episode(source)
    if mutation == "declared_calls":
        row["result"]["n_llm_decisions"] += 1
    elif mutation == "fallback_count":
        row["result"]["rule_fallback_count"] = 1
    else:
        row["result"]["primary_failure_rate"] = 1
    row["result_hash"] = digest(row["result"])
    atomic_json(path, row)
    with pytest.raises(ValueError):
        validate(gate, source)


@pytest.mark.parametrize("mode", ["absolute", "parent", "symlink"])
def test_trace_paths_cannot_escape_selected_run_directory(gate, source, tmp_path, mode):
    path, row = episode(source)
    original = source / row["traces"]["daily_log"]["path"]
    external = tmp_path / "OUTSIDE_SELECTED_RUN.jsonl.gz"
    shutil.copyfile(original, external)
    if mode == "absolute":
        row["traces"]["daily_log"]["path"] = str(external)
    elif mode == "parent":
        row["traces"]["daily_log"]["path"] = "../" + external.name
    else:
        original.unlink()
        original.symlink_to(external)
    atomic_json(path, row)
    with pytest.raises(ValueError):
        validate(gate, source)


def test_corrupted_compressed_trace_is_rejected_before_analysis(gate, source):
    _, row = episode(source)
    (source / row["traces"]["daily_log"]["path"]).write_bytes(b"historical or corrupted payload")
    with pytest.raises(ValueError):
        validate(gate, source)


def test_bundle_reverification_detects_source_episode_bytes_changed_after_build(gate, source, tmp_path):
    bundle_path = gate.build_bundle([source], tmp_path / "TEST_ONLY_bundle", TEST_MODEL, TEST_REVISION)
    path, _ = episode(source)
    path.write_text(path.read_text() + " \n")
    with pytest.raises(ValueError):
        gate.verify_bundle(bundle_path)


def test_bundle_manifest_itself_cannot_be_modified_after_build(gate, source, tmp_path):
    bundle_path = gate.build_bundle([source], tmp_path / "TEST_ONLY_bundle", TEST_MODEL, TEST_REVISION)
    value = json.loads(bundle_path.read_text())
    value["unexpected_modified_field"] = "modified"
    atomic_json(bundle_path, value)
    with pytest.raises(ValueError):
        gate.verify_bundle(bundle_path)


def test_same_run_cannot_be_counted_twice(gate, source):
    with pytest.raises(ValueError):
        gate.validate_sources([source, source], TEST_MODEL, TEST_REVISION)


@pytest.mark.parametrize("gap", ["size", "variant", "baseline", "all_second_seeds", "one_coordinate_seed"])
def test_complete_manifest_cannot_hide_missing_planned_comparison_cells(gate, source, gap):
    value = manifest(source)

    def remove(job):
        c = job["config"]
        if gap == "size":
            return job["family"] == "p01" and c["n_society"] == 6
        if gap == "variant":
            return job["family"] == "p04" and job["variant"] == "no_feedback"
        if gap == "baseline":
            return job["family"] == "p01" and c["n_society"] == 4 and c["alpha"] == 0
        if gap == "all_second_seeds":
            return c["seed"] == 2
        return job["family"] == "p01" and c["n_society"] == 4 and c["alpha"] > 0 and c["seed"] == 2

    value["jobs"] = [job for job in value["jobs"] if not remove(job)]
    atomic_json(source / "manifest.json", reseal_manifest(value))
    with pytest.raises(ValueError):
        validate(gate, source)


@pytest.mark.parametrize("kind", ["pdf", "source_table"])
def test_release_verification_detects_modified_figure_or_source_table(gate, source, tmp_path, kind):
    bundle_path = gate.build_bundle([source], tmp_path / "TEST_ONLY_bundle", TEST_MODEL, TEST_REVISION)
    bundle = json.loads(bundle_path.read_text())
    suffix = ".pdf" if kind == "pdf" else ".source.csv"
    relative = next(name for name in bundle["artifacts"] if name.endswith(suffix))
    artifact = bundle_path.parent / relative
    artifact.write_bytes(artifact.read_bytes() + b"\nModified artifact bytes\n")
    with pytest.raises(ValueError):
        gate.verify_bundle(bundle_path)


def test_existing_release_is_immutable_and_cannot_receive_replacement_artifacts(gate, source, tmp_path):
    output = tmp_path / "TEST_ONLY_bundle"
    output.mkdir()
    sentinel = output / "old_data_placeholder.pdf"
    sentinel.write_bytes(b"TEST_ONLY do not replace existing release")
    with pytest.raises(ValueError):
        gate.build_bundle([source], output, TEST_MODEL, TEST_REVISION)
    assert sentinel.read_bytes() == b"TEST_ONLY do not replace existing release"
    assert not (output / "figure_manifest.json").exists()


@pytest.mark.parametrize("mutable_revision", ["main", "latest", "v1", "weights-sha-123"])
def test_publication_gate_requires_immutable_weight_content_pin(gate, source, mutable_revision):
    # Matching a mutable tag is insufficient. The error must come from the
    # content-pin requirement, not merely the fixture's different model tag.
    with pytest.raises(ValueError, match="[Ii]mmutable"):
        gate.validate_sources([source], TEST_MODEL, mutable_revision)
