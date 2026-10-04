"""Fail-closed paper figure release from explicitly selected real main runs.

There is no CSV-only, mock, partial-data, old-data or automatic-latest mode.
An existing release is immutable. Every PDF, preview and source table is hashed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import re
from pathlib import Path
import tempfile

from jsonschema.exceptions import ValidationError

from wolfbench.llm_runtime import RUNTIME_VERSION
from wolfbench.llm_runtime.simulation import ACTION_SCHEMA, ACTION_VALIDATOR
from .registry import PAPER_SIZES, variants
from .runner import atomic_json, canonical, code_fingerprint, digest, load_complete, validate_manifest

SCHEMA = "wolfbench-paper-figure-bundle-v1"
INTERVENTION_SIZES = [300, 1000]
MIN_MAIN_SEEDS = 12
EXPECTED_HORIZON = 30
REQUIRED_TRACES = {"daily_log", "decision_log", "exposure_log", "message_log", "information_events",
                   "request_log", "backend_audit", "round_snapshots", "execution_audit", "trade_log"}
FIGURES = {
    "teaser": ("teaser.pdf", "result", ["p01"]),
    "fig2_nonlinear_response": ("fig2_nonlinear_response.pdf", "result", ["p01"]),
    "fig3_finite_size_scaling": ("fig3_finite_size_scaling.pdf", "result", ["p01"]),
    "fig4_intervention_effects": ("fig4_intervention_effects.pdf", "result", ["p04"]),
}
POLICY_FIELDS = ("model", "model_revision", "temperature", "max_tokens", "chat_template_kwargs")


def figure_code_fingerprint():
    root = Path(__file__).parent
    names = ("figure_bundle.py", "figure_rendering.py", "manuscript_figures.py", "paper_figure_catalog.json")
    original = root.parent / "paper_experiments_v3" / "figures"
    return digest({"experiment_source_hash": code_fingerprint(),
                   "presentation": {name: sha256_file(root / name) for name in names},
                   "original_plot_code": sha256_file(original / "make_paper_figures.py"),
                   "teaser_layout": sha256_file(original / "assets" / "teaser_layout.pdf")})


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contained_file(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"Unsafe artifact path: {relative}")
    path = root / rel
    if any((root / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)):
        raise ValueError(f"Symlink artifacts are not accepted: {relative}")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or external artifact: {relative}")
    return path


def _read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def _trace_rows(run_dir, descriptor):
    path = contained_file(run_dir, descriptor["path"])
    if sha256_file(path) != descriptor.get("sha256"):
        raise ValueError(f"Trace checksum mismatch: {path}")
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            canonical(record)  # Reject nonfinite JSON even in non-model traces.
            if not isinstance(record, dict):
                raise ValueError(f"Non-object trace record: {path}")
            records.append(record)
    if len(records) != descriptor.get("rows"):
        raise ValueError(f"Trace row count mismatch: {path}")
    return records


def _validate_population_evidence(row, run_dir):
    c, result, traces = row["config"], row["result"], row.get("traces", {})
    n, horizon = c["n_society"], c["horizon_days"]
    expected = n * horizon
    if result.get("runtime_version") != RUNTIME_VERSION or result.get("simulated") is not False:
        raise ValueError("Result is not the current real full-LLM runtime")
    if c.get("controller") != "llm" or any(c.get(f"{g}_controller", "llm") != "llm" for g in ("benign", "harmful")):
        raise ValueError("Primary result figures require full-LLM conditions")
    if any(result.get(k) != v for k, v in {"n_llm": n, "n_llm_decisions": expected,
            "n_population_decisions": expected, "rule_fallback_count": 0}.items()):
        raise ValueError("Population decision coverage or fallback audit mismatch")
    if c.get("scenario") != "s1" or horizon != EXPECTED_HORIZON:
        raise ValueError("Primary figure protocol must be S1 with the complete main horizon")
    if not REQUIRED_TRACES <= traces.keys():
        raise ValueError(f"Required traces missing: {sorted(REQUIRED_TRACES - traces.keys())}")
    usage = result.get("backend_usage", {})
    if (usage.get("requests") != expected or usage.get("successes") != expected or
            usage.get("failures") != 0 or usage.get("simulated_requests") != 0 or
            usage.get("remote_attempts", 0) < expected):
        raise ValueError("Missing complete remote-generation call audit")
    # Validate all trace descriptors, including evidence not used by the plot.
    decoded = {key: _trace_rows(run_dir, descriptor) for key, descriptor in traces.items()}
    snapshots = decoded["round_snapshots"]
    if snapshots != [{"day": day, "n_agents": n, "n_llm": n} for day in range(horizon)]:
        raise ValueError("Round snapshots do not cover the full population/horizon")
    if [r.get("day") for r in decoded["daily_log"]] != list(range(horizon)):
        raise ValueError("Daily evaluation log is incomplete or reordered")
    daily_failure = [r["primary_failure"] for r in decoded["daily_log"]]
    if result.get("primary_failure_rate") != float(any(r["triggered"] for r in daily_failure)):
        raise ValueError("Episode failure outcome differs from the daily evidence")
    if result.get("primary_failure_score_max") != max(r["primary_failure_score"] for r in daily_failure):
        raise ValueError("Episode failure score differs from the daily evidence")
    expected_ids = {f"{row['job_id']}:{day}:p{index:06d}": (day, f"p{index:06d}")
                    for day in range(horizon) for index in range(n)}
    requests = {}
    for request in decoded["request_log"]:
        rid = request.get("request_id")
        if rid not in expected_ids or rid in requests:
            raise ValueError("Duplicate, foreign or malformed model request identity")
        payload = json.loads(request["user"])
        if (payload.get("day"), payload.get("participant")) != expected_ids[rid]:
            raise ValueError("Request observation belongs to a different agent/round")
        requests[rid] = request
    if requests.keys() != expected_ids.keys():
        raise ValueError("Incomplete per-agent model requests")
    decisions = {}
    for event in decoded["decision_log"]:
        if "controller" not in event:  # Additional environment adjustment log.
            continue
        key = (event.get("day"), event.get("agent_id"))
        if key in decisions or event["controller"] != "llm":
            raise ValueError("Duplicate or non-LLM population action")
        ACTION_VALIDATOR.validate(event["raw_decision"])
        decisions[key] = event["raw_decision"]
    if set(decisions) != set(expected_ids.values()):
        raise ValueError("Incomplete executed-decision audit")
    audits = {}
    for audit in decoded["backend_audit"]:
        rid = audit.get("request_id")
        if rid not in requests or rid in audits:
            raise ValueError("Duplicate or foreign generation audit identity")
        if audit.get("status") != "success" or audit.get("simulated") is not False:
            raise ValueError("Failed or simulated generation cannot support a paper figure")
        if any(audit.get(key) != requests[rid].get(key) for key in ("system", "user", "seed")):
            raise ValueError("Generation does not match the agent's recorded request")
        provenance = audit.get("provenance", {})
        if any(provenance.get(key) != row["backend"].get(key) for key in POLICY_FIELDS):
            raise ValueError("Generation backend differs from the pinned main model policy")
        if audit.get("schema") != ACTION_SCHEMA or provenance.get("simulated") is not False:
            raise ValueError("Wrong action schema or simulated backend provenance")
        attempts = audit.get("attempts", [])
        if not attempts:
            raise ValueError("Generation has no transport evidence")
        payload = {"model": row["backend"]["model"], "messages": [
            {"role": "system", "content": requests[rid]["system"]},
            {"role": "user", "content": requests[rid]["user"]}],
            "seed": requests[rid]["seed"], "temperature": row["backend"]["temperature"],
            "max_tokens": row["backend"]["max_tokens"], "n": 1, "stream": False,
            "structured_outputs": {"json": ACTION_SCHEMA}}
        if row["backend"].get("chat_template_kwargs") is not None:
            payload["chat_template_kwargs"] = row["backend"]["chat_template_kwargs"]
        payload_hash = hashlib.sha256(canonical(payload).encode()).hexdigest()
        if audit.get("payload_sha256") != payload_hash or any(a.get("payload_sha256") != payload_hash for a in attempts):
            raise ValueError("Generation payload/retry hash differs from the recorded request")
        last = attempts[-1]
        if last.get("http_status") != 200 or last.get("error") is not None:
            raise ValueError("Last generation attempt is not successful")
        response = json.loads(last["raw_response"])
        choices = response.get("choices", [])
        if (not response.get("id") or response.get("model") != row["backend"]["model"] or
                len(choices) != 1 or choices[0].get("finish_reason") != "stop"):
            raise ValueError("Invalid, wrong-model or truncated completion")
        if response["id"] != audit.get("response_id") or response["id"] != last.get("response_id"):
            raise ValueError("Completion response identity differs from its transport audit")
        parsed = json.loads(choices[0]["message"]["content"])
        if parsed != decisions[expected_ids[rid]] or audit.get("decision") != parsed:
            raise ValueError("Model output differs from the population decision used")
        audits[rid] = True
    if audits.keys() != requests.keys():
        raise ValueError("Incomplete generation audit")


def _coverage(rows):
    expected = {("p01", "baseline", n) for n in PAPER_SIZES}
    expected |= {("p04", variant, n) for variant, _ in variants("p04") for n in INTERVENTION_SIZES}
    cells = defaultdict(lambda: defaultdict(set))
    coordinates = set()
    for row in rows:
        c = row["config"]
        key = (row["family"], row["variant"], c["n_society"])
        coordinate = (*key, c["alpha"], c["seed"])
        if coordinate in coordinates:
            raise ValueError(f"Duplicate independent experimental coordinate: {coordinate}")
        coordinates.add(coordinate)
        cells[key][c["alpha"]].add(c["seed"])
    if cells.keys() != expected:
        raise ValueError(f"Figure coverage mismatch; missing={sorted(expected - cells.keys())}, unexpected={sorted(cells.keys() - expected)}")
    common_seeds = None
    report = []
    for key, grid in sorted(cells.items()):
        if 0.0 not in grid or len(grid) < 2:
            raise ValueError(f"Missing clean baseline or positive-alpha grid: {key}")
        seeds = grid[0.0]
        if len(seeds) < MIN_MAIN_SEEDS or any(v != seeds for v in grid.values()):
            raise ValueError(f"Incomplete matched main seed grid: {key}")
        if common_seeds is not None and seeds != common_seeds:
            raise ValueError("P01/P04 figure comparisons require the same main seed identities")
        common_seeds = seeds
        report.append({"family": key[0], "variant": key[1], "n_society": key[2],
                       "alphas": sorted(grid), "seeds": sorted(seeds)})
    return report


def validate_sources(run_dirs, model, revision):
    """Return validated selected rows, immutable source records and coverage."""
    if not model or not revision:
        raise ValueError("Explicit model and immutable revision are required")
    if not re.fullmatch(r"(?:[0-9a-fA-F]{40}|(?:sha256:)?[0-9a-fA-F]{64})", revision):
        raise ValueError("Immutable model revision required: use a 40-hex commit or 64-hex SHA256, never main/latest labels")
    roots = [Path(p).resolve() for p in run_dirs]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Select one or more distinct explicit run directories")
    current_source = code_fingerprint()
    rows, sources, seen_jobs, policy = [], [], set(), None
    for root in sorted(roots):
        manifest_path = contained_file(root, "manifest.json")
        manifest = _read_json(manifest_path)
        validate_manifest(manifest, check_source=False)
        b = manifest["backend"]
        if (manifest["stage"] != "main" or b.get("kind") != "vllm" or
                manifest["source_hash"] != current_source or not manifest.get("grid_hash")):
            raise ValueError("Only current-source real main runs with a frozen grid may generate paper figures")
        if (b.get("model"), b.get("model_revision")) != (model, revision):
            raise ValueError("Run model/revision does not match the selected paper release")
        this_policy = {key: b.get(key) for key in POLICY_FIELDS}
        if policy is not None and policy != this_policy:
            raise ValueError("Cannot mix sampling or chat-template policies in one paper figure release")
        policy = this_policy
        fingerprints = {}
        selected_jobs = []
        for job in manifest["jobs"]:
            if job["job_id"] in seen_jobs:
                raise ValueError("Repeated jobs across selected manifests would double-count evidence")
            seen_jobs.add(job["job_id"])
            relative = f"episodes/{job['job_id']}.json"
            episode = contained_file(root, relative)
            untrusted_row = _read_json(episode)
            for desc in untrusted_row.get("traces", {}).values():
                contained_file(root, desc["path"])
            row = load_complete(episode, job, manifest)
            if row is None:
                raise ValueError(f"Incomplete/corrupt main episode: {job['job_id']}")
            fingerprints[relative] = sha256_file(episode)
            for desc in row.get("traces", {}).values():
                fingerprints[desc["path"]] = desc["sha256"]
            if row["family"] not in {"p01", "p04"}:
                continue
            row.update(backend=b, stage=manifest["stage"], run_dir=str(root))
            _validate_population_evidence(row, root)
            rows.append(row)
            selected_jobs.append(row["job_id"])
        if not selected_jobs:
            raise ValueError(f"Selected run contains no primary figure inputs: {root}")
        sources.append({"path": str(root), "manifest_hash": manifest["manifest_hash"],
                        "manifest_file_sha256": sha256_file(manifest_path),
                        "source_hash": manifest["source_hash"], "backend": b,
                        "complete_episodes": len(manifest["jobs"]),
                        "selected_job_ids": sorted(selected_jobs), "files": dict(sorted(fingerprints.items()))})
    return rows, sources, _coverage(rows)


def build_bundle(run_dirs, output, model, revision):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Figure release output already exists; use a new versioned directory")
    rows, sources, coverage = validate_sources(run_dirs, model, revision)
    generation_hash = figure_code_fingerprint()
    output.parent.mkdir(parents=True, exist_ok=True)
    from .figure_rendering import render
    # Failed validation/rendering never leaves a seemingly complete release.
    with tempfile.TemporaryDirectory(prefix=".paper-figures-", dir=output.parent) as temp:
        staging = Path(temp)
        metadata = render(rows, staging, model, revision)
        if {item["figure_id"] for item in metadata} != set(FIGURES) or len(metadata) != len(FIGURES):
            raise ValueError("Renderer did not produce exactly the manuscript figure set")
        figures, artifacts = [], {}
        for item in metadata:
            filename, kind, families = FIGURES[item["figure_id"]]
            if filename not in item["files"]:
                raise ValueError(f"Renderer missing required PDF {filename}")
            for relative in item["files"]:
                path = contained_file(staging, relative)
                artifacts[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            figures.append({**item, "filename": filename, "kind": kind, "source_families": families,
                            "sha256": artifacts[filename]["sha256"],
                            "source_job_ids": sorted(r["job_id"] for r in rows if r["family"] in families)})
        manifest = {"schema": SCHEMA, "status": "complete_real_main", "runtime_version": RUNTIME_VERSION,
                    "generator_source_hash": generation_hash, "created_at": datetime.now(timezone.utc).isoformat(),
                    "model": model, "model_revision": revision, "sources": sources, "coverage": coverage,
                    "figures": figures, "artifacts": artifacts,
                    "scope": "All current main/supplement included figures; prose and numerical tables are not rewritten."}
        if generation_hash != figure_code_fingerprint():
            raise ValueError("Experiment or figure code changed while rendering")
        for source in sources:
            root = Path(source["path"])
            if sha256_file(contained_file(root, "manifest.json")) != source["manifest_file_sha256"]:
                raise ValueError("Run manifest changed while rendering")
            for relative, checksum in source["files"].items():
                if sha256_file(contained_file(root, relative)) != checksum:
                    raise ValueError("Source experiment data changed while rendering")
        manifest["bundle_hash"] = digest(manifest)
        atomic_json(staging / "figure_manifest.json", manifest)
        staging.rename(output)
    return output / "figure_manifest.json"


def verify_bundle(manifest_path, *, verify_sources=True):
    """Recheck a release and its selected run identities; never choose newer files implicitly."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    body = {k: v for k, v in manifest.items() if k != "bundle_hash"}
    if (manifest.get("schema") != SCHEMA or manifest.get("status") != "complete_real_main" or
            manifest.get("bundle_hash") != digest(body) or manifest.get("runtime_version") != RUNTIME_VERSION):
        raise ValueError("Invalid or modified full-LLM figure release manifest")
    if manifest.get("generator_source_hash") != figure_code_fingerprint():
        raise ValueError("Figure release is stale for the current experiment/generator source")
    figures = manifest.get("figures", [])
    if len(figures) != len(FIGURES) or {f.get("figure_id") for f in figures} != set(FIGURES):
        raise ValueError("Incomplete manuscript figure catalog")
    artifacts = manifest.get("artifacts", {})
    for relative, record in artifacts.items():
        path = contained_file(manifest_path.parent, relative)
        if path.stat().st_size != record.get("size_bytes") or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Figure/source-table artifact has changed: {relative}")
    for figure in figures:
        filename, kind, families = FIGURES[figure["figure_id"]]
        if (figure.get("filename") != filename or figure.get("kind") != kind or
                figure.get("source_families") != families or filename not in figure.get("files", []) or
                filename not in artifacts or figure.get("sha256") != artifacts[filename]["sha256"] or
                any(f not in artifacts for f in figure.get("files", []))):
            raise ValueError("Figure catalog/provenance mapping was modified")
    if verify_sources:
        rows, sources, coverage = validate_sources([s["path"] for s in manifest["sources"]],
                                                 manifest["model"], manifest["model_revision"])
        if sources != manifest["sources"] or coverage != manifest["coverage"]:
            raise ValueError("Source experiment files changed after this figure release")
        for figure in figures:
            ids = sorted(r["job_id"] for r in rows if r["family"] in figure["source_families"])
            if figure.get("source_job_ids") != ids:
                raise ValueError("Figure source job identities changed")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--runs", nargs="+", type=Path, required=True)
    build.add_argument("--model", required=True)
    build.add_argument("--model-revision", required=True)
    build.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = {"manifest": str(build_bundle(args.runs, args.output, args.model, args.model_revision))}
        else:
            manifest = verify_bundle(args.manifest)
            result = {"status": "verified", "bundle_hash": manifest["bundle_hash"], "figures": len(manifest["figures"])}
    except (ValueError, OSError, KeyError, TypeError, ValidationError) as exc:
        parser.exit(2, f"Paper figure release refused: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
