"""Immutable manifest runner with per-episode atomic completion and provenance.

python -m paper_experiments.runner --help
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from . import SCHEMA_VERSION
from .registry import FAMILIES, build_cells

ROOT = Path(__file__).resolve().parents[1]
TRACE_KEYS = ("daily_log", "decision_log", "exposure_log", "message_log", "decisions", "exposures", "messages", "agent_decisions", "information_events", "request_log", "backend_audit", "round_snapshots", "execution_audit", "trade_log")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def code_fingerprint(root=ROOT):
    """Content hash includes dirty working files and configs, not just Git HEAD."""
    # Presentation-only changes must not require rerunning expensive inference.
    # Figure releases fingerprint these modules separately and still pin this
    # complete experiment/runtime hash.
    presentation_files = {"figure_bundle.py", "figure_rendering.py", "manuscript_figures.py", "paper_figure_catalog.json"}
    files = set()
    for directory in ("src", "paper_experiments"):
        for path in (root / directory).rglob("*"):
            if directory == "paper_experiments" and path.name in presentation_files:
                continue
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json"} and "__pycache__" not in path.parts and not any(part in {"outputs", "runs", "results"} for part in path.relative_to(root).parts):
                files.add(path)
    for name in ("pyproject.toml", "scripts/run_pilot.sh", "scripts/run_full_llm_paper.sh"):
        if (root / name).exists():
            files.add(root / name)
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})


def safe_url(url):
    parsed = urlsplit(url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint URL must not contain credentials, query parameters, or fragments; use an API-key environment variable")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def build_manifest(cells, *, stage, backend, source_hash=None, grid=None):
    if stage == "main" and backend.get("kind") == "vllm":
        validate_frozen_backend(grid, backend, {cell["family"] for cell in cells})
    source_hash = source_hash or code_fingerprint()
    provenance = {"schema": SCHEMA_VERSION, "source_hash": source_hash, "backend": backend, "stage": stage,
                  "grid_hash": digest(grid) if grid else None}
    run_fingerprint = digest(provenance)
    jobs = []
    for cell in cells:
        job = dict(cell)
        job["job_id"] = digest({"run_fingerprint": run_fingerprint, "cell": cell})
        jobs.append(job)
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate experiment cells in manifest")
    return {**provenance, "run_fingerprint": run_fingerprint, "jobs": jobs,
            "manifest_hash": digest({"run_fingerprint": run_fingerprint, "jobs": jobs})}


def validate_manifest(manifest, *, check_source=True):
    provenance = {k: manifest[k] for k in ("schema", "source_hash", "backend", "stage", "grid_hash")}
    if provenance["schema"] != SCHEMA_VERSION or digest(provenance) != manifest["run_fingerprint"]:
        raise ValueError("Manifest provenance checksum mismatch")
    expected = digest({"run_fingerprint": manifest["run_fingerprint"], "jobs": manifest["jobs"]})
    if expected != manifest["manifest_hash"]:
        raise ValueError("Manifest was modified after creation; generate a new manifest")
    for job in manifest["jobs"]:
        cell = {k: v for k, v in job.items() if k != "job_id"}
        if digest({"run_fingerprint": manifest["run_fingerprint"], "cell": cell}) != job["job_id"]:
            raise ValueError("Invalid job identity")
    if check_source and code_fingerprint() != manifest["source_hash"]:
        raise ValueError("Source/config files changed after manifest creation. Regenerate the manifest; old rows cannot resume")


def expected_llm_decisions(job):
    c = job["config"]
    if job["engine"] == "watts_null":
        return 0
    n = c["n_society"]
    k = c.get("harmful_count", round(c["alpha"] * n))
    mode = c.get("controller", "llm")
    fraction = c.get("llm_fraction", .5 if mode == "mixed" else 1. if mode == "llm" else 0.)
    active = 0
    for group, count in (("benign", n - k), ("harmful", k)):
        selected_mode = c.get(f"{group}_controller", mode)
        share = fraction if selected_mode == "mixed" else float(selected_mode == "llm")
        active += round(share * count)
    return active * c["horizon_days"]


def budget(manifest):
    jobs = manifest["jobs"]
    calls = sum(expected_llm_decisions(j) for j in jobs)
    return {"episodes": len(jobs), "families": dict(Counter(j["family"] for j in jobs)),
            "agent_round_decisions": sum(j["config"]["n_society"] * j["config"]["horizon_days"] for j in jobs if j["engine"] == "society"),
            "planned_llm_generations": calls, "max_completion_tokens_before_retries": calls * manifest["backend"].get("max_tokens", 0),
            "input_tokens": "measure pilot usage; context length varies by inbox and memory",
            "runtime": "measure tokens/second on target hardware; no throughput assumed",
            "simulated": manifest["backend"]["kind"] == "mock"}


def load_complete(path, job, manifest):
    try:
        row = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if row.get("status") != "complete" or row.get("job_id") != job["job_id"] or row.get("run_fingerprint") != manifest["run_fingerprint"]:
        return None
    if any(row.get(key) != job[key] for key in ("config", "family", "variant", "engine")) or row.get("simulated") != (manifest["backend"]["kind"] == "mock"):
        return None
    try:
        matches = row.get("result_hash") == digest(row.get("result"))
    except (TypeError, ValueError):
        return None
    if not matches:
        return None
    try:
        for record in row.get("traces", {}).values():
            trace = Path(path).parents[1] / record["path"]
            if not trace.exists() or hashlib.sha256(trace.read_bytes()).hexdigest() != record["sha256"]:
                return None
    except (OSError, KeyError, TypeError, AttributeError):
        return None
    return row


def create_backend(settings):
    from wolfbench.llm_runtime.backend import MockBackend, VLLMBackend
    if settings["kind"] == "mock":
        return MockBackend()
    values = {k: v for k, v in settings.items() if k not in {"kind", "api_key_env", "timeout"}}
    values["request_timeout"] = settings["timeout"]
    return VLLMBackend(**values,
                       api_key=os.getenv(settings.get("api_key_env", "VLLM_API_KEY"), "EMPTY"))


def run_manifest(manifest, output, *, backend=None, episode_fn=None, limit=None, shard_index=0, num_shards=1, check_source=True):
    validate_manifest(manifest, check_source=check_source)
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    output = Path(output)
    mode = "mock" if manifest["backend"]["kind"] == "mock" else "real"
    run_dir = output / mode / manifest["manifest_hash"][:16]
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing run directory contains a different manifest")
    atomic_json(manifest_path, manifest)
    selected = [j for j in manifest["jobs"] if int(j["job_id"], 16) % num_shards == shard_index]
    if limit:
        selected = selected[:limit]
    completed, skipped = 0, 0
    for index, job in enumerate(selected, 1):
        path = run_dir / "episodes" / f"{job['job_id']}.json"
        if load_complete(path, job, manifest):
            skipped += 1
            continue
        lock = path.with_suffix(".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise RuntimeError(f"Episode locked by another worker: {lock}. After confirming no worker is active, remove only this stale lock")
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.close(fd)
        print(f"[{index}/{len(selected)}] {job['family']} {job['variant']} N={job['config']['n_society']} alpha={job['config']['alpha']} seed={job['config']['seed']}", flush=True)
        try:
            if job["engine"] == "watts_null":
                from .null_model import run_null
                result = run_null(job["config"])
            else:
                if backend is None:
                    backend = create_backend(manifest["backend"])
                if bool(getattr(backend, "simulated", False)) != (mode == "mock"):
                    raise ValueError("Backend simulation marker does not match manifest")
                if episode_fn is None:
                    from wolfbench.llm_runtime.simulation import run_episode
                    episode_fn = run_episode
                episode_config = dict(job["config"], episode_id=job["job_id"])
                result = episode_fn(episode_config, backend)
            traces = {}
            for key in TRACE_KEYS:
                if key not in result:
                    continue
                records = result.pop(key)
                trace_path = run_dir / "traces" / f"{job['job_id']}.{key}.jsonl.gz"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = trace_path.with_suffix(".tmp")
                with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(canonical(record) + "\n")
                temporary.replace(trace_path)
                traces[key] = {"path": str(trace_path.relative_to(run_dir)), "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(), "rows": len(records)}
            canonical(result)  # fail before marking completion on NaN/non-JSON fields
            row = {**job, "status": "complete", "run_fingerprint": manifest["run_fingerprint"],
                   "simulated": mode == "mock", "completed_at": datetime.now(timezone.utc).isoformat(),
                   "result": result, "result_hash": digest(result), "traces": traces,
                   "backend_provenance": getattr(backend, "provenance", {}) if backend is not None else {"engine": "watts_null"}}
            atomic_json(path, row)
            # Audit rows are now durable per episode; do not retain the entire
            # multi-million-call suite in the serving client's Python memory.
            if backend is not None and hasattr(backend, "audit_rows"):
                backend.audit_rows.clear()
            completed += 1
        except Exception as exc:
            # Failed generations also belong in the audit. Preserve every
            # attempt across explicit episode restarts rather than silently
            # dropping invalid/refused outputs and their token expenditure.
            stamp = time.time_ns()
            errors_dir = run_dir / "errors"
            errors_dir.mkdir(parents=True, exist_ok=True)
            audit_rows = list(getattr(backend, "audit_rows", [])) if backend is not None else []
            failed_audit = None
            if audit_rows:
                audit_path = errors_dir / f"{job['job_id']}.{stamp}.backend_audit.jsonl.gz"
                with gzip.open(audit_path, "wt", encoding="utf-8") as handle:
                    for record in audit_rows:
                        handle.write(canonical(record) + "\n")
                failed_audit = {"path": str(audit_path.relative_to(run_dir)),
                                "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(), "rows": len(audit_rows)}
            error = {"job_id": job["job_id"], "error_type": type(exc).__name__, "message": str(exc),
                     "status": "failed", "backend_audit": failed_audit,
                     "backend_cumulative_stats": getattr(backend, "stats", {}) if backend is not None else {}}
            atomic_json(errors_dir / f"{job['job_id']}.{stamp}.json", error)
            atomic_json(errors_dir / f"{job['job_id']}.json", error)
            if backend is not None and hasattr(backend, "audit_rows"):
                backend.audit_rows.clear()
            raise
        finally:
            lock.unlink(missing_ok=True)
    status = {"run_dir": str(run_dir), "completed_now": completed, "resumed_complete": skipped,
              "selected_episodes": len(selected), "total_manifest_episodes": len(manifest["jobs"]), "simulated": mode == "mock"}
    atomic_json(run_dir / f"status.shard-{shard_index}-of-{num_shards}.json", status)
    return status


def int_list(raw):
    values = []
    for part in raw.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return values


def json_object(raw):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Expected a JSON object") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object, not a scalar or array")
    return value


def validate_frozen_backend(grid, backend, families):
    """Bind scientific main sampling to its frozen pilot evidence.

    P07-only model comparisons may share a grid across weights. Serving URL,
    request concurrency and other transport settings remain independently
    recorded in the manifest without redefining the sampling protocol.
    """
    settings = grid.get("pilot_backend_settings") if isinstance(grid, dict) else None
    required = {"model", "model_revision", "max_tokens", "temperature", "chat_template_kwargs"}
    if not isinstance(settings, list) or not settings or any(not isinstance(p, dict) or not required <= p.keys() for p in settings):
        raise ValueError("Frozen grid lacks complete pilot_backend_settings evidence; regenerate the frozen grid from pilot manifests")
    compared = {"max_tokens", "temperature", "chat_template_kwargs"}
    if set(families) != {"p07"}:
        compared |= {"model", "model_revision"}
    if not any(p.get("kind") == "vllm" and all(p[field] == backend.get(field) for field in compared) for p in settings):
        raise ValueError("Main model/sampling settings differ from frozen pilot evidence; rerun pilots or match model, revision, max_tokens, temperature and chat_template_kwargs (only P07-only permits different model/revision)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Print the final paper families")
    plan = sub.add_parser("plan", help="Create immutable manifest and print budget; makes no model calls")
    plan.add_argument("--families", nargs="+", default=["p01"])
    plan.add_argument("--stage", choices=["smoke", "pilot", "main"], default="pilot")
    plan.add_argument("--sizes", type=int_list)
    plan.add_argument("--seeds", type=int_list)
    plan.add_argument("--horizon", type=int, default=30)
    plan.add_argument("--grid", type=Path)
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--mock", action="store_true")
    plan.add_argument("--model", default="")
    plan.add_argument("--model-revision", default="", help="Pinned weights revision/hash, required for real main runs")
    plan.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    plan.add_argument("--concurrency", type=int, default=32)
    plan.add_argument("--max-tokens", type=int, default=512)
    plan.add_argument("--temperature", type=float, default=.7)
    plan.add_argument("--timeout", type=float, default=180)
    plan.add_argument("--retries", type=int, default=2)
    plan.add_argument("--api-key-env", default="VLLM_API_KEY")
    plan.add_argument("--chat-template-kwargs", type=json_object, default=None,
                      help='Model-specific JSON object, for example {"enable_thinking":false}; recorded in provenance')
    run = sub.add_parser("run", help="Execute/resume exactly the manifest with atomic validated rows")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(FAMILIES, indent=2, ensure_ascii=False))
        return
    if args.command == "plan":
        if args.horizon < 1:
            parser.error("--horizon must be positive")
        if not args.mock and not args.model:
            parser.error("Real manifests require --model")
        if args.stage == "main" and not args.mock and not args.model_revision:
            parser.error("Real main manifests require --model-revision for reproducibility")
        grid = json.loads(args.grid.read_text()) if args.grid else None
        if args.stage == "main" and not args.mock and grid and grid.get("simulated") is not False:
            parser.error("Real main sweeps require a frozen grid from real pilots, never mock/unknown results")
        if args.stage == "main" and grid and grid.get("pilot_source_hashes") and code_fingerprint() not in grid["pilot_source_hashes"]:
            parser.error("Source/prompts changed since the frozen pilot. Run new pilots before the main sweep")
        settings = {"kind": "mock" if args.mock else "vllm", "model": args.model or "MOCK-NOT-SCIENTIFIC",
                    "model_revision": args.model_revision, "base_url": safe_url(args.base_url),
                    "concurrency": args.concurrency, "max_tokens": args.max_tokens, "temperature": args.temperature,
                    "timeout": args.timeout, "retries": args.retries, "api_key_env": args.api_key_env,
                    "chat_template_kwargs": args.chat_template_kwargs}
        cells = build_cells(args.families, stage=args.stage, sizes=args.sizes, seeds=args.seeds, grids=grid, horizon=args.horizon)
        manifest = build_manifest(cells, stage=args.stage, backend=settings, grid=grid)
        if args.manifest.exists() and json.loads(args.manifest.read_text()) != manifest:
            parser.error("Manifest already exists with different contents; choose a new path")
        atomic_json(args.manifest, manifest)
        print(json.dumps({"manifest": str(args.manifest), "manifest_hash": manifest["manifest_hash"], **budget(manifest)}, indent=2))
    else:
        print(json.dumps(run_manifest(json.loads(args.manifest.read_text()), args.output,
                                      limit=args.limit, shard_index=args.shard_index, num_shards=args.num_shards), indent=2))


if __name__ == "__main__":
    main()
