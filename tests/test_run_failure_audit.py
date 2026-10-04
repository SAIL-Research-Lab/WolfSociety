import gzip
import json

import pytest

from paper_experiments.registry import build_cells
from paper_experiments.runner import build_manifest, run_manifest


def test_failed_requests_are_preserved_across_episode_restarts(tmp_path):
    class FailingBackend:
        simulated = True
        provenance = {"kind": "synthetic_failure"}
        stats = {"requests": 1}

        def __init__(self):
            self.audit_rows = []

    cells = build_cells(["p01"], stage="smoke", sizes=[8], seeds=[9001], horizon=1)[:1]
    manifest = build_manifest(cells, stage="smoke", backend={"kind": "mock", "model": "fixture"}, source_hash="fixture")
    seen = []

    def fail(config, backend):
        seen.append(config["episode_id"])
        backend.audit_rows.append({"request_id": config["episode_id"] + ":0:p000000",
                                   "status": "failure", "raw_response": "truncated JSON"})
        raise RuntimeError("Expected invalid output")

    backend = FailingBackend()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Expected invalid output"):
            run_manifest(manifest, tmp_path, backend=backend, episode_fn=fail, check_source=False)
        assert backend.audit_rows == []
    run_dir = tmp_path / "mock" / manifest["manifest_hash"][:16]
    traces = list((run_dir / "errors").glob("*.backend_audit.jsonl.gz"))
    assert len(traces) == 2
    for trace in traces:
        with gzip.open(trace, "rt") as handle:
            assert json.loads(handle.readline())["raw_response"] == "truncated JSON"
    assert seen == [manifest["jobs"][0]["job_id"]] * 2
    assert not list((run_dir / "episodes").glob("*.json"))
    assert not list((run_dir / "episodes").glob("*.lock"))
