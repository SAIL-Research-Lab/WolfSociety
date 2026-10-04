"""CPU integration tests of the strict batch client, including a fake server."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

from wolfbench.llm_runtime.backend import (
    BatchGenerationError, DecisionRequest, HTTPResult, MockBackend, VLLMBackend,
    benchmark_main,
)


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"seed": {"type": "integer"}}, "required": ["seed"],
}


def request(index=0, seed=None):
    return DecisionRequest(f"episode-7/day-2/agent-{index}", "Decide independently.",
                           json.dumps({"agent": index}), index if seed is None else seed)


def completion(body, *, content=None, finish_reason="stop", model=None, headers=None):
    payload = json.loads(body)
    response = {
        "id": f"chatcmpl-{payload['seed']}", "model": model or payload["model"],
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content or json.dumps({"seed": payload["seed"]})}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
    }
    return HTTPResult(200, json.dumps(response).encode(), headers or {})


@pytest.mark.parametrize("size", [1000, 2000])
def test_large_round_concurrency_barrier_and_result_identity(size):
    lock = threading.Lock()
    active = peak = 0

    def transport(url, body, headers, timeout):
        nonlocal active, peak
        payload = json.loads(body)
        with lock:
            active += 1
            peak = max(peak, active)
        # Deliberately settle requests out of their input order.
        time.sleep(0.001 * (3 - payload["seed"] % 3))
        with lock:
            active -= 1
        assert payload["structured_outputs"]["json"] == SCHEMA
        assert payload["n"] == 1 and payload["stream"] is False
        assert url.endswith("/v1/chat/completions")
        return completion(body, headers={"X-Request-Id": headers["X-Request-Id"]})

    backend = VLLMBackend("model-test", concurrency=16, transport=transport)
    decisions = backend.generate_batch([request(i) for i in range(size)], SCHEMA)
    assert decisions == [{"seed": i} for i in range(size)]
    assert 1 < peak <= 16 and active == 0
    assert backend.stats["requests"] == backend.stats["successes"] == size
    assert backend.stats["prompt_tokens"] == 11 * size
    assert backend.stats["completion_tokens"] == 5 * size
    assert len(backend.audit_rows) == size
    by_id = {row["request_id"]: row for row in backend.audit_rows}
    assert by_id[request(99).request_id]["decision"] == {"seed": 99}
    assert by_id[request(99).request_id]["input_index"] == 99
    assert backend.simulated is False


def test_network_retries_preserve_prompt_schema_seed_and_wire_identity(tmp_path):
    seen = []

    def transport(url, body, headers, timeout):
        seen.append((body, dict(headers), timeout))
        if len(seen) == 1:
            raise TimeoutError("temporary timeout")
        if len(seen) == 2:
            return HTTPResult(429, b'{"error":"busy"}', {})
        return completion(body)

    path = tmp_path / "audit.jsonl"
    backend = VLLMBackend("model-test", retries=2, retry_backoff=0,
                          api_key="test-secret", transport=transport, audit_path=path)
    assert backend.generate_batch([request(seed=123)], SCHEMA) == [{"seed": 123}]
    backend.close()
    assert seen[0] == seen[1] == seen[2]
    assert backend.stats["attempts"] == 3
    assert backend.stats["retries"] == 2
    assert backend.stats["failures"] == 0
    assert backend.stats["recovered_requests"] == 1
    row = json.loads(path.read_text())
    assert row["attempts"][0]["error"]["type"] == "TimeoutError"
    assert row["attempts"][1]["http_status"] == 429
    assert "busy" in row["attempts"][1]["raw_response"]
    assert len({a["payload_sha256"] for a in row["attempts"]}) == 1
    assert "test-secret" not in path.read_text()


def test_failed_batch_is_audited_completely_and_never_returned(tmp_path):
    def transport(url, body, headers, timeout):
        if json.loads(body)["seed"] == 1:
            return HTTPResult(400, b'{"error":"context too long"}', {})
        return completion(body)

    backend = VLLMBackend("model-test", retries=3, transport=transport,
                          audit_path=tmp_path / "failed.jsonl")
    with pytest.raises(BatchGenerationError, match="no fallback") as info:
        backend.generate_batch([request(i) for i in range(3)], SCHEMA)
    assert info.value.failed_request_ids == (request(1).request_id,)
    assert info.value.results == [{"seed": 0}, None, {"seed": 2}]
    assert backend.stats["requests"] == 3
    assert backend.stats["attempts"] == 3
    assert backend.stats["failures"] == 1
    assert backend.stats["failed_batches"] == 1
    assert len((tmp_path / "failed.jsonl").read_text().splitlines()) == 3
    backend.close()


@pytest.mark.parametrize("content,finish_reason,model", [
    ('{"seed":1}', "length", None),
    ('```json\n{"seed":1}\n```', "stop", None),
    ('{"seed":"wrong"}', "stop", None),
    ('{"seed":1,"extra":true}', "stop", None),
    ('{"seed":1,"seed":2}', "stop", None),
    ('{"seed":NaN}', "stop", None),
    ('{"seed":1e309}', "stop", None),
    ('{"seed":1}', "stop", "other-model"),
    ('[]', "stop", None),
])
def test_output_errors_fail_without_retry_until_valid(content, finish_reason, model):
    def transport(url, body, headers, timeout):
        return completion(body, content=content, finish_reason=finish_reason, model=model)

    backend = VLLMBackend("model-test", retries=3, transport=transport)
    with pytest.raises(BatchGenerationError):
        backend.generate_batch([request()], SCHEMA)
    assert backend.stats["attempts"] == 1
    assert backend.audit_rows[0]["status"] == "failure"
    # Charged generated tokens remain audited even for unusable outputs.
    assert backend.stats["completion_tokens"] == 5


def test_exhausted_timeout_does_not_convert_to_hold():
    def transport(*_):
        raise TimeoutError("server stuck")

    backend = VLLMBackend("model-test", retries=1, retry_backoff=0, transport=transport)
    with pytest.raises(BatchGenerationError):
        backend.generate_batch([request()], SCHEMA)
    assert backend.stats["failures"] == 1
    assert backend.stats["attempts"] == 2
    assert all(attempt["error"]["retryable"] for attempt in backend.audit_rows[0]["attempts"])


def test_duplicate_ids_and_remote_schema_refs_fail_before_network():
    calls = []
    backend = VLLMBackend("model-test", transport=lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="unique"):
        backend.generate_batch([request(), request()], SCHEMA)
    with pytest.raises(ValueError, match="local"):
        backend.generate_batch([request()], {"$ref": "https://example.invalid/schema"})
    assert calls == []


def test_no_response_cache_for_same_prompt_or_request_id_across_seeds():
    calls = []

    def transport(url, body, headers, timeout):
        calls.append(json.loads(body))
        return completion(body)

    backend = VLLMBackend("model-test", transport=transport)
    first = request(seed=1)
    second = request(seed=99)
    assert first.user == second.user and first.request_id == second.request_id
    assert backend.generate_batch([first], SCHEMA) == [{"seed": 1}]
    assert backend.generate_batch([second], SCHEMA) == [{"seed": 99}]
    assert [payload["seed"] for payload in calls] == [1, 99]
    assert backend.stats["cache_hits"] == 0
    assert backend.provenance["response_cache"] is False


def test_batch_called_in_running_async_loop_does_not_replace_loop():
    backend = VLLMBackend("model-test", transport=lambda u, b, h, t: completion(b))

    async def caller():
        before = asyncio.get_running_loop()
        result = backend.generate_batch([request()], SCHEMA)
        assert asyncio.get_running_loop() is before
        return result

    assert asyncio.run(caller()) == [{"seed": 0}]


def test_backend_global_bound_includes_simultaneous_batch_callers():
    lock = threading.Lock()
    active = peak = 0

    def transport(url, body, headers, timeout):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.002)
        with lock:
            active -= 1
        return completion(body)

    backend = VLLMBackend("model-test", concurrency=3, transport=transport)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(backend.generate_batch, [request(i) for i in range(15)], SCHEMA)
                   for _ in range(2)]
        for future in futures:
            assert len(future.result()) == 15
    assert peak <= 3
    assert backend.stats["requests"] == 30
    assert backend.stats["batches"] == 2


def test_mock_is_explicit_simulated_and_deterministic(tmp_path):
    backend = MockBackend(audit_path=tmp_path / "mock.jsonl")
    first = backend.generate_batch([request(7)], SCHEMA)
    second = backend.generate_batch([request(7)], SCHEMA)
    assert first == second
    assert backend.simulated is True
    assert backend.provenance["simulated"] is True
    assert backend.stats["remote_attempts"] == 0
    assert backend.stats["simulated_requests"] == 2
    assert all(row["simulated"] for row in backend.audit_rows)
    backend.close()


def test_disk_only_audit_requires_path_and_retains_no_response_cache(tmp_path):
    with pytest.raises(ValueError, match="audit_path"):
        VLLMBackend("model-test", retain_audit_rows=False)
    backend = VLLMBackend("model-test", transport=lambda u, b, h, t: completion(b),
                          audit_path=tmp_path / "rows.jsonl", retain_audit_rows=False)
    backend.generate_batch([request()], SCHEMA)
    assert backend.audit_rows == []
    assert len((tmp_path / "rows.jsonl").read_text().splitlines()) == 1
    backend.close()


def test_benchmark_replays_actual_prompts_and_reports_all_rounds(tmp_path, monkeypatch):
    import wolfbench.llm_runtime.backend as module
    observed = []

    def transport(url, body, headers, timeout):
        observed.append(json.loads(body))
        return completion(body)

    real_class = module.VLLMBackend
    monkeypatch.setattr(module, "VLLMBackend", lambda **kwargs: real_class(transport=transport, **kwargs))
    input_path = tmp_path / "source.jsonl"
    input_rows = [{"request_id": req.request_id, "system": req.system, "user": req.user,
                   "seed": req.seed, "schema": SCHEMA} for req in [request(5), request(8)]]
    input_path.write_text("\n".join(json.dumps(row) for row in input_rows) + "\n")
    output = tmp_path / "benchmark"
    status = benchmark_main(["--requests-jsonl", str(input_path), "--model", "test-model",
                             "--num-requests", "2", "--rounds", "2", "--output", str(output)])
    assert status == 0
    assert len(observed) == 4
    assert [payload["messages"][1]["content"] for payload in observed].count(request(5).user) == 2
    report = json.loads((output / "benchmark.json").read_text())
    assert report["simulated"] is False
    assert len(report["rounds"]) == 2
    assert report["cumulative_stats"]["requests"] == 4
    assert all(r["status"] == "success" and r["reported_completion_tokens"] == 10 for r in report["rounds"])
    audit = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert len({row["request_id"] for row in audit}) == 4
    assert all(row["seed"] in {5, 8} for row in audit)


def test_real_stdlib_http_transport_against_local_fake_server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append((self.path, json.loads(body), self.headers["X-Request-Id"]))
            result = completion(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(result.body)))
            self.send_header("X-Request-Id", self.headers["X-Request-Id"])
            self.end_headers()
            self.wfile.write(result.body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = VLLMBackend("local-fake", base_url=f"http://127.0.0.1:{server.server_port}/v1")
        assert backend.generate_batch([request(7)], SCHEMA) == [{"seed": 7}]
        assert seen[0][0] == "/v1/chat/completions"
        assert seen[0][1]["messages"][1]["content"] == request(7).user
        assert backend.audit_rows[0]["attempts"][0]["response_id"] == "chatcmpl-7"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_http_timeout_is_audited():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(0.15)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = VLLMBackend("local-fake", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                              request_timeout=0.025, retries=0)
        with pytest.raises(BatchGenerationError):
            backend.generate_batch([request()], SCHEMA)
        assert backend.audit_rows[0]["attempts"][0]["error"]["type"] == "TimeoutError"
        assert backend.audit_rows[0]["elapsed_seconds"] < 0.14
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
