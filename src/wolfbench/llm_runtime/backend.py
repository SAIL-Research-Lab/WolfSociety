"""Strict, GPU-independent client for batched vLLM agent decisions.

Each logical agent remains a separate chat request. The synchronous batch
barrier permits server-side continuous batching without changing a simulation
round's information timing. There is no response cache and no policy/model
fallback. vLLM KV-prefix caching is independent of response caching.

This client targets the current ``structured_outputs`` API (vLLM >= 0.12).
Pin the actual vLLM/model/tokenizer revisions in the server launch manifest.
Seeds are preserved on transport retries, but online scheduling can still
change outputs without vLLM batch invariance on supported hardware/models.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import base64
import copy
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import ssl
import statistics
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit


@dataclass(frozen=True)
class DecisionRequest:
    request_id: str
    system: str
    user: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a nonempty string")
        if not isinstance(self.system, str) or not isinstance(self.user, str):
            raise TypeError("system and user must be strings")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not -(2**63) <= self.seed < 2**63:
            raise ValueError("seed must fit a signed 64-bit integer")


@dataclass(frozen=True)
class HTTPResult:
    """Transport result; injection of this interface makes CPU testing simple."""

    status: int
    body: bytes
    headers: Mapping[str, str]


Transport = Callable[[str, bytes, Mapping[str, str], float], HTTPResult]


class CompletionValidationError(ValueError):
    """An output cannot be used; retrying identical sampling is not a repair."""


class BatchGenerationError(RuntimeError):
    """No partial batch may be applied to the simulation after this error."""

    def __init__(self, failures: dict[str, dict[str, Any]], results: list[Any]):
        self.failures = failures
        self.results = results
        self.failed_request_ids = tuple(failures)
        first = next(iter(failures.items()))
        super().__init__(
            f"{len(failures)} LLM request(s) failed; no fallback. "
            f"First: {first[0]}: {first[1]['message']}"
        )


class _HTTPStatusError(RuntimeError):
    def __init__(self, status: int):
        self.status = status
        self.retryable = status in {408, 425, 429, 500, 502, 503, 504}
        super().__init__(f"vLLM returned HTTP {status}; see raw response in audit")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _json_object_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CompletionValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CompletionValidationError(f"nonfinite JSON constant {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CompletionValidationError(f"nonfinite JSON number {value}")
    return result


def _parse_json(value: str) -> Any:
    try:
        return json.loads(value, object_pairs_hook=_json_object_pairs,
                          parse_constant=_reject_constant, parse_float=_finite_float)
    except (json.JSONDecodeError, TypeError) as exc:
        raise CompletionValidationError(f"invalid JSON: {exc}") from exc


def _validator(schema: dict):
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError(
            "LLM runtime requires jsonschema>=4.23; install the runtime dependencies"
        ) from exc
    Draft202012Validator.check_schema(schema)
    # Remote references are deliberately unsupported: validation must neither
    # fetch resources nor depend on mutable external schema documents.
    def check_refs(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"$ref", "$dynamicRef"} and not value.startswith("#"):
                    raise ValueError("only local JSON schema references are supported")
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)
    check_refs(schema)
    return Draft202012Validator(schema)


def _http_post(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> HTTPResult:
    """A total network-attempt deadline, including connect and body transfer."""
    parsed = urlsplit(url)
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    kwargs: dict[str, Any] = {"timeout": timeout}
    if parsed.scheme == "https":
        kwargs["context"] = ssl.create_default_context()
    connection = cls(parsed.hostname, parsed.port, **kwargs)
    started = time.monotonic()

    def remaining() -> float:
        left = timeout - (time.monotonic() - started)
        if left <= 0:
            raise TimeoutError(f"request exceeded {timeout:g}s deadline")
        return left

    try:
        connection.connect()
        sock = connection.sock
        sock.settimeout(remaining())
        connection.request("POST", parsed.path or "/", body=body, headers=dict(headers))
        sock.settimeout(remaining())
        response = connection.getresponse()
        chunks = []
        # Content-length exhaustion closes an HTTP/1.0 response's owning
        # socket. Do not touch that socket once read1 has completed the body.
        while not response.isclosed():
            sock.settimeout(remaining())
            chunk = response.read1(65536)
            if not chunk:
                break
            chunks.append(chunk)
        return HTTPResult(response.status, b"".join(chunks), dict(response.getheaders()))
    finally:
        connection.close()


class VLLMBackend:
    """One local vLLM model, bounded HTTP concurrency, strict JSON decisions.

    ``retries`` counts additional attempts for transient network/HTTP errors.
    Validation/length/refusal failures are never retried until a favorable
    sample appears. Retry payloads are byte-identical, including sampling seed.
    Audit stores all prompts, schema, raw bodies, attempts and reported usage;
    credentials are omitted. If an audit path is supplied, rows are flushed
    immediately when each logical request finishes, including failed batches.
    Use ``retain_audit_rows=False`` with a path for long experiments.
    """

    simulated = False
    synchronous = True
    batch_barrier = True
    name = "vllm_openai_http"

    def __init__(
        self, model: str, base_url: str = "http://127.0.0.1:8000/v1", *,
        concurrency: int = 64, request_timeout: float = 120.0,
        retries: int = 2, max_tokens: int = 512, temperature: float = 0.7,
        api_key: str | None = None, audit_path: str | Path | None = None,
        model_revision: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        retry_backoff: float = 0.25, retain_audit_rows: bool = True,
        transport: Transport | None = None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be explicitly specified")
        for label, value, minimum in [("concurrency", concurrency, 1),
                                      ("retries", retries, 0), ("max_tokens", max_tokens, 1)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{label} must be an integer >= {minimum}")
        for label, value, positive in [("request_timeout", request_timeout, True),
                                       ("retry_backoff", retry_backoff, False),
                                       ("temperature", temperature, False)]:
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"{label} must be a finite number")
            if not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, a query or a fragment")
        self.model = model
        self.base_url = base_url.rstrip("/")
        if parsed.path in {"", "/"}:
            self.base_url += "/v1"
        self.endpoint = self.base_url + "/chat/completions"
        self.concurrency = concurrency
        self.request_timeout = float(request_timeout)
        self.retries = retries
        self.max_tokens = max_tokens
        self.temperature = float(temperature)
        self.api_key = api_key
        self.model_revision = model_revision
        self.chat_template_kwargs = copy.deepcopy(chat_template_kwargs)
        if self.chat_template_kwargs is not None:
            if not isinstance(self.chat_template_kwargs, dict):
                raise TypeError("chat_template_kwargs must be an object")
            _canonical(self.chat_template_kwargs)
        self.retry_backoff = float(retry_backoff)
        self.retain_audit_rows = bool(retain_audit_rows)
        self.audit_path = Path(audit_path) if audit_path is not None else None
        if not retain_audit_rows and self.audit_path is None:
            raise ValueError("retain_audit_rows=False requires an audit_path")
        self._transport = transport or _http_post
        self._lock = threading.Lock()
        self._batch_lock = threading.Lock()
        self._audit_file = None
        self.audit_rows: list[dict[str, Any]] = []
        self._batch_number = 0
        self._stats: dict[str, int | float] = {
            "batches": 0, "failed_batches": 0, "requests": 0,
            "successes": 0, "failures": 0, "attempts": 0, "retries": 0,
            "recovered_requests": 0, "remote_attempts": 0, "simulated_requests": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "usage_missing_attempts": 0, "batch_seconds": 0.0,
            "request_seconds": 0.0, "http_seconds": 0.0, "cache_hits": 0,
        }

    @property
    def provenance(self) -> dict[str, Any]:
        config = {
            "backend": self.name, "model": self.model,
            "model_revision": self.model_revision, "revision_verified": False,
            "base_url": self.base_url, "concurrency": self.concurrency,
            "request_timeout": self.request_timeout, "retries": self.retries,
            "max_tokens": self.max_tokens, "temperature": self.temperature,
            "chat_template_kwargs": self.chat_template_kwargs,
            "retry_backoff": self.retry_backoff,
            "schema_api": "structured_outputs.json", "simulated": self.simulated,
            "response_cache": False,
            "reproducibility": "seeded; pin server/hardware and validate batch invariance",
        }
        config["configuration_sha256"] = _sha(_canonical(config))
        return config

    @property
    def fingerprint(self) -> str:
        return self.provenance["configuration_sha256"]

    @property
    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return dict(self._stats)

    @property
    def calls(self) -> int:
        return int(self.stats["requests"])

    @property
    def failures(self) -> int:
        return int(self.stats["failures"])

    @property
    def cache_hits(self) -> int:
        return 0

    def close(self) -> None:
        with self._batch_lock:
            with self._lock:
                if self._audit_file is not None:
                    self._audit_file.close()
                    self._audit_file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _payload(self, request: DecisionRequest, schema: dict) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": request.system},
                         {"role": "user", "content": request.user}],
            "seed": request.seed, "temperature": self.temperature,
            "max_tokens": self.max_tokens, "n": 1, "stream": False,
            "structured_outputs": {"json": schema},
        }
        if self.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        return payload

    def _send(self, request: DecisionRequest, body: bytes,
              headers: Mapping[str, str]) -> HTTPResult:
        return self._transport(self.endpoint, body, headers, self.request_timeout)

    def _record(self, row: dict) -> None:
        with self._lock:
            if self.audit_path is not None:
                if self._audit_file is None:
                    self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                    self._audit_file = self.audit_path.open("a", encoding="utf-8")
                self._audit_file.write(_canonical(row) + "\n")
                self._audit_file.flush()
            if self.retain_audit_rows:
                self.audit_rows.append(row)
            self._stats["requests"] += 1
            self._stats["successes" if row["status"] == "success" else "failures"] += 1
            count = len(row["attempts"])
            self._stats["attempts"] += count
            self._stats["retries"] += max(0, count - 1)
            self._stats["remote_attempts"] += 0 if self.simulated else count
            self._stats["simulated_requests"] += int(self.simulated)
            self._stats["recovered_requests"] += int(count > 1 and row["status"] == "success")
            self._stats["request_seconds"] += row["elapsed_seconds"]
            for attempt in row["attempts"]:
                self._stats["http_seconds"] += attempt["elapsed_seconds"]
                usage = attempt.get("usage")
                if not isinstance(usage, dict) or not usage:
                    if attempt.get("http_status") == 200 and not self.simulated:
                        self._stats["usage_missing_attempts"] += 1
                    continue
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    count = usage.get(key)
                    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                        self._stats[key] += count

    def _decode(self, response: dict, validator) -> dict:
        if response.get("model") != self.model:
            raise CompletionValidationError("server returned a different or missing model identifier")
        if not isinstance(response.get("id"), str) or not response["id"]:
            raise CompletionValidationError("server response has no completion id")
        usage = response.get("usage")
        if usage is not None:
            if not isinstance(usage, dict):
                raise CompletionValidationError("server token usage must be an object")
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                count = usage.get(name)
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise CompletionValidationError(f"invalid reported token usage: {name}")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise CompletionValidationError("exactly one completion choice is required")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index") != 0:
            raise CompletionValidationError("completion choice index must be zero")
        if choice.get("finish_reason") != "stop":
            raise CompletionValidationError(
                f"incomplete or refused generation: finish_reason={choice.get('finish_reason')!r}"
            )
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("refusal"):
            raise CompletionValidationError("completion message is absent or refused")
        content = message.get("content")
        if not isinstance(content, str):
            raise CompletionValidationError("completion content must be JSON text")
        result = _parse_json(content)
        if not isinstance(result, dict):
            raise CompletionValidationError("decision JSON must be an object")
        errors = sorted(validator.iter_errors(result), key=lambda e: str(e.path))
        if errors:
            error = errors[0]
            raise CompletionValidationError(f"decision schema violation at {list(error.path)}: {error.message}")
        return result

    def _generate_one(self, request: DecisionRequest, schema: dict, validator,
                      batch_number: int, index: int, queued: float) -> tuple[Any, dict]:
        started = time.monotonic()
        body = _canonical(self._payload(request, schema)).encode("utf-8")
        # ASCII transport identity avoids restrictions on arbitrary agent ids.
        wire_id = "wolfbench-" + _sha(request.request_id)
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "X-Request-Id": wire_id}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        row = {
            "request_id": request.request_id, "wire_request_id": wire_id,
            "batch_number": batch_number, "input_index": index, "seed": request.seed,
            "system": request.system, "user": request.user, "schema": schema,
            "payload_sha256": _sha(body), "provenance": self.provenance,
            "simulated": self.simulated, "started_at": _utc_now(),
            "queue_seconds": started - queued, "attempts": [], "status": "failure",
        }
        result = None
        for attempt_index in range(self.retries + 1):
            attempt_started = time.monotonic()
            attempt = {"attempt": attempt_index + 1, "started_at": _utc_now(),
                       "payload_sha256": _sha(body), "http_status": None,
                       "response_id": None, "response_headers": {}, "usage": None,
                       "finish_reason": None, "raw_response": None, "error": None}
            retryable = False
            try:
                http_result = self._send(request, body, headers)
                attempt["http_status"] = http_result.status
                attempt["response_headers"] = {
                    k: v for k, v in http_result.headers.items()
                    if k.lower() in {"x-request-id", "content-type", "server", "retry-after"}
                }
                attempt["raw_response"] = http_result.body.decode("utf-8", errors="replace")
                try:
                    http_result.body.decode("utf-8")
                except UnicodeDecodeError as exc:
                    attempt["raw_response_base64"] = base64.b64encode(http_result.body).decode("ascii")
                    raise CompletionValidationError("server body is not valid UTF-8") from exc
                if http_result.status != 200:
                    raise _HTTPStatusError(http_result.status)
                response = _parse_json(attempt["raw_response"])
                if not isinstance(response, dict):
                    raise CompletionValidationError("server JSON response must be an object")
                attempt["response_id"] = response.get("id")
                attempt["usage"] = response.get("usage")
                choices = response.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    attempt["finish_reason"] = choices[0].get("finish_reason")
                echoed = next((v for k, v in http_result.headers.items()
                               if k.lower() == "x-request-id"), None)
                if echoed is not None and echoed != wire_id:
                    raise CompletionValidationError("server echoed a different request id")
                result = self._decode(response, validator)
                row["status"] = "success"
                row["response_id"] = attempt["response_id"]
            except Exception as exc:
                retryable = isinstance(exc, (TimeoutError, ConnectionError, OSError,
                                             http.client.HTTPException))
                if isinstance(exc, _HTTPStatusError):
                    retryable = exc.retryable
                attempt["error"] = {"type": type(exc).__name__, "message": str(exc),
                                    "retryable": retryable}
            finally:
                attempt["elapsed_seconds"] = time.monotonic() - attempt_started
                row["attempts"].append(attempt)
            if row["status"] == "success":
                break
            if not retryable or attempt_index == self.retries:
                row["error"] = attempt["error"]
                break
            backoff = min(self.retry_backoff * 2**attempt_index, 5.0)
            attempt["retry_delay_seconds"] = backoff
            time.sleep(backoff)
        row["elapsed_seconds"] = time.monotonic() - started
        row["finished_at"] = _utc_now()
        if row["status"] == "success":
            row["decision"] = result
        self._record(row)
        return result, row

    def generate_batch(self, requests: Sequence[DecisionRequest], schema: dict) -> list[dict]:
        """Return input-order results only after every logical request settles.

        A batch error includes partial outputs for diagnostics, but never returns
        an incomplete list for use by the environment. IDs must be unique within
        the batch. Reusing an ID in a later invocation does not reuse its output.
        """
        materialized = list(requests)
        if any(not isinstance(req, DecisionRequest) for req in materialized):
            raise TypeError("generate_batch requires DecisionRequest instances")
        ids = [req.request_id for req in materialized]
        if len(ids) != len(set(ids)):
            raise ValueError("request_id values must be unique within a batch")
        if not isinstance(schema, dict):
            raise TypeError("schema must be a JSON schema object")
        schema = _parse_json(_canonical(schema))
        validator = _validator(schema)
        if not materialized:
            return []
        # Serialize callers so concurrency also bounds the backend globally.
        with self._batch_lock:
            self._batch_number += 1
            batch_number = self._batch_number
            queued = time.monotonic()
            results: list[Any] = [None] * len(materialized)
            failures = {}
            with ThreadPoolExecutor(max_workers=min(self.concurrency, len(materialized)),
                                    thread_name_prefix="wolfbench-vllm") as executor:
                futures = {executor.submit(self._generate_one, request, schema, validator,
                                           batch_number, index, queued): (index, request)
                           for index, request in enumerate(materialized)}
                for future in as_completed(futures):
                    index, request = futures[future]
                    result, row = future.result()
                    results[index] = result
                    if row["status"] != "success":
                        failures[request.request_id] = row["error"]
            with self._lock:
                self._stats["batches"] += 1
                self._stats["failed_batches"] += int(bool(failures))
                self._stats["batch_seconds"] += time.monotonic() - queued
            if failures:
                ordered_failures = {rid: failures[rid] for rid in ids if rid in failures}
                raise BatchGenerationError(ordered_failures, results)
            return results


def _mock_value(schema: Any, rng: random.Random, root: dict) -> Any:
    """Only a deterministic schema fixture generator, never a trading policy."""
    if schema is True:
        return None
    if schema is False:
        raise ValueError("cannot synthesize a false schema")
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        return _mock_value(target, rng, root)
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    if "enum" in schema:
        return copy.deepcopy(rng.choice(schema["enum"]))
    if "default" in schema:
        return copy.deepcopy(schema["default"])
    if "oneOf" in schema or "anyOf" in schema:
        return _mock_value(rng.choice(schema.get("oneOf", schema.get("anyOf"))), rng, root)
    kind = schema.get("type", "object" if "properties" in schema else "null")
    if isinstance(kind, list):
        kind = next((t for t in kind if t != "null"), "null")
    if kind == "object":
        return {name: _mock_value(sub, rng, root)
                for name, sub in schema.get("properties", {}).items()}
    if kind == "array":
        return [_mock_value(schema.get("items", {}), rng, root)
                for _ in range(schema.get("minItems", 0))]
    if kind == "string":
        return "x" * schema.get("minLength", 0)
    if kind in {"integer", "number"}:
        low = schema.get("minimum", schema.get("exclusiveMinimum", 0))
        high = schema.get("maximum", schema.get("exclusiveMaximum", max(low, 1)))
        if kind == "integer":
            value = math.ceil(low)
            if "exclusiveMinimum" in schema and value <= low:
                value += 1
            return value
        return (float(low) + float(high)) / 2.0
    if kind == "boolean":
        return bool(rng.randrange(2))
    return None


class MockBackend(VLLMBackend):
    """Explicit, prominently marked fixture backend for dry runs and tests."""

    simulated = True
    name = "simulated_schema_fixture"

    def __init__(self, *, responder: Callable[[DecisionRequest, dict], dict] | None = None,
                 audit_path: str | Path | None = None, retain_audit_rows: bool = True):
        super().__init__(model="__simulated__", base_url="http://mock.invalid/v1",
                         concurrency=1, retries=0, audit_path=audit_path,
                         retain_audit_rows=retain_audit_rows)
        self._responder = responder

    def _send(self, request: DecisionRequest, body: bytes,
              headers: Mapping[str, str]) -> HTTPResult:
        schema = _parse_json(body.decode("utf-8"))["structured_outputs"]["json"]
        if self._responder is not None:
            result = self._responder(request, schema)
        elif {"orders", "message", "memory"} <= set(schema.get("properties", {})):
            result = {"orders": [], "message": {
                "action": "none", "asset": "", "text": "", "sentiment": 0,
                "intensity": 0, "confidence": 0.5, "source_message_id": "",
            }, "memory": ""}
        else:
            seed = int(_sha(f"{request.seed}\0{request.request_id}"), 16)
            result = _mock_value(schema, random.Random(seed), schema)
        response = {"id": "mock-" + _sha(request.request_id), "model": self.model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": _canonical(result)}}],
                    "usage": None, "simulated": True}
        return HTTPResult(200, _canonical(response).encode("utf-8"),
                          {"X-Request-Id": headers["X-Request-Id"]})


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def benchmark_main(argv: Sequence[str] | None = None) -> int:
    """Replay exported, real agent prompts; callable for CPU HTTP integration."""
    parser = argparse.ArgumentParser(description="Benchmark actual WolfBench decision requests against vLLM")
    parser.add_argument("--requests-jsonl", required=True, type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--chat-template-kwargs")
    parser.add_argument("--output", type=Path, default=Path("output/vllm_benchmark"))
    args = parser.parse_args(argv)
    if args.num_requests <= 0 or args.rounds <= 0:
        parser.error("num-requests and rounds must be positive")
    rows = []
    for line in args.requests_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(_parse_json(line))
    if len(rows) < args.num_requests:
        parser.error(f"need {args.num_requests} actual requests; input has only {len(rows)}")
    rows = rows[:args.num_requests]
    schema = _parse_json(args.schema.read_text(encoding="utf-8")) if args.schema else rows[0].get("schema")
    if not isinstance(schema, dict):
        parser.error("provide --schema or input audit rows containing a JSON schema")
    if not args.schema and any(row.get("schema") != schema for row in rows):
        parser.error("a replay round must use one common decision schema")
    source_requests = [DecisionRequest(row["request_id"], row["system"], row["user"], row["seed"])
                       for row in rows]
    if len({req.request_id for req in source_requests}) != len(source_requests):
        parser.error("source requests must be distinct logical agent requests")
    # Audit files can settle out of order; replay their original logical order.
    if all("input_index" in row for row in rows):
        source_requests = [request for _, request in sorted(zip(rows, source_requests),
                           key=lambda item: (item[0].get("batch_number", 0), item[0]["input_index"]))]
    args.output.mkdir(parents=True, exist_ok=True)
    # Do not overwrite or append an unrelated benchmark's provenance.
    audit_path = args.output / "requests.jsonl"
    report_path = args.output / "benchmark.json"
    if audit_path.exists() or report_path.exists():
        parser.error("output already contains a benchmark; choose a new --output directory")
    kwargs = _parse_json(args.chat_template_kwargs) if args.chat_template_kwargs else None
    backend = VLLMBackend(
        model=args.model, base_url=args.base_url, model_revision=args.model_revision,
        concurrency=args.concurrency, request_timeout=args.request_timeout, retries=args.retries,
        max_tokens=args.max_tokens, temperature=args.temperature,
        api_key=os.environ.get("VLLM_API_KEY"), audit_path=audit_path,
        chat_template_kwargs=kwargs,
    )
    report = {
        "created_at": _utc_now(), "simulated": False,
        "source_requests_path": str(args.requests_jsonl.resolve()),
        "source_sha256": _sha(args.requests_jsonl.read_bytes()),
        "schema_sha256": _sha(_canonical(schema)), "provenance": backend.provenance,
        "logical_agents_per_round": args.num_requests, "rounds": [],
        "notes": ["Nonstreaming HTTP: latency is complete-response latency, not TTFT.",
                  "Later rounds may benefit from server KV-prefix reuse; no client response cache.",
                  "This replays actual exported agent observations; it does not advance market state.",
                  "Record server launch manifest and GPU inventory separately."],
    }

    def save_report() -> None:
        report["cumulative_stats"] = backend.stats
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(report_path)

    try:
        for round_index in range(args.rounds):
            requests = [DecisionRequest(f"benchmark-{round_index}/{req.request_id}",
                        req.system, req.user, req.seed) for req in source_requests]
            first_row = len(backend.audit_rows)
            before = backend.stats
            started = time.monotonic()
            failure = None
            try:
                backend.generate_batch(requests, schema)
            except BatchGenerationError as exc:
                failure = exc
            elapsed = time.monotonic() - started
            audit = backend.audit_rows[first_row:]
            durations = [row["elapsed_seconds"] for row in audit]
            total_latencies = [row["queue_seconds"] + row["elapsed_seconds"] for row in audit]
            after = backend.stats
            completion_tokens = after["completion_tokens"] - before["completion_tokens"]
            total_tokens = after["total_tokens"] - before["total_tokens"]
            summary = {
                "round": round_index, "elapsed_seconds": elapsed,
                "status": "failure" if failure else "success",
                "requests": len(requests), "successes": after["successes"] - before["successes"],
                "failures": after["failures"] - before["failures"],
                "attempts": after["attempts"] - before["attempts"],
                "requests_per_second": (after["successes"] - before["successes"]) / elapsed,
                "reported_completion_tokens": completion_tokens,
                "reported_total_tokens": total_tokens,
                "reported_completion_tokens_per_second": completion_tokens / elapsed,
                "reported_total_tokens_per_second": total_tokens / elapsed,
                "usage_missing_attempts": after["usage_missing_attempts"] - before["usage_missing_attempts"],
                "request_latency_mean_seconds": statistics.mean(durations),
                "request_latency_p50_seconds": _percentile(durations, 0.50),
                "request_latency_p95_seconds": _percentile(durations, 0.95),
                "request_latency_p99_seconds": _percentile(durations, 0.99),
                "queue_plus_request_p95_seconds": _percentile(total_latencies, 0.95),
            }
            if failure:
                summary["errors"] = failure.failures
            report["rounds"].append(summary)
            save_report()
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            if failure:
                print(str(failure), file=sys.stderr)
                return 1
            # Persistent on-disk audits already contain full prompts and replies.
            backend.audit_rows.clear()
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(benchmark_main())
