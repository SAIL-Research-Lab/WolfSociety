"""OpenRouter JSON backends with deterministic caching and budget accounting."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wolfbench.agents.llm import _loads_json_dict

from .io_utils import CACHE, ensure_dir


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Conservative defaults used only when OpenRouter usage metadata lacks cost.
# Values are dollars per million tokens.
MODEL_PRICE_PER_MTOK = {
    "openai/gpt-4.1": {"prompt": 2.00, "completion": 8.00},
    "openai/gpt-4.1-mini": {"prompt": 0.40, "completion": 1.60},
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "anthropic/claude-opus-4.8": {"prompt": 15.00, "completion": 75.00},
    "google/gemini-2.5-pro": {"prompt": 1.25, "completion": 10.00},
    "google/gemini-2.5-flash": {"prompt": 0.30, "completion": 2.50},
    "google/gemini-2.5-flash-lite": {"prompt": 0.10, "completion": 0.40},
    "google/gemini-2.0-flash-001": {"prompt": 0.10, "completion": 0.40},
    "google/gemini-flash-1.5": {"prompt": 0.075, "completion": 0.30},
    "deepseek/deepseek-v3.2": {"prompt": 0.27, "completion": 1.10},
    "deepseek/deepseek-v4-pro": {"prompt": 0.435, "completion": 0.87},
    "qwen/qwen3-235b-a22b": {"prompt": 0.20, "completion": 0.60},
    "qwen/qwen3.6-35b-a3b": {"prompt": 0.14, "completion": 1.00},
    "z-ai/glm-4.5": {"prompt": 0.60, "completion": 2.20},
    "z-ai/glm-5.2": {"prompt": 0.93, "completion": 3.00},
    "meta-llama/llama-3.3-70b-instruct": {"prompt": 0.12, "completion": 0.30},
    "meta-llama/llama-4-maverick": {"prompt": 0.15, "completion": 0.60},
}


def rough_token_count(text: str) -> int:
    return max(1, int(len(text) / 4))


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = MODEL_PRICE_PER_MTOK.get(model, {"prompt": 1.0, "completion": 3.0})
    return (
        prompt_tokens * float(price["prompt"])
        + completion_tokens * float(price["completion"])
    ) / 1_000_000.0


@dataclass
class BackendStats:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    recovered_failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    last_error: str = ""
    last_error_type: str = ""


@dataclass
class OpenRouterJSONBackend:
    """Minimal OpenRouter client for JSON-only agent plans.

    The class intentionally uses the OpenRouter HTTP endpoint directly instead
    of vLLM or an OpenAI-specific runtime. Responses are cached by
    model/system/user/temperature/max_tokens so interrupted experiment runs can
    resume without paying twice for identical calls.
    """

    model: str
    requested_model: str = ""
    fallback_models: tuple[str, ...] = field(default_factory=tuple)
    api_key: str | None = None
    base_url: str = OPENROUTER_BASE_URL
    cache_dir: Path = field(default_factory=lambda: CACHE)
    temperature: float = 0.2
    max_tokens: int = 160
    timeout: float = 60.0
    response_format: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    strict: bool = True
    max_retries: int = field(default_factory=lambda: int(os.getenv("WOLFBENCH_FINAL_OPENROUTER_RETRIES", "3")))
    json_retries: int = field(default_factory=lambda: int(os.getenv("WOLFBENCH_FINAL_OPENROUTER_JSON_RETRIES", "2")))
    retry_base_sleep: float = field(default_factory=lambda: float(os.getenv("WOLFBENCH_FINAL_OPENROUTER_RETRY_SLEEP", "1.5")))
    request_pause: float = field(default_factory=lambda: float(os.getenv("WOLFBENCH_FINAL_OPENROUTER_REQUEST_PAUSE", "0.0")))
    app_name: str = "WolfBench-Final-Hybrid"
    site_url: str = "https://openrouter.ai"
    name: str = "openrouter_json"
    stats: BackendStats = field(default_factory=BackendStats)

    def __post_init__(self) -> None:
        if not self.requested_model:
            self.requested_model = self.model

    @property
    def calls(self) -> int:
        return self.stats.calls

    @property
    def failures(self) -> int:
        return self.stats.failures

    @property
    def cache_hits(self) -> int:
        return self.stats.cache_hits

    @property
    def estimated_cost_usd(self) -> float:
        return self.stats.estimated_cost_usd

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        ensure_dir(self.cache_dir)
        last_exc: Exception | None = None
        for model in self._candidate_models():
            self.model = model
            key = self._cache_key(system, user, model=model)
            path = self.cache_dir / f"{key}.json"
            if path.exists():
                self.stats.cache_hits += 1
                cached = json.loads(path.read_text())
                return dict(cached.get("response", {}))
            max_rounds = max(1, int(self.json_retries) + 1)
            for json_attempt in range(max_rounds):
                try:
                    attempt_system = self._json_retry_system(system, json_attempt)
                    raw = self._post_with_retries(attempt_system, user, model=model)
                    content = raw["choices"][0]["message"].get("content") or "{}"
                    parsed = _loads_json_dict(content)
                    usage = raw.get("usage", {}) or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or rough_token_count(attempt_system + user))
                    completion_tokens = int(usage.get("completion_tokens") or rough_token_count(content))
                    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
                    cost = float(
                        usage.get("cost")
                        or estimate_cost_usd(model, prompt_tokens, completion_tokens)
                    )
                    self._record_usage(prompt_tokens, completion_tokens, total_tokens, cost)
                    if last_exc is not None:
                        self.stats.recovered_failures += 1
                    path.write_text(json.dumps({
                        "requested_model": self.requested_model,
                        "model": model,
                        "created_at": time.time(),
                        "system_sha": hashlib.sha256(system.encode()).hexdigest(),
                        "user_sha": hashlib.sha256(user.encode()).hexdigest(),
                        "response": parsed,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "estimated_cost_usd": cost,
                        },
                    }, indent=2, sort_keys=True) + "\n")
                    return parsed
                except Exception as exc:
                    last_exc = exc
                    if _is_model_unavailable(exc):
                        break
                    continue
        self.stats.failures += 1
        self.stats.last_error = repr(last_exc)
        self.stats.last_error_type = type(last_exc).__name__ if last_exc else "RuntimeError"
        if self.strict:
            raise last_exc or RuntimeError("OpenRouter JSON call failed")
        return {}

    def _candidate_models(self) -> list[str]:
        out: list[str] = []
        for model in (self.model, *self.fallback_models):
            if model and model not in out:
                out.append(model)
        return out

    def _json_retry_system(self, system: str, attempt: int) -> str:
        if attempt <= 0:
            return system
        return (
            f"{system}\n\n"
            "Return only one valid JSON object. Do not include Markdown, prose, "
            "lists as the top-level value, or reasoning text. If the task asks "
            "for per-asset risks, the top-level object must be keyed by asset id."
        )

    def _post_with_retries(self, system: str, user: str, model: str) -> dict[str, Any]:
        last_exc: Exception | None = None
        attempts = max(1, int(self.max_retries) + 1)
        for attempt in range(attempts):
            try:
                if self.request_pause > 0:
                    time.sleep(float(self.request_pause))
                return self._post(system, user, model=model)
            except Exception as exc:
                last_exc = exc
                if _is_model_unavailable(exc):
                    break
                if attempt >= attempts - 1:
                    break
                sleep_s = float(self.retry_base_sleep) * (2 ** attempt)
                time.sleep(sleep_s)
        raise last_exc or RuntimeError("OpenRouter request failed")

    def _record_usage(self, prompt_tokens: int, completion_tokens: int,
                      total_tokens: int, cost: float) -> None:
        self.stats.calls += 1
        self.stats.prompt_tokens += int(prompt_tokens)
        self.stats.completion_tokens += int(completion_tokens)
        self.stats.total_tokens += int(total_tokens)
        self.stats.estimated_cost_usd += float(cost)

    def _post(self, system: str, user: str, model: str) -> dict[str, Any]:
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for real OpenRouter calls.")
        payload = {
            "model": model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.response_format:
            payload["response_format"] = {"type": "json_object"}
        if self.extra_body:
            payload.update(self.extra_body)
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", self.site_url),
                "X-OpenRouter-Title": os.getenv("OPENROUTER_APP_NAME", self.app_name),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc

    def _cache_key(self, system: str, user: str, model: str | None = None) -> str:
        material = json.dumps({
            "model": model or self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": self.response_format,
            "extra_body": self.extra_body,
            "system": system,
            "user": user,
            "schema": "json_object_v1",
        }, sort_keys=True)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "requested_model": self.requested_model,
            "model": self.model,
            "fallback_models": list(self.fallback_models),
            "calls": self.stats.calls,
            "cache_hits": self.stats.cache_hits,
            "failures": self.stats.failures,
            "recovered_failures": self.stats.recovered_failures,
            "prompt_tokens": self.stats.prompt_tokens,
            "completion_tokens": self.stats.completion_tokens,
            "total_tokens": self.stats.total_tokens,
            "estimated_cost_usd": self.stats.estimated_cost_usd,
            "last_error_type": self.stats.last_error_type,
        }


def _is_model_unavailable(exc: Exception) -> bool:
    text = repr(exc).lower()
    return (
        "http 403" in text
        or "not available in your region" in text
        or "model_not_found" in text
        or "no endpoints found" in text
    )


@dataclass
class MockOpenRouterJSONBackend(OpenRouterJSONBackend):
    """Deterministic local backend used only for no-network smoke tests."""

    model: str = "mock/openrouter-json"
    strict: bool = True
    name: str = "mock_openrouter_json"

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        self.stats.calls += 1
        prompt_tokens = rough_token_count(system + user)
        completion_tokens = 30
        self.stats.prompt_tokens += prompt_tokens
        self.stats.completion_tokens += completion_tokens
        self.stats.total_tokens += prompt_tokens + completion_tokens
        return self._mock_plan(system, user)

    def _mock_plan(self, system: str, user: str) -> dict[str, Any]:
        try:
            payload = json.loads(user)
        except json.JSONDecodeError:
            payload = {}
        day = int(payload.get("day", 0))
        lower = system.lower()
        if "risk-wolfguard" in lower or "manipulation_risk" in lower:
            assets = payload.get("required_assets") or list(payload.get("risk_features", {}))
            return {
                asset: {
                    "manipulation_risk": 0.65 if "2" in asset else 0.35,
                    "cascade_risk": 0.55 if day >= 5 else 0.25,
                    "confidence": 0.8,
                }
                for asset in assets
            }
        if "benign retail" in lower:
            return {
                "risk_appetite_multiplier": 0.9 if day >= 15 else 1.05,
                "social_skepticism": 0.35,
                "share_public_message": day in {5, 10, 15},
                "message_sentiment": 0.2,
                "message_intensity": 0.25,
            }
        if "spoof" in lower:
            return {
                "spoof_size_mult": 7.0 if day < 18 else 4.0,
                "daily_cycles": 5 if day < 18 else 2,
                "side_bias": "buy" if day % 2 == 0 else "sell",
            }
        if "wash" in lower:
            return {
                "wash_volume_multiplier": 5.0 if 5 <= day <= 18 else 2.5,
                "accelerate_withdrawal": day >= 19,
            }
        if "finfluencer" in lower:
            return {
                "post_intensity": 2.1 if day < 15 else 1.1,
                "sell_today": day >= 14,
            }
        return {
            "promote_intensity": 1.9 if day < 15 else 0.8,
            "dump_today": day >= 15,
            "narrative_sentiment": 1.0 if day < 15 else -0.3,
        }


def make_openrouter_backend(
    model: str,
    cache_dir: str | Path | None = None,
    mock: bool = False,
    strict: bool = True,
    temperature: float = 0.2,
    max_tokens: int = 160,
    fallback_models: tuple[str, ...] | list[str] = (),
    response_format: bool = True,
    extra_body: dict[str, Any] | None = None,
) -> OpenRouterJSONBackend:
    cls = MockOpenRouterJSONBackend if mock else OpenRouterJSONBackend
    return cls(
        model=model,
        requested_model=model,
        fallback_models=tuple(fallback_models),
        cache_dir=Path(cache_dir) if cache_dir is not None else CACHE,
        strict=strict,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        extra_body=dict(extra_body or {}),
    )
