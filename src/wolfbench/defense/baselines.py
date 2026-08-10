"""Canonical defense baselines shipped with WolfBench v1.

Run any of these by name through the CLI::

    wolfbench evaluate --defense rule --scenario s1 --alpha 0.05

or programmatically::

    from wolfbench.defense import get_policy
    pol = get_policy("rule")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from wolfbench.agents.wolfguard import WolfGuardAgent, WolfGuardConfig
from wolfbench.defense.advanced import (
    CalibratedDistilledWolfGuardPolicy,
    CostAwareZScoreWolfGuardPolicy,
    TopologyAwareWolfGuardPolicy,
)
from wolfbench.defense.distilled import DistilledWolfGuardPolicy
from wolfbench.defense.policy import make_intervention


# ---------------------------------------------------------------- NoGuard
@dataclass
class NoGuardPolicy:
    """Reference baseline: never intervenes. Used to compute ΔH for scoring."""
    name: str = "NoGuard"
    config: WolfGuardConfig = field(default_factory=lambda: WolfGuardConfig(mode="off"))
    clean_baseline: dict = field(default_factory=dict)

    def fit_baseline(self, baseline):
        self.clean_baseline = baseline

    def decide(self, day, summary):
        return {}


# ---------------------------------------------------------------- Random
@dataclass
class RandomGuardPolicy:
    """Sanity-check baseline: each day issues random warnings/cooldowns.

    Should score near (or below) NoGuard because it pays utility cost without
    information.
    """
    name: str = "RandomGuard"
    config: WolfGuardConfig = field(default_factory=WolfGuardConfig)
    seed: int = 0
    p_warn: float = 0.15
    p_cooldown: float = 0.05
    clean_baseline: dict = field(default_factory=dict)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def fit_baseline(self, baseline):
        self.clean_baseline = baseline

    def decide(self, day, summary):
        out = {}
        for asset in summary["market"]:
            r = float(self._rng.random())
            if r < self.p_cooldown:
                out[asset] = make_intervention(asset, "cooldown", risk=0.7,
                                               reason="random")
            elif r < self.p_cooldown + self.p_warn:
                out[asset] = make_intervention(asset, "warning", risk=0.5,
                                               reason="random")
        return out


# ---------------------------------------------------------------- Rule
class RuleWolfGuardPolicy(WolfGuardAgent):
    """The z-score rule detector originally shipped as ``WolfGuardAgent``.

    It is the canonical non-learning baseline that defense submissions must
    beat to claim contribution.
    """
    name: str = "Rule-WolfGuard"


# ---------------------------------------------------------------- Oracle
@dataclass
class OracleWolfGuardPolicy:
    """Upper-bound baseline: reads explicitly supplied ground-truth harmful
    pressure and triggers proportional interventions.

    Excluded from the official leaderboard.
    """
    name: str = "Oracle-WolfGuard"
    config: WolfGuardConfig = field(default_factory=WolfGuardConfig)
    risk_warning: float = 0.001
    risk_cooldown: float = 0.005
    risk_block: float = 0.02
    clean_baseline: dict = field(default_factory=dict)

    def fit_baseline(self, baseline):
        self.clean_baseline = baseline

    def decide(self, day, summary):
        oracle = summary.get("oracle_view", {}) or {}
        out = {}
        for asset, info in oracle.items():
            active_count = float(info.get("active_harmful_count", 0.0))
            pressure = max(
                float(info.get("active_harmful_share", 0.0)),
                float(info.get("harmful_agent_share", 0.0)),
                float(info.get("harmful_pressure", 0.0)),
            )
            if not info.get("is_attack_target", 0.0) and pressure <= 0.0:
                continue
            risk = min(1.0, pressure / max(self.risk_block, 1e-9))
            if info.get("is_attack_target", 0.0) and active_count > 0.0:
                action = "block"
                risk = max(risk, 1.0)
            elif pressure >= self.risk_block:
                action = "block"
            elif pressure >= self.risk_cooldown:
                action = "cooldown"
            elif pressure >= self.risk_warning:
                action = "warning"
            else:
                continue
            out[asset] = make_intervention(asset, action, risk=risk,
                                           reason="oracle",
                                           components={"oracle_pressure": pressure})
        return out


# ---------------------------------------------------------------- LLM
def _llm_policy_factory(model: str | None = None,
                        provider: str | None = None,
                        base_url: str | None = None,
                        api_key: str | None = None,
                        strict: bool | None = None,
                        display_name: str | None = None,
                        extra_body: dict | None = None,
                        assisted: bool = False,
                        risk_only: bool = False):
    """Return an ``LLMWolfGuardAgent`` wrapper. Imported lazily so users
    without an OpenAI client installed can still use rule baselines."""
    from wolfbench.agents.llm import (
        LLMRiskWolfGuardAgent, LLMRuleAssistWolfGuardAgent, LLMWolfGuardAgent,
        RuleFallbackBackend, make_chat_backend,
    )
    env_provider = os.getenv("WOLFBENCH_LLM_PROVIDER")
    if model or provider or base_url or api_key or env_provider:
        backend = make_chat_backend(
            provider=provider, model=model, base_url=base_url,
            api_key=api_key, strict=strict,
        )
    else:
        backend = RuleFallbackBackend()
    if extra_body is not None and hasattr(backend, "extra_body"):
        backend.extra_body = extra_body
    if risk_only:
        agent_cls = LLMRiskWolfGuardAgent
    else:
        agent_cls = LLMRuleAssistWolfGuardAgent if assisted else LLMWolfGuardAgent
    if risk_only:
        if hasattr(backend, "temperature"):
            backend.temperature = float(os.getenv("WOLFBENCH_RISK_LLM_TEMPERATURE", "0.0"))
        if hasattr(backend, "max_tokens"):
            backend.max_tokens = int(os.getenv("WOLFBENCH_RISK_LLM_MAX_TOKENS", "512"))
    agent = agent_cls(backend=backend, config=WolfGuardConfig())
    label = "Risk-WolfGuard" if risk_only else "LLM-WolfGuard"
    agent.name = display_name or f"{label}({getattr(backend, 'model', 'rule_fallback')})"
    return agent


class LLMWolfGuardPolicy:
    """Factory shim for documentation; instantiate via ``get_policy('llm')``."""
    name: str = "LLM-WolfGuard"


class QwenVLLMWolfGuardPolicy:
    """Factory shim; instantiate via ``get_policy('qwen')``."""
    name: str = "Qwen3-vLLM-WolfGuard"


class LLMRiskWolfGuardPolicy:
    """Factory shim; instantiate via ``get_policy('llm_risk')``."""
    name: str = "LLM-Risk-WolfGuard"


OPEN_LLM_RISK_DEFAULTS = {
    "deepseek_risk": ("openrouter", "deepseek/deepseek-chat-v3-0324", "DeepSeek-Risk-WolfGuard"),
    "deepseek_v3_risk": ("openrouter", "deepseek/deepseek-v3.2", "DeepSeek-V3.2-Risk-WolfGuard"),
    "deepseek_v4_risk": ("openrouter", "deepseek/deepseek-v4-pro", "DeepSeek-V4-Pro-Risk-WolfGuard"),
    "qwen235b_risk": ("openrouter", "qwen/qwen3-235b-a22b", "Qwen3-235B-A22B-Risk-WolfGuard", {"reasoning": {"enabled": False}}),
    "qwen36_35b_risk": ("openrouter", "qwen/qwen3.6-35b-a3b", "Qwen3.6-35B-A3B-Risk-WolfGuard", {"reasoning": {"enabled": False}}),
    "glm45_risk": ("openrouter", "z-ai/glm-4.5", "GLM-4.5-Risk-WolfGuard", {"reasoning": {"enabled": False}}),
    "glm52_risk": ("openrouter", "z-ai/glm-5.2", "GLM-5.2-Risk-WolfGuard", {"reasoning": {"enabled": False}}),
    "llama33_70b_risk": ("openrouter", "meta-llama/llama-3.3-70b-instruct", "Llama-3.3-70B-Risk-WolfGuard"),
    "llama4_maverick_risk": ("openrouter", "meta-llama/llama-4-maverick", "Llama-4-Maverick-Risk-WolfGuard"),
    "claude_opus_risk": ("openrouter", "anthropic/claude-opus-4.8", "Claude-Opus-Risk-WolfGuard"),
    "gemini25_pro_risk": ("openrouter", "google/gemini-2.5-pro", "Gemini-2.5-Pro-Risk-WolfGuard"),
    "gpt41_risk": ("openrouter", "openai/gpt-4.1", "GPT-4.1-Risk-WolfGuard"),
    "gpt4o_mini_risk": ("openrouter", "openai/gpt-4o-mini", "GPT-4o-mini-Risk-WolfGuard"),
    "llama_risk": ("openrouter", "meta-llama/llama-3.1-8b-instruct", "Llama-Risk-WolfGuard"),
    "mistral_risk": ("openrouter", "mistralai/mistral-7b-instruct", "Mistral-Risk-WolfGuard"),
    "qwen_risk": ("vllm", "qwen3-8b", "Qwen-Risk-WolfGuard"),
}


TRACKS = {
    "noguard": "control",
    "random": "control",
    "rule": "rule_baseline",
    "zscore_guard": "traditional_baseline",
    "oracle": "oracle_upper_bound",
    "llm": "llm_from_scratch",
    "qwen": "llm_from_scratch",
    "llm_risk": "open_llm_risk",
    "qwen_risk": "open_llm_risk",
    "claude_opus_risk": "open_llm_risk",
    "deepseek_risk": "open_llm_risk",
    "deepseek_v3_risk": "open_llm_risk",
    "deepseek_v4_risk": "open_llm_risk",
    "qwen235b_risk": "open_llm_risk",
    "qwen36_35b_risk": "open_llm_risk",
    "glm45_risk": "open_llm_risk",
    "glm52_risk": "open_llm_risk",
    "llama33_70b_risk": "open_llm_risk",
    "llama4_maverick_risk": "open_llm_risk",
    "gemini25_pro_risk": "open_llm_risk",
    "gpt41_risk": "open_llm_risk",
    "gpt4o_mini_risk": "open_llm_risk",
    "llama_risk": "open_llm_risk",
    "mistral_risk": "open_llm_risk",
    "llm_assisted": "legacy_assisted_rule",
    "qwen_assisted": "legacy_assisted_rule",
    "distilled": "simulator_trained_baseline",
    "calibrated_distilled": "simulator_trained_calibrated",
    "topology_aware": "submission",
}


def get_track(name: str) -> str:
    return TRACKS.get(name.lower(), "submission")


# ---------------------------------------------------------------- Registry
BASELINES = {
    "noguard": NoGuardPolicy,
    "random": RandomGuardPolicy,
    "rule": RuleWolfGuardPolicy,
    "zscore_guard": CostAwareZScoreWolfGuardPolicy,
    "oracle": OracleWolfGuardPolicy,
    "distilled": DistilledWolfGuardPolicy,
    "calibrated_distilled": CalibratedDistilledWolfGuardPolicy,
    "topology_aware": TopologyAwareWolfGuardPolicy,
}


def get_policy(name: str, **kwargs):
    """Instantiate a baseline by short name.

    ``name`` ∈ ``{noguard, random, rule, oracle, distilled,
    calibrated_distilled, topology_aware, llm, qwen, llm_risk, qwen_risk,
    claude_opus_risk, deepseek_risk, gemini25_pro_risk, gpt41_risk,
    gpt4o_mini_risk, llama_risk, mistral_risk, llm_assisted,
    qwen_assisted}``. ``llm`` accepts
    a ``model`` kwarg (defaults to the deterministic rule fallback). ``qwen``
    uses the local vLLM OpenAI-compatible endpoint by default. ``distilled`` and
    ``calibrated_distilled`` accept ``model_path`` or read
    ``WOLFBENCH_DISTILLED_MODEL``.
    """
    key = name.lower()
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if key in {"llm", "llm_assisted"}:
        return _llm_policy_factory(
            model=kwargs.get("model"),
            provider=kwargs.get("provider"),
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            strict=kwargs.get("strict"),
            assisted=key.endswith("_assisted"),
        )
    if key == "llm_risk":
        return _llm_policy_factory(
            model=kwargs.get("model"),
            provider=kwargs.get("provider"),
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            strict=kwargs.get("strict"),
            risk_only=True,
        )
    if key in OPEN_LLM_RISK_DEFAULTS:
        defaults = OPEN_LLM_RISK_DEFAULTS[key]
        default_provider, default_model, display_name = defaults[:3]
        extra_body = defaults[3] if len(defaults) > 3 else None
        return _llm_policy_factory(
            model=kwargs.get("model") or default_model,
            provider=kwargs.get("provider") or default_provider,
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            strict=True if kwargs.get("strict") is None else kwargs.get("strict"),
            display_name=display_name,
            extra_body=extra_body,
            risk_only=True,
        )
    if key in {"qwen", "qwen_assisted"}:
        model = kwargs.get("model") or "qwen3-8b"
        return _llm_policy_factory(
            model=model,
            provider="vllm",
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            strict=True if kwargs.get("strict") is None else kwargs.get("strict"),
            display_name=f"Qwen3-vLLM{'-assisted' if key.endswith('_assisted') else ''}({model})",
            assisted=key.endswith("_assisted"),
        )
    kwargs = {k: v for k, v in kwargs.items()
              if k not in {"model", "provider", "base_url", "api_key", "strict"}}
    if key not in BASELINES:
        raise ValueError(
            f"Unknown defense baseline '{name}'. "
            f"Available: {sorted(BASELINES) + ['llm', 'qwen', 'llm_risk', *sorted(OPEN_LLM_RISK_DEFAULTS), 'llm_assisted', 'qwen_assisted']}"
        )
    if key not in {"distilled", "calibrated_distilled"}:
        kwargs.pop("model_path", None)
    return BASELINES[key](**kwargs)
