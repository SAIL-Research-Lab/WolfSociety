"""LLM hook smoke tests: rule fallback must be deterministic and equivalent."""
import pytest

from wolfbench.agents.llm import (
    OpenAIChatBackend,
    RuleFallbackBackend, LLMPumpLeader, LLMFinfluencer, LLMRuleAssistWolfGuardAgent,
    LLMRiskWolfGuardAgent, LLMWolfGuardAgent,
    OpenRouterChatBackend, VLLMChatBackend, _loads_json_dict, make_chat_backend,
)
from wolfbench.agents.wolfguard import WolfGuardConfig
from wolfbench.defense import get_policy, get_track
from wolfbench.env.environment import WolfBenchEnv
from wolfbench.scenarios.base import load_scenario


def test_llm_pump_leader_with_rule_fallback_matches_rule():
    """With RuleFallbackBackend, LLMPumpLeader must yield identical episode."""
    scen = load_scenario("s1")
    a = WolfBenchEnv(scen, n_society=300, alpha=0.05, seed=7).run()
    b = WolfBenchEnv(scen, n_society=300, alpha=0.05, seed=7,
                     llm_backend=RuleFallbackBackend(), n_llm_leaders=4).run()
    assert a.metrics.max_collapse_score == b.metrics.max_collapse_score
    assert a.metrics.collapse_day == b.metrics.collapse_day


def test_llm_leader_count_capped():
    scen = load_scenario("s1")
    # request many LLM leaders; sublinear schedule caps at 10 globally
    env = WolfBenchEnv(scen, n_society=100000, alpha=0.20, seed=1,
                       llm_backend=RuleFallbackBackend(),
                       n_llm_leaders=1000)
    n_llm = sum(isinstance(a, LLMPumpLeader) for a in env.society.attackers)
    assert n_llm <= 10, f"LLM leaders not capped: got {n_llm}"


def test_llm_finfluencer_capped_independent_of_alpha():
    scen = load_scenario("s2")
    for alpha in [0.01, 0.05, 0.20]:
        env = WolfBenchEnv(scen, n_society=5000, alpha=alpha, seed=1,
                           llm_backend=RuleFallbackBackend(),
                           n_llm_leaders=100)
        n_llm = sum(isinstance(a, LLMFinfluencer) for a in env.society.attackers)
        assert n_llm <= 5, f"alpha={alpha}: LLM finfluencers={n_llm}"


def test_llm_count_sublinear_schedule():
    """Verify the (N, alpha) → K_LLM table from paper §LLM-scaling."""
    from wolfbench.scenarios.society import strategic_leader_count
    expected = {
        (100, 0.01): 1, (100, 0.05): 2, (100, 0.10): 2,
        (1000, 0.01): 2, (1000, 0.05): 3, (1000, 0.10): 3,
        (10000, 0.005): 2, (10000, 0.01): 3, (10000, 0.05): 5,
        (100000, 0.01): 6, (100000, 0.05): 10, (100000, 0.10): 10,
    }
    for (N, a), k in expected.items():
        n_h = max(1, int(round(a * N)))
        got = strategic_leader_count(N, a, n_h)
        assert got == k, f"N={N} alpha={a}: expected {k}, got {got}"


def test_vllm_backend_is_lazy_and_strict_by_default():
    backend = VLLMChatBackend(model="qwen3-8b", base_url="http://127.0.0.1:9/v1")
    assert backend.name == "vllm_chat"
    assert backend.strict is True
    assert backend.calls == 0


def test_openrouter_backend_is_lazy_and_strict_by_default():
    backend = OpenRouterChatBackend(model="anthropic/claude-3.5-haiku")
    assert backend.name == "openrouter_chat"
    assert backend.base_url == "https://openrouter.ai/api/v1"
    assert backend.strict is True
    assert backend.calls == 0


def test_make_chat_backend_supports_openrouter():
    backend = make_chat_backend(provider="openrouter", model="openai/gpt-4o-mini", api_key="test")
    assert isinstance(backend, OpenRouterChatBackend)
    assert backend.model == "openai/gpt-4o-mini"
    assert backend.api_key == "test"


def test_qwen_policy_factory_uses_vllm_without_calling_server():
    policy = get_policy("qwen", model="qwen3-8b",
                        base_url="http://127.0.0.1:9/v1")
    assert "Qwen3-vLLM" in policy.name
    assert policy.backend.name == "vllm_chat"
    assert policy.backend.calls == 0


def test_qwen_assisted_policy_factory_is_separate_track():
    policy = get_policy("qwen_assisted", model="qwen3-8b",
                        base_url="http://127.0.0.1:9/v1")
    assert isinstance(policy, LLMRuleAssistWolfGuardAgent)
    assert get_track("qwen_assisted") == "legacy_assisted_rule"
    assert get_track("qwen") == "llm_from_scratch"


def test_llm_policy_factory_supports_openrouter_from_scratch():
    policy = get_policy("llm", provider="openrouter", model="openai/gpt-4o-mini", api_key="test")
    assert isinstance(policy, LLMWolfGuardAgent)
    assert policy.backend.name == "openrouter_chat"
    assert get_track("llm") == "llm_from_scratch"


def test_qwen_risk_policy_factory_uses_risk_wrapper_without_calling_server():
    policy = get_policy("qwen_risk", model="qwen3-8b", base_url="http://127.0.0.1:9/v1")
    assert isinstance(policy, LLMRiskWolfGuardAgent)
    assert policy.name == "Qwen-Risk-WolfGuard"
    assert policy.backend.name == "vllm_chat"
    assert policy.backend.calls == 0
    assert get_track("qwen_risk") == "open_llm_risk"


def test_llm_policy_factory_reads_provider_from_env(monkeypatch):
    monkeypatch.setenv("WOLFBENCH_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("WOLFBENCH_OPENROUTER_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    policy = get_policy("llm")
    assert isinstance(policy, LLMWolfGuardAgent)
    assert policy.backend.name == "openrouter_chat"
    assert policy.backend.model == "openai/gpt-4o-mini"


def test_llm_json_parser_strips_qwen_thinking_and_fences():
    content = '<think>draft</think>\n```json\n{"asset_2": {"action": "warning"}}\n```'
    assert _loads_json_dict(content) == {"asset_2": {"action": "warning"}}


def test_chat_backend_json_retry_recovers_without_counting_failure():
    class FlakyJSONBackend(OpenAIChatBackend):
        def __init__(self):
            super().__init__(model="test", strict=True, json_retries=1)
            self.raw_calls = 0

        def _complete(self, system, user, use_response_format, extra_body):
            self.raw_calls += 1
            if self.raw_calls == 1:
                return "not-json"
            return '{"ok": true}'

    backend = FlakyJSONBackend()

    assert backend.chat_json("", "") == {"ok": True}
    assert backend.calls == 1
    assert backend.failures == 0
    assert backend.recovered_failures == 1
    assert backend.last_recovered_error_type == "ValueError"


class CaptureBackend:
    name = "capture"

    def __init__(self, response=None, strict=False):
        self.user = ""
        self.response = response or {}
        self.strict = strict

    def chat_json(self, system, user):
        self.user = user
        return self.response


def test_llm_wolfguard_does_not_prompt_with_oracle_view():
    backend = CaptureBackend()
    agent = LLMWolfGuardAgent(backend=backend, config=WolfGuardConfig())
    summary = {
        "day": 1,
        "market": {"asset_2": {"price": 4.0, "fundamental": 4.0}},
        "social": {"asset_2": {}},
        "recent_return": {"asset_2": 0.0},
        "oracle_view": {"asset_2": {"harmful_pressure": 1.0}},
    }
    agent.decide(1, summary)
    assert "oracle_view" not in backend.user
    assert '"actions"' not in backend.user


def test_llm_wolfguard_can_act_without_rule_action_seed():
    backend = CaptureBackend({"asset_2": {"action": "warning", "risk": 0.6}})
    agent = LLMWolfGuardAgent(backend=backend, config=WolfGuardConfig())
    summary = {
        "day": 1,
        "market": {"asset_2": {"price": 4.0, "fundamental": 4.0}},
        "social": {"asset_2": {}},
        "recent_return": {"asset_2": 0.0},
        "oracle_view": {},
    }
    actions = agent.decide(1, summary)
    assert actions["asset_2"]["action"] == "warning"
    assert actions["asset_2"]["risk"] == 0.6


def test_llm_risk_wolfguard_uses_threshold_layer_and_hides_oracle():
    backend = CaptureBackend({
        "asset_2": {"manipulation_risk": 0.8, "cascade_risk": 0.6, "confidence": 1.0}
    })
    agent = LLMRiskWolfGuardAgent(backend=backend, config=WolfGuardConfig(), warning_threshold=0.5, cooldown_threshold=0.9)
    summary = {
        "day": 1,
        "market": {"asset_2": {"price": 5.0, "fundamental": 4.0}},
        "social": {"asset_2": {}},
        "recent_return": {"asset_2": 0.0},
        "oracle_view": {"asset_2": {"harmful_pressure": 1.0}},
    }

    actions = agent.decide(1, summary)

    assert "oracle_view" not in backend.user
    assert actions["asset_2"]["action"] == "warning"
    assert actions["asset_2"]["components"]["manipulation_risk"] == 0.8
    assert agent.llm_semantic_fallbacks == 0
    assert agent.llm_valid_asset_decisions == 1


def test_llm_risk_wolfguard_strict_rejects_non_asset_keyed_json():
    backend = CaptureBackend({"ok": True, "score": 0.5}, strict=True)
    agent = LLMRiskWolfGuardAgent(backend=backend, config=WolfGuardConfig())
    summary = {
        "day": 1,
        "market": {"asset_2": {"price": 5.0, "fundamental": 4.0}},
        "social": {"asset_2": {}},
        "oracle_view": {},
    }

    with pytest.raises(RuntimeError, match="invalid risk JSON"):
        agent.decide(1, summary)

    assert agent.llm_semantic_fallbacks == 1
    assert agent.llm_expected_asset_decisions == 1
