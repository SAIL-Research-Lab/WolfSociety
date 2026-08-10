"""WolfBench Defense interface and baseline policies.

WolfBench is a defense-model evaluation benchmark: any model that conforms to
the :class:`WolfGuardPolicy` interface can be plugged into the environment as
the defender. The :func:`get_policy` registry exposes the canonical baselines
shipped with WolfBench v1.
"""
from wolfbench.defense.policy import (
    Intervention,
    Observation,
    WolfGuardPolicy,
)
from wolfbench.defense.baselines import (
    NoGuardPolicy,
    RandomGuardPolicy,
    RuleWolfGuardPolicy,
    OracleWolfGuardPolicy,
    DistilledWolfGuardPolicy,
    CalibratedDistilledWolfGuardPolicy,
    CostAwareZScoreWolfGuardPolicy,
    TopologyAwareWolfGuardPolicy,
    LLMWolfGuardPolicy,
    LLMRiskWolfGuardPolicy,
    QwenVLLMWolfGuardPolicy,
    BASELINES,
    get_track,
    get_policy,
)

__all__ = [
    "Intervention",
    "Observation",
    "WolfGuardPolicy",
    "NoGuardPolicy",
    "RandomGuardPolicy",
    "RuleWolfGuardPolicy",
    "OracleWolfGuardPolicy",
    "DistilledWolfGuardPolicy",
    "CalibratedDistilledWolfGuardPolicy",
    "CostAwareZScoreWolfGuardPolicy",
    "TopologyAwareWolfGuardPolicy",
    "LLMWolfGuardPolicy",
    "LLMRiskWolfGuardPolicy",
    "QwenVLLMWolfGuardPolicy",
    "BASELINES",
    "get_track",
    "get_policy",
]
