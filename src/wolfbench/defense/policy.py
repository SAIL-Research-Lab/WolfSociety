"""Defense-model interface for WolfBench.

A WolfBench defense submission implements :class:`WolfGuardPolicy`. The
environment hands the policy a per-day :class:`Observation`-like summary
(``dict``) and consumes a per-asset :class:`Intervention`-like decision
``dict``.

We keep the runtime types as plain dicts (matching the existing environment
contract) and expose dataclasses below so that submissions can validate
their I/O statically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from wolfbench.agents.wolfguard import WolfGuardConfig


@dataclass
class Observation:
    """Compressed per-day observation handed to the defense.

    ``oracle_view`` is only populated for internal upper-bound runs such as
    :class:`OracleWolfGuardPolicy`; public submissions do not receive it.
    """
    day: int
    market: dict[str, dict[str, float]]
    social: dict[str, dict[str, float]]
    recent_return: dict[str, float]
    oracle_view: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class Intervention:
    asset: str
    action: str            # "none" / "warning" / "cooldown" / "block"
    risk: float            # in [0, 1], used for cost accounting
    target_group: str = "all"
    intensity: float = 0.0
    reason: str = ""
    components: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class WolfGuardPolicy(Protocol):
    """Interface that every WolfBench defense submission must satisfy.

    Implementations must:
    * be deterministic given the same observation stream and seed;
    * not access env internals (only the per-day summary);
    * return a dict ``{asset_id: Intervention-like dict}`` per call.
    """

    name: str
    config: WolfGuardConfig

    def fit_baseline(self, baseline: dict[str, dict[str, float]]) -> None: ...

    def decide(self, day: int, summary: dict[str, Any]) -> dict[str, dict]: ...


def make_intervention(asset: str, action: str, risk: float = 0.0,
                      target_group: str = "all", intensity: float = 0.0,
                      reason: str = "",
                      components: dict[str, float] | None = None) -> dict:
    """Helper that emits the canonical action-dict shape consumed by env."""
    return {
        "asset": asset,
        "action": action,
        "risk": float(risk),
        "target_group": target_group,
        "intensity": float(intensity),
        "reason": reason,
        "components": components or {},
    }
