"""Scenario configuration types and loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_SCENARIO_DIR = Path(__file__).resolve().parent.parent / "config" / "scenarios"

SCENARIO_FILES = {
    "s0": "s0_clean.yaml",
    "s1": "s1_pump_dump.yaml",
    "s2": "s2_finfluencer.yaml",
    "s3": "s3_spoofing.yaml",
    "s4": "s4_wash.yaml",
}


@dataclass
class AssetConfig:
    id: str
    fundamental: float
    fundamental_vol: float
    initial_liquidity: float
    sector: str = "generic"


@dataclass
class ScenarioConfig:
    id: str
    name: str
    description: str
    horizon_days: int
    assets: list[AssetConfig]
    retail: dict[str, Any]
    market_makers: dict[str, Any]
    social: dict[str, Any]
    attackers: dict[str, Any] = field(default_factory=dict)
    defense: dict[str, Any] = field(default_factory=dict)
    target_asset: str = ""

    @property
    def asset_ids(self) -> list[str]:
        return [a.id for a in self.assets]

    def asset(self, asset_id: str) -> AssetConfig:
        for a in self.assets:
            if a.id == asset_id:
                return a
        raise KeyError(asset_id)


def load_scenario(name_or_path: str) -> ScenarioConfig:
    """Load scenario by short id (s0..s4) or by filepath."""
    if name_or_path in SCENARIO_FILES:
        path = _SCENARIO_DIR / SCENARIO_FILES[name_or_path]
    else:
        path = Path(name_or_path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScenarioConfig(
        id=data["id"],
        name=data["name"],
        description=data.get("description", ""),
        horizon_days=int(data["horizon_days"]),
        assets=[AssetConfig(**a) for a in data["assets"]],
        retail=data["retail"],
        market_makers=data["market_makers"],
        social=data["social"],
        attackers=data.get("attackers", {}) or {},
        defense=data.get("defense", {}) or {},
        target_asset=data.get("target_asset", data["assets"][0]["id"]),
    )
