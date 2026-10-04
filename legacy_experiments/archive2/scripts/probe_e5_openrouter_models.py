"""Tiny real-API probe for E5 defense model availability and JSON validity."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from wolfbench.agents.llm import _WOLFGUARD_RISK_SYS, _invalid_risk_assets, _normalize_risk_plan

from .hybrid_runtime import (
    DEFENSE_MODEL_ALIASES,
    DEFENSE_MODEL_FALLBACKS,
    DEFENSE_MODEL_OPTIONS,
    DEFENSE_MODEL_RESPONSE_FORMAT,
)
from .openrouter_backend import make_openrouter_backend


ALIASES = ("deepseek_v3_risk", "qwen36_35b_risk", "glm52_risk", "llama4_maverick_risk")


def _probe(alias: str, cache_root: Path) -> dict[str, Any]:
    required_assets = ["asset_1", "asset_2"]
    system = _WOLFGUARD_RISK_SYS
    user = json.dumps({
        "day": 1,
        "required_assets": required_assets,
        "risk_features": {
            "asset_1": {"risk": 0.2},
            "asset_2": {"risk": 0.8},
        },
    })
    backend = make_openrouter_backend(
        DEFENSE_MODEL_ALIASES[alias],
        cache_dir=cache_root / alias,
        strict=True,
        temperature=0.0,
        max_tokens=256,
        fallback_models=DEFENSE_MODEL_FALLBACKS.get(alias, ()),
        response_format=DEFENSE_MODEL_RESPONSE_FORMAT.get(alias, True),
        extra_body=DEFENSE_MODEL_OPTIONS.get(alias, {}),
    )
    try:
        out = backend.chat_json(system, user)
        plan = _normalize_risk_plan(out, required_assets)
        invalid_assets = _invalid_risk_assets(plan, required_assets)
        if invalid_assets:
            raise ValueError(f"schema check failed for assets: {','.join(invalid_assets)}")
        snapshot = backend.snapshot()
        requested_model = snapshot.get("requested_model", "")
        resolved_model = snapshot.get("model", "")
        status = "ok" if resolved_model == requested_model else "ok_fallback"
        return {
            "alias": alias,
            "status": status,
            "requested_model": requested_model,
            "resolved_model": resolved_model,
            "keys": ",".join(sorted(plan.keys())),
            "recovered_failures": snapshot.get("recovered_failures", 0),
            "failures": snapshot.get("failures", 0),
            "last_error_type": snapshot.get("last_error_type", ""),
            "error": "",
        }
    except Exception as exc:
        snapshot = backend.snapshot()
        return {
            "alias": alias,
            "status": "fail",
            "requested_model": snapshot.get("requested_model", ""),
            "resolved_model": snapshot.get("model", ""),
            "keys": "",
            "recovered_failures": snapshot.get("recovered_failures", 0),
            "failures": snapshot.get("failures", 0),
            "last_error_type": snapshot.get("last_error_type", ""),
            "error": repr(exc)[:500],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="final_hybrid_experiments/outputs/e5_openrouter_probe.csv")
    parser.add_argument("--cache-root", default="/private/tmp/wolfbench-openrouter-probe-v5")
    parser.add_argument("--aliases", default=",".join(ALIASES))
    args = parser.parse_args()
    aliases = [item.strip() for item in args.aliases.split(",") if item.strip()]
    cache_root = Path(args.cache_root)
    rows = [_probe(alias, cache_root) for alias in aliases]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for alias in aliases:
        row = next((item for item in rows if item["alias"] == alias), None)
        if not row:
            continue
        if row["status"] in {"ok", "ok_fallback"}:
            label = "OK" if row["status"] == "ok" else "OK_FALLBACK"
            print(alias, label, "requested=", row["requested_model"], "resolved=", row["resolved_model"], "keys=", row["keys"])
        else:
            print(alias, "FAIL", "requested=", row["requested_model"], "last_model=", row["resolved_model"], "error=", row["error"])
    print(f"Wrote {out_path}")
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
