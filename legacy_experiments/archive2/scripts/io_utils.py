"""Small file/env helpers for final mixed-agent experiments."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
CACHE = ROOT / "cache" / "openrouter"
MANIFESTS = ROOT / "manifests"


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def env_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def env_float_list(name: str, default: str) -> list[float]:
    return [float(item) for item in env_list(name, default)]


def env_int_list(name: str, default: str) -> list[int]:
    return [int(item) for item in env_list(name, default)]


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    rows = list(rows)
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_manifest(name: str, payload: dict[str, Any]) -> None:
    ensure_dir(MANIFESTS)
    write_json(payload, MANIFESTS / f"{name}.json")

