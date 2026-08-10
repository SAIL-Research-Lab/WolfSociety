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


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_jsonl(row: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str, allow_nan=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv_from_jsonl(jsonl_path: str | Path, csv_path: str | Path) -> None:
    jsonl_path = Path(jsonl_path)
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    fieldnames: list[str] = []
    seen: set[str] = set()
    if not jsonl_path.exists():
        write_csv([], csv_path)
        return
    with jsonl_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        with jsonl_path.open() as source:
            for line in source:
                if line.strip():
                    writer.writerow(json.loads(line))


def append_manifest(name: str, payload: dict[str, Any]) -> None:
    ensure_dir(MANIFESTS)
    write_json(payload, MANIFESTS / f"{name}.json")
