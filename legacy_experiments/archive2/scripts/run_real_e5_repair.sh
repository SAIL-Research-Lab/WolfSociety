#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OPENROUTER_ENV_FILE="${OPENROUTER_ENV_FILE:-$HOME/.wolfbench/openrouter.env}"
OPENROUTER_ENV_FALLBACK="${OPENROUTER_ENV_FALLBACK:-/private/tmp/wolfbench-secrets/openrouter.env}"

if [[ -z "${OPENROUTER_API_KEY:-}" && -f "$OPENROUTER_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$OPENROUTER_ENV_FILE"
fi
if [[ -z "${OPENROUTER_API_KEY:-}" && -f "$OPENROUTER_ENV_FALLBACK" ]]; then
  # shellcheck disable=SC1090
  source "$OPENROUTER_ENV_FALLBACK"
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required. Export it or save it to $OPENROUTER_ENV_FILE." >&2
  exit 1
fi
export OPENROUTER_API_KEY

PY="${PY:-.venv/bin/python}"
BASE="${BASE:-real_e5_wolfguard_benchmark}"
OUT="${OUT:-real_e5_wolfguard_benchmark_repaired}"
REQUESTED_DEFENSES="${DEFENSES:-zscore_guard,oracle,qwen235b_risk,glm45_risk,gpt41_risk,gemini25_pro_risk}"
PROBE_OUT="${PROBE_OUT:-final_hybrid_experiments/outputs/e5_openrouter_probe.csv}"
FILTER_BY_PROBE="${FILTER_BY_PROBE:-1}"
STRICT_PROBE="${STRICT_PROBE:-0}"

export WOLFBENCH_FINAL_OPENROUTER_RETRIES="${WOLFBENCH_FINAL_OPENROUTER_RETRIES:-3}"
export WOLFBENCH_FINAL_OPENROUTER_JSON_RETRIES="${WOLFBENCH_FINAL_OPENROUTER_JSON_RETRIES:-2}"
export WOLFBENCH_FINAL_OPENROUTER_RETRY_SLEEP="${WOLFBENCH_FINAL_OPENROUTER_RETRY_SLEEP:-1.5}"
export WOLFBENCH_FINAL_OPENROUTER_REQUEST_PAUSE="${WOLFBENCH_FINAL_OPENROUTER_REQUEST_PAUSE:-0.05}"

echo "[1/2] probing OpenRouter defense models"
if ! PYTHONPATH=src:. "$PY" -m final_hybrid_experiments.scripts.probe_e5_openrouter_models --out "$PROBE_OUT"; then
  if [[ "$STRICT_PROBE" == "1" ]]; then
    echo "Probe failed and STRICT_PROBE=1; stopping." >&2
    exit 1
  fi
  echo "Probe reported unavailable or schema-invalid models; continuing with available requested defenses." >&2
fi

DEFENSES="$REQUESTED_DEFENSES"
if [[ "$FILTER_BY_PROBE" == "1" ]]; then
  DEFENSES="$(PYTHONPATH=src:. "$PY" - "$REQUESTED_DEFENSES" "$PROBE_OUT" <<'PY'
import csv
import sys
from pathlib import Path

requested = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
probe_path = Path(sys.argv[2])
probe_aliases = {"qwen235b_risk", "glm45_risk", "gpt41_risk", "gemini25_pro_risk"}
ok_aliases = set()
if probe_path.exists():
    with probe_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                ok_aliases.add(row.get("alias", ""))

kept = [
    defense for defense in requested
    if defense not in probe_aliases or defense in ok_aliases
]
print(",".join(kept))
PY
)"
fi

if [[ -z "$DEFENSES" ]]; then
  echo "No repair defenses left after probe filtering." >&2
  exit 1
fi
echo "Repair defenses: $DEFENSES"

echo "[2/2] repairing E5 rows"
PYTHONPATH=src:. "$PY" -m final_hybrid_experiments.scripts.repair_e5_failed \
  --base "$BASE" \
  --out "$OUT" \
  --defenses "$DEFENSES"

echo "E5 repair complete: final_hybrid_experiments/outputs/$OUT"
