#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
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
  echo "OPENROUTER_API_KEY is required. Save it to $OPENROUTER_ENV_FILE or export it." >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing venv python at $VENV/bin/python" >&2
  exit 2
fi

cd "$REPO"
export OPENROUTER_API_KEY
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/wolfbench-mpl}"
export WOLFBENCH_FINAL_OPENROUTER_RETRIES="${WOLFBENCH_FINAL_OPENROUTER_RETRIES:-3}"
export WOLFBENCH_FINAL_OPENROUTER_RETRY_SLEEP="${WOLFBENCH_FINAL_OPENROUTER_RETRY_SLEEP:-1.5}"
export WOLFBENCH_FINAL_OPENROUTER_REQUEST_PAUSE="${WOLFBENCH_FINAL_OPENROUTER_REQUEST_PAUSE:-0.02}"

PY="$VENV/bin/python"
TRACK="${TRACK:-all}"
SEEDS="${SEEDS:-1,2,3,4,5}"
PLAN_INTERVAL="${PLAN_INTERVAL:-5}"
COMMON=(--continue-on-error --quota-mode micro_full --plan-interval "$PLAN_INTERVAL")
if [[ "${APPEND_EXISTING:-0}" == "1" ]]; then
  COMMON+=(--append-existing)
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  COMMON+=(--skip-existing)
fi
if [[ -n "${ALPHA_GRID:-}" ]]; then
  COMMON+=(--alpha-grid "$ALPHA_GRID")
fi

validate_out() {
  local out_name="$1"
  "$PY" - "$out_name" <<'PY'
import csv
import sys
from pathlib import Path

out = sys.argv[1]
path = Path("final_hybrid_experiments/outputs") / out / "data.csv"
rows = list(csv.DictReader(path.open(newline="")))
errors = [row for row in rows if row.get("status", "ok") != "ok"]
print(f"{out}: rows={len(rows)} errors={len(errors)}")
if rows and len(errors) == len(rows):
    print(f"{out}: all rows failed; stopping batch", file=sys.stderr)
    raise SystemExit(3)
PY
}

"$PY" -m compileall -q final_hybrid_experiments

if [[ "$TRACK" == "s1" || "$TRACK" == "all" ]]; then
  "$PY" -m final_hybrid_experiments.scripts.run_e1_refined_scaling \
    --track s1 \
    --out real_e1_refined_s1 \
    --seeds "$SEEDS" \
    "${COMMON[@]}"
  validate_out real_e1_refined_s1
fi

if [[ "$TRACK" == "s2" || "$TRACK" == "all" ]]; then
  "$PY" -m final_hybrid_experiments.scripts.run_e1_refined_scaling \
    --track s2 \
    --out real_e1_refined_s2 \
    --seeds "$SEEDS" \
    "${COMMON[@]}"
  validate_out real_e1_refined_s2
fi

"$PY" -m final_hybrid_experiments.scripts.plot_e1_refined
