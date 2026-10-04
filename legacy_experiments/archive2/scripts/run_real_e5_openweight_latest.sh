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
OUT="${OUT:-real_e5_openweight_latest}"
PROBE_OUT="${PROBE_OUT:-final_hybrid_experiments/outputs/e5_openweight_latest_probe.csv}"
# DeepSeek V4 Pro currently returns empty JSON on the public risk-scoring schema
# probe, so the official open-weight track uses the latest DeepSeek row that
# passes the interface contract.
OPENWEIGHT_DEFENSES="deepseek_v3_risk,qwen36_35b_risk,glm52_risk,llama4_maverick_risk"
REQUESTED_DEFENSES="${DEFENSES:-noguard,rule,zscore_guard,topology_aware,oracle,$OPENWEIGHT_DEFENSES}"

export WOLFBENCH_FINAL_OPENROUTER_RETRIES="${WOLFBENCH_FINAL_OPENROUTER_RETRIES:-3}"
export WOLFBENCH_FINAL_OPENROUTER_JSON_RETRIES="${WOLFBENCH_FINAL_OPENROUTER_JSON_RETRIES:-2}"
export WOLFBENCH_FINAL_OPENROUTER_RETRY_SLEEP="${WOLFBENCH_FINAL_OPENROUTER_RETRY_SLEEP:-1.5}"
export WOLFBENCH_FINAL_OPENROUTER_REQUEST_PAUSE="${WOLFBENCH_FINAL_OPENROUTER_REQUEST_PAUSE:-0.05}"
export WOLFBENCH_LLM_SEMANTIC_RETRIES="${WOLFBENCH_LLM_SEMANTIC_RETRIES:-2}"

export WOLFBENCH_FINAL_DEFENSES="$REQUESTED_DEFENSES"
export WOLFBENCH_FINAL_N_GRID="${WOLFBENCH_FINAL_N_GRID:-1000}"
export WOLFBENCH_FINAL_SEEDS="${WOLFBENCH_FINAL_SEEDS:-1,2}"
export WOLFBENCH_FINAL_ALPHAS_S1="${WOLFBENCH_FINAL_ALPHAS_S1:-0,0.01,0.015,0.02,0.03}"
export WOLFBENCH_FINAL_ALPHAS_S2="${WOLFBENCH_FINAL_ALPHAS_S2:-0,0.001,0.0015,0.002,0.003}"
export WOLFBENCH_FINAL_ALPHAS_S3="${WOLFBENCH_FINAL_ALPHAS_S3:-0,0.05,0.075,0.1}"
export WOLFBENCH_FINAL_ALPHAS_S4="${WOLFBENCH_FINAL_ALPHAS_S4:-0,0.02,0.03,0.05}"

echo "[1/3] probing latest open-weight OpenRouter defense models"
if [[ "${SKIP_PROBE:-0}" == "1" ]]; then
  echo "Skipping probe because SKIP_PROBE=1"
else
  PYTHONPATH=src:. "$PY" -m final_hybrid_experiments.scripts.probe_e5_openrouter_models \
    --aliases "$OPENWEIGHT_DEFENSES" \
    --cache-root /private/tmp/wolfbench-openweight-latest-probe \
    --out "$PROBE_OUT"
fi

echo "[2/3] running E5 latest open-weight leaderboard"
PYTHONPATH=src:. "$PY" -m final_hybrid_experiments.scripts.run_e5_wolfguard_benchmark \
  --preset pilot \
  --out "$OUT" \
  --continue-on-error

echo "[3/3] building E5 leaderboard"
PYTHONPATH=src:. "$PY" -m final_hybrid_experiments.scripts.build_e5_leaderboard \
  --out "$OUT"

echo "E5 latest open-weight complete: final_hybrid_experiments/outputs/$OUT"
