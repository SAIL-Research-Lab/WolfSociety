#!/usr/bin/env bash
# Replay actual per-agent request JSONL against the real endpoint. Synthetic
# random-token benchmarks do not measure this simulator's grammar/prompt costs.
set -euo pipefail

if [[ ${1:-} == --help ]]; then
  cat <<'HELP'
Usage: MODEL=served-id REQUESTS_JSONL=episode/llm_audit.jsonl \
         bash scripts/benchmark_vllm.sh [benchmark CLI args]

Input lines contain request_id, system, user, seed and optionally schema.
The LLM runtime audit JSONL is accepted directly. Each unique agent prompt is
replayed separately; fewer than NUM_AGENTS source requests is an error.
Export one full round with 1000 or 2000 agents from a dry run first.

Environment: BASE_URL=http://127.0.0.1:8000/v1 NUM_AGENTS=1000 ROUNDS=3
  CONCURRENCY=64 REQUEST_TIMEOUT=120 RETRIES=2 MAX_TOKENS=512 TEMPERATURE=.7
  SCHEMA_JSON=path.json       Optional if every input audit row embeds schema
  BENCH_OUTPUT=output/vllm_benchmark
  MODEL_REVISION             Optional declared revision matching server manifest
  CHAT_TEMPLATE_KWARGS_JSON  Optional JSON, e.g. {"enable_thinking":false}
  VLLM_API_KEY               Optional endpoint API key; never written to audit
  PYTHON_BIN=python3

Benchmark sends genuine generation requests. It has no mock/fallback mode.
Reports round time, requests/s, reported tokens/s and latency percentiles;
records every request and error. This measures functional throughput, not
agreement with a rule baseline. Extra CLI arguments override defaults below.
HELP
  exit 0
fi

: "${MODEL:?Set MODEL to the served model identifier}"
: "${REQUESTS_JSONL:?Set REQUESTS_JSONL to actual per-agent request/audit JSONL}"
PYTHON_BIN=${PYTHON_BIN:-python3}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

args=(--requests-jsonl "$REQUESTS_JSONL" --model "$MODEL"
  --base-url "${BASE_URL:-http://127.0.0.1:8000/v1}"
  --num-requests "${NUM_AGENTS:-1000}" --rounds "${ROUNDS:-3}"
  --concurrency "${CONCURRENCY:-64}"
  --request-timeout "${REQUEST_TIMEOUT:-120}" --retries "${RETRIES:-2}"
  --max-tokens "${MAX_TOKENS:-512}" --temperature "${TEMPERATURE:-0.7}"
  --output "${BENCH_OUTPUT:-$PROJECT_DIR/output/vllm_benchmark}")
if [[ -n ${SCHEMA_JSON:-} ]]; then
  args+=(--schema "$SCHEMA_JSON")
fi
if [[ -n ${MODEL_REVISION:-} ]]; then
  args+=(--model-revision "$MODEL_REVISION")
fi
if [[ -n ${CHAT_TEMPLATE_KWARGS_JSON:-} ]]; then
  args+=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_JSON")
fi
exec "$PYTHON_BIN" -m wolfbench.llm_runtime.backend "${args[@]}" "$@"
