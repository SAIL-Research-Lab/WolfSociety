#!/usr/bin/env bash
# Launch one OpenAI-compatible endpoint. TP shards a replica; DP creates
# independent replicas. The required single-node GPU count is TP * DP.
# https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/
# https://docs.vllm.ai/en/stable/features/structured_outputs/
# https://docs.vllm.ai/en/stable/usage/reproducibility/
set -euo pipefail

if [[ ${1:-} == --help ]]; then
  cat <<'HELP'
Usage: MODEL=/path/or/hf-model bash scripts/serve_vllm.sh [extra vllm serve args]

Environment:
  MODEL                     Required local model path or Hugging Face model id
  MODEL_REVISION            Optional immutable weight commit; record for paper runs
  TOKENIZER_REVISION        Defaults to MODEL_REVISION when given
  SERVED_MODEL             API model identifier (defaults to MODEL)
  TP=1 DP=1                 Tensor/data parallel sizes; requires TP*DP GPUs
  CUDA_VISIBLE_DEVICES      Standard vLLM device selection, passed through
  HOST=127.0.0.1 PORT=8000   Serving endpoint
  MAX_MODEL_LEN=8192        Prompt plus output budget
  MAX_NUM_SEQS=256          Active sequences per DP rank, not population size
  GPU_MEMORY_UTILIZATION=.9 DTYPE=auto SERVER_SEED=0
  PREFIX_CACHING=1 CHUNKED_PREFILL=1
  VLLM_BATCH_INVARIANT=1    Scheduling invariance (supported models/hardware only)
  STRUCTURED_OUTPUT_BACKEND=xgrammar
  SERVER_MANIFEST           Optional JSON provenance output path
  PYTHON_BIN=python3 VLLM_BIN=vllm

Install/pin vLLM >=0.12 in the GPU environment. Older guided_json is unsupported.
Extra CLI arguments are passed verbatim; do not duplicate the managed flags.
HELP
  exit 0
fi

: "${MODEL:?Set MODEL to the model path or Hugging Face model id}"
PYTHON_BIN=${PYTHON_BIN:-python3}
VLLM_BIN=${VLLM_BIN:-vllm}
TP=${TP:-1}
DP=${DP:-1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
SERVED_MODEL=${SERVED_MODEL:-$MODEL}
export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}

if ! command -v "$VLLM_BIN" >/dev/null; then
  printf 'vLLM executable not found: %s\n' "$VLLM_BIN" >&2
  exit 1
fi
for parallel in "$TP" "$DP"; do
  if [[ ! $parallel =~ ^[1-9][0-9]*$ ]]; then
    printf 'TP and DP must be positive integers.\n' >&2
    exit 1
  fi
done

cmd=("$VLLM_BIN" serve "$MODEL"
  --served-model-name "$SERVED_MODEL"
  --host "$HOST" --port "$PORT"
  --tensor-parallel-size "$TP" --data-parallel-size "$DP"
  --max-model-len "${MAX_MODEL_LEN:-8192}"
  --max-num-seqs "${MAX_NUM_SEQS:-256}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
  --dtype "${DTYPE:-auto}" --seed "${SERVER_SEED:-0}"
  --generation-config vllm
  --enable-request-id-headers
  --structured-outputs-config.backend "${STRUCTURED_OUTPUT_BACKEND:-xgrammar}")

if [[ -n ${MODEL_REVISION:-} ]]; then
  cmd+=(--revision "$MODEL_REVISION" --tokenizer-revision "${TOKENIZER_REVISION:-$MODEL_REVISION}")
elif [[ -n ${TOKENIZER_REVISION:-} ]]; then
  cmd+=(--tokenizer-revision "$TOKENIZER_REVISION")
fi
if [[ ${PREFIX_CACHING:-1} == 1 ]]; then
  cmd+=(--enable-prefix-caching)
else
  cmd+=(--no-enable-prefix-caching)
fi
if [[ ${CHUNKED_PREFILL:-1} == 1 ]]; then
  cmd+=(--enable-chunked-prefill)
else
  cmd+=(--no-enable-chunked-prefill)
fi
cmd+=("$@")

# Read version from the same Python environment in which vLLM was installed.
# Save exact argv without secrets. API authentication uses VLLM_API_KEY directly.
"$PYTHON_BIN" - "${SERVER_MANIFEST:-}" "${cmd[@]}" <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
from datetime import datetime, timezone

version = importlib.metadata.version("vllm")
match = re.match(r"(\d+)\.(\d+)", version)
if not match or tuple(map(int, match.groups())) < (0, 12):
    raise SystemExit(f"vLLM >=0.12 required for structured_outputs; installed {version}")
argv = sys.argv[2:]
redacted = []
hide_next = False
for token in argv:
    if hide_next:
        redacted.append("[REDACTED]")
        hide_next = False
    elif token == "--api-key":
        redacted.append(token)
        hide_next = True
    elif token.startswith("--api-key="):
        redacted.append("--api-key=[REDACTED]")
    else:
        redacted.append(token)
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "vllm_version": version, "python": platform.python_version(),
    "argv": redacted,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
    "host_platform": platform.platform(),
    "weight_revision": os.environ.get("MODEL_REVISION"),
    "tokenizer_revision": os.environ.get("TOKENIZER_REVISION", os.environ.get("MODEL_REVISION")),
    "note": "Archive GPU/driver inventory and model/tokenizer contents with the experiment."
}
if sys.argv[1]:
    path = Path(sys.argv[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(manifest, ensure_ascii=False), file=sys.stderr)
PY

exec "${cmd[@]}"
