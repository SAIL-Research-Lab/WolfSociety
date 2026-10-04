#!/usr/bin/env bash
# Usage: scripts/run_pilot.sh MODEL MODEL_REVISION OUTPUT_ROOT [runner-plan flags...]
# Set EXECUTE=1 to run after the printed budget; default only creates a manifest.
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 MODEL MODEL_REVISION OUTPUT_ROOT [--families p01 ...] [--sizes 100,300,...]" >&2
  exit 2
fi
MODEL=$1
MODEL_REVISION=$2
OUTPUT_ROOT=$3
shift 3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
MANIFEST="${MANIFEST:-$OUTPUT_ROOT/pilot.manifest.json}"
"${PYTHON:-python}" -m paper_experiments.runner plan --stage pilot \
  --families p01 --sizes 100,300 --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --base-url "${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}" \
  --concurrency "${CONCURRENCY:-32}" --manifest "$MANIFEST" "$@"
if [[ "${EXECUTE:-0}" == 1 ]]; then
  "${PYTHON:-python}" -m paper_experiments.runner run --manifest "$MANIFEST" --output "$OUTPUT_ROOT" \
    --shard-index "${SHARD_INDEX:-0}" --num-shards "${NUM_SHARDS:-1}"
fi
