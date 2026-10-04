#!/usr/bin/env bash
# Usage: scripts/run_full_llm_paper.sh MODEL MODEL_REVISION FROZEN_GRID OUTPUT_ROOT [runner-plan flags...]
# Execute=1 enables inference after an immutable manifest and budget have been written.
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "Usage: $0 MODEL MODEL_REVISION FROZEN_GRID OUTPUT_ROOT [--families all] [--seeds 1-12]" >&2
  exit 2
fi
MODEL=$1
MODEL_REVISION=$2
FROZEN_GRID=$3
OUTPUT_ROOT=$4
shift 4
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
MANIFEST="${MANIFEST:-$OUTPUT_ROOT/main.manifest.json}"
"${PYTHON:-python}" -m paper_experiments.runner plan --stage main \
  --families all --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --base-url "${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}" \
  --concurrency "${CONCURRENCY:-32}" --grid "$FROZEN_GRID" --manifest "$MANIFEST" "$@"
if [[ "${EXECUTE:-0}" == 1 ]]; then
  "${PYTHON:-python}" -m paper_experiments.runner run --manifest "$MANIFEST" --output "$OUTPUT_ROOT" \
    --shard-index "${SHARD_INDEX:-0}" --num-shards "${NUM_SHARDS:-1}"
fi
