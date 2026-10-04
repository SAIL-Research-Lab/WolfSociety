#!/usr/bin/env bash
# Build the complete figure release from explicit, finished real-main runs.
# PAPER_DIR optionally installs the verified PDFs into a manuscript checkout.
set -euo pipefail
if [[ $# -lt 4 || ${1:-} == --help ]]; then
  echo "Usage: $0 MODEL IMMUTABLE_REVISION OUTPUT_DIR RUN_DIR [RUN_DIR ...]" >&2
  echo "Install the client with .[plot]. PAPER_DIR optionally enables verified manuscript installation." >&2
  exit 2
fi
MODEL=$1
MODEL_REVISION=$2
FIGURE_OUTPUT=$3
shift 3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python}" -m paper_experiments.figure_bundle build \
  --model "$MODEL" --model-revision "$MODEL_REVISION" --output "$FIGURE_OUTPUT" --runs "$@"
if [[ -n ${PAPER_DIR:-} ]]; then
  "${PYTHON:-python}" -m paper_experiments.manuscript_figures install \
    --paper-dir "$PAPER_DIR" --manifest "$FIGURE_OUTPUT/figure_manifest.json"
fi
