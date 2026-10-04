#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO"

export TRACK="${TRACK:-s1}"
export APPEND_EXISTING="${APPEND_EXISTING:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export SEEDS="${SEEDS:-1,2,3,4,5,6,7,8,9,10,11,12}"
export WOLFBENCH_E1_REFINED_N="${WOLFBENCH_E1_REFINED_N:-200,300,500,1000,2000}"

bash final_hybrid_experiments/scripts/run_real_e1_refined.sh
