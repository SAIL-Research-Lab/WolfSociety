#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run E6b: matched absolute LLM quota.

This experiment holds the number of LLM-controlled agents fixed across N:
  fixed_b4_h1 = up to 4 benign LLM agents and 1 harmful LLM agent.

It writes to:
  final_hybrid_experiments/outputs/real_e6_matched_quota

The goal is to test finite-size scaling while removing the default N-dependent
LLM quota as a confound.
EOF
  exit 0
fi

if [[ -z "${OPENROUTER_API_KEY:-}" && -f /private/tmp/wolfbench-secrets/openrouter.env ]]; then
  # shellcheck disable=SC1091
  source /private/tmp/wolfbench-secrets/openrouter.env
  export OPENROUTER_API_KEY
fi

for n_value in 300 1000; do
  PYTHONPATH=src:. .venv/bin/python -m final_hybrid_experiments.scripts.run_e6_controller_ablation \
    --out real_e6_matched_quota \
    --seeds 1,2,3,4,5 \
    --n-values "${n_value}" \
    --controller-variants fixed_b4_h1 \
    --continue-on-error \
    --append-existing \
    --skip-existing
done

PYTHONPATH=src:. .venv/bin/python -m final_hybrid_experiments.scripts.plot_e6_controller_ablation \
  --out real_e6_matched_quota
