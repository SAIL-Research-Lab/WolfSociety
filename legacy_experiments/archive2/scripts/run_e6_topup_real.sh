#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the real E6 controller-ablation top-up.

This appends missing rows into:
  final_hybrid_experiments/outputs/real_e6_controller_ablation

It uses seeds 1-5, skips completed ok rows, and covers the refined E6 grid:
  N=300:  0,.002,.004,.006,.008,.010,.0125,.015,.0175,.020,.025
  N=1000: 0,.004,.006,.008,.010,.012,.014,.016,.018,.020

The run is chunked by controller variant and N so each completed chunk is
written to disk before the next chunk starts.
EOF
  exit 0
fi

if [[ -z "${OPENROUTER_API_KEY:-}" && -f /private/tmp/wolfbench-secrets/openrouter.env ]]; then
  # shellcheck disable=SC1091
  source /private/tmp/wolfbench-secrets/openrouter.env
  export OPENROUTER_API_KEY
fi

for controller in behavioral_only mixed_default higher_llm_quota; do
  for n_value in 300 1000; do
    PYTHONPATH=src:. .venv/bin/python -m final_hybrid_experiments.scripts.run_e6_controller_ablation \
      --out real_e6_controller_ablation \
      --seeds 1,2,3,4,5 \
      --n-values "${n_value}" \
      --controller-variants "${controller}" \
      --continue-on-error \
      --append-existing \
      --skip-existing
  done
done

PYTHONPATH=src:. .venv/bin/python -m final_hybrid_experiments.scripts.plot_e6_controller_ablation \
  --out real_e6_controller_ablation
