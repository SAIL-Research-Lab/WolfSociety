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
  echo "OPENROUTER_API_KEY is required. Save it to /private/tmp/wolfbench-openrouter.env or export it." >&2
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
COMMON="--preset pilot --continue-on-error"

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

"$PY" - <<'PY'
import json
import os
import urllib.request

api_key = os.environ.get("OPENROUTER_API_KEY", "")
if not api_key:
    raise SystemExit("OPENROUTER_API_KEY is empty after env loading.")
request = urllib.request.Request(
    "https://openrouter.ai/api/v1/models",
    method="GET",
    headers={"Authorization": f"Bearer {api_key}"},
)
try:
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception as exc:
    raise SystemExit(f"OpenRouter network preflight failed: {type(exc).__name__}: {exc}") from exc
if "data" not in payload:
    raise SystemExit("OpenRouter network preflight returned an unexpected payload.")
print(f"OpenRouter network preflight ok: {len(payload.get('data', []))} models visible")
PY

WOLFBENCH_FINAL_SEEDS=1,2,3 \
WOLFBENCH_FINAL_N_GRID=100,300,1000 \
WOLFBENCH_FINAL_ALPHAS_S1=0,0.01,0.015,0.02,0.03 \
"$PY" -m final_hybrid_experiments.scripts.run_e1_scaling --out real_e1_scaling $COMMON
validate_out real_e1_scaling

WOLFBENCH_FINAL_SEEDS=1,2,3 \
WOLFBENCH_FINAL_N_GRID=300 \
WOLFBENCH_FINAL_ALPHAS_S1=0,0.01,0.015,0.02,0.03 \
WOLFBENCH_FINAL_ALPHAS_S2=0,0.003,0.005,0.0075,0.01 \
WOLFBENCH_FINAL_ALPHAS_S3=0,0.05,0.075,0.1 \
WOLFBENCH_FINAL_ALPHAS_S4=0,0.02,0.03,0.05 \
"$PY" -m final_hybrid_experiments.scripts.run_e2_mechanisms --out real_e2_mechanisms $COMMON
validate_out real_e2_mechanisms

WOLFBENCH_FINAL_SEEDS=1,2,3 \
WOLFBENCH_FINAL_N_GRID=300 \
WOLFBENCH_FINAL_ALPHAS_S1=0,0.01,0.015,0.02,0.03 \
"$PY" -m final_hybrid_experiments.scripts.run_e3_social_dynamics --out real_e3_social_dynamics $COMMON
validate_out real_e3_social_dynamics

WOLFBENCH_FINAL_SEEDS=1,2,3 \
WOLFBENCH_FINAL_N_GRID=300 \
WOLFBENCH_FINAL_ALPHAS_S1=0,0.01,0.015,0.02,0.03 \
WOLFBENCH_FINAL_ALPHAS_S2=0,0.003,0.005,0.0075,0.01 \
WOLFBENCH_FINAL_ALPHAS_S3=0,0.05,0.075,0.1 \
WOLFBENCH_FINAL_ALPHAS_S4=0,0.02,0.03,0.05 \
"$PY" -m final_hybrid_experiments.scripts.run_e4_quota_robustness --out real_e4_quota_robustness $COMMON
validate_out real_e4_quota_robustness

WOLFBENCH_FINAL_SEEDS=1,2 \
WOLFBENCH_FINAL_N_GRID=1000 \
WOLFBENCH_FINAL_DEFENSES=noguard,zscore_guard,topology_aware,oracle,deepseek_v3_risk,qwen235b_risk,glm45_risk,llama33_70b_risk,gpt41_risk,gemini25_pro_risk \
WOLFBENCH_FINAL_ALPHAS_S1=0,0.01,0.015,0.02,0.03 \
WOLFBENCH_FINAL_ALPHAS_S2=0,0.001,0.0015,0.002,0.003 \
WOLFBENCH_FINAL_ALPHAS_S3=0,0.05,0.075,0.1 \
WOLFBENCH_FINAL_ALPHAS_S4=0,0.02,0.03,0.05 \
"$PY" -m final_hybrid_experiments.scripts.run_e5_wolfguard_benchmark --out real_e5_wolfguard_benchmark $COMMON
validate_out real_e5_wolfguard_benchmark

"$PY" -m final_hybrid_experiments.scripts.plot_final_figures \
  --out real_e1_scaling \
  --out real_e2_mechanisms \
  --out real_e3_social_dynamics \
  --out real_e4_quota_robustness \
  --out real_e5_wolfguard_benchmark

"$PY" -m final_hybrid_experiments.scripts.build_e5_leaderboard --out real_e5_wolfguard_benchmark
