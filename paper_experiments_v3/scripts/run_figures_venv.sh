#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Project virtual environment not found at .venv/bin/python" >&2
  echo "Create it with: python3 -m venv .venv && .venv/bin/python -m pip install -e '.[plot]'" >&2
  exit 1
fi

export MPLCONFIGDIR="${TMPDIR:-/tmp}/wolfbench_mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/wolfbench_xdg_cache"

.venv/bin/python -m paper_experiments_v3.figures.make_paper_figures "$@"
