# WolfBench paper experiments v3

This package contains the current experiment runners, shared runtime,
post-processing analyses, and theory utilities for the WolfBench study. It
imports the simulator from `src/wolfbench/` and does not depend on archived
code or manuscript sources.

Generated rows, caches, tables, and figures are intentionally excluded from
version control. Each run writes a frozen configuration beside its local
artifacts so results can be audited without committing those artifacts.

## Requirements

Install WolfBench from the repository root:

```bash
python -m pip install -e ".[dev]"
```

For real OpenRouter experiments, also install the LLM extra and provide the
credential through the environment:

```bash
python -m pip install -e ".[dev,llm]"
export OPENROUTER_API_KEY=your-key
```

Never place a real credential in tracked configuration or source files.

## Validation first

Run the integration and clean-state sanity check with the deterministic mock
backend:

```bash
PYTHONPATH=src:. python -m paper_experiments_v3.experiments.p00_validate \
  --profile smoke \
  --mock \
  --quota-mode standard
```

The command covers S1--S4 and writes local artifacts under
`paper_experiments_v3/outputs/p00_validate/`.

## Experiment profiles

All runners accept the shared arguments below:

- `--profile smoke`: minimal integration run.
- `--profile pilot`: protocol and grid validation.
- `--profile paper`: full configured run.
- `--mock`: deterministic local backend; no external API call.
- `--model`: override the OpenRouter-compatible model identifier.
- `--quota-mode`: choose the configured strategic-agent allocation.
- `--out`: select a local output directory name.
- `--continue-on-error`: record failed cells instead of stopping immediately.

Use smoke and pilot profiles to validate a grid before starting any expensive
run. A real run without `--mock` fails closed when no API credential is
available.

## Core runners

```bash
PYTHONPATH=src:. python -m paper_experiments_v3.experiments.p01_nonlinear_scaling --profile smoke --mock
PYTHONPATH=src:. python -m paper_experiments_v3.experiments.p02_size_decomposition --profile smoke --mock
PYTHONPATH=src:. python -m paper_experiments_v3.experiments.p03_cross_scenario --profile smoke --mock
PYTHONPATH=src:. python -m paper_experiments_v3.experiments.p04_game_phase --profile smoke --mock
PYTHONPATH=src:. python -m paper_experiments_v3.experiments.p05_information_cascade --profile smoke --mock
PYTHONPATH=src:. python -m paper_experiments_v3.experiments.p06_role_robustness --profile smoke --mock
```

P07 and P08 are optional robustness/utility studies. P09--P11 are targeted
audit runners. See `EXPERIMENTS.md` for their scientific roles and reporting
constraints.

## Analyses

Analyses read completed local run artifacts and write derived files back into
ignored output or generated-figure directories:

```bash
PYTHONPATH=src:. python -m paper_experiments_v3.analysis.scaling --run p01_nonlinear_scaling
PYTHONPATH=src:. python -m paper_experiments_v3.analysis.p02_decomposition \
  --run p02_size_decomposition \
  --p01-run p01_nonlinear_scaling
PYTHONPATH=src:. python -m paper_experiments_v3.analysis.game_phase --run p04_game_phase
PYTHONPATH=src:. python -m paper_experiments_v3.analysis.information --run p05_information_cascade
PYTHONPATH=src:. python -m paper_experiments_v3.analysis.roles --run p06_role_robustness
```

Analysis requires the corresponding local outputs. The repository publishes
the code and frozen protocol, not precomputed evidence.

## Reproducibility contract

Every completed run should retain locally:

- the frozen protocol and benchmark version;
- one episode-level row per scenario, size, alpha, seed, and variant;
- grouped descriptive summaries;
- success and error counts;
- model/backend usage metadata when an external model is used.

Treat the episode seed as the independent unit. Do not combine benchmark
versions, and do not report an estimated midpoint unless the evaluated grid
brackets the target probability.