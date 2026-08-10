# WolfBench

WolfBench is a research prototype for studying harmful-agent scaling and
defense policies in controlled financial agent societies. It provides a
reproducible simulator, public scenario configurations, defense interfaces,
finite-size evaluation utilities, and a separate set of paper experiment
runners.

The repository is intentionally code-first. Generated experiment rows, model
artifacts, figures, and manuscript sources are not versioned.

## Scope

WolfBench supports three connected tasks:

- **Scaling analysis:** vary the harmful-agent fraction and society size under
  a fixed scenario protocol.
- **Defense evaluation:** compare policies through the same closed-loop
  environment and public observation interface.
- **Mechanism studies:** test how network reach, shared-state feedback,
  information flow, and behavioral heterogeneity affect collective outcomes.

This is a research scaffold rather than a production market model. Scenarios
are case-inspired controlled environments, and empirical conclusions should be
reported only for the protocol and parameter range that were evaluated.

## Installation

WolfBench requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the optional hosted/local LLM and plotting dependencies when needed:

```bash
python -m pip install -e ".[dev,llm,plot]"
```

## Quickstart

List the public scenarios:

```bash
wolfbench scenarios
```

Run a deterministic episode:

```bash
wolfbench run --scenario s1 --alpha 0.02 --n-society 200 --seed 1
```

Run a small scaling sweep:

```bash
wolfbench scaling \
  --scenario s1 \
  --alpha 0,0.02,0.05 \
  --n-society 100,200 \
  --seeds 2
```

Evaluate a built-in defense on an explicit small seed set:

```bash
wolfbench evaluate \
  --defense rule \
  --scenario s1 \
  --alphas 0,0.02,0.05 \
  --n-society 200 \
  --seeds 1,2
```

Use `--out PATH` on commands that support it when a machine-readable local
result is needed. Generated outputs are ignored by Git.

## Defense interface

A defense receives only the public daily summary and returns one intervention
per asset. Evaluator-owned validation and costs are applied after the policy
returns.

```python
from wolfbench.agents.wolfguard import WolfGuardConfig


class MyDefense:
    name = "MyDefense"
    config = WolfGuardConfig()

    def fit_baseline(self, baseline: dict) -> None:
        self.baseline = baseline

    def decide(self, day: int, summary: dict) -> dict[str, dict]:
        return {}
```

Load a custom policy with a dotted class path:

```bash
wolfbench evaluate \
  --defense my_package.policies:MyDefense \
  --scenario s1 \
  --split public_test
```

Built-in baselines include no-defense and random controls, rule-based and
topology-aware policies, a simulator-trained distilled policy, an oracle upper
bound, and optional OpenAI-compatible LLM policies. The oracle is not an
eligible public-observation defense.

## Scenarios

| ID | Controlled scenario | Primary mechanism |
| --- | --- | --- |
| S0 | Clean market | calibration/control |
| S1 | Social pump-and-dump | promotion, diffusion, coordinated exit |
| S2 | Finfluencer scalping | central placement and copy-trading |
| S3 | Spoofing/layering | displayed depth and cancellation |
| S4 | Wash trading/fake liquidity | manufactured volume signals and withdrawal |

Scenario YAML files and public split definitions live under
`src/wolfbench/config/` and are included in the installed package.

## Paper experiment runners

The current reproducibility code lives under `paper_experiments_v3/`. It is
separate from the installable `wolfbench` package but imports the same simulator
implementation.

Run the four-scenario integration check without an external model:

```bash
PYTHONPATH=src:. python -m paper_experiments_v3.experiments.p00_validate \
  --profile smoke \
  --mock \
  --quota-mode standard
```

Run a small nonlinear-scaling pilot with the deterministic mock backend:

```bash
PYTHONPATH=src:. python -m paper_experiments_v3.experiments.p01_nonlinear_scaling \
  --profile smoke \
  --mock
```

See `paper_experiments_v3/README.md` and
`paper_experiments_v3/EXPERIMENTS.md` for the experiment registry and analysis
entry points. Runners write local artifacts under
`paper_experiments_v3/outputs/`; those artifacts, generated tables, and figures
are excluded from version control.

Real OpenRouter runs require the optional LLM dependencies and an
`OPENROUTER_API_KEY` environment variable. Copy `.env.example` only as a local
template; never commit a populated environment file.

## Tests

Run the public Python test suite from the repository root:

```bash
pytest -q
```

The suite covers end-to-end scenarios, defense metrics, public-observation
isolation, optional LLM wrappers with deterministic test doubles, trajectory
export, the distilled baseline, and the paper-v3 social-dynamics utilities.

## Project website

The academic project page is a React/Vite application under `website/`.

```bash
cd website
npm ci
npm run check
npm run dev
```

The browser animations are conceptual illustrations, not online simulator
runs. The website keeps a small set of paper-facing summary values in source;
raw experimental CSV files and generated figures are not bundled.

Pushes that change the website run its checks, production build, and GitHub
Pages deployment after Pages is configured to use GitHub Actions.

## Repository layout

```text
src/wolfbench/                 installable simulator and CLI
tests/                         public regression and integration tests
paper_experiments_v3/
  experiments/                experiment runners
  runtime/                    shared execution and artifact contract
  analysis/                   post-run analyses
  theory/                     analytical models and derivations
  configs/protocol.yaml       smoke, pilot, and paper profiles
  scripts/                    serial execution helpers
docs/                          public theory and calibration notes
website/                       academic project website
.github/workflows/             Python CI and website deployment
```

## Reproducibility and generated artifacts

- Keep scenario, seed, population-size, harmful-fraction, model, and quota
  settings with every local run.
- Treat the episode/seed as the independent experimental unit.
- Do not report a critical midpoint when the evaluated grid does not bracket
  the target probability.
- Keep conceptual animations, theoretical assumptions, fitted quantities, and
  measured results visually and textually distinct.
- Generated outputs, caches, datasets, figures, model weights, and paper files
  are intentionally excluded from this repository.

## License

WolfBench is released under the MIT License. See `LICENSE`.