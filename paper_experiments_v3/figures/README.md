# Original paper figure code

The current full-LLM exporter **reuses the plotting functions in
`make_paper_figures.py`**, including their publication layout, colors, markers
and figure types. `paper_experiments/figure_rendering.py` only adapts verified
new episode fields, exports source data and composes the original teaser.

## Current results

Follow `SERVER_RUNBOOK.md`, section 6:

```bash
scripts/build_paper_figures.sh MODEL IMMUTABLE_REVISION OUTPUT_DIR RUN_DIR [RUN_DIR ...]
```

This entry point rejects old/mock/pilot/incomplete data before calling the
original plot functions. It never calls `read_rows` or `read_csv` here. The old
hard-coded threshold annotation and axis windows now follow the supplied data.
Scaling captions no longer assume that the new result must decrease/subscale.
The active P04 names use the same grouped forest; high deliberation is labeled
as such instead of pretending it is numerical QRE precision.

| Current output | Original plotting function | New input |
|---|---|---|
| `teaser` | Original artwork + the same Fig. 2/3 artists | P01 |
| `fig2_nonlinear_response` | `figure2_p01_nonlinear_response` | P01 failure probability and severity |
| `fig3_finite_size_scaling` | `figure3_p01_finite_size_scaling` | P01 boundaries, counts, power fits and seed bootstrap |
| `fig4_intervention_effects` | `figure4_intervention_effects` | P04 matched-size shifts and paired-seed bootstrap |

`assets/teaser_layout.pdf` preserves the final manuscript's original vector
artwork. All old empirical paths/text in its A/B plot regions were removed by
PDF redaction. `assets/teaser_layout.json` records that preparation. Current
curves are inserted on every build; the original result PDF is never an input.
The static mechanism panel is a conceptual illustration, not a new estimate of
its decomposition exponents. New captions must distinguish that illustration
from the empirical panels.

## Explicit historical reproduction

```bash
./paper_experiments_v3/scripts/run_figures_venv.sh --allow-historical-data
```

Only this opt-in CLI reads v3 CSVs under `paper_experiments_v3/outputs` and writes
historical diagnostics to `paper_experiments_v3/figures/generated`. It is not the
current paper release path. The wrapper uses the project's `.venv`.

The non-primary P02/feedback plotting functions remain available for historical
inspection. Their proxy quantities are not exact causal response measurements.
See `legacy_experiments/final_paper_experiment_catalog.json` for those historical
experiment dependencies.
