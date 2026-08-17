# Paper figure builders

This directory builds the main-paper figures for the v3 rewrite.

Run after the relevant paper-profile experiments and analyses finish:

```bash
cd Harm-Scale
./paper_experiments_v3/scripts/run_figures_venv.sh
```

The wrapper intentionally uses the project virtual environment at `.venv`,
instead of the active conda/base Python.

Outputs are written to:

```text
paper_experiments_v3/figures/generated/
```

The script writes both `.pdf` and `.png` versions.

It uses seaborn if the local Python environment can import it. If seaborn is
broken or unavailable, it falls back to Matplotlib's bundled `seaborn-v0_8`
style with the same colorblind-safe palette and paper rcParams.

## Figure mapping

| Output | Paper figure | Inputs |
|---|---|---|
| `fig1_overview` | Figure 1: overview schematic | no experiment data |
| `fig2_p01_main_results` | Figure 2: nonlinear collapse and scaling | P01 shards |
| `fig3_p02_decomposition` | Figure 3: scaling decomposition | P02 shards + P01 scaling |
| `fig4_mechanism` | Figure 4: feedback-information mechanism | P04 and P05 shards |
| `fig_p04_feedback_available` | Current polished P04-only mechanism figure | P04 shards |
| `table2_scaling_results.csv` | main scaling result table | P01 shards |

## Important interpretation rule

Figure 3 currently uses proxy quantities:

- `attack_magnitude_proxy = K / N^liquidity_exponent`
- `failure_gain_proxy = primary_failure_score_max / attack_magnitude_proxy`

Do not label these as exact \(h_N\) or exact \(\chi_{T,N}\) unless the simulator
writes direct perturbation-response measurements. The paper text should keep the
word "proxy" for these panels.

## Completeness

The script filters to rows with `status == ok`. If P03/P04/P05 still contain
error rows, generated figures are diagnostics, not final paper figures.
