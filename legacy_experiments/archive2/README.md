# Final mixed-agent experiments

This folder contains code and outputs for the final paper-facing experiments.
Paper writing stays in `AuthorKit27`.

All final population experiments now run the bounded-rational network signaling
game by default. A bounded set of strategic population agents can use
OpenRouter JSON calls, while the scalable population mixes five plain-language
roles: Risk-Averse, Value-Oriented, Trend-Following, Social-Following, and
Aggressive. Agents have noisy private signals, finite attention, adaptive
source trust, and endogenous post/reshare/challenge choices. Mathematical
implementations such as QRE and satisficing remain internal mechanisms rather
than paper-facing role names. The implementation does not use vLLM.

See `EXPERIMENTS.md` for the authoritative experiment inventory and run order,
and `PAPER_STORY.md` for the v3 paper narrative. Outputs from earlier social
game versions are historical and must not be combined with v3 results.

Smoke test:

```bash
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_smoke
```

Real OpenRouter runs require `OPENROUTER_API_KEY`:

```bash
export OPENROUTER_API_KEY=...
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e1_scaling --preset paper
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e2_mechanisms --preset paper
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e3_social_dynamics --preset paper
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e4_quota_robustness --preset paper
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e5_wolfguard_benchmark --preset paper
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e7_behavioral_realism --preset paper
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e8_network_signal_game --preset paper
```

E7 directly addresses the homogeneous/rational-agent objection by comparing
the old weighted-score controller, the mixed-role population, and homogeneous
risk-averse, trend-following, social-following, and aggressive counterfactuals. E8 performs
causal social-mechanism ablations and also writes `agent_decisions.csv`,
`messages.csv`, and `exposures.csv` for information-theoretic audit.

Generate the independent mean-field predictions before looking at E8 outcomes,
then calculate paired mechanism contrasts after the run:

```bash
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.network_game_theory --out network_game_theory
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.analyze_e8_signal_game --out e8_network_signal_game
```

The propositions, proof sketches, cascade criterion, and falsifiable validation
plan are in `final_hybrid_experiments/theory/network_signal_game.md`.

Generate figures after a run:

```bash
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.plot_final_figures --out e1_scaling
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.plot_calibrated_figures --out e1_scaling
```

For paper-facing plots, prefer the calibrated figures. They report
scenario-aligned primary-risk scores and subtract the alpha=0 mixed-agent
baseline, which avoids treating clean S1/S2 social diffusion as attack-induced
failure. Raw binary failure-rate plots are still useful as appendix diagnostics.

For the main E1 threshold plot, run a denser S1 alpha sweep around the observed
transition region:

```bash
WOLFBENCH_FINAL_SEEDS=1,2,3,4,5,6,7 \
WOLFBENCH_FINAL_N_GRID=1000 \
WOLFBENCH_FINAL_ALPHAS_S1=0,0.02,0.04,0.05,0.055,0.06,0.065,0.07,0.075,0.085,0.10 \
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e1_scaling --out real_e1_scurve --preset pilot --continue-on-error

PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.plot_calibrated_figures --out real_e1_scurve
```

The wider S1 grid is intentional: after the v2 hold-cost/participation
calibration, the N=100 mixed-population pilot changes from 0/3 primary failures
at alpha=0.05 to 3/3 at alpha=0.075. Do not reuse threshold estimates from the
old weighted-score population. Re-pilot S2--S4 before committing their final
paper grids.

Useful controls:

- `WOLFBENCH_FINAL_MOCK=1` runs without network using a deterministic mock backend.
- `WOLFBENCH_FINAL_POPULATION_MODEL` changes the population LLM model.
- `WOLFBENCH_FINAL_N_GRID`, `WOLFBENCH_FINAL_SEEDS`, and `WOLFBENCH_FINAL_ALPHAS` override grids.
- `WOLFBENCH_FINAL_QUOTA_MODE=low|standard|high` changes the bounded LLM quota schedule.
- `WOLFBENCH_FINAL_DEFENSES` selects E5 defenses.

Use `--quota-mode behavioral_only` for a zero-API-cost policy-population run;
the network game and all information metrics remain active. Use `--mock` to
exercise the LLM wrapper path without external calls.

Outputs are written under `final_hybrid_experiments/outputs/<run_name>/`.

Within one episode, days must run sequentially because the market, social graph,
portfolio, and LLM observations depend on previous days. Independent
scenario/alpha/seed/defense episodes can be parallelized if each worker writes
separate partial outputs before merging and OpenRouter rate limits are respected.
