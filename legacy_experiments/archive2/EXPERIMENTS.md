# WolfBench final experiments ledger

This file is the source of truth for paper-facing experiments. Update it when a
runner, grid, metric, model version, or paper claim changes.

## Current benchmark version

- Active environment: `network_signal_game_v3`.
- Active experiment root: `final_hybrid_experiments/` only.
- All outputs produced before v3 are historical calibration artifacts. They
  must not be mixed with v3 rows or quoted as final paper results.
- v3 paper-facing retail roles are Risk-Averse, Value-Oriented,
  Trend-Following, Social-Following, and Aggressive. QRE, satisficing, and
  heuristics are implementation details, not role names.

## The three paper claims

| Claim | Question | Required evidence |
|---|---|---|
| C1: nonlinear collapse | Does failure rise sharply rather than linearly with harmful fraction? | Dense alpha response curves, continuous-risk check, logistic-vs-linear comparison, repeated seeds |
| C2: harmful-agent scaling | Does the critical harmful fraction fall as society size grows? | N-specific thresholds, bootstrap bands, critical harmful counts, fixed-K and liquidity/social decompositions |
| C3: social mechanism | When and why does social information override private information? | Mean-field coupling test, causal social interventions, conditional MI, transfer entropy, conflict-following cascade criterion |

The paper should present C1 and C2 as the population-level phenomenon and C3
as the mechanism explaining when the phenomenon is amplified. Defense results
demonstrate benchmark utility but are not a fourth central scientific claim.

## Canonical paper experiment blocks

Although code retains E1--E8 names for reproducibility, the paper should group
them into five blocks:

1. **Collapse and size scaling:** E1a, E1b, E1c.
2. **Cross-scenario scope:** E2.
3. **Game/information-theoretic social mechanism:** E3a, E3b, and code-E8.
4. **Agent and implementation robustness:** E4, E6, E7.
5. **Defense benchmark:** E5, preferably in the appendix unless space permits.

There are currently ten executable experiment families when E1 and E3
subexperiments are counted separately, plus the smoke test and independent
theory predictor.

## Experiment inventory

### E0: four-scenario smoke test

- Runner: `scripts/run_smoke.py`
- Purpose: integration only; never scientific evidence.
- Checks: S1--S4, LLM wrapper path, v3 role installation, output schema.
- Run:

```bash
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_smoke \
  --mock --quota-mode standard --out v3_smoke_all
```

### E1a: coarse nonlinear response and size sweep

- Runner: `scripts/run_e1_scaling.py`
- Claims: C1 and preliminary C2.
- Varies: `N`, `alpha`, seed.
- Primary outputs: collapse probability, continuous primary-risk score,
  transition width, provisional `alpha_c(N)`.
- Current issue: the default grid is useful for pilot discovery but is not a
  substitute for an N-specific bracket.
- Pilot:

```bash
WOLFBENCH_FINAL_SCENARIOS=s1 \
WOLFBENCH_FINAL_N_GRID=100,300,1000 \
WOLFBENCH_FINAL_SEEDS=1,2,3 \
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e1_scaling \
  --preset pilot --mock --quota-mode behavioral_only --out v3_e1a_pilot
```

### E1b: refined finite-size scaling

- Runner: `scripts/run_e1_refined_scaling.py`
- Claims: main evidence for C1 and C2.
- Varies: N-specific alpha grid and repeated seeds.
- Required analysis: grid-interpolated midpoint, logistic midpoint sensitivity,
  10--90 transition width, seed bootstrap, linear-vs-logistic likelihood/AIC,
  and `K_c=N*alpha_c`.
- Status: v3 grids are exploratory brackets. Only S1 N=100 was previously
  bracketed; pilot every N before a paper run.
- Run one pilot bracket:

```bash
WOLFBENCH_E1_REFINED_N=100,300 \
WOLFBENCH_E1_REFINED_SEEDS=1,2,3 \
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e1_refined_scaling \
  --track s1 --mock --quota-mode behavioral_only --out v3_e1b_s1_pilot
```

### E1c: fixed-count and liquidity/social decomposition

- Runner: `scripts/run_e1_size_decomposition.py`
- Claim: causal support for C2.
- Varies: society size N, requested harmful count K, social channel, liquidity
  exponent `q in {0, 0.5, 1}`.
- Why needed: without it, a reviewer can argue that falling `alpha_c` is a
  mechanical consequence of the simulator's `L(N) proportional to N^0.5`
  liquidity rule.
- Key comparison: whether the size ordering persists under per-capita depth
  (`q=1`) and how much disappears when social transmission is removed.
- Run:

```bash
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e1_size_decomposition \
  --mock --quota-mode behavioral_only --out v3_e1c_size_decomposition
```

### E2: cross-scenario scope

- Runner: `scripts/run_e2_mechanisms.py`
- Claims: scope/robustness of C1--C2 across S1--S4.
- Role in paper: S1 remains the dense main case. S2 is a directional
  replication. S3/S4 should report continuous excess risk when no 50% crossing
  is resolved; do not manufacture `alpha_c`.
- Current issue: S2--S4 v3 thresholds have not been calibrated. First use broad
  log-spaced alpha grids, then refine only resolved crossings.
- Run:

```bash
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e2_mechanisms \
  --preset pilot --mock --quota-mode behavioral_only --out v3_e2_pilot
```

### E3a: topology and feedback interventions

- Runner: `scripts/run_e3_social_dynamics.py`
- Claim: coarse causal support for C3.
- Conditions: no social channel, baseline, high exposure, high reach, harmful
  hub placement, reflexive feedback.
- Paper use: a topology intervention table or appendix figure. This experiment
  alone is not a deep social-dynamics contribution because it does not measure
  private-vs-social information.
- Needed change in interpretation: call it the topology audit, not the full
  social-dynamics analysis.

### E3b: game-theoretic social phase diagram

- Runner: `scripts/run_e3_social_phase_diagram.py`
- Analyzer: `scripts/analyze_e3_phase_diagram.py`
- Claim: direct test of C3 and Proposition 1.
- Controlled population: value-oriented QRE agents, matching the analytical
  assumptions; mixed roles are included as an external-validity reference.
- Interventions: weak coupling, baseline, high response precision, high
  conformity, high reach, high attention, and combined strong coupling.
- Key test: conditions with measured coupling proxy above one should exhibit a
  sharper response/more frequent cascade than conditions below one.
- Run:

```bash
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e3_social_phase_diagram \
  --preset pilot --mock --quota-mode behavioral_only --out v3_e3b_phase_pilot
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.analyze_e3_phase_diagram \
  --out v3_e3b_phase_pilot
```

### E3c / code-E8: information-theoretic network signal game

- Runner: `scripts/run_e8_network_signal_game.py`
- Analyzer: `scripts/analyze_e8_signal_game.py`
- Claim: direct test of C3 and Proposition 2.
- Conditions: private-only, content-only, social-proof-only, full game, low/high
  attention, precise/noisy private signals, static trust, shuffled sender,
  delayed messages, and hub placement.
- Event artifacts: `agent_decisions.csv`, `messages.csv`, `exposures.csv`.
- Main estimands: bias-corrected `I(A;M|V,X)`, `I(A;V|M,X)`, social dominance,
  social-to-trade transfer entropy, and private/social conflict-following rate.
- Run and analyze:

```bash
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.run_e8_network_signal_game \
  --preset pilot --mock --quota-mode behavioral_only --out v3_e3c_information_pilot
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.analyze_e8_signal_game \
  --out v3_e3c_information_pilot
```

### E4: LLM quota robustness

- Runner: `scripts/run_e4_quota_robustness.py`
- Purpose: implementation robustness, not a main scientific result.
- Question: do low/standard/high LLM quotas materially change conclusions?
- Paper placement: appendix, with one sentence in the main text.

### E5: WolfGuard defense benchmark

- Runner: `scripts/run_e5_wolfguard_benchmark.py`
- Purpose: demonstrate that the response curves support cost-aware defense
  evaluation near the boundary.
- Current problem: many model comparisons can overwhelm the paper's three-part
  story. Put the full leaderboard in the appendix; retain at most NoGuard,
  one public baseline, WolfGuard, and oracle in the main paper.
- Do not run before v3 near-critical grids are recalibrated.

### E6: LLM-allocation scaling robustness

- Runner: `scripts/run_e6_controller_ablation.py`
- Actual purpose: behavioral-only versus mixed/high/fixed LLM quota across N.
- Naming problem: despite the filename, this is not a behavioral controller
  ablation. In the paper call it **LLM allocation robustness**.
- Status: v3 alpha brackets are exploratory and require a pilot.

### E7: plain-language role realism

- Runner: `scripts/run_e7_behavioral_realism.py`
- Purpose: answer the reviewer objection that all retail agents are identical
  rational score maximizers.
- Conditions: legacy score, mixed roles, all risk-averse, all trend-following,
  all social-following, and all aggressive.
- Required evidence: role/action conditional mutual information, trade-rate
  gap, action entropy, alpha=0 sanity check, and whether C1/C2 survive plausible
  role-mixture changes.
- Paper placement: short robustness subsection plus full appendix table.

### Independent theory predictor

- Runner: `scripts/network_game_theory.py`
- Theory text: `theory/network_signal_game.md`
- Purpose: generate coupling/multiplicity predictions before inspecting E3b.
- Run:

```bash
PYTHONPATH=src:. python -m final_hybrid_experiments.scripts.network_game_theory \
  --out v3_theory_predictions
```

## Required run order for the rewritten paper

1. Run E0 and the complete test suite.
2. For every scenario, verify alpha=0 has zero harmful primary failures while
   benign diffusion metrics can remain nonzero.
3. Pilot E1a and refine E1b brackets separately for each N.
4. Freeze grids and seeds; run E1b paper cells.
5. Run E1c before making a causal size-scaling claim.
6. Generate theory predictions, then run E3b and E3c without post-hoc sign
   changes.
7. Run E2 for scope, followed by E7/E6/E4 robustness.
8. Recalibrate near-critical cells and only then run E5 defenses.

## Paper figure plan

| Paper location | Evidence |
|---|---|
| Figure 1 | Conceptual overview: nonlinear response and leftward size shift |
| Figure 2 | Environment loop plus five plain-language roles and network signal game |
| Figure 3 | E1b response curves, alpha_c(N), and critical harmful count |
| Figure 4 | E1c size/liquidity/social decomposition |
| Figure 5 | E3b coupling phase diagram plus E3c private-vs-social information boundary |
| Table 1 | Four case-grounded scenario cards |
| Table 2 | Cross-scenario summary with unresolved crossings explicitly marked |
| Appendix | Role realism, LLM allocation/quota, full event metrics, defense leaderboard |

## Non-negotiable reporting rules

- Never combine rows from different `social_game_version` values.
- Never report a midpoint if the tested alpha grid does not bracket 50% failure.
- Always show alpha=0 sanity results.
- Report critical harmful count as well as critical fraction.
- Distinguish benign diffusion reach from harmful safety cascade.
- Theory signs must be registered before inspecting paper-scale E3 results.
- Pilot/mock/behavioral-only results are calibration evidence, not final LLM
  benchmark results.
