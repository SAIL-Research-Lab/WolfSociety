# P0 causal reruns

This directory is an independent, from-scratch rerun of the three highest-priority
reviewer questions. It does not treat any historical output as evidence.

## Questions

1. Does fixed-count dilution persist when the simulator is rerun with native
   liquidity exponents `q = 0, 0.5, 1`?
2. Does the collapse boundary move when the behavioral-environmental feedback
   path is cut while trades and market impact are still executed?
3. Does the response-surface exponent change between subcritical and
   near-transition severity regimes, explaining the response/boundary mismatch?

## Causal conditions

- `closed_loop`: current simulator, including social and market feedback.
- `no_environment_feedback`: treated trades still clear and determine outcome
  metrics, but later agents receive a paired alpha=0 market-observation replay.
  The same condition also disables return-dependent amplification inside social
  delivery. Social messages themselves remain available.
- `no_social_propagation`: market feedback remains, while exposure, resharing,
  and conformity are disabled.
- `direct_harmful_first_hop`: harmful-source messages can reach immediate
  neighbors, but second-hop bot resharing and benign retransmission are removed.
  This is the cleaner test of direct exposure versus a social cascade.

The no-environment condition is therefore an intervention on information flow,
not a post-hoc normalization of an unchanged trajectory.

## Runtime scope

The frozen default is 12 seeds. It uses `behavioral_only`, so it is a controlled
mechanism experiment and makes no external API calls. This is deliberate: the
current shell has no OpenRouter credential, and mock output must not be reported
as real LLM evidence. The runner records the current simulator source hash in
every row.

Run a pilot first:

```bash
cd /Users/zhangyuejun/Documents/aaai
WolfBench-main/.venv/bin/python p0_causal_reruns_20260821/run_experiments.py \
  --experiment all --seeds 1,2,3
```

Continue to the frozen 12-seed run (completed cells are skipped):

```bash
WolfBench-main/.venv/bin/python p0_causal_reruns_20260821/run_experiments.py \
  --experiment all
```

Analyze only the fresh rows:

```bash
MPLCONFIGDIR=tmp/mpl WolfBench-main/.venv/bin/python \
  p0_causal_reruns_20260821/analyze_results.py
```

Raw checkpoints are under `results/raw/`; analysis tables, figures, and the
claim audit are under `results/analysis/`.
