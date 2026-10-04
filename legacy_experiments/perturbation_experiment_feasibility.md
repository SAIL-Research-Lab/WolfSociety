# Independent Perturbation Experiment Feasibility

## Decision

The experiment is feasible, but it should remain optional for the current
submission. The revised paper no longer depends on it. If added, its purpose
is to measure a direct local response to a controlled social-state
perturbation, not to repair the exponent closure.

## Why implementation needs a small simulator change

`WolfBenchEnv.run()` currently owns the complete daily loop. Agent
observation, decisions, market clearing, social propagation, and metric
updates are executed inside one method. The environment does not expose a
public day-level `step()` interface or a full state snapshot and restore
mechanism. Existing backend snapshots record controller usage rather than the
complete simulator state.

A paired perturbation should therefore be implemented through a narrow hook
inside `run()`, immediately before `_build_observation()`. The hook should
receive the day, current market state, social state, and a dedicated
perturbation random generator. It should not consume the base episode random
stream.

## Recommended perturbation

Inject a deterministic, unit-norm increment into the benign agents' social
evidence for one target asset on a pre-specified day. Run a matched control
episode with the same seed and no increment.

The perturbation should:

- leave harmful count, graph, controller allocation, market depth, and agent
  parameters unchanged;
- use a fixed direction and magnitude across society sizes;
- be applied after prior-day state is available and before agents observe the
  current day;
- use a separate random stream if recipient sampling is required;
- include a feedback-disabled comparison as a negative control.

## Primary readouts

Use direct excess responses between paired perturbed and control episodes:

- benign harmful exposure;
- reshare or message propagation;
- induced buy volume or net demand;
- subsequent price and liquidity response.

Collapse risk can be a secondary readout. The primary response should not be
divided by \(K/N^\ell\), because that would reintroduce the normalization
identity under audit.

## Suggested stages

1. Smoke test: \(N\in\{100,1000\}\), two seeds, perturbation on and off.
2. Pilot: \(N\in\{100,300,1000\}\), six seeds, feedback on and off, 72
   episodes.
3. Full audit: \(N\in\{100,300,1000,2000\}\), twelve seeds, feedback on and
   off, 192 episodes.

Fit the size dependence of the paired excess response and report paired
seed-bootstrap intervals. Pre-specify the perturbation day, magnitude,
recipient rule, and primary readout before the full audit.

## Recommendation for the current paper

Submit the closure-reframed version without waiting for this experiment. Add
the perturbation audit only if the hook can be implemented and validated
without changing the base episode trajectories. Its value would be an
independent causal response measurement, while the current intervention
results already provide the main mechanism evidence.
