# WolfBench bounded-rational network signaling game

This document maps the analytical model to the WolfBench simulator. The formal
theorems and the finite-size derivation live in
`agent_society_dynamics_principles.md`. It is deliberately separate from fitted
simulation results: no current smoke or pilot output is treated as evidence.

## 1. Agents and information

Retail agent `i` has a persistent role and an individualized parameter vector.
Its private observation is a noisy signal `v_i,t` of fundamental value,
compressed under private capacity `C_v`. The agent sees only a subset `z_i,t`
of messages from its network neighborhood under a distinct social-attention
capacity `C_s`. A message contains content, source identity, confidence,
and visible social proof. Source trust changes after the realized return shows
whether previously attended messages were directionally accurate.

The population contains distinct decision processes, not one score rule with
different coefficients:

1. Risk-Averse agents wait for strong evidence, trade small, and may challenge
   a doubtful message;
2. Value-Oriented agents use noisy private value estimates with quantal choice;
3. Trend-Following agents use sequential thresholds and inertia;
4. Social-Following agents respond to neighbors and visible social proof;
5. Aggressive agents use larger positions and noisy, impulse-prone reactions.

After trading, an agent makes a bounded-rational signaling choice among
silence, an original post, a reshare, and (when disagreement is visible) a
challenge. Utilities combine conviction, reputation, coordination/social proof,
source trust, and the cost of speaking. Logit choice allows mistakes. Thus
social information is endogenous: actions affect messages, messages affect
neighbors, trades affect prices, and realized prices update trust.

## 2. Mean-field game

For a binary reduction, let `a_i` be `-1` or `+1`, and let
`m = E[a_i]` be aggregate directional demand. The utility difference between
buying and selling is

`Delta U = eta*alpha + theta*s + K*m`,

where `alpha` is harmful pressure, `s` is average private evidence, and

`K = gamma*d*q_s(C_s)`,

with conformity `gamma`, effective degree/reach `d`, and processed social share
`q_s(C_s)=C_s/(C_s+C0)`. Under a logit quantal response with precision `beta`, a
symmetric equilibrium satisfies

`m = tanh{ beta/2 * [eta*alpha + theta*s + K*m] }`.             (1)

### Proposition 1: unique response versus social tipping

Let `J = beta*K/2`.

- If `J <= 1`, equation (1) has a unique stable fixed point for every external
  pressure.
- If `J > 1`, an interval of external pressures has three fixed points (two
  stable and one unstable), so a small change in harmful pressure can trigger a
  discontinuous social cascade.

**Proof.** The derivative of the right-hand side of (1) with respect to `m` is
`J*sech^2(.)`, whose maximum is `J`. For `J < 1` the map is a contraction and
the fixed point is unique. At zero pressure, when `J > 1`, the slope at the
origin exceeds one while the map remains bounded in `[-1,1]`; symmetry and
continuity therefore give two additional nonzero fixed points. The outer fixed
points have derivative below one and are stable; the origin is unstable. The
boundary case follows by continuity. QED.

For `J>1`, set `u=sqrt(1-1/J)`. The two homogeneous spinodal fields are

`h_spin^+ = K*u - (2/beta)*atanh(u)`,

`h_spin^- = (2/beta)*atanh(u) - K*u = -h_spin^+`.

The positive field destroys the negative metastable branch, and the negative
field destroys the positive metastable branch. For a positive attack field, the
lower branch disappears at

`alpha_spin^+ = [h_spin^+ - theta*s]/eta`.                     (2)

Equation (2) is **not** the benchmark's empirical midpoint `alpha_c`. It is a
branch-destruction field under a particular initialization. Indeed `h_spin^+`
increases with `K`, because stronger coordination deepens both metastable
branches. The simulator's `alpha_c` also depends on initialization, stochastic
seeding, attack placement, finite horizon, and the collapse readout. The
paper-facing comparative statics must therefore use the Jacobian and
finite-horizon susceptibility derived in the formal framework, not equate the
spinodal with the observed response midpoint.

For heterogeneous network agents, the scalar `J` is replaced by the spectral
radius of the fixed-point Jacobian. Mean degree times mean conformity is only a
proxy and can fail under hub placement or directed, non-normal influence.

## 3. Information-theoretic cascade criterion

Let `V` be the raw private signal, `Z_v` its internal representation, `M` raw
social information, `Z_s` its attended representation, `X` public market
history, `R` role, and `A` the trade. Assume the two raw channels affect action
only through their respective representations, with separate constraints

`I(V;Z_v | M,X,R) <= C_v`,

`I(M;Z_s | V,X,R) <= C_s`.

### Proposition 2: channel-specific influence bounds

`I(A;V | M,X,R) <= C_v`,

`I(A;M | V,X,R) <= C_s`.

**Proof.** Apply conditional data processing to `V -> Z_v -> A` given
`(M,X,R)` and to `M -> Z_s -> A` given `(V,X,R)`. QED.

This produces an operational definition. Estimate

`I_social = I(A;M | V,X,R)` and `I_private = I(A;V | M,X,R)`,

then report `D = I_social/(I_social+I_private)` only above a preregistered
minimum-information floor. A strong empirical cascade
requires both high `D` and a high conflict-follow rate: among decisions where
private and social signals disagree, the action follows the social direction.
The Schreiber transfer-entropy estimand should use a lagged source,
`I(A_t;M_{t-1} | A_{t-1},X_t,R)`. A same-step exposure-to-action conditional
mutual information may also be useful, but must not be called transfer entropy.

`D=1/2` is an information crossover, not the game-theoretic phase boundary.
The formal framework proves a local weak-signal expression for `D` and gives
counterexamples to any universal `J=1 iff D=1/2` identity. A socially driven,
self-sustaining cascade requires both sufficient feedback gain and social
information dominance.

These quantities are diagnostics, not causal effects by themselves. A future
intervention experiment would obtain
causal leverage from paired interventions: private-only, content-only,
proof-only, full game, reduced attention, static trust, shuffled source
identity, delayed messages, and hub placement.

## 4. Falsifiable validation plan

Future role experiments reject the “all agents are identical rational score maximizers” explanation
only if the mixed population has multiple policy families in the event log and
the principal benchmark conclusions are not confined to `legacy_score`.

Future mechanism experiments support the network-game mechanism only if:

- full-game and proof-only conditions increase social dominance and
  conflict-following relative to private-only;
- reducing attention reduces social information flow;
- shuffling sender identity removes the benefit or harm caused by adaptive
  trust while leaving message volume approximately controlled;
- delaying messages reduces lagged social-to-trade transfer entropy;
- increasing reach or hub placement strengthens the finite-horizon response in
  the direction predicted by the Jacobian/resolvent framework.

Failure of any sign is informative and should be reported rather than absorbed
by post-hoc parameter changes.

## 5. Theoretical lineage

The design combines [quantal response equilibrium (McKelvey and Palfrey,
1995)](https://doi.org/10.1006/game.1995.1023), [rational inattention (Sims,
2003)](https://doi.org/10.1016/S0304-3932(03)00029-1), [information cascades
(Banerjee, 1992)](https://doi.org/10.2307/2118364), [transfer entropy
(Schreiber, 2000)](https://doi.org/10.1103/PhysRevLett.85.461), and
[zero-intelligence constrained trading (Gode and Sunder,
1993)](https://doi.org/10.1086/261868). The implementation uses these as
modeling principles; it does not claim that the finite WolfBench population
exactly satisfies the mean-field assumptions.
