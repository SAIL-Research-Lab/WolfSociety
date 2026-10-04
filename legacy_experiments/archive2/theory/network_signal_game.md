# WolfBench bounded-rational network signaling game

This document states the analytical model and falsifiable predictions used by
the final-hybrid experiments. It is deliberately separate from fitted
simulation results: the propositions are properties of the stylized model,
while E7/E8 test whether the richer simulator exhibits the predicted signs and
tipping behavior.

## 1. Agents and information

Retail agent `i` has a persistent role and an individualized parameter vector.
Its private observation is a noisy signal `v_i,t` of fundamental value. The
agent sees only a capacity-constrained subset `z_i,t` of messages from its
network neighborhood. A message contains content, source identity, confidence,
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

`K = gamma*d*q(C)`,

with conformity `gamma`, effective degree/reach `d`, and processed social share
`q(C)=C/(C+C0)`. Under a logit quantal response with precision `beta`, a
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

For `J>1`, set `u=sqrt(1-1/J)`. The positive magnitude of the spinodal pressure
is

`h_c = K*u - (2/beta)*atanh(u)`,

and the lower branch disappears at

`alpha_c = [h_c - theta*s]/eta`.                               (2)

Equation (2) preregisters the main comparative statics: greater reach,
conformity, processing of social information, or payoff responsiveness moves
the system into the multiple-equilibrium regime and changes the cascade
threshold. If `J<=1`, experiments should show smooth amplification rather than
a discontinuous transition.

## 3. Information-theoretic cascade criterion

Let `V` be the private signal, `Z` its internal capacity-constrained
representation, `M` attended social information, `X` public market history,
and `A` the trade. Assume private information can affect the action only
through `Z` once `(M,X)` is given, and the information-processing constraint is

`I(V;Z | M,X) <= C`.

### Proposition 2: bounded private influence

`I(A;V | M,X) <= I(Z;V | M,X) <= C`.

**Proof.** Conditional on `(M,X)`, the model imposes the Markov chain
`V -> Z -> A`. The first inequality is the conditional data-processing
inequality; the second is the capacity constraint. QED.

This produces an operational definition. Estimate

`I_social = I(A;M | V,X)` and `I_private = I(A;V | M,X)`,

then report `D = I_social/(I_social+I_private)`. A strong empirical cascade
requires both high `D` and a high conflict-follow rate: among decisions where
private and social signals disagree, the action follows the social direction.
Transfer entropy `I(A_t;M_t | A_{t-1},X_t)` checks that social exposure adds
predictive information beyond action persistence and current price state.

These quantities are diagnostics, not causal effects by themselves. E8 obtains
causal leverage from paired interventions: private-only, content-only,
proof-only, full game, reduced attention, static trust, shuffled source
identity, delayed messages, and hub placement.

## 4. Falsifiable validation plan

E7 rejects the “all agents are identical rational score maximizers” explanation
only if the mixed population has multiple policy families in the event log and
the principal benchmark conclusions are not confined to `legacy_score`.

E8 supports the network-game mechanism only if:

- full-game and proof-only conditions increase social dominance and
  conflict-following relative to private-only;
- reducing attention reduces social information flow;
- shuffling sender identity removes the benefit or harm caused by adaptive
  trust while leaving message volume approximately controlled;
- delaying messages reduces social-to-trade transfer entropy;
- increasing reach or hub placement strengthens cascade reach and moves the
  observed transition in the direction predicted by Proposition 1.

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
