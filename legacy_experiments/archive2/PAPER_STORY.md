# WolfBench paper story for the v3 benchmark

## One-sentence thesis

A harmful minority can trigger a nonlinear society-level collapse because
bounded agents switch from private evidence to socially reinforced signals;
as society size and network amplification grow, the harmful fraction required
to reach this regime can fall.

This sentence connects all three contributions. The paper should not read as
three unrelated studies of thresholds, scaling, and social networks.

## The three contributions

1. **Phenomenon:** WolfBench reveals a finite-size nonlinear collapse response:
   risk stays low, rises rapidly in a narrow harmful-fraction interval, and
   saturates.
2. **Scaling:** the location of that response shifts with society size. The
   critical harmful fraction may fall even while the absolute harmful count
   grows, so a larger society can be proportionally more vulnerable.
3. **Mechanism:** a bounded-rational network signal game explains when the
   shift is amplified. Game theory predicts a coupling boundary; information
   theory measures when social information overtakes private information and
   produces a cascade.

The benchmark and defense evaluation are artifacts enabled by these
contributions, not additional co-equal claims.

## Causal story

The main text should repeatedly use the same chain:

`harmful pressure -> local social signals -> bounded attention and source trust
-> ordinary-agent adoption -> coordinated trades -> price feedback -> stronger
social proof -> nonlinear collapse`.

Society size enters through the number of potential adopters, network paths,
and market depth. The E1c decomposition is necessary to show which channel is
responsible rather than attributing every size effect to “emergence.”

## Recommended section order

1. Introduction: isolated-agent evaluation misses collective tipping.
2. WolfBench: four case-grounded scenarios, five simple retail roles, shared
   social-market loop, primary event definitions.
3. Theory:
   - mean-field network signal game and the `J>1` multiplicity condition;
   - finite-attention data-processing bound;
   - preregistered comparative statics.
4. Experimental design and estimands.
5. Result 1: nonlinear response in S1.
6. Result 2: finite-size shift and E1c decomposition.
7. Result 3: social coupling phase diagram and private/social information
   crossover.
8. Scope: S2--S4, role robustness, LLM allocation.
9. Defense benchmark as a compact utility demonstration.
10. Limitations and cross-domain mechanism validation.

## How to explain the five retail roles

Use behavior, not mathematical jargon:

- Risk-Averse: waits for strong evidence, trades small, challenges doubtful
  social claims.
- Value-Oriented: compares price with a noisy personal estimate of value.
- Trend-Following: follows recent price direction and tends to persist.
- Social-Following: copies trusted neighbors and visible popularity.
- Aggressive: takes larger, faster, occasionally impulsive positions.

After this plain-language description, one sentence can say that the internal
implementations combine quantal response, satisficing, heuristics, noisy
signals, and inventory constraints. Do not name a role “QRE” or
“zero-intelligence” in the main text.

## Main figures

- Figure 1: the two empirical signatures, nonlinear response and size shift.
- Figure 2: closed-loop benchmark plus the five role cards.
- Figure 3: dense E1 response curves and `alpha_c(N)` with uncertainty.
- Figure 4: fixed-K and liquidity/social decomposition of the size effect.
- Figure 5: left panel game-theoretic coupling phase diagram; right panel
  social-vs-private information dominance and conflict-following cascade rate.

The defense leaderboard, LLM model list, and full role ablations should not
interrupt this sequence.

## Claims that must be removed until v3 reruns finish

The numerical thresholds currently written in `AuthorKit27` were obtained from
older controller/social-game versions. In particular, do not retain the old
S1 values such as `alpha_c=0.055 at N=100` or `0.0052 at N=2000` after changing
the agent system. They can be restored only if v3 reruns independently recover
them.

Similarly, existing statements that exposure is weak but hub placement is
dominant must be re-estimated under v3. The mechanism wording can stay as a
hypothesis, not a result, until E3a--E3c are rerun.

## Statistical standard

- Show raw seed-level binary outcomes and continuous risk.
- Compare nonlinear and linear response models only on grids that resolve the
  transition.
- Bootstrap seeds, not individual agent-day events, for episode-level claims.
- Treat agent-day information estimates as within-episode diagnostics; use
  episode/seed as the independent unit for uncertainty.
- Report unresolved/censored thresholds explicitly.
- Use paired seeds for all causal interventions.

## Positioning and limitations

The defensible claim is not that WolfBench is a realistic financial market or
that a universal phase transition has been proven. It is a controlled
measurement instrument showing a reproducible finite-size nonlinear response
and testing a named social-information mechanism.

Future work should transport the same mechanism—not the finance parameters—to
other domains: recommender systems, public-information diffusion, shared
resource allocation, or multi-agent coordination. Cross-domain validation
should preserve the interventions on reach, attention, trust, social proof,
and feedback.
