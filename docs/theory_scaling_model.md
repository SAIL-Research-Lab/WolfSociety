# Harmful-Agent Finite-Size Scaling Model

## Status

This note derives the finite-size relation used by the Agent Society Dynamics
framework. It is theory-first: no smoke or pilot output is treated as evidence.
The complete feedback and information framework is in
`paper_experiments_v3/theory/agent_society_dynamics_principles.md`.

The word **law** below means a consequence of explicit scaling assumptions. It
does not mean a universal thermodynamic law.

## 1. Society response around the clean state

Let `x*_N` be the clean expected-action state of an `N`-agent society. Linearize
the bounded-rational network update around it:

```text
delta x_(t+1) = J_N delta x_t + E_N delta h_t,              (1)
```

where `J_N` is the heterogeneous social--market Jacobian. Its spectral radius
controls local asymptotic stability, but the benchmark has a finite horizon and
may use a directed, non-normal graph. Therefore define the one-sided
constant-shock failure gain

```text
chi^+_T,N(v) = max_{1 <= t <= T}
               [c' sum_{k=0}^{t-1} J_N^k E_N v]_+.          (2)
```

`v` is the signed attack direction, `c` is the society-level readout, and
`[y]_+=max(y,0)`. If the attacker may choose either sign of `v`, replace the
positive part by an absolute value. This gain
captures topology, role mixture, conformity, market feedback, and transient
amplification in a single response quantity.

For a stable, symmetric normal system and long horizon,

```text
chi_infinity,N ~ 1 / (1 - Lambda_N),                         (3)
```

provided the attack and readout overlap with the leading eigenmode. Equation
(3) is not generally valid for directed non-normal systems; equation (2) is the
primary definition.

## 2. Attack aggregation

Write the payoff/public-state field produced by harmful fraction `alpha` as

```text
h_N(alpha) = eta_N alpha N^delta v_N + o(alpha N^delta).     (4)
```

`delta` is the attack-aggregation exponent. It is mechanism-specific. Fix a
norm, normalize `||v_N||=1`, and assume
`eta_N -> eta_bar in (0,infinity)`. The normalization is necessary: without it,
powers of `N` can be moved arbitrarily between `N^delta` and `v_N`.

### Social exposure

If one harmful source reaches `r_N` recipients and `r_N ~ N^delta_s`, then
average exposure from `K=alpha N` sources is proportional to

```text
K r_N / N = alpha r_N ~ alpha N^delta_s.                    (5)
```

Thus `delta=delta_s`. Fixed reach gives `delta=0`; growing reach or hub capture
gives `delta>0`.

### Market pressure

If harmful order flow is proportional to `K=alpha N` and effective liquidity
depth is `L_N ~ N^ell`, then

```text
harmful market pressure ~ alpha N / L_N
                         ~ alpha N^(1-ell),                  (6)
```

so `delta=1-ell`. Per-capita liquidity (`ell=1`) removes this direct size
amplification; sublinear depth (`ell<1`) creates it.

### Placement

Hub placement need not change `delta`. It can instead increase the overlap of
`v_N` with a high-gain mode of the resolvent in (2). Mean degree therefore
cannot identify placement gain.

## 3. Finite-size fragility theorem

Let the clean society be a signed distance `b_N` from its scenario-specific
failure surface, with `b_N -> b_bar in (0,infinity)`. Assume nonvanishing
attack alignment:

```text
N^-zeta chi^+_T,N(v_N) -> chi_bar in (0,infinity).           (7)
```

This condition states explicitly that the normalized attack retains a
nonvanishing overlap with failure-relevant response modes. Any systematic
`N`-dependence of hub/mode alignment belongs in `zeta`, not in the constant.
`zeta` is the society-susceptibility exponent. Combining (2), (4), and (7), the
leading score displacement is

```text
Delta S_N(alpha) ~ eta_bar chi_bar alpha N^(delta+zeta).     (8)
```

Setting `Delta S_N(alpha_c)=b_N` gives

```text
alpha_c(N) ~ A N^-nu,          nu = delta + zeta,            (9)
K_c(N) = N alpha_c(N) ~ A N^(1-nu).                         (10)
A = b_bar / (eta_bar chi_bar).
```

Therefore, when

```text
0 < nu < 1,                                                   (11)
```

larger societies require a smaller harmful **fraction**, while the absolute
number of harmful agents still grows sublinearly. This is the theoretical bridge
between network/market microstructure and the paper-facing finite-size pattern.

The decomposition

```text
observed scaling exponent = attack aggregation + susceptibility
nu = delta + zeta                                             (12)
```

is the central falsifiable identity. A fit of `alpha_c(N)` alone cannot identify
the mechanism.

## 4. Relation to the nonlinear response curve

For a fixed scenario, the empirical response may be summarized by

```text
Pr(C=1 | N,alpha) = sigmoid[s_N(alpha-alpha_c(N))],           (13)
```

with transition width

```text
w_N = 2 log(9) / s_N.                                        (14)
```

Equation (13) is an estimator, not a theorem. Random initial conditions,
heterogeneity, and finite-horizon noise smooth the deterministic failure
surface. The midpoint estimated from (13) must not be equated with the
homogeneous mean-field spinodal. The spinodal is initialization- and
branch-specific; the benchmark midpoint is a probability-level estimand.

## 5. Nonlinear and metastable regime

The derivation of (9) assumes local response around the clean state. It may fail
when:

- the attack begins outside the linear neighborhood;
- the society is already in a multiple-equilibrium regime;
- collapse is dominated by stochastic barrier crossing;
- the collapse surface itself moves with `N`;
- the finite horizon truncates the leading response mode.

In those cases, `alpha_c(N)=A N^-nu` is a scaling ansatz to test, not a proved
property of the simulator. The paper should distinguish the linear-response
theorem from the nonlinear empirical law candidate.

## 6. Comparative statics and defenses

Inside the monotone linear-response regime,

```text
alpha_c(N) = b_N / [eta_N N^delta chi^+_T,N(v_N)].          (15)
```

Hence:

```text
partial alpha_c / partial eta_N             < 0,
partial alpha_c / partial chi^+_T,N(v_N)    < 0,
partial alpha_c / partial delta     < 0  for N>1.             (16)
```

This gives three defense routes:

1. reduce attack effectiveness `eta`;
2. reduce finite-horizon feedback gain `chi^+_T,N`;
3. reduce size-dependent aggregation `delta`.

Improving private information capacity is an epistemic intervention. It may
change the clean equilibrium, collapse margin, or information-dominance axis,
but it is not automatically the same as reducing network feedback.

## 7. Mechanism heterogeneity

| Scenario | Candidate aggregation channel | Main theoretical object |
| --- | --- | --- |
| S1 social pump | harmful reach plus social--market feedback | `delta_social`, `chi^+_T,N` |
| S2 influencer | central placement and mode overlap | attack direction `v_N`, `chi^+_T,N(v_N)` |
| S3 spoofing | order pressure relative to depth | `delta=1-ell` |
| S4 wash trading | volume-signal reach and liquidity illusion | channel-specific `delta`, readout `c` |

The exponents need not match across scenarios. Cross-domain generality should be
claimed for the decomposition in (12), not for one universal numerical value of
`nu`.

## 8. Evidence ladder for later experiments

| Level | Required evidence | Permitted wording |
| --- | --- | --- |
| Pattern | resolved `alpha_c(N)` on several sizes | “finite-size scaling pattern” |
| Law candidate | stable exponent with uncertainty and holdouts | “protocol-specific empirical law” |
| Mechanism-backed law | separately estimated `delta`, `zeta`, and `nu ~= delta+zeta` | “Agent Society Dynamics fragility law” |
| Universal law | invariance across domains and protocols | not currently claimed |

Future experiments should manipulate attack aggregation and susceptibility
separately. Without that separation, a declining `alpha_c` is descriptive and
cannot establish the proposed mechanism.
