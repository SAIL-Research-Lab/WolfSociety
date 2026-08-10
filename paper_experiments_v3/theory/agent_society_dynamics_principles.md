# Agent Society Dynamics: a theory-first framework

## Status and claim boundary

This note is the formal theory layer for WolfBench v3. It does **not** use the
current smoke or pilot outputs as evidence. Statements below are separated into:

- **theorems**, which hold inside the stated mathematical model;
- **principles**, which are model implications to be tested in the simulator;
- **empirical laws**, which must not be claimed until the corresponding
  estimands have been measured on frozen, replicated experiments.

The framework has two independent coordinates and one finite-size law:

1. a **feedback coordinate** measuring whether social perturbations reproduce;
2. an **information coordinate** measuring whether social evidence outweighs
   private evidence in decisions;
3. a **fragility law** connecting population size to the harmful fraction
   required to cross a society-level failure margin.

The feedback and information coordinates are related through behavior, but they
are not mathematically identical. In particular, there is no general identity
`feedback threshold = information crossover`.

## 1. Microfoundation: bounded-rational network choice

There are `N` agents. Agent `i` takes a directional action
`a_i,t in {-1,+1}`; hold can be added as a third action without changing the
information bounds below. Let `x_i,t = E[a_i,t | I_i,t]` be expected action.

Agent `i` observes:

- raw private evidence `V_i,t`, compressed to `Z^v_i,t` under private capacity
  `C^v_i`;
- raw social messages `M_i,t`, compressed to `Z^s_i,t` under social-attention
  capacity `C^s_i`;
- public state `X_t`, including market history;
- neighbors' expected actions through a nonnegative influence matrix `W_N`.

Private capacity and social-attention capacity are different objects. Write

```text
q_v(C^v_i) in [0,1],        q_s(C^s_i) in [0,1].
```

The binary logit response is

```text
x_i,t+1 = tanh{ beta_i/2 * [
    h_i(alpha,N,X_t)
    + theta_i q_v(C^v_i) Z^v_i,t
    + gamma_i q_s(C^s_i) sum_j W_ij x_j,t
] } .                                                     (1)
```

`beta_i` is response precision, `theta_i` is private-evidence sensitivity,
`gamma_i` is social conformity, and `h_i` contains harmful pressure and public
market feedback. The five retail roles correspond to different response maps
and parameter distributions. Equation (1) is an analytical reduction, not a
claim that every simulated role literally uses the same logit rule.

Market feedback can be included by augmenting `x_t` with market-state nodes, so
that `W_N` represents the combined social--market feedback operator.

## 2. Principle I: the spectral feedback law

Let `x*` be a clean fixed point and define

```text
B       = diag(beta_i/2),
Gamma  = diag(gamma_i q_s(C^s_i)),
D(x*)  = diag(1 - (x*_i)^2),
J_N    = D(x*) B Gamma W_N.                                (2)
```

`J_N` is the Jacobian of the expected-action update at the fixed point. Define
the local social reproduction number

```text
Lambda_N = spectral_radius(J_N).                           (3)
```

### Theorem 1: uniqueness and local stability

For the map in (1):

1. if `||B Gamma W_N||_infinity < 1`, it is a contraction for every external
   field and has a unique fixed point;
2. a fixed point `x*` is locally asymptotically stable under synchronous mean
   updates when `spectral_radius(J_N) < 1`;
3. it is locally unstable when `spectral_radius(J_N) > 1`.

**Proof.** The derivative of `tanh` is at most one. Therefore the global
Lipschitz constant in the infinity norm is bounded by
`||B Gamma W_N||_infinity`; Banach's fixed-point theorem proves part 1. The
linearization around `x*` is `delta x_(t+1) = J_N delta x_t`. Standard discrete
linear-system stability gives parts 2 and 3. QED.

The contraction condition is sufficient, not necessary. The spectral condition
is local. In a directed or non-normal network, `Lambda_N < 1` does not rule out
large transient amplification within a finite episode.

For a fixed, signed perturbation direction `v`, define the one-sided
finite-horizon failure susceptibility

```text
chi^+_T,N(v) = max_{1 <= t <= T}
               [c' sum_{k=0}^{t-1} J_N^k E_N v]_+,         (4)
```

where `E_N = D(x*)B` maps an external payoff field into actions and `c` maps
actions into the society-level failure score and `[y]_+=max(y,0)`. This definition
does not count a response that moves away from the failure surface as harmful.
If the adversary may choose either sign of the direction, use the two-sided
susceptibility

```text
chi^+-_T,N(v) = max_{1 <= t <= T}
                |c' sum_{k=0}^{t-1} J_N^k E_N v|.          (4a)
```

Equivalently, one may impose directional alignment so that the cumulative
readout is nonnegative at its maximizing time. Requiring every individual term
or every time point to have the same sign is sufficient but unnecessarily
strong. Equation (4), rather than `Lambda_N` alone, is the correct control
quantity for a finite-horizon directed agent society.

### Homogeneous corollary

If agents are homogeneous, the clean equilibrium is zero, and the network is
`d`-regular with unnormalized neighbor influence, then

```text
Lambda = beta * gamma * q_s(C^s) * d / 2.                  (5)
```

The scalar fixed point becomes

```text
m = tanh[ beta/2 * (h + gamma d q_s(C^s) m) ].             (6)
```

At zero field, `Lambda <= 1` gives a unique symmetric response, while
`Lambda > 1` gives the familiar symmetry-breaking multiplicity. This is the
Brock--Durlauf/Curie--Weiss special case of the network result, not the main
novelty of WolfBench.

For `Lambda > 1`, let `K=gamma d q_s(C^s)` and define

```text
u = sqrt(1 - 1/Lambda),
h_spin^+ = K u - (2/beta) atanh(u),
h_spin^- = (2/beta) atanh(u) - K u = -h_spin^+.             (7)
```

`h_spin^+` is the positive field at which the negative metastable branch
disappears; `h_spin^-` is the negative field at which the positive metastable
branch disappears. In words, equation (7) gives the spinodal field that destroys
the metastable branch opposite to the field direction. Differentiating the
positive branch gives `partial h_spin^+ / partial K = u > 0`. Stronger
coordination deepens both metastable branches. The spinodal must **not** be
identified with the
benchmark's empirical midpoint `alpha_c`, whose value depends on initialization,
noise, finite horizon, attack placement, and the collapse readout.

## 3. Principle II: the dual-capacity information law

Assume that, conditional on public state `X` and role `R`, raw private evidence
affects action only through `Z^v`, and raw social messages affect action only
through `Z^s`. Impose separate channel constraints

```text
I(V; Z^v | M,X,R) <= C^v,
I(M; Z^s | V,X,R) <= C^s.                                  (8)
```

### Theorem 2: channel-specific influence bounds

```text
I(A;V | M,X,R) <= C^v,
I(A;M | V,X,R) <= C^s.                                     (9)
```

**Proof.** Conditional on `(M,X,R)`, the model imposes the Markov chain
`V -> Z^v -> A`; conditional data processing gives the first bound. Conditional
on `(V,X,R)`, `M -> Z^s -> A` gives the second. QED.

Define

```text
I_private = I(A;V | M,X,R),
I_social  = I(A;M | V,X,R),
D         = I_social / (I_social + I_private),              (10)
```

only when `I_social + I_private` exceeds a preregistered information floor.
`D > 1/2` means social information has the larger measured conditional
association. It is not, by itself, a phase-transition theorem or a causal claim.

### Proposition 3: local information crossover

Consider a weak-signal binary logit model conditional on `(X,R)`:

```text
Pr(A=1) = sigmoid[u_0 + epsilon(b_v V + b_s M)],
```

where the residual private and social signals are centered and have finite
conditional variances. A second-order expansion of Bernoulli KL divergence gives

```text
I_private = epsilon^2 K_0 b_v^2 Var(V | M,X,R) + o(epsilon^2),
I_social  = epsilon^2 K_0 b_s^2 Var(M | V,X,R) + o(epsilon^2),   (11)
```

with `K_0 > 0` determined by the baseline choice probability. Consequently,

```text
Omega = [b_s^2 Var(M | V,X,R)] /
        [b_v^2 Var(V | M,X,R)],
D = Omega/(1+Omega) + o(1).                                 (12)
```

Thus `D = 1/2` corresponds locally to `Omega = 1`, an information crossover.
It does **not** correspond generally to `Lambda_N = 1`: `Omega` compares the two
input channels, while `Lambda_N` measures feedback around the social network.

## 4. Principle III: a socially driven cascade needs two conditions

The pair `(Lambda_N, Omega_N)` gives a two-axis phase description.

| Feedback | Information | Interpretation |
| --- | --- | --- |
| `Lambda < 1` | `Omega < 1` | private-anchored and dynamically damped |
| `Lambda < 1` | `Omega > 1` | socially driven choices, but no self-sustaining feedback |
| `Lambda > 1` | `Omega < 1` | latent coordination instability, but private evidence remains the stronger input |
| `Lambda > 1` | `Omega > 1` | self-reinforcing, socially dominated cascade regime |

For finite-horizon non-normal networks, replace the binary feedback label by the
measured one- or two-sided gain: a society can be transiently fragile even when its
asymptotic spectral radius is below one.

This yields the **cascade conjunction principle**:

> A self-reinforcing social-information cascade requires both sufficient
> feedback gain and social-over-private information dominance.

The two requirements can move separately. Increasing private signal quality can
lower `Omega` without changing the social network Jacobian. Increasing network
reach can raise `Lambda` without making social messages more informative than
private evidence. These counterexamples rule out a universal
`Lambda=1 iff D=1/2` identity.

## 5. Principle IV: the finite-size fragility law

Let the aggregate harmful field produced by harmful fraction `alpha` satisfy

```text
h_N(alpha) = eta_N * alpha * N^delta * v_N
             + o(alpha N^delta),                             (13)
```

where `delta` is the **attack-aggregation exponent**. It captures how the total
pressure from `K=alpha N` harmful agents is diluted or amplified by reach,
placement, liquidity, and normalization. To make `delta` identifiable, fix a
norm and normalize `||v_N||=1`; otherwise arbitrary powers of `N` can be moved
between `N^delta` and `v_N`. Assume also
`eta_N -> eta_bar in (0,infinity)`.

Examples:

- if one harmful source reaches `r_N ~ N^delta` recipients, average social
  exposure scales as `alpha N^delta`;
- if total harmful order flow is `alpha N` and liquidity depth is
  `L_N ~ N^ell`, market pressure scales as `alpha N^(1-ell)`, so
  `delta = 1-ell`;
- hub placement changes the attack direction `v_N` and its overlap with the
  leading response mode, even if mean degree is unchanged.

### Assumption 4.1: nonvanishing attack alignment

Let the clean society's signed distance to the failure surface be `b_N`, with
`b_N -> b_bar in (0,infinity)`. For the fixed attack direction, assume

```text
N^-zeta chi^+_T,N(v_N) -> chi_bar in (0,infinity).           (14)
```

This makes explicit that the normalized attack direction has a nonvanishing
asymptotic overlap with the failure-relevant response modes. If the attacker may
choose the sign, replace `chi^+` by `chi^+-`. The assumption includes cases in
which hub alignment changes with `N`; such changes contribute to `zeta` rather
than silently entering the constant.

`zeta` is the **susceptibility exponent**. In a symmetric near-critical system,
`1-Lambda_N ~ N^-zeta` implies the infinite-horizon susceptibility grows as
`N^zeta`; in a directed system, `zeta` may instead be driven by non-normal
finite-time amplification.

### Theorem 4: finite-size critical-fraction scaling

Within the linear-response region around the clean equilibrium,

```text
alpha_c(N) ~ A N^-nu,        nu = delta + zeta,             (15)
K_c(N) = N alpha_c(N) ~ A N^(1-nu).                         (16)
A = b_bar / (eta_bar chi_bar).
```

**Proof.** By differentiating the fixed-point/dynamic response around the clean
state, the society-level score shift is
`Delta S_N = eta_N alpha N^delta chi^+_T,N(v_N)
+ o(alpha N^delta chi^+_T,N)`. Setting `Delta S_N=b_N`, using the three limits
above, and solving for `alpha` gives (15) with the stated `A`; multiplying by
`N` gives (16). QED.

This theorem explains the otherwise counterintuitive regime

```text
0 < nu < 1:
    alpha_c(N) decreases,
    K_c(N) still increases, but sublinearly.                 (17)
```

The exponent is not universal. It decomposes into an attack-aggregation term
and a society-susceptibility term. Different mechanisms can therefore share the
same qualitative leftward threshold shift while having different exponents.

The theorem is a local asymptotic statement. Near a metastable spinodal or under
large shocks, nonlinear barrier crossing replaces linear response; equation
(15) then becomes an ansatz to test rather than a theorem to assert.

## 6. Intervention duality

The framework separates three defense targets:

1. **feedback defenses** reduce `Lambda_N` or finite-horizon gain `chi^+_T,N`
   (limit reach, slow resharing, reduce visible proof, damp market feedback);
2. **epistemic defenses** reduce `Omega_N` (improve private evidence, provenance,
   source calibration, or conflict warnings);
3. **attack-surface defenses** reduce `eta` or `delta` (limit hub capture,
   harmful placement gain, order size, or cross-channel coupling).

Under the monotone linear-response assumptions of Theorem 4, reducing any of
`eta_N`, `delta`, or `chi^+_T,N` raises `alpha_c`. Increasing private capacity need
not change `alpha_c` through the same equation; it operates through the
information axis and must be modeled as a change in the clean equilibrium,
decision coefficient, or collapse margin.

## 7. What later experiments must identify

No current pilot output is used here. A future theory test must estimate the
following separately:

1. `Lambda_N` or, preferably for the directed finite-horizon system,
   signed `chi^+_T,N` under controlled perturbations;
2. `I_social`, `I_private`, their uncertainty, and conflict-following, with role
   and public state included in the conditioning set;
3. the attack-aggregation exponent `delta` under social-only and market-only
   channel interventions;
4. the susceptibility exponent `zeta` under fixed attack input;
5. whether the fitted finite-size exponent satisfies `nu ~= delta+zeta`;
6. whether interventions move the feedback and information coordinates in the
   preregistered directions.

Only item 5 turns the observed `alpha_c(N)` trend into a mechanism-backed
finite-size law. Without the decomposition, the correct wording is “empirical
scaling pattern,” not “agent-society principle.”

## 8. The four paper-facing principles

1. **Spectral feedback law.** Society-level amplification is governed by the
   Jacobian/resolvent of the interaction network, not by the rationality of an
   average agent.
2. **Dual-capacity information law.** Private reasoning and social attention
   impose separate information bounds; social dominance is an information
   crossover, not an equilibrium identity.
3. **Cascade conjunction principle.** A social cascade requires both feedback
   gain and social-over-private decision dominance.
4. **Finite-size fragility law.** The critical harmful fraction is controlled by
   attack aggregation times society susceptibility:
   `alpha_c(N) ~ N^{-(delta+zeta)}`.

Together these principles form the proposed Agent Society Dynamics framework.
They are domain-portable: a new domain changes the influence operator, private
and social channels, attack aggregation, and collapse readout, while preserving
the same theoretical objects.

## 9. Theoretical lineage

- McKelvey and Palfrey (1995), quantal response equilibrium:
  <https://doi.org/10.1006/game.1995.1023>
- Banerjee (1992), herd behavior and social learning:
  <https://doi.org/10.2307/2118364>
- Brock and Durlauf (2001), discrete choice with social interactions and the
  interaction threshold: <https://doi.org/10.1111/1467-937X.00168>
- Sims (2003), rational inattention and finite information capacity:
  <https://doi.org/10.1016/S0304-3932(03)00029-1>
- Schreiber (2000), transfer entropy:
  <https://doi.org/10.1103/PhysRevLett.85.461>

WolfBench's theoretical contribution is not the scalar `tanh` bifurcation. It is
the separation and experimental identification of network feedback,
private-versus-social information competition, and finite-size attack
aggregation in heterogeneous agent societies.
