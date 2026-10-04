# Fresh P0 causal experiment audit

Analysis status: **frozen 12-seed**. All analyzed rows have `fresh_rerun=1`; historical rows are excluded.

## 1. Native liquidity rerun

- q=0: b_N=-0.764, b_K=1.046, nu_response=0.269 (124 episode rows).
- q=0.5: b_N=-0.811, b_K=1.022, nu_response=0.207 (121 episode rows).
- q=1: b_N=-0.822, b_K=1.103, nu_response=0.255 (120 episode rows).

Resolved fresh liquidity boundaries: 6/6. An unresolved cell is not extrapolated.

## 2. Closed-loop ablation

Important metric audit: the S1 primary score is the minimum of a social-threshold score and a market-threshold score. It is therefore structurally zero when social propagation is disabled; that fact is not causal evidence by itself.

- closed_loop, N=300: alpha_c=0.0429 (interpolated).
- closed_loop, N=1000: alpha_c=0.0275 (interpolated).
- direct_harmful_first_hop, N=300: unresolved (not_bracketed).
- direct_harmful_first_hop, N=1000: unresolved (not_bracketed).
- no_environment_feedback, N=300: alpha_c=0.0429 (interpolated).
- no_environment_feedback, N=1000: alpha_c=0.0267 (interpolated).
- no_social_propagation, N=300: unresolved (not_bracketed).
- no_social_propagation, N=1000: unresolved (not_bracketed).

Interpret boundary movement only after all three paired conditions are bracketed. The no-environment intervention changes information flow during simulation; it is not post-hoc normalization.
Absolute market thresholds are also audited against paired alpha=0 runs. If alpha=0 already exceeds a market threshold, only clean-normalized market/retail contrasts are interpreted.

## 3. Response-regime crossover

- realized_severity / R_lt_0.3: nu_response=0.406, b_N=-0.564, b_K=0.950, rows=65, unique cells=7.
- realized_severity / R_0.3_to_0.6: nu_response=0.028, b_N=-0.252, b_K=0.260, rows=27, unique cells=6.
- realized_severity / R_0.6_to_1: nu_response=0.270, b_N=-0.174, b_K=0.238, rows=29, unique cells=5.
- realized_severity / R_ge_1: nu_response=0.364, b_N=-0.326, b_K=0.512, rows=23, unique cells=4.
- realized_severity / all_subcritical_R_lt_1: nu_response=0.207, b_N=-0.811, b_K=1.022, rows=121, unique cells=11.
- distance_to_fresh_boundary / alpha_over_alpha_c_lt_0.5: nu_response=0.161, b_N=-0.832, b_K=0.991, rows=84, unique cells=7.
- distance_to_fresh_boundary / alpha_over_alpha_c_0.5_to_0.9: nu_response=0.140, b_N=-0.469, b_K=0.545, rows=48, unique cells=4.
- distance_to_fresh_boundary / alpha_over_alpha_c_0.9_to_1.3: nu_response=0.731, b_N=-0.008, b_K=0.029, rows=24, unique cells=2.
- distance_to_fresh_boundary / alpha_over_alpha_c_ge_1.3: nu_response=0.144, b_N=-0.332, b_K=0.388, rows=60, unique cells=5.

### Fresh discrepancy decomposition

For the current pilot, nu_response=0.207 and the two-size boundary exponent is nu_b=0.369, leaving a gap of -0.162.
Under a near-boundary correction R ~ N^(b_N-kappa) K^b_K, the algebraic residual is kappa=b_K(nu_response-nu_b)=-0.165.
This kappa is a target for the stratified tests above, not yet an independent theoretical finding.

Severity-conditioned fits are diagnostic rather than automatically causal: conditioning on realized R can change cell support. A crossover claim requires ordered movement with adequate N/K support and seed-bootstrap intervals, not just point estimates.

## Reporting guardrails

- Do not call this real-LLM evidence; the fresh mechanism run is behavioral-only.
- Do not report a critical fraction when the evaluated grid fails to bracket 50% failure.
- Do not claim a universal scaling law from the two-size boundary check.
- Preserve the source hash and frozen config when extending the run.
