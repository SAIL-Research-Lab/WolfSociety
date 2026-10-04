# Closure Validity Audit

This audit uses existing fixed-K episode data only. It does not modify or refit the paper.

## 1. Definition-implied versus empirical content

For `log R = a + b_N log N + b_K log K`, the proxy definition
`G = R N^ell / K` implies exactly:

- `zeta = b_N + ell`,
- `delta + zeta = 1 + b_N`, because `delta = 1 - ell`,
- the response-contour prediction is `nu = 1 + b_N / b_K`, not
  `1 + b_N`, unless `b_K = 1`.

Numerically, the largest fitted identity error across depth rules is 2.831e-15. Thus the cancellation of ell is definition-implied to numerical precision.
Holding the baseline simulator responses fixed and changing only the analyst normalization from ell=0 to ell=1 changes zeta from -0.784 to 0.216, while delta+zeta remains 0.216 in every row.
In the baseline fixed-K data, b_N=-0.784 and b_K=1.575. The proxy sum is 0.216, while the direct response-contour formula gives 0.503.

## 2. Independent size-response signal in subcritical fixed-K data

Using only cells with P02 failure prevalence below 1/2 gives 300 episodes across 25 (N,K) cells. The direct risk-size coefficient is b_N=-0.894 (paired-seed bootstrap 95% CI [-0.960, -0.852]).
The harmful-count elasticity is b_K=1.683 (95% CI [1.664, 1.763]). The contour-implied exponent is 0.469 (95% bootstrap interval [0.435, 0.501]).
The sign of b_N is empirical and is not created by the gain normalization. However, its interpretation as an amplification mechanism remains limited because R is the same score used to define collapse.
At each fixed K with at least three subcritical sizes, the direct size slopes range from -1.007 to -0.573; the negative direction is not driven by pooling different harmful counts.

## 3. Boundary prediction without held-out midpoints

The strict test fits the subcritical response surface after removing every response row at the target N, uses the pre-specified score threshold R=1, and uses no boundary midpoint for fitting or calibration.
It obtains mean absolute log error 0.390, 3/6 predictions within 1.5x, 6/6 within 2x, and predicted nu=0.463 versus observed nu=0.222.
For context, the existing leave-one-size-out constant-boundary baseline has mean absolute log error 0.212, and direct boundary power-law extrapolation has 0.092. The strict subcritical response prediction does not beat either midpoint-based baseline.

## Decision

The audit supports a real fixed-K size-response signal, but it does not support treating delta + zeta as an independent two-component validation. The ell cancellation is algebraic, and the stronger no-midpoint prediction must be judged by the errors above. Closure is best treated as a descriptive response-contour consistency check unless an independent perturbation-response measure is added.

Full results are in `algebra_audit.csv`, `reparameterization_audit.csv`,
`per_k_subcritical_signal.csv`, `subcritical_signal.csv`,
`no_midpoint_prediction_summary.csv`, and `no_midpoint_prediction_detail.csv`.
