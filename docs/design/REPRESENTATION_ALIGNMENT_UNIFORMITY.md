# Response representation alignment and uniformity

This optional v2 research objective addresses two observations together: overly
similar response representations and loss of same-input view consistency when
dispersion alone is strengthened. The auxiliary loss is not a replacement for
prediction or either active distillation objective. Disabled weights preserve
the historical model, checkpoint identity and execution path.

For the same input `(control_row_id, perturbation_condition)`, take the Student
Response CLS from each of its two Global views. These are the existing final
RMSNorm outputs, before the distillation projector. Normalize only the auxiliary
copies to unit length, obtaining `u1` and `u2`. Neither view is detached.

\[
L_A = N^{-1}\sum_i(1-u_i^{1\top}u_i^2),\qquad
L_U = \tfrac12\sum_{v=1}^2
\log\left(|E|^{-1}\sum_{(i,j)\in E}
\exp(2t(u_i^{v\top}u_j^v-1))\right).
\]

`E` contains unordered pairs with the same canonical `cell_type::batch` context
and different perturbation conditions. Duplicate `(control,p)` inputs are
removed before device and microbatch partitioning. Each valid pair has equal
weight; contexts with more valid pairs consequently contribute more. Report
those proportions in calibration. Alignment includes all unique inputs.
An empty eligible pair population is invalid, not a successful zero loss.
No truth expression is used to choose pairs. Different perturbations may still
have similar biology: this is an explicit distributional hypothesis, not proof
that every pair is a true biological negative.

The training objective adds `a(k) * (lambda_A * L_A + lambda_U * L_U)`, where
`a(k)` warms up over completed optimizer updates and resumes from the saved
optimizer count. The initial research kernel scale is `t=2`. Balanced and
stronger coefficients must be fixed from training-side calibration before
formal configuration generation; test results cannot select them. There is no
target rank or pairwise cosine margin. The uniformity term can be negative.

The complete effective batch is the statistical population. Cross-device
gather remains differentiable; concatenation across gradient accumulation
precedes the nonlinear log-mean-exp. Averaging microbatch or device-level
uniformity losses defines a different objective and is forbidden. Distributed
parameter-gradient tests must match the full-population calculation after the
optimizer's usual gradient synchronization. Preserve the existing center and
EMA update boundaries.

Small diagnostic probes and validation batches are not the sealed training
population. They explicitly mark AU terms unavailable and report the remaining
prediction/distillation objective. They must not silently report an incomplete
AU term as a full-batch training loss. Full-population no-update calibration is
separate from these probes and records its sampling, device topology and RNG.

Validation covers matched effective rank and relative variance, raw and
unit-normalized representations, GG/GL/LL positives and matched controls across
conditions, prototype entropy/JS and correct-versus-mismatched distillation,
and prediction quality. Improving rank or reducing this auxiliary loss alone
does not establish a useful representation or successful distillation.

The mechanism is informed by Wang and Isola, ICML 2020,
[Understanding Contrastive Representation Learning through Alignment and
Uniformity on the Hypersphere](https://proceedings.mlr.press/v119/wang20k.html).
For unit vectors, squared distance is `2*(1-cosine)`; our alignment is half the
paper's squared-distance alignment. Context filtering and same-condition
exclusion are project adaptations and do not inherit a theorem about global
uniformity. The native implementation does not import an external package.
