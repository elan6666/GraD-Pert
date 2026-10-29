# R15-DIR: projected response readout direction supervision

This separately identified v2 representation research candidate keeps the R14
training recipe and changes one auxiliary objective. It is not an R50 benchmark
configuration. The authorized screening budget remains two epochs, 300 successful
updates, global batch 320 (80 cells per rank, two ranks, accumulation two), seed 1,
without per-epoch validation. Epoch-two last is the prespecified endpoint.

## Evidence and hypothesis

R14's retained validation predictions deteriorated relative to retained R3-RI,
despite increased raw response-distance RSA. A subsequent zero-update diagnostic
used 16 TRAIN conditions and three private stochastic recipes per condition.
Its raw auxiliary-CLS/truth distance RSA was 0.649--0.748; after controlling truth
amplitude, rank-residual RSA was 0.00075--0.06976, with intervals crossing zero.
Global shared gradient cosine averaged +0.0743, so these measurements do not
support a universal strong-gradient-conflict explanation. These observations
motivate, but do not establish, a problem with radial auxiliary supervision.

The single falsification is whether removing the requirement to match projected
target magnitude yields useful direction geometry **and** better prediction.
This is not evidence that raw CLS norms are a shortcut: their norms were nearly
constant, and response strength can be represented in angles.

## Exact isolated change

For the original complete condition group, first average its Response CLS:

$$
\bar z_p=\frac{1}{n_p}\sum_i z_{p,i},\qquad u_p=W\bar z_p+b.
$$

Use the original random Linear256-to-256 head, with its original bias, exactly
once under the original BF16 forward context. Only the resulting readout and
detached fixed target are converted to FP32 for normalization and reduction:

$$
L_{\mathrm{DIR},p}=1-
\left\langle
\frac{u_p}{\max(\|u_p\|_2,10^{-6})},
\frac{\operatorname{stopgrad}(t_p)}{\max(\|t_p\|_2,10^{-6})}
\right\rangle.
$$

The former auxiliary objective was coordinate-mean MSE of the same two
unnormalized vectors. Do not normalize each cell before averaging. Do not add
another predictor, remove bias, change EMA, add a norm penalty, or change the
prediction decoder. Cosine is invariant to positive rescaling of the **readout**,
not necessarily the raw CLS, because the retained bias can convert radius into
angle. The new objective makes no guarantee of avoiding collapse.

The fixed bank is unchanged: 1322 eligible TRAIN conditions, the original
context-matched complete control slots, permitted 4775 expression columns,
original centering, frozen Gaussian QR projection and global RMS. All 1322
conditions remain; no reliability filter or resampling is introduced. Target
norms at or below epsilon fail explicitly. Observed readout norm/clamp counts
are diagnostic scalar summaries, never a reason to silently change the loss.
Weak-response directions can be noisy and become relatively more influential.

The main sampler, graph/query/view identities, private RNG, original group
schedule, rank-mean compensation `weight*world*150/1322`, primary prediction MSE,
SSL1 condition and SSL2 DINO/iBOT remain unchanged. SSL1 node remains off. The
auxiliary path does not invoke the prediction decoder or Teacher.

## Initialization calibration and version separation

Recalculate the coefficient from fresh seed-one initialization using the exact
original first eight epoch-zero batches and their assigned complete groups:

$$
\lambda=\min\left(1,
0.1\frac{\operatorname{median}\|g_{\mathrm{prediction}}\|_2}
{\operatorname{median}\|g_{\mathrm{direction,unit}}\|_2}\right).
$$

The parameter population is the original `student.response.parameters`, excluding
the auxiliary head. Sum signed group gradients and rank-average vectors before
taking norms. Do not average norms, choose batches, reuse the numerical R14
coefficient, or tune against validation/test. Calibration performs zero updates,
restores protected state/RNG and records the exact eight input identities. Its
unit-weight configuration is calibration-only; no fit may use it before the new
coefficient, source, config and calibration decision are sealed and published.

Historical MSE options and runtime identity remain unchanged when the new optional
field is absent/default. Only `response_anchor_loss=cosine_direction` adds the
new objective descriptor. Old checkpoints continue using their original native
sources. No R3-RI/R14 baseline is retrained or relabeled with a new source.

## Execution and interpretation

Before fit, require a matching clean local/GitHub/server commit and the original
native ten-update/full-group/resident-main-gradient/save-restore/300-control
inference preflight, with explicit direction identity and defined-target audit.
Use the existing resource gates and allocator. Keep the full successful run
zero-PKL; scientific arrays and checkpoints remain on `/data/yilangliu`.

Postfit uses the same 445 validation conditions and ordered controls, all 18
Pearson variants with actual finite denominators, own-target-masked response
Pearson/MSE and paired condition intervals. No test values select the candidate.
Reuse the fixed medium/crossed panels and separately report pre-head CLS,
readout and prediction geometry, direction/amplitude-controlled RSA, effective
rank with its sample ceiling, relative variance, same-input/condition Global and
Local relationships, and actual prototype entropy/JS/CE/KL when available.

Strong screening support requires improvement over retained R3-RI in all three
all-gene Pearson families, positive own-mask Pearson delta with paired CI above
zero, reduced own-mask MSE with paired CI below zero, and corroborating useful
representation evidence. Report DEG separately, including its limited unseen
finite denominator. Two epochs and one training seed are only a screen.
Higher rank or raw RSA alone, a falling head cosine loss alone, or recovery only
from R14 does not solve the overall scientific problem. If geometry improves
without prediction, redesign the next isolated mechanism; do not stack changes
onto this run or tune its checkpoint retrospectively.

## Method references

- [BYOL, NeurIPS 2020, section 3.1 equation 2](https://arxiv.org/abs/2006.07733):
  normalized predictor-target squared L2 equals twice one-minus-cosine. Its EMA
  target, nonlinear predictor, two-view objective and training guarantees are
  not reproduced here; the target here is a frozen noisy population estimate.
- [Relational Knowledge Distillation, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html):
  pair-distance/triplet-angle supervision is a different mechanism. R15 does
  not claim to implement it.
