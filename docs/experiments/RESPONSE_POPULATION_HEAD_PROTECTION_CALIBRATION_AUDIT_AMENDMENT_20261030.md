# R17-GPM-HG0: population supervision through the prediction head without its parameter gradient

## Evidence and hypothesis

This is one fresh two-epoch v2 representation screening candidate, not R50, a
restart of P1, or a repeated RI baseline. Its parent intervention is R16-GPM.
Reuse the exact retained RI, R14, R15 and R16 checkpoints/results.

R16 full frozen val445 has native all5000 MSE .0059831773 versus RI .0056349207
(+6.18%). Seen4775 all-gene Pearson improves, whereas unseen225 all-gene
Pearson point estimates decrease in all three definitions. Ownmask Pearson
improves from .33565185 to .35825907. Its deep506 Response CLS effective rank
15.06477/256 exceeds RI10.98483, while relative variance .0179643 is below
RI .0225348. Teacher SSL2 CLS H remains near log16384 and JS near .000168.
Native medium/crossed zero-update closure was independently validated. These
observations do not establish scientific success or prove which parameters
caused the unseen-output regression.

The falsifiable hypothesis is that allowing the extra population supervision to
update the shared decoder parameters changes cross-gene calibration differently
from routing that same supervision through the decoder into the upstream
representations. R16 does not contain a raw-control output skip: native absolute
prediction is Softplus(score), and delta=prediction-control is derived. RI is a
hidden-state condition-injection residual, not an expression-space residual.

ForkMerge (NeurIPS2023, https://arxiv.org/abs/2301.12618, sections3.1–3.2)
distinguishes negative transfer/generalization from gradient conflict. It motivates
checking actual target performance rather than accepting auxiliary-loss reduction.
This experiment is not ForkMerge, PCGrad or a claimed guarantee from that paper:
there is no branch merge, gradient-cosine gate, validation-based weight search,
or additional learned readout. The intervention below is a project hypothesis.

## One changed gradient route

Let phi be the upstream model parameters, theta the existing native prediction
MLP parameters, and h=H_phi(control,condition,gene). Population supervision uses
exactly the existing forward D_theta([h,e_p]) followed by Softplus, FP32 cell mean
and fixed TRAIN treated-mean MSE. Only on this auxiliary branch, use detached
parameter values of D_theta. This preserves the head-input Jacobian and sets
its auxiliary parameter gradient to zero. Primary prediction still trains both
phi and theta normally; the existing SSL objectives and Teacher behavior remain.

Do not detach the score, token or condition. Do not clear head.grad after
backward: that would also erase already accumulated primary gradients. Do not
change global requires_grad flags. Use native decoding with an opt-in parameter
route and functional detached parameter values; retain the default old forward.
Checkpoint recomputation, autocast and private RNG must use the same route.

## Fixed recipe and boundaries

- Same fresh seed1 initialization and native main sampler; 2epochs and exactly
  300 successful updates; global320=80/rank×2ranks×accum2; epoch2 last.
- Same complete1322 TRAIN groups,47807 selected treated rows, 4–40/group,
  private group schedule and allowed4775 query/target axis. Thirteen small
  conditions retain the primary task but skip auxiliary. Excluded225 numerical
  training columns stay forbidden in every branch; graph identity remains legal.
- Same native graph, RI injection, GELU, Softplus, norms, biases, masks/views,
  active SSL1condition/DINO/iBOT weights, EMA/center clocks, primary optimizer,
  warmup/cosine schedule and resource policy. No per-epoch validation, no test tuning.
- Same TRAIN bank and target arrays: retain their original preparation provenance;
  a new model source does not retroactively become their preparation source.
- Same original first-eight zero-update calibration rule on student.response
  parameters, excluding prediction-head parameters from the calibration norm.
  The auxiliary input gradient should match R16 at identical initial parameters,
  but verify rather than assign that equality. Calibration weight1 is only a
  temporary config; seal the actual measured lambda into the final self-contained
  fit config using min(1,.1*medianGpred/medianGaux). No manual extra multiplier.
- Only this candidate is fitted; RI/R16 are retained comparisons, not reruns.
  Equal optimizer-update budgets do not mean equal forwards/GPU time versus RI.

## Engineering acceptance before fitting

Default-true behavior and historical identity remain unchanged. False must be a
strict boolean and legal only for the supported population-native-head path.
Metadata records the auxiliary parameter-gradient exclusion; evaluation must not
read TRAIN banks. Tests must prove forward/loss equality, preserved upstream
auxiliary gradients, absent head auxiliary gradients, intact pre-accumulated main
head gradients, normal primary-head updates, state/RNG/Teacher/center invariants,
and activation-checkpoint recomputation compatibility. Source is published clean
and matched locally/GitHub/new server checkout before native execution.

Before the first CUDA execution, freeze the route-audit precision: identical-input
score/loss equality is exact in FP32 and BF16. CPU head-input and Response
gradient replay is exact. CUDA head-input and Response gradients use absolute
tolerance2e-6 and relative tolerance0, recording actual maximum absolute error,
tensor support/missing/shape and nonfinite checks. Auxiliary head-parameter
gradients must be exactly absent/zero; already accumulated primary-head
gradients and all protected state/RNG remain byte-identical. Negative tests must
reject errors above2e-6. Do not change these thresholds after a failed audit.
Audit replays do not contribute to the eight calibration measurements or lambda.

Actual server first-eight calibration and a ten-step integration/full-state
restore gate precede formal two-epoch fitting. Preserve a new run ID for every
attempt, zero-PKL closure and all failure receipts. No gate is passed by a report
or a synthetic test alone. Do not edit an active server source checkout.

## Scientific endpoints and rejection

After fitting, run native observer parity, the exact frozen val445/ordered300
control protocol with all18 Pearson metrics and ownmask/prehead RSA, then the
same medium1335/5317, deep128/506 and crossed32/128/16 panels. Match source,
config, checkpoint and sample identities. Report all/seen/unseen and all-gene/DEG
finite denominators and uncertainty where actually available; no NA→0 filling.

Primary scientific acceptance remains joint representation, input-specific
Teacher/Student distillation and predictive utility, not auxiliary loss alone.
Compare to R16 for the routing hypothesis and to retained RI for actual progress.
If unseen/global predictive utility remains worse, or seen gains disappear,
reject the claim that direct auxiliary decoder updates were the main obstacle.
An improved two-epoch point estimate is provisional; keep the frozen confirmation
boundary and do not tune against test. A negative result triggers the next
mechanism/literature/design iteration; it does not complete the overall task.

## Deferred alternatives

Historical MR1 and U24 already have negative evidence and are not restarted here.
Value masking is not whole-column output holdout. Control-anchored nonnegative
inverse-Softplus decoding requires resolving the zero-expression identity,
nonnegative output and induction-gradient tradeoff; no epsilon is selected in
this trial. No extra masks, output activation, variance/rank/contrastive loss or
Teacher schedule is stacked onto the gradient-route intervention.

## first8-calibration-only

The parameter provenance fragment names the sole source of the weight: the
original first-eight calibration rule. The published initialization probe uses
weight1 only to measure unit auxiliary gradients; a final fit config must contain
the actual frozen calibrated weight and its decision receipt. This reference
does not prescribe weight1 for formal fitting.

## Initial native audit failure and bounded instrumentation

Calibration attempt1 at source d6d105e67a4e01416b9cb0c26a414dba76835aef
failed on both ranks before completion: Response-gradient maximum absolute
difference .0001220703125 exceeded the preregistered2e-6 bound. No fit was
started and this receipt is not usable as calibration acceptance. The original
source/config/run and zero-PKL failure are retained. The cause is unresolved.

The next source adds failure-only repeatability instrumentation: one extra replay
per route, each under the existing protected state/private RNG context. Record
within-default, within-protected and cross-route scalar comparison summaries,
then raise the same original failure. Do not change tolerances, selected first
eight measurements, the lambda rule, or normal successful receipt semantics.
These summaries are diagnostic evidence rather than acceptance criteria.

## Amendment identity and frozen original

The complete text above is preserved verbatim from commit
`faaff68967f2cf973985fe04a2e819af06c68c47`; that 9009-byte document has SHA256
`6d746ef1ae52286ec06b9bfa78c095f53a2d6b9146335d6f91e5ad8183e513ca`.
It contains the original design followed by the calibration provenance fragment
and the actual attempt1 failure/instrumentation amendment. Those later additions
are post-failure evidence, not retrospective preregistration or passed calibration.

The original [R17 design](RESPONSE_POPULATION_HEAD_PROTECTION_20261030.md) is
frozen to its exact 7710-byte content at commit
`d6d105e67a4e01416b9cb0c26a414dba76835aef`, SHA256
`c57a42794c6f7ece98b0ac510cb2ccb9a18c7eb3045acf680e7683b01567afc6`.
This restores the genuine versioned bytes bound by the original DESIGN_STEP
evidence; no historical receipt or managed workflow state is rewritten. Future
calibration evidence and amendments must use their own immutable version/path.

The 2e-6 absolute/zero relative CUDA gradient bound, exact score/loss and primary
head/state/RNG requirements remain unchanged. Attempt1 remains failed, its cause
remains unresolved, and this documentation repair provides no fit authorization
or scientific acceptance. Retained RI/R16 baselines are not rerun.

Historical configuration references ending in the original document's
`#first8-calibration-only` fragment retain their original recorded text. The
fragment definition is preserved above in this amendment: only the first eight
measurements determine lambda; formal fitting requires the actually measured
weight and a valid calibration decision. This relocation changes no configuration,
calibration source identity, loss, optimizer, data boundary or acceptance gate.
