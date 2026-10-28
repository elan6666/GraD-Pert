# Fixed TRAIN response group anchor

This is a falsifiable representation experiment, not a claimed improvement.
The reference is retained R3-RI epoch2-last (training/evaluation source
`f0cb20ebb2efab749970f2bf5d205ed2f358592c`), without retraining it.

## Mechanism

Keep native RI joint prediction and self-distillation. Add a fixed target for
each of the 1,322 eligible selected TRAIN conditions (47,807 treated rows).
Thirteen small conditions remain in main training and have no anchor target.
Only the 4,775 permitted expression columns are read for target construction;
the 225 excluded output-gene values and validation/test responses are not used.

For condition p and context k, use original selected treated proportions n_pk/N_p:

    delta_p = sum_k (n_pk/N_p) * (mean(treated_pk) - mean(TA union TB controls_k))
    r_p = delta_p - mean_condition(delta)

Project r with one PCG64(20261028) Gaussian reduced-QR matrix, fix diagonal signs,
scale by sqrt(4775/256), then divide by one TRAIN-only global RMS. Do not normalize
each condition. The projection is never learned or rerolled. Abort if the fixed
4,096-pair squared-distance relative RMS error exceeds .25, or projected ownmask
split-half cross-S is nonpositive/nonfinite.

Construct N_p fixed matched control slots from TA/TB (4<=N_p<=40). Preserve the
original context proportions; use deterministic per-condition/context hash order
and cyclic reuse when necessary. The input contains control expression, gene and
perturbation identity and native priors; target response values are loss targets
only. Main sampling and main losses remain unchanged.

    L_p = mean_d((Linear(mean_i ResponseCLS(control_slot_pi,p)) - target_p)^2)

Use the complete group in one differentiable forward and immediately backward.
An average of individual-cell target MSEs is a different loss and is forbidden.
The new Linear256->256 has ordinary random weight/bias initialization in an
isolated RNG stream, before native optimizer construction. Its weight uses native
Muon routing; its bias uses AdamW. There is no decoder or Teacher forward on this
auxiliary path. The Teacher retains an unused EMA mirror of the readout; no target
bank values enter it, and no extra EMA/center update is made.

## Exact update budget and calibration

From scratch, seed1, two epochs, 150 updates/epoch, global320=80x2 GPUsx2
accumulation. No per-epoch validation; retain/evaluate last. Do not warm-start a
trained reference. This adds 47,807 Student row-forwards and 1,322 group graph
builds per epoch, so it is not an equal-FLOP baseline comparison.

Hash-permute conditions each epoch. Position i maps to step i%150 and rank
(i//150)%world. Each condition appears once per epoch. Before native gradient
rank-mean reduction, clipping and the single optimizer step, add each local
group gradient with coefficient lambda*world*150/1322. Empty local group lists
are valid. This equalizes group coefficients, not fixed-theta gradients across
an epoch whose parameters change. Modulo scheduling is imbalanced (722/600
groups per epoch at world2); report measured throughput.

Freeze lambda once at initialization using exactly the first eight sealed native
epoch0 TRAIN batches and their assigned complete groups:

    lambda = min(1, .1 * median(G_prediction) / median(G_anchor_unit))

G is the norm on shared `student.response` parameters, excluding the readout.
Sum signed gradients across groups/microbatches and rank-mean across the full
world before taking a norm. Do not average rank norms. Zero/nonfinite gradients
fail without resampling. Calibration performs no optimizer/EMA/center update and
restores all protected state and RNG. It does not fit to validation metrics.

CUDA preflight must cover a complete40 group with main gradients resident,
native gradient reduction, checkpoint restore and zero-PKL closure. Capacity
failure aborts this configuration; do not silently split groups into per-cell
losses. Auxiliary RNG scopes include backward recomputation, preserving main
view/dropout clocks. Target/cache arrays are external, never checkpoint buffers.
Default-disabled execution keeps old parameter keys and initialization streams.

For this opt-in trial, the training instance also slices permitted columns before
reading `.X`, then scatters into the original-width zero array. Selected allowed
query values and global gene IDs are unchanged. Evaluation uses a separate,
unconfigured instance and preserves the full output axis. Older runs did exclude
the 225 columns from model inputs/labels, but their loader physically read them;
do not rewrite that history as a new leakage finding.

## Acceptance

Use the fixed val445 and original 300-control manifests, retain all18 native
Pearson axes and finite counts. Primary evidence requires improved ownmask
allowed-expression delta Pearson (paired condition-bootstrap lower95(delta)>0),
all three native seen-expression all-gene Pearson point estimates no lower,
native and condition-balanced ownmask MSE no worse, and improved pre-readout
ResponseCLS response-distance RSA with pair-union direct-target masking. Use
10,000 condition bootstrap draws, seed20261028; do not treat condition-pair
distances as independent observations. Baseline newly needed diagnostics may
reuse its checkpoint with zero updates; never retrain it.

Keep the same medium panel, same/different condition and context comparisons,
GG/GL/LL similarities, raw/normalized effective rank, relative variance and
prototype entropy/JS/CE/KL. A lower anchor loss, higher rank, sharper prototypes
or better label separation alone does not pass. Small fixed probes cannot
represent complete TRAIN groups: report anchor loss/gradient as unavailable,
not zero. Add a frozen condition-derangement negative control. Genuine transfer
to the 225 unseen output columns still needs an independent column-holdout
confirmation; do not use their test labels to redesign this trial.

Report actual training/evaluation SHA, target-bank preparation SHA and hashes,
configuration, checkpoint, populations, extra compute and cross-version limits.
The overall scientific problem remains open until joint task-relevant gains are
demonstrated. Failed hypotheses trigger evidence-led redesign, not success claims.

## Endpoint and identity clarification before fitting

For primary val RSA, collect the pre-readout, post-final-RMSNorm ResponseCLS from
the native prediction query blocks. Weight each block by its actual gene count
divided by the full5,000 output-gene count, then average across the same300
controls. Here there are five1,000-gene blocks with equal1/5 weights. Apply no
additional normalization. This block-pooled endpoint is distinct from a medium
probe's single1,000-query CLS; do not combine their rank/variance statistics.
Use identical definitions for the retained R3 checkpoint and the candidate.

The self-contained configuration seals both bank-manifest and bank-array SHA256.
Training verifies the array bytes. Evaluation reconstructs the same identity and
readout architecture from configuration, without loading TRAIN target/cache
arrays. This avoids an identity mismatch between training and evaluation while
keeping scientific arrays server-only and outside evaluation input consumption.
The unit-weight calibration leaf is an engineering input, not a fit permission
or a second hyperparameter arm. Its first-eight BF16 calibration produces the
single final loss coefficient; publish a separate immutable fit configuration.

## Frozen-checkpoint analysis entry points

`scripts/v2/response_anchor_postfit.py run --spec FILE --spec-sha256 SHA`
uses the original endpoint's native evaluation package in separate GPU workers.
The same val445/300-control forward produces the native18 metrics, allowed-axis
own-target-masked prediction scores and block-pooled ResponseCLS. The pure NumPy
`gradpert.evaluation.response_geometry` module implements condition-paired
bootstrap with reranked weighted distance ties, not independent pair bootstrap.
The original R3 training source/checkpoint and the analysis source are recorded
separately. Private endpoint arrays and row identities remain on the server.

`scripts/v2/response_anchor_medium.py seal|medium|panel|validate --spec FILE
--spec-sha256 SHA` reuses the original medium population (1335/5317, deep128/506,
gradient32/125) and matched128 panel, without family/queue registration. It
installs an instance-scoped allowed-column-first reader before TRAIN diagnostic
reads, including a guarded fallback for the original R3 package. Partial probes
mark the complete-group anchor term unavailable. Gradient probes observe
derivatives but never update optimizer, Teacher or centers.

Parallel analysis workers must bind an exact live common dispatcher identity
and verify ancestry before native resource admission. No foreign-job whitelist
or memory rule changes. Local synthetic tests do not replace the first-condition
native prediction/observer parity or the medium reference/fast parity gates.
Analysis code is published independently of the immutable active fit checkout;
postfit-stage completion never implies overall scientific acceptance.
