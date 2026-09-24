# Lessons

## Final-epoch reporting needs the actual final checkpoint

- User correction 2026-09-10: automatically report both best-validation and
  final-epoch test results for R50. Best-only cleanup made historical LR-low
  final unavailable. Never fabricate it, relabel an intermediate archive, or
  silently rerun training. Preserve best/last before test loading or cleanup,
  bind each role to epoch/step/source/config/checkpoint hashes, and explicitly
  alias identical final/best checkpoints. Smoke remains validation-only.

## Loss coefficients require a normalization and gradient rationale

- User correction (2026-09-09): the redesigned experiment program should not
  begin with arbitrary coefficients or a blanket remove-one-loss matrix.
  Preserve the original E3 prediction/condition/masked/spread weights1/.8/.4/.1
  for training-recipe selection. MSE, soft-target CE and log-distance penalties
  have different scales and reductions; equal coefficients do not imply equal
  influence, and raw loss magnitudes do not measure gradient contribution.
- Before proposing changed weights, inspect reductions and training-only
  weighted gradients on shared parameters, state the hypothesis, and freeze
  the comparison before validation results. Do not claim optimality from an
  old test ranking or a single batch. See R50_E3_REDESIGN.md.

## Instantiate the complete model before claiming a capacity ratio

- Correction (2026-09-09): the proposed 128-embedding, two-layer/two-head128
  graph, projector256/32, basal/decoder128 design was incorrectly estimated
  at 3.82M parameters. CPU module instantiation counted 6,338,568 (49.417:1
  against 128,266 training perturbed cells), including both graph sources
  and both Student and Teacher. It does not satisfy the agreed 20--30:1.
- Prevention: count every instantiated parameter, show component subtotals,
  include frozen Teacher parameters, and distinguish budget estimates from
  verified counts before changing configs or claiming a design fits.

## A changed baseline must reverse dependent ablations where necessary

- Requirement correction: the successor A0 changed from eight to four local
  views. Merely editing A0 would make the old four-local L2 identical to the
  reference and would silently destroy the single-factor matrix.
- Correct migration: regenerate every successor row from the new A0, redefine
  L2 as eight locals, derive proportional masks as `2/4` and `1/4`, advance the
  matrix identity, and invalidate old config/run hashes without rewriting the
  old evidence.
- Prevention: after any baseline change, compute each row's resolved parameter
  diff against the new baseline and require it to equal the declared allowlist.
  Also audit derived quantities and every secondary config generator that uses
  A0 as its base.

## Cross-profile attribution should choose the smallest measured boundary

- Mistake avoided: static inspection suggested both scalar GPU synchronization
  and sparse-union construction, but either could have been too small to matter
  on the real A0 path.
- Correct evidence: the Torch profile found only about 14 ms of
  `_local_scalar_dense` across three profiled steps, while an independent
  five-step cProfile attributed 52.829 seconds cumulative to 342
  `build_sparse_union` calls. A real first-batch CPU microbenchmark then showed
  exact tensors with median union preparation falling from 8,276.7 to
  2,310.5 ms.
- Prevention: require a stage profile, a second profiler at the suspected
  implementation boundary, and an exact real-shape microbenchmark before
  editing. Optimize only that boundary, preserve a same-commit reference path,
  and reserve end-to-end claims for serial same-GPU ABBA.

## Linux capacity checks must use reclaimable available memory

- Mistake: the CUDA preflight used `SC_AVPHYS_PAGES`, which measures currently
  free pages. Hashing multi-gigabyte immutable inputs filled the Linux page
  cache and made a host with about 241 GiB reclaimable memory appear to have
  only 2--4 GiB available.
- Correct evidence: `/proc/meminfo` `MemAvailable` is the kernel estimate for
  memory that can be allocated without swapping; `MemFree`/free pages exclude
  safely reclaimable cache.
- Prevention: Linux capacity gates use `MemAvailable` first, retain the POSIX
  free-page probe only as a fallback, and test both paths. A failure before
  native/model construction must be receipted as preflight failure and must
  not be validated as if native identity files already existed.

## Stage memory evidence must follow retained autograd lifetime

- Mistake: aggregate step memory and static call counts could not explain a
  first-step OOM in a model that accumulates many independent local forwards
  before either backward path begins.
- Correct evidence: record allocated/reserved memory at nested phase
  boundaries. In the RingInduced A0, each completed eight-condition local index
  retained about 8.47 GiB, while Student globals retained about 5.44 GiB. The
  fourth local index then failed, directly identifying activation lifetime as
  the capacity blocker.
- Prevention: capacity probes emit ordered entered/completed/failed events for
  every global, local index and condition view. Nested observers use a stack;
  a single active-stage scalar rejects valid outer completions after an inner
  event and corrupts the failure receipt.

## Checkpointing stateful graph forwards requires private buffer state

- Mistake: saving and restoring shared BatchNorm buffers around checkpoint
  recomputation either applies running-stat updates twice or mutates tensors
  that autograd saved, causing a version-counter failure.
- Correct implementation: run the original local forward through a functional
  module with private working buffers, mirror its one update into real buffers
  without placing that mutation on the autograd dependency path, and recompute
  from separate pre-view buffer clones. Preserve the checkpoint RNG state and
  the original independent view order.
- Prevention: exact-effect tests compare every output, loss, gradient,
  optimizer state, Student and Teacher parameter/buffer state, centers and RNG
  after a complete step. An output-only checkpoint test is insufficient.

## Receipt producers and validators must share one exact predicate schema

- Mistake: the P1 worker emitted the valid
  `publication_receipt_equals_p0` source predicate, but the terminal validator's
  exact allowlist omitted it. After the import path was repaired, the validator
  still rejected every legitimate terminal receipt as malformed.
- Correct evidence: a replay with repository root plus `src` on `PYTHONPATH`
  reaches the validator; a synthetic copy of the preserved exact CUDA-OOM
  receipt with the unrelated old observer failure removed is then classified
  as `capacity_failed` with exit code 10.
- Prevention: derive producer and validator keys from one shared constant and
  keep an end-to-end replay test. Unit fixtures must include every field the
  real producer emits; a self-consistent reduced fixture can hide schema drift.

## Local telemetry must not depend on a long-lived archive transport

- Mistake: local-private Trackio mode repeated a live Hugging Face Bucket
  preflight when each queued row started. The reverse SOCKS endpoint expired
  during the long formal queue, so H1/H2 lost local scalar capture even though
  their native training receipts completed normally.
- Correct boundary: local scalar collection and remote archival are separate
  best-effort stages. A sealed launch-time private-Bucket receipt can bind the
  intended destination, while a later transport outage must be receipted as an
  archive failure rather than preventing owner-only local collection.
- Prevention: before the next tracked formal lineage, make the sidecar consume
  a hash-pinned launch preflight, test proxy loss between rows, and defer the
  live Bucket operation until post-row archival. Never replay this completed
  lineage to manufacture missing telemetry.

## Capacity policy must enumerate every authorized non-sentinel row

- Mistake: a fresh M capacity queue used the bounded performance worker, but
  only H4 was registered as capacity-only. M4 ran because it belonged to the
  frozen sentinel; M1 then failed before claiming an attempt because it was
  absent from both allowlists.
- Correct boundary: keep the eight-row sentinel immutable and maintain a
  separate explicit allowlist for user-authorized M, W, O, and H4 capacity
  probes. These additional rows may run only `p1_capacity`, never timing or
  profiling stages, and remain `scientific_completion=false`.
- Prevention: before a multi-row capacity launch, test every exact variant ID
  through the same preclaim predicate used by the worker, plus negative tests
  proving non-authorized rows and all timing/profile stages remain rejected.

## One-step capacity probes must include degenerate graph views

- Mistake: the sparse Transformer capacity probe completed its first batch,
  but a later STRING-only condition produced a valid isolated-anchor local
  graph with one node. Training BatchNorm then failed because a singleton view
  has no estimable batch variance.
- Correct boundary: independent graph views keep their own normalization order.
  A singleton view uses the BatchNorm layer's existing running statistics
  without updating them; ordinary views with two or more nodes retain the
  original BatchNorm path exactly. Do not merge views or duplicate nodes merely
  to manufacture normalization samples.
- Prevention: graph-encoder integration gates must include connected,
  disconnected and isolated perturbation anchors beyond the first frozen
  batch. Test singleton forward/backward finiteness and unchanged running
  means, variances and batch counters.
# Post-fit scheduling

- User correction (2026-09-11): do not require an entirely idle fixed GPU for
  post-fit testing. The legacy R50 queue delayed LR-mid despite free GPU0.
  Chain tests directly after successful training, retain memory safety checks,
  and use an explicit nonduplicate handoff for already-waiting evaluations.

## Normalize Norman single-perturbation condition keys before interaction fits

- Mistake (2026-09-23): the first dataset-only interaction audit looked for
  singles by bare gene name, while canonical Norman stores them as `GENE+ctrl`.
  It silently returned zero fitted double conditions although measured singles
  and doubles were present.
- Prevention: parse condition components, map every `GENE+ctrl` single to its
  gene component, and require a positive known double count in the integration
  check. Do not interpret a zero-count scientific result before inspecting
  canonical condition strings and expected coverage.

## Condition-level counts do not guarantee within-batch comparison coverage

- Finding (2026-09-23): in the five-dataset audit, four single-gene datasets
  had many cells per condition overall, yet only 3–20 conditions per dataset
  had at least 20 perturbed and 20 control cells in one common batch.
- Prevention: report the eligible condition denominator for each matched-batch
  distribution test. Do not extrapolate significance fractions from this tiny
  subset or replace it silently with unmatched pooled controls.

## Never build in an active source-identity snapshot

- Mistake (2026-09-23): `python -m build` was run from the same server checkout
  used by an active v2 capacity probe. It created ignored
  `src/gradpert.egg-info/` after the probe's startup source check. Git remained
  clean, but the project's content-tree hash changed; the 32-update partial
  run was stopped and explicitly audited rather than used as capacity evidence.
- Prevention: run builds in a disposable clean clone or stage the source into a
  separate build directory. Keep every active training checkout read-only;
  recheck both Git status and content-tree SHA before accepting a run receipt.

## Advance the authorized dependency after a monitor confirms completion

- Context: the scheduled v2 batch-128 capacity check verified a passed 128-step
  receipt while the overall request still required recording the result and
  preparing the formal Jurkat baseline.
- Mistake or misunderstanding: I reported the completed test and listed its
  successor as a future step instead of executing the already authorized
  documentation and preflight handoff in the same turn.
- Correct understanding and evidence: a successful stage receipt changes the
  active work set; it does not complete the overall outcome. A newly discovered
  repair is another dependency to track before returning to the original path.
- Prevention rule: after verifying a stage, immediately execute the next
  authorized ready action, then update the existing monitor's target and
  cadence. Also update the project's Byte state/entry documents so a future
  scheduled run sees the current stage instead of an older status snapshot.
  Stop only at a genuine external wait, unresolved blocker, or completed
  overall outcome.
- Status: active
# Ratio descriptions must name their scope

- Mistake: interpreted the requested 2:1 as a Cell-versus-Response depth ratio,
  then as an uneven KDA/DSA allocation between encoders.
- Correction: the user specified the same internal two-KDA/one-DSA-MLA sequence
  in **each** encoder. The old configuration was three KDA plus one terminal
  DSA/MLA in each encoder.
- Prevention: write layer sequences per module and compute total depth before
  giving parameter estimates or editing architecture configs.

## Resolve B0 labels against the user's current experiment meaning

- Mistake (2026-09-24): I initially treated “launch B0” as the historical
  two-row generator group (`prediction_only` plus `joint_ssl`) and generated
  both local candidate configs. Neither was published or launched.
- Correction: the user specified that this B0 is the **complete** compact v2
  baseline with prediction, SSL1 and SSL2; `prediction_only` is outside the
  authorized formal launch. The tentative files were removed.
- Prevention: for reused experiment labels, inspect the current run lineage
  and apply the user's latest explicit definition before creating a formal
  queue or allocating GPUs. Keep a separate run ID for each scientific model.
