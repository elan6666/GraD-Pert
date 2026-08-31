---
id: 033
status: in_progress
wave: 6
depends_on: [028, 032]
updated_at: 2026-08-31T16:30:00+08:00
---

# Plan 033 — four-local A0 performance engineering and H execution

## Objective

Freeze the new four-local A0 and its direct single-factor matrix, measure the
real single-GPU execution path, implement only exact-effect optimizations
supported by those measurements, and then run A0/H1/H2/H3 with row-level
parallelism across at most two physical GPUs. L is configured but does not run.

## Scientific coordinate

- A0: Nadig Jurkat HVG512 plus all targets, RingInduced, local node ratio
  `1/2`, four locals, anchor-mask-view ratio `0/1`.
- H1/H2/H3 change only requested HVG count to 1024/2048/5000; local count
  stays four and local node budget remains ratio-derived.
- L1 stays Fanout; L2 becomes eight locals; L3 stays node ratio `1/4`; L4 and
  L5 retain mask ratios `1/2` and `1/4`, resolving to `2/4` and `1/4`.
- All M/W/D/E/O rows inherit the four-local A0 so their declared scientific
  differences remain unchanged.

## Performance protocol

1. Publish one clean commit with matrix ID
   `nadig_jurkat_vnext_ratio_graph_v3`; bind local, GitHub and a fresh server
   checkout to that exact commit and branch ref.
2. On one idle RTX 5090, run a bounded training-only four-local A0 capacity
   pass and stage/Torch/Python profiling. No validation or test reader opens.
3. Attribute step wall, GPU busy/idle intervals, view construction, sparse
   union, transfers/synchronization, Teacher, Student global/local,
   prediction, backward/update, data input and logging. Select only a measured
   material bottleneck.
   For selective checkpointing, test checkpointed-local counts `4,3,2,1,0`
   until the minimum passing count is established; do not assume a capacity
   result from the old eight-local lineage.
4. Preserve a same-commit reference path. Exact-effect gates compare views,
   ordered edges/unions, RNG, forwards, losses, every gradient, optimizer,
   Student/Teacher buffers and parameters, centers and prediction content.
5. Accept an optimization only after serial same-physical-GPU ABBA with fresh
   roots and identical batches/instrumentation. Primary acceptance is lower
   step wall without p90 or memory regression; utilization is diagnostic, not
   a substitute for throughput or exactness.

## Measured checkpointing target

- The sealed four-local capacity pass completed six training-only steps with
  10.035/10.138 GB peak allocated/reserved memory and at least 22.812 GB free;
  receipt SHA
  `a6befedb276df507e393cb30b723208546b600ed7c3cdf4e14480c08239192d0`.
- The sealed Torch profile completed two warmup plus three profiled steps. Its
  2.914--3.080 s measured steps attribute 1.356--1.494 s to Student locals,
  0.526--0.682 s to view construction and 0.622--0.632 s to backward/update;
  receipt SHA
  `61e653a4274f241025ff7bb798b576c2bbc1664a19ef8e82bca0e8ef31367393`.
- The next implementation therefore exposes an ordered checkpointed-local
  count without changing any scientific factor. Fresh capacity attempts test
  `3,2,1,0` after the already-passed all-four reference, followed by complete
  exact-effect and ABBA gates before a formal H launch.
- The clean `0467372` capacity boundary completed counts three and two; count
  one failed before any optimizer step with a real CUDA OOM and count zero was
  consequently not launched. The minimum passing count is therefore two,
  sealed by receipt SHA
  `06bac22e5cc0fb6a3f34314bb9dfa68b349e7b67e2889c4fa2884b68ad36f1ab`.
- The exact-effect gate extends first-step evidence with prediction, complete
  gradient, optimizer, Teacher, center, combined model and CPU/CUDA RNG hashes.
  It runs the all-four reference and count-two candidate with deterministic
  algorithms on the same physical GPU before any timing acceptance.
- Count two passed exactness but was rejected by same-GPU ABBA because its
  paired median ratio was 1.026487 (2.65% slower); count four remains default.
- A fresh real Python profile selected repeated RingInduced edge and
  incident-node scans. The immutable source-aware incoming-edge index reduced
  exact 32-view construction from median 983.661 to 833.647 ms (15.251%).
- Deterministic CUDA equality passed at evidence SHA
  `2b4f241aba137cdd138f576ca20a27b1f954f2b59dcbf874ed2a53b8f177f088`.
  Same-GPU ABBA passed at SHA
  `7e82fe419da63a2d85b8786af6751acafbc0ddfd1caf2bc1488cba7cda1ea579`:
  paired ratio 0.848836, 15.116% and 452.540 ms lower wall, improved p90,
  identical peak GPU memory, zero retry/OOM/PKL and no truth access.

## Formal execution

- Only after performance acceptance, create a fresh H-only lineage.
- Run A0/H1/H2/H3 as independent one-GPU rows with at most two rows active.
  Use two hash-pinned queues only after per-row capacity checks; H3 may run
  alone when its larger graph requires isolated capacity.
- Launch one out-of-process private Trackio sidecar per formal row. It may read
  only allowlisted `train_steps.csv`, `validation.csv`, `run_meta.json` and the
  pre-test training receipt; performance phases never launch it. Treat the
  dashboard as provisional telemetry, not scientific evidence.
- Bind the private dashboard target to
  `elan68681/grad-pert-vnext-ablations` and its explicit private Bucket
  `elan68681/grad-pert-vnext-ablations-bucket`; creation remains blocked until
  the server is privately authenticated with Hub write permission. The future
  hash-pinned queue launcher owns sidecar start, shutdown and local receipt
  validation for each row.
- Each row uses exactly 10 epochs, 5,820 ordered steps, 10 validations, seed 1,
  no early stopping, one test evaluation from `best.pt`, exact three metrics,
  `metrics_only`, zero persistent PKL and only `best.pt`.
- Never launch L or another module without new user authorization.

## Acceptance criteria

- Generated 25-row configs have exact hashes, no stale old L2 path, and every
  row differs from A0 only by its declared factor.
- Local and exact-commit server pytest/Ruff/format/mypy/build gates pass.
- Four-local capacity/profile and exact-effect/ABBA evidence are complete and
  truth-free; old eight-local numbers are labeled historical only.
- Trackio pre/postflight proves private Space/Bucket identity and exact formal
  source/run identity. Its local client attempts to enqueue 5,820 train scalar
  points and 10 validation points while uploading no test metric or artifact;
  because Trackio 0.37 delivery is best effort, the dashboard receipt remains
  provisional with `remote_sync_verified=false`. A dashboard failure does not
  alter scientific status.
- The private Bucket and owner credential are ready, but private Space creation
  returned `402 Payment Required`; never replace it with a public Space absent
  explicit user authorization.
- A0/H1/H2/H3 share exact split and ordered 300-control/truth hashes and pass
  their complete formal receipt validators.
