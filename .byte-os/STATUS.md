---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: execution
current_workflow: byte-auto
next_workflow: byte-auto
review_verdict: pass
iteration_count: 1
harness_status: ready
hard_blocked: false
updated_at: 2026-09-03T11:30:00+08:00
---

# Status

- Plan 035 is complete. Exact source `75a2c2b` finished formal D3--D6; every
  row passed 5,820 contiguous steps, ten validations, one best-checkpoint
  evaluation, three finite metrics, identical canonical split/control/truth
  identities, zero PKL and best-only retention. Neither Transformer fusion nor
  perturbation width 256 improved all three metrics consistently, so additive
  A0 remains the default without an equivalence claim.
- The user reauthorized continuation of the prior ablation sequence. The next
  module is a fresh L-only lineage for L1--L5; interrupted Plan 034 L roots stay
  sealed and must never be resumed or relabeled. M remains behind the L module
  barrier.
- Plan 035 superseded the old Plan 034 queue and is now complete. The earlier
  L1/L2 roots remain sealed as user-interrupted; L3--L5 and M1/M2/M4 did not
  start in that lineage. Completed E and D evidence is immutable. The next
  formal priority is the newly authorized fresh L1--L5 lineage.
- D3/D4 use 64-wide perturbation states with direct `concat(b,p)` and
  `concat(b,p,T([b,p]))`; D5/D6 repeat those fusion choices with 256-wide
  perturbation states. For D6, only the Transformer token path applies learned
  `256 -> 64` projection, while the decoder retains raw `p256`. The scientific
  matrix is now 29 rows; the historical eight-row performance sentinel remains
  unchanged and treats the four new rows as unselected.
- Plan 034 is active. The user authorized formal E, D, L and M execution in
  that exact module order under the completed four-local A0 coordinate. Rows
  inside one module may use at most two GPUs, but a hard module barrier blocks
  D until E passes, L until D passes and M until L passes.
- The requested GenePT prior label `Protein+Reactome+SIGNOR` is the exact
  existing GenePT-Seed `protein-pathway` / `Seed-GO-ProteinPathway` artifact,
  not a second embedding. Server evidence binds 17,730 by 2,048
  `doubao-embedding-vision` vectors at SHA
  `34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`.
  A fresh Nadig Jurkat coverage preflight remains required before E launch.

- Plan 033 is complete. Exact source `845c10a` finished formal four-local
  A0/H1/H2/H3. Every row passed 5,820 contiguous steps, ten validations, one
  best-checkpoint evaluation, exact three metrics, identical canonical
  split/control/truth identities, zero PKL and best-only checkpoint retention.
  The single-seed H point estimates are mixed and do not replace A0 or establish
  equivalence. L remains paused.
- Trackio is explicitly partial for this lineage: A0/H3 owner-only local stores
  completed, but private Bucket archives failed; H1/H2 local clients were
  blocked by the expired reverse SOCKS endpoint. Native scientific status and
  receipts remain authoritative and complete.
- All `f1c14d8` formal queues are stopped by user request. GPU0 A0 is preserved
  as interrupted at 1,094/5,820 steps; GPU1 H3 is preserved as interrupted at
  1,874/5,820 steps. Neither is failed or scientifically complete, no L row
  started, both GPUs were released, and the run root retained zero persistent
  PKL. The old automation was deleted so it cannot relaunch the superseded
  coordinate.
- Plan 033 replaces Plan 030. The active A0 now has four RingInduced local
  views at ratio `1/2`; the successor matrix began as the 25-row
  `nadig_jurkat_vnext_ratio_graph_v3` identity and Plan 035 advances it to the
  29-row `nadig_jurkat_vnext_ratio_graph_v4`. H/M/W/D/E/O inherit four locals. L1 is
  Fanout with four locals; L2 is the direct eight-local count ablation; L3
  keeps four locals at ratio `1/4`; L4/L5 resolve mask ratios to `2/4` and
  `1/4`. L execution remains paused.
- The accepted eight-local sparse-union implementation remains valid history,
  but its 56.253% result is not a four-local speed claim. Fresh four-local
  profiling selected repeated RingInduced edge/incident scans. At `1fc1576`,
  the immutable source-aware index passed exact CUDA state equality and serial
  same-GPU ABBA: paired ratio 0.848836, 15.116% and 452.540 ms lower step wall,
  improved p90, identical GPU memory, zero retry/OOM/PKL and no truth access.
  Formal scope is A0/H1/H2/H3 with at most two independent single-GPU rows;
  L remains paused.
- The v3 generator is deterministic and the integrated local surface passes
  544 tests with 2 honest skips under the locked dev/data/model/tracking
  extras, including 21 Trackio sidecar tests and all prior migration gates.
  Ruff, format on 255 files, compileall and diff checks pass. Strict mypy
  passes on all 76 source files and the isolated wheel/sdist build passes.
  Exact-commit `1fc1576` server gates pass 578 tests with one honest frozen
  skip, Ruff, format on 256 files, strict mypy on 76 files and isolated build.
- Private Hugging Face Trackio support is implemented as an optional,
  out-of-process formal-only sidecar. It exposes allowlisted loss,
  validation, stage-throughput and one-GPU telemetry curves while excluding
  test metrics and every artifact/data surface. It is disabled for all
  capacity/profile/exact-effect/ABBA lineages. Twenty-one focused sidecar tests
  pass. The selected private targets are Space
  `elan68681/grad-pert-vnext-ablations` and Bucket
  `elan68681/grad-pert-vnext-ablations-bucket`; owner authentication and the
  private Bucket are verified, while private Space creation returns `402
  Payment Required`. No public substitute is authorized. The user chose to
  continue with owner-only local Trackio stores archived to the private Bucket;
  receipts must state that live Space sync is unavailable.

- Exact-effect performance engineering is accepted at clean commit `7332cc1`.
  The deterministic CUDA hard gate is exact for non-timing metrics, views,
  unions, every gradient, Student/Teacher and optimizer state, centers and
  CPU/CUDA RNG. Serial one-GPU ABBA (5 warmups + 20 measured steps per arm)
  reduced paired median step wall from the reference 11.167/11.082 seconds to
  4.842/4.891 seconds: ratio 0.437470, 56.253% reduction, 2.286x speedup and
  6.258 seconds saved per step. Both p90 pairs improved; peak allocated and
  reserved GPU memory increased only 0.017% and 0.041%, with zero retry/OOM,
  zero PKL and no validation/test access. Plan 028 is complete. Plan 030 was
  the next authorized work at that coordinate but is now superseded by Plan
  033.

- The clean `a7beebf` eight-row sentinel is sealed: A0/H3/L1/L2/D2/E2 pass
  one complete training-only step; M4 is an authorized CUDA-OOM capacity
  failure and W1 an authorized post-step memory-headroom failure. The A0 Torch
  profile records 11.3--12.0 seconds per warm/profile step, with Student locals
  at 7.9--8.43 seconds (about 68--70%). The independent cProfile attempt
  completed five steps and attributes 55.481 seconds cumulative to 345
  `_batched_sparse_union` calls and 52.829 seconds to 342 `build_sparse_union`
  calls. Both profiles are truth-free performance evidence with zero PKL.
- The isolated candidate now replaces only sparse-union preparation with an
  exact ordered CPU-vectorized implementation and keeps a same-commit
  `reference` selector. Server Torch tests prove bit-exact union tensors and a
  complete first-step trajectory including non-timing metrics, views, RNG,
  every gradient, model/optimizer/Teacher state and centers. On the real A0
  first batch (8 conditions, 66 views, 1,404-node locals), exact CPU union
  preparation fell from median 8,276.7 ms to 2,310.5 ms (3.58x, 72.1%). Full
  candidate gates pass 523 server tests with one honest frozen-reference skip,
  Ruff, format on 247 files, strict mypy on 74 source files and an isolated
  build. The later clean-commit CUDA exact-effect and serial ABBA gates passed
  as recorded above.

- The clean `b37963e` eight-row sentinel is sealed as failed. A0 entered its
  first Student-local phase and raised a real CUDA OOM after about 30.65 GiB
  was allocated; H3 was stopped by the peer-failure gate. Source remained
  clean, both GPUs were released and the whole lineage retained zero PKL.
- Preserved stage telemetry localizes the capacity blocker: about 5.44 GiB
  after Student globals, 13.92/22.40/30.87 GiB after local indices 0/1/2, then
  OOM while entering local index 3. The roughly 8.47-GiB retained increase per
  local index is activation-state growth, not allocator fragmentation.
- The isolated successor implements non-reentrant activation checkpointing for
  Student local forwards only. It preserves ordered independent forwards and
  RNG, and isolates stateful Exphormer BatchNorm buffers so recomputation does
  not apply a second running-stat update. A full synthetic reference-versus-
  optimized step is bit-exact for non-timing metrics, gradients, optimizer,
  Student/Teacher state including buffers, centers, RNG and first-step health.
- The nested stage observer now maintains an active-stage stack, and future
  queue validators must include both repository root and `src` in their import
  closure. A prelaunch replay additionally found and repaired an exact-schema
  mismatch: the worker emitted `publication_receipt_equals_p0` but the terminal
  validator allowlist omitted it. The corrected validator classifies an exact
  CUDA allocator OOM as capacity evidence only when observer/source/input/PKL
  gates are intact. Current gates pass 437 local non-notebook tests with 10 honest
  dependency/reference skips and 518 server tests with one honest frozen-
  reference skip; Ruff/format pass, strict mypy passes on 74 source files and
  the isolated build succeeds. Fresh exact-commit publication, CUDA capacity/
  exact-effect/profile and ABBA timing remain pending.

- The user replaced the 25-row CUDA performance census with an exact eight-row
  representative sentinel: A0, H3, L1, L2, M4, W1, D2 and E2. The scientific
  matrix remains 25 rows; formal A0/H1--H3/L1--L5 remains 10 epochs and starts
  only after exact-effect optimization review.
- Preserved P1-v4 stopped before native/model construction. Its A0 resource
  gate falsely used Linux free pages after multi-gigabyte input hashing (about
  2.2 GiB) while `MemAvailable` was about 241 GiB. The worker and profiler now
  share a `MemAvailable`-first probe, and resource-preflight failure has a
  dedicated receipt path that does not require nonexistent native identities.
- The eight-row sentinel and the two infrastructure repairs pass 84 focused
  local tests. Full repository gates, review, publication, fresh server P0/P1
  lineage and real CUDA profiling remain pending.

- The user replaced every E-row `emb_b` dependency with the GenePT-Seed
  `Seed-GO-ProteinPathway` master artifact (SHA
  `34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`).
  Plan 032 selects the exact runtime axis from its 17,730 nonzero 2,048-wide
  vectors; the old 17-missing-target skip no longer describes this lineage.

- The Plan 031 25-row CUDA census is superseded by the eight-row performance
  sentinel. Its 25-row CPU-only static matrix/graph/GenePT validation remains
  useful prerequisite evidence. Short runs remain performance evidence only.
- Plan 031's CPU-only launch closure is now implemented in the isolated
  branch. A static P0 tool binds the exact 25-row matrix, clean source content
  tree, canonical/source/split identities, all four nested H axes, every
  resolved rational local-view contract, and the one Seed-GO-ProteinPathway
  preflight without importing CUDA or constructing model/evaluation objects.
  A separate expression-free manifest freezes exactly the first 110 semantic
  batch identities (ordered rows, conditions, sampled controls and gene-name
  anchors) from the real seed-1 batch scheduler.
- Every census worker now verifies the hash-pinned P0 and batch manifest before
  claiming an attempt root, rejects a batch before its optimizer step on the
  first identity mismatch, preserves append-only preclaim and in-run failures,
  requires P2/profile to cite a completed P1 receipt on the same physical GPU,
  rehashes every sealed data/graph/GenePT input before and after native
  execution, rechecks the clean published GitHub source content tree after the
  run, and requires native identity receipts plus whole-attempt zero PKL.
  Aggregation independently revalidates the same lineage, final immutable-input
  evidence and per-step batch SHA values. P0 and batch-manifest destinations
  are atomically claimed with exclusive creation so concurrent attempts cannot
  overwrite one another.
- Current local gates for this uncommitted integration are 83 focused
  performance/data tests and 413 non-notebook tests with 10 honest local
  dependency/reference skips; Ruff, format on 158 files, compile checks and
  whitespace checks pass. Exact-commit server pytest/mypy/build gates remain
  required before live P0 generation.
- The generic bounded census worker supports only P1 capacity, P2 timing
  and the separate diagnostic profiler. It preserves the ordered attempted
  batch even when the first native step fails before returning, distinguishes
  attempted batches from completed steps, emits truth-free/non-scientific
  evidence, and never exposes P3 or a formal epoch entrypoint.
- The first exact A0 capacity process at clean commit `454ab1ff` passed source,
  config, data, split, graph, idle-GPU, RAM and disk preflight but failed before
  returning step 0: PyTorch had allocated about 30.65 GiB on the 31.36-GiB
  RTX 5090 and a 30-MiB allocation failed. No validation/test truth was opened;
  the receipt/log are immutable. This makes stage-level memory bisection the
  first measured engineering task.
- Plan 029 is complete: local and server gates passed at `454ab1ff` with 396
  tests and one honest frozen-reference skip, Ruff, format on 235 files,
  strict mypy on 73 source files and isolated package build.
- Historical dependency chain Plan 032 -> Plan 031 -> Plan 028 -> Plan 030 is
  complete through performance acceptance but superseded before formal
  completion. Plan 033 alone governs the active four-local performance and
  A/H-only program.

- Plans 029 and 028 describe the prior eight-local successor. Their code and
  evidence remain preserved, while the active v3 A0 is HVG512+targets on the
  sealed 2,809-node graph, RingInduced locals at exact ratio `1/2` (floor cap
  1,404), four locals and mask-view ratio `0/1`. H varies only requested HVG
  count 512/1024/2048/5000; L rows remain configured as direct single-factor
  comparisons but are not authorized to run.
- Plan 029 implementation is integrated in isolated branch
  `codex/vnext-performance`: exact rational config fields, runtime B/M
  resolution and receipts, generalized L=4/8 consistency loss, generic H graph
  materialization, new 25-row matrix, legacy v1/pilot compatibility, a
  training-only bounded A0 profiler, runtime schema-v2 semantic enforcement,
  pre-model anchor-capacity validation and persisted realized-view evidence.
  Primary-agent local gates pass 366 tests with 10 honest dependency/evidence
  skips when notebook tests are excluded; Ruff, format on 156 files,
  compileall, deterministic config regeneration and diff checks pass. The
  local interpreter lacks Torch, anndata, nbformat, mypy and build; full server
  Torch/anndata/nbformat/strict-mypy/build gates remain required before
  publication is accepted.
- Performance selection remains deliberately open. The old Fanout preindex
  benchmark is retained only for L1. The first server artifact will be the
  unoptimized RingInduced A0 capacity/profile receipt; no sparse-union or Ring
  optimization is authorized until a measured stage contributes at least 10%
  and 100 ms per step (or causes the capacity/memory gate).
- Superseded 42e/8221/276d run evidence remains immutable and cannot satisfy
  the new coordinates. No successor 10-epoch queue has launched.

- Plan 022 is in local implementation. The user replaced the historical
  effective four-term B2 loss weights `1.0/0.1/0.1/0.01` with explicit formal
  weights `1.0/0.8/0.4/0.1` for prediction, condition consistency, masked-node
  consistency, and spread. All five formal configs and the fail-closed schema
  encode the new values; the runtime composes gradients from the directly
  weighted terms and records the effective mapping. Historical pilot configs
  and completed results retain their old identity. No new training has been
  launched, and any future run requires a new synchronized commit/namespace.
  Local gates passed 195 tests with 9 honest dependency/receipt skips, Ruff,
  format, all 30 config verifications, and isolated build. Strict mypy and the
  Torch-backed objective test remain exact-environment server gates because the
  local environment has no Torch/PyG.
- Plan 021 is in implementation at the local successor of `b6e3854`. The user
  selected full-graph, all-seven-systems B2 as the default native profile and
  restored the full ceiling to 200 epochs with validation-only early stopping
  `patience=10`. All five formal native configs now encode that profile and
  remain `metrics_only`; schema/trainer/matrix tests enforce the new budget and
  a synthetic full run stops after exactly 10 consecutive non-improving
  validations. Local focused gates passed 19 tests with one honest Torch/PyG
  skip plus Ruff/format; full local pytest/build remain dependency-limited and
  will run on the exact synchronized server commit before execution. The next
  server execution is Nadig Jurkat seeds 1--4 on the one frozen canonical
  split, preceded by a fresh exact-commit seed-1 one-epoch B2 integration gate.
  Goal mode remains inactive for long training.
- Plan 020 execution and strict review are complete at clean synchronized
  commit `ddf40fd`. B2 systems-only and B3 combined each completed exactly 10
  epochs/5,820 ordered optimizer steps with 10 validations, no test truth
  during fit, one final test evaluation, one retained `best.pt`, and zero PKL.
  The 71-check verifier passed. B3 trained in 5,223.127 s versus B2's 7,221.773
  s: 1.383x faster and 27.68% less wall time. B3-minus-B2 metric deltas were
  -0.007801, -0.002219, and -0.019724 for TxPert, TriShift, and Systema;
  these are descriptive and do not establish effect equivalence. The runs used
  separate GPUs but shared host CPU, RAM, and storage, so their absolute timing
  is not mixed with the earlier sequential one-epoch pilot. The evidence and
  documentation commit `8aa7a90` passed 217 server tests with 3 honest skips,
  Ruff, format, strict mypy on 66 source files, isolated build, public push,
  and clean server synchronization.
- Plan 019 is delivered at clean public commit `167e31a`. Exact-commit server
  gates passed 215 tests with 3 honest prepared-receipt skips, Ruff, format,
  strict mypy on 66 source files, isolated wheel/sdist build, and clean-tree
  verification. No further B0 execution is authorized or required.
- Plan 019 execution and strict validation are complete. The fresh B0 timing
  coordinate ran exactly once at clean commit `7bed1f0`, used the canonical
  6,506-node/222,654-edge graph, disabled all seven systems groups, completed
  one epoch/582 steps, evaluated once, retained only hash-pinned `best.pt`, and
  left zero PKL. Its 44-check strict verifier passed; receipt SHA-256 is
  `acc60de269b85e16b6164a1bd4035acc869ca2a62b183d32f09d457d18e63920`.
  Historical c240 B0 remains untouched with the same manifest SHA-256.
- The exact 30-coordinate one-epoch matrix is complete and sealed across two
  explicit lineages: retained `c240157` GraD-Pert/nonlearned results and
  `2bf2771` GEARS/TxPert results. The reviewed small stages contain 195 and 325
  selected files; the combined audit passed all five cross-model fairness
  identities and all 15 learned checkpoint hashes. Retained c240 coordinates
  honestly predate `inference_recipe.json`; this limitation is recorded rather
  than backfilled.
- Project-wide artifact policy and README delivery are complete at clean,
  public, synchronized commit `0b8d3c7`: every model and nonlearned baseline
  defaults to zero persistent PKL, while explicit `single_pkl` is the only
  opt-in. The authorized cleanup removed 126 non-active experiment PKLs
  (355.10 GiB) with a sealed server receipt.
- Plan 018 is complete. Its historical c240 B0 remains immutable; plan 019
  added a separate metrics-only timing coordinate rather than overwriting it.
  B1 changed only the graph axis; B2 enabled all seven semantics-preserving
  systems optimizations on the full graph; B3 combined them. Every timing
  coordinate was exactly one epoch and selection was speed-only.
- B1 local implementation is complete: a separate pilot config keeps all 5,000
  expression/output/evaluation genes, directly recomputes raw-data Top-500
  HVGs, requires exact equality with the frozen normalized-dispersion ranking,
  unions every candidate target, and re-prunes both public graph sources. The
  native runtime now accepts an independently sealed graph axis smaller than
  the expression axis and emits per-stage speed/memory receipts. No B2 systems
  optimization is enabled in the B1 config. Local gates passed 185 tests with
  9 honest dependency/receipt skips, Ruff, format, build, and diff check;
  strict mypy remains a server gate because the local venv lacks Torch/PyG.
- B1 server execution and strict validation are complete at synchronized clean
  commit `0a4d339`. It completed exactly one epoch/582 optimizer steps, used no
  test truth during fit, evaluated test once, retained only hash-pinned
  `best.pt`, left zero PKL, and matched B0's canonical/split/ordered-300-control
  and truth row IDs exactly. The runtime graph had 2,798 nodes and 89,561
  nonself edges while all expression/output/evaluation axes remained 5,000.
  After 10 warmup steps, 572 measured steps ran at 0.6931 steps/s and 152.31
  cells/s; one-epoch training wall was 844.180 s. The three one-epoch metrics
  are retained only as non-decisional evidence.
- B2 all-seven systems-only implementation is complete locally at `a9f47ff`
  on B0's full graph axis. It combines merged HDF5 fallback reads, an exact
  control-expression cache, background prefetch/pinned/nonblocking transfer,
  resident fixed graph tensors, validation expression caching, buffered logs
  with required flushes, and one checkpoint serialization plus reflink/copy.
  It also records first-step batch/control/tensor/view/loss/parameter/update
  evidence and runtime activation/fallback/timing/memory receipts. Local gates
  passed 190 tests with 9 honest dependency/receipt skips, Ruff, format,
  isolated build, and diff check. The synchronized server lineage through
  `a17b8e7` passed 213 tests with 3 honest prepared-receipt skips, Ruff,
  format, strict mypy on 66 source files, isolated build, and clean-tree
  verification. Its formal systems-only run later completed at `2e30fb5` and
  passed all 41 strict checks; the historical c240 B0 remained untouched.
- B3 completed at exact clean local/GitHub/server commit `44ae7ff`. Local gates
  passed 191 tests with 9 honest dependency/receipt skips, Ruff, format, and
  build; server gates passed 214 tests with 3 honest skips, Ruff, format,
  strict mypy on 66 source files, isolated build, and clean-tree verification.
  The formal run completed one epoch/582 optimizer steps, used no test truth
  during fit, evaluated test once, retained only hash-pinned `best.pt`, and
  left zero PKL. Its graph has 2,798 nodes/89,561 nonself edges while all
  expression/output/evaluation axes remain 5,000. The strict validator passed
  49/49 checks; validation receipt SHA-256 is
  `65cf90788dc6a213148b35de0685cb1216d50d127a19d21b1cd175c6801c4274`.
- The rebuilt speed-only comparison selects B3. Actual one-epoch training
  walls were B0 2,951.487 s, B1 844.180 s, B2 718.681 s, and B3 507.718 s;
  full-epoch wall throughput was 43.46, 151.94, 178.47, and 252.63 cells/s.
  B0→B1 isolates graph reduction without systems (3.496x/71.40% less time),
  B0→B2 isolates all seven systems groups on the full graph (4.107x/75.65%),
  B1→B3 isolates systems on the reduced graph (1.663x/39.86%), and B2→B3
  isolates graph reduction with systems active (1.416x/29.35%). Combined B3
  is 5.813x faster than the new B0 timing baseline. Comparison receipt SHA-256
  is `d4da6aac3a71cf3fcf2aba645d1c423fe1a4f52ae593a49e0f0361b0a20defe1`.
- The original warmup-excluded receipt throughput is retained but not used as
  actual wall throughput for prefetch-enabled B2/B3: its stage sum adds data
  read time to GPU step time even when those stages overlap. Selection uses
  monotonic `one_epoch_training_wall_ms` and throughput recomputed from the
  same full-epoch wall. The new B0 timing coordinate uses that same monotonic
  protocol. All three metrics remain explicitly non-decisional; no one-epoch
  unchanged-effect claim is made.

- The first d6f9 GEARS/K562 hard gate completed one epoch successfully, but
  strict artifact validation found frozen GEARS had retained two framework
  PKLs (`checkpoints/best/config.pkl`, 4.0 MiB, and
  `official_adapter/custom_split.pkl`, 10.5 KiB). The entire run/contract/log
  lineage is preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-d6f9-gears-framework-pkl`.
  No remaining external queue was launched. The local repair now hashes and
  removes those reproducible metadata files, retains only `model.pt`, and
  enforces a whole-run zero-PKL postcondition before success.
- User-locked artifact policy is now plan 017: all 30 self-contained configs
  explicitly default to `metrics_only`, retaining the best checkpoint, complete
  inference recipe, exact ordered control/truth row IDs, hashes, and small
  metrics without a persistent PKL. Optional `single_pkl` emits only
  `artifacts/result.pkl`, with selected control expression deduplicated into a
  shared pool and exact per-condition 300 ordered indices/IDs. Server training
  remains stopped; this source/config change requires a fresh synchronized
  lineage before any future execution.
- Plan 017 local implementation verification passed: 177 tests with 9 honest
  dependency/server-receipt skips, Ruff check/format, compileall, diff check,
  and package build. The local environment lacks Torch/PyG, so strict mypy's
  remaining errors are dependency-derived; server static/full gates remain
  pending while server work is stopped. No commit, push, server sync, training,
  or historical-output rewrite has occurred for plan 017.
- Commit `c6418df` passed the TxPert RPE1 hard gate and produced strictly
  validated GEARS K562/RPE1/Jurkat/Norman results, but TxPert HepG2 then failed
  before model construction because the isolated HepG2 adapter cache exposes
  canonical `cell_type` while frozen TxPert requires its exact `cell_line`
  observation column. Both queues were stopped; the failed HepG2 root,
  interrupted GEARS HepG2 peer, logs, and matrix receipts are preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-c641-txpert-hepg2-missing-cell-line`.
  Plan 016 is the bounded adapter-column repair; any source change creates a
  fresh external-run namespace and c641 outputs will not be mixed with it.
- The earlier all-learned one-epoch-only execution policy is complete and is
  superseded for native GraD-Pert only by Plan 021. GEARS and TxPert remain
  one-epoch-only. Nadig Jurkat B2 seed 1 may now enter a maximum 200-epoch full
  run with validation-only patience 10 after its exact-commit smoke gate.
- Formal-v2 at clean commit `c240157` has valid manifests for all five
  GraD-Pert runs, all 15 nonlearned runs, and GEARS K562. GEARS RPE1 and Jurkat
  completed their optimization epoch but failed inside the frozen official
  `compute_metrics`: their source AnnData lacks DE rankings, the current
  `skip_calc_de=True` path emits a one-index sentinel, and SciPy Pearson rejects
  vectors shorter than two. Both queues stopped fail-closed with no full task.
- Plan 009 is the bounded repair: run the frozen official GEARS DE-ranking path
  on the already test-free train+validation AnnData, preserve the failed c240
  evidence, and establish a fresh benchmark-runner lineage. The existing c240
  GraD-Pert Nadig Jurkat run remains the immutable optimization B0 and will not
  be rerun.
- First repair commit `dc8e24a` passed all server quality gates and its GEARS
  K562 run completed, but RPE1/Jurkat then exposed two/six singleton
  train+validation conditions for which frozen Scanpy refuses a t-test. The
  follow-up keeps every shared condition, uses the official ranking unchanged
  for rankable groups, and supplies singleton groups a stable full-gene order
  only for GEARS' internal one-epoch bookkeeping. Shared evaluation continues
  to mark singleton DE metrics unavailable.
- Follow-up commit `5f82d73` proved the singleton ranking maps themselves are
  complete, but also exposed ordering inside the frozen API: with
  `skip_calc_de=True`, `new_data_process` builds its PyG cache before the
  adapter attaches those maps. RPE1/Jurkat consequently retained one-index
  pre-ranking graphs and failed after the epoch; K562 passed only via its older
  correctly ranked cache. That lineage and all three involved cache directories
  are preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-5f82-pre-ranking-graphs`.
  Plan 009 now attaches rankings before the single frozen official graph build
  and verifies exactly 20 DE indices per retained condition before fit.
- Commit `dddc767` validated that repair on GEARS K562, RPE1, Jurkat, and
  Norman. The first TxPert task then exposed a separate storage compatibility
  issue: frozen Anndata 0.11.4 cannot read Anndata 0.13.2's explicit null
  encoding at `/uns/log1p/base`. The stopped lineage is preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-dddc-txpert-anndata-null`.
  Plan 010 removes only that adapter-copy key when its value is null; a real
  server cross-version fixture proves the sanitized file preserves expression
  values and loads successfully in the frozen environment.
- The `c127380` hard gate showed the exception actually occurs earlier, while
  frozen Anndata opens the immutable canonical H5AD before an adapter copy is
  built. That stopped lineage is preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-c127-canonical-null-reader`.
  The follow-up registers the inspected Anndata 0.13.2 `null` 0.1.0 read
  semantics (`return None`) in the frozen runner's in-memory registry before
  `CanonicalTrainingData`, then retains the adapter-copy write sanitation. It
  does not edit the canonical file or frozen environment.
- Commit `9207e8c` passed that canonical-read boundary and reached the frozen
  official TxPert data module, then failed on its first CUDA `torch.cat`. The
  isolated official environment uses Torch 2.6/cu124, whose wheel supports only
  through `sm_90`, while both server RTX 5090 GPUs require `sm_120`. The failed
  root and log are preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-9207-txpert-sm120`.
  Plan 011 keeps GraD-Pert and GEARS environments unchanged and builds a new
  TxPert-only Torch 2.7/cu128 environment from the frozen official lock plus a
  narrow committed CUDA override. The runner will reject the environment unless
  exact versions, CUDA 12.8, `sm_120`, a core CUDA kernel, and a PyG extension
  kernel all pass before data/model work.
- The first new-environment resolution exposed one bounded transitive conflict:
  Torch 2.7 requires SymPy 1.13.3 while the Torch 2.6 lock pins 1.13.1. The
  failed 84 KiB environment is preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-txpert-env-10a-sympy`; plan 011
  now overrides exactly that additional pin and nothing else.
- The completed `d75d93a` TxPert-only CUDA 12.8 environment passed `sm_120`,
  core CUDA, PyG scatter, and dependency gates. Its RPE1 hard gate then failed
  before optimizer step 1 because frozen TxPert retained its construction
  device `cuda:1` while Lightning's `devices=1` selected visible `cuda:0`.
  The full stopped lineage is preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-d75-txpert-device-binding`.
  Plan 012 isolates each assigned physical GPU at the subprocess boundary and
  presents it to frozen TxPert as its official local `cuda:0`, with explicit
  matrix/runtime receipts; GraD-Pert and GEARS behavior remains unchanged.
- Commit `b06f29f` passed the process-local GPU preflight and reached the first
  official TxPert training batch. It then failed before optimizer step 1 at
  frozen `z_p[p]`: the official dataset extension represents 11,485
  control-only training rows as string lists `["ctrl"]`, while treatment rows
  already use the official numeric control ID `-1` and the frozen model indexes
  a tensor with every component. The entire b06f run/log lineage is preserved
  under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-b06f-txpert-pert-index-shape`.
  Plan 013 performs one bounded adapter translation from the official control
  label to `data_module.pert2id["ctrl"]`, rejects every other nonnumeric or
  unknown component, and records before/after hashes and counts before fit.
- Commit `687681f` proved that translation and its real-data preflight, but the
  subsequent hard gate reproduced the same first-step failure. Direct evidence
  then showed why: Lightning calls the frozen data module's `setup("fit")`
  again when it is passed as `datamodule=`, replaces `train_data` with a new
  object, and restores all 11,485 string control rows after the adapter receipt
  was sealed. The complete 6876 run/log lineage is preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-6876-txpert-lightning-reset`.
  Plan 014 passes the already-adapted loader returned by the frozen official
  `train_dataloader()` into Lightning, retaining the official dataset,
  collate/shuffle/batch, training step, optimizer, and one-epoch semantics while
  preventing the destructive second setup.
- Commit `a5f3473` closed the second-setup boundary and completed the full RPE1
  epoch with a positive training receipt. Prediction then exposed a distinct
  Lightning teardown boundary: registered model parameters had been returned
  to CPU while frozen Exphormer's device-owned index tensors remained on local
  `cuda:0`. The full a5f3 lineage is preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-a5f3-txpert-postfit-device-teardown`.
  Plan 015 restores the official module to the requested inference device after
  fit/checkpoint, matching the frozen official inference entrypoint's explicit
  `.to(device)`, and receipts the observed parameter/buffer devices before any
  canonical test prediction.
- Commit `c6418df` validated plan 015 end to end on TxPert RPE1: exactly one
  epoch, 2,143 optimizer steps, post-fit parameter/buffer restore to local
  `cuda:0`, one test evaluation, exact three-metric schema, and shared fairness
  hashes all passed. Plan 015 is complete; the HepG2 column failure is a new
  pre-model adapter boundary tracked separately by plan 016.
- 2026-08-25 execution override: GraD-Pert full runs now use `max_epochs=100`
  and train/evaluation batch size 256. The ad6 full gate/final watcher were
  stopped before any full task launched. Completed ad6 smoke/nonlearned outputs
  remain evidence but the native capacity gate and learned smoke matrix must be
  rerun at the next clean commit before full training.
- Performance work in progress: batch disconnected graph views per encoder
  call and reuse the two actual branch-backward traversals for gradient
  diagnostics, without changing losses, view identities, gradient ownership,
  optimizer→Teacher EMA→center order, splits, or evaluation.
- First batch-256 capacity attempt at `4d68da40` exhausted all frozen prototype
  candidates but emitted no structured failure receipt. The log is retained;
  the gate is being repaired to atomically persist every rejected candidate and
  terminal failure before any memory-policy change is considered.
- The repaired receipt at `3e366fd` proved allocator reservation fragmentation:
  at 8,192 prototypes K562 peaked at 18.17GB allocated but 27.39GB reserved,
  just above the 26.71GB threshold. The next evidence-led iteration freezes
  expandable segments and caps the speed-first candidate set at 16,384/8,192.
- The same-code 64-vs-256 comparison is complete and selects batch 256. Both
  passed five datasets × 128 real steps at 16,384 prototypes. Batch 256 reduced
  estimated epoch time by at least 3.223×, retained at least 24.55% headroom
  below the 85%-of-initial-free threshold, and raised cell throughput by
  3.84--4.04×. The hash-pinned decision is tracked at
  `registry/capacity/batch_comparison.development.json` and uses no test metric.

- Goal mode: paused during current server training; no active Codex goal.
- Project state: root Git repository and standalone package shell exist; plans
  002 and 003 are complete, and native model plan 004 is active.
- Discussion source: Codex task `01a00ab3-9864-7032-98d9-45f6d0016838`, `TxPert/grad-pert`, and `.byte-os/DISCUSSION.md`.
- Confirmed scope: all five datasets through `canonical_ready`, training, evaluation, and manifests.
- Training route: one B2 configuration; no B3 and no ablation matrix.
- Benchmarks: GEARS, TxPert, matched-control mean, global train delta, and general train delta with Norman additive seen singles.
- Fairness: one canonical condition split/evaluation manifest; paired run seeds and shared 300-control evaluation draws.
- Metric contract: frozen union with three distinct Pearson headlines: TxPert macro delta, TriShift delta, Systema Pearson.
- Artifact contract: versioned condition-level PKL plus Parquet/JSON/H5AD; notebooks consume artifacts only.
- Execution policy: all 15 learned model/dataset integrations stop after exactly
  one epoch under the current user amendment; no model continues to a full run.
- Server topology: formal compute only on `/data/yilangliu`; local/GitHub/server must share one clean commit; large artifacts remain server-only.
- Historical batch-64 hardware decision: two RTX 5090 with 32607 MiB each; the sustained global
  gate selected `K_head=16384`. Candidate 65,536 failed on K562 at step 34,
  32,768 failed on Jurkat at step 20, and 16,384 completed 128 real steps on
  each dataset. Its worst peak reserved memory was 24,226,299,904 bytes on
  Jurkat, below the 28,168,037,990-byte threshold. This receipt does not qualify
  the new batch-256 implementation; a fresh capacity receipt is required.
- Harness: ready; Claude context ready; Codex context ready; `AGENTS.md` ready.
- Config contract: exactly 30 self-contained model/dataset YAML files verified with 30 unique byte hashes; no global config/default/merge/inheritance path.
- Evidence iteration 1 migrated all five splits to the frozen
  official GEARS default-graph representability intersection without
  reshuffling retained conditions. Data/hash verification, graph rebuild and
  evaluator-state rebuild passed. The sustained `K_head` refit is complete and
  the five native configs now freeze 16,384.
- Publication gate: cleared. Explicit approval was received on 2026-08-24, the
  complete source and CI history was pushed to public `origin/main`, and local
  `HEAD` and the remote ref were byte-identical at `5a9109f` immediately before
  this status update. No CI file was removed and no history was rewritten.
- Server source gate: restored and verified. The non-Git storage root remains
  untouched; public `main` was cloned to the new clean checkout
  `/data/yilangliu/GraD-Pert/source`, whose HEAD matches local/GitHub
  `a8a7247c3ac54e570f67cf0b392397e9309abb66` before the current launcher fix.
- Data state: all five official sources remain sealed and all five datasets are
  `canonical_ready` under `datasets-v2`. The canonical H5AD hashes did not
  change; split/control hashes did. Server verification passed for all five,
  and prior split/evaluation/graph receipts are recoverable under
  `/data/yilangliu/GraD-Pert/superseded/20260824-gears-default-intersection`.
- Local prepared/result hygiene: prior datasets-v1 and seed-0 small artifacts
  are preserved under explicit `superseded/` roots. Current v2 receipt chains
  are marked pending sync rather than being inferred from terminal output.
- Local verification: 137 tests pass and 9 optional Torch/PyG/anndata or
  pending-v2-receipt tests skip; all 30 self-contained configs, Ruff
  lint/format, the 20-module available-dependency strict mypy surface, and the
  offline wheel/sdist build pass. Full local strict mypy cannot complete because
  the intentionally lightweight local environment has no Torch/PyG; server
  policy/config/type gates passed before the latest resume/launcher additions,
  so a fresh complete server regression remains required after synchronization.
- Evidence iteration 2 closed the server-operations implementation gap. The
  dry-run-first matrix now freezes 15 learned smoke, 15 paired nonlearned and
  20 native full tasks. Full runs are machine-gated on exact one-epoch training
  receipts, checkpoints, no test Truth during fit, shared commit/config
  identities, and per-dataset equality of canonical-data/split/300-control
  hashes across all three learned models.
- Small-file synchronization now has a sealed staging layer: only allowlisted,
  size-bounded, non-symlink files from named `small_results` or one explicit
  receipt root can enter a new staging tree, and the transferred tree must
  reverify with no missing, extra or changed files.
- Evidence iteration 3 strengthened the leakage and source boundaries. GEARS
  and TxPert now finish fit, checkpoint hashing and training receipt sealing
  before the canonical test reader exists; development Git worktrees reject a
  declared commit that differs from HEAD. The current pre-server review verdict
  is `block`, with exact unresolved formal-execution evidence listed under
  `.byte-os/reviews/2026-08-24-pre-server-review.md`.
- The final notebook surface is now executable rather than implicit. An
  explicit-source, dry-run-first builder seals a catalog only for the exact 45
  formal coordinates (all six models at paired seed 1 plus native seeds 2--4),
  one source commit, shared per-dataset fairness hashes, one native config per
  dataset, and exact three-metric schemas/denominators. The benchmark notebook
  was re-executed and uses the strict final loader; no formal catalog is claimed
  before the actual server results exist.
- TriShift architecture audit: source-level call chain recorded in
  `docs/provenance/TRISHIFT_ARCHITECTURE_ALIGNMENT.md`; the explicit hash-pinned
  ResultCatalog and evaluator-only condition bundle are implemented and tested,
  without newest-result discovery or runner-side Truth.
- Notebook handoff: the read-only benchmark notebook is generated and executed;
  it reports no result until an explicit hash-pinned catalog is synchronized.
- Fresh server regression: pass on the clean checkout with Torch 2.13 CUDA
  enabled on two RTX 5090 GPUs: 153 tests passed, three datasets-v2 receipt-sync
  tests skipped, Ruff passed, strict mypy passed on all 62 source files,
  isolated wheel/sdist build passed, all 30 configs verified, and Git remained
  clean. The earlier access-window notice is superseded by this live evidence.
- Fresh data gate: all five server datasets passed full `data verify --all` as
  `canonical_ready`; their canonical-data, split and ordered 300-control hashes
  were observed directly from the storage root.
- Official-runner gate: all five GEARS and all five TxPert configs passed the
  guarded official-checkout preflight at commits `f374e43` and `08d82ee`.
  The two GEARS graph resources were hard-linked into the dedicated official
  cache only after their hashes matched all five registry entries.
- The first exact 15-task smoke dry run selected no execution and exposed a
  launcher bug before training: resolving interpreter symlinks collapsed every
  isolated Python path to `/usr/bin/python3.12`. The launcher now validates but
  preserves each virtualenv path, with a subprocess regression test.
- The repaired launcher passed the fresh server gate at `1412939`: 154 tests,
  three honest pending-receipt skips, Ruff, strict mypy on 62 source files,
  isolated build, all 30 configs, clean tree and a 15-task no-write dry run.
- The first deliberate K562 smoke attempt failed before training because the
  server's direct `git ls-remote` path to GitHub timed out. Its empty run,
  failed matrix receipt and log are retained for superseding; no GPU work or
  scientific artifact was produced.
- A loopback-only remote SOCKS forward on the SSH ControlMaster restored live
  GitHub verification and returned public `main=1412939`. Git identity commands
  now also disable interactive prompts and fail closed after 30 seconds.
- Next action: publish/synchronize the bounded Git identity fix, archive the
  failed empty attempt, repeat all gates at the final frozen commit, then rerun
  the first exact one-epoch GraD-Pert smoke through the verified SSH tunnel.
- Commit `28e859a` passed the complete server regression and source-parity gate.
  All 15 nonlearned runs completed and their run identities, formal lifecycle,
  three-metric schemas/denominators, task receipts, and per-dataset fairness
  hashes passed strict verification. These results remain evidence but must be
  superseded and rerun if the final source commit changes.
- The first real native K562 smoke completed all 1,346 training steps and wrote
  both checkpoints, then failed before its single test evaluation while loading
  `best.pt`: `torch.load(map_location=cuda)` had moved the saved CPU RNG tensor
  to CUDA, while `torch.set_rng_state` requires a CPU ByteTensor. GPU0 stopped
  fail-closed; RPE1 was deliberately interrupted before it could encounter the
  same deterministic failure. Both incomplete runs, logs and receipts are
  recoverably archived under
  `/data/yilangliu/GraD-Pert/superseded/20260824-checkpoint-rng-device-failure`.
- A bounded repair now validates RNG tensor types and moves CPU and CUDA RNG
  states back to CPU before PyTorch restoration. A CUDA checkpoint regression
  reproduces the server path. Local Ruff/format/diff checks pass; the local
  environment has no Torch, so server CUDA regression and a fresh full gate are
  required before the next frozen formal launch.
- Metrics-only persistence is now enforced across each whole successful run
  root. GEARS hashes and removes its reconstructible `config.pkl` and
  `custom_split.pkl`, retains `model.pt`, and writes a small checkpoint-retention
  receipt. The first repaired hard-gate launch at `e68712e` stopped before model
  construction because the formal `git ls-remote` publication check timed out;
  its zero-PKL failure evidence is preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-e687-git-ls-remote-timeout`.
  Renewing the loopback reverse SOCKS forward restored the exact public-main
  identity check. The next action is a fresh, hash-pinned namespace whose launch
  contract explicitly carries that verified proxy environment, followed by the
  single GEARS K562 hard gate only.
- The `e867c69` GEARS K562 zero-PKL hard gate passed strictly, and GEARS Jurkat
  also passed before the next independent boundary was observed. TxPert RPE1
  reached evaluated-output sealing but was correctly rejected because its
  official adapter cache still contained `splits/train_test_split.pkl` and
  `splits/subgroup.pkl`. Both queues were stopped; failed TxPert RPE1,
  interrupted GEARS HepG2, exact queue logs and receipts are preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260826-e867-txpert-split-pkl`.
  The bounded repair validates both recorded hashes before deleting either
  split input after official fitting and emits `adapter_cache_retention.json`;
  a fresh synchronized commit, full gates and new external namespace are now
  required.
- The fresh `2bf2771` formal external lineage completed all five GEARS and all
  five TxPert coordinates at exactly one epoch. Both queues returned zero, all
  ten evaluated run manifests are present, and the complete formal run root has
  zero persistent PKL. The project-wide default is now explicitly documented:
  GraD-Pert, GEARS, TxPert, and every nonlearned baseline use `metrics_only`;
  only an explicit `single_pkl` request may retain one deduplicated result PKL.
- User-authorized cleanup removed 126 non-active historical experiment PKLs,
  reclaiming 355.10 GiB. Checkpoints, configs, manifests, logs and small results
  were preserved; the current formal root plus all data/environment caches were
  excluded. Historical c240 metrics can be audited from sealed small evidence
  but cannot be recomputed from the intentionally deleted prediction matrices.
- A post-run staging dry run found that `config.resolved.yaml` was missing from
  the small-file extension allowlist. The bounded follow-up adds YAML/YML to the
  existing size/symlink/hash gates and regression-tests resolved-config staging.
- The next dry run measured necessary ordered-ID evidence rather than guessing:
  the largest file is 6.42 MiB, while the external and retained-c240 selections
  total 100.26 and 118.10 MiB. Small-sync defaults are therefore bounded at
  8 MiB per file and 128 MiB total; binary/scientific suffixes remain forbidden.
- Both small-result stages were executed, transferred, and reverified locally.
  External: 195 files, 105,131,234 bytes, file-list SHA-256
  `2dd0b89886e28c503369981585d4500e3dd62a87dd327ac20a831bd7ce1ec8a7`.
  Retained c240: 325 files, 123,833,824 bytes, file-list SHA-256
  `925009e128bec63e7c5bfb20ed6bb6c54054f655bd30e55f9329d2540b05bed0`.
- The explicit two-lineage one-epoch audit passed exact 30 coordinates with
  coordinate SHA-256
  `a5fe297272cb3cf6f7c0b8d4587027538e725e183eefa4a257b90134aa818891`.
  Each dataset has one shared protocol/canonical/split/ordered-300-control/Truth
  identity across all six models; the three metric IDs and one-test lifecycle
  also passed. All 15 learned checkpoint files exist on the server and match
  their manifest hashes.
- Historical limitation: all 20 retained c240 coordinates predate
  `inference_recipe.json`. Their small metrics/manifests/configs and exact row
  IDs remain auditable, and the five learned checkpoints remain hash-valid, but
  they must not be described as having the newer complete recipe bundle. The
  ten new external coordinates all carry a valid `metrics_only` recipe.
- Nadig Jurkat speed pilot B2 completed and passed 41 strict checks at clean
  commit `2e30fb5`. It trained one epoch/582 steps on the full 6,506-node graph,
  retained only checkpoint `best.pt` with SHA-256
  `1229f46c44955f940e9a0972dc2d540af231cc16dea2a2fce52255bb000c8649`,
  performed one test evaluation, and left zero PKL. Its 10-warmup/572-measured
  timing was 718.681 s training wall, 0.6270 steps/s and 137.80 cells/s; the
  three recorded metrics remain explicitly non-decisional. All seven systems
  groups passed runtime and first-step equivalence checks. The merged-read
  fallback was enabled but honestly recorded zero runtime batches because the
  full control cache served all 582 batches. Strict validation receipt SHA-256
  is `1f0c50357463d303e9be19d6a3845306c685b453f744d63dc5d1ffb3b2a70fa2`.
- B3 implementation, execution, strict validation, comparison, and reviewed
  small-evidence transfer are complete. The server retains all checkpoints and
  scientific data; the repository tracks only the 56 KiB contract/verifier/
  comparison evidence bundle and the final pilot review.
