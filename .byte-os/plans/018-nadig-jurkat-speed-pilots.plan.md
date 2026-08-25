# Plan 018 — Nadig Jurkat one-epoch speed pilots

## Objective

Measure three sequential GraD-Pert performance variants on the immutable Nadig
Jurkat setup without rerunning B0. The pilot selects on speed only; it records
the three benchmark metrics but makes no claim that one epoch proves unchanged
effect.

The labels below are performance-pilot labels. They do not introduce a new
GraD-Pert model route: the native model remains the product's B2 architecture.

## Immutable comparison contract

- B0 is the existing c240
  `smoke/gradpert_b2/nadig_jurkat/seed-1` coordinate; never rerun it.
- Same dataset, canonical split, seed 1, ordered 300-control manifest, batch
  size 256, prototype count 16,384, expandable allocator, hardware, and timing
  protocol wherever applicable.
- B1, B2, and B3 each train exactly one epoch.
- Every variant remains `metrics_only` and leaves zero persistent PKL.
- Each code phase uses a new clean local/GitHub/server synchronized commit and
  a fresh server namespace. A source change invalidates that pilot lineage.

## Variants

### B1 — graph-only

- Expression input/output and evaluation axes remain the frozen Top-5000 HVGs.
- Recompute Top-500 HVGs directly from the same filtered, normalized, log1p
  matrix with the same Scanpy method and parameters used for Top-5000.
- Verify the direct Top-500 set and order equal the first 500 genes in the
  frozen `dispersions_norm` ranking.
- Graph nodes are Top-500 HVGs union every candidate perturbation gene.
- Do not enable any of the seven systems optimizations.

### B2 — systems-only

- Preserve B0's full graph axis.
- Enable all seven semantics-preserving optimizations together:
  1. merge perturbed/control HDF5 reads and restore exact row order;
  2. cache control expression;
  3. background prefetch with pinned memory and nonblocking transfer;
  4. keep fixed graph tensors resident on GPU;
  5. cache validation truth/control expression;
  6. buffer training logs with required epoch/checkpoint/failure flushes;
  7. serialize one checkpoint and create the peer best/last file by
     reflink/copy with a safe fallback.
- Do not change data, batch/control IDs, RNG, losses, gradient ownership,
  update order, evaluation, or split.

### B3 — combined

- Use B1's reduced graph axis and all seven B2 systems optimizations.

## Required receipts

- Actual expression/output/evaluation gene counts and graph node/edge counts.
- Cold start and cache-build time.
- One-epoch wall time, warmup/measured steps, steps/s, and cells/s.
- Stage timings for data read, host-to-device, view build, teacher/student
  forwards, prediction, backward, validation, logging, and checkpointing.
- Peak allocated/reserved GPU memory and peak CPU RAM.
- Three metric values, explicitly labelled non-decisional for this speed pilot.
- B0 timing-field gaps are reported honestly; B0 is never retrained to fill
  missing instrumentation.

## Equivalence and health gates

- Exact batch and paired-control row IDs.
- Pre-transfer tensor hashes.
- Graph/view RNG state and mask hashes.
- First-step losses and parameter/update hashes.
- Optimizer → Teacher EMA → center update order.
- Checkpoint/resume state equivalence.
- Safe fallbacks when pinning, async transfer, caches, or reflink are
  unavailable.
- Targeted tests followed by full pytest, Ruff, format, strict mypy, and build.

## Execution checklist

- [x] Implement and verify B1 locally (185 tests, 9 honest dependency/receipt
      skips; Ruff, format, isolated build, and diff check passed).
- [x] Commit, publicly push, synchronize, and pass server gates for B1 at
      `0a4d339` (204 server tests, 3 honest skips, Ruff, format, strict mypy,
      isolated build, clean exact local/GitHub/server identity).
- [x] Launch and validate B1. It completed one epoch/582 steps with exact B0
      split/canonical/ordered-control/truth identity, zero PKL, Top-5000
      expression/output/evaluation, 2,798 graph nodes, 89,561 nonself edges,
      and checkpoint SHA-256 `453642fc82609df65d78ab29afd2879878e25ea8361f66a583cfb858901235ec`.
      The measured training wall was 844.180 s, with 0.6931 steps/s and 152.31
      cells/s after 10 warmup steps. Recorded metrics are non-decisional.
- [x] Implement and verify all seven B2 optimizations together at `a9f47ff`.
      Local gates passed 190 tests with 9 honest dependency/receipt skips,
      Ruff, format, isolated build, and diff check. The implementation adds
      explicit equivalence, activation, fallback, timing, and memory receipts;
      the formal run remains pending.
- [x] Commit, publicly push, synchronize, and pass server gates for B2. The
      implementation/evidence lineage through `a17b8e7` passed 213 server
      tests with 3 honest prepared-receipt skips, Ruff, format, strict mypy on
      66 source files, isolated build, and clean-tree verification.
- [x] Launch and validate B2. It completed one epoch/582 steps with exact B0
      canonical/split/ordered-300-control identity, exact B1 ordered control and
      truth row IDs, zero PKL, the full 6,506-node graph with 222,654 nonself
      edges, and checkpoint SHA-256
      `1229f46c44955f940e9a0972dc2d540af231cc16dea2a2fce52255bb000c8649`.
      The measured training wall was 718.681 s. Actual full-epoch wall
      throughput was 0.8098 steps/s and 178.47 cells/s. The original
      warmup-excluded serial stage sum reports 0.6270/137.80 but is not used as
      actual throughput because prefetch overlaps data preparation and GPU
      work. All seven optimization groups passed their
      runtime/equivalence checks; the enabled merged-read fallback was dormant
      because the complete control cache served all 582 batches. Validation
      receipt SHA-256 is
      `1f0c50357463d303e9be19d6a3845306c685b453f744d63dc5d1ffb3b2a70fa2`.
- [x] Implement the explicit self-contained B3 config and its combined-contract
      regression test.
- [x] Commit, publicly push, synchronize, and pass server gates for B3 at
      `44ae7ff` (191 local tests/9 honest skips; 214 server tests/3 honest
      skips; Ruff, format, strict server mypy on 66 files, isolated builds, and
      exact clean three-way identity).
- [x] Launch and validate B3. It completed one epoch/582 steps with exact
      B0/B1/B2 fairness identities, zero PKL, one retained `best.pt`, 5,000
      expression/output/evaluation genes, and the exact B1 2,798-node/89,561-
      edge graph. All seven systems groups were requested and active; the
      merged-read fallback remained dormant because the complete control cache
      served all 582 batches. The strict validator passed 49/49 checks.
- [x] Compare speed receipts, record metrics without effect claims, review, and
      deliver. B3 trained in 507.718 s at an actual full-epoch 252.63 cells/s:
      1.663x faster than B1 on the same reduced graph and 1.416x faster than B2
      under the same systems optimizations. B0 timing remains unavailable and
      was not backfilled. The three metrics are non-decisional.
