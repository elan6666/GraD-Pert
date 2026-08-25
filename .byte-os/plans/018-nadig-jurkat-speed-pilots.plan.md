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
- [ ] Commit, publicly push, synchronize, and pass server gates for B1.
- [ ] Launch and validate B1; suspend goal execution while it runs.
- [ ] Implement and verify all seven B2 optimizations together.
- [ ] Commit, publicly push, synchronize, and pass server gates for B2.
- [ ] Launch and validate B2; suspend goal execution while it runs.
- [ ] Implement and verify B3.
- [ ] Commit, publicly push, synchronize, and pass server gates for B3.
- [ ] Launch and validate B3; suspend goal execution while it runs.
- [ ] Compare speed receipts, record metrics without effect claims, review, and
      deliver.
