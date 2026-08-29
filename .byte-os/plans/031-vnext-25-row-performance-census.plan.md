---
id: 031
status: in_progress
wave: 3
depends_on: [029, 032]
updated_at: 2026-08-29T19:15:00+08:00
---

# Plan 031 — B2-vNext 25-row capacity and performance census

## Implementation readiness

- The non-CUDA P0 checker, expression-free 110-step semantic batch manifest,
  bounded P1/P2/profile worker, native phase observer, P1-to-P2 same-GPU gate,
  zero-PKL scan and independent aggregation validators are implemented. P0 and
  batch evidence use atomic exclusive output claims; workers rehash all sealed
  data, graph and GenePT files and the published source content tree both
  before and after execution.
- H512/H1024/H2048/H5000 graph materialization and the cross-H nested lineage
  audit are sealed server-side. The new exact-commit P0 receipt and frozen
  batch manifest still must be generated after publication; the checkboxes
  below remain open until live receipts exist.
- No new CUDA census process or formal H/L row has launched from this source.

## Objective

Measure every registered Nadig Jurkat vNext coordinate through an adaptive,
truth-free performance funnel before selecting any implementation optimization.
Treat capacity failure as evidence, separate profiler attribution from timing,
and never use short-run metrics to choose a scientific ablation row.

## Frozen scope

- Exact schema-v2 25-row matrix at
  `configs/ablations/nadig_jurkat/matrix.json`.
- One canonical split, seed 1, batch 256, 16,384 prototypes, exact config and
  graph identities, `metrics_only`, expandable allocator and zero persistent
  PKL.
- A0/H/L/M/W/D/E/O rows are measured as registered. A performance harness may
  not change graph size, local ratio/count, source order, encoder, feature
  mode, decoder, objective, precision or batch composition to make a row fit.
- Validation and test objects remain unopened throughout the census.
- Every short-run receipt declares `evidence_class=performance_training_only`
  and `scientific_completion=false`; it cannot satisfy a formal row.
- The four E rows use only the Plan 032 `Seed-GO-ProteinPathway` artifact and
  exact runtime-axis selection. No old `emb_b` availability receipt applies.

## Adaptive funnel

### P0 — static and materialization preflight for all rows

- [ ] Validate exact matrix/config/source/data/split identities and semantic
      one-factor declarations.
- [ ] Materialize missing HVG1024/2048/5000 graph axes from the same frozen
      full-cell-line source workflow and seal one cross-H nested-set, target
      coverage, graph-order and topology audit before measuring H rows.
- [ ] Resolve actual runtime graph N, topology/order hashes, local budget and
      mask count, model parameter count and GenePT availability before CUDA.
- [ ] Classify an unavailable row with a sealed preflight receipt; never
      silently substitute another prior, graph or feature mode.

### P1 — one complete-step capacity gate for all eligible rows

- [ ] Run every eligible row in a fresh process/root through one complete
      forward, both gradient paths, optimizer step, Teacher EMA and center
      update.
- [ ] Capture an atomic stage-progress and memory receipt even when the step
      fails before returning metrics.
- [ ] Record CPU RSS, CUDA allocated/reserved/free/peak, retry/OOM counters,
      physical GPU identity and competing processes at every stage boundary.
- [ ] Preserve the existing A0 failure: zero completed steps and approximately
      30.65 GiB allocated before a 30 MiB allocation failed. Do not overwrite
      or relabel it.
- [ ] Seal order-sensitive row, condition, sampled-control and anchor IDs for
      the exact batch and bind all later stages to prefixes of one frozen
      110-step training-batch schedule.

### P2 — short timing census for capacity-passing rows

- [ ] Run 5 warmup plus 20 uninstrumented measured steps with the same ordered
      batch/global-step sequence per reference/optimized comparison.
- [ ] Report raw step/stage values and p50/p90/p95/p99, not only a mean.
- [ ] Record actual unique-condition count, local nodes/edges/budget hits,
      dynamic union and encoder-forward counts and system activation/fallbacks.
- [ ] Keep profiler traces out of timing acceptance.

### P3 — focused attribution

- [ ] Run separate profiler processes using wait 1, warmup 1, active 3 for one
      representative of every distinct execution shape and every capacity or
      high-variance row.
- [ ] Attribute Ring selection/induction, sparse-union preparation/alignment,
      scalar synchronization, D2H/H2D, Teacher, Student global/local,
      prediction, both backward paths and update stages.
- [ ] Extend a row to 10 warmup plus 100 measured steps only when 20-step
      variance or attribution is insufficient, or when it represents the
      implementation target selected for optimization.
- [ ] Freeze promotion before execution: relative MAD above 10%, p95/p50 above
      1.25, first-half/second-half median drift above 10%, reserved-memory
      growth above 5%, near-headroom status, or selected implementation target.
      Promote the matched A0 comparator to the same depth.

### P4 — one-epoch lifecycle confirmation after optimization

- [ ] Run one epoch only for A0 and the smallest representative set covering
      every modified execution path. This verifies loader/prefetch, logging,
      checkpoint and epoch-boundary overhead with validation disabled; it is
      not a scientific result and constructs neither validation nor test data.

## Write scope

- `scripts/performance/` for the matrix census, launch-contract generation,
  analysis and report generation.
- `src/gradpert/training/` and `src/gradpert/modeling/` only for identical
  reference instrumentation and atomic failure-stage evidence.
- `src/gradpert/execution/` only for strict schema-v2 census launch and receipt
  binding.
- Focused `tests/performance/`, `tests/training/`, `tests/execution/` and Byte
  OS experiment/evidence/review files.

## Non-goals

- No test evaluation or short-run scientific ranking.
- No 10/100/200-epoch queue in this plan.
- No reduced batch, prototypes, graph, views or precision as a capacity fix.
- No optimization selected from static code inspection alone.
- No cross-row speed difference presented as effect equivalence.

## Analysis and optimization selection

- Capacity is the first optimization objective. A row that cannot complete one
  exact step is component-bisected before timing work.
- A speed target must account for at least 10 percent and 100 ms of median
  uninstrumented step wall, or be the measured cause of capacity failure or
  more than 20 percent of peak memory.
- Common-path work and backend-specific work are reported separately. Faster
  scientific variants are not implementation optimizations.
- Plan 028 may implement only the smallest target authorized by the census.

## Acceptance criteria

- Exact 25-row baseline census closure: P2 complete, preregistered unavailable,
  blocked prerequisite, or preserved capacity-failure receipt for every
  registered variant. A capacity-failed row is analyzed but is not falsely
  described as timed or completed.
- Every launched process is source/config/data/graph/GPU hash-pinned and uses a
  new root. Validation/test guards prove no truth access.
- OOM and other failures preserve the last entered/completed stage and memory
  state without masking the original exception.
- Timing and profiler modes use identical scientific inputs but distinct fresh
  processes and roots.
- The aggregate report keeps capacity, speed attribution and scientific
  evidence in separate sections and authorizes at most one first optimization
  target.

## Verification

```text
python -m pytest -q tests/performance tests/training tests/execution
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m build
```
