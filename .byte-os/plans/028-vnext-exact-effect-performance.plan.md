---
id: 028
status: pending
wave: 2
depends_on: [029]
updated_at: 2026-08-29T17:10:00+08:00
---

# Plan 028 — B2-vNext measured exact-effect performance engineering

## Objective

Measure the real successor A0 path first, then reduce only a demonstrated
material implementation bottleneck while keeping every scientific input, view,
model operation, update, validation output and prediction exactly unchanged.

## Frozen scientific coordinate

- Nadig Jurkat canonical split and ordered 300-control/truth manifests.
- Shared Teacher/Student global graph: pre-split full-cell-line HVG512 plus all
  targets, 2,809 nodes, topology SHA
  `ba22af6e9e9a558533aaae850f619840ea2d717310eb3362a52476c3c1ea9128`.
- RingInduced locals, effective node budget `floor(2,809 * 1/2) = 1,404`,
  eight locals, local anchor-mask ratio 0.
- STRING+GO Exphormer-MG, seed 1, batch 256, 16,384 prototypes, loss
  `1.0/0.8/0.4/0.1`.
- Same optimizer, update order, Teacher EMA, centers, validation and one-test
  lifecycle; `metrics_only`, expandable allocator, zero persistent PKL.

## Write scope

- `src/gradpert/graphs/`, `src/gradpert/modeling/`, `src/gradpert/training/`
  only where a measured bottleneck requires a change.
- `src/gradpert/execution/` for identical reference/optimized instrumentation
  and environment receipts.
- Focused tests, `scripts/performance/`, this plan, the standalone performance
  Markdown, Byte OS evidence/review/delivery records.

## Non-goals

- No change to graph coverage, views, source order, encoder layers, precision,
  prototypes, loss, batch order, validation or evaluation.
- No GAT/Fanout/RingInduced scientific comparison presented as optimization.
- No 10-epoch H/L run used to select implementation changes.
- No performance conclusion from concurrent GPU runs or overlapping summed
  stage timers.

## Evidence-led iterations

### Iteration 1 — unoptimized real A0 profile

- [ ] Publish/synchronize a clean reference commit after plan 029 gates.
- [ ] Bind the profiler to a hash-pinned A0 contract covering config bytes,
      protocol/split, graph manifest/gene/source/topology hashes, optimizer,
      loss, systems, artifact mode and physical GPU identity.
- [ ] Verify one idle physical GPU and no competing host/storage workload.
- [ ] Run a training-only bounded real-data capacity gate without constructing
      validation or test data. Stop immediately after the declared step count;
      do not produce or transfer an additional batch.
- [ ] Fail capacity on allocator retries/OOMs, non-finite state, occupied GPU,
      insufficient disk/RAM, or remaining VRAM below
      `max(4 GiB, 15% physical VRAM)`.
- [ ] Record warmup-excluded stage distributions, peak memory, GPU UUID/driver/
      clocks/power/temperature/utilization, CPU/RAM/disk/thread state.
- [ ] Capture profiler counts and time for view construction, union preparation,
      `_local_scalar_dense`, D2H/H2D, Teacher global, Student global, Student
      local, prediction and backward.
- [ ] Select an implementation target only if the measured evidence shows a
      material residual contribution. Record a no-change decision if it does not.
- [ ] Preserve a minimal atomic failure receipt for preflight, training,
      profiler teardown and telemetry failures without masking the primary error.

### Iteration 2 — smallest measured fix

- [ ] Reproduce the chosen bottleneck in a focused benchmark or profiler test.
- [ ] Implement the smallest semantics-preserving change.
- [ ] Preserve exact ordered views, sparse-union tensors, channel and local-edge
      order, per-view BatchNorm execution and dropout RNG.
- [ ] Compare captured/restored reference/optimized forward tensors, losses,
      every gradient, model/optimizer/Teacher/center/RNG state exactly.
- [ ] Retain the already measured Fanout incoming-index optimization for the L1
      path only; do not count it as an A0 speed improvement unless profiling
      executes Fanout.

### Iteration 3 — matched timing and review

- [ ] Pass full pytest, Ruff, format, strict mypy and isolated build on the
      exact synchronized commit.
- [ ] Pass serial one-step reference/optimized exact-effect hard gates.
- [ ] Run serial single-physical-GPU ABBA timing with identical instrumentation,
      warmed state and fresh roots.
- [ ] Report raw replicates, paired median ratio, p50/p90/p95/p99 stages,
      resource telemetry and every equality hash.
- [ ] Complete independent review and ship only if exact-effect passes.

## Acceptance criteria

- The first server artifact is a reference profile, not an optimized run.
- Capacity/profile receipts prove validation and test data handles were never
  constructed; this claim is scoped to post-materialization model fitting and
  does not describe the preregistered full-cell-line HVG preparation.
- The shipped optimization maps to a measured bottleneck and has before/after
  evidence under the same frozen coordinate.
- Timing/telemetry are the only allowed equality exclusions.
- Any scientific-state mismatch rejects the candidate without weakening gates.

## Verification

```text
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m build
```

Server commands and exact contract hashes are frozen only after the clean
reference/optimized selector commit is synchronized.
