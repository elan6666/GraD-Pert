# Lessons

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
