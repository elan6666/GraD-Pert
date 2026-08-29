# B2-vNext exact-effect performance engineering

## Scope

This work optimizes the implementation of the successor Nadig Jurkat A0
configuration preregistered in
`docs/experiments/VNEXT_GRAPH_SCALE_AND_LOCAL_ABLATIONS.md`. It does not change
or compare scientific factors. The graph axis, RingInduced local views at 50%
of the actual runtime graph, eight locals, STRING+GO Exphormer-MG, batch size
256, 16,384 prototypes, seed, loss, optimizer/update order, validation and test
protocol remain fixed.

Changing to GAT or Fanout, a different local ratio, fewer views, AMP/TF32, or a
different validation schedule is outside this exact-effect comparison.

## Measured baseline

The superseded Fanout-256 formal run recorded roughly 7.1 seconds per step in a
recent observed window, with view construction at roughly 5.0--5.1 seconds.
Those values were sampled while both server GPUs shared host CPU, RAM and
storage. They are retained as historical evidence only and must not be used to
select an optimization for the new RingInduced-50% A0.

The deterministic local microbenchmark uses the actual vNext node scale and a
matched edge scale: 2,809 nodes, 89,888 nonself edges, 8 locals per condition,
four incoming Fanout hops and budgets 256/512. Five post-warmup repetitions
gave the following medians before candidate implementation:

| Conditions | Locals | Budget | Reference ms | Preindexed ms | Speedup | Reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 256 | 168.636 | 54.995 | 3.066x | 67.39% |
| 1 | 8 | 512 | 172.048 | 55.246 | 3.114x | 67.89% |
| 5 | 40 | 256 | 608.335 | 64.813 | 9.386x | 89.35% |
| 5 | 40 | 512 | 606.685 | 65.103 | 9.319x | 89.27% |
| 8 | 64 | 256 | 941.861 | 71.808 | 13.116x | 92.38% |
| 8 | 64 | 512 | 948.680 | 72.889 | 13.015x | 92.32% |

Every complete ordered local-view SHA matched exactly. A separate
`tracemalloc` run for the 5-condition Fanout-256 case measured 11.51 MB versus
8.73 MB peak traced Python allocation, a 24.10 percent reduction. Tracing
inflates wall time, so only its allocation comparison is used.

The superseded Fanout cause is concrete: the reference path scans and sorts the
entire graph once per local view. A typical 5-condition step does this 40 times.
The candidate builds the same immutable, source-ordered incoming-edge index
once per engine. This remains relevant to the new L1 Fanout row, but it is not
assumed to accelerate the new RingInduced A0.
The recorded local environment is Apple M5 with 10 logical CPUs, 16 GiB RAM,
macOS Darwin 25.6.0 arm64 and Python 3.13.13. These values are not extrapolated
to RTX 5090 end-to-end throughput.

## Second measured hypothesis

Static code-path counting for a typical 5-condition step gives 45 independent
encoder forwards and 44 dynamic sparse-union builds. The current sparse-union
path first materializes edge tensors on the GPU, performs scalar `.item()`
validations, copies edge rows back to Python/CPU, sorts and rebuilds final GPU
tensors. This is a hypothesis, not an implementation authorization: the new
real A0 must first show its profiler counts and stage costs. Training views must
remain independent because Exphormer-MG BatchNorm running statistics and
dropout RNG order are scientific state.

## Exact-effect gates

- Exact node IDs and ordered per-source edges/weights for every view.
- Exact masks, warnings and complete view hashes for budgets 256/512,
  1/5/8-condition batches, multi-anchor, isolated-anchor and boundary cases.
- Exact sparse-union edge index, channel membership and first-source local edge
  order before any union optimization.
- Captured/restored RNG with exact forward tensors, losses, every gradient,
  optimizer state, model state, Teacher EMA, centers and RNG after one step.
- Exact validation and prediction content hashes. Timing/telemetry fields are
  the only equality exclusion.
- Serial single-physical-GPU ABBA timing with identical instrumentation,
  warmed caches and no competing GPU/host jobs. Report raw timings and paired
  median ratios rather than the overlapping serial stage sum.

## Training-only measurement boundary

The capacity, profiler and uninstrumented timing phases are bounded training
measurements. They must not construct a validation or test data reader, warm a
validation cache, invoke a validation callback, or produce batch `N+1` after
the declared final update. Their receipts derive access state from unopened
guards rather than hard-coded booleans.

This boundary starts after the preregistered graph artifact exists. The graph
axis itself is deliberately fitted on the complete filtered cell line before
the condition split; that transductive preprocessing choice is separately
receipted and is not described as train-only.

Before CUDA, the measurement contract binds exact config bytes, protocol and
split hashes, ordered graph-gene/source/topology identities, RingInduced ratio
resolution, optimizer/loss/update order, system flags, artifact mode, allocator
and the requested logical device to one physical GPU UUID. Capacity fails on a
competing process, allocator retry or OOM, non-finite state, insufficient host
or disk headroom, or free VRAM below `max(4 GiB, 15% of physical VRAM)`.
Every phase attempts an atomic failure receipt even when preflight or profiler
teardown fails; teardown errors are recorded separately and never replace the
primary error.

## Current boundary

The superseded `276d` queue has stopped and its evidence remains immutable. The
candidate is isolated in a separate worktree. Before any implementation target
is selected, the new ratio-based A0 configuration and reference path must pass
source gates, an idle-GPU capacity check, and real server profiling. Strict
mypy, CUDA exact-effect gates and matched end-to-end timing remain pending.
