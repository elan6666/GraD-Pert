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

This hypothesis is now confirmed by two sealed real-A0 diagnostics at clean
commit `a7beebf`. The five-step Torch profile records about 11.3--12.0 seconds
per warm/profile step; Student locals account for 7.9--8.43 seconds (about
68--70%). The separate cProfile attempt completed the same five-step
training-only protocol and attributes 55.481 seconds cumulative to 345
`_batched_sparse_union` calls and 52.829 seconds to 342 `build_sparse_union`
calls. By contrast, `_local_scalar_dense` contributes only about 14 ms across
the three Torch-profiled steps, so host scalar synchronization alone is not the
selected explanation.

The smallest selected throughput change is an ordered CPU-vectorized sparse
union. It consumes the existing ordered per-source graph-view pairs, preserves
the exact lexicographic edge/channel and first-source local-edge order, and
transfers only the final edge index, membership and local-edge tensors. A
same-commit environment selector retains the reference implementation for the
exact CUDA hard gate and ABBA comparison; the selected implementation is
written into the resident graph runtime receipt.

On the real A0 first batch (8 unique conditions, 2 globals plus 64 locals,
every local exactly 1,404 nodes), three CPU repetitions gave reference times
of 8,214.3/8,327.9/8,276.7 ms and optimized times of
2,312.5/2,310.5/2,308.7 ms for all 66 unions. Every union tensor was exactly
equal. The medians are 8,276.7 and 2,310.5 ms: 3.58x speedup and 72.1 percent
reduction for union preparation. This microbenchmark is target-specific
evidence, not an end-to-end throughput claim; the latter still requires serial
same-GPU ABBA.

The clean `7332cc1` deterministic CUDA hard gate subsequently passed exact
non-timing metrics, views and unions, every gradient, Student/Teacher model
state, optimizer state, centers, and CPU/CUDA RNG. Serial single-GPU ABBA then
ran A1 reference, B1 CPU-vectorized, B2 CPU-vectorized, and A2 reference. Each
arm used five warmups and twenty measured steps on the identical frozen batch
sequence. Reference p50 was 11,166.911/11,082.219 ms; optimized p50 was
4,842.376/4,890.634 ms. The paired median ratio was 0.437470: a 56.253 percent
wall reduction, 2.286x speedup, and 6,258.060 ms median absolute reduction per
step. Optimized p90 improved in both pairs. Peak allocated/reserved GPU memory
rose only 0.017/0.041 percent, CUDA retry/OOM counters were zero, and all arms
retained zero PKL with no validation or test access. The implementation passes
the preregistered timing and capacity gates.

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

## Representative performance sentinel

Performance capacity and bottleneck discovery use eight representative rows
from the unchanged 25-row scientific matrix:

| Row | Performance path covered |
|---|---|
| A0 `a0_ratio_ring_half` | primary RingInduced-50% optimization target |
| H3 `h3_hvg5000_ratio_half` | largest runtime graph and worst graph-memory case |
| L1 `l1_fanout_ratio_half` | alternate Fanout local-view construction |
| L2 `l2_ring_half_count4` | dynamic four-local path and consistency-term count |
| M4 `m4_adaptive_source_gat` | alternate multi-source GAT encoder |
| W1 `w1_string_edge_feature` | STRING edge-feature path |
| D2 `d2_control_transformer` | heaviest decoder path |
| E2 `e2_genept_id_residual` | Seed-GO-ProteinPathway full-axis projection |

The sentinel is capacity/performance evidence only and never scientific
completion. M4 is used instead of M1 because M1 and W1 both exercise the
single-source STRING-GAT family. An objective-only row is omitted because the
current native step still computes those auxiliary loss tensors before their
configured scalar weights are applied; zeroing a weight does not remove that
compute path. L2 gives more distinct implementation coverage.

P1 runs one complete step for all eight rows. Real profiler attribution remains
A0-led; later short timing is added only where it helps select or regression-
check an implementation optimization. The formal A0/H1--H3/L1--L5 experiments
remain separate fixed 10-epoch scientific runs after exact-effect and matched
ABBA performance gates pass.

## Current boundary

The superseded `276d` queue and the failed 25-row P1-v4 launch remain immutable.
P1-v4 did not enter native/model construction: its host gate used free pages
after hashing large inputs and falsely reported about 2.2 GiB, while Linux
`MemAvailable` showed about 241 GiB reclaimable memory. The implementation now
uses `MemAvailable` and preserves this state as a distinct resource-preflight
failure instead of demanding nonexistent native identity receipts.

The first eight-row sentinel at clean commit `b37963e` is sealed as failed and
must not be repaired or relaunched. A0 failed during its first training step
with a real CUDA out-of-memory error: about 30.65 GiB was already allocated on
the 31.36-GiB RTX 5090 when another 30 MiB allocation was requested. The
stage-progress receipt localizes the retained memory growth precisely:

| Completed phase | Allocated CUDA memory |
|---|---:|
| Student global views | about 5.44 GiB |
| Local index 0, eight condition views | about 13.92 GiB |
| Local index 1, eight condition views | about 22.40 GiB |
| Local index 2, eight condition views | about 30.87 GiB |
| Local index 3 | entered, then OOM |

The approximately 8.47-GiB increase per completed local index identifies
retained Student-local autograd activations as the capacity blocker; this is
not a fragmentation diagnosis. The failed line also exposed two measurement-
harness defects: nested stage events require a stack, and the validator must
run with both the repository root and `src` on its import path. Those failures
are operational evidence only and do not alter the observed CUDA exception.

The smallest selected implementation change is non-reentrant activation
checkpointing for Student local graph forwards only. It preserves the original
forward order and recomputes each local forward during backward with preserved
RNG. Exphormer-MG BatchNorm buffers are isolated per checkpointed view: the
original forward applies its single running-stat update to the real encoder,
while recomputation uses private pre-view buffers and cannot update the real
state. Globals, prediction, Teacher, view construction, graph/channel/edge
order, dropout order, losses and update order are unchanged.

Synthetic full-step reference-versus-checkpointed tests require exact non-
timing metrics, every parameter gradient, Student and Teacher state including
BatchNorm buffers, optimizer state, both centers, RNG and first-step health.
These CPU gates, the clean eight-row capacity lineage, real A0 profiles,
deterministic CUDA exact-effect gate, and serial same-GPU ABBA now pass. The
accepted sparse-union implementation is ready for the preregistered formal H/L
lineage; none of the performance evidence is relabeled as scientific
completion.
