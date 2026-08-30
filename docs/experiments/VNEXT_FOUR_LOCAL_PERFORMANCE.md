# B2-vNext four-local A0 performance engineering

## Scope

This is the active performance protocol for matrix
`nadig_jurkat_vnext_ratio_graph_v3`. Its scientific coordinate is fixed:
HVG512 plus all perturbation targets, RingInduced local node ratio `1/2`, four
local views, no local-anchor masking, STRING+GO Exphormer-MG, seed 1, batch
256, 16,384 prototypes, direct loss weights `1.0/0.8/0.4/0.1`, and the frozen
optimizer/update/evaluation protocol.

Changing graph scale, builder, local count or ratio, precision, view/forward
order, BatchNorm/dropout order, loss, validation or evaluation is a scientific
change and cannot be called an exact-effect performance optimization. The
accepted eight-local sparse-union evidence is historical; it motivates code
candidates but does not establish speed for this coordinate.

## Measured baseline protocol

After exact local/GitHub/server commit parity and full server gates, use one
idle physical RTX 5090 and fresh roots:

1. one training-only capacity step;
2. two warmups plus three active Torch-profiler steps;
3. an independent Python profile around the same bounded training path;
4. five warmups plus twenty unprofiled measured steps for timing acceptance.

All phases bind the same frozen semantic batches and keep validation/test
readers unopened. Receipts record source/config/data/split/topology identities,
allocator state, GPU UUID/driver/clocks/power/temperature, host CPU/RAM/disk and
competing processes, zero persistent PKL, and `scientific_completion=false`.

The profile must separate:

- Ring boundary selection, induced-edge extraction and view-stat aggregation;
- ordered-pair remapping, undirected checks, expander/self-loop preparation,
  lexicographic sorting, membership construction and final host-to-device
  transfer;
- Student-local original forward and checkpoint backward recomputation;
- Teacher, Student global, prediction and backward/update stages;
- data read, prefetch, host-to-device transfer, scalar/receipt synchronization;
- CUDA kernel-active time, idle gaps and stage wall distributions.

Sparse `nvidia-smi` utilization samples are supporting telemetry only. The
primary outcome is synchronized step wall and tail latency with exact state.

## Candidate hypotheses, not preselected changes

No candidate is implemented unless the new real A0 attributes at least 10% of
step wall and 100 ms per step to its boundary.

- **Selective local checkpointing.** Test capacity in order from four
  checkpointed local indices down through three, two, one and zero. Select the
  least checkpointing that still retains at least 4 GiB and 15% physical-VRAM
  headroom with zero retry/OOM. A lower-count attempt may be skipped only after
  a failed higher-count attempt and a verified monotonic retained-activation
  contract prove it cannot fit; preserve that failure evidence. No capacity
  result is assumed in advance.
- **Prepared-step overlap.** If CPU view/union preparation creates measured GPU
  idle gaps, double-buffer the next pure-CPU plan while the current GPU step
  runs. Default-stream forward, BatchNorm, dropout, optimizer and receipt order
  must remain exact; final flush state and log rows must match.
- **Static union preparation cache.** If confirmed material, cache immutable
  expander/self-loop arrays and identical global-union structure, or avoid
  redundant channel-membership work only when exact tensor/order gates prove
  equivalence.

A local 2,809-node, eight-condition, four-local microbenchmark found that
caching expander pairs reduced 34 union preparations from median 1,309.41 to
1,143.57 ms (12.66%, 165.84 ms), but this is hypothesis evidence only. A
naive NumPy Ring selection/merge prototype was exactly equal yet 9.8% slower
than the reference and is rejected unless a different server measurement
justifies revisiting it.

## Exact-effect and timing gates

Reference and candidate must match exact ordered batch identities, complete
view hashes, ordered edges/weights, sparse-union tensors, forward tensors,
losses, every gradient, Student and Teacher parameters and buffers, optimizer
state, centers, CPU/CUDA RNG, predictions and all non-timing receipt fields.
Asynchronous work additionally requires exact final state and log order after
flush.

Timing uses serial same-GPU A1-B1-B2-A2 with five warmups and twenty measured
steps per arm. Acceptance requires:

- median wall reduction at least 10% and 100 ms in both paired comparisons;
- neither optimized p90 worse by more than 5%;
- peak allocated/reserved memory not worse by more than 5%;
- required VRAM headroom, zero allocator retry/OOM, zero PKL and no truth
  access.

Only after review and a new clean synchronized commit may the formal A0/H1/H2/
H3 queue launch sequentially on one GPU. L remains paused.

## Formal loss and utilization dashboard

Formal A0/H rows launch the optional private Hugging Face Trackio sidecar
documented in [HUGGING_FACE_TRACKIO.md](HUGGING_FACE_TRACKIO.md). It tails only
the existing allowlisted scalar training/validation receipts, so the dashboard
shows loss components, validation trajectory, stage times, throughput and the
single selected GPU without entering the training process. Trackio is disabled
for capacity/profile/exact-effect/ABBA work and never receives test metrics,
predictions, datasets, checkpoints or row-level identities. Its live view is
auxiliary and provisional; native receipts remain authoritative.
