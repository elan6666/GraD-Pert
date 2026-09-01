# B2-vNext graph-scale and local-view ablations

## Status and purpose

This document preregisters the four-local successor to the interrupted `42e`,
`8221`, `276d`, and `f1c14d8` local-graph lineages. Those run roots remain
immutable descriptive evidence, but none of their A/H/L coordinates may
satisfy this successor matrix.

The experiment asks two separate questions on Nadig Jurkat:

1. **H module:** how does the shared Teacher/Student global graph scale affect
   prediction when the Student local-view coverage ratio is held fixed?
2. **L module:** how do local construction, coverage, count, and anchor masking
   affect prediction when the global graph axis is held fixed?

Every row uses the same canonical split, ordered 300-control/truth manifests,
seed 1, batch size 256, 16,384 prototypes, Exphormer-MG STRING+GO encoder,
loss weights `1.0/0.8/0.4/0.1`, exactly 10 epochs with 10 validations and no
early stopping, `metrics_only`, and zero persistent PKL. Validation is the only
model-selection surface; test truth is opened once after training from the best
checkpoint.

## Axis and ratio definitions

For requested HVG count `H`, materialize one ordered runtime graph axis from the
complete filtered cell line before the condition split:

```text
V_global(H) = ordered_union(Top-H normalized-dispersion HVGs,
                            all representable canonical perturbation targets)
N_global(H) = |V_global(H)|
```

Teacher global views and Student global views use the same `V_global(H)`. The
Teacher receives only the two global views. The Student receives the same two
global views, its local views, and the clean prediction view. This program does
not introduce asymmetric Teacher/Student embedding tables or graph axes.

Local node coverage is a ratio of the **actual runtime graph size after target
union**, not a ratio of the requested HVG count:

```text
requested_local_ratio = p / q
effective_local_node_budget = floor(N_global(H) * p / q)
```

The effective budget must still retain every active anchor. The run receipt
records `H`, ordered-axis and topology hashes, `N_global(H)`, requested ratio,
effective integer budget, quotient/remainder, every realized local node count,
and realized coverage summaries. Exact integer arithmetic is used; Python float
rounding is not. The current sealed `H=512` axis has `N_global=2,809`, so
`R=1/2` resolves to an effective cap of 1,404 nodes and `R=1/4` resolves to
702 nodes. A run fails before model construction if active anchors exceed the
derived cap; the runtime never silently expands a ratio-derived budget.

Local anchor masking uses a separate ratio:

```text
effective_mask_count = local_view_count * local_anchor_mask_view_ratio
```

Formal configs fail closed if this product is not integral. They never round a
mask count silently. Both the requested ratio and effective count are receipted.

## Default A0

| Factor | A0 value |
|---|---|
| Global axis | `HVG512 + all targets` (`N_global=2,809` in the sealed graph) |
| Teacher/Student global views | same complete runtime node axis |
| Local builder | `RingInduced` |
| Local node budget | `50%` of actual `N_global` (`1,404` for H=512) |
| Local views per condition | `4` |
| Local anchor-mask ratio | `0%` |
| Encoder and sources | Exphormer-MG, ordered STRING then GO |

The 50% local default is substantially larger than the superseded fixed-256
coordinate. Before any 10-epoch run, it must pass an idle-GPU capacity and
throughput gate on the real data. A capacity failure does not authorize silently
changing this scientific coordinate; it requires a newly reviewed matrix.

## H module — shared global graph scale

The only scientific factor is `graph_hvg_count`. The local ratio remains 50%,
so its effective integer budget is a declared derived consequence of graph
scale rather than an independently tuned factor.

| ID | Global axis | Local builder | Local coverage | Interpretation |
|---|---|---|---:|---|
| H0/A0 | `HVG512 + targets` | RingInduced | 50% | reference |
| H1 | `HVG1024 + targets` | RingInduced | 50% | larger global context |
| H2 | `HVG2048 + targets` | RingInduced | 50% | larger global context |
| H3 | `HVG5000 + targets` | RingInduced | 50% | full prepared-HVG context |

All four axes are computed pre-split from the same filtered cell line and must
be materialized, ordered, hashed, and graph-pruned before the matrix launches.
Results do not justify selecting a new row after test inspection.

## L module — local-view factors

The global graph is fixed at the sealed `HVG512 + targets` axis. Every row is
A0 plus exactly one named factor.

| ID | Builder | Node ratio | Local count | Anchor-mask ratio | Only changed factor |
|---|---|---:|---:|---:|---|
| A0 | RingInduced | 50% | 4 | 0% | reference |
| L1 | Fanout | 50% | 4 | 0% | builder |
| L2 | RingInduced | 50% | 8 | 0% | local count |
| L3 | RingInduced | 25% | 4 | 0% | local node coverage |
| L4 | RingInduced | 50% | 4 | 50% (`2/4`) | anchor-mask ratio |
| L5 | RingInduced | 50% | 4 | 25% (`1/4`) | anchor-mask ratio |

The matrix generator must construct every row from A0 and validate a semantic
diff allowlist. Derived values such as effective node or mask counts are not
additional factors. Any other unregistered scientific difference fails config
generation.

## Performance-first execution gate

No formal A/H queue launches until exact-effect performance engineering has
completed for the new four-local A0 coordinate. L is configured and frozen but
remains paused by user instruction. Performance work follows this order:

1. Run the current accepted implementation as the real-data A0 reference in a
   capacity check and profiler on one idle physical GPU. Measure before choosing
   another implementation target.
2. Record stage distributions and profiler evidence for view construction,
   sparse-union preparation, host/device transfers and synchronizations,
   Teacher global, Student global, Student local, prediction, backward, logging,
   peak memory, clocks, power, temperature, CPU, RAM, disk, and competing
   processes. Validation and test readers remain unopened.
3. Optimize only a measured material bottleneck. Do not reduce views, graph
   coverage, sources, layers, prototypes, precision, losses, validation, or
   evaluation to obtain speed.
4. Require exact reference/optimized view structures, ordered unions, RNG,
   outputs, losses, every gradient, optimizer/model/Teacher/center state, and
   prediction content hashes. Timing and operational telemetry are the only
   equality exclusions.
5. Run serial single-GPU matched ABBA timing after the exact-effect one-step
   gate. Concurrent two-GPU timing is not accepted as an absolute speed claim.

Only a reviewed, clean, locally/GitHub/server synchronized performance commit
may become the source lineage for the A/H runs.

## Formal H results

The four-local A/H lineage `formal-vnext-ah-845c10a-v2` completed at exact
source `845c10a`. Each row has 5,820 ordered optimizer steps, ten validations,
one evaluation from `best.pt`, the exact three metrics, zero persistent PKL and
only the retained best checkpoint. All rows share the exact canonical split,
condition order, ordered 300-control rows and truth-row identities.

| Row | Runtime nodes | Local cap | TxPert macro delta | TriShift delta | Systema Pearson |
|---|---:|---:|---:|---:|---:|
| A0 / HVG512 | 2,809 | 1,404 | 0.251406 | 0.245002 | 0.018933 |
| H1 / HVG1024 | 3,238 | 1,619 | 0.251408 | 0.239399 | 0.017970 |
| H2 / HVG2048 | 4,115 | 2,057 | 0.247875 | 0.225980 | 0.017752 |
| H3 / HVG5000 | 6,487 | 3,243 | 0.247738 | 0.234845 | 0.031904 |

These are single-seed descriptive direct comparisons. Increasing the global
HVG count did not improve all three metrics consistently: H3 has the highest
Systema point estimate, while A0/H1 have the highest TxPert macro point
estimates and A0 has the highest TriShift point estimate. The results do not
establish equivalence or general superiority and do not authorize test-selected
configuration changes. A0 remains the preregistered default; L remains paused.

The reviewed compact evidence is in
`.byte-os/evidence/vnext-performance/formal-ah-four-local-845c10a.json`.
Large scientific artifacts remain on `/data/yilangliu`.
