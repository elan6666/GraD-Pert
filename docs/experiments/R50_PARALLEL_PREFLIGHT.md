# R50 parallel execution preflight

2026-09-10: user requests parallel progress. Preserve LR-mid on GPU0 and the
existing best/final evaluation queue. GPU1 is idle after REF/LR-low tests.
The sequential scientific matrix is unchanged: do not invent the LR winner.

First measure the unchanged R50 REF on GPU1: a fresh32-step solo probe, then
two fresh32-step probes together. Four warmup steps per process; identical
seed/config, original50-epoch schedule horizon and native training engine.
Each allocator is capped at40% of total VRAM. Synchronize first-step readiness
with a barrier so both paired workers have initialized before measuring.
Validation/test constructors and callbacks are guarded; no test selection,
no checkpoints/PKL are expected, scientific_completion=false. Native interrupted
status is intentional only when accompanied by a complete32-step probe receipt.

Acceptance for a concurrency candidate: all three exact32-step receipts,
matching source/config/GPU, no evaluation access or persistent PKL, zero OOM or
allocator retry, free memory>=4GiB, measured overlap>=80%, aggregate training
throughput>=1.15x solo. Report individual slowdown and aggregate throughput,
not utilization alone. GPU0 LR-mid remains active: this measures conditional
throughput on the shared host, not isolated hardware benchmarking.

Passing short probes is NOT permission to run arbitrary two full experiments.
Next stage must test full-epoch peak capacity including validation/checkpoint
overlap for the proposed pair. On probe failure stop only probe-owned workers;
preserve roots, never kill LR-mid or other users. No automatic retries.

Preparation endpoint: clean published/server identity, CPU tests/lint/format/
types/build, exact source/config/prior/publication hashes, idle GPU1 and adequate
host/disk resources, fresh run/session root, pinned launcher. End active goal
before CUDA launch; restore the existing hourly quiet grad-pert-r50 monitor.
Monitor root COMPLETE.json or failed log/worker receipts once per hour; after
terminal, pause monitor and start the next bounded analysis/preparation goal.
Until LR-mid finishes, prepare schedule/batch candidates and capacity only.
After LR selection, dispatch independent comparisons when their parent is fixed.
