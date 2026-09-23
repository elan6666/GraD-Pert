# GraD-Pert v2 full-model dual-GPU performance and capacity

Status: engineering evidence, 2026-09-23. This is the 33,908,867-parameter
Top500/GenePT-PCA256/SwiGLU model on Nadig Jurkat, with prediction, graph
distillation, cell distillation, and cross-rank KoLeo all enabled. Two RTX 5090
cards train together; default effective global batch is 128, with 32 cells per
rank per microbatch and two accumulated microbatches. The project five-epoch
warmup+cosine recipe, frozen canonical split, and default row-mean losses are
unchanged. Source, resolved config, data identity, and both GPU measurements are
in each immutable server receipt under `/data/yilangliu/GraD-Pert/development/`.

## Same-batch performance comparison

The bounded `benchmark_only` probe runs five complete optimizer updates, omits
the first from the speed estimate, and exercises checkpoint save/reload. It
does **not** establish sustained capacity, validation quality, convergence,
or inference capacity. Throughput is the summed global cells divided by the
slowest-rank measured update time; memory is the maximum allocated on either
card. The runs use the same dataset, global batch 128, seed, view/loss recipe,
and GPUs, but are sequential and may experience host/GPU timing variation.

| Frozen training SHA | Change | Cells/s | Median measured update | Peak allocated |
|---|---|---:|---:|---:|
| `cb517eac3930493bf56e6a1983193f16c8bce7a1` | original query chunk 8 and two KoLeo backward traversals | 4.561 | 27.50 s | 21.51 GB |
| `8ea0e27f6650c3d275e63c89be199a983c46bc31` | query chunk 32; equivalent matrix products | 4.583 | 27.80 s | 21.51 GB |
| `2eb4c4ed75dd44bd0d5f61c671b68b166970ddc7` | combine cell and global KoLeo into one backward | **6.511** | **19.36 s** | **25.91 GB** |

The final candidate is 42.7% faster in this short comparison. Its higher
activation lifetime costs about 4.40 GB/card. The attention-only change was
large in a synthetic layer microbenchmark but only 0.5% in the full update;
do not attribute the complete improvement to query chunking. Timed full-step
diagnostics on the intermediate source measured about 7.9 s cell forward,
22.5 s backward, 0.06 s graph-loss forward, and 0.02 s gradient reduction.
The four backward calls were approximately 9.6, 7.3, 5.1, and 0.6 s; the
third was the deferred KoLeo pass. The fused update removes that repeated
encoder traversal while preserving the same cell and KoLeo scalar terms.

The underlying dot products and loss sum are algebraically unchanged. CPU
float32 output/input/parameter gradients matched the prechange operator;
53 targeted component/distributed tests and the full 212-test v2 suite passed
for the fused engine. Ruff lint/format, an isolated wheel/sdist build, and a
two-GPU `gradpert train --dry-run` using the current default config passed.
Mypy still reports five errors in unchanged training-step/reduction code;
the one new redundant-cast error was removed in source `3c1a34d` with no
runtime operation change. BF16
matrix multiplication and changed query partitioning can change roundoff and
dropout draws, so bitwise-identical trajectories are not claimed. This is
training-system performance evidence, not a claim that validation metrics
or final model quality are unchanged.

## Batch boundary

| Per-rank microbatch × accumulation | Global batch | Evidence | Peak allocated | Result |
|---|---:|---|---:|---|
| 32 × 2 | 128 | 128 updates plus validation inference, frozen `0a9a67f` | 27.80/27.80 GB | passed; default scientific batch |
| 36 × 2 | 144 | 128 updates plus validation inference, frozen `0a9a67f` | 31.68 GB | passed sustained capacity |
| 38 × 2 | 152 | 5 updates, frozen `0a9a67f` | 32.42 GB | passed short test; only ~1.25 GB/card total headroom |
| 40 × 2 | 160 | one update passed, longer probe failed at update 3 | — | OOM; not capacity-valid |
| 48 × 2 | 192 | one-update attempt | — | OOM at update 1 |

The `performance_no_outer_checkpoint` compute-only candidate OOMed before its
first update at global batch 128. It is retained solely to reproduce the
negative result. Do not silently disable outer checkpointing in the default.
The highest **128-update validated** profile is batch 144. Its clean receipt is
`v2-m36-capacity-clean-0a9a67f/receipt.json`, SHA256
`aa78cce7b4312d7150744fdae25c25265b37a633cd828a954721327a26e2f1a4`.
Both cards completed 128 updates, checkpoint save/reload at step 64, and
300-control validation inference with output shape 300×5000 in 13.67 s. The
measured post-warmup throughput was 6.871 cells/s; total training wall time was
2,868 s. Peak allocated memory was 31.68/31.56 GB on GPUs 0/1. The reported
single-condition validation loss (0.01242) is only an engineering finiteness
check and must not be used to select a scientific model or hyperparameters.
Batch 152 has only a five-update pass with narrow headroom; no exact physical
maximum above 144 is claimed. The user-selected ablation default remains global
batch 128; a hardware capacity point is not an automatic experimental default.
The exact-configuration default batch-128 probe also passed on both GPUs at
source `0a9a67f272978476e104952f3a715fa41969105d`. Its clean receipt is
`v2-m32-capacity-clean-0a9a67f/receipt.json`, SHA256
`51323c14610b8094dbe25e73c9c6ce4663c3325498e9873db97b2f1e9a59b269`.
It completed 128 updates, checkpoint continuation at update 64, and
300-control validation inference with output shape 300×5000. Training took
2,657.49 s, with measured post-warmup throughput 6.622 cells/s; peak allocated
memory was 27.80/27.80 GB on GPUs 0/1. The single-condition validation loss
(0.009516) only checks that inference is finite; it is not a formal baseline
metric. The source checkout stayed clean at the pinned commit. This result
establishes sustained capacity for the selected batch, not five-epoch model
quality or best/last test performance.

The first 128-step batch-144 attempt was operator-stopped after 32 committed
updates: a packaging build had created ignored `src/gradpert.egg-info/` in the
active checkout after startup, changing the content-tree hash. Its
`v2-m36-capacity-0a9a67f/audit-stop.json` records the mismatch, and that
partial run is **not** capacity evidence. The replacement
`v2-m36-capacity-clean-0a9a67f` used a fresh checkout verified against the
same pinned publication receipt; no packaging or tests ran in that checkout.
Its post-run Git status was clean and the content-tree SHA matched the startup
receipt.

Receipt stems: `v2-throughput-baseline-cb517ea`,
`v2-throughput-optimized-8ea0e27`, `v2-throughput-fused-2eb4c4e`,
`v2-m36-throughput-0a9a67f`, `v2-m38-throughput-0a9a67f`,
`v2-fused-m40-throughput-2eb4c4e`, and
`v2-fused-m48-integration-2eb4c4e`.
