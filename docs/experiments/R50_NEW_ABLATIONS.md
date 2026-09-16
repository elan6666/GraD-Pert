# R50 new ablations, 2026-09-16

Authorization: nine requested ablations, fresh roots, 50 epochs, one project
experiment per GPU on two GPUs. Each eligible row first passes one real update
with checkpoint roundtrip and resource evidence. A step is not sustained capacity
or scientific completion. Best is minimum validation prediction loss; retain and
test both best and epoch-50 last. No early stop or test-based selection.

Parent is the existing independent bf938 batch1024 / LR=.001 / AdamW repeat.
Its historical Pearson checkpoint selection is not relabeled loss selection.
All new rows use seed 1, eval batch256, same canonical splits/controls/prior,
4 graph layers, projector hidden2048/bottleneck256/K16384 unless named below.
These are paired configuration ablations, not independent statistical seeds.

| Row | Change | Eligibility |
| --- | --- | --- |
| r50n_t1_u1 | EMA .990 plus existing GLM5MuonSplit_v1 optimizer | implemented combination, not single-factor |
| r50n_batch2048 | train batch2048, LR remains .001 | implemented; equal epochs, fewer optimizer updates |
| r50n_k32768 | prototype count32768 | implemented |
| r50n_k8192 | prototype count8192 | implemented |
| r50n_string | STRING only, unchanged node axis and sparse Transformer | implemented |
| r50n_source_gat | existing native adaptive_source_gat_fusion, dropout .2 | native architecture package, not official GAT-Hybrid parity |
| r50n_prediction_only | auxiliary loss weights all zero, supervised graph prediction only | no teacher/projector forward, EMA, centers or augmented views |
| r50n_gat_mlg | paper-defined supra-graph GAT (GAT-MLG) | implemented natively from the published supplement S1.3 definition; see `R50_GRAPH_ENCODER_DEFS.md`; queued after all existing lanes |
| r50n_hybrid_bmp | paper-defined bidirectional message passing | implemented natively from the published supplement S1.3.2 definition; see `R50_GRAPH_ENCODER_DEFS.md`; queued after all existing lanes |

The source-GAT row uses the existing preregistered native per-source towers and
node-adaptive fusion, including its .2 dropout contract (baseline .1). It is an
architecture-package comparison, not a pure single-operator change. No frozen
official source is copied or imported. MLG/BMP remain visible but cannot launch.

Prediction-only preserves dormant checkpoint teacher/projector tensors for
serialization compatibility. They do not affect prediction, receive gradients,
or update. Report total and active parameters separately when comparing compute;
do not claim physical removal or maximum memory optimization.

Queue: first complete all eligible 1-step checks, two workers, one per physical
GPU. Then each lane trains its successful rows and immediately runs best/last
tests. Failed rows are preserved and not retried automatically. Existing default
15-percent/4-GiB headroom and 40-percent allocator cap remain intact; single-GPU
scheduling is not permission to relax a gate. No occupied GPU is used at entry.
Monitor checks new training attempts every10min for the first hour, then hourly;
new later starts reset the intensive window. Only this queue owns new runs.
