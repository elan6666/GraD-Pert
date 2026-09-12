# R50 batch1024 independent factors

Status: implementation and pre-CUDA verification, not launched or completed.
Parent source: `7f31ba47af9f6ad54f3ff6d6d6f3fc0af514a6fc` (published main).

All rows independently inherit the resolved batch1024 reference configuration,
not the winner of another row. Preserve E3 initialization and its sealed prior.
No GenePT comparison, combined optimizer/schedule row, or cell-pool KoLeo run.
50 epochs, seed1, batch1024, evaluation batch256, no early stopping; automatic
test of validation-best and true-last checkpoints. Test scores never select
hyperparameters. Training/evaluation commits and checkpoint hashes are separate.

| ID | Only scientific factor changed |
| --- | --- |
| r1024_s1 | Endpoint LR warmup/cosine; peak .001, floor 1e-6, warmup .16 |
| r1024_s2 | Endpoint LR warmup/cosine; peak .001, floor .0002, warmup .16 |
| r1024_u1 | Existing GLM5MuonSplit_v1 optimizer, fixed baseline LR |
| r1024_t1 | EMA start .990, same end/schedule |
| r1024_t2 | EMA start .994, same end/schedule |
| r1024_p1 | Projector hidden 1024; bottleneck unchanged |
| r1024_p2 | Two graph layers; widths unchanged |
| r1024_c1 | Batch condition-equal expression MSE, gamma0, no directional term |
| r1024_l1 | Fanout instead of RingInduced, same half-node budget |
| r1024_l2 | All annotated essential runtime genes plus current anchors |
| r1024_l3 | RingInduced matched to L2 exact per-condition node count |

L2/L3 explicitly override the nominal half budget; no truncation of essential
genes. L3 fails if the anchor-reachable graph cannot provide the exact count.
Four locals remain; repeated Essential-only local sets are expected and receipted,
not replaced with artificial random variation. Non-membership in the annotation
does not establish biological nonessentiality. Frozen DepMap20Q1 annotation:
`9dd185edc9876e0dd77f405f0fb488c465d8557a3491619a47fc0d6ef779aee2`.

## Launch and monitoring handoff

Publish clean source and run exact server gates before any CUDA. First run every
coordinate for one real training update with full 50-epoch schedule horizon,
validation/test access guards and save/load checks. Up to four independent
processes across two GPUs, at most two per GPU, each capped at 40% allocator
memory. This leaves headroom but is not a guarantee of sustained capacity or
speedup. Preserve external jobs; use only available slots. No automatic retry,
configuration reduction, or overwrite on failure.

After reviewing successful receipts and actual overlap/resource telemetry,
`scripts.server.run_r1024_full` runs a fresh formal root followed immediately by
both checkpoint tests. Keep the same two slots per GPU across fit and evaluation;
do not spawn extra evaluation processes outside that limit. Failed rows remain
failed evidence and need a fresh reviewed lineage; safe rows may proceed.

Monitor `grad-pert-r50` every 30 minutes, succinct progress each check. Inspect
processes, exit receipts, compact logs, source/input identity, resources and
whole-root PKL count once. At terminal stages pause the monitor before a bounded
analysis/preparation goal. No goal during CUDA waiting. No busy polling.
Do not call a one-step receipt a completed 50-epoch result.
