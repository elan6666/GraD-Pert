---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: experiment_supervision
current_workflow: byte-auto
next_workflow: byte-auto
review_verdict: pending_formal_best_last
hard_blocked: false
updated_at: 2026-09-24T04:00:00+08:00
---

# Current state

The active GraD-Pert v2 work is the five-epoch Nadig Jurkat ablation baseline.
The complete Top500/GenePT-PCA256/SwiGLU model, global batch 128, and
two-GPU row-mean protocol are fixed at training source
`536458333e437252178ffe493c5c50c9064e7615` and config SHA256
`8471f5ea68f4801406497985291a0088116ff5a8435b82f85e55c611548b06c6`.
The exact-source two-GPU integration check passed before launch.

- Active run ID:
  `nadig_jurkat-seed1-20260923T194747Z-1b7eda2abd6441f592d0834e1e275e88`.
- Server run root: `/data/yilangliu/GraD-Pert/runs-v2-glm53-current/`
  followed by that ID. The training log is
  `/data/yilangliu/GraD-Pert/development/v2-jurkat-baseline-5364583.log`.
- Last checked: process alive, epoch state 0/5, GPUs 0 and 1 occupied by the
  training job. `COMPLETE.json` and best/last test receipts were absent.
- Existing heartbeat: `grad-pert-v2-batch128`, retargeted to this baseline,
  every two hours with a brief report on every check. It should inspect live
  process, epoch state, logs, GPUs, and terminal receipts; a missing process
  alone is not completion evidence.

Next dependency: after five committed epochs, verify both checkpoint roles and
their actual test receipts, then record comparable metrics and provenance.
If a failure creates an in-scope repair, record and resolve that dependency
before returning to the baseline; retarget the same monitor to the live stage.
Do not silently start another ablation group. The detailed active ledger is
[STATE.md](STATE.md); capacity and throughput evidence is in
[GRADPERT_V2_GLM53_DUAL_GPU_PERFORMANCE.md](../docs/experiments/GRADPERT_V2_GLM53_DUAL_GPU_PERFORMANCE.md).

Historical R50 batch-1024 status from 2026-09-14 remains in Git history and
its dedicated experiment documents. It does not describe this v2 run.
