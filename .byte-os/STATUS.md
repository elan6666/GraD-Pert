---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: compact_v2_capacity_complete
current_workflow: byte-auto
next_workflow: byte-auto
review_verdict: published_and_dual_gpu_capacity_passed
hard_blocked: false
updated_at: 2026-09-24T20:50:00+08:00
---

# Current state

The five-epoch Nadig Jurkat v2 ablation baseline was stopped at the user's
request on 2026-09-24. The active work is a new compact v2 model and dual-GPU
capacity retest, **not** a replacement formal five-epoch training run.
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
- Stopped state: all training parent/rank processes exited and both GPUs
  released this run's memory. **1/5 epochs committed**. Epoch 1 completed
  1,081 updates and selected `epoch-0001.pt` as provisional best/last by
  validation prediction loss `0.0052692545`. Validation Pearson values:
  TxPert `0.142970`, TriShift `0.187574`, Systema `0.067022`. These are
  validation metrics, not test results. `COMPLETE.json` and best/last test
  receipts remain absent. The interrupted epoch 2 has no committed receipt;
  do not report this run as a completed five-epoch baseline.
- The `grad-pert-v2-batch128` heartbeat was deleted after the requested stop.

New design: both Cell and Response Encoders have two KDA layers followed by one
DSA/MLA layer; all four Student/Teacher distillation heads use 8192 prototypes.
The Student count is 23,123,847. An isolated server CPU snapshot passed 227 v2
tests; lint, format, and package build passed. The scoped change is published
as `974c5eadf35a95fbc3ba46cffe6d9ab34cdd4e94`; its clean immutable server
checkout and publication receipt passed the formal source-identity check.
The default global-batch-128 two-GPU one-step integration and checkpoint reload
passed. Short probes passed global batches 192, 208, and 224; global 240 OOMed
at step 3. The full 128-step probes at 224 and 208 OOMed after five and 23
completed steps, respectively; neither is a sustained-capacity result. The
128-step dual-GPU probe at global 192 **passed** with checkpoint continuation
and 300×5000 validation inference. Its clean receipt is
`/data/yilangliu/GraD-Pert/development/v2-compact-974c5ea-capacity-m48-128/receipt.json`,
SHA256 `71d952f90466052a52855a819090d9c5e03ba3e96081eadc03ac2cd3c896529d`.
Measured throughput was 8.486 cells/s after warmup, 7.992 cells/s end to end;
peak allocated memory was 31.01/31.05 GB on GPUs 0/1. Thus 192 is the highest
**sustained-validated** batch under this protocol, while the exact physical
maximum between 192 and 208 is unmeasured. The scientific default stays 128.
The stopped five-epoch run remains
historical evidence only; do not resume it or start an ablation group. The detailed ledger is
[STATE.md](STATE.md); capacity and throughput evidence is in
[GRADPERT_V2_GLM53_DUAL_GPU_PERFORMANCE.md](../docs/experiments/GRADPERT_V2_GLM53_DUAL_GPU_PERFORMANCE.md).

Historical R50 batch-1024 status from 2026-09-14 remains in Git history and
its dedicated experiment documents. It does not describe this v2 run.
