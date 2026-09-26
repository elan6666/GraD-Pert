---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: relay_performance_preflight
current_workflow: byte-auto
next_workflow: byte-auto
review_verdict: method_published_diagnostics_locally_validated
hard_blocked: false
updated_at: 2026-09-26
---

# Current state

Method published as `7e4c669e5043986209339a0c8a613b02645356d3`; Student27,279,662,
same frozen EMA Teacher. Full B0 retains prediction+SSL1+SSL2. Local acceptance
and disclosed historical failures: [receipt](../docs/experiments/relay-stage-a.json).

User restored server access and requested Goal supervision. Goal active; SSH
works again. Diagnostics published `d9c1fbfb4a7864be316426fd2a7b26797ad7b138` and verified
on clean server source. Supported server environment passed263 v2 tests;
dual-card complete update + resume passed at micro2/accum2/global8. Short final-
update CPU/GPU traces were exported, but summary parsing failed on CUDA-mirrored
annotations. Repair and reanalysis are in progress; no formal training or timer. GPU0 idle100%
telemetry is anomalous but both cards passed CUDA arithmetic; analyze traces. [Plan](../docs/design/GRADPERT_V2_RELAY_METHOD_PLAN.md) and STATE.md carry
remaining diagnostics→optimization→capacity→five-epoch-B0 dependencies. No new
CUDA throughput, capacity, or scientific results claimed.

The material below is historical status for earlier variants and stopped runs.

2026-09-25 method update: new Jurkat v2 config
`configs/v2/hamiltonian_mla_jurkat/gradpert_v2/nadig_jurkat.yaml` selects
three fixed bidirectional Hamiltonian expander cycles, two graph layers with
updated-neighbor propagation, and 2 KDA + 1 full MLA block in each Cell and
Response Encoder, with no DSA in the new profile. Old configs and stopped B0
remain unchanged. Full formulas and review issues are in
`docs/design/GRADPERT_V2_HAMILTONIAN_MLA_METHOD.md`. The inherited global
batch192 is **not capacity-certified for this changed architecture**; do not
start formal training before new two-card sustained capacity evidence, clean
source publication and method review. No training was restarted by this edit.
Local v2 regression: 230 tests passed; Ruff lint/format and wheel/sdist build
passed. This does not establish two-GPU throughput or capacity.
The four-fold fixed-axis cross-cell **design contract** now points to the same
new mechanism; it still lacks axis-specific prior/graph artifacts and a
validated training adapter, so no cross-cell GPU run is implied.

The old five-epoch Nadig Jurkat v2 ablation baseline was stopped at the user's
request on 2026-09-24. The subsequent compact v2 B0 formal run was also
**stopped at the user's request on 2026-09-25 02:28 +08**, before the first
epoch committed. It used the complete prediction + SSL1 + SSL2 model;
`prediction_only` was never launched. The compact-model capacity retest remains
valid engineering evidence, not a scientific five-epoch result.
The earlier stopped four-layer Top500/GenePT-PCA256/SwiGLU model, global batch
128, and two-GPU row-mean protocol are fixed at training source
`536458333e437252178ffe493c5c50c9064e7615` and config SHA256
`8471f5ea68f4801406497985291a0088116ff5a8435b82f85e55c611548b06c6`.
The exact-source two-GPU integration check passed before launch.

- Historical stopped run ID:
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
maximum between 192 and 208 is unmeasured. The user selected the passed
**global batch 192**, micro48/rank × accumulation2 × two ranks, for this new B0
formal run. The older stopped run remains historical evidence only.

Stopped compact B0 run ID:
`nadig_jurkat-seed1-20260924T134139Z-e5df6111138747e08ba5a582e37c1ed2`.
Run root:
`/data/yilangliu/GraD-Pert/runs-v2-b0-compact192-fc90a0d/` plus the ID.
Training source is clean published commit
`fc90a0d992373e619976504c82bd6f94c930d7b2`; config SHA256
`674ea9ab4160e10e85de7f6da78137257efee510c97e2f9852a474a55e81acf5`.
Exact-source two-GPU one-step integration passed before launch. At the user
stop, the epoch journal and history were **0/5** (749 planned updates per
epoch), with only the epoch-0000 initial checkpoint and no finite validation
selection. The targeted parent/torchrun processes received SIGTERM; parent,
wrapper, torchrun, and both ranks exited, and GPUs 0/1 returned to 2 MiB each.
No `COMPLETE.json` or best/last test receipts exist. The two-hour heartbeat
`grad-pert-v2-b0-compact192-formal` was deleted. Do not resume or overwrite
this run ID; no other ablation row was launched. The detailed ledger is
[STATE.md](STATE.md); capacity and throughput evidence is in
[GRADPERT_V2_GLM53_DUAL_GPU_PERFORMANCE.md](../docs/experiments/GRADPERT_V2_GLM53_DUAL_GPU_PERFORMANCE.md).

Historical R50 batch-1024 status from 2026-09-14 remains in Git history and
its dedicated experiment documents. It does not describe this v2 run.
