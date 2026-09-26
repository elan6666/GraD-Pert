---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: single_pass_performance_preflight
current_workflow: byte-auto
next_workflow: byte-auto
review_verdict: single_pass_locally_validated
hard_blocked: false
updated_at: 2026-09-26
---

# Current state

## Resumed single-pass performance engineering — 2026-09-26

Single-pass integration passed1/1 complete update on both GPUs with checkpoint
save/reload; receipt dry-run reviewed79,212bytes then copied to
`docs/experiments/single-c073a37-preflight/integration/receipt.json`.
Baseline12updates now running, profile follows. Finite successor synthetic
CUDA Graph probe queued behind both GPU locks: PID3400405, stem
`development/single-c073a37-cudagraph-check` with `.stage/.log/.pid/.exit`.
It requires all three preflight receipts passed and queue exit0 before GPU use.
This unchanged generic kernel tool checks two live final-state invocations and
carried-state gradients; it is not a model backend adoption or a two-pass model
training run. Final-state write implementation and model default stay eager.
Synthetic success alone will not establish end-to-end speed or update parity.

Current live bounded queue: `/data/yilangliu/GraD-Pert/development/single-c073a37-preflight`,
PID3399679; same stem `.sh/.pid/.log/.exit/.stage`. Stages: target30tests →
dual-GPU full-loss integration+checkpoint reload → 12update baseline (3warmup)
→ 4update bounded final-update profile. All stage success receipts gate the next.
Training SHA `c073a37c3c038561fb3d167b2a8f53f8650544bd`, immutable
`development/source-v2-single-c073a37`; publication
`development/gradpert-single-clean-publication-c073a37.json`, SHA256
`3287827f59329ea3be6185493c5e816e24d527027101cbad0277fbc1e7bf3ace`.
Fresh target tests30passed; integration processes verified alive. GPUs0/1 locked.
The first local publication receipt was rejected: ignored editable-install
`src/gradpert.egg-info` altered the local tree hash. Fresh clean local clone and
server now agree on tree a7d78edc8a11dc3f39a469247382dac0e612139bd7ed1ee47f25d57bfec52ca8.
Rejected receipt retained; no identity checks bypassed. Use clean publication
staging for subsequent releases, not the editable test worktree.

User explicitly resumed the original Goal after the single-pass correction.
The pause below is historical and superseded. Current method commit
`f6d84bebb3519a5cf94c91e38ea0b90854e1a952`: single random-order writing pass,
unchanged final-state readout. Resume only new run IDs; old dual-pass ABBA
and CUDA Graph queue remain stopped. SSH works; both5090s verified idle.
Next: publish the same method with a conservative micro2×accum2×world2=batch8
profiling config; verify fresh immutable server checkout and full-loss integration
with checkpoint reload. Then collect single-pass throughput/CPU-GPU timeline,
optimize evidenced bottlenecks with complete-update parity, and validate sustained
capacity before full B0 five epochs and best/last tests. No other ablations.
Goal active; no enabled automation. Old batch192 is not a certified new-method
capacity; base config batch values are placeholders until capacity acceptance.


## Latest user override — performance work paused; single-pass KDA

2026-09-26: Goal paused at user request. Stopped both verified process trees:
ABBA wrapper3386002 (including active B2 ranks) and waiting CUDA Graph
wrapper3389345, stopping the successor before freeing GPU locks. Post-stop
process inspection found no owned processes and nvidia-smi no compute apps.
Server evidence: `/data/yilangliu/GraD-Pert/development/user-pause-single-pass-20260926.json`.
A1/B1 completed receipts remain historical; ABBA is incomplete, B2 interrupted,
A2 and CUDA Graph probe have not been accepted as completed.

Only method change: `relay_passes: 1`, new self-contained current config
`configs/v2/single_pass_jurkat/gradpert_v2/nadig_jurkat.yaml`. All graph, cell,
response self/cross KDA write once, then queries read the final state. Self CLS
writes once at the end; graph target-only readout and cross control K/V remain.
Historical missing setting means two passes; old checkpoints/config identities
remain valid. Student parameter count unchanged at27,279,662; EMA same structure.
Local validation: 298 v2 tests passed (83.78s), plus the subsequently added
single/dual joint-loss parameterization passed both cases. Ruff and scoped mypy
passed. Receipt: `docs/experiments/single-pass-method-validation.json`.
New single-pass CUDA performance/capacity remains untested while paused.
Pre-change clean published baseline: f5267dfa3c34f3bf5993848d315ed4bbaf19c02b.
No new GPU test, capacity sweep, training or monitor is authorized to resume
by this correction. Earlier active/running statements below are historical.


Method published as `7e4c669e5043986209339a0c8a613b02645356d3`; Student27,279,662,
same frozen EMA Teacher. Full B0 retains prediction+SSL1+SSL2. Local acceptance
and disclosed historical failures: [receipt](../docs/experiments/relay-stage-a.json).

User restored server access and requested Goal supervision. Goal active; SSH
works again. Diagnostics published `d9c1fbfb4a7864be316426fd2a7b26797ad7b138` and verified
on clean server source. Supported server environment passed263 v2 tests;
dual-card complete update + resume passed at micro2/accum2/global8. Short final-
update CPU/GPU traces were exported, but summary parsing failed on CUDA-mirrored
annotations. Repair5186003 passed4 tests; both original traces successfully
reanalyzed with separate analysis SHA. About1.972million kernels/update suggests
small-op/chunk fusion as next diagnostic direction; profiled timings are not
normal throughput evidence. A1 passed40/40: median26.63s/update at global8. Candidate569ae1b passed
FP32 kernel checks but failed strict BF16 output tolerance; failure preserved,
default stays eager. Intermediate-rounding retry also failed; compilation is not adopted.
Independent per-layer graph validation candidate passes275 local v2 tests
including bitwise checkpoint/dropout gradient/RNG checks; full two-card
update parity exposed GPU repeatability differences. Published4898de3 reference-repeat
also fails strict gradient tolerance (max1.3163e-4), with exact inputs/RNG.
Deterministic reference/candidate diagnostic passed two complete updates on
both GPUs with bitwise-equal losses/gradients/model/optimizer and exact RNG.
Same-source ABBA throughput queue runs40updates each at global8. A1 passed,
median26.323s/update. B1 passed at25.983s/update; B2 automatically started.
Single-pair throughput differs by only1.33%; B2/A2 remain pending;
no throughput improvement or default adoption claimed. Next CUDA Graph replay
probe is published279696a, CPU helper checks pass, and queued behind ABBA with
explicit successful-receipt/resource gates; its GPU behavior remains unverified. No formal training or timer; Goal continues. STATE.md has receipts
and remaining gates. GPU0 idle100%
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
