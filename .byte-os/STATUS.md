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

Sequence replay candidate published `3c308ddb5d163659cfbd87e325a92d1ea191bd64`;
clean immutable server `development/source-v2-replay-3c308dd` verified against
clean local publication. Publication `development/gradpert-replay-publication-3c308dd.json`,
SHA256 d7fbdbd15c8cd2e6dde21e09ba1b328c091e51d95ea67390561eff18de011e26.
Live bounded parity wrapper PID3407939, stem
`development/single-3c308dd-replay-parity`; `.sh/.pid/.stage/.log/.exit`, diagnostic
log `.run.log`, per-rank receipts inside stem directory. Both GPU locks, timeout
1200s; target3routing tests then deterministic2update reference/candidate check.
Reference profiling_m2_a2 and candidate replay_m2_a2, same single-pass method,
global8. Comparison includes nonzero-LR update2, all gradients/optimizer/EMA/
centers, inputs and RNG. No throughput queue or formal training launched until
candidate terminal evidence is inspected. Recompilation/cache/OOM failure is
candidate evidence, never grounds to silently relax precision or change views.

Opt-in sequence CUDA replay candidate prepared (relay_kernel=cudagraphs),
self/cross gene scans and CLS write only; graph neighborhoods remain eager.
Unchanged eager arithmetic inside compiled cudagraphs backend, no fusion.
Optimizer-boundary step markers only, cloned final outputs retain storage
ownership; finite64 shape-recompile budget, capture skips rejected. Variable
view shape capture costs/memory remain unvalidated and may reject the candidate.
Default remains eager; config single_pass_jurkat/replay_m2_a2 is engineering-only.
38 targeted tests passed, scoped mypy4files and ruff passed. CPU tests validate
dispatch/eager equivalence; actual replay proof requires target GPU parity.
Next publish clean source and run deterministic two-update dualGPU parity,
including nonzero-LR step2 and actual graph/skip counters. No throughput or
formal adoption before full-update and sustained memory evidence.

Replacement queue3404789 completed exit0: profile4/4 passed; synthetic
CUDA Graph12/12 cases passed, no skipped graphs or recaptures during timing.
FP32/BF16,grad/no-grad,live carried states and all five input gradients checked
at unchanged3e-5/3e-4 tolerance; max observed absolute error2.33e-10, RNG exact.
B2,T257 speedup4.83–5.87x; B64,T94 only1.008–1.046x; CLS-likeT1 2.43–3.08x.
These are synthetic paired-call timings, NOT model/update speedup or proof of
full-model parity. Small receipts/summaries dry-run (111900+21401bytes) then
copied to `docs/experiments/single-38af3ce-profile-replay/{profile,replay}/`.
Both GPUs verified no compute apps after completion. No monitor or GPU job now.
Next active engineering: opt-in model replay candidate, preserving eager default;
handle variable view shapes, live output/state lifetimes, checkpoint recomputation,
full nonzero-LR update/EMA/center/RNG parity and graph cache memory before any
capacity/formal run. Graph-neighborhood replay alone has weak speed evidence;
prioritize sequence KDA and ensure capture overhead does not erase gains.

Supersedes pending c073a37 capture/replay queue: explicitly stopped diagnostic
postprocessing (raw traces+summaries preserved, no terminal profile pass) and
its waiting successor. Both process trees gone, GPUs empty; stop evidence
`development/single-c073a37-postprocess-interrupted.json`. Integration and
12-step baseline remain passed. No training checkpoint/run was restarted.
Replacement immutable source `development/source-v2-profile-38af3ce`, SHA
`38af3ce7a9b00ba4f8876af6e9521c49cb555edb`, clean local/server/publication identity
verified. Model/config tree identical to c073a37, a7d78edc8a11dc3f39a469247382dac0e612139bd7ed1ee47f25d57bfec52ca8;
only diagnostic script behavior changed. Publication
`development/gradpert-profile-publication-38af3ce.json`, SHA256
`79ecb1e787512e32f78eae81f1b7a4ed5fa61096fa8ae0be0a9d930e1443f19a`.
New bounded queue PID3404789: `development/single-38af3ce-profile-replay`,
`.sh/.pid/.stage/.log/.exit`. Target12tests → dualGPU4step profile (900s timeout)
→ synthetic GPU0 replay (1800s timeout), both GPU locks held. Child outputs
append `-profile`/`-replay`; no detailed operator aggregation requested.
Inspect this queue, not stopped predecessors. Goal active, no automation.

Profiler postprocessing issue observed: after ~4.77GB/rank trace and small
summary had been saved, lazy key_averages aggregation ran >5minutes at one CPU
core/rank and ~49millionKiB RSS/rank. Host still had ample available RAM;
original processes kept intact under2400s wrapper timeout. Do not label hung
or terminal solely because .stage/receipt has not advanced.
Prepared diagnostic-only repair: detailed operator table is now opt-in via
`--profile-operator-table` (requires profile-last-update). Default still exports
raw trace, region summary and GPU interval union, without key_averages. No
model, loss or optimizer change. Tests5profile+7probe-policy pass; profile test
checks complete update/RNG equality, default forbids expensive aggregation,
explicit mode exports table. Scoped mypy2files and ruff pass. Applies only to
future immutable source; current c073a37 capture is not modified.

Both single-pass trace summaries have been generated and copied after bounded
dry-run (12,618bytes), `docs/experiments/single-c073a37-preflight/profile/`.
Rank0/1 kernel counts1,414,510/1,414,379 in ~31.51s profiled windows;
GPU interval-union fractions0.16365/0.20411 include profiler overhead, not normal
utilization. Graph forwards ~5.19/5.01s student plus3.18/3.01s teacher;
backward18.52/17.74s CPU-inclusive; gradient reduction14–16ms. Nested times
must not be summed. Evidence prioritizes launch overhead/KDA regional replay.
Raw traces (~4.77GB/rank) remain server-only. Profile workers3401379/3401380
are still live in CPU summary/operator aggregation, not terminal success;
queue retains locks and the synthetic replay successor remains gated.

Single-pass baseline passed12/12 (3warmup,9timed), source c073a37 unchanged.
Median update19.57964s; p9519.83892s; measured cells/s including data0.387596;
peak allocated3,505,913,344bytes. Global8 is diagnostic only. Data wait1.09–1.45s
per update is a modest fraction; prioritize KDA CPU launch/synchronization
investigation before assuming prefetch alone solves the bottleneck. Method
change from dual to single pass is not an equivalent-implementation speedup.
Receipt dry-run88,894bytes then copied to
`docs/experiments/single-c073a37-preflight/baseline/receipt.json`.
Queue has transitioned to final-update profiling; wrapper3399679 live. Replay
candidate wrapper3400405 still waits on GPU locks and verified predecessor gates.

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
