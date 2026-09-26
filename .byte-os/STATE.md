## Resumed single-pass performance engineering — 2026-09-26

ABBA stopped terminal exit1: A1 passed12/12, B1 failed10/12 on both ranks,
Dynamo recompile_limit64 exceeded at chunk_delta_final_state; no B2/A2 launched.
Full-sequence static replay is rejected for current random-length views; do not
raise cache limit and repeat or adopt synthetic speedups as training evidence.
Small reviewed receipts saved docs/experiments/single-3a2980d-replay-abba/{A1,B1}/
(88,989+75,379bytes); failure traceback remains server *-B1.log. GPUs cleared.
Immediately launched independent checkpoint-removal parity PID3418239, stem
`/data/yilangliu/GraD-Pert/development/single-473466d-checkpoint-parity`;
source/publication473466d-standalone as below, bothGPUlocks,1200s timeout.
.stage/.run.log/.exit/.pid and rank receipts at stem/. Both configs identical
eager profiling_m2_a2, only candidate sequence checkpoints disabled; graph
checkpoint retained. Deterministic two-update check, no throughput claim.
Default eager/checkpointed unchanged. Inspect this result before new GPU job.

Next diagnostic source ready (not launched while ABBA owns GPUs):
`/data/yilangliu/GraD-Pert/development/source-v2-replay-473466d-standalone`,
SHA473466d0b373b5486ed024adab69e538d18b6e16, publication
`development/gradpert-replay-publication-473466d.json` SHA256
d6b28c717edb24081ec1dafb2cde5bbc3a71a11d5b871c133511a55765fdf796.
Clean identity verified. Uses --candidate-no-sequence-checkpoint with identical
profiling_m2_a2 configs, --deterministic; reference checkpointing retained.
Previous attempted shared clone source-v2-replay-473466d failed checkout because
Git alternates nesting exceeded depth; preserved unused, never launched. Repaired
by24MiB full source-only bundle and independent clone. No active source altered.

ABBA live at B1 after A1 passed12/12. A1 median17.1801s, inclusive0.43615cells/s,
peakallocated14,551,056,384bytes (13.55GiB), default checkpointed priorbaseline
19.5796s /3,505,913,344bytes. This is one reference observation, not ABBA outcome.
Prepared separate candidate-only checkpoint-disable parity flag, requiring
identical eager architecture/config on both sides and rejecting reference-repeat
or both-sides overrides. This checks against the real checkpointed default;
the previous replay parity disabled checkpointing on both sides.8 helper/CLI
tests plus scoped lint/typecheck passed. GPU validation of this new diagnostic
must wait for current ABBA to release GPUs; no duplicate job launched.

ABBA throughput diagnostic launched PID3413600, stem
`/data/yilangliu/GraD-Pert/development/single-3a2980d-replay-abba`.
Immutable source `development/source-v2-replay-3a2980d`, fullSHA
3a2980d2eee30faa1ad99fb4481262701879d262, publication
`development/gradpert-replay-publication-3a2980d.json` SHA256
a19bc42a1af60768863314c2ba640a8847f8bf1d905f9825d387d752682650f6.
A1/B1/B2/A2: eager/replay/replay/eager, all no-sequence-checkpoint override;
graph checkpoint remains. Global8=micro2*accum2*2GPUs. Each12 updates,
3warmup+9timed, timeout1200s each, bothGPUlocks. Logs/receipts: stem-label.log
and stem-label/receipt.json; wrapper .stage/.pid/.exit. Stops on failure.
14 targeted tests passed locally and scoped mypy/ruff passed. Inspect live
result, compare schedule/view hashes, timings and peak memory before adoption.
This diagnostic is not sustained capacity or formal training. Goal stays active.

Checkpoint-isolation diagnostic PASSED on both ranks, exit0, GPUs released.
Source a63ac39a3b9237cf57c5782bcc7f5939fa6e2edb; two full updates (second
nonzero LR), exact input/RNG identity, all compared losses/gradients/objective/
optimizer max_abs0. Each rank captured10 graphs, skipped0. Receipts:
docs/experiments/single-a63ac39-replay-no-sequence-checkpoint/ (41,598byte
reviewed transfer). Scope: BOTH sides disable Cell/Response checkpointing;
graph checkpoint remains. This isolates the failed checkpoint/replay combination,
not proof of a production speedup or capacity. Default remains eager.
Next: bounded throughput measurements with identical checkpoint override on
reference/candidate, then compare memory/throughput to checkpointed baseline.
Probe override restricted to benchmark-only and explicitly receipted; no formal
capacity/training can silently use it. Goal active, no timer or formal B0 yet.

Checkpoint-isolation diagnostic now running, PID3411374, stem
`development/single-a63ac39-replay-no-sequence-checkpoint`; same .stage/.run.log/
.pid/.exit and rank-receipt layout,1200s timeout, bothGPUlocks. Clean source
`development/source-v2-replay-a63ac39` at a63ac39a3b9237cf57c5782bcc7f5939fa6e2edb.
Publication development/gradpert-replay-publication-a63ac39.json SHA256
c2652c2080543178f443eb07ca98f66acf17dc2599dd7a505f84401956e8f8a0.
Both reference/candidate sequence checkpoints disabled by explicit diagnostic
flag; graph checkpointing retained. Formal configs unchanged. Six parity-helper
tests passed locally (corrects preceding count typo). Outcome pending.

835a9ad donation-disabled candidate also FAILED on both ranks with exactly
same cudagraph lifetime invariant in first cell backward; exit1, no candidate
update passed. Receipts saved under docs/experiments/single-835a9ad-replay-parity/
after35,700byte dry-run. Buffer donation alone is not the fix.
Next diagnostic isolates checkpoint interaction: --no-sequence-checkpoint on
update_parity disables Cell/Response checkpointing on BOTH reference and
candidate, graph checkpointing unchanged. Explicit diagnostic receipt flag;
formal configs/runners untouched. This is fault isolation, not adoption or a
claim of faster training.6 parity helper tests and scoped mypy/ruff pass.

Ownership-repair hypothesis published835a9ad663ca0546b2cf36da02d1df69a1dc9521,
clean immutable `development/source-v2-replay-835a9ad`, tree
b3103a1d8457898db791584716a8abcefbdcf1ab6acfa0f5f9184b932eb74ccd;
publication `development/gradpert-replay-publication-835a9ad.json` SHA256
64ed51fa9535eaf91f48d984b7dd728b408bd1ad12df6e26d5b404532e12a735.
New parity wrapper PID3409783, stem `development/single-835a9ad-replay-parity`;
same .stage/.run.log/.pid/.exit and rank receipt layout,1200s timeout/bothGPUlocks.
Only donation policy changes relative to failed replay candidate. Reference
still eager, checkpointing unchanged. Inspect terminal result before deciding
next optimization; do not repeat unchanged failing configuration. No model
optimization adopted and no capacity/formal job yet.

3c308dd sequence replay full-update candidate FAILED on both ranks during
first accumulated cell backward, before any candidate update comparison:
"graph recording observed an input tensor deallocate ... did not occur during
replay" in PyTorch cudagraph_trees.check_invariants. Exit1, processes gone,
GPUs clear. Both failure receipts dry-run35,700bytes then copied to
`docs/experiments/single-3c308dd-replay-parity/`. Do not adopt or call parity passed.
All303 local v2 tests passed; that does not cover CUDA storage lifetimes.
Inspected installed PyTorch runtime and functorch donated_buffer config.
Next bounded repair hypothesis: disable saved-buffer donation during regional
capture to stabilize checkpoint-recomputation ownership. Same method/tolerances,
checkpoint setting/views/losses unchanged.9 targeted tests plus scoped mypy pass;
new test asserts policy restoration and independent differentiable output copy.
GPU rerun still required; disabling donation is not yet established as a fix.

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

# Current stage: performance instrumentation and target preflight

User explicitly requested Goal-mode supervision on 2026-09-26 after restoring
VPN. Goal is active for performance diagnosis/optimization and the dependent
capacity/full-B0 workflow; no scheduled monitor. The first GPU profile capture is complete and its failed summary parser has been repaired; both saved traces were reanalyzed.
Worktree `/Users/elan/code/grad-pert-v2-build`; preserve dirty root and old runs.

Method stage A published: `7e4c669e5043986209339a0c8a613b02645356d3`.
Local acceptance: `docs/experiments/relay-stage-a.json`; method/plan and stage B
tool details: `docs/design/GRADPERT_V2_RELAY_METHOD_PLAN.md`.
Student27,279,662, Teacher same frozen. New graph-local relay KDA, response
self→cross, full prediction+SSL1+SSL2. v2 253+1 tests passed; same5 historical
failures reproduced on baseline; no new regression. Local Python3.11 remains
separate from supported target environment evidence.

Stage B tools prepared locally: bounded last-update profiler, nested phase
labels, GPU busy interval unions, optional memory trace, per-rank timing/RSS,
ordered batch plan/RNG hashes, custom warmup and no default mid-step barriers.
49 targeted tests plus5 additional CLI rejection tests pass; scoped tool mypy
and all-tree lint/format pass. Profiled complete optimizer/EMA/center/RNG update
matches unprofiled. Published diagnostics/guidance SHA `d9c1fbfb4a7864be316426fd2a7b26797ad7b138`; clean local/GitHub/server identity verified.
First probe config is full-size model at micro2×accum2×world2=batch8. This is
only a conservative diagnostic candidate, not a new scientific default.

Fresh SSH is authenticated. Publication receipts must come from a clean
worktree: local build egg-info contaminated the old temporary7e4c669 receipt;
that receipt is invalid for server comparison. d9c1fbf uses the clean receipt.

Server supported environment Python3.12.3/Torch2.13.0+cu130 passed all263 v2
checks. Both GPUs passed tiny CUDA arithmetic/synchronization; GPU0 idle100%
telemetry remains anomalous, so use trace timings, not utilization alone.
Immutable source `/data/yilangliu/GraD-Pert/development/source-v2-relay-d9c1fbf`.
Publication `/data/yilangliu/GraD-Pert/development/gradpert-relay-publication-d9c1fbf.json`,
SHA256 `0d13c02e2bf8a5599da7a41dc2d184511b9c025aa524031d83f5bc1eac471eee`.
Config m2/a2 SHA256 `069251ca1d0cf3dbd16f73466d3621a74894a57ce8df306203c5db81f1ca8692`.
Dual-card integration passed1/1 with checkpoint reload; receipt
`/data/yilangliu/GraD-Pert/development/relay-d9c1fbf-m2-integration/receipt.json`,
SHA256 `e2f84d94521a8fb484d5c715c2f6956fc905ed36b05be37bab9e6c95097e5a7c`.
Cold update28.942s/data4.386s/peak3.313GB; not steady throughput or capacity.

Failed 3-update diagnostic (last update captured with allocation events):
`/data/yilangliu/GraD-Pert/development/relay-d9c1fbf-m2-profile`;
log/PID/exit use same stem with .log/.pid/.exit, own GPU0/1 locks,600s time cap.
Both7.5GB traces were exported, then trace_summary rejected the CPU capture
annotation plus CUDA-mirrored gpu_user_annotation as two windows. Original
rank failure receipts/logs and traces remain unchanged; no active GPU process
or formal run. Fix selects only the CPU window and excludes GPU mirrored
regions from CPU-inclusive totals, with a synthetic mirrored-event regression.
Repair published as5186003d54646d1f7f45723cc9617076349427c9;4 targeted tests
including complete-update/RNG parity passed. Both existing traces reanalyzed
successfully, in development/relay-d9c1fbf-profile-reanalysis-5186003, with
capture and analysis SHAs separated. Small summaries dry-run-reviewed and
copied to docs/experiments/relay-profile-d9c1fbf/ (104534bytes total).
Original status remains failed(profile_only),3/3 updates completed,105.118s
training wall including cold/profile overhead. Do not retrofit successful run.
Rank0/1 windows43.645/43.665s; GPU interval union22.207%/21.988%; kernel
counts1,971,993/1,972,124. These include profiler overhead, not normal utilization
or speed. Rank0 nested CPU graph Student7.357s,Teacher4.233s; backward26.456s;
data materialization.012s,assemble1.173s,gradient average.017s. Elementwise
multiply alone212752calls/2.721summedGPUseconds. Prioritize measured small-op
fusion/chunk kernels and recompute overhead before generic data-loader tuning;
first obtain unprofiled steady baseline, then full-update numerical/gradient/RNG
checks and controlled ABBA. Unprofiled baseline A1 now active:
`/data/yilangliu/GraD-Pert/development/relay-d9c1fbf-m2-steady-A1`;
PID3371948, .pid/.log/.exit/.sh same stem, dual exclusive locks,3600s limit;
40 full updates with10 warmups/30 measured. Original immutable d9 source/config.
No formal training. Check its live receipt and process before admitting another
GPU job. Opt-in compiled chunk candidate published and deployed; existing defaults
remain eager.269 distinct local v2 tests passed (268 suite +1 additional policy
test), four-source mypy and lint pass. Candidate source569ae1bbda8fe905fd53e8dd2208c27354d1a550 deployed clean to
`/data/yilangliu/GraD-Pert/development/source-v2-relay-569ae1b`; tree
07bd9659a9a8a5769d62ae61695202ce84c5864a3e9e0986ac139e8473ef1c66.
Publication development/gradpert-relay-publication-569ae1b.json hash
8f67ad305a23dbcc3996e9ca26c9c5accea50784c9ae3fed4418d3d0944cdb46.
Baseline A1 passed40/40; original queue and kernel-check processes both
terminal. Measured median26.634557s,p9527.352516s,including-data.2887537cells/s,
peak allocated3,547,394,560/reserved4,011,851,776bytes. Small receipt copied after
dry-run to docs/experiments/relay-steady-d9c1fbf-A1/receipt.json, SHA256
1159d98616a01a0e877802f881bf0c0b4ac4f966991b26a9dd57808d8d27b2a7.
Kernel probe569ae1b failed BF16 B64/L94 after FP32 cases passed;71 output elements
exceed3e-5/3e-4, max absolute.00048828125. No threshold relaxation; default eager.
Failure receipt docs/experiments/relay-kernel-candidate-569ae1b/receipt.json hash
ffc60bfa0aacbdc3f16782dc54bb5bedae98d5c12ecf37d9c314df60ef3efca4.
30a597a75ddfa10fced9b5a503c5699cadde3222 tested the same kernel with
emulate_precision_casts=True; same71 BF16 mismatches. It is rejected too.
Server source development/source-v2-relay-30a597a remains immutable;
publication91ca4093766941f736c65c158cb5d07c98734a5094c8dfead39654e7aec425e4.
Both development/relay-30a597a-kernel-check and dependent update-parity wrappers
exited1; dependency gate prevented the latter from running. No GPU work active.
Do not retry these output IDs or relax tolerances. Compiler remains opt-in,
not accepted/default. Next independent candidate: relay_validate_once=True,
hoists graph-neighborhood legality checks outside per-chunk/recompute calls,
without changing floating operations, RNG or losses. Default remainsFalse.
CPU checkpoint/dropout bitwise-output/gradient/RNG tests and invalid-neighbor
checks pass; full local v2 suite275passed, scoped4file mypy/lint pass.
New validated_m2_a2 config differs only this flag from profiling_m2_a2.
Update-parity tool supports exactly one of kernel or validation execution
changes, using initial.pt+same RNG/data and original LR/EMA for two full steps;
checks gradients/losses/Student/Teacher/centers/optimizer/RNG. 3 helper tests
passed; first actual two-card diagnostic was published as09845938ff1e0caa8e6b94ee40d52bf536adf1f5
and launched in development/relay-0984593-validation-parity (PID3380495, now
terminal exit1). Failed BEFORE any optimizer update while hashing a zero-stride
singleton Long tensor: contiguous() can preserve a singleton stride0, so viewing
as bytes fails. Initial checkpoint/log/rank receipts retained. Diagnostic fix
forces contiguous-format clone before byte view, covered by a new int64/bfloat16
singleton+broadcast test;4 helper tests/mypy/lint pass. Model source unchanged.
Fix published8aae56d6e383b2b250eb8b2fdcfc186cd537709b, deployed clean
source-v2-relay-8aae56d; publication hash23a3f9a11ca3078b1a2a10b5150a3526001423837070f2903dc796e323d590b0.
Development/relay-8aae56d-validation-parity failed after candidate step1:
inputs+all RNG digests exact, losses/objective parameters exact; gradients max
.00011331774294376373 and optimizer max.000011331867426633835. Failed gradient
paths include control_cls, graph.embedding and graph.layers.0 read projections.
Do not call this passed. Dependent relay-8aae56d-m2-validated-B1 was gated off;
no throughput measurement ran. Both wrappers terminal, GPUs idle.
Next diagnose numerical repeatability WITHOUT changing tolerance: add explicit
--reference-repeat (identical config/architecture enforced) and --deterministic
(diagnostic-only deterministic algorithms, disabled autograd multithreading,
CUBLAS_WORKSPACE_CONFIG=:4096:8). Test unchanged reference twice in normal mode,
then candidate vs reference in deterministic mode. If nondeterministic scatter/
GEMM explains differences, record that evidence; it is a hypothesis for now.
These flags never alter formal training defaults or benchmark settings. Five
parity-helper tests, scoped mypy/lint pass. Published4898de30dbb8ccdc09ce5bcbc8938ae2b4992cc1;
clean source development/source-v2-relay-4898de3, publication SHA256
f08a9de3ec3c48e8b5c40132bb31ee60e5d27d0a4c0b0d22436cb32f2778c231.
Reference-repeat development/relay-4898de3-reference-repeat exited1: both ranks
have exact inputs/RNG/loss/model state but gradient max1.31627545e-4,
optimizer max1.31627312e-5. The original implementation itself is not strictly
repeatable in normal GPU mode; specific nondeterministic operator unresolved.
Small rank receipts copied after dry-run (44506bytes) to
 docs/experiments/relay-parity-4898de3/reference-repeat/.
First update LR is zero from the sealed warmup, so exact updated parameters alone
are not evidence of update equivalence. Inspect nonzero-LR step2 as well.
Deterministic diagnostic development/relay-4898de3-validation-deterministic
passed2/2 on both GPUs, exit0. Every compared loss/gradient/objective/optimizer
tensor is bitwise identical; inputs and all RNG hashes exact. It includes a
nonzero-LR second update. Small receipts copied after dry-run (41370bytes) to
 docs/experiments/relay-parity-4898de3/validation-deterministic/.
The validation-hoist candidate passes controlled mathematical/full-update
checks; performance benefit not established, defaults unchanged.
CURRENT finite ABBA queue development/relay-4898de3-validation-abba,
PID3386002, same-stem .sh/.log/.pid/.exit/.current; exclusive GPU0/1 throughout.
Order A1,B1,B2,A2; child roots append -A1/-B1/-B2/-A2, with .log/.exit files.
A uses profiling_m2_a2, B validated_m2_a2; same immutable4898 source, global8,
full prediction+SSL1+SSL2, normal GPU mode,40 updates each (10warmup+30timed).
A1 passed40/40, update median26.3230s,including-data0.2915533cells/s; small
receipt copied after dry-run to docs/experiments/relay-abba-4898de3/A1/receipt.json.
B1 passed40/40: update median25.9832s,including-data0.2954293cells/s; receipt
copied after dry-run to docs/experiments/relay-abba-4898de3/B1/receipt.json.
A1/B1 source/environment/data/ordered batches/view RNG matched exactly; B1
configuration hash verified. Single-pair throughput ratio1.01329 is small and
not yet evidence of stable improvement. B2 automatically started; A2 follows.
Do not adopt or claim speedup until the repeated comparison completes.
Each run max3600s, queue stops on any process/receipt failure. This is bounded
performance engineering, not formal/capacity evidence. At terminal completion,
compare exact row/view RNG hashes, pair medians/p95/throughput/memory; decide
adoption only on repeatable benefit. No deterministic settings leak into timing.
Original, failed and stopped runs remain immutable.
While ABBA runs, next independent candidate prepared: synthetic
scripts/v2/benchmark_relay_cudagraph.py, unchanged eager final-state function
inside Torch's AOTAutograd cudagraphs backend (no Inductor fusion). No model or
config edit. Tests compare two live carried states and all input gradients
against the independent recurrent reference. Probe covers FP32/BF16,grad/no-grad,
masked writes/nonzero state,strict numerical/RNG/capture/replay gates and memory.
Requires idle GPU and publication identity; do not overlap the ABBA resource
locks. CUDA capture and speed remain unverified. Tool published3a05875bb43e88cf7004a79a1df6b23c87e64775;
clean server copy passed7 helper checks on supported Python3.12/Torch2.13.
Pre-GPU review then kept the single-token CLS-like case active (mask starts at
index1, not0); a length-one recurrence/gradient test now covers that boundary.
Corrected tool published279696a2c995d136d64838f3c3090fc4992a6839;
source development/source-v2-relay-279696a is clean, supported-server3 probe
checks pass (8 local probe+parity checks). Publication development/
gradpert-relay-publication-279696a.json SHA256
7905c5ffe67c47804d876e3a9b159006367f555cd7408a5fe85f3caf973b1a1f.
Dependent wrapper development/relay-279696a-cudagraph-check,PID3389345,
same-stem .sh/.log/.pid/.exit/.stage, waits up to14400s for ABBA's GPU locks.
It requires ABBA queue exit0 plus all four passed40-step receipts before any
CUDA call, then runs a bounded1800s GPU0 synthetic check (both locks held).
Current stage waiting-for-abba-lock; no CUDA probe result yet. This is a finite
prerequisite queue, not a timer or parallel GPU job. Inspect actual wrapper/rank
processes and terminal receipts; do not restart based on a stale .stage file.
Prepared scripts/v2/compare_benchmarks.py for the current one-factor ABBA:
requires all four terminal passed unprofiled receipts, matching clean source,
config hashes/one changed validation setting, data/ordered batches/view RNG,
rank populations and CPU thread/affinity settings. Recomputes whole-step rank
maxima and reports descriptive ratios, both pairs and run variability without
calling two replicates statistically significant or numerical/capacity proof.
Eight focused rejection/aggregation tests and scoped lint/mypy pass. Run against
actual four small receipts after completion; partial A1 alone cannot show gain.
Capacity handoff audit found scripts/v2/capacity_sweep.py silently forced OMP=1,
which would discard the current explicit OMP=2 setting. Changed child environment
to preserve caller-supplied OMP/MKL/OpenBLAS settings, keep legacy OMP=1 only when
unset, and record the four whitelisted execution variables in sweep.json. No
GPU/model/scientific config changes. Three focused sweep checks and lint pass;
scoped mypy passes with missing external yaml stubs excluded (import-untyped).
This repair applies to future sweeps only; active ABBA/probe source stays frozen.
No timer, Goal active. Remaining full-update/throughput/capacity/B0requirements
unchanged. No new default optimization has been accepted.

Additional user request complete: updated EasyConnect skill with native Swift AX
and private Security.framework credential path; both installed copies match,
6 decision tests, live healthy UI/TCP/SSH checks and Codex forceReload recognition
passed. No live logout/login was forced. AGENTS contains fallback command.

Next: read short CPU/GPU timeline, then ranked measured
optimizations with full-update parity/ABBA, sustained128+ batch search, then only
complete Jurkat B0 five epochs and best/last tests. No other ablations. Preserve
full run/source/config/data/environment identity. Old batch192 is uncertified.
Update Byte and active execution state at each transition; testing30min/formal2h
cadence applies if scheduled monitoring is later used; current user asked Goal
mode supervision, so no duplicate timer is enabled.

## Historical state, superseded by the relay-method stage above

Previous heading: Hamiltonian expander / full MLA method implementation.

The user requested replacing the old random-permutation expander with
TxPert-style fixed bidirectional Hamiltonian cycles, allowing graph layers
to read updated neighbors, and dropping DSA for full MLA. Preserve the old
stopped B0 and all historical config/source identities. The new explicit
Jurkat config is `configs/v2/hamiltonian_mla_jurkat/gradpert_v2/nadig_jurkat.yaml`.
It keeps 5 epochs and a **candidate** global batch192; capacity must be
remeasured before formal GPU work. Detailed method and open scientific risks:
`docs/design/GRADPERT_V2_HAMILTONIAN_MLA_METHOD.md`. This stage is local
implementation, tests, documentation and Git publication only, not a
training restart. The historical capacity records below refer to older
architectures and are not evidence for this method.
Local validation: 230 v2 tests, Ruff lint/format, and wheel/sdist build passed.
The fixed-axis four-fold cross-cell design is updated to this same method;
its dataset-specific artifacts and execution adapter remain pending.

Overall outcome: deliver a runnable, clean, published Top500/GenePT-PCA256/
SwiGLU v2 model on two RTX 5090 cards, measure useful batch capacity, and make
this the parent for future Jurkat v2 ablations without modifying v1 or historical
B0/B1 runs. The user-selected scientific default remains global batch 128 and
five training epochs; capacity candidates are engineering evidence only.

Core stage acceptance: the complete model and default config are published;
the bounded two-GPU performance comparison, 128-update capacity at the highest
validated batch 144, full v2 tests, ablation parent/CLI smoke, and design docs
are complete. The capacity-run source is `0a9a67f272978476e104952f3a715fa41969105d`
(the immutable server checkout is
`/data/yilangliu/GraD-Pert/development/source-v2-capacity-clean-0a9a67f`). Work in
`/Users/elan/code/grad-pert-v2-build`; preserve the separate dirty root checkout.

Measured at global batch 128 on both cards: 5-step bounded probe baseline
`cb517ea` 4.561 cells/s, 21.51 GB/card; current fused KoLeo backward `2eb4c4e`
6.511 cells/s, 25.91 GB/card (+42.7% throughput). An attention-only optimization
had negligible full-step benefit. CPU/distributed checks for fused gradients:
53 passed. Detail, limitations, and immutable receipt stems:
`docs/experiments/GRADPERT_V2_GLM53_DUAL_GPU_PERFORMANCE.md`.

Batch boundary: 144 passed a clean 128-update plus 300-control validation
inference probe, receipt
`/data/yilangliu/GraD-Pert/development/v2-m36-capacity-clean-0a9a67f/receipt.json`
SHA256 `aa78cce7b4312d7150744fdae25c25265b37a633cd828a954721327a26e2f1a4`.
It is the highest sustained-tested batch, not an exact physical maximum. 152
passed only 5 updates and reached 32.42 GB/card; 160 failed by update 3; 192
failed at its first update. Disabling outer checkpointing OOMed at global 128.

Completed server process: exact-default global-batch128 128-update plus
validation-inference capacity probe, output
`/data/yilangliu/GraD-Pert/development/v2-m32-capacity-clean-0a9a67f`, log
same stem plus `.log`, clean read-only source checkout
`source-v2-capacity-clean-0a9a67f`, GPUs 0,1. Its receipt passed 128/128
updates, checkpoint continuation, and 300×5000 validation inference; receipt
SHA256 is `51323c14610b8094dbe25e73c9c6ce4663c3325498e9873db97b2f1e9a59b269`.
Both GPUs are idle after completion. Measured throughput was 6.622 cells/s,
with 27.80/27.80 GB peak allocated memory. Source HEAD remains the pinned
`0a9a67f` commit and the server checkout is clean.
The earlier distinct 128-step attempt `v2-m36-capacity-0a9a67f` was stopped
after 32 updates because a concurrent build introduced ignored egg-info into
its active source tree; `audit-stop.json` preserves both hashes. It is not
capacity evidence. The replacement checkout's tree hash matched its pinned
publication receipt before launch and remained clean after the successful 144
probe. Never build in that checkout. Do not overwrite run IDs or change the
active server checkout. The completed capacity receipt is engineering evidence.
The exact published source `536458333e437252178ffe493c5c50c9064e7615`
then passed the same-config two-GPU one-update integration preflight: receipt
`/data/yilangliu/GraD-Pert/development/v2-5364583-b128-integration/receipt.json`,
SHA256 `c8882f28d279ec5cde045fc1b3f9cdfd9be2d8371930994ecb6ad85c2498cc2c`.
Its publication receipt is `gradpert-5364583-publication.json`, SHA256
`37b58e67236317c969b5350e5f25dd1ce0ae3bc16f94eb7e20ad5c19ce8018cf`.

The formal five-epoch Jurkat v2 baseline is now active on GPUs 0,1 at global
batch 128: run ID
`nadig_jurkat-seed1-20260923T194747Z-1b7eda2abd6441f592d0834e1e275e88`,
run root `/data/yilangliu/GraD-Pert/runs-v2-glm53-current/` plus that ID,
log `/data/yilangliu/GraD-Pert/development/v2-jurkat-baseline-5364583.log`,
parent PID file with the same stem and `.pid`. The source checkout is
`/data/yilangliu/GraD-Pert/development/source-v2-formal-5364583`; config
SHA256 `8471f5ea68f4801406497985291a0088116ff5a8435b82f85e55c611548b06c6`.
At handoff the process was alive, epoch state was 0/5, and both GPUs were
occupied by this run. The existing heartbeat `grad-pert-v2-batch128` has been
retargeted to the formal baseline and verified active at every two hours, with
user-requested reports on each check. Verify five committed epochs and actual
best/last tests before treating this run as complete; keep training SHA fixed
at `5364583` even if documentation-only GitHub main advances.

2026-09-24 11:53 +08 milestone: epoch 1/5 committed after 1,081 optimizer
updates. Journal SHA256
`02fe76ec4cd46489a58116a095a3059ac9f6382c1e97f95fe3ce419f7ac76dcb`;
provisional best/last `epoch-0001.pt` SHA256
`c3e1134e441db33556765458f009c88f5bd321389539c1bda1a8118a40647d85`.
Validation prediction loss `0.005269254464738538`; validation Pearson values
TxPert `0.14296990652273225`, TriShift `0.1875737367080305`, Systema
`0.06702235003118254`. The parent and both rank workers remained alive;
`COMPLETE.json` and best/last **test** receipts were absent. Continue training
the existing run through five epochs; no new run ID or ablation group.
Full v2 test suite: 212 passed; lint,
format and isolated wheel/sdist build passed. Mypy retains five unrelated
existing errors in `training/step.py` and `training/v2/reductions.py`.
The formal five-epoch scientific run described above was later stopped by the
user. Capacity and one-step preflight receipts are not scientific best/last
results.

2026-09-24 18:05 +08 user stop: the formal Jurkat v2 run was interrupted
after one committed epoch while the second epoch had no committed receipt.
Verified parent PID 2770034 and torchrun PID 2770045 received SIGTERM;
both rank workers 2770069/2770070 exited. GPUs 0/1 returned to 2 MiB each.
`COMPLETE.json` and best/last test receipts were absent. Preserve the run ID,
epoch-1 checkpoint, source/config hashes, and all existing receipts; do not
mark the five-epoch baseline complete or restart it. The two-hour heartbeat
`grad-pert-v2-batch128` was deleted. Current scope is discussion of speed
changes (Cell/Response Encoder depth 2:1 and four heads 16384→8192
prototypes); no new code or experiment has been launched.

2026-09-24 18:37 +08 new authorized outcome: implement the compact v2 parent
and retest maximum useful dual-GPU batch without restarting the stopped formal
run. User clarified that **each** of Cell and Response has two KDA layers and
one terminal DSA/MLA layer, not a 2:1 depth ratio between encoders. Four
Student projectors and their four Teacher copies use 8192 prototypes; center
buffers follow. New default Jurkat config is explicit; unmodified historical
v2 configs resolve to the old three-KDA plus one-terminal-layer architecture.
Expected Student count: 23,123,847. Isolated server development snapshot
`/data/yilangliu/GraD-Pert/development/v2-compact-preflight-20260924-39473bd`
passed 82 targeted and 227 total v2 tests, Ruff lint/format and isolated
wheel/sdist build. Mypy retained exactly the five previously documented errors
in unchanged `training/step.py` and `training/v2/reductions.py`. This is
prepublication CPU evidence, not a GPU capacity receipt. Next: scoped commit
and GitHub push; clean immutable server checkout and exact publication receipt;
two-GPU short failure bracket then 128-update/validation confirmation at the
highest practical batch. No monitoring is active yet.

2026-09-24 19:07 +08 publication and capacity milestone: scoped source/config
change `974c5eadf35a95fbc3ba46cffe6d9ab34cdd4e94` is on GitHub `main` and
in clean immutable server checkout
`/data/yilangliu/GraD-Pert/development/source-v2-compact-974c5ea`.
Publication receipt `gradpert-v2-compact-publication-974c5ea.json` has SHA256
`626bb795287187ea1accec20eacb1bdb906c6bf1ed95fc371ead270424c13760`;
formal server source identity passed. Default global batch128 dual-GPU one-step
integration plus checkpoint reload passed; its receipt SHA256 is
`b7b75b95c22e4ff6cb43aa9f40d39f353a4cc230381d6b63d5fa775bc5698554`.
Five-step short probes passed batch192, 208, and 224, with batch224 throughput
9.5172 cells/s and 32,338,694,656 bytes peak allocated. Batch240 OOMed at
step3. The 128-step batch224 probe
`/data/yilangliu/GraD-Pert/development/v2-compact-974c5ea-capacity-m56-128`
failed at step6 with CUDA OOM after five completed steps; batch224 is not a
sustained-capacity result. The failed receipt and log are preserved.

The next 128-step probe is running at global batch208, microbatch52 per GPU,
accumulation2, GPUs 0/1, with the same published source and allocator contract.
Run root `/data/yilangliu/GraD-Pert/development/v2-compact-974c5ea-capacity-m52-128`,
PID file same stem plus `.pid`, log same stem plus `.log`, exit-code file same
stem plus `.exit`; config SHA256
`89832263f13fe41a3eda0e0793fea8567df28616d07f92416ae31594b2b84172`.
Check 128 completed steps, checkpoint continuation and 300-control validation
inference before accepting it. If it OOMs, preserve its run and probe 192 with
a distinct ID after the GPUs are idle. The 30-minute heartbeat
`grad-pert-v2-compact-batch-capacity` is active and reports at every check.
This is engineering capacity evidence only; no new five-epoch training or
ablation has been launched.

2026-09-24 19:41 +08 capacity follow-up: the batch208, micro52, accumulation2
128-step run failed on both ranks from CUDA OOM in backward at update24, with
23 completed updates. GPU 0 had about 120 MiB free and the attempted tensor
was 138 MiB. Exit code 1; failed receipt SHA256
`b52aad0d0d1874f4ec04d556d991504123defe0b1d95fe62156221535cd5811a`.
The failure is not a sustained-capacity result. Both GPUs returned to idle.
The next independent full probe started at global batch192, micro48 per GPU,
accumulation2, using the same immutable source and publication receipt:
`/data/yilangliu/GraD-Pert/development/v2-compact-974c5ea-capacity-m48-128`.
Its config SHA256 is
`674ea9ab4160e10e85de7f6da78137257efee510c97e2f9852a474a55e81acf5`;
PID/log/exit file names use that run stem. The existing 30-minute monitor was
retargeted to this run. Require all 128 updates, checkpoint continuation and
300-control validation before accepting batch192. If it fails, retain evidence
and narrow to a smaller unused candidate with a new run ID.

2026-09-24 20:50 +08 capacity completion: the new compact v2 source
`974c5eadf35a95fbc3ba46cffe6d9ab34cdd4e94` completed the dual-GPU
global-batch192 (micro48/rank × accumulation2 × two ranks) 128-step probe.
Exit code 0; receipt status `passed`, 128/128 updates, checkpoint continuation
SHA256 `1610f227056f5565697bde00c2ba95345b4a1df3f80b70bec11ff98a3ee1ffbb`,
300×5000 validation inference in 11.986 s, and both GPUs free afterward.
Receipt `/data/yilangliu/GraD-Pert/development/v2-compact-974c5ea-capacity-m48-128/receipt.json`
SHA256 `71d952f90466052a52855a819090d9c5e03ba3e96081eadc03ac2cd3c896529d`.
Post-warmup throughput 8.486 cells/s; end-to-end throughput 7.992 cells/s;
training wall time 3,074.805 s. Peak allocated memory: GPU0 31,011,834,880
bytes, GPU1 31,045,261,312 bytes. The single-condition validation prediction
loss 0.008950 is only an inference finiteness check, not a scientific result.
The source checkout remains clean at its pinned commit. Batch208 failed at
24th update and batch224 at sixth, so 192 is the highest sustained-validated
batch among tested candidates; the exact physical threshold between 192 and
208 remains unknown. Scientific default batch128 and five-epoch plan remain
unchanged. No new formal train or ablation was launched. The capacity monitor
can now be removed.

2026-09-24 21:44 +08 new formal stage: the user selected the sustained-passed
global batch192 profile and authorized B0 formal training. The user then
clarified that B0 here means only the complete prediction + SSL1 + SSL2
baseline, **not** the `prediction_only` contrast in the older group generator.
The tentative two-row local generated group was removed before publication;
no prediction-only training launched. The existing self-contained compact-m48
configuration has micro48/rank, accumulation2, world2, batch192, lambda1=1,
lambda2=0.1, seed1 and five epochs. Source, GitHub main and a fresh immutable
server checkout matched published SHA
`fc90a0d992373e619976504c82bd6f94c930d7b2`; clean source identity and
publication receipt SHA256
`a77a9e63771f1ffc516d123f7280e8cf04401c408f34c820d150dfd487c5c770`
passed. Exact-source, exact-config dual-GPU one-step integration and checkpoint
reload passed, receipt SHA256
`7cdeb101149bbfe5264215b474e2e38d449ca461c6312f11aafc0aa57c0b73e4`.

Formal B0 run ID
`nadig_jurkat-seed1-20260924T134139Z-e5df6111138747e08ba5a582e37c1ed2`,
root `/data/yilangliu/GraD-Pert/runs-v2-b0-compact192-fc90a0d/` plus ID,
started from sealed plan
`/data/yilangliu/GraD-Pert/development/v2-b0-fc90a0d-b192.launch-plan.json`,
SHA256 `8f4dac881eb98bf5f3ed1bf7562e51e897d8ece58cde146727008fc61b2df03a`.
Config SHA256 `674ea9ab4160e10e85de7f6da78137257efee510c97e2f9852a474a55e81acf5`;
runtime SHA256 `54691399183a502fcf7f939e0f4e3bad587b00677fc84545cf6d20a582a547c6`.
Parent PID file `/data/yilangliu/GraD-Pert/development/v2-b0-fc90a0d-b192.pid`
contains 2953737; log and exit-code files use the same stem. At first check,
parent was alive, epoch journal 0/5 with 749 updates/epoch, and GPUs 0/1
were both active. Require five committed epochs, best/last tests and
`COMPLETE.json` before reporting a scientific result. The heartbeat
`grad-pert-v2-b0-compact192-formal` checks every two hours and reports every
check. Do not modify the active source, resume the old run, or launch any other
ablation row.

2026-09-25 02:28 +08 user stop: the compact complete B0 run above was
terminated before its first epoch committed. The targeted torchrun PID 2953749
and launcher parent PID 2953737 received SIGTERM after their command lines
were matched to the sealed B0 launch plan. Wrapper PID 2953738 and rank PIDs
2953761/2953762 also exited; GPUs 0/1 returned to 2 MiB each. The fit journal
remains 0/5 with 749 planned optimizer updates per epoch, `history.json` is
empty, and only the epoch-0000 initial checkpoint exists. There is no finite
validation selection, `COMPLETE.json`, or best/last test receipt. Preserve the
run ID, published training SHA `fc90a0d992373e619976504c82bd6f94c930d7b2`,
config SHA256 `674ea9ab4160e10e85de7f6da78137257efee510c97e2f9852a474a55e81acf5`,
and existing run files; do not resume or relabel this interrupted run. The
two-hour `grad-pert-v2-b0-compact192-formal` heartbeat was deleted. No other
ablation or `prediction_only` job was started. Current stage is discussion of
graph propagation and expander design; no server training is active from this
B0 run.
