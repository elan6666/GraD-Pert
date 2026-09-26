## 2026-09-26：完整校验否决首版，定位归约精度并重测

67fd8e1完整更新失败，损失/梯度超原容差，输入RNG一致；不采用、不跑其吞吐。
根据Torch原生strided reduction实现修正行轴求和结合顺序，7ae9a8b独立18组
全部40阶段/输出/梯度逐元素相等。修正后完整校验PID3439468进行中，未宣布通过。
当前路径/后续动态N编译问题见STATE.md。保持原eager默认，无B0或其他消融启动。

## 2026-09-26：mHC融合微基准通过，完整更新校验运行中

单向KDA/final-state读取保持；新融合候选仅针对mHC 20轮Sinkhorn执行，默认未改。
两张5090共18合成case输出与梯度通过，局部前反向约7–8倍提速，不能外推整模型。
61本地相关测试及18服务器定向测试通过。首版Triton scope失败保留，已修。
完整双卡校验已启动：PID3436999，single-67fd8e1-sinkhorn-parity，
源码67fd8e1883ecb5d1145fe179d797347057f29394，两边同config，全局8，完整双蒸馏。
先比对两次完整更新再端到端ABBA；状态/路径见STATE.md。未启动B0或其他消融。

## 2026-09-26：单向方法继续性能目标；关联归因进行中

全量trace统计完成并入库：1414510次GPU kernel、43200次logsumexp，
后者对应mHC Sinkhorn。累计耗时非互斥关键路径，未宣称任何新加速。
已发布301be18 CPU scope/launch/kernel三遍关联分析工具，6定向测试通过；
服务器CPU任务PID3432732，900s有界，stem single-301be18-trace-attribution-rank0。
无GPU任务；持续容量/B0待机制优化验证。当前详情见STATE.md与性能报告。

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

User steering prioritizes mechanism-depth KDA optimization before sustained capacity/B0.
Sweep3429104 exit1:72passed/80OOM/88notrun; no activeGPU job. Receipts saved.
Route in performance report: quantify existing trace, fused decay-Gram fwd/bwd
without expanded pair-channel temporary; not yet implemented. Byte state corrected.

CPU prefetch ABBA complete exit0, all12/12 and comparable identity audit.
Mean inclusive step increases2.4476%; do not adopt. Saved reviewed356,094byte
receipts and comparison under docs/experiments/single-ec2384a-prefetch-abba/.
Immediately launched current-method capacity integration sweep PID3427171,
stem development/single-a1d55fa-capacity-integration, source/publication below.
Both GPUs verified idle, locks held,1800s timeout, micro8/16/32/48/64, accum2.
Only one-step boundary search; continue sustained validation after terminal evidence.

CPU prefetch B2 passed12/12, median21.5217s/inclusive0.36685cells/s, wait2.2775%.
A2 now running; both prefetch repeats so far do not exceed A1 throughput.
Capacity integration wrapper prepared/uploaded (.sh at planned stem), not launched.

## 容量测试发布就绪（尚未运行）

源码 `/data/yilangliu/GraD-Pert/development/source-v2-capacity-a1d55fa`，
SHA `a1d55faff4a55d624e5061b7eb70e76d93f24c9b`，干净身份已核验。
发布 `development/gradpert-capacity-publication-a1d55fa.json` SHA256
`d00729ca8d66fe5dede305af80a94aa22fcba3298f0424154dd9f11cdf10e67c`。
单步扫描dry-run `development/single-a1d55fa-capacity-integration.plan.json`，
SHA256 `53e743e31d453259ed1136afbb4854b4b67ccb72e6566c667ea6be35f1ad678d`。
候选micro8/16/32/48/64，eager/重计算/同步数据路径，未加--execute。
待当前ABBA完成和取舍后再启动；如果启用预取则需要先正式配置化并重新发布，
不可让benchmark-only override冒充容量/正式设置。


CPU prefetch B1 passed12/12, median21.8183s, wait fraction2.5462%,
inclusive0.35858cells/s versus A1 .36947 (~2.95% slower). Waiting decreased
but update grew; CPU contention is a hypothesis, not established cause.
B2 now running in same ABBA wrapper3421214. Do not adopt based on wait fraction;
await B2/A2 for paired throughput conclusion. No reruns or other GPU job.

CPU prefetch ABBA A1 passed12/12, median20.6005s, inclusive0.36947cells/s;
B1 running at latest check. compare_benchmarks.py now accepts explicit
--execution-factor cpu_prefetch, requiring identical configs and A/B/B/A flags
without checkpoint override;9 audit tests and scoped mypy/ruff passed.

Prepared self-contained current-method capacity candidates under
configs/v2/single_pass_jurkat/capacity_m{8,16,32,48,64}_a2/gradpert_v2/.
All parsed and validate_profiles passed: only micro/global batch differ from
profiling_m2_a2; single-pass final-state eager, checkpointed, fullSSL1+SSL2,
world2 accumulation2. No GPU capacity launched while prefetch ABBA is active.
After optimization decision: publish selected execution configuration, run
ascending one-step integration to bound memory (expand if64 passes, refine if
failure); then128+ updates/checkpoint continuation/300-control inference near
boundary with fresh run IDs. Never call one-step boundary stable maximum.
Recheck prefetch benefit at selected larger batch; current ABBA only global8.

CPU prefetch parity PASSED two updates on both ranks: exact inputs/RNG and
max_abs0 gradients/loss/objective/optimizer. Exit0; reviewed41,682byte receipts
saved docs/experiments/single-ec2384a-prefetch-parity/. Complete local v2 suite
312passed in85.78s (two existing TorchScript deprecation warnings).
Immediately launched throughput ABBA PID3421214, stem
`/data/yilangliu/GraD-Pert/development/single-ec2384a-prefetch-abba`.
Source ec2384ade9fa66458eac2ff40c2dbff2fd067dc8/publication as below. A1/B1/B2/A2
all identical checkpointed eager profiling_m2_a2 configs; B alone --cpu-prefetch.
12updates each,3warmup9timed,1200s timeouteach,bothGPUlocks. Both parity receipts
are checked before benchmark. .stage/.pid/.exit; stem-label.log and
stem-label/receipt.json. Compare schedule/RNG/data/identity plus inclusive
throughput and memory after all complete. Existing compare_benchmarks.py CLI
is specific to validation-hoist; do not mislabel this as that experiment.
Default remains synchronous pending speed evidence; no capacity/formal yet.

CPU prefetch full-update parity now running PID3420187, stem
`/data/yilangliu/GraD-Pert/development/single-ec2384a-prefetch-parity`;
.stage/.run.log/.exit/.pid and rank receipts at stem/, bothGPUlocks,1200s.
Source `development/source-v2-prefetch-ec2384a`, fullSHA
ec2384ade9fa66458eac2ff40c2dbff2fd067dc8, clean identity verified; publication
`development/gradpert-prefetch-publication-ec2384a.json` SHA256
43f6146774258e336d6777e4fef1b439d9380e49af75a310bb9b3a090442d2fa.
Identical eager checkpointed profiling_m2_a2 config, only candidate CPU lookahead
changed; --deterministic --candidate-cpu-prefetch. Inspect terminal result before
throughput benchmark. Default remains synchronous; no formal/capacity launch.

Checkpoint-removal parity PASSED both ranks, exit0, two full updates with exact
inputs/RNG and max_abs0 losses/gradients/objective/optimizer. Reviewed41,616byte
receipts copied to docs/experiments/single-473466d-checkpoint-parity/. Not adopted:
A1 suggests ~12% faster update but ~4x memory; need sustained capacity tradeoff.
New opt-in CPU view/data lookahead implemented, default disabled. One worker,
one batch ahead; private NumPy generator, public state committed only at yield;
CPU-only assembly, consumer-thread device transfer. Early close/error preserves
consumed RNG, worker joined. Diagnostic flags isolate prefetch from other changes.
20 relevant tests, scoped mypy/ruff passed. Next publish and run full two-update
parity (--candidate-cpu-prefetch, identical eager checkpointed configs), then
benchmark if passed. No current GPU job; no timer/formal B0/capacity yet.

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
