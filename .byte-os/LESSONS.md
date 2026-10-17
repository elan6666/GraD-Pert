## 2026-10-16：新实验家族必须穿过真实评估leaf入口

439efcb已支持DINO两臂的prepare/runner/medium，却漏接eval_batch_capacity.validate_eval_binding白名单，导致10-update通过后在评估推理前失败。cfe641b修复并增加四个真实leaf参数调用，以及mode、步数、顺序和缺行的拒绝测试。预防：新family既测试生成/prepare，也逐一调用全部训练、评估、collector和分析子入口；不能把runner.verify通过当成完整链路通过。失败queue不原地重试，发布新版本和ID重验。跨agent收据SHA必须程序计算传递，主独立校验原文件，不手工补写或信任转写字符串。

## 2026-10-13：设备别名不等于设备不一致

已确认Accelerate1.9单进程device可为cuda（index=None），与显式cuda:0直接torch.device比较为False。CPU集成没有覆盖此GPU guard。预防：将未编号CUDA绑定到真实current_device再比较；测试当前index非0、显式错卡、CPU/CUDA类型错配和多进程拒绝。不得用无条件接受cuda或删除隔离检查来修复。正式门仍必须使用真实GPU。

## 2026-10-05：组合实验与并行实验需区分

用户“2、3开启”在本次明确指②＋③同时作用于同一模型，单次从头20轮，不能据编号自动拆为两组训练。将是否组合的含义写进配置名、计划和队列行数；启动核验必须确认仅一行E23。

# Fixed-epoch protocols must cover terminal collection and queue advancement

Confirmed 2026-10-02/03: training/CLI/schema accepted v2_fixed_6, but result collector retained only1/3/5/50 and failed after a successful six-epoch E1. Individual launch/capacity checks did not exercise post-training next_action. Prevention: every supported protocol needs a synthetic complete history/journal/checkpoint/test fixture traversing collector and queue skip_complete, including joint-only selection and invalid evidence. Collector rejection must preserve completed runs; recovery imports validated completed rows and gives remaining rows fresh IDs instead of rerunning successful work.

# Lessons

## Relay masked-node queries need a nonzero learned initialization

- Confirmed during the new relay full-update test: a zero graph mask token
  stayed zero through bias-free query projections and relay blocks. Repeated
  normalization derivatives produced nonfinite gradients although forward
  losses were finite. Anomaly detection traced the failure to graph query
  projection backward; nonzero normal initialization (std 0.02) removed it.
- Prevention: test heavily masked graph views through a complete optimizer
  update, with dropout and accumulation, and assert finite gradients. Forward
  checks alone do not validate a masked multilayer recurrent mechanism.
  Apply this initialization only to the new relay profile; keep history intact.

## Final-epoch reporting needs the actual final checkpoint

- User correction 2026-09-10: automatically report both best-validation and
  final-epoch test results for R50. Best-only cleanup made historical LR-low
  final unavailable. Never fabricate it, relabel an intermediate archive, or
  silently rerun training. Preserve best/last before test loading or cleanup,
  bind each role to epoch/step/source/config/checkpoint hashes, and explicitly
  alias identical final/best checkpoints. Smoke remains validation-only.

## Loss coefficients require a normalization and gradient rationale

- User correction (2026-09-09): the redesigned experiment program should not
  begin with arbitrary coefficients or a blanket remove-one-loss matrix.
  Preserve the original E3 prediction/condition/masked/spread weights1/.8/.4/.1
  for training-recipe selection. MSE, soft-target CE and log-distance penalties
  have different scales and reductions; equal coefficients do not imply equal
  influence, and raw loss magnitudes do not measure gradient contribution.
- Before proposing changed weights, inspect reductions and training-only
  weighted gradients on shared parameters, state the hypothesis, and freeze
  the comparison before validation results. Do not claim optimality from an
  old test ranking or a single batch. See R50_E3_REDESIGN.md.

## Instantiate the complete model before claiming a capacity ratio

- Correction (2026-09-09): the proposed 128-embedding, two-layer/two-head128
  graph, projector256/32, basal/decoder128 design was incorrectly estimated
  at 3.82M parameters. CPU module instantiation counted 6,338,568 (49.417:1
  against 128,266 training perturbed cells), including both graph sources
  and both Student and Teacher. It does not satisfy the agreed 20--30:1.
- Prevention: count every instantiated parameter, show component subtotals,
  include frozen Teacher parameters, and distinguish budget estimates from
  verified counts before changing configs or claiming a design fits.

## A changed baseline must reverse dependent ablations where necessary

- Requirement correction: the successor A0 changed from eight to four local
  views. Merely editing A0 would make the old four-local L2 identical to the
  reference and would silently destroy the single-factor matrix.
- Correct migration: regenerate every successor row from the new A0, redefine
  L2 as eight locals, derive proportional masks as `2/4` and `1/4`, advance the
  matrix identity, and invalidate old config/run hashes without rewriting the
  old evidence.
- Prevention: after any baseline change, compute each row's resolved parameter
  diff against the new baseline and require it to equal the declared allowlist.
  Also audit derived quantities and every secondary config generator that uses
  A0 as its base.

## Cross-profile attribution should choose the smallest measured boundary

- Mistake avoided: static inspection suggested both scalar GPU synchronization
  and sparse-union construction, but either could have been too small to matter
  on the real A0 path.
- Correct evidence: the Torch profile found only about 14 ms of
  `_local_scalar_dense` across three profiled steps, while an independent
  five-step cProfile attributed 52.829 seconds cumulative to 342
  `build_sparse_union` calls. A real first-batch CPU microbenchmark then showed
  exact tensors with median union preparation falling from 8,276.7 to
  2,310.5 ms.
- Prevention: require a stage profile, a second profiler at the suspected
  implementation boundary, and an exact real-shape microbenchmark before
  editing. Optimize only that boundary, preserve a same-commit reference path,
  and reserve end-to-end claims for serial same-GPU ABBA.

## Linux capacity checks must use reclaimable available memory

- Mistake: the CUDA preflight used `SC_AVPHYS_PAGES`, which measures currently
  free pages. Hashing multi-gigabyte immutable inputs filled the Linux page
  cache and made a host with about 241 GiB reclaimable memory appear to have
  only 2--4 GiB available.
- Correct evidence: `/proc/meminfo` `MemAvailable` is the kernel estimate for
  memory that can be allocated without swapping; `MemFree`/free pages exclude
  safely reclaimable cache.
- Prevention: Linux capacity gates use `MemAvailable` first, retain the POSIX
  free-page probe only as a fallback, and test both paths. A failure before
  native/model construction must be receipted as preflight failure and must
  not be validated as if native identity files already existed.

## Stage memory evidence must follow retained autograd lifetime

- Mistake: aggregate step memory and static call counts could not explain a
  first-step OOM in a model that accumulates many independent local forwards
  before either backward path begins.
- Correct evidence: record allocated/reserved memory at nested phase
  boundaries. In the RingInduced A0, each completed eight-condition local index
  retained about 8.47 GiB, while Student globals retained about 5.44 GiB. The
  fourth local index then failed, directly identifying activation lifetime as
  the capacity blocker.
- Prevention: capacity probes emit ordered entered/completed/failed events for
  every global, local index and condition view. Nested observers use a stack;
  a single active-stage scalar rejects valid outer completions after an inner
  event and corrupts the failure receipt.

## Checkpointing stateful graph forwards requires private buffer state

- Mistake: saving and restoring shared BatchNorm buffers around checkpoint
  recomputation either applies running-stat updates twice or mutates tensors
  that autograd saved, causing a version-counter failure.
- Correct implementation: run the original local forward through a functional
  module with private working buffers, mirror its one update into real buffers
  without placing that mutation on the autograd dependency path, and recompute
  from separate pre-view buffer clones. Preserve the checkpoint RNG state and
  the original independent view order.
- Prevention: exact-effect tests compare every output, loss, gradient,
  optimizer state, Student and Teacher parameter/buffer state, centers and RNG
  after a complete step. An output-only checkpoint test is insufficient.

## Receipt producers and validators must share one exact predicate schema

- Mistake: the P1 worker emitted the valid
  `publication_receipt_equals_p0` source predicate, but the terminal validator's
  exact allowlist omitted it. After the import path was repaired, the validator
  still rejected every legitimate terminal receipt as malformed.
- Correct evidence: a replay with repository root plus `src` on `PYTHONPATH`
  reaches the validator; a synthetic copy of the preserved exact CUDA-OOM
  receipt with the unrelated old observer failure removed is then classified
  as `capacity_failed` with exit code 10.
- Prevention: derive producer and validator keys from one shared constant and
  keep an end-to-end replay test. Unit fixtures must include every field the
  real producer emits; a self-consistent reduced fixture can hide schema drift.

## Local telemetry must not depend on a long-lived archive transport

- Mistake: local-private Trackio mode repeated a live Hugging Face Bucket
  preflight when each queued row started. The reverse SOCKS endpoint expired
  during the long formal queue, so H1/H2 lost local scalar capture even though
  their native training receipts completed normally.
- Correct boundary: local scalar collection and remote archival are separate
  best-effort stages. A sealed launch-time private-Bucket receipt can bind the
  intended destination, while a later transport outage must be receipted as an
  archive failure rather than preventing owner-only local collection.
- Prevention: before the next tracked formal lineage, make the sidecar consume
  a hash-pinned launch preflight, test proxy loss between rows, and defer the
  live Bucket operation until post-row archival. Never replay this completed
  lineage to manufacture missing telemetry.

## Capacity policy must enumerate every authorized non-sentinel row

- Mistake: a fresh M capacity queue used the bounded performance worker, but
  only H4 was registered as capacity-only. M4 ran because it belonged to the
  frozen sentinel; M1 then failed before claiming an attempt because it was
  absent from both allowlists.
- Correct boundary: keep the eight-row sentinel immutable and maintain a
  separate explicit allowlist for user-authorized M, W, O, and H4 capacity
  probes. These additional rows may run only `p1_capacity`, never timing or
  profiling stages, and remain `scientific_completion=false`.
- Prevention: before a multi-row capacity launch, test every exact variant ID
  through the same preclaim predicate used by the worker, plus negative tests
  proving non-authorized rows and all timing/profile stages remain rejected.

## One-step capacity probes must include degenerate graph views

- Mistake: the sparse Transformer capacity probe completed its first batch,
  but a later STRING-only condition produced a valid isolated-anchor local
  graph with one node. Training BatchNorm then failed because a singleton view
  has no estimable batch variance.
- Correct boundary: independent graph views keep their own normalization order.
  A singleton view uses the BatchNorm layer's existing running statistics
  without updating them; ordinary views with two or more nodes retain the
  original BatchNorm path exactly. Do not merge views or duplicate nodes merely
  to manufacture normalization samples.
- Prevention: graph-encoder integration gates must include connected,
  disconnected and isolated perturbation anchors beyond the first frozen
  batch. Test singleton forward/backward finiteness and unchanged running
  means, variances and batch counters.
# Post-fit scheduling

- User correction (2026-09-11): do not require an entirely idle fixed GPU for
  post-fit testing. The legacy R50 queue delayed LR-mid despite free GPU0.
  Chain tests directly after successful training, retain memory safety checks,
  and use an explicit nonduplicate handoff for already-waiting evaluations.

## Normalize Norman single-perturbation condition keys before interaction fits

- Mistake (2026-09-23): the first dataset-only interaction audit looked for
  singles by bare gene name, while canonical Norman stores them as `GENE+ctrl`.
  It silently returned zero fitted double conditions although measured singles
  and doubles were present.
- Prevention: parse condition components, map every `GENE+ctrl` single to its
  gene component, and require a positive known double count in the integration
  check. Do not interpret a zero-count scientific result before inspecting
  canonical condition strings and expected coverage.

## Condition-level counts do not guarantee within-batch comparison coverage

- Finding (2026-09-23): in the five-dataset audit, four single-gene datasets
  had many cells per condition overall, yet only 3–20 conditions per dataset
  had at least 20 perturbed and 20 control cells in one common batch.
- Prevention: report the eligible condition denominator for each matched-batch
  distribution test. Do not extrapolate significance fractions from this tiny
  subset or replace it silently with unmatched pooled controls.

## Never build in an active source-identity snapshot

- Mistake (2026-09-23): `python -m build` was run from the same server checkout
  used by an active v2 capacity probe. It created ignored
  `src/gradpert.egg-info/` after the probe's startup source check. Git remained
  clean, but the project's content-tree hash changed; the 32-update partial
  run was stopped and explicitly audited rather than used as capacity evidence.
- Prevention: run builds in a disposable clean clone or stage the source into a
  separate build directory. Keep every active training checkout read-only;
  recheck both Git status and content-tree SHA before accepting a run receipt.

## Advance the authorized dependency after a monitor confirms completion

- Context: the scheduled v2 batch-128 capacity check verified a passed 128-step
  receipt while the overall request still required recording the result and
  preparing the formal Jurkat baseline.
- Mistake or misunderstanding: I reported the completed test and listed its
  successor as a future step instead of executing the already authorized
  documentation and preflight handoff in the same turn.
- Correct understanding and evidence: a successful stage receipt changes the
  active work set; it does not complete the overall outcome. A newly discovered
  repair is another dependency to track before returning to the original path.
- Prevention rule: after verifying a stage, immediately execute the next
  authorized ready action, then update the existing monitor's target and
  cadence. Also update the project's Byte state/entry documents so a future
  scheduled run sees the current stage instead of an older status snapshot.
  Stop only at a genuine external wait, unresolved blocker, or completed
  overall outcome.
- Status: active
# Ratio descriptions must name their scope

- Mistake: interpreted the requested 2:1 as a Cell-versus-Response depth ratio,
  then as an uneven KDA/DSA allocation between encoders.
- Correction: the user specified the same internal two-KDA/one-DSA-MLA sequence
  in **each** encoder. The old configuration was three KDA plus one terminal
  DSA/MLA in each encoder.
- Prevention: write layer sequences per module and compute total depth before
  giving parameter estimates or editing architecture configs.

## Resolve B0 labels against the user's current experiment meaning

- Mistake (2026-09-24): I initially treated “launch B0” as the historical
  two-row generator group (`prediction_only` plus `joint_ssl`) and generated
  both local candidate configs. Neither was published or launched.
- Correction: the user specified that this B0 is the **complete** compact v2
  baseline with prediction, SSL1 and SSL2; `prediction_only` is outside the
  authorized formal launch. The tentative files were removed.
- Prevention: for reused experiment labels, inspect the current run lineage
  and apply the user's latest explicit definition before creating a formal
  queue or allocating GPUs. Keep a separate run ID for each scientific model.

## Check the reference numerical repeatability before blaming an execution change

- Evidence (2026-09-26): unchanged full two-GPU relay model repeated from the
  same checkpoint and exact input/RNG digests differs by 1.3163e-4 in gradients
  (relay-4898de3-reference-repeat). The candidate discrepancy was 1.1332e-4.
- Correction: strict normal-mode A/B failure alone does not isolate the
  candidate as the cause; the responsible nondeterministic operator remains
  unresolved. First-step model equality is also weak when warmup LR is zero.
- Prevention: preserve thresholds, add an unchanged-reference control and
  diagnostic deterministic comparison, and inspect gradients/optimizer plus
  a nonzero-LR update. Do not claim equivalence or speed before those checks.

## Clean publication staging

2026-09-26: Git-clean editable test worktree can still contain ignored
`src/gradpert.egg-info`, which participates in the source-tree identity hash.
Publication from it differed from a fresh server checkout and correctly failed.
Generate release receipts from a clean clone at the published SHA; compare tree
hashes before any CUDA work. Never bypass the identity check or delete unrelated
local artifacts merely to match a release.

## Bounded profiler postprocessing

Large KDA traces contain over a million kernels per update. Torch key_averages
can lazily materialize event objects after trace export, taking minutes and
~47GiB RSS per rank in the observed single-pass probe. Keep detailed operator
tables opt-in; preserve raw traces and small summaries by default. Profiler
overhead, including postprocessing, is not model training throughput.


### Bound immutable-checkout object-store dependencies
Repeated shared clones of previous shared clones eventually exceeded Git's
alternates nesting limit and failed checkout (473466d staging). Use an independent
source-only bundle/clone or a bounded stable object store; never repair by repacking
an active immutable source. Verify clean source/publication identity after staging.

## 2026-10-09：不要把最大容量测试当作每次启动门槛

错误：沿用历史128步持续测试，给三个小机制改动逐组重复长压力测试。用户修订：以后默认10次完整优化更新即可，保留恢复和推理检查。预防：新v2队列使用短程10步门；历史长压力测试只作为历史证据，不自动重跑或把短检验称为长期容量证明。


## 2026-10-11：CPU 合成测试必须隐藏 CUDA

- 已确认错误：发布 GM1 后在服务器直接运行 pytest，checkpoint RNG 保存调用
  `torch.cuda.get_rng_state_all()`，使 CPU 测试也创建498MiB CUDA上下文。
  pytest PID2337434 属于我们 UID1002、在训练控制器树外；共享GPU保护因此
  正确停止 BR。不能把它描述成其他用户任务或模型 OOM。
- 修正：服务器CPU测试用 `scripts/v2/run_cpu_tests.py` 的隔离子进程，
  在导入 torch 前强制 `CUDA_VISIBLE_DEVICES=""` 并检查 CUDA 不可用；
  CPU 配置解析、checkpoint 核验同样隐藏 GPU。真实CUDA子训练仍显式选择卡。
- 预防：部署时不能仅凭测试名称或张量device=cpu判断无CUDA副作用。
  不放宽资源保护；保留失败队列证据，按原科学源码、配置、日程及完整状态
  从最后已提交 epoch 恢复，并单独记录恢复控制器版本/attempt。


## 2026-10-11：未启动CUDA任务时资源不足应等待

- 证据：恢复控制器在验收屏障、active为空时发现新UID1003 ScButterfly
  占用GPU0约720MiB；新PID不能继承原pinned白名单，原保护正确拒绝。
  但将此未启动阶段记为终态失败会导致不必要的恢复队列重建。
- 预防：无活动CUDA子任务时已知占用/显存不足保持CPU资源等待，GPU lease
  与子训练均不得提前启动；实际任务运行时继续5秒fail-close。设备身份
  漂移、坏配置或遥测失败不能吞成等待，也不自动改白名单或信号其他用户。


## 2026-10-12：短等待的exec session不是跨agent持久标识

子agent对主exec session49703读取返回Unknown process id；完整服务器checkpoint验证JSON确实已产生，主在helper结束后确认exit0。之后短监督交付公开PID/identity、固定日志、exit收据和结果文件路径；不假设exec session handle可跨agent调用。区分结果文件核验与进程exit证据，两者缺失时不能声称执行完成。


## 2026-10-12：晚到review与真实runtime契约

实施agent FINAL后进入idle，send_message不会开启修复turn；晚到finding要followup_task，并确认running。容量启动校验曾错用mock里的source_root，真实runtime使用repository_root（native源定位来自repository_root函数）；27 CPU检查未捕捉现实字段，server prepare在GPU派发前KeyError。接入边界测试应使用真实已脱敏runtime字段结构，先对照native消费点，再mock行为；不能凭fixture发明字段或给旧失败配置补假字段绕过。失败source/runtime/pub/log保留，新版本新ID重试。

## 2026-10-12: checkpoint before interrupting validation

Current lifecycle commits epoch checkpoints after validation. A training step counter at the final epoch is not proof that the final weights are on disk. Before cancelling a validation process, check epoch journal, TRAIN_COMPLETE and full checkpoint SHA. If final weights exist only in memory and no supported cancel/save interface exists, do not kill them; preserve updates until the nearest verified checkpoint boundary. Future cancellation support should save a receipt-backed training checkpoint before validation.

## 2026-10-12: align legacy and current handoff pointers

The relay state helper advances current_handoff/history, but older readers can still follow a stale top-level handoff. At transfer, verify all reader-used pointer fields and their SHA against the new exact run/attempt; synchronize them as part of the owned handoff record before announcing transfer. A correct owner/run alone does not prove the reader has the correct handoff.

## Filter scientific terminal markers on the server before printing

Native evaluation COMPLETE may embed hundreds of conditions and ordered cell IDs, and controller COMPLETE may duplicate the entire result. Printing the full marker caused an oversized tool output. Inspect bounded key lists, select identities and 18 summary metrics on the server, assert a small serialized size before transfer, and retain complete scientific details on the server. Do not mistake a file named receipt for a small artifact.

## 2026-10-13: distinguish official baselines from native comparison anchor
User corrected “恢复基线” to official scDFM/GEARS, not native BR. Reuse completed BR400 as the native reference and forbid BR/Scouter rerun in new matrix; identify model names and immutable terminal receipts before interpreting a resume request.

## 2026-10-13: launch each official arm with its frozen Python
GEARS CPU integration was incorrectly launched with scDFM Python, and correctly failed anndata-version guard. Always select runtime[python] per arm (GEARS and scDFM dependency sets differ), never reuse the previous arm command interpreter. Retain the failed log; rerun with fresh output identity.

## 2026-10-14: masked output gradients and keyed evaluation references

Excluding a gene from numeric supervision does not freeze its parameters: dense zero-gradient rows still receive coupled Adam L2 updates. For gene-specific readouts, audit parameter scale and actual per-cell/mean delta, rather than interpreting null Pearson as0 or blaming the evaluator. Keep this protocol effect distinct from ordinary unseen-perturbation generalization.

Evaluation-state condition IDs may cover more populations or use a different order than test draws. Reuse the public evaluator's condition-ID lookup; do not index reference means by test loop position or assume ordered-list equality. The CPU diagnostic assertion caught this wrong assumption before results were saved; production evaluator already used the correct lookup.

## 2026-10-15：参考可重复性与profiler成本

相同CUDA参考也可出现iBOT单ulp和梯度归约波动；保留重复证据，严格对照两臂使用同一确定性算子与cuBLAS工作区，不放宽误差门限。百万级事件的完整trace达数GiB并污染walltime；正式测速用小聚合计时，重型profile独立记录且不下载。
# Continue authorized dependencies after repair

- Mistake: after a postfit report failure was repaired, the next already approved
  ablation was left idle and described as needing a new authorized stage.
- Correction: preserving failed queues requires a fresh continuation identity,
  not fresh user permission for the unchanged authorized next arm.
- Prevention: completion/repair acceptance must name the next approved dependency
  and either launch it with its existing gates or record a concrete external blocker.
  Status questions do not cancel the standing execution request.
