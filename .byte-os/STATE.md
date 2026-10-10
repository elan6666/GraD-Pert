## 2026-10-10：统一 MLP 恢复双卡累积（当前覆盖单卡计划）

旧单卡控制器/N0/U24已按用户指令停止，停止核验SHA256 eb4f0e46a67c0618ed6671a0a40e01a98344af70944b3e0501cab55159af9fa6。旧产物保留。新十组配置：world2、每卡微68、accum2/global272；先N0→CG1→U24→MR1→P1→C1（第六组暂定），其余四组不派发。CG1 rank-local control对齐及P1完整群体MMD适配已完成，本地验证/发布阶段；CUDA10步预检及正式训练尚未开始。Owner主会话，旧监督/返回监控PAUSED。详见主实现工作树计划 GRADPERT_V2_UNIFIED_DDP_20261010.plan.md。

## 2026-10-05：唯一合并E23从头20轮队列已交监督

训练SHA e27f49166dae688eb35444331a1ef1916238efe7，固定config7214e3…c08df8。合并四头16384和SSL1/SSL2各(1,1,0)，mHC4/global272/cap40；本地114、服务器133检查通过。队列 v2-cap40-e23-twenty-e27f491-20261004T203648Z-5d34d6ae attempt1 已真实启动，监督ACK时容量预检24/128，尚非正式epoch。通过后自动fresh20及best/last测试。指定监督会话 owner；20min监控ACTIVE与exacttarget/run已由主会话复核，返回fallback PAUSED。无需主会话重复查服务器。当前共享权威状态位于 /Users/elan/.codex/worktrees/v2-gene-exposure-eval/grad-pert/.byte-os/coordination/state.json；发布状态只是交接快照。证据 .byte-os/evidence/v2-cap40-e23-twenty-20261005/、ACK .byte-os/coordination/receipts/v2-cap40-e23-twenty-supervisor-ack-20261005T0447Z.json。历史六轮结果不变，旧分别两臂20轮解释已覆盖。

## 2026-10-05：用户澄清为②＋③合并、从头20轮

仅一个E23配置：prototypes16384，SSL1/SSL2各(1,1,0)，mHC4；其余cap40/global272/joint-only/frozen tests保持。此前两条20轮请求解释已被覆盖。准备合并配置和资源门槛队列，尚未启动新作业。

## 2026-10-05：E2/E3 20轮新请求准备中

用户授权②③20epoch；从头20轮或完整状态6→20的起点待用户选择。fixed20 schema/planner/native/collector已发布，代码SHA f3620b0d439778a7f1ed21d894e3d06bcdea0b3f；本地112、服务器119定向检查及Ruff通过，服务器源码前后干净。旧六轮结果冻结，新配置/双卡预检/正式队列待起点确定后准备。当前GPU1有其他任务，继续既有空闲门槛。没有新训练或监控启动。计划 `.byte-os/plans/GRADPERT_V2_CAP40_E2_E3_TWENTY_EPOCH_20261005.plan.md`。

## 2026-10-05：cap40 三组消融全部完成并独立验收

B0/E1/E2/E3均6/6、1056更新、min-joint best=last epoch6、真实best/last测试、相同冻结数据/300-control有序清单与表达分组、checkpoint哈希/来源核验、zeroPKL。替换队列COMPLETE；E1未重跑。最终比较和loss分项已归档：`docs/experiments/GRADPERT_V2_CAP40_THREE_ABLATIONS_20261001.md`；独立证据 `.byte-os/evidence/v2-cap40-three-ablations-20261001/final-comparison-20261005/independent-acceptance.json`。单seed描述性结果，E3为组合消融，不按测试调参。正式墙时分解/峰值缺失已明确记录；E2预检128通过。Owner为主会话，整体任务已完成，无后续作业；监督及返回监控均DELETED。此记录覆盖以下未完成/等待/监督状态。

## 2026-10-03: collector repaired, E1 accepted, E2/E3 queue running

Repair SHA2871c4e79a70b7d0d4f900866ce019316b68670f published and immutable server source matches;110 local/server directed tests/Ruff passed. Native src tree unchanged; collector fixed6 bug repaired without changing scientific protocol. Main independently accepted original E1sixepochs/bestlast true tests/hashes/zeroPKL; it remains train/evalSHAac220f5. VPN/SSH connection is verified after Mac unlock, superseding prior connection block. New queue v2-cap40-remaining-2871c4e-20261003T072407Z attempt1/controller742925 only E2/E3 freshIDs plus validated imported E1; old failed queue preserved. Luna startup verification finished. Exact supervisor independently ACKed live E2 capacity preflight24/128, source/runtime/E1 linkage and ACTIVE20min monitor; formal E2/E3 has not started at ACK. Return fallback remains PAUSED. Owner supervisor; main does not duplicate queries. Automatic E2capacity->E3integration->E2/E3fresh6epoch/tests; failures return to main. ACK .byte-os/coordination/receipts/v2-cap40-remaining-supervisor-ack-20261003T0735Z.json; E1 acceptance .byte-os/evidence/v2-cap40-three-ablations-20261001/e1-independent-acceptance-20261003.json. Overall three-ablation experiment unfinished; prior waiting/failed/blocking snapshots below are historical.

## 2026-10-03: E1 terminal collector repair in progress

Original cap40 ablation queue attempt1 failed only after E1 six-epoch child exit0, during collect_run missing fixed6 support; E2/E3 formal not started. Main acknowledged exact handback; both monitors PAUSED. Collector repair and safe completed-E1 import now pass110 tests/Ruff, preserving native src/configs and old run identities. New remaining E2/E3 queue requires server acceptance/immutable release and launch/ACK gates. SSH reset plus locked Mac currently prevents VPN recovery; user asked to unlock only, no duplicate run launched. Evidence: .byte-os/evidence/v2-cap40-three-ablations-20261001/collector-repair-validation.json. This supersedes old queue-running snapshots below; overall ablations remain incomplete.

# 2026-10-01: active approved cap40 three-ablation queue

Owner is exact supervisor codex:01a0df0b-4142-7df1-86c0-d959471d80a1. Queue v2-cap40-ablations-ac220f5-20261001T060835Z attempt1/controller368656, source ac220f5e90e9e5c0c9900cdb2c285a629297c93d clean. CPU waiting_for_idle_gpus is verified; no GPU child. Native preflights then E1/E2/E3 fresh6epochs and best/last tests advance automatically. No E2 capacity claim or silent fallback. Supervisor-owned ACK saved and future monitor grad-pert-v2 ACTIVE20min confirmed; return fallback PAUSED until terminal handback. Queue/source/config identities and exact row roots in coordination/state.json and launch receipt. Main accepts terminal outcomes, fixes core failures, compares final results to accepted B0. No other experiments authorized; historical stopped/completed states below remain historical.

# 2026-10-01: execute approved cap40 three-ablation stage

Main owns build/publish/launch. Existing B0 is accepted; E1/E2/E3 execution now authorized. CPU-only resource wait controller will preserve occupied GPUs and automatically advance exact preflights then formal rows. E2 capacity failure stops without changing batch or losses. New row/run identities and long-run supervision must be verified after launch. No Goal requested.

# 2026-10-01: cap40+mHC terminal independently accepted by main

Read-only server audit verified6/6,1056updates,minimum-joint best=last=epoch6,real best/last tests,source/config/data/selection/checkpoint identities,exit0,COMPLETE,noFAILURE,zeroPKL,and stopped target PIDs. Receipt:.byte-os/evidence/v2-mhc-cap40-six-acceptance-20261001/receipt.json. Both monitors PAUSED. No ablation launched; three configs remain design-only with E2capacity unverified. Earlier pending independent review and heartbeat guard mismatch are superseded by this explicit main acceptance.

# Planned 2026-10-01: three independent cap40 B0 ablations

E1 streams1; E2 four independent prototype heads16384 with hidden2048/bottleneck256;
E3 SSL1(1,1,0), SSL2(1,1,0), both stage multipliers1. User corrected SSL2 DINO/iBOT
from2 to1. Each compares directly to currentcap40+mHC B0; six fresh epochs,seed1,
targetm68×accum2×world2=272,existing val/test/selection. No new runs or monitors.
Three self-contained configs passed schema/options/full-parity checks and2 existing
cap40/mHC config tests. GPU integration and E2 sustained capacity remain unverified.
Current run/owner/20-minute monitor remain unchanged; after its terminal acceptance,
return to main for the next stage. Plan:.byte-os/plans/GRADPERT_V2_CAP40_THREE_ABLATIONS_20261001.plan.md.
Local evidence:.byte-os/evidence/v2-cap40-three-ablations-20261001/config-validation.json.

# Live cap40 mHC six-epoch B0 2026-09-30

Run nadig_jurkat-seed1-20260930T112531Z-f7ca7a77fa6848bb811f706d00e5daad, attempt1, source aeac5fd94123af0b73810259e5e2985228b11d65.
Root /data/yilangliu/GraD-Pert/runs-v2-mhc-cap40-six-aeac5fd/nadig_jurkat-seed1-20260930T112531Z-f7ca7a77fa6848bb811f706d00e5daad.
Source /data/yilangliu/GraD-Pert/development/source-v2-mhc-cap40-aeac5fd, configSHA 1401174171770bb5c7ae580db55f9dd9d4907d8da6a37a42b86a64d747f412d0.
RuntimeSHA 19b87960e43dcd36bda67a87645c9b205e1ec55a4bf6b2817407034534e7b89c, publicationSHA 095782e81d633c3336ab7b2823d95f65e0be67fef82ab2ec569376bfaeb2b213.
PID225555, log /data/yilangliu/GraD-Pert/development/v2-mhc-cap40-aeac5fd-six-formal-r1.log, same stem .pid/.exit.json.
Dual RTX5090 GPU0,1: m68 x accumulation2 = global272. mHC4, full loss, seed1,
6fresh epochs/176updates each; original val/test/reference, joint_only validation.
95local/server checks and dual complete-objective integration/checkpoint reload pass.
Start confirmation: epoch1 step2, ranks225585/225586 alive, no terminal markers.
Luna verification ended. Main prepares exact supervisor20min lease; final result
requires6/6,best/last true tests, hashes, COMPLETE/exit0 andzeroPKL.
Handoff .byte-os/coordination/handoffs/2026-09-30-v2-mhc-cap40-six-supervise.md.
Old no-auto-launch and old3+3 monitor states below are superseded history.

# Active 2026-09-30: cap40 mHC fresh six-epoch B0

User authorized cap40 and mHC. Implementation is v2-only frozen train row IDs,
47836/1335 conditions; canonical val/test/control and evaluation references unchanged.
Exact X/obs/var parity verified between analysis/training canonical; uns differences
remain separately recorded. Bound selection SHA aa0916dfb91fff78d7a43f7d7325879a11c1a904e9bf31586420ca11cce84132.
Next: tests, scoped main publication, immutable server, dual m68 a2 integration,
fresh six-epoch full B0 and designated20-min supervisor handoff. No run launched
yet. Main owns build; old monitors paused. Plan GRADPERT_V2_MHC_CAP40_SIX_EPOCH_20260930.plan.md.
This new authorized stage supersedes prior no-auto-launch analysis-only status.

# Delivered 2026-09-30: Jurkat proportional sampling and four-version comparison

Analysis SHA8638c6557d779ed2290a00b0882964597ca81d3b, clean local/GitHub/server.
Server source: /data/yilangliu/GraD-Pert/development/source-jurkat-proportions-8638c65.
Queue development/jurkat-proportions-8638c65-20260930T103836Z, exit0.
Runs development/jurkat-proportion50-8638c65-20260930T103836Z and proportion25
same suffix; comparison jurkat-sampling-comparison-8638c65-20260930T103836Z.
50%64475rows,delta=.262513065;25%32246rows,delta=.166474252.
Original128266/.373931664;cap4047836/.261779657.100split repeats,1331valid,
4singletons missing;sameoriginal5000genes/control/reference assignments.
All1335conditions/nontrain110711rows/6506variables and retainedmatrix/metadata
exact.12tests local/server; exit0/COMPLETE/source/hash/parity/quotas/CSV accepted.
ConfigSHA50%=93201429299fb0bf8345f62d2adcb14f0286a5572526bd935672084b86367584.
ConfigSHA25%=7d4cb61931b6aa4094abe8b8bfe9d5f64ec5a3717068d8cc87f190931a9da0aa.
Acceptance and all complete artifact identity index:
docs/experiments/data/jurkat-sampling-comparison-8638c65-20260930T103836Z/.
Chinese report/PDF:docs/experiments/JURKAT_SAMPLING_COMPARISON_20260930.md.
All small transfers reviewed/hash-verified; scientific large files serveronly.
Luna supervisor ended; no active task/automation/training/defaultswitch.
Current requested analysis delivered; do not auto-launch mHC B0 or new dataset.
Prior states below are historical evidence, not active wait ownership.

# Live CPU analysis 2026-09-30T10:38Z

Source 8638c6557d779ed2290a00b0882964597ca81d3b, clean three-way identity.
Server source /data/yilangliu/GraD-Pert/development/source-jurkat-proportions-8638c65.
Queue /data/yilangliu/GraD-Pert/development/jurkat-proportions-8638c65-20260930T103836Z.
PID214600, queue .plan.json/.pid/.exit.json/.driver.log.
Runs: jurkat-proportion50-8638c65-20260930T103836Z, then proportion25 same suffix.
Comparison: jurkat-sampling-comparison-8638c65-20260930T103836Z, same development root.
Queue planSHA6778aa63bf88710ebb78a85ccc022c422eacd4535c7b27eab417b2e8030af8bd.
CPU-only4threads, seed42,100splits. Local/server12checks pass.
Luna cap40_cpu_watch owns bounded read-only supervision; main waits without
repeat server querying. Next terminal main acceptance -> four-version report.
No GPU training, default switch or scheduled monitor. Prior snapshots follow.

# Active stage 2026-09-30: Jurkat proportional sampling analysis

Additional methods recovered from prior discussion: condition-wise50% and25%.
Only training perturbations sampled, round-half-up and minimum2whenpossible;
all controls/nontrain rows and retained values remain exact. CPU-only4threads,
100balancedhalf-splits, same original control/gene reference as cap40.
12local checks passed. Next: scoped main publication, clean server tests,
sequential new analysis IDs with Luna read-only bounded waiting; terminal
acceptance and four-version Chinese report. No training/default change.
Plan: .byte-os/plans/GRADPERT_JURKAT_PROPORTIONS_20260930.plan.md.
Historical states below remain original evidence.

# Current 2026-09-30: Jurkat train-only cap40 analysis delivered

Accepted run: jurkat-cap40-5a8a7bb-20260930T094437Z.
Server root: /data/yilangliu/GraD-Pert/development/ plus that run ID.
Clean analysis SHA: 5a8a7bb6712bc4facdfeb13d8dd629bd0f250880.
Configuration SHA256: 1e692732fca3fad57a713ada9475be71a9bce785563d7a5d1b53ec01850cb80f.
Derived H5AD SHA256: 9cb901ead750d82f96cf85016e409b863643e340c043ff8c3aef48d5e48ef98b.
Source directory: /data/yilangliu/GraD-Pert/development/source-jurkat-cap40-5a8a7bb.
Acceptance receipt: docs/experiments/data/jurkat-cap40-5a8a7bb-20260930T094437Z/
jurkat-cap40-5a8a7bb-20260930T094437Z-acceptance.json.

128266→47836 training rows, all 1335 conditions; total158547×6506.
100 split repetitions, seed42, same original context-matched control pool;
original/cap condition-equal Pearson delta=.373931664/.261779657 (1331 valid).
Four singleton conditions omitted; 866 conditions lose at least one batch.
Full retained values and obs/var exact, original H5AD/manifests unchanged,
exit0/COMPLETE/source-clean/zero-PKL accepted. Main independently verified
identity, all rows/partitions/quotas/metadata and267000 score rows.
All small downloads had explicit size/hash checks; H5AD/selection/repeat CSV
remain on server. Report and full plot-data index:
docs/experiments/JURKAT_CAP40_RESULT_20260930.md (Chinese two-column PDF linked).

Luna bounded supervisor finished; no monitor, GPU training or automatic B0
launch remains assigned. No default dataset or model/training setting changed.
Analysis-only derivative is not canonical_ready. Next action is scientific
review of this tradeoff; the mHC joint-only B0 remains ready and unlaunched.
The prior two failed analysis directories/source and transport evidence remain
preserved; following sections are historical snapshots.

# Current repair: preserve categorical metadata on cap40 export

Run jurkat-cap40-af41538-20260930T093845Z computed all scores but failed strict
obs metadata equality after AnnData slicing pruned unused categories. Old outputs
are unaccepted and preserved. The repair restores original obs/var dictionaries
before native writing; no weakened assertions or changed expression/statistics.
Regression includes removed-only and unused ordered levels; 8 tests pass.
Next: publish, server tests, new CPU run and full terminal acceptance. No training.

# Current repair: cap40 canonical gene-symbol identity

The first CPU run jurkat-cap40-1af4195-20260930T093216Z failed before
sampling: H5AD var index uses ENSG while the canonical gene axis uses gene_name.
Fix uses the frozen registry symbol column and validates expression/observation
order hashes. The same 8 tests now include distinct ENSG/symbol metadata and pass.
Failure evidence remains under its old run ID. The server GitHub transport also
timed out; a hash-verified Git bundle restored publication without touching old
checkouts. Next: publish this repair, same-source server tests and a new CPU run.
No GPU training or default dataset change. Original cap40 scope stays intact.

# Current 2026-09-30: approved train-only Jurkat cap40 analysis

Main owns implementation/publication and terminal acceptance. A single bounded CPU
analysis will be supervised read-only by a Luna same-task agent; no cross-chat
monitor or new training is assigned. Cap40 selects min(40, Np) perturbation rows
per frozen training condition with proportional batch quotas, seed42, without
replacement. All other rows/genes/values stay exact. Compare original/fixed sample
with 100 disjoint batch-balanced half splits and condition-equal Pearson delta;
same original context-matched control pool for both. Full-expression Pearson,
retained/removed and overlapping sample/full diagnostics are auxiliary.

Pre-change published source: 5d602a02158af679a4d384f677176ec910fa238a.
8 targeted local tests pass. Next: publish scoped code/docs to main, immutable
server checkout and same-source tests, launch exactly one CPU analysis, inspect
COMPLETE/exit0 and exact data invariants, retrieve only small summary/figure files,
then deliver Chinese comparison report. No model/default data change or GPU run.
The mHC joint-only B0 implementation below remains ready but unlaunched.

# Current 2026-09-30: next fresh B0 restores mHC and uses joint-only validation

Owner: main chat. The requested bounded update is implemented and locally
verified and published on main as
`49456353927ea893f470247a9e4d13e547041dc8` (remote identity verified).
No training, server mutation or monitor was started by this update.
The next B0 config is
`configs/v2/mhc_joint_only_jurkat/three_epoch_m68_a2/gradpert_v2/nadig_jurkat.yaml`,
SHA256 `156c7a0314073fb2901f6d56f8667c657a1d4878c283a1d28ac67b3bdd1ba01c`.
It restores Student/EMA Teacher four-stream mHC, retains full prediction + SSL1
+ SSL2, and chooses `validation_mode=joint_only`: fixed-view joint validation
and its components only, joint-loss best selection, no val population prediction
or Pearson/reference materialization. Terminal best/last tests retain the frozen
300-control, six all/DEG Pearson, exposure groups and zero-PKL protocol. Global
batch272 is micro68 per GPU × accumulation2 × two GPUs; three epochs, seed1,
existing warmup/cosine and loss weights. The historical mHC m74 OOM and m68
three-epoch completion justify the config choice but do not prove new-source
capacity. Historical no-mHC configs/checkpoints and v1 remain unchanged.

Pre-change pushed main identity was independently verified as
`6d5cb7e4d05dcaeb7c8105dc04562a2627e9459a`. Targeted tests: 107 passed; scoped
Ruff/format and six-source-module mypy pass; wheel/sdist build pass. Expanded
checks: 538 passed, 11 CUDA/Triton skips, five unrelated failures reproduced
at the exact baseline. Details: `docs/experiments/GRADPERT_V2_MHC_JOINT_ONLY_B0.md`.
Next action is delivery of this update; any future formal run
needs a clean matching server release, same-config CUDA integration check and
new run ID. No supervision lease is needed while no job is assigned. Existing
unrelated coordination state/receipts are excluded from this code publication.

# Current 2026-09-30: full-state v2 B0 3→6 continuation accepted

The exact run `nadig_jurkat-seed1-20260929T143029Z-d83c2632bba14bddb595f8b8c0734805` completed attempt 1 with exit 0, cumulative 6/6 epochs, protocol-selected joint-loss best epoch 2, epoch-6 last, both six-Pearson test receipts, matching clean source/config/checkpoint identities, no failure marker and zero PKL. The main chat independently verified the server artifacts and hashes against `.byte-os/coordination/receipts/v2-continue6-c47f84e-terminal-20260930.json`. The run-specific `grad-pert-v2` monitor is PAUSED; no training or ablation was started on handback. Results and interpretation: `docs/experiments/GRADPERT_V2_NO_MHC_B0_CONTINUE6_RESULT_20260930.md`. The next action is scientific discussion or a separately authorized experiment, not further polling of this terminal run. The prior “running” section below is historical.

# Current 2026-09-29: full-state v2 B0 continuation from epoch 3 to 6 is running

Outcome: continue the completed no-mHC Jurkat B0 from its epoch-3 `last.pt` through epochs 4–6 at constant LR `2e-4`, then verify cumulative 6/6 epochs, joint-loss best selection, best/last test receipts, source/config/checkpoint/data identities, and zero PKL. This is a new 3+3 training stage, not a retroactive six-epoch cosine run. The immutable parent run and v1 remain untouched. Published implementation/source commit `c47f84e796aed994fdd060a9afbe14a95fa64069` adds the explicit six-epoch policy and self-contained config; server tests passed 79/79, dry-run passed, and same-config dual-GPU full-state one-update probe restored the parent, advanced optimizer step 1302→1303 at LR `0.0002`, changed trainable parameters, and saved/reloaded a checkpoint. Probe receipt: `/data/yilangliu/GraD-Pert/development/v2-continue6-c47f84e-full-probe-r1/COMPLETE.json`.

Active run ID `nadig_jurkat-seed1-20260929T143029Z-d83c2632bba14bddb595f8b8c0734805`, root `/data/yilangliu/GraD-Pert/runs-v2-continue6-c47f84e/` plus that ID; config SHA256 `66b6abdd0070d111a8275ba85739c6279cd5373559464b62eb0bb2be8473f956`; parent `last.pt` SHA256 `5daef1bac40c19a20ba4ee54a2637e1f187e0d1f502010ba06708fc45a303978`. Server checkout `/data/yilangliu/GraD-Pert/development/source-v2-continue6-c47f84e`, publication receipt SHA256 `e1001c1e4a962d729c2feb57b2ca17e96716978b586aff0f9d6dc9e473c6b23b`, runtime SHA256 `73d83138c6736337ffd25f1d495a3f9fd1a50fa2ace9f886b3016d48f4d1466f`. GPU 0,1; microbatch 74 per GPU × accumulation 2 = global batch 296; seed 1. Wrapper PID file, log, exit JSON share stem `/data/yilangliu/GraD-Pert/development/v2-continue6-c47f84e-formal-r1`; driver plan has exact run identity. Launch verification saw parent epoch 3 committed, epoch 4 live, wrapper/torchrun/both ranks alive, both GPUs computing, no exit/COMPLETE/FAILURE. The prior three-epoch parent is never resumed in place. Handoff: `.byte-os/coordination/handoffs/2026-09-29-codex-v2-continue6-supervise.md`; launch receipt: `.byte-os/coordination/receipts/v2-continue6-c47f84e-launch-20260929.json`. Supervision lease target is `codex:01a0df0b-4142-7df1-86c0-d959471d80a1` with monitor `grad-pert-v2` every 20 minutes only while this run is active. On terminal success or failure, supervisor returns the baton to `codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b`, pauses the monitor, and the main chat performs independent acceptance/repair. No other ablation is authorized by this run.

# Current 2026-09-29: v2 reusable training and evaluation interfaces delivered

Overall outcome: v2 single-/multi-GPU training and evaluation instructions, hash-bound independent checkpoint evaluation, and new full-state/LoRA training stages are implemented. Exact same-run resume remains distinct; old B0 and v1 behavior are preserved. Baseline `1b0f7064ff42311b8ac3615a13e84f5943ca43ba` was verified pushed before edits. Implementation `1cba4272f4d4c28281ab5efca0cf13057cb605d0` and evaluation progress/environment follow-up `5852acd46c9a1a8f1fa79827b76e4c502ad049fb` were published on main, with clean immutable server checkouts at `/data/yilangliu/GraD-Pert/development/source-v2-checkpoint-workflows-{1cba427,5852acd}`. Both 3+2 modes passed server dry-run. Actual dual-GPU one-update probes restored completed B0, changed trainable parameters at LR `2e-4`, saved/reloaded checkpoints, and reached optimizer step 1303; hash-verified `COMPLETE.json` files are under `/data/yilangliu/GraD-Pert/development/v2-checkpoint-workflows-1cba427-{full,lora}-probe-r1`. Single-GPU dry-run also passed. Full independent dual-GPU evaluation of the old `last.pt` under immutable `1cba427` completed at `/data/yilangliu/GraD-Pert/development/v2-checkpoint-workflows-1cba427-dual-eval-r1/COMPLETE.json` (SHA256 `21fab046917048aa5cb5aa55cec485bbd2d1aa3bf7b0342ddc4d5259738ae7e4`): all 592 conditions and all 9,920 numerical result values exactly match the original `fit/last-test.json`; frozen control/reference hashes, query recipe, and expression-exposure record also match. DEG metrics are finite for 590/592 conditions, as in the original. Entire evaluation root has zero PKLs; its processes exited and GPUs returned idle. The follow-up `5852acd` added progress and environment metadata and passed targeted server tests/dry-run; this metadata-only follow-up did not itself receive a full 592-condition GPU run. Expanded v2 suite had 452 passed, 11 skipped, and one unrelated pre-existing failure because `configs/v2/single_pass_jurkat/capacity_m66_a2/gradpert_v2/nadig_jurkat.yaml` is absent from baseline HEAD; targeted Ruff/format and wheel build passed. No formal 3+2 training or monitor is active. Do not stage unrelated historical `.byte-os/coordination` files.

## Historical completed B0 (2026-09-29)

The no-mHC single-stream/joint-validation run `nadig_jurkat-seed1-20260928T184412Z-2d6425f1e28e46b0b285af6096d03bd3` attempt 1 has completed three epochs (1302 optimizer updates), exit 0, with `COMPLETE.json` and both best/last tests. The main chat independently re-read the server's `COMPLETE.json`, `fit/epoch_state.json`, `fit/history.json`, and best/last test JSON and matched their SHA256 values to the supervisor's terminal receipt. The entire run root contains zero PKLs. The clean training/evaluation source is `ca7884e4e9b70bb55a61d97442467ff2531337b3`; the config SHA256 is `4c5ba9b39689b9d7993e023694cdedd0268668aa0f9e2f922e6c82e5cb25887e`. The protocol-selected best is epoch 2 by minimum validation joint loss 4.0076390792; last is epoch 3 with validation joint loss 4.0519638342. Full results and their interpretation are in `docs/experiments/GRADPERT_V2_NO_MHC_B0_RESULT_20260929.md`. The `grad-pert-v2` monitor is PAUSED because this run is terminal. No rerun or further ablation was started. The supervisor's terminal receipt is `.byte-os/coordination/receipts/nadig_jurkat-seed1-20260928T184412Z-2d6425f1e28e46b0b285af6096d03bd3-terminal-20260929T1032Z.json`; the completed-run handoff is `.byte-os/coordination/handoffs/2026-09-29-codex-v2-no-mhc-b0-complete.md`. Earlier sections below are historical snapshots, not the current run state.


## 2026-09-29：主会话完成 v2 验证/测试输出协议更新

主会话已发布 `10193b6edffca802410fef837fd918844d4cc6e3`：新 v2 运行的 checkpoint 选择改为固定视图完整 joint validation loss；best/last 默认分别输出全表达轴与按实际训练表达暴露划分的两组各三种 Pearson；进度显示总/轮内 step、速度、比例和验证/测试阶段。Teacher/center 在验证时只读，验证 RNG 不推进训练 RNG；一次条件推理同时形成三组指标。隔离服务器 v2 回归 434 通过，扩展相关测试 48 通过（既有缺配置测试 1 项跳过），最终定向 37 通过；Ruff 全仓通过。`mypy` 剩余 29 条位于未改旧模块，未宣称全仓 typecheck 通过。新源码已推送 main；**未启动训练或评估**。历史 B0 与已有分组结果保持原身份，不按新协议追认。若后续获授权启动新 B0，先使用新 ID、干净服务器 checkout、配置哈希与新 joint validation 容量预检；现无长时间监督租约。

## 2026-09-28：主会话接回结果；当前无活动训练交接

监督会话完成 `v2-best-expression-exposure-9ea0815-20260928` 终态核对并交回主会话：服务器结果 `best-stratified-test.json` SHA256 `603ffa7e2cba7f78e53f0566a73b56fdceaab0af00eb99ba1fe791bc84e8f840`，exit0、zero-PKL、GPU 空闲；`grad-pert-v2` 心跳 PAUSED。按先前授权已实现并发布后续 v2 运行的步数/阶段/损失/吞吐与验证、测试条件进度；源码 `87b44c8c1e8840c876bb63cdf783486aa40f7364`。当前没有新 GPU 任务或监督租约，旧 run 不重启。下一动作由主会话与用户讨论分组结果或安排明确授权的后续实验，不能以本分组指标调参测试集。

## 2026-09-28T04:39:59+08:00：m74 正式三轮 OOM 后的容量回退

m74 已在首轮中途 OOM，128 步通过不能证明整轮安全；旧 run 和 FAILURE 保留，不恢复。当前用已发布来源门版本 `8dfb267adb25591393602066ef1226e7cbb76174`，对 m68／图目标行64／序列chunk16 启动独立双卡128步持续探针，服务器根 `/data/yilangliu/GraD-Pert/development/v2-gate-m68-r64-s16-128-after-b0-oom-r2`；r1 因启动环境缺少 PYTHONPATH 在导入阶段失败，r2 已补齐。准备了新的三轮 m68 自包含配置，但须以探针终态、显存余量与吞吐决定是否采用；不能把进行中视为通过。本次只改变物理/全局 batch，完整预测＋SSL1＋SSL2、累积2和既定图/序列chunk不变。


## 2026-09-28T04:34:09+08:00：三轮 B0 attempt1 因 OOM 失败，交回主会话

精确运行 `nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143` 退出码1，`fit/epoch_state.json`停在0/3（434 updates/epoch），无epoch-0001、history、COMPLETE、best/last测试或test root；全run根扫描0个PKL。rank0 在GPU0上 dropout 申请128 MiB时OOM，报告仅余9.62 MiB，进程占31.34 GiB；rank1随后在NCCL ALLREDUCE seq5573超时300027 ms并SIGABRT。训练进程已全部退出，双卡回到0%/2 MiB。源码与配置身份匹配。failure JSON原件保存在服务器且19,977字节已核验SHA；小型副本和日志哈希索引记录在 `.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-terminal-failure-20260928.json`。本run不重启、不评估、不修改配置；等待主会话决定新的有界步骤。监督心跳暂停并交回。详细handoff：`.byte-os/coordination/handoffs/2026-09-28-b0-m74-three-epoch-oom-return.md`。


## 2026-09-28T04:12:23+08:00：服务器连接恢复，三轮 B0 仍在训练

TCP/22与SSH已恢复。精确进程 wrapper3829919、torchrun3830019、rank3830095/3830096均存活；`epoch_state`仍0/3（434 updates/epoch），history为空，只有epoch-0000初始checkpoint；无exit、COMPLETE或failure，测试根尚未创建。源码干净且提交/config SHA匹配。GPU0 29%/31944 MiB，GPU1 29%/32084 MiB，compute PID与两个rank对应。有限日志无错误；继续按20分钟监督。收据：`.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-supervise-check-20260928T2011Z.json`。


## 2026-09-28T03:53:52+08:00：服务器连接暂不可用

当前 EasyConnect 服务门户已登录且列出 `10.24.1.91`，但该资源显示 L3VPN 服务启动失败；辅助检查 TCP/22 与 SSH 均不可达。执行一次常规恢复并从已填表单登录后，页面仍报告启动失败；未做路由/DNS/代理更改。训练状态未知；上次可达观察为 epoch 0/3、进程存活且无 exit/COMPLETE/failure，不可视为当前状态。连接证据：`.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-connectivity-incident-20260928.json`。主会话已被通知；监控保持 ACTIVE，下一周期再做有限连接核验。

## 2026-09-28 首次监督确认

首次监督确认（2026-09-28T03:12:05+08:00）：Byte 所有权与 run ID/attempt 门禁匹配；服务器 wrapper、torchrun 与两个 rank 均存活，epoch 0/3；服务器源码干净且提交/配置哈希匹配。GPU0 为18%/30584 MiB，GPU1 为0%/30564 MiB。未见 COMPLETE、failure 或 exit 标记。20分钟心跳已启用。记录：`.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-supervise-check-20260928.json`。

## 当前运行：B0 m74/chunk16 三轮，监督交接中（2026-09-28）

运行ID `nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143`、尝试1、服务器根 `/data/yilangliu/GraD-Pert/runs-v2-b0-gate-m74-3ep-e821173/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143`，父PID3829919，源码 `e821173b4d11268b621ac8299e45b8d6d272b7e7`，配置SHA `3f1d3d9dde6bf151eeb3916809863eda0020fa89d8c82c74005d19c4d424dedd`，runtime SHA `988968f35f1e97558be2ec32deeb285e3a33781624da8a0407314083c8baab6c`，发布收据SHA `d5da951138794e9e5944754e40bc7536c8f32769f5e7e327060d2bf1c56e563b`。启动时launcher/torchrun/双rank存活，run manifest和计划已写，双卡占用；没有完成或测试收据。精确交接 `.byte-os/coordination/handoffs/2026-09-28-codex-b0-m74-three-epoch-supervise.md`，接下来须核对20分钟监控确实ACTIVE且绑定监督对话；终态要求3/3轮、COMPLETE、best/last、训练/评估身份及zero-PKL。任何失败保存证据，复杂修复交回主会话，不覆盖本运行。下方选型段落为历史。

## 当前阶段：m74/chunk16 选定，三轮完整B0发布预检（2026-09-28）

双RTX5090、完整预测＋SSL1＋SSL2、累积2下，来源门源码 `8dfb267adb25591393602066ef1226e7cbb76174` 的m74／图行64／序列16完成128/128步、checkpoint续跑和单条件300-control推理，exit0；峰值allocated/reserved 32,310,246,400/32,621,199,360 bytes。m76/78/80的持续测试OOM，失败原证据保留在服务器各独立ID。12步同路径吞吐排序m74两次13.794/13.791 cells/s，高于m66／序列32的12.687；m70／序列32短测OOM。选定全局batch296；收据与局限见 `docs/experiments/GRADPERT_V2_GATE_BATCH_CHUNK_20260928.md`。用户新授权独立3 epoch完整B0及best/last测试，本地已准备 `v2_fixed_3` 策略和 `configs/v2/source_key_gate_b0_jurkat/three_epoch_m74_a2/gradpert_v2/nadig_jurkat.yaml`；下一动作是定向测试、提交推送main、干净不可变服务器checkout、同配置预检，然后新运行ID启动。旧B0不恢复、不覆盖。当前无本任务GPU运行；下方更早阶段均为历史快照。

## 当前阶段：来源门容量与chunk配对（2026-09-28）

目标是双RTX5090、累积2、完整损失下，最大经过128步持续验证的物理微批，同时比较可用chunk组合的含数据等待吞吐与峰值显存。先前新来源门m66全局264只通过单步；旧图版m66持续通过、m68第22步OOM不可移作新证明。第一组九项双卡单步已在不可变源码`f090b726ba0df0e02a2fb058e727236c9d9dfcef`全部通过，原始receipt在服务器`/data/yilangliu/GraD-Pert/development/v2-gate-capacity-f090b72-{variant}-integration-r1/`；m70 seq16较seq32少用约3.45GB峰值预留，rows32未减峰值且较慢，故追加seq16 m72/74/76/77/78/80及seq32 m72。配置设计与序贯预检→持续容量→同batch吞吐门槛在`docs/experiments/GRADPERT_V2_GATE_BATCH_CHUNK_20260928.md`。下一步发布扩展配置的干净提交，使用新服务器checkout及独立ID查上界，随后选择候选做128步持续容量。训练等待交同任务子代理只读监督，主会话不重复轮询。旧B0保持停止，非容量测试不启动。

## 当前默认：图来源门开启、原分块、batch264（2026-09-28）

用户要求结束大chunk默认尝试，同时保留新图编码器。新版本 `d2410efdc81a9690644ef2d165c2dea374ba0fbb` 已推送 main；自包含配置 `configs/v2/source_key_gate_jurkat/one_epoch_m66_a2/gradpert_v2/nadig_jurkat.yaml` 只在先前 `optimized_single_pass_jurkat/one_epoch_m66_a2` 上增加 `graph_source_key_gate=true`。图邻域KDA scan32、序列继承scan32、图目标row64；旧配置和独立chunk对照保持可复现。每卡微批66×累积2×双卡，全局264；这是旧模型已通过128步容量的batch，**不是新门控已验证的持续容量**。

新配置 SHA256 `f0f69871fdd8e797645d85b2faf434740d685be7c9370b6f063da4ac3fb3e844`；干净服务器源码 `/data/yilangliu/GraD-Pert/development/source-v2-gate-default-d2410ef-from-local`，publication收据SHA256 `020b778d4db5aa78e8ee7e7b31eec057c3a3f81a7c8d84782e3f0b8a3afd6dc8`。双卡单步完整更新1/1通过并重载checkpoint，峰值 allocated/reserved 31,064,793,600 / 31,427,919,872 bytes；收据 `/data/yilangliu/GraD-Pert/development/v2-gate-default-d2410ef-m66-integration-r1/receipt.json`。服务器81项定向测试通过，实验进程已退出、GPU空闲。下一次正式训练前须做新门控版本的持续容量验证；本轮未启动B0或其他消融。下方大chunk工程测试结论为历史，不能再读作后续默认。

## 当前任务完成：第四层来源门与图64／序列256分块（2026-09-28）

用户指定的方法与工程配置已经实施：图第4层边来源逐维key门零初始化并保留来源偏置，图邻域KDA scan chunk64、Cell/Response序列KDA chunk256；图目标row chunk64/96/128各有独立配置。基线提交`9a820e1d2a2dfc38bdcfee9370ecb1e518733b2f`先验已推送；实现提交`657b63ad509f5511b3cb4a585be0dc44a92aff65`及回退微批配置提交`2043e09a4f4cb29317b9879da68e32e8d953ee1c`均推送main。最终干净不可变服务器源码`/data/yilangliu/GraD-Pert/development/source-v2-gate-chunks-2043e09`；publication收据`/data/yilangliu/GraD-Pert/development/gradpert-v2-gate-chunks-2043e09-publication.json`，SHA256`898df95cdfbcba7fb0e8ba54515ca317e4d08489d418a129c247d30d075b6734`。服务器定向80项测试通过。

双RTX5090全局batch128（m32×累积2）下，门控单独单步通过，chunk-only与完整组合反向OOM；失败ID/收据均保留。统一全局batch64（m16×累积2）下六组4步完整预测＋SSL1＋SSL2均通过，最后一步学习率非零，细胞批次顺序哈希相同。参考3.373 cells/s、9.14GB/卡；完整组合3.201 cells/s、26.39GB/卡；row96组合3.716 cells/s，row128组合3.963 cells/s，row128反序复测3.978对参考3.376。图行块扩大虽有重复短程速度收益，但此前严格语义审计未通过逐目标随机分配与有限精度等价；本次无持续128步和一轮效果证明，不改变正式性能默认或启动B0。完整配置/结果/收据/限制见`docs/experiments/GRADPERT_V2_SOURCE_GATE_CHUNK_20260928.md`。当前没有由本任务运行的GPU进程；旧B0/收据原样保留。下一步若用户授权正式采用某候选，先解决随机语义与容量，再以新run ID继续；下方历史计划不表示正在运行。

## 当前阶段：chunk 对照已结案，B0 保持停止（2026-09-27）

主会话完成双卡扫描32/48/64及图目标行64/96/128工程对照。没有通过“语义＋端到端收益＋容量”全部门槛的候选；维持扫描32、图行64和 `relay_sequence_chunk_size=None`。λ₂=1 及蒸馏②实际权重0.8/0.4/0.1已写入未来新配置；以后 v2 新实验按1 epoch，不覆盖旧B0。旧训练已停在0/5，不启动新B0或其他消融。报告 `docs/experiments/GRADPERT_V2_CHUNK_SWEEP_20260927.md`，工程测试源码 `d61af46d9e4282c8e5b69e9514d1134407d2b8e5`，收据均保留服务器。下方“训练运行中”等记述是历史快照，以本节为准；监控保持暂停。

## 当前执行：m68 OOM 后验证 m66（2026-09-27）

主会话拥有容量与后续 B0 启动。NUMA m68 全局272新 ID 128步测试在 rank1 第22步反向 OOM，exit1；`rank-1-failure.json` 是终态，顶层 receipt 停在16步旧状态。原始失败证据保留于 `/data/yilangliu/GraD-Pert/development/v2-numa-m68-28f447a-capacity-20260927T0929Z`；GPU0/1 已空闲。下一动作：从干净已推送 `28f447a24a400659ea09f9c0f6d735ea7e4ec8f1` 只变微批68→66、全局272→264，定向推送 main，建立干净不可变服务器源及发布收据，独立 ID 单步预检→128步持续、checkpoint 续跑及300-control。若 m66 OOM，保留失败并选 m64，在新源码复测。容量通过才启动新 ID 完整预测+SSL1+SSL2五轮及 best/last，长时训练交指定监督会话；旧停训 B0 不恢复。下方 m68 进行中快照是历史记录。

## 当前执行：m68 128步持续容量（2026-09-27 09:29 UTC）

主会话拥有容量阶段。干净发布源码 `28f447a24a400659ea09f9c0f6d735ea7e4ec8f1`、服务器 checkout `/data/yilangliu/GraD-Pert/development/source-v2-numa-m68-28f447a`、publication SHA256 `3e03290ea72f9d03fc05224413c07bbc012bc1213ebc47e44dcabb2a9aeee5e9`、配置 SHA256 `ab8c358ecdd7d6ee398120ebcaafd048bb25814263eb5b2e8271f479662735a9`。m68单步预检独立 run 1/1 exit0，持续 run `/data/yilangliu/GraD-Pert/development/v2-numa-m68-28f447a-capacity-20260927T0929Z` 已启动，父PID3689686；脚本 SHA256 `3c503b566a822048757ffb77e21d513512974f09b5e4d4b0097c746349ab8f42`。只读 Luna 短时监督，主会话收到失败/终态即推进：通过则核对128步、恢复、300-control及可用 batch 后启动新 ID 完整 B0；OOM则保留失败并评估m66或回退m64，不覆盖本 run。两张GPU由本测试占用，无其他消融；旧B0不恢复。下方是历史快照。

## 当前所有权／动作（2026-09-27 09:24 UTC）

主会话持有性能阶段；NUMA A1/B1/B2/A2 队列 `complete`/`exit0`，Luna 短时监督已结束，两卡空闲。比较收据 SHA256 `813bd55bef4715ceb430944d251cc7ee7e319fbd29043996405d0cb3c2886ed4`，同源码 `32b2bfd88f36b9a7fad435c533dc825ef81942b4`、同配置 SHA256 `23a2942804b73cdf92871e36808846f50382ded41b1588a68da76ebc0e69985b`，同有序 batch/视图 RNG 起点；只变 rank 本地 NUMA 绑定。配对吞吐几何比 1.05083，显存无增加，决定采用于当前服务器。下一动作：将文档及自包含 m68 容量候选定向提交推送 main，在新干净不可变服务器 checkout 发布收据并做双卡128步持续容量；失败保留，安全回退 m64。容量门槛通过后新运行 ID 完整 B0 五轮与 best/last。旧 B0 中止不恢复；跨会话旧监督仍暂停。下方章节是历史快照。

## 当前所有权／动作（2026-09-27）

主会话负责当前性能构建；无活动双卡任务或长时监督。短图融合 `27aec3643cc613b17f3b87eecebf05d1453743ba` 的双卡确定性完整更新失败，收据 `/data/yilangliu/GraD-Pert/development/v2-short-27aec36-parity-m2-datafix-20260927/` 保留，不进入容量/B0。KDA 形状常量复用在干净发布 `32b2bfd88f36b9a7fad435c533dc825ef81942b4` 上已双卡两步严格一致，A1/B1/B2/A2 各12步通过；含数据等待吞吐比 1.01051/1.01361，几何均值 1.01206，比较收据 `/data/yilangliu/GraD-Pert/development/v2-kda-constants-32b2bfd-abba-20260927T0826Z/comparison.json`。收益较小，保留 opt-in。后续区域 CUDA Graph 与图边相同基因/来源的预投影都已在局部梯度门槛失败，未推送/未用于正式更新；日志及诊断副本保留在服务器 development。当前下一步：针对双卡跨 NUMA 拓扑与约130万 kernel 发射检查各 rank CPU 亲和调度，做同源串行整步对照；若无收益即停止该方向。再决定持续最大batch与新ID完整B0五轮。旧已停止B0不恢复，旧跨会话监督保持暂停。当前正在运行的任务以实时进程/收据为准；下方旧状态是历史快照。

## 当前执行：主会话性能工程（2026-09-27，新授权）

用户重新授权性能调研、双卡实测、等价机制优化及其后容量与全新 B0 正式运行；旧 B0 仍中止且不可覆盖。主会话拥有 build 阶段，当前无跨会话监督/定时监控；短时测试用 Luna 子代理只读监督。当前源码 `681d4fb609644d51c22f6b4e51dd61256adc3310` 的 m64 128步容量已通过，m72 OOM 保留。m32 完整更新 profile 已 exit0、6/6 passed，收据及 trace summary 位于 `/data/yilangliu/GraD-Pert/development/v2-mech-681d4fb-m32-profile-20260927T0727Z`；普通更新中位17.550秒，GPU/kernel与图发射详情已记录在 `docs/experiments/GRADPERT_V2_SINGLE_PASS_PERFORMANCE.md`。第一候选是独立短图邻域的融合最终状态递推：FP32局部加速约4.5倍，首版BF16 value梯度在极少量舍入边界失败；保留失败证据并采用原块求解计算该梯度的混合候选，相关服务器测试59项通过，BF16局部前后向2.803→1.782ms。尚未证明真实双卡完整更新等价或整体加速。临时checkout `/data/yilangliu/GraD-Pert/development/source-v2-short-provisional-20260927T0750Z` 仅用于不洁诊断测试，不能正式运行。下一动作：发布仅opt-in候选→双卡完整非零LR多步等价→同配置ABBA端到端，未通过则淘汰/修复。之后持续容量与新ID完整B0。细则见 `.byte-os/plans/GRADPERT_V2_MECHANISM_PERFORMANCE_20260927.plan.md`；若启动超过一小时的正式运行，再建立可返回主会话的耐久监控，不使用旧暂停心跳。

## 当前交接：用户暂停实验及监督（2026-09-27）

主会话 `01a0c01a-0611-7a90-b3e8-8ad7e017748b` 已接回所有权。B0 运行 `nadig_jurkat-seed1-20260927T041339Z-e35f5edb8e6248479462589ff3cef41f` 按用户要求 SIGTERM 中止，全部训练进程退出，双卡空闲；已提交 epoch 0/5，仅保留初始 `epoch-0000.pt`，无 COMPLETE、best/last 或测试收据。中止证据见 `.byte-os/coordination/receipts/b0-681d4fb-user-stop.json`（SHA256 `9e792365f37ec635483eb1da5a61dc993e65fa2bb30298e1bc68f09153af5451`）。监督会话停止工作，`grad-pert-v2`、`grad-pert-v2-b0-return` 两条自动化均暂停。下一步等待用户明确重新指示；不得自动恢复或启动实验、重启监督。下方运行中状态均为历史快照。

## 当前交接：完整 B0 五轮双卡训练已启动，交监督会话

m64 持续容量接受条件全部通过（exit0、128/128、checkpoint continuation、300-control、源码/配置身份），容量 receipt SHA256 `31a0798737a60bd318a861fd5e4d947a51f4d943ac0f613d0328ca4a3a7bfc04`。完整 B0 run ID `nadig_jurkat-seed1-20260927T041339Z-e35f5edb8e6248479462589ff3cef41f` 已启动，父 PID 3613045、双 rank 存活、`fit/epoch_state.json` 初始 epoch0/5；五轮及 best/last 未完成。源码 `681d4fb609644d51c22f6b4e51dd61256adc3310`，配置 SHA256 `23a2942804b73cdf92871e36808846f50382ded41b1588a68da76ebc0e69985b`。长时监督边界、路径与返程心跳见 `.byte-os/coordination/handoffs/2026-09-27-b0-681d4fb-supervise.md`。监督会话接手后才开启本运行的 20 分钟心跳；终态即暂停/删除。此节覆盖下方 m64 进行中记录。

## 当前交接：m64 容量测试主会话接手监督

活动 run `v2-teacher-681d4fb-m64-capacity-oomfallback-20260927-1203`，PID 3602712；2026-09-27T04:17:04Z 实查 32/128、双 rank 存活、无 exit/failure。主会话与子代理按新流程负责约1小时内测试的有界监督；跨对话监督心跳已暂停。m72 OOM 与 m64 fallback 证据见 `.byte-os/coordination/handoffs/2026-09-27-teacher-m64-capacity-return-main.md`。容量 acceptance 仍需 128/128、checkpoint reload 与300-control；未启动 B0。通过后主会话立即核验并启动此前已授权的完整 B0 五轮；失败则保存证据并决定修复/回退，不能把一次报告当作阶段终点。

监督规则：预计约一小时以内的有界测试，由本会话子代理只读监督并向主会话回报；更长的训练/消融才交指定跨会话监督。跨会话定时心跳仅在其负责的活动后台任务期间启用；终态或交回主会话时暂停/删除，需要新长任务时再开启。主会话收到任何监督终态后，在同一轮推进已授权的下一依赖。规则已同步到两份项目 `AGENTS.md`。

## 2026-09-27：m72 双卡128步容量测试已启动

Teacher Sinkhorn 优化双卡ABBA comparable，描述性总耗时下降3.84%，保留；m72×2×2=全局288单步预检1/1 passed，峰值预留显存31,279,022,080 bytes。已在干净发布源码681d4fb609644d51c22f6b4e51dd61256adc3310上启动m72的128步持续容量工程测试，PID3599854、运行根`/data/yilangliu/GraD-Pert/development/v2-teacher-681d4fb-m72-capacity-20260927`；当前只确认进程和0/128初始收据。监督交接见`.byte-os/coordination/handoffs/2026-09-27-teacher-capacity.md`。容量/B0仍未完成；此节覆盖下方历史空闲记录。

## 2026-09-27：Teacher Sinkhorn ABBA 已启动，交监督会话

源码 681d4fb609644d51c22f6b4e51dd61256adc3310 的本地 v2 351 测试、服务器 CUDA 16 测试通过；双卡确定性两步完整更新精确通过。非确定性校验与同路径重复对照均未过严格梯度阈值，保留收据，不混作候选失效证据。A1/B1/B2/A2 全模型 12 步串行队列已启动，PID 3531359，A1 收据真实存在；运行根 `/data/yilangliu/GraD-Pert/development/v2-teacher-681d4fb-abba-20260927`。按 `.byte-os/coordination/handoffs/2026-09-27-teacher-abba.md` 交由指定监督会话观察，完成或失败立即交回主会话。容量持续测试与 B0 五轮未启动。此节覆盖下面历史“无活动任务”记录。

## 最新状态：主会话负责工程，监督会话负责后台等待（2026-09-27）

主会话 `01a0c01a-0611-7a90-b3e8-8ad7e017748b` 负责模型设计、论文核对、性能实验设计、实现、验证、发布与启动；监督会话 `01a0df0b-4142-7df1-86c0-d959471d80a1` 只在有明确交接的后台阶段接管检查。Teacher 无梯度 Sinkhorn 存储消除候选的双卡 ABBA 已于 2026-09-27 完成：四组 12/12、`queue.exit=0`、比较为 comparable，描述性总耗时下降 3.84%；这不是正式训练或科学结果。当前没有活动 GPU 任务，所有权已交回主会话，由主会话审阅并决定下一有界阶段。持续容量与完整 B0 五轮及 best/last 仍未完成；不自动启动其他消融。

用户指定不用 Goal 模式；工具当前也报告无活动 Goal。已建立监督会话的 20 分钟心跳 `grad-pert-v2`，启动后立即触发一次。没有交接中的后台运行时仅核对状态并保持安静。主会话每次交接必须写清阶段、运行 ID/PID/日志、不可变源码与配置哈希、GPU/时间边界、预期完成收据、失败处理和下一步；监督会话以服务器实时证据核对。发现故障或里程碑后立即保存证据并消息唤起主会话，不等下一轮；主会话接回后监督会话停止对该任务重复操作。20 分钟是周期检查上界，单靠定时器无法在两次检查间瞬时发现故障；新后台任务如需故障发生即唤起，应在启动时加独立退出/失败事件通知，并先验证其可靠性。用户暂停/停止指令优先；全部任务完成时删除心跳。

上面的双会话安排覆盖下文旧的“Goal active／无定时监控”和阶段 F 的旧检查间隔，不改动历史实验事实。

## 显式mHC配置完整更新通过；下一步无梯度存储消除候选


### 显式后端配置完整更新通过

9d9bad86e16286b590b2bb31c5df5103ecfbbccb干净发布服务器source-v2-opt-9d9bad8，
publication SHA256 da83f52334078f742997eb4d2e3c850abba1b5234c13e09bfaceaa7688977294。
同profiling_m2_a2协议，仅candidate配置从native改auto；无诊断融合开关。
两rank两步全部比较最大差0，输入/RNG相同，第二步LR7.796055196070788e-8。
exit0，收据single-9d9bad8-config-parity/共42414bytes经dry-run归档。
这验证正式配置接入等价；不是新的吞吐或持续容量结果。当前GPU无活动任务。
下一文献指导候选是Teacher无梯度Sinkhorn的无用中间存储消除，详见文献矩阵。

FLA冻结完整前后向已读，直接替换会保留逐token输出，不能宣称省算。
文献矩阵新增具体候选：Teacher无梯度时每token2560bytes反向概率无需保存，
保持40归一化与Student梯度路径；先独立数值/完整更新再ABBA，未实现。
原目标仍为最终执行组合→持续双卡最大batch→完整B0五轮best/last。

# 当前状态 — 2026-09-26

## 授权与目标

历史状态：当时的目标为新方法升级与性能工程 → 双卡持续容量 → 完整B0五轮及best/last；2026-09-27 起改用上方双会话工作流。
不启动其他消融；不恢复旧停止运行。默认单向KDA、最终S统一读取，完整双蒸馏。
此处“无定时监控”为当时记录；现以顶部状态为准。保留历史来源，禁止修改活动服务器源码或覆盖runID。

## 当前阶段：融合mHC Sinkhorn完整双卡更新校验

trace两轮分析均完成，结果在docs/experiments/single-38af3ce-profile-replay/costs/。
1414510 kernels；logsumexp去重CPU区间1.5996s、129600 kernel累计0.166s。
不把嵌套时间相加；主要机制还包括图邻域/重计算/KDA临时量。先验证mHC融合，
保持20轮与解析梯度，不减少迭代、不改模型数学。KDA Gram候选尚未实施。

独立Triton融合算子18/18双设备合成case通过，原始收据
`docs/experiments/single-59305ee-sinkhorn-probe/receipt.json`。
最大输出/梯度差4.92e-7/3.88e-7，原4.33–5.04ms→0.514–0.656ms；
4096token临时峰值14,221,312→11,010,048bytes。非完整训练吞吐，不作采用依据。
首版4f855f9编译scope错误已修，失败收据保留；未改活动源码。
本地61测试通过，服务器18定向测试通过。

前次完整更新67fd8e1失败，第1candidate更新gradient最大差1.6681e-4，
ssl2_koleo亦超容差，输入/RNG完全一致。两rank收据已入库，不采用，不跑其ABBA。
经安装Torch Reduce.cuh证明strided四元素和逐项结合，修正Triton行轴归约；
7ae9a8b微基准18/18组、全部40阶段、输出及梯度逐元素相等，BF16差异0。
证据docs/experiments/single-7ae9a8b-sinkhorn-probe/receipt.json。

动态N cab78a1已通过双卡30组微基准：全部40阶段/输出/梯度exact，
每卡forward/backward各1编译variant。完整两rank两更新也全部max差0（含loss标量），
第二步明确非零LR7.796055196070788e-8。收据已dry-run复核281810bytes入库
`docs/experiments/single-cab78a1-sinkhorn-parity/`，未将第一步LR0冒充非零更新。

本地完整v2回归340passed（113b96a阶段）；接口类型修复后20定向测试及4文件mypy通过。
ABBA已终止exit0，四组12/12、完整审计comparable。
A1/B1/B2/A2含等待均值21.3293/20.4358/20.2642/21.3046秒；吞吐+4.7519%。
收据docs/experiments/single-b64745b-sinkhorn-abba-m32/；源码
b64745b7cc2a752ab992a32c62a234193a676cad，config SHA256
c03f83643f208af475765d2491401fd2de0b6b0091d96c7243712b21152b5a8f。
mHC通过完整精确更新和代表性batch性能门槛，保留；正式配置接入待完成。
独立Gram probe源码c7e89e8525a149026bdb17354e289e42fda0a28a，exit0、18组通过容差，
梯度exact、前向最大差4.47e-8；小形状偏慢，不采用默认。
收据docs/experiments/single-c7e89e8-gram-probe/receipt.json。
当前无活跃GPU任务，两卡各2MiB/0%；下一步核对Gram归约顺序并据新假设优化，
再完整更新/真实吞吐验证。不得放宽容差，不重复无新假设的失败候选。
保持单向KDA统一final-state读取；持续容量和完整B0尚待完成。

先前容量短检查全部终止：a1d55fa micro8/16/32/48/64 passed1/1，
3ba9a96 micro72 passed1/1+restore、micro80 OOM、88未启动。
各原始收据在对应docs/experiments/single-*-capacity-integration目录。
单步不证明持续容量；机制优化验证前不启动容量长测/正式B0。

## 已完成的性能取舍

CPU预取ABBA已终止exit0，四组12/12通过且配置/数据/RNG/源码审计通过。
A1/B1/B2/A2含等待均值21.6529/22.3103/21.8072/21.4106秒。
预取平均慢2.4476%，故不采用；同步路径默认保持。两次双卡更新数值完全
一致但不构成速度收益。详见 `docs/experiments/single-ec2384a-prefetch-abba/comparison.json`
及同目录原始收据（已dry-run356,094bytes后同步）。

性能总结 `docs/experiments/GRADPERT_V2_SINGLE_PASS_PERFORMANCE.md`。
全序列CUDA Graph候选10/12触发重编译64上限，淘汰当前路径，不增加上限重跑。
无序列重计算两次更新精确一致但显存约3.27→13.55GiB，暂不采用。
默认单向final-state eager+重计算；旧batch192不能证明当前方法容量。
完整本地v2测试312passed；新增benchmark因子审计9passed。

## 运维

本地工作树 `/Users/elan/code/grad-pert-v2-build`，定向提交推送HEAD:main。
服务器Python `/data/yilangliu/GraD-Pert/source/.venv/bin/python`。
SSH `ssh -S none -o BatchMode=yes -o ConnectTimeout=10 10.24.1.91`，当前可用。
服务器无rg，可用Python筛选输出。数据/权重/大trace只留服务器；小收据先dry-run。
共享Git克隆不可无限叠加；独立基底source-v2-replay-473466d-standalone可用。
干净发布应从无ignored egg-info的独立本地克隆生成，保持源码身份核验。
旧运行细节及已停止PID仅在STATUS/history/Git保留，不当作活动任务。
## 当前所有权／动作（2026-09-27 08:27 UTC）

主会话拥有当前性能阶段；已完成的双卡完整更新 parity 为 `32b2bfd88f36b9a7fad435c533dc825ef81942b4`、两 rank 两步 passed、输入/RNG/全部损失/梯度/optimizer/Teacher/center 差值零，第二步 LR 非零。短时 ABBA 队列 `/data/yilangliu/GraD-Pert/development/v2-kda-constants-32b2bfd-abba-20260927T0826Z` 正运行，PID `3668186`，顺序 A1/B1/B2/A2；仅本会话 Luna 子代理只读监督，旧跨会话心跳保持暂停。原 B0 中止且不恢复。队列终态后主会话立即审计总步时、含等待吞吐、峰值显存、启动成本与哈希；有效才进入新源码128步持续容量，否则淘汰并依据关键路径选择下一机制。用户授权范围仍含新 ID 完整 B0 五轮及 best/last，不启动其他消融。下方章节为历史快照。


## 2026-10-10 Unified MLP first batch implementation

Only N0/U24/MR1/P1/C1/O1/VH/S1-L4/CG1/S12-L4 are authorized. New common MLP, masked response, population mean/MMD, conditional graph, raw shared readout and independent Local views implemented. Target tests133 pass; full v2 regression737 pass/11 environment skips/1 missing historical fixture failure reproduced on immutable40fb032. Scoped Ruff/format/mypy pass, no first-batch review blocker. Student trainable counts N0/U24/CG1=42699822/42980782/45596478; frozen prior13324288. Candidate batch32/world1/accum1 requires CUDA10-update restore and300-control gates. No scientific completion or capacity claim yet.

Plan: plans/GRADPERT_V2_UNIFIED_MLP_FIRST_BATCH_20261010.plan.md. Report: docs/experiments/GRADPERT_V2_UNIFIED_FIRST_BATCH_20261010.md. Source implemented in a415d8c; next publish final snapshot, immutable server checkout, worst-first gates, then automatic six-epoch two-lane first batch only. Main owns build; old U4/U3 monitors remain deleted. VPN UI/TCP/SSH recovered and GPUs idle.
