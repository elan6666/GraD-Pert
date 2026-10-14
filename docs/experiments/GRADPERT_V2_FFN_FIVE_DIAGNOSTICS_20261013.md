# Five FFN arms after official scDFM and GEARS

## Current override: 2026-10-14

The user prioritizes DP→FN→FG→FB→PO while deferring scDFM evaluation. The explicit
`scdfm_fit_gears_terminal_evaluation_deferred_v1` policy verifies original completed
scDFM fit/checkpoint hashes and GEARS real terminal metrics; it never invents a
scDFM COMPLETE or uses its interrupted 48/592-condition evaluation as a result.
The strict all-evaluations default remains unchanged. Original scDFM last.pt stays
server-only with SHA256 `9dff79f7e2836482ea6f272dd619c9df967e73f8e5b16f911679c3d557c62a8c`.

Source `1fa8cd48d89b1bf590a9fdf73a633505d2e522a8` passed 134 local/server
checks and DP ten-update/restore gate. Its fresh queue
`native-ffn-five-deferred-1fa8cd4-20261014` then failed before evaluation computation:
the eval leaf allowed only historical BR queue families. DP receipt SHA256
`275b27b94c2a6cf5954d86a355b57ef0675bcf7c2d377e08d494f378314ede40`;
failure receipt SHA256 `0cd174237b4d1d4c89e89384430bd6459c5bf5a5fae6cf4bbfe3eb63e5dc6561`.
DP eight measured updates took about21.84–23.21seconds each; peak allocated
26,903,244,800bytes/reserved27,277,656,064bytes. These are short engineering
measurements, not scientific results. No formal arm launched.

Repair adds the exact five-arm family/order/mode/ten-step leaf check, preserving
per-row source/config/publication/GPU/batch bindings and inference tolerances.
Tests accept the valid leaf and reject extra BR, wrong mode/steps/GPU/config.
Fresh immutable publication and a new queue are required; old failed evidence
and original scDFM checkpoint are preserved. Scientific configurations unchanged.

User authorization: 2026-10-13. Clean pre-change main/GitHub source: `141989eeb18719aad2d790012eb6fc7366c31e9b`. This family does not rerun BR or completed Scouter arms. The baseline means official scDFM/GEARS; completed native BR400 is the comparison anchor (training `6a8c612bd7733d2f03b5c041aeff63c5f6ec8339`).

## Protocol and scope

Canonical Jurkat cap40, same split/expression exclusion, frozen 300 controls, common DEG and Systema reference; native four epochs from scratch, no validation/best, last-only 18 Pearson entries with valid condition counts. Global400 = micro100 × two GPUs × accumulation2. Evaluation uses both GPUs with cell batch300; all ten-update gates include restore and 300-control inference. Candidate diagnostics are read-only and never select a checkpoint. No data or scientific matrices leave the server.

| Arm | Only mechanism change from BR | Notes |
|---|---|---|
| DP | Four prototypes weight_norm and first-epoch LR0 | g=1, directions keep existing Linear initialization; moments continue, Teacher EMA and centers continue. Combined parameterization/early-update strategy; not all DINOv2 initialization/optimizer settings. |
| FN | Pre-LayerNorm at FFN and vector-role MLP inputs | FP32 statistics, eps1e-6, scalar expression lifted first; existing FFN norms replaced, attention/router/KV/output norms retained. LN beta retained. |
| FG | GELU FFNs | Main hidden1536 vs SwiGLU1024 matches matrix weight count; GenePT2048→4096→1024→256 remains unchanged; KDA/mHC mechanism gates retained. |
| FB | All FFN/role MLP Linear biases disabled | Does not remove LN beta, graph source biases, KDA decay parameters, or mHC routing offsets. |
| PO | Softplus final expression score | `prediction=softplus(score)` computed FP32; `delta=prediction-control`. Positive expression permits negative delta; no clipping. Finite scores cannot produce exact zero. |

All common graph/Cell/Response, views, losses, Teacher/center, data, Muon/AdamW routing and schedules remain fixed except declared parameterization changes. Scalar prediction outputs keep AdamW when FN nesting changes module paths. FG matches core weights, not every activation-memory/FLOP cost. DP weight_norm changes initial prototype row scale and parameter optimization; it is not an equivalent speed transformation.

## Fixed training diagnostics

Opt-in `process_diagnostics=true` only in this new family; defaults and historical checkpoint architecture payload unchanged. At initialization and epoch boundaries a private fixed training recipe selects four rows and allowed expression queries. No validation/test expression input is introduced; graph prior nodes remain permissible. Existing basal prediction-vs-SSL2 gradient diagnostics remain. New graph probe compares weighted prediction/SSL1/SSL2 on `graph.adapter`, `graph.layers` and all graph parameters, recording zero derivative honestly.

Recorded small aggregates:

- Four Student/Teacher head entropies, marginal entropy, JS sample dependence, argmax usage/top1 and bounded actual CE-pair decomposition `CE=H(Teacher)+KL`.
- Pre/post projector and per-layer graph/Cell/Response gene/CLS centered variance, relative variance and covariance effective rank.
- Prediction negative fraction/magnitude, delta mean/std/RMS, random-paired effect RMS, control-copy gain and worst1% error contribution.
- Gradient norms/cosines, center statistics, input/gene/view hashes, sampling counts, elapsed diagnostic time and capacity peak memory.

Budget: first128 feature rows/dimensions≤256; deterministic prefixes are not unbiased population estimates. Statistics are rank0 local fixed probes, not global training KoLeo/population metrics. Actual training reductions/cross-rank neighborhoods do not change. Optional three intervention forwards are implemented but off by default: rolledcontrol is explicitly not context matched; rolledcondition counts actual changed rows; alternate KDA order seed preserves gene axis. Full context-matched interventions, seen/unseen token attribution, CKA and gate/source coverage panels are not yet materialized by this first version. Final 18-metric grouping remains unchanged. Do not infer biological usefulness or complete collapse from any one statistic.

Diagnostics use eval mode and autograd.grad, never optimizer steps/EMA/center commits. Module modes, temporary hooks/CE observer, pending statistics, auxiliary counters, RNG and existing `.grad` are restored. CPU nonzero-LR two-step comparison with and without probes is exact. GPU engineering preflight must include initial/post-update probes before scientific launch; historical two-step passes do not prove long-term model effect.

## Official recovery and dependency

scDFM batch48 FP32 OOM and prior contiguous pinned-offload parity failure remain historical. New adapter retains saved-tensor shape, strides and storage offset, including broadcast views and offset alignment. It calls the original official model/train_step/MMD/optimizer unchanged. Prefix/span copies can increase CPU/pinned memory and transfer cost; no speed or GPU capacity benefit is claimed before actual measurement.

New official queue runs strict none-a/none-b/cpu three-update GPU comparison (gradients, checkpoint optimizer/scheduler, RNG, prediction) at original rtol1e-6/atol1e-7, then real ten-update/restore/300-control preflight, scDFM four epoch-equivalent sample budget (3988 updates), last evaluation, then GEARS four epochs and last evaluation. Any failed numeric or scientific receipt stops continuation. Unknown foreign jobs cause CPU resource waiting before CUDA dispatch; no whitelist expansion is implied.

Native controller validates both official COMPLETE and per-run last evaluation/checkpoint/config/control/truth/hash/zero-PKL evidence before GPU leases or native probes. After all native ten-step/evaluation capacity gates, existing MAIN_ACCEPTANCE returns to main for evidence review; supervisor must wake main at this milestone. No automatic scientific BR run is present.

## Verification at build stage

155 combined adapter/native/config/queue/probe checks and six strict-numeric-gate checks passed locally. Read-only review checked dependency ordering and layout restoration; no GPU/scientific completion is inferred. Actual deployment/run identities and gates are recorded in Byte state/compact receipts.

## Primary mechanism sources

- [DINOv2 prototype head](https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/dino_head.py), [LR0 first-epoch implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/train.py): direct parameterization adaptation; other projector initialization/widths remain our experiment.
- [DINO](https://arxiv.org/abs/2104.14294): entropy and teacher-target diagnostics. Statistical observables adapted, not upstream model runtime.
- [Official scDFM frozen source](https://github.com/AI4Science-WestlakeU/scDFM/tree/8de47e7d3d443939f93f768f7902c66e0e82eda2): external runner directly invokes official code; saved activation storage is adapter-side only.
- [CKA](https://proceedings.mlr.press/v97/kornblith19a.html): proposed later diagnostic, not implemented/claimed in current receipt.

## 2026-10-13：GPU数值门通过，官方恢复与后续五组依赖已启动

训练发布SHA `9607981fd7d44b3982574167132d06f739bc1b7e`，服务器独立干净checkout已核对。最终scDFM/GEARS各自冻结环境CPU十更新、恢复预测精确和zero-PKL均通过。scDFM保留布局的激活卸载GPU三更新对照通过：两组各1065张量最大绝对差0，原rtol1e-6/atol1e-7未变。真实官方batch48十步fit已完成，300-control预检收据尚待核对；这不是正式训练或科学结果。

官方新attempt2 `official-recovery-9607981-20261013-r2` controller2818852已运行；先scDFM预检/3988更新及last评估，再GEARS四轮及last评估。依赖native队列 `native-ffn-five-9607981-20261013-r2` controller2820803仅CPU等待两组实际18指标终态，不占GPU锁。然后DP→FN→FG→FB→PO十步及评估容量门，PREFLIGHT_COMPLETE后须主会话验收并生成MAIN_ACCEPTANCE才训练，不能由监督绕过。BR/Scouter不重跑。旧missing-parent失败和CLI/Python启动失败原样保留。机器资源门仅新增用户刚授权的UID1003 TotalVI s2脚本，上限不变。精确队列SHA/数值证据/阶段见evidence/official-ffn-recovery-20261013/launch-9607981-r2.json；长监督ACK/定时激活待单独收据，当前仍主owner。
