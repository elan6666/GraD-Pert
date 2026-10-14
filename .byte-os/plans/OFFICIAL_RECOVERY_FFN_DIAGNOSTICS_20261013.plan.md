# 2026-10-14 用户授权更新：先运行五组，scDFM评估延期

## 2026-10-14 FB数值门后的工程接续

3ee8ef3新队列DP/FN/FG全部工程门、FB十步通过，FB GPU0 eval300与reference128超固定2e-6容差，终态failed，无正式fit。失败保留并回交主，两个monitor暂停。不得把误差阈值放宽或覆盖旧队列。

只测试执行分块256→192→128，取通过的最高候选；每次均保留同一300control清单、5000基因轴、seed、query1000、指标/DEG/归约，训练仍四轮/global400/m100×双卡×accum2/无验证/last。增加明确的eval-only工程入口，只运行所选已授权FFN arm两卡容量门，不跑训练/不写科学COMPLETE或MAIN_ACCEPTANCE。工程输出明确标为EVAL_CAPACITY_COMPLETE。失败收据现在保存实际abs/relative误差统计，原阈值不变。先在FB检查候选，再所有五臂同一batch重新通过完整门和主验收，才正式派发；128是既有参考执行路径，不将同batch一致宣称为跨batch数值验证。所有候选config自包含/版本化，旧300家族默认和历史不改。

GEARS用户请求的constant-vector审计已完成，详见docs/experiments/GEARS_UNSEEN_ZERO_DELTA_AUDIT_20261014.md：官方输出没有硬置零，未监督基因特异最终读出经官方coupled Adam weight decay退化到~1e-41，三个固定条件的预测均值Δ精确零；checkpoint保持。没有修改GEARS优化器、架构或旧结果。

此段覆盖下文“必须等待两官方完整测试终态才派发native”的旧顺序要求。用户明确要求先推进之前五组，scDFM评估以后再优化，保留当前pt。五组仍为DP→FN→FG→FB→PO，各从头四轮、global400=m100×2卡×accum2、不逐轮验证、只测last的18个Pearson及原训练过程诊断；BR/Scouter/scDFM不重跑。

只调整native调度依赖，不改变native模型、五组配置/manifest、数据/表达剔除、损失/优化器或评估协议。运行时显式 completion_policy=scdfm_fit_gears_terminal_evaluation_deferred_v1 和 user:2026-10-14-native-five-scdfm-eval-deferred：要求原scDFM RUN/fit receipt/source/config/runtime完整身份、3988/3988和四轮预算完成、last.pt路径/SHA未变，以及GEARS真实终态/18指标/zeroPKL。scDFM测试仍deferred_by_user，不写COMPLETE、不把48/592部分结果当成完成。原strict默认仍要求两官方完整评估，未知policy/缺授权/变更checkpoint/未完成fit/GEARS不齐必须拦住。此policy和proof也绑定在PREFLIGHT_COMPLETE/MAIN_ACCEPTANCE中。

工作步骤：验证scDFM checkpoint与GEARS终态及live资源；定向测试/发布调度源码；新immutable server/source/runtime/queue；仅精确退休旧CPU waiting控制器，旧run/日志/权重保留；五组各10update+restore及双卡eval容量检查，主核对全部门后写精确MAIN_ACCEPTANCE；自动DP→FN→FG→FB→PO四轮last评估，交唯一指定监督20min，失败交主，旧monitor闲置保持暂停。短预检由Luna只读监督，主不重复查询。

2026-10-14初始资源：每卡UID1003 run_totalvi_source_validation_development.py约800MiB，在既有白名单内。继续原90%/每卡外部≤1GiB/max1/余量≥2GiB/allocator expandable_segments:True；不放宽。修改前clean/pushed baseline1c9830bd01c89f58569b1e6791fd724d4189b0f7已核对。

---

# Official recovery then five FFN experiments

Owner: codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b. Published clean base: 141989eeb18719aad2d790012eb6fc7366c31e9b. No BR or Scouter rerun.

Outcome: finish official scDFM and GEARS (four epochs, existing expression holdout and frozen 300-control/18-metric last evaluation), then run five native four-epoch last-only arms FB/FG/FN/DP/PO using the completed BR400 anchor. Preserve failed receipts and immutable releases. No native arm dispatch before both official baseline terminal acceptances.

Current repair: scDFM batch48 FP32 OOM; CPU saved-activation offload GPU multistep parity failed. Diagnose ordinary repeatability and layout changes under original tolerance before adopting a repair. No approximation of MMD or model rewrite.

Native specifications: FB disables Linear biases in all FFN/role MLPs, retains normalization beta and mechanism biases; FG uses GELU in all role FFNs, transformer hidden1536 matches original SwiGLU hidden1024 weights; FN vector-role pre-LayerNorm eps1e-6 with FP32 statistics, scalar expression lifted first, existing FFN norm replaced not doubled; DP four prototype layers weight_norm(g initialized1), first epoch LR0 (optimizer moments continue), original widths/MLP/init conventions explicitly recorded; PO score head Softplus directly predicts expression, delta=prediction-control, all response inference paths consistent. No silent default changes.

Diagnostics: fixed training-only probes at initialization/epochs; prototype entropy/JS/KL and pre/post projection variance/rank; gene/control/response token variance, delta/negative prediction/tail metrics; fixed-seed order and control/perturbation interventions; shared graph and basal loss gradient norms/cosine at bounded cadence. No excluded expression enters training probes, no optimizer/EMA/center/RNG mutation. Aggregate files only, budget and elapsed time recorded.

Disjoint delegation: native implementation agent owns config/v2.py, modeling/v2/*, optimizer/runtime update hooks and its new tests; diagnostics agent owns NEW process-diagnostic module and NEW tests only, does not wire runtime/lifecycle. Main owns official adapters, config generators/queues, runtime diagnostic integration, lifecycle, documentation/publication/launch. Agents do not commit or launch jobs.

Acceptance: scoped local regression; baseline parity with disabled options; nonzero-LR multistep/checkpoint/EMA/center/RNG tests; official real GPU parity and ten updates/restore/300-control gate; clean local/GitHub/server matching release; actual run launch; unique supervisor ACK and active monitor/return wake. Long tasks handed to designated supervisor 01a0df0b-4142-7df1-86c0-d959471d80a1, bounded probes to Luna without duplicate polling. No Goal requested.
