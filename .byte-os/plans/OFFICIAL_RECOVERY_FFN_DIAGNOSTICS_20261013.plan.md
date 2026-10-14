# 2026-10-14 用户授权更新：先运行五组，scDFM评估延期

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
