# BR400四组与best/last评估

## 已授权的新协议

仅BR（用户称BR0）、F1、N1、GM1。全部从头4epoch、seed1、Jurkat cap40；global400=每卡micro100×双卡×累积2。每轮joint-only验证，按最低验证joint loss保存best，第4轮last；结束后分别测试best与last的18项Pearson（全部/见过表达/未见表达×TxPert/TriShift/Systema×全基因/统一DEG）。即使best和last内容相同也保留两个role收据。冻结300-control、表达剔除、data/split/DEG与既有role-specific norm/bias/activation不变；Transformer FFN无截断，图来源门、合法邻域、KDA单向末态读出、逐层扰动注入/control cross、Cell/Response mHC等依原BR。

BR旧400短测只证明旧BR配置10更新范围内可用；新增加joint验证且F1/GM1结构不同，需要每组新配置精确10更新检查。最高batch测量与正式效果结果分别报告。原BR四轮last结果/旧192各组ID/原源码不改；待派发192 continuation已被本协议覆盖，未启动。

| arm | 相对新BR的单变量改动 |
|---|---|
| BR | 恢复功能B0角色配置；冻结原GenePT2048，共享2048→4096→1024→256，GELU隐藏+LayerNorm输出 |
| F1 | 普通MLP隐藏切SwiGLU；既定隐藏维度/Norm/bias不变 |
| N1 | FFN输入改RMSNorm；其他保持BR |
| GM1 | 图四层attention/FFN残差加入mHC4；Cell/Response已mHC4不变 |

完整双蒸馏：预测/condition/node/DINO/iBOT各1，spread/KoLeo0，四heads16384prototype。4轮warmup+cosine依新400实际steps/epoch配置；batch改变会改变optimizer更新数/EMA-center次数和训练轨迹，不宣称与原192等价。

## 容量与资源

每GPU物理free>=2048MiB，aggregate本任务<=85%；仅允许UID1003精确run_scbutterfly_archived_predictions.py每GPU1个<=1GiB共用，不调整资源授权。所有nativeCUDA必须PYTORCH_ALLOC_CONF=expandable_segments:True。先GM1/F1的完整10更新、非零LR、checkpoint恢复及val300-control实际推理，再其余；只在四组和joint验证均通过后主验收启动。失败保留独立ID/日志/收据，不能静默降400/跳组或覆盖旧源。

评估batch目标300：每条件固定只有300个control，超过300没有额外计算并发。四组以val control/all5000gene推理检查最大300及固定较小chunk的有限精度差；不能用test分数选evalbatch。初始容量探针不冒称更新后模型验证，10步后的实测推理另行要求。共同evalbatch应在训练前确定并写入自包含config，保证独立PT评估配置hash完全匹配。若300失败，明确保存并以容量证据选择更小共同batch，重新发布配置后才能正式启动。

训练400与最大eval并发会竞争显存，不预设能共存。此family使用双卡fit→双卡best/last测试→下一fit顺序，防止旧25%cap/大eval互相挤占。dual评估按条件分片，300是每个worker的control细胞chunk，不是训练batch。首轮与每次切换均报告实际epoch/steps/phase、源和配置hash；进程消失不等于完成。

## 实施与证据

预修改版本832a9262af73b9a3a56e4cb92fdf413572882d3a已发布。新实现、CPU检查、评估容量/训练容量/正式运行和monitor尚待新收据补充。旧m100最高十步400+16.376cells/s/minfree5004/5024MiB仍历史，不作为本family已通过。计划 .byte-os/plans/GRADPERT_V2_BR400_BEST_LAST_20261012.plan.md。

## Local verification

Both new group/capacity suites together57passed1TorchJITdeprecationwarning10.89s, exit0, hiddenCUDA; Ruff/format/diff passed. Real producer/validator interoperation, cross-arm6dataSHA re-sealing rejection, manual main barrier, true native best+last samecheckpoint/incorrectbest/missingmetric/PKL cases covered. Boundedjoint11protectedstates and eval128↔300 fixedatol/rtol2e-6. Not GPU evidence yet. Small local receipt .byte-os/evidence/br400-best-last-20261012/local-validation.json.

## Source2898882 first GPU gate failed

ServerCPU checks exit0; queuev2-br400-2898882-20261012T043957Z exactSHA6dc193717d020b76c9da67f5c546acdd9eb28cbb982cb3c87356b867770f487d. GM1 native10/10/restore/nonzeroLR/finitegrad/boundedjoint100 andGPU0/GPU1 fresheval300 vs128passed. F1 ownGPU28108MiB percard exceeded85%budget27715.95MiB, free3994MiB; guard terminatedourworkers, noCUDA OOM evidence. F1receiptstep0/running isinterrupted—notpass. ControllerfailureSHA3811c1591359db439712664a6417b3fd22b73c05f0ba5514fd2665c737489ed7; allownedworkersgone, noMAIN_ACCEPTANCE/noformalfit. BR/N1notprobed. EquivalentGenePT mapping activationrecompute repair proposedwithinapproved400/resourcebounds; noresult/effectimprovementclaim. Oldroot immutable.

## 等价显存修复：GenePT 映射激活重计算

F1 的原始 GenePT 映射有 gate/up 双支路，训练反向会保存宽度4096的激活。新版本仅对 Student 在有梯度的训练映射使用非重入 activation checkpoint：前向仍计算相同映射，反向按需重算内部激活；Teacher、评估与无梯度路径直接计算。参数与映射公式、Norm/bias、随机顺序、loss及更新时点均不变。四组共同启用以避免比较混入不同内存策略，旧配置默认关闭、旧架构payload不增加字段。

四组均验证3次非零学习率完整更新，包括GM1图mHC的嵌套重计算；输出、梯度、optimizer、Teacher/EMA、center/pending、Torch及NumPy随机状态精确对齐。CPU BF16 autocast映射和反向也对齐。CPU float32、128基因、真实F1映射宽度的去重非持久激活保存量为9,569,280→0字节；参数与冻结先验存储已排除。这仅说明重计算机制有效，不代表CUDA峰值或吞吐收益。

新配置目录 `configs/v2/br400_best_last_checkpoint_20261012`；新预检顺序F1→GM1→BR→N1，正式顺序仍BR→F1→N1→GM1。须在新源码重新通过全部双卡10更新和300-control容量/一致性门，主验收后才能启动四轮训练。资源85%与每卡2GiB余量保持原规则，失败不自动降batch400。

## TotalVI 小任务共用授权

新6e2b93c队列在CUDA子任务启动前发现UID1003的 `run_totalvi_source_validation_development.py`，原ScButterfly名单不允许，保护门安全终止且旧运行保留。用户随后明确允许此程序按相同上限共用。新版schema3政策仅增加该精确程序，必须绑定新的授权reference；旧ScButterfly政策兼容。两种程序合计仍每卡最多一个外部任务、占用≤1GiB、物理free≥2GiB、我们aggregate≤85%，不扩大其他UID/程序授权。127项相关CPU测试通过；新正式工作仍等待四组精确GPU预检。
