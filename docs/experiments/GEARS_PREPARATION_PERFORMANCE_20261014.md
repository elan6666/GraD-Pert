# GEARS 十步预检：CPU准备修复

训练配置保持官方 GEARS `f374e43e197b295016d80395d7a54ddb81cc6769`，Jurkat cap40、test靶点表达剔除、四轮、共同300-control/last18指标。真实GPU预检仍恰好10次优化器更新，不是128步。1781是数据准备时的扰动条件数。

## 瓶颈与机制

原科学源 `9607981fd7d44b3982574167132d06f739bc1b7e` 的适配器在每个target row重复验证整控制候选池、深复制control/target AnnData。新源 `7798605001bda2a03933b71e84cdfde7faf811c5` 缓存同context且候选顺序完全相同的只读控制池/target视图，仍逐row调用官方采样/构图，仍均匀有放回。同context候选tuple共享但不可变。

此外，准备输入的canonical float32 CSR共有174,706,361个非零值；原build_prepared转COO再转回CSR，反复Python栈停在该转换，5分钟仍未完成，另一未加栈的探针21分钟未形成收据。新路径直接映射CSR列索引：`indices_new = allowed[indices_old]`，不展开每个非零值的行坐标；严格递增allowed轴保持列排序。data/indptr独立复制；其他格式或含重复项的CSR保留原回退语义。表达身份、允许轴、行/候选顺序及数值不变。

原GEARS官方allowed-only DE/dropout排序已经写入H5AD，可复用但必须逐项核对缓存文件/原RUN/data/config/env/method SHA、obs/var及**完整sanitize X值**，不能仅信manifest或用非零状态推断允许轴。排序方法未改、未重算或引入其他先验。

## 验证与数据

- 本地130项bench/parallel/BR检查及75项数据/资源检查通过；服务器合并196项通过。四格式含重复CSR与原转换逐数组对照、输入不变、控制候选/官方graph/audit/RNG回归。
- 同源官方CPU门：10 updates、恢复预测逐位一致、zeroPKL。source7798605，config SHA `b7c2b23170bb5a16bb61e4199590977e8e7c8c81ad1c2e56a8d39ce4760b0c1c`。小收据 `.byte-os/evidence/gears-preparation-20261014/cpu/receipt.json`，SHA `f4d8985ac84bf0436ffc09869ee032c73487187ff24f1c27e41996b562213a61`。
- 真实prepare阶段：CanonicalTrainingData初始化0.804s、允许表达读取10.910s、build_prepared2.583s。矩阵101084×4775、CSR/float32/canonical。这是阶段对比，不是训练吞吐。
- 真实8细胞/16graphs对照exit0，完整input加载和缓存校验30.465s；原逐row构图10.674s，新缓存构图0.03738s，局部285.58倍。官方图、配对audit及NumPy/Torch RNG哈希完全相同。仅8细胞局部测量，不能外推整套训练加速比或长期效果。
- parity收据 `.byte-os/evidence/gears-preparation-20261014/parity/receipt.json`，SHA `d9911a4683424e3bc8136573e884a36d885eded857457901ae15aa3b035e5286`。原排序缓存SHA `4f67d273913f0dc9336abcdc65ad75ea8fa56fb12b6a8296ac5d6c11dbd094ea`。

## 失败证据与调度

保留edf19aa原21min探针及stage1中断收据、7798605 stage2未知进程消失的INCOMPLETE.json；stage2没有exit/最终parity，原因未知，不能归因于OOM或宣称通过。撤去外部faulthandler wrapper后直接发布的plain3探针exit0/parity通过。原并行队列随后核实已被外部GPU资源门保护终止GEARS/dispatcher，scDFM原worker仍在；保护未被放宽。修复的live-migration入口因真实terminal状态拒绝（另一次CLI因缺`.` PYTHONPATH失败），均未发送信号/覆盖旧run。

当前GPU1重新空闲，主以实际FAILURE SHA、旧controller/GEARS确实不存在、原scDFM PID/tick/cmd仍一致、原资源门通过为前提生成新的终态接续sealed schedule；不伪造running状态，不恢复旧GEARS ID。新schedule `official-parallel-gears-prep-7798605-20261014` SHA `a1624e75e72b1bcbc1e2e393da616f8fb8af8e93c2e7c9d9650149db08ce00fa`，mixed-source parent queue SHA `d88f7a8c2b6b935b9214bc86fb07b22dda2e2fe14f2b300ab259433868a417a3`。scDFM保持9607981及PID2823360，GEARS新源7798605/fresh run `gears-jurkat-seed1-20261013T172304Z-77a6c09777cb45b695d416e96dc063ec`。新runtime SHA `2ebb7c637db82e1ce8f1bd71842460447a2c0f0459ff809f121ba85012765571`，publication SHA `4b59164763f50491d83e5972f719e271dc8d86f05c9db0e09684b316f8e44b3f`。

native五组DP→FN→FG→FB→PO依赖已换到这个mixed-source parent；旧纯CPU等待器2820803精确退出，旧queue/收据保留。新native queue `native-ffn-five-gears-prep-7798605-20261014` SHA `dd02a982d4693ff3aa431d65e5175846231fdf8752ca28f0147f005e75e7f44d`，仍CPU等两官方实际18指标终态，再10步/恢复/评估容量预检，须主MAIN_ACCEPTANCE后才允许正式native四轮。尚无新的GEARS GPU预检、正式结果或FFN结果完成证据。

启动复核：GEARS完整条件构图1781/1781用5:49，仍需真实10update/checkpoint/common-inference收据；scDFM最新3000/3988。native首次新queue因CUDA_VISIBLE_DEVICES未显式为空被CPU gate拒绝，未进入CUDA，旧记录保留；新r2 queue SHA583496f99cba00325055d34ed16d61c08a8bde8393ba199d2377740e18728e72、PID2897522已核实空CUDA且waiting_official_baselines。此r2覆盖前文native r1活动表述；handoff-7798605-supervise.md记录完整身份和监督/返回规则。
