# 消融共同分析模板 v4：同清单、分组、冻结权重与可审计交付

更新于2026-10-15。当前验收范围是 medium_cap40_stratified，不是全量训练细胞。
本模板覆盖旧v3仍残留的47836行／64梯度条件等全量条目，历史证据不重写。

## 固定口径

- 1335个训练条件全覆盖，每条件最多4个按context分层的真实细胞记录，实际5317行；不复制singleton。
- 固定128个深度条件／506行；32个梯度条件／125行，均为前述清单子集。
- context=cell_type::batch；cell_line单独保留。control_treatment始终control，paired_perturbation另列。
- 固定truth/control/query/view/mask/edge配方和行顺序，绑定完整manifest SHA256，不能仅以相同seed替代。
- 单细胞query1000个训练允许表达基因；本次训练侧诊断不偷看225个剔除表达基因数值。图身份表示可读，与表达标签可见性分开。
- 深度基因秩用每细胞最多128个确定性哈希基因，秩上限127；多view和多token不能增加独立细胞N。
- 每个条件内部先平均，再条件等权；row_mean同时保留但只是本平衡抽样行均值。深度分层子集不宣称总体无偏。
- last.pt只读，不做optimizer／EMA／center更新；所有state_before/state_after与COMPLETE证据必须一致。

## 每个实验按同一顺序报告

1. 单变量与原冻结配置：实际层／映射／bias／原型规则的作用范围，不能按简称猜配置。
2. 训练SHA、分析SHA、完整config／checkpoint／manifest SHA、epoch与角色。跨分析代码版本单列；说明是否只改调度／观测，不给旧权重贴最新HEAD。
3. 数据范围、cell_line/cell_type/batch/condition/gene/view各轴与实际N；超过4context的条件不能声称全部batch覆盖。
4. 原历史训练每轮全部分项loss、LR、EMA、时间、已采集梯度；缺项写未采集，不能从last恢复过去。
5. 冻结last原型：四头×Student/Teacher×view，H_sample、H_marginal、JS、N_eff、top1、argmax、center；明确condition/node/cell/masked-token单位。
6. 实际蒸馏配对CE/H_pair/KL，按有效mask过滤；只有相同配对、温度与权重才能使用CE=H+KL，不用全局H相减。
7. 表征：GenePT→图→Cell→Response→投影输入/瓶颈。混合秩、单细胞基因秩、同基因跨control、同条件CLS、固定query+context的条件间均值、跨view各自分表。mHC明确stream均值。
8. 输出MSE与复制control对照、负值/零值/极值、变化RMS／斜率／方向、大误差贡献；随机control配对不是同细胞真实反事实。训练允许查询诊断不替代未见表达测试。
9. 梯度graph.adapter/graph.layers/graph的预测/SSL1/SSL2加权norm、cosine，实际有效条件N；只读探针，不推断整段训练的冲突比例。
10. 真实耗时、启动与报告成本、峰值allocated显存、失败/重做成本。双卡一组一卡时用实际重叠和makespan描述，不以瞬时利用率宣称加速倍数。
11. 正式测试完整18项：3 Pearson×全基因/统一DEG×all/seen/unseen，原last测试receipt、有效/总条件数。不要重跑测试来选参。
12. 结论分已观测、机制解释、未验证假设、限制；一seed／一细胞系不声称稳定泛化或因果机制。
13. 所有产生的图都嵌入，逐图标轴、单位、独立N、分组、读法。全部CSV/JSON列路径、粒度、大小、SHA及本地/server位置。大表和checkpoint留server。

## 统计定义

C=(1/N)Σ(x−μ)(x−μ)^T；p_k=λ_k/Σλ；effective_rank=exp(−Σp_k log p_k)。
relative_variance=tr(C)/(tr(C)+||μ||²)。常量矩阵秩0；不足两行标undefined，不能补0。
JS=H(mean p)−mean H(p)，使用累计概率重算，不平均各chunk的JS。
输出和配对差值区间以扰动条件bootstrap1000次；不代表多seed方差。
条件内CLS秩必须去重control、固定condition/context并报告真实N；N≤4本身限制秩，不能当作坍塌。

## 图号与数据交付

每臂01–04为历史小probe；05覆盖、06原型、07表征、08输出、09梯度为当前medium补测。
full_prototypes/full_representations是历史文件名，标题和正文必须指明medium。
总比较C01正式18测试、C02 Teacher H/JS、C03混合表征秩、C04同条件输出MSE差。
五臂共45图＋4比较图=49图；密集图可放大，但精确数值必须能对应CSV。
原报告与原收据不覆盖；本地修复相对链接另存派生副本，并核验所有49图片链接。
完整图数据索引包括report声明的输出与COMPLETE声明的原始分析文件；逐细胞大表只列服务器路径。

## 调度与终态

同一精确父进程管理每卡一个分析worker，启动和每条件检查均核验父UID/PID/start/cmd；不弱化外部资源规则。
短门通过＋实际并行进度＋独立main验收后交唯一监督；终态回交main，暂停等待监控和return fallback，读回状态。
主独立核验终态与逐文件SHA，审核图片／报告，再完成交付；消息通知或进程消失本身不算完成。

## 新FG+PO／node0基线的补充规则（2026-10-15）

后续B0、VH、S12-L4、CG1、U24、MR1、P1、C1、O1、K2-fixed-prefix仍使用上述中等范围。
从已冻结medium manifest深拷贝实际truth/control/context/query、deep/gradient行与view seed，
以`diagnostic_population_sha256`绑定这些轴，不重新分层采样。
每组按实际训练源码和完整配置重新封存view/mask/edge配方；VH/S12的配方差异是已声明的实验变量，
不能宣称其完整view manifest与B0逐字节相同。原DP等五臂完整清单和报告不覆盖。

新B0的SSL1 node权重为0；不为了填满表而额外调用该头，CE、熵与瓶颈缺失标为未采集，不能填0。
MR1的masked-response辅助forward单独标记，不算作SSL2 Global或Local，跨view表仅比较实际蒸馏视图。
CG1只共享前三层图前缀：全图身份轴可报告共享H3；不同control的第四层H4必须在对应主路径／视图分开，
不能将H3混入其他组的H4比较。相同control的遮蔽表达仍不能从图条件摘要绕过。

每个新臂训练并完成last18正式测试后，自动保存旧小probe报告、重封中等清单、执行本臂reference/fast三条件
1e-8正确性门，再执行fast中等诊断与共同中文报告。新pipeline验收包含科学输出／状态／checkpoint／源码／
配置和全部图表hash；启动、seal、probe、诊断和绘图分别计时。父训练控制器释放GPU锁，分析worker仍获取
正常GPU0锁，避免父子重复持同一锁；分析完成才继续下一组。双卡训练／评估协议不变。

本次实际性能机制的完整中等配对验收是新队列准备的依赖，短探针通过不允许直接启动正式新消融。
# Mandatory Global/Local condition comparisons (2026-10-17)

All future medium ablation reports include the same additional paired-view module,
without changing the frozen basic/deep/gradient population. Record the protocol
`condition-stratified-global-local-1`, model/config/checkpoint/source identities,
all scalar CSV/JSON tables and the paired-view figure with reading guidance.

- SSL1 unit: graph perturbation-condition vector before projection, one row per
  condition for a static graph; control-conditioned graphs retain control IDs.
- SSL2 unit: response CLS before projection, deduplicated by control+condition
  within cell line/type/context. Views never count as independent cells.
- Compare GG/GL/LL for same sample, same condition/different control, different
  condition/different control, and different condition/same control when available.
  Match cell line/type/context; separate batch/cell-line effects, and explicitly
  mark unavailable groups (currently only Jurkat) rather than fabricating values.
- Report raw cosine, cosine centered using a common Global centroid, MSE between
  features, pair means versus condition/condition-pair equal means, valid counts,
  independent rows/controls/conditions, per-view centered rank/relative variance,
  within-condition and between-condition-mean ranks. No independent-pair CI claim.
- For disabled SSL objectives, Student view probes are diagnostic-only; no Teacher
  target/center update or artificial SSL CE. Per-cell matrices remain server-only.
- PT1 additionally records phase/optimizer reset/LR, preserves its pretrain
  checkpoint and marks prediction quantities unavailable before finetuning.
