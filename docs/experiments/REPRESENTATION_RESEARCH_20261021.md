# GraD-Pert 表征与蒸馏：四轮自适应研究协议

日期：2026-10-21。主会话负责设计、实现、验收和分析；长时间训练交给既有监督会话。用户授权四轮逐轮设计的迭代，每个候选先训练两轮。当前只定义第一轮，不预先填满四轮队列。

## 研究问题与边界

当前问题是 Response 表示的差异集中、跨输入区分较弱，以及蒸馏概率接近均匀。它们是待解释的现象，尚不能统称整个模型坍塌。目标是保留任务相关变化，并使这些变化能支持扰动响应预测。

不将 DINOv2 的秩、方差或余弦设为细胞模型的达标线。视觉类别、扰动条件、图像 patch、基因 token 和预训练预算不同。当前有效秩使用中心化协方差特征值的熵，不混称使用奇异值权重的 RankMe。

交叉熵必须拆成 Teacher 熵与 KL：

$$CE(q_T,p_S)=H(q_T)+KL(q_T\Vert p_S).$$

目标变尖锐造成的 CE 降低，不等于 Student 学到更多输入信息。秩提高也可能只是增加幅度很小的差异、保留基因身份或放大 batch；需同时检查幅度、分层配对与响应效用。

## 第一轮为何选择注入残差

已有冻结中等诊断提示，B0 单细胞内基因 token 有效秩由 Cell 输出约29.09降至第一层条件注入后9.26、self后6.33、cross后3.66，最后约2.02。该观察定位了收缩位置，但不证明单一模块是原因。B0-DP虽然提高部分混合样本统计量，任务结果未同步改善；继续叠加低温、高EMA、换头不利于区分原因。

现有 Response 每层将状态替换为条件融合 MLP 输出。第一轮检验：保留输入状态的恒等路径，能否减少反复融合造成的差异损失，并保留有用响应。该机制是本项目的可证伪假设，不声称文献已证明有 mHC 的 KDA 会坍塌。

|实验|条件注入|其余配置|
|---|---|---|
|RS-B0|$X'=F_l([X;e_p])$|冻结当前 B0，重新从头训练2轮|
|RS-RI|$X'=X+F_l([X;e_p])$|只开启 `response_injection_residual`|

残差覆盖各 Response 层、四条 stream 和 CLS；Student/Teacher同构。无新增参数、门、归一化或缩放，不改 self/cross/FFN。必须分别记录 MLP 分支和实际注入状态，避免把二者混为同一取样点。

共同配置：Jurkat cap40、seed1、global batch320（80×双卡×累积2），两轮，无逐轮验证、无best选择。保留B0的GELU、非负预测、共享GenePT映射、SSL1 node=0及其余损失/优化器/视图设置。LR和EMA保持相同端点与相对调度，压缩到两轮；历史四轮实验的epoch2不是该对照。两组应有相同参数初始化及数据/RNG规则。

自包含配置及完整配置哈希：`configs/v2/representation_round1_20261021/manifest.json`。源基线为已发布干净commit `1cac75157a08405e980e858ce0f7919534eec5ca`。正式运行使用最终发布的新commit、独立run ID与新服务器checkout，不修改活动P1源码或结果。

## 共同诊断与任务效用

1. 保留原中等清单、样本轴、哈希和统计定义，供历史描述性并列；只用同一两轮对照作首轮干预比较。输出训练分项、梯度诊断、Teacher分布、主干谱/秩/方差及已有视图分析。
2. 另封存训练侧交叉面板，目标32个条件×4个不同control。同一context内使用相同control与同一允许表达基因panel，不能跨cell line/type/batch拼接。实际有效数量和缺失组必须报告。它补充而不替换中等清单。
3. 分开同输入跨Global/Local、同扰动异control、同control异扰动、异control异扰动。静态SSL1图按扰动计，不将重复control算独立图样本；Control表示不要求区分其未输入的扰动。
4. 分别测条件注入输入/分支/输出、self、cross、FFN及最终归一化；四stream与平均后状态分开。报告有效秩、相对/绝对方差、均值/总能量，区分单细胞跨基因与同基因跨control的统计轴。
5. 原型比较使用真实温度与center，Student各视图对Teacher Global；排除同Global配对。匹配context、控制错配组成，报告正确和错误配对的CE/H/KL及差值、有效样本/条件数。不存在的SSL1 node不补零冒充测量。
6. 交叉面板是输入干预，复制的truth ID仅供组装元数据，绝不作为真实配对反事实标签打分。任务效用使用独立的固定验证split及既有control/inference协议，训练结束后评估一次。

四轮方案选择只使用train/validation。最终测试延后到设计锁定后，以原协议评估last；“研究阶段完成”与“最终测试科学完成”使用不同状态，不伪造缺失test结果。

## 保留、否定与继续研究

- 若注入后的幅度/差异更稳定，正确配对优势及响应验证也改善，则保留RI作为有依据的候选；仍需后续稳定性验证。
- 若只提高gene身份差异、秩或扰动可分性，而预测效用没有改善，结论是机制局部改变，研究目标未达成。
- 若改动未阻止后续self/cross收缩，或任务误差变差，应否定该修复的充分性，利用逐层结果重新定位下一轮。
- 两轮可能不足以收敛；无收益不等同任何预算下无效。不得根据一次seed的小幅波动宣称解决，不为凑结论变更统计定义。

每轮完成后先形成中文报告、全部图及聚合表、来源/配置/checkpoint哈希，再决定下一轮。保留原B0锚点和各轮父版本；已经确定无效的组合不自动带入下一轮。四轮后仍未解决则明确尚存问题，不宣称保证达到DINO水平。

## 运行与监督

当前P1仍按原scope完成fit、last18和中等分析；之后停止旧C1/O1/K2队列。首轮研究在独立源码发布和10步/恢复/推理容量检查通过后推进，资源规则保持原授权，训练由唯一监督会话跟踪。终态回到主会话独立核验后才进入分析与下一轮设计。

所有正式数据、权重、前向与指标生成只在服务器运行。拉取前审阅清单，仅回传小型聚合CSV/JSON、图、报告与收据；原始矩阵和checkpoint留在服务器。永久分析目录：`experiment-analysis/representation-research-20261021/`（Git忽略）。

## 文献依据与适用限制

- [Understanding Dimensional Collapse, ICLR 2022](https://arxiv.org/abs/2110.09348)：区分低维退化与完全相同输出；其线性InfoNCE理论不是KDA定理。
- [Attention is Not All You Need, ICML 2021](https://proceedings.mlr.press/v139/dong21a.html)：提示检测token同质化及残差作用；纯attention结论不能直接套到当前mHC+FFN。
- [RankMe, ICML 2023](https://proceedings.mlr.press/v202/garrido23a.html)：谱可辅助评估，但不能单靠rank跨方法排名；保持其奇异值定义与本项目定义的区别。
- [Understanding Self-supervised Learning Dynamics without Contrastive Pairs, ICML 2021](https://proceedings.mlr.press/v139/tian21a.html)：EMA与其他机制的联合作用值得检查，不提供当前两轮最佳EMA。
- [What Makes for Good Views for Contrastive Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/4c2e5eaae9152079b9e95845750bb9ab-Abstract.html)：视图需保留任务信息，增强越强并非越好。
- [VICReg, ICLR 2022](https://arxiv.org/abs/2105.04906)：幅度与冗余是不同问题；其expander、尺度、权重不能直接移植到256维归一化基因表示。

详细文献核查与历史结果证据保留在永久分析目录的 `literature_handoff.md`、`evidence_handoff.md`、`code_audit_handoff.md`，与本轮新结果分开。
