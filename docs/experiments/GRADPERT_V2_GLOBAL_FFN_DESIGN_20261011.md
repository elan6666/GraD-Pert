# 全局 FFN 消融讨论稿（2026-10-11）

状态：仅讨论，未修改模型、未启动实验、未调整活动队列/监督。用户确认4 epoch；基线为此前讨论的恢复后B0，GenePT使用GELU。GenePT规则不变，候选仅切换内部FFN；所有候选关闭截断。此次扩展设计包括FFN归一化方式和Linear bias。

## 已授权执行范围（覆盖下方较大讨论矩阵）

只运行BR→F1→N1，各从头4 epoch。撤回统一bias=False：BR按历史各角色的bias规则；F1/N1保留同一规则。关闭所有FFN截断。图普通残差、Cell/Response mHC4，不新增图mHC。GenePT frozen6506×2048，共享2048→4096→1024→256；BR/N1两隐藏激活GELU，F1首隐藏用gate/up4096、down1024后SiLU、最后linear256，图输出LayerNorm保持。F1的普通MLP角色保留历史隐藏宽度，不沿用N0统一4d扩宽；Transformer本来就是SwiGLU。N1只改变FFN输入Norm：wrapper现有Norm替换，表达特殊入口在1→256升维后加入RMSNorm，其他MLP输入一次RMSNorm；eps1e-6，不改变attention/mHC路由自身Norm或输出Norm。新fields只在restored profile允许，历史default架构身份不变。

双卡micro48×accum2/global192，eval cell batch恢复功能B0的128；冻结300-control抽样口径和三组18项Pearson保持。以功能B0配置8961c93f…为母本，保留原视图RNG方式（不带入N0独立视图RNG），数据/表达剔除/损失/teacher-center更新时点/优化器不变。训练无逐轮验证，仅第4轮last测试。固定4轮warmup+cosine；旧6轮曲线只作预算诊断，不作相同LR轨迹声明。spread/KoLeo权重0，condition/node/DINO/iBOT各1，四头16384。

旧统一MLP五组队列已经由监督会话身份匹配停止；N0完整终态保留，CG1 epoch1 step199/250中断，U24/MR1/P1未启动。原source/run/config/checkpoint不修改。详见canonical stop receipt shared-m48-bf42185-stop-20261011T0823Z.json。新运行使用新源码/配置/ID。

CPU合成形状统计（非GPU容量证据）：
- BR Student trainable: 46854190; frozen raw prior: 13324288.
- F1 Student trainable: 57878062; frozen raw prior: 13324288.
- N1 Student trainable: 46858542; frozen raw prior: 13324288.

## 历史讨论：固定条件

拟以功能B0骨架为基础，保留当前双卡micro48、accumulation2、global192及共享可学习GenePT方式。GenePT冻结原始2048维先验，所有基因共用映射，无独立可训练基因表；用户此前指定的2048→4096→1024→256尺寸和输出LayerNorm保持。第一、第二隐藏表示均有激活，输出256维保持线性投影后归一化。

全局切换范围：图4层、Cell3层、Response3层FFN、表达编码非线性部分、逐层扰动融合、预测隐藏网络、四个蒸馏头的隐藏网络，以及GenePT映射。Teacher与Student同构切换。Q/K/V、MLA压缩、最终标量预测、prototype输出不添加激活；KDA门控、mHC路由与任务本身所需Sigmoid等不是FFN消融目标。

恢复功能B0各模块已有的Norm位置/类型/eps、bias规则和Dropout位置/概率；在候选之间固定它们。统一的是FFN形式，并不把所有模块强行改成相同尺寸或额外加入pre-RMSNorm。只有GenePT保持新共享映射和上述新尺寸。保持图合法邻域/来源门、随机顺序、单向KDA写入/末态读取、mHC、视图/损失/Teacher-center更新、数据划分/表达剔除和300-control评估不变。

## 建议矩阵

基线命名BR（restored B0），不是全局GELU的F0：图/Cell/Response FFN沿用B0的SwiGLU但关闭截断；表达、扰动融合、预测隐藏层、蒸馏头以及新的共享GenePT映射采用GELU。BR恢复各角色的历史Norm、bias、dropout，并保留当前batch和新GenePT映射。此前F0只表示全局GELU候选，不是用户已确认的基线。

| 编号 | FFN类型 | 非线性隐藏变换 |
|---|---|---|
| F0 | 普通GELU | GELU(Wx+b) |
| F1 | 普通SiLU | SiLU(Wx+b) |
| F2 | 普通ReLU | ReLU(Wx+b) |
| F3 | GEGLU | GELU(Wg x+bg) ⊙ (Wu x+bu) |
| F4 | SwiGLU | SiLU(Wg x+bg) ⊙ (Wu x+bu) |

两层FFN在上述隐藏变换后使用线性输出。GenePT为三个投影阶段：第一隐藏变换2048→4096采用本行类型，4096→1024后再用该类型对应的GELU/SiLU/ReLU激活，1024→256保持线性后LayerNorm。GEGLU/SwiGLU的第二隐藏激活分别为GELU/SiLU；不额外引入第二套门控，确保与此前讨论的4096 gate/up→1024 down→256 output布局一致。

首批固定各角色隐藏宽度，普通三组内参数一致，门控两组内参数一致。门控与普通组之间多一套投影，属于结构/容量对比，不能把收益单独归因于激活。报告实际参数/MAC/吞吐；参数匹配对照留作后续，不为本轮公平性默默更改用户指定GenePT尺寸。标准二投影FFN与三投影GLU的匹配公式（忽略bias）为 m_g=m_p*(d_in+d_out)/(2*d_in+d_out)；只有d_in=d_out时是2/3。

原始机制来源：[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。这是既有机制的任务适配，不是新机制贡献。

## 扩展：Norm、bias与残差（讨论建议，未定执行列表）

所有组固定BR的FFN类型、尺寸及其余训练协议，除非本组专门切换激活/门控。不同轴先独立比较，不做完整笛卡尔积。

| 编号 | 相对BR改动 | 范围/解释 |
|---|---|---|
| NR | 统一FFN前RMSNorm | 全部FFN/MLP输入各一次归一化；标量表达先升维。Transformer由wrapper提供的FFN norm直接替换，不叠加第二次 |
| NL | 统一FFN前LayerNorm | 与NR相同位置和eps/scale设定，LayerNorm先不使用额外beta，以隔离归一化统计差异 |
| NN | 删除FFN输入Norm | 仅移除本轴的FFN输入归一化；注意力/路由内部Norm、既定输出Norm与L2投影不变 |
| BA | FFN Linear全部bias=True | 包括GenePT/表达/融合/预测隐藏/蒸馏隐藏网络；不改注意力投影、mHC路由、KDA门、标量/原型最后输出、Norm affine |
| BN | FFN Linear全部bias=False | 同BA边界；BR的混合bias另作锚点 |
| RG | 图也使用4路mHC | 图前三层KDA及第四层MLA的attention/FFN残差均切换；Cell/Response不变 |

NR与NL的类型对照使用相同位置、eps=1e-6与可训练scale；NL的beta先关闭。它们相对BR也改变了原本没有pre-norm的MLP覆盖面，因此BR→NR不能单独归因于RMS vs LayerNorm；NR↔NL才是统计方式的更干净对照。Norm bias/affine独立于Linear bias，不用同一配置开关混改。

补充候选：统一post-FFN Norm、pre+post sandwich、Norm affine关闭、LayerNorm beta开启；需要各自明确输入/输出维度及避免与已有GenePT输出LayerNorm重复。先不把这些扩展成全组合。

图层当前为LayerNorm+普通attention/FFN残差，而Cell/Response为4路mHC；统一mHC不是恢复B0或FFN等价改动。建议保留BR作为普通图残差锚点，RG专门检验图mHC。图streams按基因ID存储，路由只混合同一基因的streams；KDA仍按合法邻域独立建状态/只输出目标query，MLA不新增邻域边。不把GenePT等无残差映射强行包成mHC。Teacher同构，图多路表示层间不提前折叠，只有既定输出边界读出256维。需要重新确认图target分块下的邻域/梯度/显存语义，不宣称mHC必然改善浅层图模型。

机制来源：[RMSNorm](https://proceedings.neurips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)、[mHC](https://arxiv.org/abs/2512.24880)。这两项属于不同机制轴。

## 已确认轮数：4 epoch 初筛

只读提取11个历史运行的fit/history.json；原始run ID、source/config身份与history SHA256存于[loss-summary.json](../../.byte-os/evidence/ffn-design-history-20261011/loss-summary.json)，逐轮小表见[epoch-losses.csv](../../.byte-os/evidence/ffn-design-history-20261011/epoch-losses.csv)。没有拉取科学大产物或重算GPU指标。

| epoch | 功能B0 prediction | N0 prediction | 功能B0 joint | N0 joint |
|---|---:|---:|---:|---:|
|1|0.09949|0.10173|36.241|33.729|
|2|0.05918|0.10160|11.844|9.315|
|3|0.05685|0.10143|2.701|4.280|
|4|0.05573|0.08201|1.152|2.385|
|5|0.05509|0.05709|0.850|1.809|
|6|0.05489|0.05591|0.776|1.642|

多数功能/未见基因组在第3轮的joint已有差异，但K1/K2/B0的prediction差距很小；N0前3轮prediction停滞、第4轮才下降，第5轮明显追近。第2轮N0 joint低于B0却prediction更差。因此4轮是降低误判迟发收敛的预算折中，不能称已证明4轮稳定区分全部FFN或最终模型效果。

拟所有新候选从头4轮、相同cap40/seed/batch/loss与统一4轮warmup+cosine；不计算逐轮验证/预测评估，保留逐轮训练分项loss与gradient norm，结束后评估第4轮last。报告三个Pearson的全部/见过/未见表达组及all/统一DEG、有效条件数，测试不选超参数。小幅差异须后续多seed，不以训练joint一个标量判胜负。

历史为6轮日程，只用于预算诊断；新4轮cosine会重新安排LR，绝不声称等价于历史6轮的前4轮。新的基线需重新训练，旧clipped B0/N0仅历史参考。容量按用户已确认10步预检，不把10步通过写成长期稳定或科学结果。

## Coexistence preflight input binding repair

2026-10-11 source0916536c6132b4a44cb04de5e171e6cc825cf41f started torchrun beside the two pinned ScButterfly jobs (GPU0:720MiB; GPU1:718MiB). F1 failed at0/10 optimizerupdates: the config generator inherited the historical PCA256 file from restored B0 while enabling raw learned GenePT projection. Runtime rejected it before training; no capacity or speed conclusion is possible. Initial receipt inference_exercised=true records intent, not completed inference. Actual checkpointrestore and300-control inference did not pass. Failed queue v2-ffn-coexist-0916536-20261011T095120Z is retained.

Corrected allthree config artifactbindings to the existing original2048prior (17730rows, SHA34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318; live servermetadata verified). Frozen graphgene-ID selection remains unchanged. This is a configbug fix to the already confirmed GenePT design; no model or trainingprotocol change. Testnow asserts the correctpath/hash for every arm. Freshsource/queuepreflightrequired.
