# Jurkat 小模型三组纯预测对照

主对照直接复用 Nadig Jurkat 的 `small_softmax_mhc_no_distill`，
run `nadig_jurkat-seed1-20261109T143945Z-08918456488f4d2fbc2a4f67a5d8409c`，
训练源码 `9061c632fad7d4c853e17322caca5c739c7066b0`。
三组都保持宽度 256、四流 mHC、纯表达 MSE、关闭所有蒸馏、Softplus、
基因身份和表达拼接投影、固定训练行和图拓扑、seed 1、两完整轮各 47,836 行、
1000 查询基因。冻结验证只用第 2 轮 last，按原参考、条件顺序及同序每条件 300
个对照，评估切块与原小模型一致。开启现有训练诊断和执行预设。

| 组 | 唯一变化 | 训练 batch 与预算 | 参数（可训练 / 总计） |
|---|---|---|---|
| `small_softmax_mhc_deep_no_distill` | Graph 4→6、Cell 3→6、Response 3；预测头 512→256→256→1，两个 GELU | global 320 / micro 80 / accum 4，两轮 300 更新 | 13,071,276 / 26,395,564 |
| `small_softmax_mhc_random_gene_no_distill` | 以每图节点一行的可训练 6506×256 标准正态表替换 GenePT 查表和映射 | 同上 | 9,181,010 / 9,181,010 |
| `small_softmax_mhc_max_batch_no_distill` | 原骨架和 GenePT，双卡 DDP batch 容量实验 | 由容量筛选冻结；global=2×micro，accum 1；每轮更新数为 ceil(47836 / global) | 9,943,890 / 23,268,178 |

加深组新隐藏矩阵走 Muon，标量输出层权重与偏置走 AdamW，原默认模型参数键、
架构载荷和路由保留。初始化顺序因新增层改变，结果归于加深组合。
随机表使用运行 seed 创建私有 CPU 生成器，采样 N(0,1) 后沿冻结图节点顺序查表，
经现有图邻域注意力参与预测并由 MSE 更新。训练和推理不加载 GenePT 或其收据。
删除旧映射的初始化会改变公共模块的随机轨迹；记录实测差异，不宣称共享权重逐项一致。
训练数据采样保持原 seed 与构造规则。

参考边界：[公开 TxPert Exphormer 实现](https://github.com/valence-labs/TxPert/blob/08d82eea86746b044cf7531f4ec8c5f60e1cb73f/gspp/models/pert_models/exphormer.py#L105-L110)
从默认可训练嵌入表读取扰动节点向量，再进入图层（同文件 L204–L217）。
GraD-Pert 只参考节点查表的初始化思路，使用原生 256 维 Graph、原基因轴、
独立 RNG 和自己的训练/推理实现。

容量筛选固定两张 RTX 5090，各卡 90% 进程显存上限、allocator、BF16、层重计算、
GenePT 映射重计算和初始化诊断。用同一 runner 的工程模式，每个候选独立进程，
单卡 microbatch 依次试 80、96、128、160、224、320、448、640，必要时在通过/更新 OOM 间二分到 4 的步长。
候选通过需完成三次满 batch 更新（含首步、优化器状态和有限梯度），保留耗时、峰值
allocated/reserved、命令与收据。仅实际 optimizer update 内的 CUDA OOM 可收紧
边界；资源准入失败、诊断失败、进程消失等不算 batch 容量证据。超过预注册上界未测，
或相邻 1–3 个未测时只称“最大已验证值”。无验证/测试分数参与选取。
获胜配置在发布源码内单独封存并重新通过标准 10 步、保存恢复和 300-control 推理，
之后才启动完整两轮。报告同时列出旧有效 batch 320、所选单卡 microbatch 与双卡 global、末批及实际
warmup/cosine 步数；更大单卡 microbatch 和更大有效 batch 是两个不同问题。两张卡都必须空闲并通过资源准入，不占用运行中的作业。

每组独立运行并留各自预检与启动回执，不能通过时明确排队或失败原因。
完成后在实验平台按正式 run ID 归档真实曲线、冻结表达验证和 CPU 训练诊断。
没有蒸馏模块的指标记作不适用。
