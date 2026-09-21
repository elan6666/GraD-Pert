# GraD-Pert v2：讨论记录与候选实验设计

更新：2026-09-21。状态：设计讨论，不是已实现模型或训练授权。
本文区分用户已确认方向、建议默认值、仍待核实事项。历史 v1/B2 的正式协议不被本文改写。


后续实施与实验路线见[分阶段计划](../../.byte-os/plans/GRADPERT_V2_IMPLEMENTATION_AND_EXPERIMENTS.plan.md)（当前供审阅，尚未执行）。

## 当前设计总图（用户于 2026-09-21 指定）

当前视觉基准为 [单画布总图 PNG](../../overview-design/2026-09-21-gradpert-v2-mechanisms/gradpert_v2_overall.png)，
对应 [可编辑 Draw.io](../../overview-design/2026-09-21-gradpert-v2-mechanisms/gradpert_v2_overall.drawio)。
上方为九组组件机制展开，下方为预测与双蒸馏总流程。后续设计图以这一版为基准；
十页分图保留作组件编辑来源，早期拥挤总图仅为历史版本。
图中星号仍表示待冻结细节，选定设计图不等于确认所有候选值或授权模型实现/训练。

## 0. 本次从服务器核实的父配置

父行 `r50n_t1_u1`，源码 SHA `53d02e17086ed6fac67920cd781130da9688b07c`。
训练 run manifest、启动收据与当前冻结配置的配置 SHA256 一致：
`6c56c73a12e054eef561522b80484b7c577a632f4f43a1ebd729af5d2479792f`。
来源为服务器
`/data/yilangliu/GraD-Pert/development/source-r50-new-53d02e1-v1/configs/r50/r50n_t1_u1/gradpert_b2/nadig_jurkat.yaml`。

| 参数 | 已核实 v1 父值 |
|---|---|
| epochs / batch / seed | 50 / 1024 / 1；无早停 |
| optimizer / LR / weight decay / scheduler | GLM5MuonSplit_v1 / .001 / 0 / none |
| prediction / condition / masked-node / spread 权重 | 1 / .8 / .4 / .1 |
| gene embedding / graph output / basal latent | 128 / 64 / 64 |
| graph layers / configured heads / configured head dim | 4 / 2 / 128；具体有效张量布局还需查主干实现 |
| projector hidden / bottleneck / prototypes | 2048 / 256 / 16384 |
| Global / Local | 2 / 4；RingInduced local预算为图节点数1/2 |
| EMA起点 / Student温度 / Teacher温度 / center momentum | .99 / .1 / .04 / .9 |
| graph / prediction dropout | .1 / .2 |
| batch condition上限 | 8个不同条件 |

以下全 token 模块是 **v2设计改造**：当前默认 d=256，d=128 用于宽度对照；
不是声称 v1 的64维扰动/细胞 latent 已经具有该宽度。

## 1. 已确认的方向

- 以用户指定的 v1 **50 epoch、batch1024、split Muon** 为参照，而非旧 AdamW 默认。
- 全基因图覆盖表达矩阵基因与扰动靶点，保留 GO/STRING 来源信息。
- 首次图读取按每来源 Top20 邻居限制；之后允许 KDA/非因果 MLA token 交互。
  Expander 是另外的传播机制，不应把其边冒充生物 Top20 边。
- 基因依冻结的“出现顺序”排列；裁剪保留相对顺序。额外靶点的追加次序须冻结。
- 训练随机选取约 1000 个允许观察的基因。测试表达列在现有预处理矩阵上移除：
  这是表达列留出协议，不宣称所有上游预处理都完全未见这些基因。
- 测试时使用更多/全部基因是候选，尚未通过容量及 token 数分布偏移验证。
- GenePT-Seed 初始化可训练基因嵌入表；来源感知图主干由 Control 与扰动分支共享。
- 不加入 MoE、短因果卷积或四 token pooling。
- Student 用于表达预测，不 detach 扰动表示；Teacher 无预测反向梯度、通过 EMA 更新。

## 2. 三种表示与两种蒸馏，不能混淆

```text
基因知识图 → Student 图编码器 → 扰动靶点聚合 → e_p ───────────┐
                    ↓                                     │
              基因表示 + control 表达                     │
                    ↓                                     │
             Cell Encoder → H_c、control CLS s_c           │
                              │                            │
                              └── concat(H_c,g, e_p) ──────┘
                                          ↓
                         预测 Transformer + response CLS
                                  ↓                  ↓
                            基因响应 tokens       response CLS z_cp
                                  ↓                  ↓
                            逐基因 Δx 头         蒸馏② projection head
                                  ↓
                             x_control + Δx

扰动图/condition 表示 e_p → 原有蒸馏① head
```

- `s_c`：Control Cell Encoder 的细胞状态，功能上对应 v1 basal latent s。
- `e_p`：扰动图条件表示，蒸馏①的对象。
- `z_cp`：融合 control 与扰动后得到的响应 CLS，蒸馏②的对象。
- **蒸馏①不是 control CLS 蒸馏；蒸馏②也不是另加一套 Cell Encoder CLS 蒸馏。**
- Teacher 共用一份 EMA 图编码器，既服务蒸馏①也产生蒸馏②的 e_p^T。
  蒸馏②所需 Cell Encoder、预测 Transformer 和投影头也需有明确的 EMA 对应。
  共享参数每个 optimizer step 只更新一次 EMA。
- Teacher 不额外融合外部 control CLS 上下文；这不意味着移除其自身 CLS token。
  早期 local3/4 注入其他细胞 CLS 的方案暂不作为本版默认。

## 3. CLS 的信息流与遮蔽

每个候选四层单元：三层顺序 KDA，随后一层非因果 MLA。
CLS 放在基因序列末尾：KDA 能把前方基因信息传到 CLS，MLA 再让基因读取 CLS。
mHC 应包围相应注意力与 FFN 残差子层，不是仅在整个单元结束后使用一次。
最终逐基因输出必须位于这种 CLS→基因的信息通路之后。
末尾 CLS 可接收全部前缀，不保证其有限状态能无损保存全部信息。
双向 KDA 留作独立对照；双向可达不等于排列不变性。

表达遮蔽定义：

\[
e_g=E(g,\mathcal G),\quad a_g=\mathrm{ExprMLP}(x_{c,g}),\quad
\tilde a_g=\begin{cases}m_{expr}&g\in M\\a_g&g\notin M\end{cases}
\]
\[
(\tilde H_c,\tilde s_c)=\mathrm{CellEncoder}(\{e_g+\tilde a_g\}).
\]

gene embedding 与 expr embedding 按元素相加（同为 d 维），不使用未定义的 Fuse。
Control CLS 是额外可学习的末尾 token，与基因 tokens 共同经过编码器获得 s_c；
它不是直接压缩原始表达矩阵，也不是在最终基因输出后额外做一次 pooling。
保留 gene embedding，在表达融合及跨基因交互之前遮 expr embedding。
Teacher 使用同一 control 的未遮蔽表达。Student 不得旁路读取未遮蔽的全局 CLS、
表达派生 KV 或其他缓存。静态知识图 KV 可以保留。残差输出的 x_control 不输入蒸馏头。
这防止直接泄漏，但不禁止模型从相关基因合理推断被遮表达。
留出测试基因的表达不参与训练 Teacher、目标或数据派生图。

## 4. 视图与损失

Teacher 两个 Global；Student 两个匹配 Global 加四个 Local。
用户接受提高裁剪比例，但具体数值未冻结。
建议起点 Global 70%–100%、Local 30%–70%；比例按节点数，不是图像 resize。
建议蒸馏②从本次 1000 个可见基因池裁剪；蒸馏①从其条件图裁剪。
**分母仍需最终确认**：全图 100% 与本次 1000-query 池 100% 不等价。
Local 下限必须大于零，并有最小有效节点数。保留扰动条件标识。
图节点裁剪是否同时限制 KV 支持集须独立配置：仅缩 query 不叫“缩小整个图”。
蒸馏②最初可固定静态全图 KV，只裁表达/query 轴，并如实命名。

蒸馏②包括候选 DINO-CLS、iBOT-response-token、KoLeo-response-CLS：

- DINO：同一 control 与同一扰动的跨视图响应 CLS 投影分布对齐。
- iBOT：匹配 Global 中按基因 ID 对齐；Student 的被遮表达位置对齐 Teacher 对应响应 token。
  不是恢复原始表达的另一份 MSE，也不是再次给同一个旧 graph iBOT 项计权。
- KoLeo：Student Global 的投影前归一化响应 CLS；两种 Global 分别计算。
  增强视图副本不能直接当成彼此最近邻负样本。批内条件重复需要审计。
- 蒸馏①保留原 condition/node/spread 的角色；开启时核对实际 Muon 父配置的权重。
- KoLeo 是否按扰动或细胞类型分组后续消融，不预先宣称分组更好。

总目标：
\[
L=L_{pred}+\lambda_1L_{SSL1}+\lambda_2L_{SSL2}.
\]

默认先 λ1=λ2=0。核对父配置后，建议开启蒸馏①时 λ1=1，且
L_SSL1=.8 L_condition+.4 L_masked-node+.1 L_spread，以保留原权重。
蒸馏②是新增目标，建议 λ2=.1 为起点；内部 DINO:iBOT:KoLeo=.8:.4:.1（用户确认），
因此在候选 λ2=.1 下有效系数为 .08/.04/.01。它们是待验证值，不是最优值保证。
这里取代初稿“两个外部λ均从.1起”的建议，避免把继承的SSL1悄悄缩小十倍。
各项按有效细胞、基因、视图对/遮蔽位置分别取均值，禁止随 local 数简单累加放大。
记录 raw/weighted loss、各路径梯度范数与冲突，不能仅凭 loss 数值接近判平衡。

## 5. 建议默认超参数（待容量验证，不是已确认运行配置）

| 项目 | 建议起点 | 解释/边界 |
|---|---|---|
| 训练轮数 | 50 | 不早停，best 按验证预测 loss，另测 epoch-50 last |
| 有效 batch | 1024 | microbatch、累积、GPU 数独立记录；不宣称物理 batch1024 可行 |
| 基因 query | 1000 | 只从训练允许的表达轴抽样，保持冻结顺序 |
| 主宽度 d | 256（用户确认） | gene adapter、expr、Hc、e_p、两种 CLS 同宽 |
| GenePT 输入维度 | 取既有 artifact 元数据 | 不用截断/填零强行变成 d；用可训练适配层 |
| concat 融合 | 512→256 | Hc 与广播 e_p 融合 |
| FFN 宽度 | 1024 | 固定4d、非 MoE，不做倍率消融 |
| Cell / 预测编码器 | 各 4 层 | 各一个 3 KDA + 1 MLA 单元 |
| 初始图读取 | 2 层候选 | 与 v1 层数不能混称相同；后续 2/4 层对照 |
| 注意力头 | 固定4，不做头数消融 | KDA 内核支持与 MLA latent rank 要分别校验 |
| MLA KV latent rank | 64 候选 | 这是缩小模型的项目设置，不照搬大模型比例 |
| mHC | 4 路候选 | 全部相关残差子层一致；基线普通残差单独对照 |
| dropout | 固定0.1，不消融 | 全矩阵固定；不自动照搬视觉模型 drop-path |
| 蒸馏 head | hidden2048 / bottleneck256 / prototypes16384 | 两蒸馏独立 head/中心；不是预测输出维度 |
| 优化器 | split Muon + 辅助 AdamW | Embedding/norm/bias/output 不按 ndim 简单扔进 Muon |
| 基础 peak LR | 1e-3 | 每组实际 LR/形状修正显式记录 |
| weight decay | 0 | 已核实父配置值，建议继承 |
| 默认 LR schedule | 项目既有 warmup+cosine 配方 | warmup 16%，floor/peak=0.2；不消融日程，来源见第15节 |
| EMA | .99→1 cosine 候选 | 使用完整 50-epoch optimizer-step horizon |
| Student / Teacher 温度 | .1 / .04 固定候选 | 不同时引入温度 warmup；非 DINOv2 全套默认复刻 |
| expr mask | 50% Global 样本，选中者遮10%–50% | 与图裁剪独立；只对可观察表达轴 |
| Global / Local | 2 / 4 | SSL 关闭不执行无用视图与 Teacher |
| Global / Local 比例 | .7–1 / .3–.7 候选 | 分母按第4节单独声明 |

两种蒸馏的 condition/CLS head 与 node head 是否共享是额外设计选择：建议同一
蒸馏内部先延续已核实的头策略，不跨蒸馏共享投影参数或中心统计。只在被遮节点上
计算 node logits，避免保留 B×全部基因×16384 的巨型 logits 张量。

batch 累积不能自动等价于大批 KoLeo：最近邻实际在哪个 microbatch/跨卡集合计算，
必须独立记录。EMA 与中心更新按 optimizer step 定义；不能累积一次却更新多次 Teacher。
全基因图 KV 是否按细胞复制、1000-token 六视图和 mHC 内存均未验证。

## 6. LR 机制：按用户最新决定冻结

peak LR 对比 1e-4 / 1e-3。移除 warmup-only、cosine-only；warmup 时长
沿用项目既有配方：warmup占总更新步数16%，cosine末值为peak LR的20%，
不再比较5%/10%/16%。来源见第15节，明确标为项目设定，不是GLM5官方日程。
断点恢复不能重新预热；容量 smoke 不压缩正式50轮日程。
所有 Muon/AdamW 组应用统一 schedule 倍数，保留各自已审计的 LR 映射。

## 7. 分阶段消融，不展开全笛卡尔积

第一阶段固定结构、学习率、视图定义、数据与种子：

| ID | 预测 | SSL1 | SSL2 | 目的 |
|---|---|---|---|---|
| V2-L0 | 开 | 关 | 关 | 全 token 模型预测基线 |
| V2-L1 | 开 | 开 | 关 | 原扰动表示蒸馏贡献 |
| V2-L2 | 开 | 关 | 开 | 响应 CLS/token 蒸馏贡献 |
| V2-L12 | 开 | 开 | 开 | 互补或冲突 |

关闭一个 SSL 时同时关闭它自己的 node/KoLeo/spread 正则，不能残留隐藏辅助损失。
L12 与 L1/L2 的差异按联合目标解释；交互可用 M12-M1-M2+M0 描述，并报告跨种子不确定性。
旧 v1 Muon 是外部结构参照，不冒充与 V2-L0 仅差一个因素的消融。

第二阶段在固定选定配置上逐轴比较；只按验证集决策，测试仅最终报告：

| 因子 | 候选水平 | 必须固定/报告 |
|---|---|---|
| λ1 / λ2 | λ1=.3/1/3；λ2=.03/.1/.3 | 一次只变一项，内部比例不变 |
| peak LR | 1e-4 / 1e-3 | 固定 batch、schedule |
| 有效 batch | 双5090压力测试后冻结2–3个可行档位 | 首轮固定 LR；报告microbatch/累积/更新步数，不能同时线性缩放 LR |
| Local 数 | 0 / 2 / 4 / 8 | Global=2，loss按视图对平均，记录计算量 |
| Teacher/匹配 Student Global 比例 | .32–1 / .7–1 / 1–1 | Local固定；全量Global仍可有独立遮蔽 |
| Local 比例 | .05–.32 / .3–.7 / .5–.8 | Global固定，节点数下限固定 |
| DINOv2-inspired 视图组合 | G .32–1、L .05–.32、8 locals | 三因素组合对照，不称单因素 |

第三阶段结构/正则：单向 vs 双向 KDA；普通残差 vs mHC；宽度128/256；
Cell/预测深度4/8（分开变）；SSL2 DINO-only、+iBOT、+KoLeo、全开；
KoLeo 不分组/条件分组/关闭。不要把蒸馏①与②的 KoLeo 同时改掉后归因到某一分支。
条件分组须定义“细胞类型”“扰动ID”“完整条件”，小于2个有效样本时不计算最近邻项。

每个关键四组对照建议至少3个匹配种子；探索阶段可先固定一组种子做筛选，
不得把单种子差异写成稳定结论。固定50轮是等暴露而非等 FLOPs，报告吞吐、显存、总耗时。
所有实验仍需独立实现/测试/版本发布/容量门禁与明确训练授权。

## 8. GenePT-Seed 缺失补全约束

复用 `/Users/elan/code/DinoGenePT` 的语料、身份映射、embedding/cache/export 管线。
不能从“GenePT”名字推断向量空间或语料一致。

1. 在服务器确定实际父 artifact 与 manifest，核对模型及版本、原始维度、归一化、
   输入模板、截断规则、GO/蛋白/通路等语料组件和来源版本。
2. 计算 v2 全轴（表达基因∪靶点）相对 artifact 的缺失，先消除可核实的别名/ID 映射问题。
   旧图覆盖不等于全表达轴覆盖；未知映射不能猜测。
3. 只为真实缺失基因构造同一冻结配方的文本；复用 `gene + text_sha256 + model` 缓存。
   增加新基因会改变整体语料 hash，但已有基因文本指纹不得变化；新manifest记录父hash。
4. 生成模型、维度、后处理必须一致。DinoGenePT 历史文档明确禁止把1536维Ada旧向量
   与2048维Doubao补向量混合。维度相等也不能单独证明同一向量空间。
5. 用新 artifact 保存扩展结果，不覆盖父向量；逐行检查旧向量未变、有限值、覆盖、排序与哈希。
6. Ark 走项目 Agent Plan endpoint；Keychain 一次性注入，不落盘、不打日志。
   embedding 和表达矩阵都只在服务器处理，不下载本地。
7. 原语料/模型无法恢复时报告阻塞，不静默换配方或以随机向量冒充 GenePT-Seed。

本次核对：SSH 已恢复；未调用API、未生成新嵌入。本地 Ark credential 存在，
DinoGenePT client 使用 `https://ark.cn-beijing.volces.com/api/plan/v3`。
服务器既有 artifact `seed-go-protein-pathway-master-aligned.npz` 实际为
17730×2048、`doubao-embedding-vision`；重算SHA256与父配置一致：
`34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`。

语料 manifest 的 profile 为 `protein-pathway`，每来源最多8条、每字段最多2000字符。
注意该profile **包含SIGNOR**，以及基础语料、UniProt、InterPro、Reactome；
不能只读名字就把SIGNOR漏掉，也不能换成SIGNOR-masked/shuffled或HPA版本。
语料输出hash：`7a952fa7feaf2f19f3e810e8b50532ee0cfd479deb93d62b47dfebedcc9daa3c`；
向量生成manifest的文本指纹：`c892dff33cf0b757ac9a044e6664d2b606928c3582f55279c614aa91fdcc8ffe`。
若将来确需补全，还须核验冻结生成代码、模型版本/服务漂移与来源文件hash，
不能只凭同名API模型断言新旧向量完全可比。

2026-09-21服务器只读精确键覆盖检查：

| canonical数据集 | 表达列数 | 表达轴∪既有图轴 | 缺失键数 |
|---|---:|---:|---:|
| Nadig HepG2 | 5000 | 6530 | 0 |
| Nadig Jurkat | 5000 | 6506 | 0 |
| Replogle K562 essential | 5000 | 5657 | 0 |
| Replogle RPE1 essential | 5000 | 6386 | 0 |
| Norman | 5045 | 5045 | 0 |

当前矩阵没有发现需要付费补全的缺失项。此检查不是新v2靶点清单、别名消歧或原始
全转录组覆盖验收；v2最终基因轴冻结后仍须核对全部靶点及身份映射。

## 9. 主要未决点与证据边界

- 裁剪分母、KV是否一起裁剪、图连接性与靶点保留规则。
- MLA/KDA 的缩小维度、内核与 mHC 实际容量；默认数值均是建议。
- 父配置与run身份已核实；新增KDA/MLA/mHC参数如何落入split Muon/AdamW仍需逐参数审计。
  本地当前主树不是那份新实验配置，不用100轮配置替代。
- KoLeo 重复条件与 microbatch 分组、两类 SSL 梯度竞争留待消融。
- 1000-query训练到全基因推理的变化，以及留出表达列场景的 Δx 质量，必须单独验证。
- 留出基因报告 Δx 指标、逐条件误差及复制 control 对照；整体表达相关性不能独证成功。

参考：本仓 `docs/experiments/R50_GLM5_MUON.md`（历史方案，非待跑队列）；
DinoGenePT `docs/GRADPERT_EXTENSION.md`、`src/dinogenept/embedding.py`。
官方 DINOv2 config：
https://github.com/facebookresearch/dinov2/blob/main/dinov2/configs/ssl_default_config.yaml
（本记录借鉴视图/损失角色；不声称复刻其整套默认，实施前固定源commit。）
GLM-5 报告：https://arxiv.org/abs/2602.15763 。

## 10. 2026-09-21 后续确认：监督分支、采样与结构消融

本节记录用户本轮确认；仍为设计阶段，不授权启动训练，也不修改 v1 协议。

### 10.1 已确认的监督边界

- “真实表达监督只放第一行”解释为独立 prediction branch，不是 batch 第一条细胞。
  四组 loss 消融统一使用该分支、相同约1000基因采样策略、输入处理和监督次数。
  Global/Local 仅贡献蒸馏损失，不额外计算真实表达预测损失。
- 保留固定扰动替换 control、固定 control 替换扰动、阻断 response CLS→gene
  信息通路三种诊断。报告预测变化与误差，不把敏感性越大等同于生物学正确。
- 泛化拆为两轴：固定推理 token 预算比较训练表达可见/留出基因；固定评估基因
  比较1000/更多/全部上下文。表达列留出与扰动条件留出分别记录。
  “更多/全部”不是已验证能力。所有目标基因必须被实际输出；不能用缺失预测补零。
- 共享参数不等于共享视图，不触发嵌套蒸馏。图输入/裁剪/mask 不同时不能直接复用输出。

### 10.2 采样与归一化消融

用户要求把“SSL1按不同条件平均、SSL2按有效(control,p)平均”作为待验证因子，
不宣称其优于现有行级加权。比较时固定样本流及所有其余损失定义；记录重复率、
每条件贡献、有效样本数和梯度范数。输入完全相同的重复图前向可复用；独立增强视图
不能因条件相同而被误删。

增加 scDFM-inspired 单扰动采样：先取一个训练条件 p，再取 B 个该条件目标细胞和
B 个 control；不是同一细胞的真实前后配对。主候选关闭 KoLeo，不自动引入 MMD/flow
matching，也不改变既定 prediction loss。确认现有 spread 的实际定义后再映射开关，
不能只凭名称把所有 spread 都当作 KoLeo。

| 对照 | 每有效更新的扰动组织 | KoLeo | 解释 |
|---|---|---|---|
| S-M0 | 多扰动，固定每条件配额 | 关 | 采样对照基点 |
| S-S0 | 单扰动 B 个细胞 | 关 | 用户指定的新分支，与M0隔离采样因素 |
| S-M1 | 与M0相同 | 开 | 混合条件下的KoLeo贡献 |
| S-S1 | 与S0相同 | 开，额外候选 | 仅用于研究交互，不是单扰动分支默认 |

第一轮先比较 M0/S0；不得仅用 M1/S0 把采样和关KoLeo两个变化归为一种收益。
单扰动条件级表示去重后只有一个样本，条件级最近邻项应跳过；不能拿它的增强副本
充当负样本。SSL2即使同扰动仍可有不同control，本轮S0明确关闭该处KoLeo。
DDP/梯度累积下，S0暂建议整个有效 optimizer update 使用同一p，两卡与各microstep
一致；每卡/每microstep另抽p属于另一种实验。B仍需区分物理、每卡和有效batch。
有放回采样记录独立cell ID数；单条件不足B时不能声称获得B个独立细胞。
50轮需固定与基准相同的抽样槽位/更新预算，并报告真实唯一细胞覆盖；不能沿用
scDFM人为Dataset长度重新定义一轮。条件均匀/按频率采样也是独立因子。

### 10.3 新增结构消融设计

| 因子 | 对照 | 隔离要求 |
|---|---|---|
| 全token交互 | 同gene/expr输入、同图先验的共享逐基因MLP vs token交互 | 同输出轴和监督；MLP也可用同一pooling状态 |
| v1到v2总体变化 | 冻结v1父行 vs v2 prediction-only | 仅称整体架构比较，不能归因一个组件 |
| 交互机制 | 普通非因果Transformer vs 3 KDA+1 MLA | 同宽、相同总层数、普通残差起步；报告参数/FLOPs |
| KDA方向 | 单向 vs 双向 | 排序、宽度、损失固定；双向不等于排列不变 |
| mHC | 普通残差 vs mHC | 固定注意力结构；4路只是候选 |
| GenePT初始化 | 同维可训练表随机初始化 vs GenePT初始化 | 相同adapter/参数量/图/优化分组；冻结embedding另作因子 |
| 宽度 | 默认256，对比128；512暂不纳入 | 固定层数、视图、有效batch、训练预算 |
| Cell深度 | 4/8 | 固定response深度 |
| Response深度 | 4/8 | 固定Cell深度 |

四组SSL消融仍优先；上述结构逐轴展开，不生成全部笛卡尔积。
GenePT含知识语料，GO/STRING另为图先验，初始化消融不能宣称同时剥离了所有外部知识。

## 11. scDFM 官方代码核对：1000基因与单条件batch

核对日期2026-09-21，官方仓库固定commit：
`2cf6bca1f044e74c4e1dc586892c0495880cf125`。
只读下载了相关源码，无模型运行/科学数据下载。

- 训练：`src/script/run.py:85-93` 的 train_step 对表达轴randperm后取
  infer_top_gene（示例1000）；source/target/gene IDs共用索引，整批共用一组。
  新step重新抽样。源码直接使用随机排列；GraD-Pert仍按冻结出现顺序重排，不能照搬顺序。
- 推理：`src/data_process/data.py:210-211` 先在 adata_test 上选择1000 HVG并切列；
  `src/script/run.py:133-170,218-238` 用切列后的control及对应gene IDs，建立1000维噪声，
  在整个ODE过程中保持这一基因集合。不是1000一块遍历全部5000输出。
- 官方此处HVG使用测试表达，不能移植到本项目。我们的推理清单应来自训练侧统计或
  预先指定清单，保存ID/hash；表达留出评估清单不得按测试响应挑选。
- batch：实际入口使用 `PerturbationDataset`（data.py:362-398），每item均匀选一个p，
  有放回抽B个target和B个control。外层DataLoader batch_size=1因为item本身就是一批。
  Dataset长度为条件数×1000，主循环按steps停止；不能将这个“epoch”视作全细胞一遍。
- 论文Appendix A.3描述只输出选定子集，未使用全词表imputation head。
  A.4.3报告H800、d512/L4/H8、batch96、Adam5e-5、100000步、cosine到1e-6。
- 该commit的示例run.sh实际指定d128、batch48、200000步、LR5e-5；
  config_flow.py未覆盖默认又是d512、batch32。三者必须分开，不能混成一个官方配置。
  示例推理默认RK4/20个时间点也不同于论文Euler/100步；本轮未验证checkpoint对应哪套。

来源：
[论文](https://arxiv.org/html/2602.07103v1)；
[训练与推理入口](https://github.com/AI4Science-WestlakeU/scDFM/blob/2cf6bca1f044e74c4e1dc586892c0495880cf125/src/script/run.py)；
[数据和采样](https://github.com/AI4Science-WestlakeU/scDFM/blob/2cf6bca1f044e74c4e1dc586892c0495880cf125/src/data_process/data.py)；
[示例配置](https://github.com/AI4Science-WestlakeU/scDFM/blob/2cf6bca1f044e74c4e1dc586892c0495880cf125/run.sh)。

## 12. 双GPU参数讨论：估算与实测边界

2026-09-21两次只读SSH均连接超时，未取得当前GPU占用。既有STATUS记录两张RTX5090，
历史容量记录约31.36 GiB/卡；本节基于这份历史硬件信息，不声称当前两卡可独占。
现有每GPU两任务、每进程40%容量门禁不在本轮放宽。DDP复制模型，不能把两张32GB
当作一张64GB使用；模型分片是额外工程方案，不在本轮默认。

以下是候选容量搜索点，不是可运行batch承诺或最大值；第5节d128仍为原讨论起点。

| 档位 | 主宽度 | Cell/Response层数 | FFN | 头数候选 | 单卡物理microbatch首次试探 |
|---|---:|---|---:|---:|---:|
| 小型参照 | 128 | 4/4 | 512 | 2（head64） | 8 |
| 建议主力候选 | 256 | 4/4 | 1024 | 4（head64） | 4或8 |
| 扩容候选 | 512 | 4/4 | 2048 | 8（head64） | 2或4 |
| 更深压力探针 | 512 | 8/8 | 2048 | 8（head64） | 1或2 |

注意力头/维度必须适配实际KDA/MLA内核。宽度消融建议固定head_dim=64并披露头数变化；
这与旧d128四头候选不同，尚待选择。mHC4路、MLA rank64/128/256均属可单独控制的候选，
宽度消融先冻结明确的rank规则，不同时随意改变多个容量旋钮。

有效batch1024仍可通过microbatch×GPU数×累积步数定义，例如16×2×32=1024，
不是声明microbatch16已通过。KoLeo近邻集合另记，累积不等价于大批联合近邻。
采用bf16、激活检查点、Teacher no_grad、视图分次计算/反传等是待实现的内存优化；
不能把共享静态gene KV按B个细胞无意复制，不能缓存跨optimizer step的可训练输出。

举例仅一个bf16 logits张量：B=16、两Global各1000节点、mask=25%、16384原型，
大小16×2×1000×.25×16384×2字节≈250MiB；全1024则约15.6GiB。
若softmax/logits实际为fp32还要翻倍，且未计梯度、Teacher、主干和图视图。
若稠密注意力显式物化N×N矩阵，1000→5000使该项增25倍；使用高效内核也不消除
运算增长。因此宽度、batch、视图数、上下文长度没有可以同时取满的独立最大值。

未来容量验证：在已发布一致源码与资源授权下，带全部视图、两套SSL、optimizer step、
EMA测试真实完整更新；从小microbatch递增，报告allocated/reserved峰值、空闲余量、
吞吐和内核版本。再做短持续验证，单步通过不是50轮完成。所有CUDA进程保留
PYTORCH_ALLOC_CONF=expandable_segments:True。当前未实施任何容量测试。

建议讨论主力d256/4+4，d512作为后续扩容因子；不能依据scDFM单路径H800配置承诺
我们双编码器、EMA、多视图、mHC模型的d512/batch96能在5090运行。LR仍先固定
既定Muon1e-3参照，scDFM的Adam5e-5不能直接作为Muon等效学习率；另做独立LR消融。

## 13. 2026-09-21 最新设计决议与图注意力语义

本节及前文同步后的值优先于早期候选。讨论顺序为超参数 → 模型参数 → 功能设计。

- weight decay 默认0，仅比较1e-5；dropout固定0.1。
- Cell / Response Encoder各4层、注意力头4、FFN 4d固定，不做这三项消融。
- 图默认 GO + STRING + expander，暂不开展图来源/expander开关消融。
- 蒸馏①默认按不同扰动条件平均；按样本行平均保留作归一化对照。
  蒸馏②按有效(control,p)样本统计，不能把两种归一化混为一谈。
- 两套蒸馏分别设计全局分布项、masked-token项与spread/KoLeo项的开关组合；
  蒸馏①历史masked-node/spread尚不能未经核实直接称为标准iBOT/KoLeo。
- 保留全token交互、KDA/MLA对普通注意力、mHC对普通残差、GenePT初始化消融；
  scDFM-inspired单扰动B细胞采样作为独立对照，关闭KoLeo，并报告这一联合变化。
- 保留固定p换control、固定control换p，以及阻断CLS→gene通路的诊断。
- 真实表达监督只在主预测分支；Global/Local只用于蒸馏。

### 13.1 图 mask 的位置

图02及总图的运算关系为 Q/K/V投影 → QK得分 → 加图mask及候选来源项 →
Softmax → 加权V → 输出投影 → 残差/FFN。图mask限制可注意的边，不是将Q/K/V置零：

\[
A_{ij}=\operatorname{softmax}_j(q_i^\top k_j/\sqrt{d_h}+b_{ij}+M_{ij}),
\quad M_{ij}=0\ (j\in\mathcal N(i)),\quad M_{ij}=-\infty\ (j\notin\mathcal N(i)).
\]

邻域为 GO Top20、STRING Top20、expander及self的并集；来源项b尚待冻结。
实现可预先索引邻居，只对保留边计算得分；不要求先物化完整query×全基因矩阵。
这是图mask；表达mask必须在Cell Encoder的表达融合与token交互前生效。

### 13.2 scDFM 对照（官方冻结版本）

核对commit `2cf6bca1f044e74c4e1dc586892c0495880cf125`：
`GeneEncoder.forward`先用`mask_padded[x[0]]`取得所选基因对应mask行，
再构造全部memory基因embedding；未在这一调用路径按邻居裁剪K/V。
`CrossAttentionTransformerLayer`将query、全memory的key/value和`attn_mask`
传给`nn.MultiheadAttention`。mask准备发生在前面，但其注意力数学作用位置是
QK得分之后、Softmax之前；参考图箭头指向整个模块只是省略内部运算。
不能据此把普通带mask的注意力称为已经实现稀疏边计算。

来源：[GeneEncoder](https://github.com/AI4Science-WestlakeU/scDFM/blob/2cf6bca1f044e74c4e1dc586892c0495880cf125/src/models/origin/layers.py#L73-L89)、
[CrossAttentionTransformerLayer](https://github.com/AI4Science-WestlakeU/scDFM/blob/2cf6bca1f044e74c4e1dc586892c0495880cf125/src/models/origin/blocks.py#L227-L319)。

## 14. 后续batch决议（2026-09-21）

用户确认：后续batch消融根据双5090完整训练链的压力测试结果生成，不预设512/1024/2048。
实测选出稳定且高吞吐的参考batch，再冻结2–3个可行档位；其他超参数消融固定该参考值。
前文1024仅为历史父配置/早期候选，不再作为v2正式实验强制batch。
记录物理microbatch、GPU拓扑、梯度累积和有效batch；KoLeo实际近邻集合单独核对。

## 15. 已确认的v2项目LR配方（2026-09-21）

用户确认沿用项目已有warmup+cosine，明确为项目设定，不称GLM5官方默认。
采用已发布`19342809e0cacfe70196da32c356471f2f78d9e0`中
`configs/r50/g2_schedule/gradpert_b2/nadig_jurkat.yaml`的日程：
warmup占总optimizer steps的0.16，cosine末值为peak LR的0.2，首步LR为0，
末个使用步骤精确到达floor；不重启、不做warmup比例消融。
LR消融将peak与floor同比缩放（1e-3→2e-4、1e-4→2e-5），所有组保持日程规则。

## Capacity probe settings

`configs/v2/capacity/gradpert_v2/nadig_jurkat.yaml` 是双5090容量搜索的起始探针，
不是正式消融batch选择：microbatch=2、accumulation=1、单进程；两卡分别测量。
完整d256、4层Cell+4层Response、双蒸馏、全部16384原型均保留。
压力测试中明确固定的项目候选：expander degree=8/seed=1，图mask=.25，
图edge dropout=.1；表达Global比例.7–1、Local比例.3–.7，2G+4L；
Global逐细胞mask概率.5，mask比例.1–.5。它们不是scDFM官方默认的声明。
推理使用显式1000基因有序块覆盖全部表达轴，eval cell batch=2；
全基因上下文是另一个推理配方，需要独立容量证据。
正式batch水平仍只能由完整链路实测收据生成。

### A1 逐基因 MLP 对照的确切定义

`attention=per_gene` 将 Cell/Response 每层注意力子层替换为
`Linear(d,d)→GELU→Linear(d,d)`；保留四层、FFN、归一化、残差与mHC设定。
表达维度内各基因独立处理，图读出与扰动条件注入不变。
最后的CLS槽改为gene输出的均值只读汇总，供蒸馏使用；它不回流到gene预测。
该组检验token交互的贡献，不是参数量严格匹配的对照，报告实际参数量和吞吐。
测试要求单基因表达干预不改变其他gene token，而汇总CLS能够响应干预。
