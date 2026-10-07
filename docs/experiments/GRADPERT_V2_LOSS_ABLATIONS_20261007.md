# GraD-Pert v2：预测 loss 与掩码重建消融设计

日期：2026-10-07。状态：**IMPLEMENTED_LOCAL，原生实现与自包含配置已写入；权重1本地验证通过，尚未发布/启动，无新科学结果。**
用户范围：设计此前讨论的 L0–L3、M1–M2 六组消融。本文件不改变已有运行、注意力消融队列或论文正文。

后续用户指令（2026-10-07）：**“等下优先跑这些实验”**。六组列为下一批优先执行范围；建议顺序 L0→L1→L2→M2→L3→M1。父队列已成功结束并独立验收，继续实现/验证/发布/容量/启动。执行安排见 [优先执行记录](../../.byte-os/plans/GRADPERT_V2_LOSS_PRIORITY_20261007.plan.md)。
最新安排：**当前A2的6epoch及final测试完成并验收后立即续接本消融**，不等待其他未来实验。已有返回监控已更新；实现/验证/发布与容量门槛仍须通过。

执行分工：主会话 `codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b`；本轮执行会话 `codex:01a10391-22a3-78e1-9799-93ea9d6ffdd9` reports_to 主会话；长任务监督 `codex:01a0df0b-4142-7df1-86c0-d959471d80a1`。完成/失败/需设计决策时最终返回主会话，执行会话不是长期 main。

## 1. 科学问题与证据边界

问题：当前预测 MSE 的早期下降，是否对应扰动特异响应的学习？改变误差惩罚、条件权重或增加掩码表达重建，能否改善未见扰动的预测？

训练曲线下降快本身不能证明 MSE 太简单，也不能证明模型只复制 control。当前输出是 `prediction = control + delta`；大量变化较小的基因、噪声及各任务梯度比例都可能解释该现象。先做同输入诊断，再做受控比较。

本实验是 **Jurkat 遗传扰动、同细胞背景、未见单扰动条件预测**。不能据此声称跨细胞类型、药物或其他模态泛化，也不能把预测相关性解释为机制正确。

三个待检验因素：

1. 误差形状：平方误差 versus 四次方误差。
2. 预测监督权重：按细胞行平均 versus 按当前有效全局 batch 的扰动条件平均。
3. 表征约束：原有自蒸馏之外，增加真实表达数值的掩码重建。

CellFM 的掩码损失仍然使用 MSE；其关键变化是隐藏输入和双重数值重建监督，而不是换一种更复杂的误差函数。

## 2. 冻结参照与拟用协议

已核对的父版本：`3d3f5ad2d6b831baed1eead45d61862fdea2db33`。
父配置：`configs/v2/cap40_functional_b0_ka/B0/gradpert_v2/nadig_jurkat.yaml`，SHA256：
`8961c93fd42bf0141722a42e23a3577c9209e5cdfb9aea1e8bba3c5de92467a5`。
核对来源为该版本的本地独立 worktree；此处不声称已检查当前 GitHub、服务器或队列状态。

新 loss 实现应从父版本建立可追溯后继版本，六组使用同一新发布 SHA；历史 B0 只作背景参照，**L0 要在新实现上重新训练**。

| 项目 | 六组共同设置（设计提案） |
| --- | --- |
| 数据 | Nadig Jurkat，既有 canonical 数据、条件 split 和 representability intersection |
| 训练选择 | 原 cap40 receipt；SHA256 `aa0916dfb91fff78d7a43f7d7325879a11c1a904e9bf31586420ca11cce84132` |
| 训练表达访问 | 保持 `exclude_test_target_expression=true`；新增 mask 也只能在允许训练表达的基因上产生 |
| 架构 | B0：Graph 3 KDA + 1 sparse MLA；Cell 2 KDA + 1 MLA；Response 2 self/cross KDA + 1 self/cross MLA |
| 其余模型项 | width256、heads4、MLA rank64、mHC4、GenePT trainable、16384 prototypes、单次随机顺序、final-state reads |
| 原有监督 | 预测项系数1；SSL1 condition/node=(1,1)、spread=0；SSL2 DINO/iBOT=(1,1)、KoLeo=0；lambda1=lambda2=1 |
| 数据采样 | 原 sampler、control 配对、每次 query1000、graph/view 分布；不因 loss 更改 sampler |
| 预算 | fresh seed1、完整6 epochs、无 validation、无早停、仅 final epoch6 last 测试；best=null |
| Batch | 272 = micro68 × accumulation2 × world2；若容量不支持，停在容量阶段，不能单独降低某一组 batch |
| 优化器 | GLM5MuonSplit_v2、weight decay0、LR 1e-3→2e-4、warmup fraction0.16、六 epoch cosine |
| 测试 | 原有每条件300 control 的有序 IDs、truth IDs、hash、推理 recipe 和评价实现完全一致 |

这里的六 epoch 是新 loss 家族的**拟用探索协议**，不是给所有 R50 工作增加例外，也不是已有注意力队列授权的扩展。未来运行需要另行落实新家族、版本与运行身份。用户已明确授权按此协议执行六组。

## 3. 六组实验矩阵

记预测误差 `e_ig = prediction_ig - truth_ig`，每行 query 基因数为 Q。
`a_i(p) = mean_g |e_ig|^p`，p∈{2,4}；N 为有效行数，C 为这些行中出现的条件集合。

- 行平均：`R_row(a) = sum_i a_i / N`。
- 条件平均：`R_cond(a) = mean_c [mean_{i:condition_i=c} a_i]`。
- `L_SSL` 为原有四个活跃 SSL 分量之和，权重、统计更新与 reduction 均冻结。

| ID | 预测监督 | 新增辅助监督 | 决定性对照 | 要回答的问题 |
| --- | --- | --- | --- | --- |
| L0 | `R_row(a(2))` | 无 | 全部组的匹配基线 | 原始 MSE 的实际表现 |
| L1 | `R_row(a(4))` | 无 | L1−L0；L3−L2 | 强调大误差是否改善响应预测 |
| L2 | `R_cond(a(2))` | 无 | L2−L0；L3−L1 | 降低 batch 内高频条件的权重是否有益 |
| L3 | `R_cond(a(4))` | 无 | 与 L0/L1/L2 构成 2×2 | 误差形式与条件权重是否存在交互 |
| M1 | 与 L0 相同 | `1 L_gene-mask` | M1−L0 | 基因级数值掩码重建是否有益 |
| M2 | 与 L0 相同 | `1 L_gene-mask + 1 L_CLS-mask` | M2−M1；M2−L0 | 加入 CLS 数值重建后的净增益 |

各组总目标为 `L_prediction + L_SSL + L_aux`。M1/M2 是增加辅助任务；不删除扰动预测 MSE，也不替换现有 iBOT。
四次方组保持预测系数1，不附带 DEG 加权、方向项、非零基因过滤或新梯度裁剪。所有原有优化/裁剪设置冻结。

### L1：大误差惩罚

意义：平方误差的梯度为2e，四次方为4e³；相对于小误差，它会更关注大误差。
这不保证数值梯度更大：|e|较小时四次方梯度反而更小。因此必须记录梯度，不能用 loss 高低直接解释监督强弱。

最强替代解释：收益可能来自预测/SSL 相对梯度权重变化，或少量噪声离群点。记录训练残差绝对值分位数 p50/p90/p99、最极端1%基因位置对预测 loss 的贡献、预测与 SSL 梯度范数及裁剪频率。
若仅降低训练目标而扰动评价不改善，不能支持“更适合扰动预测”。

### L2：条件权重

意义：同一全局 optimizer batch 内，一个条件无论占几行都先计算自己的均值，再在出现的条件间平均。
它不是让整个 epoch 的每个条件出现次数相同；cap40/sampler 保持原样。记录各条件暴露次数与实际累计有效权重，检查改动大小。

实现必须汇总 **world2 × accumulation2 的完整有效行群体**后计算预测权重 `1/(|C| n_c)`，不能分别在 rank 或 microbatch 内条件平均再求均值。
必须验证 DDP 梯度平均及 accumulation 缩放，与单进程完整 batch 参考梯度相同。

当前 `loss_reduction` 会影响预测以及 SSL 路径；**不能把全局配置切成 condition_mean 来实现 L2/L3**。需要预测专用 reduction，保持 SSL 的原有 row_mean 及既有例外。
若变化由 sampler、SSL reduction 或不同优化步数造成，该组对照无效。

### L3：交互

按 primary metric 计算预先指定的描述性交互：
`I = (metric_L3 - metric_L2) - (metric_L1 - metric_L0)`。
I>0 表示在条件平均下，四次方的增益更大；单 seed 不证明稳定交互。四组完整报告，不能只保留有利组合。

## 4. M1/M2 的掩码路径与防泄漏

辅助任务仅使用与主任务相同训练 batch 的 **control 表达**。目标是隐藏部分 basal 表达后恢复其原始值。
选择 control 重建，是为了保留推理时可获得的信息边界；它可能只改善 basal 表征，不一定改善扰动响应，这正是 M1/M2 要检验的。

### 输入、mask 与输出

1. 在每行 Q=1000 个合法 query 位置，均匀无放回选择恰好200个位置（20%），包括合法的零表达位置；不依据表达大小或 DEG 选择。
2. 原始 control 留作 loss target。辅助前向中的被选位置移除表达嵌入贡献，用独立可学习 mask embedding 代替；保留 gene ID/先验特征及显式 mask 标志。
3. 通过共享的 basal Cell encoder 重新得到 masked token 表征和 **masked-control CLS**。该辅助路径不调用 Response encoder、不输入扰动后 truth、不读取主路径的干净 CLS。
4. 基因头：`Linear(256,1)` 从每个 masked-control token 的隐状态预测 control 数值，输出只在 mask 位置计分。
5. CLS 头：将 masked-control CLS 与对应 gene feature 各投影到256维，以内积除sqrt(256)加标量 bias 解码该基因的 control 数值；输出只在相同 mask 位置计分。这是原生设计，非官方 CellFM decoder 的逐项复刻。
6. 两头均直接输出标量，**无原始 control 值的 residual/skip**；mask 位置的真值只能在 loss 中读取。

允许的 graph/gene 先验不包含这些 control 的隐藏表达数值。不得在额外缓存、normalization 统计、干净 CLS 或跨路径 tensor 中泄漏被隐藏值。
同一输入双路径：主路径仍完整 control→Response→control+delta→扰动 MSE；辅助路径 masked control→Cell→数值重建。

### 辅助损失

`L_gene-mask = sum_ig m_ig (gene_recon_ig - control_ig)^2 / sum_ig m_ig`。
`L_CLS-mask = sum_ig m_ig (CLS_recon_ig - control_ig)^2 / sum_ig m_ig`。
分母在完整有效全局 batch 上计算；padding/无效行权重0。空 mask 为0且不发生除零（正常 Q1000 固定 mask 不会为空）。

M1：λgene=1、λCLS=0；M2：λgene=λCLS=1。用户于2026-10-07在启动前明确修订权重为1；固定设置，不根据 test 调整。
两个数值头是 student 的训练辅助模块，独立于原有 Teacher/center 路径；不改变 EMA/center 更新次数或顺序。
头的参数按原有优化器分组规则显式注册，记录配置；不能静默使用另一套 optimizer。

M1/M2 使用相同两个辅助头的构造和初始化；M1 的 CLS 头不参与目标和更新。分别报告总参数与实际活跃参数。
先按原顺序初始化共享主模型，再用独立 RNG 初始化辅助头，避免辅助模块改变主模型初始参数。
mask RNG、辅助 dropout 和顺序 RNG 使用独立可恢复状态，不能消耗主任务后续 query/view/顺序/dropout 随机流。
仅独立 mask generator 不足以满足这一要求，辅助前向的所有随机源都需隔离和 checkpoint。

推理使用原干净 control 路径，辅助头不参与输出。不要声称 mask 模型在测试时获得更多表达或额外上下文。

### 能支持什么，不能支持什么

- M1−L0：整个基因数值重建辅助任务的净效果；新增头、输入损坏、监督与计算共同变化。
- M2−M1：增加 CLS 重建项的净效果；不能严格把收益归因于 CLS 汇总机制，因为辅助损失总系数也增加。
- M2−L0：双头辅助方案的整体效果，不是完整 CellFM 与 GraD-Pert 的比较。
- 若要进一步宣称 CLS 机制更好，建议另加 `M2-weight`（gene2、CLS0）作总系数对照；系数相等仍不保证梯度相等。该组是后续建议，不属于核心六组。

## 5. 先行诊断与训练记录

诊断代码已实现，正式数据诊断仍待服务器执行，data_state=derivable；只使用训练条件，不访问 held-out truth。

**D0：复制 control 基线。** 对固定训练诊断 batch、同 query 和配对 control/truth 计算：
`L_copy = MSE(control, truth)`，`L_model = MSE(prediction, truth)`，同时记录二者差值。
L_copy>0 时记录 `1 - L_model/L_copy`；L_copy=0 时该比例标为 undefined，保留原始两值。
固定诊断 batch 在初始化及每 epoch 末以 eval mode 计算，各组一致；不替代正式训练 loss。
这只是配对训练诊断，不能等同于正式300-control测试的 MSE。

**D1：梯度贡献。** 同一固定训练诊断 batch 上，分别计算 weighted prediction、四项 SSL 的总和、gene-mask、CLS-mask 对共享 Cell 参数的 L2 梯度范数及 prediction/SSL 梯度余弦。
仅记录，不参与 optimizer step、不更新 Teacher/center、不影响采样 RNG；L0–L3 的 aux 标为不适用。
两组 mask 表征和条件平均监督覆盖范围不同，范数不能单独作为好坏判断。

每 epoch 保存：原始各 loss、带权贡献、joint、**独立计算的统一 row-MSE**、D0、D1、残差分位数、condition 暴露/累计权重、梯度裁剪/非有限状态、训练秒数与峰值显存。
统一 row-MSE 必须六组都记录；不能比较 L0 的 MSE 数值和 L1 的四次方数值判断模型质量。
训练曲线无 validation 数据，不能补造 validation 曲线。正式训练、诊断及指标计算都在 `/data/yilangliu`。

## 6. 评价、统计单位与判断规则

预先指定 primary：**TxPert 定义的 DEG macro Pearson**，保持既有 DEG 评价规则。
DEG 只在 evaluator 用于计分，不能借 test DEG 给训练 loss 选基因或加权。
Secondary：TxPert/TriShift/Systema 的 all/DEG 六项；统一测试 MSE；seen/unseen-expression 分组；耗时与显存。
同时报告每指标有效条件数、缺失/undefined 原因。不可把缺失相关性填0。

统计单位为 held-out 扰动条件；配对差值使用两个模型都有效的相同条件，不能把300 controls或细胞数当独立重复。
若将来获准多 seed，模型训练 seed 是额外重复层。单 seed 的条件 bootstrap 不包含训练随机性的全部不确定性。

预先指定五个主对照：L1−L0、L2−L0、L3−L0、M1−L0、M2−M1；M2−L0 与交互作为描述性补充。
报告 macro 差值与配对条件 bootstrap 95%区间（10,000次，固定 analysis seed20261007）；保留条件间依赖的局限。
如果进行显著性检验，使用双侧条件级配对置换并对五个主对照做 Holm 校正；单 seed 下结果只解释为固定训练实例的探索证据。

设计判断规则：

- primary 差值>0：记作该 seed 的描述性改善；区间跨0：记作不确定。
- primary 差值≤0：不支持本次设置带来改善；不等于否定整个 loss 家族。
- 四次方组若训练残差尾部集中度明显上升但 primary 无改善，记录为与“偏向离群误差”相容，不能宣称已证明原因。
- 条件平均组若全局权重没有发生可测变化，记录为该 batch 组成下弱干预，不能声称条件均衡无用。
- mask 组若重建 loss 下降但扰动 primary 不改善，只能说明辅助重建学到部分数值规律。
- 工程一致性失败（输入访问、预算、reduction 或 identity 不一致）意味着比较无效，修复后用新 run ID 重做，保留旧失败记录。

不使用 test 选择 λ、mask ratio、epoch、checkpoint 或注意力配置，也不据此反复挑选新设置。
所有已设计组结果完整保留；正式优越性结论需独立预算的多 seed 确认，不能只挑有利 seed。

## 7. 优先级、成本与后续范围

| 层级 | 范围 | 数据状态 | 预算与目的 |
| --- | --- | --- | --- |
| minimum decisive | D0/D1 + L0/L1/L2/M2 | 训练数据 existing；诊断 derivable；新运行 new_required | 4 fresh×6 epochs=24 model-epochs，先区分误差/条件权重/完整辅助方案 |
| recommended | 补 L3、M1，完成核心六组 | new_required | 总计36 model-epochs，完成2×2与双头拆解 |
| ideal | 全六组 seeds1/2/3，另研究 M2-weight 等强度对照 | new_required | 六组多 seed 本体108 model-epochs；扩展对照另计，不由当前设计自动授权 |

model-epochs 不是 GPU 小时估计。M1/M2 有额外 Cell 前向/反向；不从现有 B0 耗时推定其成本。执行前容量探测必须记录实际 step 时间、显存与恢复能力。
六组固定更新预算，额外计算属于辅助方案成本；不是严格等 FLOPs 比较。禁止将更新次数和计算预算同时变化而仍声称单因素隔离。
本轮固定 B0，不叠加 K1/K2/A1/A2。后续 attention×loss 或其他数据集实验需新设计，避免归因混淆。
最终材料目的：Methods 中记录目标与信息访问；消融表/训练诊断图报告真实结果；负结果和成本放补充。

## 8. 实现与验收合同

拟新增预测专用设置：error_power=2/4、prediction_reduction=row_mean/condition_mean；auxiliary_mask_ratio=0/0.2、lambda_gene_mask、lambda_cls_mask。
当前 schema 已支持这些设置，配置生成器与启动器逐项核对，其他设置必须等于父配置。每组生成自包含配置并显式记录所有固定项。

实现位置以父版本为准：

- `src/gradpert/training/v2/objective.py`：预测误差与独立 reduction；辅助 loss 合成和日志。
- `src/gradpert/training/v2/reductions.py` 及现有 accumulation/DDP 更新路径：预测专用完整行群体权重，不扰动 SSL。
- `src/gradpert/modeling/v2/model.py`：可复用的 masked basal 编码入口与 student 数值头，保持主预测输出。
- `src/gradpert/training/v2/views.py`、checkpoint 状态及 config schema：mask/辅助随机流及完整恢复。

关键验收（合成单元验证已完成，M1/M2权重1验证通过；服务器容量/正式阶段尚未执行）：

1. L0 与父版本同输入输出、loss、梯度、optimizer/EMA/center 一次更新一致；不存在辅助计算或随机流改变。
2. p=2/4 与手算一致；condition 重复数不均、跨 rank、accumulation、无效行下，全局目标及梯度与完整 batch 参考一致。
3. 所有组 SSL 的权重、reduction、view、Teacher/center 时序保持一致。
4. 被 mask 的真值替换为任意值时，固定 mask 下辅助输出不变（gene ID和可见输入固定）；targets 变化只影响 loss。主任务不得访问扰动 truth 输入。
5. mask位置计分、padding排除、global分母、空mask安全；CLS使用masked前向；无control残差捷径。
6. M1/M2 初始共享参数相同；辅助头初始化及前向不改变主随机流；两组在CLS权重0的受控检查中相同。
7. 非零 LR 更新能给对应辅助头和共享 Cell 参数传梯度；关闭辅助项恢复L0行为；正式推理输出不依赖辅助头。
8. checkpoint 保存/恢复辅助头、optimizer与独立RNG；recomputation和中断恢复后更新一致。
9. 正式前所有组在同一新发布版本、精确配置和固定 batch 上通过128-update容量/重载/推理门槛；容量通过不是科学结果。

代码修改前核实干净已发布父版本；修改后完成针对性测试并按项目规则发布到 main。
服务器使用新的不可变 checkout，不修改现有运行源码；训练和评价 SHA、checkpoint SHA256、配置/data/split/control/truth/环境 hash 全部收据化。
任何正式 CUDA 进程使用 `PYTORCH_ALLOC_CONF=expandable_segments:True`；zero-PKL、仅小收据可回传、所有大数据/模型留服务器。
初次设计未创建运行或监控。用户后来明确要求当前6epoch后立即执行，本次更新复用现有完成/返回监控；不改变旧训练进程、源码或配置，不创建重复监督。

## 9. 冻结参考与来源

- 本项目父版本 `3d3f5ad2d6b831baed1eead45d61862fdea2db33`：`objective.py` 中完整query的平方误差；`model.py` 中control+delta；`reductions.py` 中全局行群体权重。属于已核对的实现证据，不是新 loss 实验结果。
- CellFM：Zeng et al., Nature Communications 16, 4679 (2025)，[论文](https://www.nature.com/articles/s41467-025-59926-5)，Methods Eq.10–12。掩码表达重建与CLS表达重建，不是CLS细胞类型分类损失。
- 官方代码冻结版本 `72c9f4a9580a3716058c184900ed14a65151ed8f`：[model.py](https://github.com/biomed-AI/CellFM/blob/72c9f4a9580a3716058c184900ed14a65151ed8f/model.py#L147-L152)、[MaskedMSE](https://github.com/biomed-AI/CellFM/blob/72c9f4a9580a3716058c184900ed14a65151ed8f/loss_function.py#L31-L44)，作为设计来源。
- CellFM 官方遗传扰动 GEARS tutorial 的 [loss_fct](https://github.com/biomed-AI/CellFM/blob/72c9f4a9580a3716058c184900ed14a65151ed8f/tutorials/Perturbation/gears/utils.py#L252-L277)：gamma=2 时四次方、按条件计算，并带过滤和 hard-sign 项。L1/L3 **只借鉴误差形式**，L2/L3只借鉴条件权重思想；不声称复刻全部下游 loss。
- 不加入官方 hard-sign 方向项：sign 的通常梯度为0，不能把它当成有效可微方向监督。若另设计平滑方向项，必须独立命名和测试。
- 本次重用前序已审读的论文/冻结代码证据，未重新检查网页更新或论文完整性公告。官方 CellFM 源码许可边界保留；未来采用原生方程实现，不复制或重命名官方源码，不引入上游 runtime。

## 10. 待填结果台账

| ID | 状态 | training SHA | evaluation SHA | run ID | checkpoint/epoch | primary | CI | 六项/seen/unseen/MSE/成本收据 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | IMPLEMENTED_LOCAL | 待发布（本文件同版本） | 待评价 | 无 | 无 | 无 | 无 | 无 |
| L1 | IMPLEMENTED_LOCAL | 待发布（本文件同版本） | 待评价 | 无 | 无 | 无 | 无 | 无 |
| L2 | IMPLEMENTED_LOCAL | 待发布（本文件同版本） | 待评价 | 无 | 无 | 无 | 无 | 无 |
| L3 | IMPLEMENTED_LOCAL | 待发布（本文件同版本） | 待评价 | 无 | 无 | 无 | 无 | 无 |
| M1 | IMPLEMENTED_LOCAL | 待发布（本文件同版本） | 待评价 | 无 | 无 | 无 | 无 | 无 |
| M2 | IMPLEMENTED_LOCAL | 待发布（本文件同版本） | 待评价 | 无 | 无 | 无 | 无 | 无 |

## Implementation receipt, weight1 (2026-10-07)

The native heads are initialized after the main model and Teacher, using a private CPU RNG. The gene scalar decoder uses the existing scalar-output AdamW route; CLS matrices use Muon. M1 CLS parameters are inactive, and neither Teacher contains auxiliary parameters.

Fixed diagnostics use the first four rows of a privately sampled training batch, at initialization and epoch boundaries in eval mode. The gradient scope is shared Cell, expression and control CLS parameters. Diagnostics do not update optimizer, EMA or centers, and preserve the main RNG. Timing separates training and diagnostics; peak memory is explicitly rank0 only.

The frozen parent and candidate L0 matched bit-for-bit across two nonzero-LR relay updates in the same CPU environment: main model, Teacher, centers, original metrics and Torch RNG. See the loss-l0-parity receipt. Weight1 tests cover M1/M2 multistep updates, shared encoder and active-head gradients, hidden-value exclusion, checkpoint/optimizer/RNG continuation, recomputation and evaluation loading. Changing auxiliary heads leaves inference output identical. Scientific results remain absent until formal runs and final tests complete.
