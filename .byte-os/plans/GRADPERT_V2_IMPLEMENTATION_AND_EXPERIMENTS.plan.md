# GraD-Pert v2：方法完成、双5090压力测试与分组实验计划

日期：2026-09-21。状态：用户已授权实施，当前实现及容量测试进行中；完成证据见 V2_IMPLEMENTATION_STATE.md。
本计划承接用户明确要求的后续顺序，不把早期“仅讨论”永久解释为禁止未来实现。
实施依据用户后续明确执行授权。本计划不更改正在运行的v1任务。

## 1. 目标、设计来源与执行顺序

交付可在config选择v1/v2的独立GraD-Pert v2方法；测定两张RTX5090的实际容量与吞吐；
生成按类型组织、可单组启动的消融配置和脚本；完成基线及分组消融的50轮best/last评估。

方法来源：[设计文档](../../docs/design/GRADPERT_V2.md)。
用户指定视觉基准：[总图](../../overview-design/2026-09-21-gradpert-v2-mechanisms/gradpert_v2_overall.png)，
[可编辑源](../../overview-design/2026-09-21-gradpert-v2-mechanisms/gradpert_v2_overall.drawio)。
图中星号属于阶段A必须补全的规格，不能由绘图推断为已冻结实现。

| 阶段 | 交付 | 退出条件 |
|---|---|---|
| A 方法定稿 | 运算、张量、视图、损失、优化器的完整规格 | 所有实现所需星号有明确值/规则、证据与候选标记 |
| B 完整实现 | v2模块、公共训练流程适配、版本选择、恢复与评估 | 合成契约测试和v1兼容回归通过，源码发布可追溯 |
| C 双5090压力测试 | 容量表、吞吐表、执行配置建议 | 完整训练链持续稳定，保存/恢复/评估可运行 |
| D 实验包 | 分组矩阵、独立YAML、运行/检查/汇总脚本 | 所有config验证和队列dry-run通过 |
| E 基线 | 默认v2锚点、v1及外部/非学习基线 | 50轮best/last收据完整或明确标缺失 |
| F 分组消融 | 先超参数，再loss、结构、采样、泛化 | 每组按预注册因子比较，验证集决策，最终报告 |

不预估未经实测的GPU小时数。C阶段按step耗时×预计更新数给出资源预算，D据此封存矩阵。

## 2. v1冻结与版本选择设计

已检查当前入口：`config/schema.py`、`config/native.py`、`execution/native.py`及
`modeling/modules.py`；现有原生身份为`gradpert_b2`，已有vNext配置不能被重新解释成v2。

建议新增`model.version`选择器（v1/v2），v2身份为`gradpert_v2`；两者必须匹配：

```yaml
# 接口提案，仅展示版本选择；不是现在就能运行的完整config
model_id: gradpert_v2
model:
  model_id: gradpert_v2
  version: v2
  # family、source及完整带来源parameters由阶段D写全
```

- 新v1配置可显式`model.version: v1`，身份仍为`gradpert_b2`。
- 旧v1/vNext YAML无version时按原逻辑解析；不得改写已有文件、哈希、参数默认或manifest。
  新增字段不能进入旧配置的历史canonical哈希计算；通过版本化序列化保持一致。
- v2必须显式选择，独立参数schema，未知参数/版本组合报错，禁止失败后静默降级v1。
- 新增`modeling/v2/`、`training/v2/`、`config/v2.py`，必要时增加`execution/v2.py`。
  公共CLI只做薄分派；复用已有数据/评估/收据接口，避免在v1训练循环遍布v2条件判断。
- 2026-09-21 用户进一步明确：v2 基于 v1 公共工程流程开发，不另建平行实验框架。
  数据集条件划分、GEARS 可表示性交集、300-control 清单、数据读取、指标定义、
  主入口和来源校验直接复用既有实现；不得重新划分或仅以相同随机种子代替清单一致。
  新增逻辑限于 v2 模型、输入/视图处理、采样、损失及其必要的运行适配。
  已新增的生命周期、best/last 评估编排和曲线输出需逐项审查重复实现，优先使用公共接口；
  因双教师状态、双卡同步等差异需要保留的部分，应收敛为适配层并记录理由。
  此项复用审查未完成，不能将当前独立生命周期标为最终工程交付。
  评估采用 metrics_only，保留服务器 best.pt/last.pt 与小型指标、曲线、复现收据；不持久化预测矩阵或 PKL。
- v1模型数学行为、loss、optimizer、checkpoint键、默认训练/推理保持不变。
  允许兼容性入口改动，但必须证明旧配置解析、固定输入输出/梯度和恢复路径不变。
- checkpoint显式保存model_version、architecture/config hash、完整训练状态；v1/v2
  互相误加载必须拒绝。旧v1 checkpoint沿原路径继续可读，不自动升级。
- 数据、图轴或缓存配方改变时使用新的内容哈希/路径，不覆盖v1缓存与结果。

实施前先核对当前dirty工作树归属，保护已提交并推送的前置基线，不将其他任务修改打包。
独立目录优先；共享入口存在重叠时使用隔离checkout或先处理归属。每阶段代码通过检查后
按项目规则发布main，正式服务器工作使用相同clean commit，绝不修改活跃checkout。

## 3. A：方法规格补全

| 组件 | 已确认内容 | 实施前必须补全 |
|---|---|---|
| 基因输入/图读取 | GenePT可训练表；GO/STRING Top20+expander+self；图mask在得分后Softmax前 | 基因轴/映射、来源融合与去重、来源偏置、图层数、norm/残差、空邻域规则 |
| Cell Encoder | gene+expr逐元素相加；末尾可学习control CLS；d256、4层/4头/FFN4d | ExprMLP形状、激活、norm、token顺序和padding |
| KDA/MLA/mHC | 每编码器3 KDA+1非因果MLA；不加短因果卷积/MoE/四token池化 | 原论文/冻结代码证据、门控/状态更新、MLA rank、mHC流数及初始化、算子数值范围 |
| Response Encoder | Hc与广播ep拼接再投影；独立response CLS；Δx残差预测 | 多靶点聚合、输出头、control CLS是否仅诊断输出、逐基因输出轴 |
| 蒸馏① | ep/节点；权重.8/.4/.1；默认按不同条件平均 | 原condition/masked-node/spread精确定义，能否称iBOT/KoLeo，头共享策略 |
| 蒸馏② | response CLS/token；DINO/iBOT/KoLeo=.8/.4/.1 | λ2=.1仍为起点候选；视图分母、匹配位置、温度/center、KoLeo集合 |
| 联合训练 | 主预测分支才有真实表达监督；ep不detach；Teacher EMA/no_grad | EMA覆盖模块、更新时序、loss分母、分布式/累积语义 |
| 优化/日程 | split Muon；LR1e-3默认，对比1e-4；WD0对比1e-5；dropout.1 | 逐参数分组及Muon映射；GLM5对齐的确切日程/冻结来源，禁止猜warmup比例 |

全方法必须包含SSL开/关路径，不先用缺失蒸馏的模型宣称完整v2已实现。
默认G/L=2/4、mask比例、mHC流数/MLA rank等保留候选身份直到本阶段记录明确选择。
表达mask在融合前替换表达嵌入；Teacher不访问真实目标表达；mask基因不能从缓存旁路泄漏。
固定4头做d128/256对照，核对两种head dimension的内核支持；不把头数一起改掉。

验收：形成逐模块输入/输出shape表、前向/反向图、teacher更新顺序、推理配方及参数来源表。
若官方内核不适配5090、非因果或小维度，先做等价参考算子验证；不悄悄替换为另一种机制。

## 4. B：完整工程实现与必要验证

依次实现：版本schema/工厂 → 图读取 → Cell/Response及混合层 → 双蒸馏/EMA →
优化器/日程 → 完整train/val/infer → checkpoint恢复/评估/收据 → CLI分派。

关键测试与验收：
- 固定小图上稀疏注意力与masked-dense参考的输出/梯度一致；邻域外权重为零。
- control CLS、ep、response CLS区分；表达mask泄漏测试；Teacher无梯度，Student
  预测梯度可到ep；每optimizer step仅一次EMA/center约定更新。
- 四种SSL开关均可完成训练、验证、保存、恢复及best/last测试；关闭loss时不残留对应项。
- KDA小序列数值参考、MLA非因果可达性、mHC残差形状/归一化；bf16误差与fp32参考对照。
- view/token/padding匹配与loss分母；单条件/重复样本/空mask的合法行为。
- 累积/DDP逐项验证可加loss的尺度；KoLeo不宣称跨microbatch等价，记录真实近邻集合。
- 小合成样本可过拟合；同状态恢复的下一步输出/loss/LR/EMA一致到预定数值容差。
- v1旧配置哈希、模型构造、固定输入预测/梯度、checkpoint恢复和默认CLI保持原行为。

运行相关单元/契约测试及项目lint、类型与build检查。记录原有失败与本次回归，不能归零掩盖。
GPU内核正确性与真实数据集成在服务器完成；本地只用合成数据，不拉取科学矩阵。

## 5. C：两张RTX5090压力测试（工程容量，不是科学消融）

开测前读取实时型号/显存/占用/驱动环境，记录GPU UUID与外部进程；本计划不承诺空卡。
两卡DDP各自持有模型，不把显存加成一张大卡；默认不引入分片训练。
所有CUDA启动必须设置`PYTORCH_ALLOC_CONF=expandable_segments:True`。
保留当前资源门禁，不复用文档里未经刷新确认的历史40%配额；需要独占时先确保资源许可。

### C1 正确性与容量搜索

1. d256、query1000、完整双SSL、2G/4L，使用单个同步双卡分布式任务，world_size=2、accumulation=1；
   按每卡 physical microbatch 逐级搜索并细化边界，单卡数据不能确定正式双卡默认值。
   前一点成功才加大。临界OOM独立子进程退出，回退/细搜，不影响其他任务。
2. 首步必须含前向、反向、optimizer、EMA和center；成功后才做至少128连续更新的稳定性段。
   记录初始优化器状态分配峰值、峰值allocated/reserved、step分位耗时、cells/s、
   各view/token数量、nonfinite、梯度、CPU内存及数据加载等待。
3. 在最大稳定点和保守运行点重复检验，选择稳定吞吐配置，不直接用OOM边缘点正式训练。
4. 同样探测prediction-only、SSL1-only、SSL2-only，避免按轻量路径容量安排完整路径。
5. 单独探测训练/验证/300-control推理与best/last评估；更多基因推理按增加token测试，
   与训练microbatch结果分开。恢复后继续训练，排查EMA、缓存、日志的持续内存增长。

### C2 双卡拓扑与系统开关

- 用户已明确采用单任务2卡同步分布式训练；默认容量、吞吐和H3均依据此拓扑，记录通信时间。
- 每卡2任务只在两进程峰值有余量、现有门禁允许时测试；用总完成时间决定是否值得。
- bf16、activation checkpointing、视图串行分次反传作为工程开关；比较同数学目标，
  不因省显存减少view数或偷换KoLeo集合。EMA固定在所有梯度更新完成后。
- batch消融水平由本阶段实测产生，不预设512/1024/2048。先报告各拓扑最大稳定microbatch与
  高吞吐运行点，再给出可行microbatch/world_size/accumulation组合及对应B_eff。
  B_eff=单卡microbatch×DDP world_size×累积；独立两任务不属于同一个batch。
- 累积只保证适用的可加损失尺度；KoLeo/center/DDP统计必须明确。不能容纳所需近邻集合
  时报告限制，禁止把梯度累积称为已实现全batch KoLeo。

输出：`capacity.json/csv`、`throughput.csv`、错误收据、所选execution profile及其配置哈希。
至少在首个开发数据集走通训练→验证→保存→评估；其余已集成数据集逐个容量检查。
新数据集沿用项目的一epoch集成门槛；压力测试与少量更新不算50轮科学结果。

## 6. D：分组消融矩阵与运行顺序

默认以Nadig Jurkat作消融数据集（用户已确认）；固定seed1筛选，
关键对比用1/2/3配对复核。五个既定数据集用同一冻结条件split与300-control清单。
每组开始前冻结父配置；组内每行只改声明因子，不把上一行赢家实时滚动作为下一行父值。
跨组父配置变化必须记录依赖和parent hash。测试结果不参与挑选。

| 组 | 类型/优先级 | 水平与建议行数 | 固定项/解释 |
|---|---|---|---|
| B0 | 基线，先跑 | v2 prediction-only、v2完整双SSL；v1参照及外部/非学习基线 | B0双SSL为主要消融锚点，权重/外部λ先冻结 |
| H1 | 超参数，第一组 | LR 1e-3/1e-4（2行） | 其余固定，项目 warmup+cosine（16% warmup，末值/峰值0.2）；非 GLM5 官方日程 |
| H2 | 超参数，第二组 | WD 0/1e-5（2行） | 采用H1验证决策，记录父配置 |
| H3 | 超参数，第三组 | C阶段实测后冻结2–3个可行batch档位 | 固定LR，不自动线性缩放；50epoch导致更新数不同须报告 |
| P1 | 模型参数 | d256/128（2行） | 固定4层/4头/FFN4d、rank规则与相同数据流 |
| L0 | loss主效应 | 无SSL/仅SSL1/仅SSL2/双SSL（4行） | 相同主预测结构；B0同配置可复用 |
| L1 | SSL1内部组件 | condition/node/spread三个二值开关共8组合 | SSL2关；000与L0重合；标注node/spread真实定义 |
| L2 | SSL2内部组件 | DINO/iBOT/KoLeo三个二值开关共8组合 | SSL1关；000与L0重合；保留单独正则组的研究边界 |
| L3 | loss权重，后续可选 | λ1=.3/1/3；λ2=.03/.1/.3 | 一次变一个外部λ，内部.8/.4/.1保持；不是首轮超参数组 |
| A1 | token交互 | 全token vs匹配输入先验的逐基因MLP | 披露参数量/计算量，保留相同监督与输出轴 |
| A2 | 注意力机制 | 3KDA+1MLA；3普通注意力+1MLA；3KDA+1普通注意力；全普通注意力 | 4行分别分离两种机制；普通注意力的可达性/因果性按替换位置匹配 |
| A3 | 残差机制 | mHC开/普通残差（2行） | 固定主宽度；额外参数量单独报告 |
| A4 | 初始化 | GenePT/random（2行） | 均可训练，同维adapter和图先验 |
| S1 | 平均策略 | 不同条件平均/按样本行平均（2行） | 固定完全相同样本流，仅改SSL1分母 |
| S2 | batch组织 | 多条件KoLeo关/单条件B细胞KoLeo关/多条件KoLeo开（3行） | 前两行隔离采样，第一与第三隔离KoLeo；单条件默认关 |
| G1 | 泛化 | 同token预算的可见/留出表达列；同评估轴的1000/更多/全量上下文 | 两轴分别报告，不混成一个因素 |
| D1 | 诊断 | 固定p换control；固定control换p；阻断response CLS→gene | 可先用已有checkpoint评估；重训阻断另列，不混淆两种结果 |

首轮H1/H2各2个配置坐标，H3按压力测试冻结2–3档；父配置一致且配置完全相同的锚点可去重；随后P1。
H组建议在完整双SSL模型上筛选，以免只优化prediction-only而不适用于最终方法。
H3必须等C完成后再生成数值配置，未测试的档位不入队。以稳定且高吞吐的B_ref为参考，
优先选一个较小档、参考档和一个更大可行档；比例不预设，较大档不可行就保留两档。
B_ref同时用于H1/H2等其他组的固定batch。不能因为梯度累积理论上能放大B_eff，就将
未经完整目标验证的大batch当成压力测试通过。容量profile必须绑定完整双SSL、视图、
数据集/token规模；跨数据集不可直接照搬，显存不同需复测并披露执行差异。
H3固定KoLeo近邻集合/center更新约定，若无法保持则标为联合变化，不声称纯batch因果。
若要比较物理microbatch本身，在C中固定B_eff作为工程吞吐比较，与H3分开。
不做warmup时长、warmup-only/cosine-only、dropout、层数、头数、FFN倍率消融；
图默认GO+STRING+expander，暂不做图来源开关消融。其他views/温度大范围搜索暂不入队。

组L/A/S等在超参数冻结后运行，不展开跨组笛卡尔积。全组件组合、完整双SSL交互可在
单组结论后追加，需新矩阵版本，不能在观察测试分数后修改原行。

## 7. 配置、脚本与结果交付布局（计划路径，尚未生成）

```text
configs/v2/<group>/<variant>/<dataset>.yaml   # 每个文件完整独立，无extends/隐藏继承
experiments/v2/matrix.yaml                   # 分组、依赖、种子、父配置、角色
experiments/v2/registry.json                 # resolved config hash、来源及去重身份
scripts/v2/generate_configs.py               # 从已冻结矩阵生成完整YAML
scripts/v2/validate_matrix.py                # schema/单因素差异/身份/重复运行检查
scripts/v2/capacity_probe.py                 # C阶段，接入既有资源与源码门禁
scripts/v2/run_group.py                      # 默认dry-run，按组/数据集/seed明确启动
scripts/v2/collect_results.py                # 收据驱动汇总、曲线和缺失项
```

示意命令（阶段D实现后才可用）：

```bash
python scripts/v2/validate_matrix.py --matrix experiments/v2/matrix.yaml
python scripts/v2/run_group.py --group H1 --dataset nadig_jurkat --seeds 1 --dry-run
python scripts/v2/run_group.py --group H1 --dataset nadig_jurkat --seeds 1 --launch
```

每份config携带版本、模块、全部loss权重、所有优化组、日程、视图、采样、执行profile、
评估配方与参数来源。保留现有SourcedValue约定；结构化参数采用独立严格schema。
矩阵差异检查忽略run ID/说明字段后验证唯一科学因子；工程microbatch等变化显式记录。
脚本必须可重入，启动前去重/锁定；只恢复同一run ID的匹配状态，不覆盖完成/失败证据。
依赖组完成后输出验证集选择记录，再物化下一组完整配置；未来组未定父值不能伪装ready。

## 8. E/F：基线、正式运行与科学验收

- 默认锚点先跑，随后H1→H2→H3→P1；冻结超参数后运行L→A→S→G/D。
- 外部基线先核查既有结果收据；同split、表达/评估轴、300-control、训练轮数与源码证据
  才可作为正式参照，否则新run ID重跑。旧v1模型不改，但其数据兼容性必须如实披露。
- 范围为v1、GEARS、TxPert、已授权可用的Scouter及现有非学习基线；Scouter适配状态先查，
  未接入不假装能运行。外部模型只调用冻结官方实现与其配置，不用v2超参数替代。
- 公平性主基准维持现有可比任务；表达列留出G1单独定义评估任务，不把不支持的外部模型
  强行搬到不同输入任务后仍称公平。所有必要数据和预处理派生物有版本与哈希。
- 学习模型正式50epoch、不提前停止；best按验证预测loss，保留epoch50 last；每行结束即
  安排两checkpoint测试，不等全组完成。每epoch保存loss、支持的三类Pearson与曲线。
- 超参数只看验证集；即使测试自动落盘也不参与下一组选择。选择程序不读取测试目录。
- 所有训练/图构建/真实推理在`/data/yilangliu`；local/GitHub/server clean SHA一致。
  训练与评估SHA、checkpoint SHA256/epoch/role、配置/数据/split/ordered rows/environment
  哈希一并记录；数据和权重不拉回本地，结果同步限小文件。
- 复用原生零PKL/外部临时PKL清理和结果收据契约；失败、OOM、缺测明确记录。
- v2调度采用用户指定的单任务同步双卡拓扑，每个任务占用两卡；各消融行顺序执行。
  外部基线保留官方模型配置与单任务拓扑，并遵守实时资源门禁。已有实验不抢占、不杀停。

最后交付：按类型的结果表、每组配置差异表、best/last训练曲线、配对种子统计、
模型参数/显存/吞吐成本表、全部收据链接，以及哪些方法结论被实验支持/未被支持。

## 9. 当前可确认与下一步

截至代码 `173f81b7cc7f083497d86398f94f10d658d7e049`，v2 模型、联合损失、版本分派、
恢复与评估适配及分组工具已经实现，并有合成测试和真实数据容量收据。不能继续将当前状态
描述为“仅有计划”，也不能据此宣称完成整个方法交付。

验收分项见 [V2_ACCEPTANCE.md](../V2_ACCEPTANCE.md)。当前优先项为外层训练/评估适配
审查、完整验证集生命周期检查、实测 batch 冻结与 H3 配置、G1/D1 运行入口及多数据集预检。
尚未启动正式50轮基线/消融，也未移交 ZCode。容量探针只证明收据覆盖的工程行为。
