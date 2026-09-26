# GraD-Pert v2 邻域接力方法与交付计划

状态：2026-09-26 单向扫描修改通过本地验证；用户已明确恢复原性能工程目标。当前配置为 `configs/v2/single_pass_jurkat/gradpert_v2/nadig_jurkat.yaml`，显式 `relay_passes: 1`。此前 `relay_jurkat` 双遍配置及其性能记录保留为历史；缺省字段仍解释为历史双遍，保证旧 checkpoint 配置身份不变。除扫描遍数外，新旧自包含配置完全相同。更新后须重新测性能，不能用旧双遍收据证明新方法吞吐或容量。

## 方法合同

输入采用固定的 6506 个基因 ID 轴与可训练的 GenePT-PCA256 表。ID、表达、图边、扰动靶点和预测标签始终按 ID 对齐；随机化仅作用于 KDA 扫描顺序。训练表达采样默认排除测试扰动靶基因，先验节点与边保留，测试允许这些基因的 control 表达。

图边是 GO/STRING 各入边 Top20、三个种子 1 的双向 Hamiltonian 环和 self 的去重并集，保存四位来源。无虚拟节点。图层 1–3：对**每个目标节点 i**，只在其合法邻域 N(i) 内抽取随机排列，按该排列单向写入状态，不再反向接力。完成一遍后仅目标 i 用自己的 query 读取末态，得到唯一的新 h_i；邻居在该局部扫描里只负责写入。每层的所有目标同步更新，下一层读取上一层的新节点值。第 4 层是仅对 N(i) 读取的稀疏 MLA，保留边来源得分偏置。无全图共享 KDA 状态。图局部视图对节点集合取诱导子图；采样和邻域展开不得泄漏到局部视图外。

Cell 和 Response 各为 2 个单遍末态读取 self-KDA 块加一个全量、非因果 MLA 块，宽 256、4 头、MLA KV 秩 64、4 个 mHC stream、SwiGLU FFN、dropout 0.1。每个 self-KDA 块随机排列基因，单向写入所有基因，末尾 CLS 写入一次；全部基因及 CLS 从同一末态读出。下一层状态重新置零。训练时每个视图/样本抽取排列，Cell、Response 和控制交叉读取使用一致的 ID 对齐；验证/测试用固定种子排列并记录种子。随机增强不是数学排列不变性，需报告重排敏感性。

Response 从 Cell 基因状态和独立 response CLS 开始。每层首先通过各层独立的 Linear(512,256)→GELU→Linear(256,256) 将每个 stream 与扰动图表示 e_p 融合，替换入口状态而不另加入口融合残差，再 self，再读取 control 基因状态的 cross，最后 FFN；前两层 cross 使用单遍 KDA 的 control K/V 写入与 response Q 末态读取，末层用全量 cross MLA。control CLS 不作为 cross K/V。响应末态基因表示与 e_p 再拼接预测 Δ，yhat=x_control+Δ；SSL2 使用最终 response CLS，SSL1 使用 e_p。不得把扰动真实表达输入预测路径。

训练仍为完整 B0：主视图 1000 个表达基因上预测 MSE，SSL1 图 2 Global+2 Local，SSL2 表达 2 Global+2 Local，两套 Global 比例均为 60%–90%、Local 为 25%–50%，取 round 后的节点数；SSL1 分母是图节点、两种 Local 分别从 GO/STRING 扩展取节点诱导子图，SSL2 分母是本次1000查询基因、两种 Local 都是随机基因子集。图视图强制包含本次条件靶点（可能使极小图上的比例超出目标）；教师只读未遮蔽 Global，Student 读全部四视图。四个投影头均为 256→2048→256→8192。权重：MSE + 1×(0.8 condition + 0.4 node + 0.1 spread) + 0.1×(0.8 DINO + 0.4 iBOT + 0.1 KoLeo)。默认统一 row_mean；condition_mean 是整套 S1 消融。图 node、spread 和教师 center 按现有例外。KoLeo 的跨双卡/累积窗口候选排除**同一扰动条件**，至少两个非 control 条件进入每个有效比较集合，不设最多 8 条件。

## 单遍覆盖规则（2026-09-26）

每个 KDA 层从零状态开始，按训练随机排列执行一次原有衰减与 delta 写入；self 分支随后写 CLS 一次。最终统一用 `y_g = S_*^T q_g` 读取，不回退为逐 token 中间状态输出。图分支仍为每个目标的独立合法邻域状态，仅目标 query 读取，无 CLS；cross 分支仅 control 基因写入，response query 读取。层数、参数量、随机顺序、图边、所有损失与 EMA Teacher 同构规则不变。下文性能工作恢复，旧的“双遍/接力”测量只作历史证据。

## 总交付与判定原则

交付一个方法完整、版本可追溯的 v2，随后提供同方法、同训练协议下的优化前后实测报告、推荐双卡运行配置和最高持续验证容量，最后运行完整 B0 五轮。历史模型的 batch192 和吞吐不充当新方法的基线。

主要优化目标是**完成相同训练工作所需的总时间**：分别报告 cells/s、optimizer updates/s、训练一轮时间、验证时间、五轮加 best/last 测试总时间估计。GPU utilization 是诊断信号，不能替代吞吐；不预设加速比例，不承诺完全消除有依赖关系的空闲。

方法正确性基线、性能参考实现和最终运行配置各有独立身份。优化比较先固定有效全局 batch、微批次、累积次数、样本/视图和数值精度；容量搜索另列实验，避免把更大的 batch 当成算子优化收益。

## 阶段 A：完成模型升级，冻结可比较的参考版本

1. 把上面的合同逐项对应到实现、配置和测试。确认图邻域内写入、仅目标读取、四层同步传播；Cell/Response 的单遍写入、CLS 写入一次、每层清零；逐层扰动注入、self→cross→FFN、独立 response CLS、输出残差头。对“融合 MLP”与当前线性残差注入等实现差异逐项核对，不把能运行等同于符合设计。
2. 核对两套蒸馏的视图数、裁剪比例、mask、配对和分母；明确图裁剪与表达裁剪的各自采样轴。Teacher 同构 EMA、训练时顺序增强、推理确定性、center 更新频率都进入合同。实际计数 Student、Teacher、各子模块参数量和优化器状态规模。
3. 随机顺序只重排写入，不改变基因 ID 对齐。验证同一细胞推理在不同评估 batch 分块下的一致性，并把种子写入配置/收据。随机训练本身不能证明排列不变；额外报告固定样本多排列的输出波动，作为模型性质而非优化收益。
4. 所有适用 loss 统一 row_mean/condition_mean；两卡条件计数和 KoLeo 同条件排除、累积窗口候选集合保持一致。loss 协议必须独立于注意力变体，避免注意力消融顺带改变 KoLeo。随机采样无条件数上限，保证有效全局 batch 至少两个扰动条件；尾批不得偷偷超过容量配置，记录覆盖率和任何尾批处理。
5. 完成训练一步/多步、数值和全参数梯度、EMA/center 更新、checkpoint 恢复、best/last 评估接口及 v1/历史 v2 兼容检查。包括 padding、单节点邻域、变长视图、不同条件计数及失败恢复。小合成训练用于查错，不能证明真实数据模型效果。
6. 核实最终测试结果；分目录执行以避开既有测试模块同名冲突。对类型检查失败区分环境、原有错误与新增错误。代码审查后定向提交推送，验证本地/GitHub/服务器不可变源码一致。此时只完成方法参考版本，不宣称性能优化已完成。

**退出证据：**需求—实现—测试对应表、真实参数统计、测试日志及限制、干净完整 Git SHA、自包含配置 hash。未达到这一点，不开始优化比较。

## 阶段 B：建立 CPU/GPU/通信的性能证据

### B1. 环境与固定工作负载

在已有双 RTX 5090 的授权范围内记录 CPU 核数/NUMA、可用内存、存储、GPU 显存、PCIe 链路、P2P 可用性、驱动/CUDA/PyTorch 和 NCCL 环境。先读实际拓扑，再判断通信可能走的路径；必要的小型带宽测试独立运行，不与模型计时互相干扰。

从训练划分固定一组真实 batch 与视图计划，覆盖常见和较大条件数、600–900 的表达 Global、250–500 的 Local、不同邻域长度以及尾批。所有大数据、trace、checkpoint 留在服务器。先用保守微批次跑完整预测+SSL1+SSL2，短预热后取得可重复的参考耗时，不从旧 batch192 起步假定可容纳。

### B2. 三层测量，避免计时本身制造瓶颈

| 层级 | 必须回答 | 测量与产物 |
| --- | --- | --- |
| 端到端 | 同样细胞和更新量需要多久 | 预热后整段 wall time、cells/s、update 中位数/p95、峰值 allocated/reserved、CPU RSS；包含数据等待和必要通信 |
| 阶段时间线 | CPU、两卡及通信在哪里互相等待 | CPU 采样/索引/视图、H2D、图编码、Cell、Response self/cross、四头投影、SSL1/SSL2、KoLeo gather、backward、梯度归约、optimizer、EMA/center、验证和保存 |
| 热点算子 | 哪个具体 kernel/分配/主机调用值得改 | 短窗 torch.profiler；按服务器可用工具深入系统/GPU trace，统计 launch 数、同步点、拷贝量、内存流量、矩阵尺寸及分配生命周期 |

使用阶段标签、CUDA events 和 CPU/GPU trace 关联异步工作。正式吞吐测试不在每个子步骤插入全设备 synchronize；诊断性的串行分段计时单独报告。Profiler 的运行与无 profiler 的吞吐对照分开，说明采样开销。

每张卡分别记录，不只报告平均 GPU 利用率。分解空闲为：CPU 供给不足、H2D 等待、主机同步、短 kernel 发射间隙、卡间负载不均、通信等待、验证/保存停顿。报告临界路径及热点占比，按可节省的端到端时间排序。当前 `capacity_probe.py` 的聚合 data_wait 只是线索，不足以完成这项诊断。

**退出证据：**一份参考配置、重复计时结果、CPU/双卡时间线、内存生命周期图、按成本排序的瓶颈表。每个优化项都必须对应一个已测到的瓶颈。

## 阶段 C：按瓶颈推进深度优化

每个候选依次执行“假设 → 最小改动 → 正确性对照 → 相同工作负载实测 → 保留或回退”。以下是候选清单，不代表全都启用。

| 方向 | 重点工作 | 必须保持的语义 |
| --- | --- | --- |
| CPU 数据与图准备 | 找出 Python 循环、重复邻域构建和索引转换；预存静态邻接/来源/映射，批量采样；测 worker 数与 BLAS/torch 线程竞争，设置有界预取 | train split、ID 顺序对应、每次视图抽样、edge dropout 和 mask 分布不变；不缓存会随参数更新的图表示 |
| CPU/GPU 流水调度 | CPU 准备 batch n+1 同时 GPU 算 n；pinned staging、有界双缓冲、异步 H2D 与独立 copy stream；events 标记数据可用和缓冲回收 | tensor 生命周期和 stream 依赖正确；不覆盖 GPU 正在读的缓冲；CPU 内存/队列不无限增长 |
| KDA chunk 与末态计算 | 对短邻域和长表达序列分别测 chunk16/32/64/128；只求最终 S，批量处理独立邻域，减少 padding、临时矩阵和小 kernel；先测原生向量化/编译，必要时再实现融合 kernel | 正向→逆向真实状态接力、FP32 状态、衰减/delta 更新、CLS 规则、来源信息、所有参数梯度及输入梯度保持一致 |
| 图/编码器重复计算 | 统计每个微批/视图实际重算次数；识别相同输入和参数时才可共享的投影/邻接计算；评估向量化和形状分桶 | 不跨 optimizer/EMA 更新缓存表示，不复用本应独立的随机视图、顺序、dropout；分桶 padding 不成为有效 token |
| 激活与反向调度 | 绘制累积期间的 autograd 保存量；比较整层/子层/邻域分块 checkpoint，释放真实不再依赖的中间结果；评估精确重算减少保留图的方案 | KoLeo 仍在同一完整跨卡累积集合选异条件近邻，并保留远端被选邻居的梯度；不得 detach 或缩小候选集合换速度 |
| 双卡通信 | 测梯度归约、KoLeo gather/backward、center 及指标 collectives 的占比；合并小通信、减少冗余统计交换；若有收益，再设计梯度分桶与计算重叠 | 当前是自定义归约，不能直接套 DDP no_sync；先确认梯度何时最终就绪，特别是 cell 和 graph 两路共享参数，保持全局权重和更新顺序 |
| 编译和 launch 开销 | 只在 trace 显示热点时测 torch.compile、合法形状桶或 CUDA Graph 候选；记录首轮编译成本、重编译和 graph break | 保持随机增强，不能固化一组 mask/排列；五轮总耗时需覆盖编译成本，不能只报稳态最快 kernel |
| 验证和小产物 | 单独 profile 300-control 推理、指标与 checkpoint 写入；允许将不可变 CPU 快照后的序列化工作重叠 | frozen control/truth ID、best/last 选择和保存一致；检查点必须原子完成，不能后台线程读取正在更新的参数 |

调度示意：

```text
CPU：准备 n+1 ───── 准备 n+2 ───── 准备 n+3
H2D：      拷贝 n+1 ───── 拷贝 n+2
GPU：计算 n ───────── 计算 n+1 ───────── 计算 n+2
```

只把互相独立且资源允许的工作放到空隙中。Teacher 与 Student 共享 GPU 资源，另开 stream/进程未必提高吞吐；优先修复本训练流水线，评估进程并发时以总吞吐和显存余量为依据。不会为了“填满 GPU”擅自启动其他消融。

## 阶段 D：证明提速且未偷改训练目标

- **算子层：**保存同输入、同随机计划；比较前向、最终状态和全参数/输入梯度。FP32 严格参照，BF16 单独记录误差；阈值在查看候选收益前按参考实现数值波动确定，失败不事后放宽。
- **更新层：**同一 checkpoint、样本、视图、mask 和 RNG 计划运行相同优化步，比较每项 loss、梯度、Student/Teacher 参数、优化器状态、center 与恢复后的连续训练。涉及 checkpoint/随机数的调度尤其检查 RNG 状态；允许的浮点重排差异需单列。
- **性能层：**同双卡独占时段串行 A/B/B/A 比较，初始预热不少于 10 次更新并确认稳定；每段默认至少 30 次测量更新，两轮 ABBA。时长过高时在实验前记录调整后的预算与置信限制。报告每段结果、差值范围、中位数/p95、峰值显存和端到端 cells/s，不能只挑最快一次。
- **实际保留：**模块 microbenchmark 加速但整步没有收益的不作为默认优化。重复测量不能排除噪声、错误变多或显存不可持续的候选不进入默认。每项收益与累计收益分开，不能简单把局部百分比相加。
- **效果边界：**短程同种子训练及验证可排查明显退化，不能证明泛化效果完全相同。改变精度、裁剪、loss、候选集合、模型层数等属于方法改变，不能混入本阶段；测试集不用来选择优化或超参数。

**退出证据：**优化前后不可变 SHA、配置/数据/环境 hashes、分项及完整更新一致性结果、重复性能表、失败/回退记录、可复现命令。没有可靠提速也如实给出瓶颈与限制。

## 阶段 E：双卡持续容量与推荐 batch

1. 使用通过 D 阶段的实现，优先固定累积 2、双卡，搜索每卡微批容量；明确全局 batch = micro × 2 卡 × accumulation。另测达到相同全局 batch 的可行组合，比较吞吐、保留激活与通信代价。
2. 短探针只缩小边界；遇到 OOM 保留收据并收紧范围。最终候选至少完成 128 次 optimizer update、checkpoint 续跑、完整 300-control 验证，以及包含更多条件/更长 Global 的合法高负载视图。观察 reserved/allocated 与 CPU RSS 是否持续增长。
3. 输出“最高持续验证值”和“建议日常训练值”两项；后者考虑显存余量和总耗时，可低于最大值。未覆盖的容量区间标为未测，旧192不再作容量证明。
4. 容量选择后的自包含配置再次做同源码双卡集成验证。batch/累积变化与纯实现加速分开归因；更新数、LR 日程和有效 KoLeo 集合按最终协议核对。

## 阶段 F：完整 B0 五轮及后续交接

方法、优化、容量和发布身份全部通过后，按此前已授权范围仅启动 Nadig Jurkat 完整 B0：预测+SSL1+SSL2、5 epoch、双卡，自动完成 best/last 测试及小收据。其他消融不随本任务自动启动。验证/测试协议复用既有生命周期，不重写主训练逻辑。

长测试阶段按用户要求每 30 分钟检查；正式五轮每 2 小时，并每次汇报。阶段结束时同次检查核对终态证据并推进已授权的下一步，更新同一个监控和 Byte 状态。遇到新修复项 C，加入剩余任务和监控依赖；复杂修复返回主动工程阶段，暂停该任务的等待监控。只有实际注册并验证的监控才称“已开启”。

当前已恢复阶段 A 的主动工程工作，尚未恢复 GPU 工作或创建定时查询。整体进度、当前阶段、验收证据和下一步只维护在 `.byte-os/STATE.md` 与现有 STATUS 中；大型日志/trace 留在服务器，文档引用小收据。


## 阶段 A 验收记录（2026-09-26）

预变更基线：`95388f5bd365ae3768080f5c2e39f1fc79e672ef`，已核对本地 HEAD 与 GitHub main 一致。可机器读取的小收据见 [relay-stage-a.json](../experiments/relay-stage-a.json)。代码提交后以包含本文的不可变提交作为方法参考版本；服务器预检单独登记。

| 合同 | 实现位置 | 新验证 |
| --- | --- | --- |
| 最终状态、真实接力、CLS 单次写入 | operators.py: chunk_delta_final_state / RelayDeltaAttention | 与逐 token FP32 参考比较状态及 k/v/衰减/门控/初态梯度；chunk1/3/32 |
| 目标专属合法邻域、同步层更新 | model.py: RelayGraphLayer / GeneGraph | selected 与全图读取一致；诱导子图外节点无影响 |
| 每层 MLP→self→cross→FFN | model.py: RelayResponseEncoder | 主损失、双蒸馏、masked expression 不变性、CLS 通路诊断 |
| Teacher、优化器、center、随机数 | engine.py / checkpoint.py | 含 dropout、累积2、全损失的 checkpoint 开/关完整更新对照与保存恢复 |
| 评估与历史兼容 | relay_order / V2Architecture.payload | 同细胞 eval 分块1/2/5一致；旧架构不新增空 seed 字段；best/last 载入 |
| 全局异条件 KoLeo | reductions.py / engine.py | 双进程远端近邻梯度；双进程完整更新各卡仅1条件仍有有效异条件候选，更新后模型/Teacher/center一致 |
| multiscale 与随机混合 batch | views.py / training/data.py | 图裁剪/靶点保留/诱导边；尾批不超容量、每行覆盖一次 |

Student 实际 **27,279,662** 个参数；同构 EMA Teacher **27,279,662**（冻结，不交给优化器）。Graph 6,085,672；expression 66,816；Cell 3,415,154；Response 4,987,539；prediction 131,585；四个投影头各3,148,032；另有3×256个直接 token 参数。优化器实际状态内存以后续实测为准。

实现边界：正向只读 CLS 的中间诊断输出不用于任何目标，故不物化；其不写状态的语义保持。推理种子显式为1，同层同视图采用与 cell 分块无关的排列。新 relay 图的 mask token 改为 normal(std=.02) 初始化，避免零 query 多层 normalize 造成非有限梯度；旧模型初始化不变。KoLeo 排除同条件通过独立显式参数控制，不再由注意力类型推断。新父配置下旧 A1/A2 注意力开关会连带移除 self/cross 架构，因此生成器明确拒绝，待保留架构的单因素设计后再启用；不影响完整 B0。

验证结果：v2全套253通过，随后新增双进程完整更新1项通过（合计254个不同用例）；历史目录1039通过、7跳过、5失败。基线独立快照复现完全相同5项失败，涉及旧配置矩阵及R50错误提示，不属于新增回归，未改v1来掩盖它们。全项目mypy当前12项错误，基线19项；变更的 operators/model/config/views 定向检查通过，缺失yaml stub等历史/环境错误仍保留。全树 Ruff、format、diff check及 wheel/sdist构建通过。

本地验证环境为 Python3.11 + Torch2.13（辅助依赖在临时目录），不等同于项目要求的Python≥3.12服务器运行环境。CUDA数值、真实数据、内存、吞吐和目标环境预检尚无本版本证据。根目录一次性pytest收集有既有同名模块冲突，故v2与其余目录分开运行。


## 阶段 B 工具准备（2026-09-26，尚无 CUDA 测量）

方法参考提交为 `7e4c669e5043986209339a0c8a613b02645356d3`。性能采集复用 `scripts/v2/capacity_probe.py`，新增 `profile_capture.py` 仅在探针的最后一次更新临时包装调用，正常训练不加载或开启这些标记。记录 CPU 数据 materialize/transfer/view，Student/Teacher 图、Cell、Response/self/cross/四头，SSL1/SSL2、backward、gather/梯度归约、optimizer 和 EMA/center。异常退出恢复原方法。CPU 完整优化更新的参数、优化器、center、RNG 对照通过；这不替代 CUDA 验证。

- `--benchmark-only --steps 40 --warmup-steps 10`：预热10、计时30；无profiler，无步内额外同步，不在计时段保存checkpoint。整步边界同步仍保留并明确记录。
- `--benchmark-only --steps 3 --profile-last-update`：先预热、短基线、最后1次完整更新抓取trace。该短探针是最初的成本诊断，不充当重复ABBA提速证明；最后一次数据准备和更新都被捕获。receipt kind为profile_only，不可纳入容量证明。
- `--profile-memory` 仅与上一选项联合：记录分配事件与小型时间线；不启用record_shapes/with_stack，避免不必要的tensor保留。无CUDA事件时报告unknown，不能报告GPU利用率0。
- `--sync-phase-timing` 是另一次诊断性同步计时，与profile互斥；默认关闭。旧容量报告兼容缺少此字段的历史收据，明确关闭时不伪造零通信耗时。
- 记录每卡更新时长、数据等待、CPU RSS/CPU时间、allocated/reserved、原始计时序列、中位数/p95、精确有序batch行计划hash、view RNG前后hash、硬件/NUMA/PCIe拓扑、source/config/data/environment身份。GPU忙碌时长对并发kernel取区间并集；嵌套CPU区域不能当作互斥阶段相加。
- 首个保守配置 `configs/v2/relay_jurkat/profiling_m2_a2/gradpert_v2/nadig_jurkat.yaml`：完整27.28M模型，micro2×accum2×双卡=8，和候选192仅物理/global batch不同。先单步完整更新与恢复，再短trace，确认可测后执行充分预热的无profiler基准。

工具本地定向验证49项通过，补充CLI非法组合/allocator失败关闭5项通过；三份工具mypy、全树Ruff和format通过。benchmark不含checkpoint，单步integration和128+capacity仍保留恢复验证；profile收据不能宣称容量。

测量依据：[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)、[CUDA异步执行与stream语义](https://docs.pytorch.org/docs/main/notes/cuda.html)、[pin_memory/non_blocking说明](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)。Profiler开销与稳态吞吐严格分开，不能用“另开stream”或单个GPU利用率数字作为加速证明。


## 2026-09-26 首个执行优化候选：chunk 区域编译（未采用为默认）

双卡时间线各记录约197.2万kernel，一次更新的嵌套CPU图前向与反向耗时显著；
原始trace和重新解析SHA见 `docs/experiments/relay-profile-d9c1fbf/`。
第一候选保留32-token chunk的三角求解、causal mask-before-exp、输入/状态精度
和两遍接力语义，将重复单chunk提取为`delta_final_block`，以Inductor区域编译
融合逐元素/归约工作。外层扫描、随机排序、图邻居、loss/Teacher更新保持原调用。
自包含候选配置 `configs/v2/relay_jurkat/compiled_m2_a2/gradpert_v2/nadig_jurkat.yaml`
仅显式增加`relay_kernel=inductor`；所有现有配置继续eager，历史payload省略默认字段。
动态维覆盖crop和尾chunk；fullgraph编译失败直接报错，不把静默eager回退计作加速。
首次编译成本、缓存目录与稳态更新分开记录。

训练engine在autocast之外backward，因此编译区域显式使用
`torch._functorch.config.patch(backward_pass_autocast="off")`并恢复原设置。
[PyTorch 编译反向语义](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_backward.html)
[区域编译与首次成本](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.reducing_compile_time.html)。
本地AOT数值、梯度、极端衰减、RNG和默认身份测试通过；全v2 268项通过，随后
额外反向精度上下文测试通过（269个不同测试）。四份源文件mypy、Ruff通过。
这些不等于CUDA Inductor或BF16通过。

待执行`benchmark_relay_kernel.py`：固定seed、合法mask no-op、非零初始状态、
1/94/257长度、2/64行、FP32与BF16 autocast，前向和全部输入梯度严格对照
atol3e-5/rtol3e-4并记录实际误差，3次预热+10次同步计时，冷编译单独记录。
任何数值失败保留收据，不放宽阈值来接受候选。随后必须同配置完整双卡更新、
Student/optimizer/Teacher/center/RNG核对，再进行ABBA稳定吞吐比较；只有实测
整步收益且科学语义验证通过，才更新默认执行后端。


完整更新验证入口 `scripts/v2/update_parity.py` 使用同一initial.pt恢复，不同kernel
使用同样两次真实数据更新；配置仅允许relay_kernel字段变化。沿用封存LR/EMA日程、
双卡全局loss人口、相同有序batch/view/RNG；截获梯度后执行原optimizer.step，
逐步核对loss、全部梯度、Student/Teacher/center和optimizer状态。RNG与输入哈希
严格相等，浮点比较atol3e-5/rtol3e-4。CPU拷贝只用于诊断，不用于吞吐计时。
梯度比较在服务器内存完成；检查点留服务器，导出小比较收据。两卡通过标志一起归约，任一卡失败
均停止，不继续ABBA或正式训练。本地3项证据辅助测试、mypy和lint通过；真实双卡
验证尚未执行，须等待kernel数值检查通过。


2026-09-26实测A1：原始d9源完整双卡global8，40次更新（10预热+30计时），
中位数26.6346s、p9527.3525s，含data的0.288754cells/s，峰值allocated3.547GB，
reserved4.012GB。收据见`docs/experiments/relay-steady-d9c1fbf-A1/receipt.json`。
569ae1b编译候选FP32通过，但BF16 B64/L94输出71/1048576元素超出既定阈值，
最大绝对误差0.00048828125；全部失败证据见
`docs/experiments/relay-kernel-candidate-569ae1b/receipt.json`，候选不采用。
下一修正显式设置编译options `emulate_precision_casts=True`，保持eager中间舍入；
目标环境Torch2.13已确认该选项存在且默认为False。保持原输入/误差阈值，重新验证，
尚不声称这是唯一误差来源或已经解决。参考
[PyTorch实际lowering](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/lowering.py)。


30a597a保留中间精度舍入后，BF16 B64/L94仍出现完全相同的71个超阈值元素；
说明该flag没有修复当前偏差。整chunk编译暂时搁置；其后排队的完整更新验证
在依赖门禁终止，没有执行GPU训练，不把它记为模型数值失败或通过。

下一项独立调度候选 `relay_validate_once=true` 将图层每个chunk内部重复的
`valid.any(-1).all()` CPU布尔读取，移到整层forward一次检查。所有行仍须拥有
合法邻居；私有chunk仅读取已检查的数据，直接调用attention.graph默认仍检查。
不改变KDA/MLA浮点运算、dropout、排序、邻域或任何梯度公式。新开关默认False，
自包含候选 `configs/v2/relay_jurkat/validated_m2_a2/gradpert_v2/nadig_jurkat.yaml`。
CPU checkpoint/dropout测试输出/所有参数与输入梯度/RNG逐位相等；示例的标量
读取从多次降到一次；空邻域在两种策略下都失败。真实GPU节省多少尚待整步测量。
完整更新工具现在只允许一次改变relay_kernel或relay_validate_once中一个，
会将具体执行差异写入收据；该调度候选保持eager，无需通过已搁置的编译候选。


真实validation-once更新对照（8aae56d）在首步：input/RNG完全一致、loss和
更新后模型状态一致，但gradient最大差1.133e-4超出原阈值；optimizer差1.133e-5。
尚不能区分GPU梯度累加非确定性与候选影响；不接受候选，不运行排队吞吐。
诊断工具新增原实现重复对照（强制配置/架构相同）与显式确定性模式；后者仅
用于数学/完整更新诊断，设置deterministic_algorithms、关闭autograd多线程与
CUBLAS_WORKSPACE_CONFIG=:4096:8。先测reference-repeat，再确定性A/B，收据
记录模式，误差阈值不变。正式训练与吞吐benchmark均不继承这些诊断设置。


4898de30dbb8ccdc09ce5bcbc8938ae2b4992cc1 实测补充：原实现重复对照也在梯度
出现1.3163e-4差异（输入/RNG一致），因此普通GPU执行存在数值重复性误差。
确定性A/B的两次完整更新在双卡均通过，loss、全部梯度、Student/Teacher/center
和optimizer逐位相同，input/RNG严格一致。第二次更新学习率非零；第一步日程
LR=0，所以单看首步模型参数相同没有证明力。收据保留在
`docs/experiments/relay-parity-4898de3/`，未放宽容差或修改正式训练确定性策略。

验证外移候选进入同源码ABBA测量：每次完整global8、40步（10预热、30计时），
A保持逐chunk校验，B仅整图层一次校验，CPU线程均2，双卡独占、无profiler。
有界队列`development/relay-4898de3-validation-abba`按A1/B1/B2/A2串行，
任何失败停止后续项。首次原始d9 A1仅作历史参考，不混入这组受控ABBA统计。
还未证明提速；当前默认仍不采用候选，也未认证新模型最大batch或启动正式B0。


## 下一候选：保持 eager 运算的 CUDA Graph 重放（尚未 CUDA 验证）

目标服务器 Torch2.13 的实际 `torch/_dynamo/backends/cudagraphs.py` 使用
AOTAutograd + boxed_nop + cudagraph_trees；它没有请求 Inductor 逐元素融合。
独立探针 `scripts/v2/benchmark_relay_cudagraph.py` 编译原有
`chunk_delta_final_state` 整个区域（内部 compiled=False），模型/配置默认不变。
先验证同一区域多次调用、第一次结果仍存活且作为第二次状态的生命周期；两个
输出都参与 loss，全部五个输入梯度均比较。覆盖 FP32/BF16、1/94/257长度、
2/64行、训练梯度/Teacher无梯度路径，mask no-op 与非零初态；固定 RNG。
使用与已通过完整更新相同的确定性诊断环境与原阈值，反向在autocast之外。
四次预热、十次计时；首次编译/捕获单独记录，每个计时输出也检查数值。
要求capture manager/recorded graphs实际存在，任何静默skip或预热后重新录制
都判失败；记录实际backend源码SHA与Torch Git版本、内存。每次独立更新边界
只调用一次mark_step_begin，不在两个相互依赖的区域之间标记新步。

这是独立 synthetic probe，不代表模型集成或提速；未添加新默认backend。
动态裁剪带来的形状缓存/图池内存，以及完整模型checkpoint重算、多视图、多次
累积backward生命周期都仍需随后验证。当前ABBA不被这项准备工作修改或中断。
