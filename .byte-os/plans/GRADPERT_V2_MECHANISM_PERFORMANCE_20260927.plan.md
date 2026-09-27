# GraD-Pert v2 机制性能工程（2026-09-27）

## 目标与边界

用户本轮重新授权性能调研、测试和等价优化；完成后复测双卡持续最大 batch，随后以新运行 ID 执行完整 B0 五轮和 best/last。旧 B0 `nadig_jurkat-seed1-20260927T041339Z-e35f5edb8e6248479462589ff3cef41f` 是用户主动停止的未完成运行，不恢复、不覆盖。正式方法保持单向 KDA 最终状态统一读取、每个合法图邻域独立建状态、随机顺序、图来源、视图、预测+SSL1+SSL2、跨卡 KoLeo、Teacher/center 时点与数据划分。

## 已核对基线

- 本地/远端 `main` 均为 `681d4fb609644d51c22f6b4e51dd61256adc3310`；服务器 `source-v2-teacher-681d4fb` 是不可变运行源码。当前 GPU0/1 均空闲（2026-09-27 07:23Z 只读核对）。
- m64×accum2×双卡，全局256的 128 步容量通过；服务器收据 `/data/yilangliu/GraD-Pert/development/v2-teacher-681d4fb-m64-capacity-oomfallback-20260927-1203/receipt.json`：20.183 秒/update 中位，11.8629 cells/s（含等待），数据等待 6.23%，每卡 peak allocated/reserved 30,648,811,520/30,987,517,952 bytes，checkpoint 续跑及300-control推理通过。m72 在 2/128 OOM。二者不能直接给出其他源码的容量。
- 旧 `38af3ce` trace 的 31.74 秒 profile 窗口含约 141 万 kernel；Student/Teacher graph 范围分别关联 190,106/169,597 kernel，GPU 区间并集 rank0 5.14 秒，但 profiler 开销和嵌套范围使其只能用于定位假设。通信及同步、重计算与显存生命周期尚需当前 `681d4fb` 新 trace。旧预取 ABBA 整体慢 2.45%；整段 CUDA Graph 变长重编译失败；Gram 融合完整梯度不合格，均保留负结果，不重复原候选。

## 优先级与决策门

1. **当前版本完整更新关键路径。** 同配置/种子隔离运行普通计时与单步 profile；分开记录启动/编译、CPU物化与视图、传输、发射间隔、GPU kernel与copy区间、反向重计算、梯度聚合/同步、optimizer及Teacher/center。CPU scope为嵌套项，禁止相加作关键路径。用 m32 安全捕获详细 trace，并用 m64 复核非 profile 的代表性完整更新/显存；若 m64 profile 有安全容量，再补 m64 trace。测量不碰 test truth。
2. **图邻域 KDA 执行合并（首选候选，须由新 trace 确认）。** 当前每个目标邻域长度短、每图视图多次读写状态，块内 Gram/下三角解/状态更新分成大量小算子。参照 Kimi KDA chunk 的块解与 FlashAttention 的 IO 模型，考虑按邻域长度组织 target，保持每个邻域内原有随机次序、目标和来源向量不变；将块内中间量限制在片上或复用 workspace，只有最终状态/目标 read 写出。先报告 padding、kernel 数、寄存器/显存与数值风险，再决定 fused kernel 或 batched block。旧 fused Gram 虽在小探针快，却没过整步梯度，不能复用为已验证候选。理论上界只由当前图范围的非重叠关键路径份额确定。
3. **选择性激活保留/重计算。** 参考 Checkmate 的 profile 成本思想，用实测每子图 saved bytes、重计算时间和 m64 余量，求不改 forward/RNG 的可行保留集合；当前 m64 peak allocated 已达30.65GB，接近单卡32GB，不能简单关闭全层 checkpoint。验证不同 batch 及峰值生命周期。
4. **调度与跨卡重叠仅在可用窗口成立时试验。** 旧预取负结果说明等待比例下降不等于步时下降；先用 CPU/GPU 对齐时间线定位可重叠的真实依赖窗口，核对 H2D pinned/stream、NUMA 绑定、gradient all-reduce 与 KoLeo all-gather。跨卡拓扑为 SYS，不假设通信免费。若计算是主导且通信窗口小，则不改调度。增加 GPU 并发须比较两任务总 cells/s、p95 步时、峰值显存和稳定性，不能只看利用率。

每个候选先记录浪费来源、Amdahl 上界、实现与新瓶颈；独立输出/所有输入及参数梯度、双卡多步非零LR完整 optimizer/EMA/center/RNG身份；再做同源码/配置的串行 A/B/B/A 全模型吞吐、启动成本、数据等待、p95和峰值显存；持续容量只在选定执行方案后复测。非等价候选转方法讨论，不混入此计划。

## 文献到工作负载的映射

| 原始来源 | 借鉴点与适用性 | 当前结论 |
| --- | --- | --- |
| [Kimi Linear KDA](https://arxiv.org/abs/2510.26692)及[FLA作者实现](https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/ops/kda/chunk.py) | channel decay＋delta 的块内计算与分块传播；本模型只读最终S、图邻域短，不能直接替换其逐token输出算法 | 适配候选，未复现 |
| [FlashAttention](https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) | 片上分块、减少HBM中间量与反向重算的IO成本模型；不是改用softmax注意力 | 方法论适配 |
| [Checkmate](https://proceedings.mlsys.org/paper_files/paper/2020/hash/0b816ae8f06f8dd3543dc3d9ef196cab-Abstract.html) | profile驱动激活保留/重计算选择 | 成本模型适配，未引入其求解器 |
| [GNNAdvisor](https://www.usenix.org/conference/osdi21/presentation/wang-yuke) | 图负载形状感知的任务组织；本模型有顺序敏感写入，不能任意交换邻居 | 分组策略候选 |
| [SALIENT](https://proceedings.mlsys.org/paper_files/paper/2022/hash/afacc5db3e0e85b446e6c7727cd7dca5-Abstract.html)及[NVIDIA CUDA最佳实践](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | CPU准备、传输与GPU流水；旧预取负结果要求先测真实依赖窗口和总吞吐 | 暂不采用新预取 |
| [PyTorch DDP系统论文](https://arxiv.org/abs/2006.15704) | 梯度桶/反向通信重叠；当前KoLeo全局近邻与统一归约语义禁止改变集合或更新时点 | 只有通信主导才评估 |

这里没有直接复现任何论文或声称原创系统贡献。新增文献假设与源实现核对继续记入 `docs/experiments/GRADPERT_V2_PERFORMANCE_LITERATURE.md`；前后实测记入 `docs/experiments/GRADPERT_V2_SINGLE_PASS_PERFORMANCE.md`。

## 门槛更新：短图融合失败与精确路径候选

短图融合已在干净发布源码 `27aec3643cc613b17f3b87eecebf05d1453743ba` 的双卡确定性完整首步失败，最大梯度差0.005747，损失、中心及优化器状态亦越过原容差。局部形状测试通过不能抵销此失败；淘汰该候选，不进行ABBA与容量。第一次 parity 的数据根误设也单独保留失败，不覆盖同ID。

下一优先级是**保留原块解算术与反向，消除重复不可训练形状常量构造**：以长度/设备/dtype缓存遮罩与单位矩阵，其他运算及RNG原样。trace中 `eye` 31842次，说明这一重复有实测规模；局部BF16前后向2.738→2.632ms及60项输出/梯度严格相等只证明值得做完整更新验收。预计收益上限低于图融合原设想，但实现代价、数值风险与显存占用更小；若双卡ABBA总吞吐无收益即淘汰。更大幅度的图投影重复/发射融合须先测具体子路径并满足同一完整更新门槛，不能凭重复率直接改算术。

2026-09-27 更新：常量复用已在干净发布 `32b2bfd88f36b9a7fad435c533dc825ef81942b4` 的双卡两步确定性完整更新通过，输入/RNG、所有损失/梯度/optimizer/Teacher/center 差值零，第二步 LR 非零。m64×累积2×双卡的 A1/B1/B2/A2 串行12步 benchmark-only 均已通过、队列 exit0。A/B 配对含数据等待吞吐比 1.01051/1.01361，几何均值 1.01206；有序 batch 与视图 RNG 起点哈希一致。比较收据 `/data/yilangliu/GraD-Pert/development/v2-kda-constants-32b2bfd-abba-20260927T0826Z/comparison.json`，SHA256 `aa594a8ab33ce2032ac81960ccfae92db2d31f9b77f7d33464604680f545aeef`。保留为可选小优化，不把这约1.2%的描述性结果作为主要优化终点。

若 ABBA 未显示稳定端到端收益，下一较高潜力机制按 **发射瓶颈优先**：将图 KDA 里形状稳定的子区域单独编译/捕获，视图构造、随机排列、动态尾块仍在区外；先做实际图块形状直方图，限制捕获变体，逐项记录冷编译、重放显存和 checkpoint backward。依据是完整更新约130万GPU kernel/116万 launch，而整段序列 CUDA Graph 在64次变长重编译后失败；[PyTorch CUDA Graph约束](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)明确要求固定形状/地址。此方案只借鉴区域化调度，不靠提高重编译上限。另一候选是[MLSys 2022 GNN计算图重排](https://proceedings.mlsys.org/paper_files/paper/2022/hash/b559156047e50cf316207249d0b5a6c5-Abstract.html)启发的节点/来源组合预投影，针对253371边、6506节点的重复变换；KDA的顺序写入和边来源不会移位，且线性因式分解的 BF16 舍入/梯度风险先通过微基准和完整更新证伪。两者均属待验证性能候选，不是已批准改变方法的提案；选择以实测节省上界、完整更新与双卡整体吞吐为准。


## 2026-09-27 调度决策与剩余验收

区域 CUDA Graph 捕获和相同基因/来源预投影均在 BF16 严格梯度门槛失败，已淘汰且保留原始日志，不提高编译上限或放宽容差。PyTorch 内建 NUMA 绑定在真实双卡 A1/B1/B2/A2 完整更新取得1.05240/1.04927两次配对含等待速度比，峰值预留无变化，比较收据 SHA256 `813bd55bef4715ceb430944d251cc7ee7e319fbd29043996405d0cb3c2886ed4`。本机采用绑定，不声称原创或跨硬件保证。当前尚无证据支持再开一个低收益开关试验；性能机制阶段进入已授权的容量及正式训练验证。

下一门槛：新自包含 m68×accum2×双卡=272 配置，只改 batch 相关字段；发布新 Git main SHA 和不可变服务器源后做先单步、再128步持续、checkpoint恢复与300-control。若 m68 OOM，记录失败并视显存余量检查 m66 或回退已通过 m64，优先保证稳定及总训练时间。采用 NUMA 绑定的正式 B0 必须新 ID、五轮完整预测+SSL1+SSL2、验证选 best、保留 last 并双角色测试；旧用户停止的 B0 仅作历史证据。所有实验依然只用两张已授权5090，不启动其他消融。
## 容量边界更新（2026-09-27）

NUMA m68 全局272单步通过，但128步持续测试在第22步 rank1反向 OOM，exit1，GPU仅余53.62 MiB，故淘汰为正式 batch；失败原始收据保留。下一有根据的候选 m66 全局264，源于已通过的 m64 与已失败的 m68 间的2样本/卡二分；只改自包含配置的 microbatch、train_batch_size，其他模型/损失/RNG/评估不变。m66须新干净 Git/服务器源码、独立 run ID、1步和128步收据、checkpoint与300-control通过；若失败改选 m64，并在最终发布源码复测，不因一次预检宣称容量。NUMA A/B证据和机制取舍保持不变。通过后用全新 ID 启动完整五轮 B0，不启动其他消融。
