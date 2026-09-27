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
