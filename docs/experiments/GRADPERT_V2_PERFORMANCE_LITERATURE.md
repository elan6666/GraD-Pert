# 性能工程文献驱动矩阵

## 2026-09-27：文献假设的双卡验证结论

[PyTorch 官方 NUMA 绑定接口](https://docs.pytorch.org/docs/main/elastic/numa.html)解决多 socket 机器的 worker/设备局部性问题；我们仅适配其现有 `--numa-binding=node` 执行选项，非原创机制。真实双5090完整更新 A1/B1/B2/A2 各12步、同源码/配置/随机计划，B/A 配对含等待速度比1.05240和1.04927，几何均值1.05083，预留显存不变。因此在当前服务器采用；不能外推到其他拓扑或宣称长期模型效果。详见[性能报告](GRADPERT_V2_SINGLE_PASS_PERFORMANCE.md#numa-本地-cpu-绑定整步对照终态2026-09-27)。这项有实测收益的调度借鉴与失败的局部 CUDA Graph/图投影候选并列记录，不能抹去后者的精度反证。

## 2026-09-27：机制证伪与 CPU/GPU 亲和调度

区域捕获参照 [PyTorch `make_graphed_callables` 官方接口](https://docs.pytorch.org/docs/main/generated/torch.cuda.make_graphed_callables.html) 与 [CUDA Graph 内存生命周期说明](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)。本项目的固定64目标图块在单次捕获能得到原值/梯度，但实际连续两块、前块状态仍参与反向时产生非有限梯度；`torch.compile(..., backend="cudagraphs")` 两种精度配置在 BF16 输入梯度也超过原阈值。前向图捕获＋原生重算反向仍在 BF16 两块链上超阈值。这些是本负载/当前实现的反证，不说明 PyTorch 机制本身有通用错误；当前候选淘汰，保留服务器隔离测试日志。

对 [MLSys 2022 GNN 计算图重排](https://proceedings.mlsys.org/paper_files/paper/2022/hash/b559156047e50cf316207249d0b5a6c5-Abstract.html) 的更保守检验是**不做代数分配律，只把完全相同的 `(基因, 四位来源)` 投影算一次**。Jurkat 冻结图共有253371条合法边、27202个不同组合，理论重复9.31倍。128目标 BF16 局部探针四组前向张量完全相同，但反向对输入和多组投影权重的累加次序改变，既定3e-5/3e-4阈值失败；所以不做整步/容量/正式训练，也不把9.31倍重复误称为可实现的整步加速。

下一调度假设依据 [PyTorch NUMA 绑定官方文档](https://docs.pytorch.org/docs/main/elastic/numa.html)、[PyTorch 性能调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)、[NCCL CPU/内存亲和说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/user-guide/docs/troubleshooting/performance_and_tuning.html)。本机 `nvidia-smi topo -m` 显示 GPU0 近 NUMA0、GPU1 近 NUMA3，现有 rank 可在0–127任意逻辑核调度，而当前完整更新约130万次 kernel、CPU 发射为显著成本。已用服务器 Torch2.13 `torchrun --numa-binding=node` 小探针验证 rank0 实际亲和 0–15/64–79、rank1 为48–63/112–127；它是官方已有机制，不是原创。现在同一干净源码/配置做 A1/B1/B2/A2 全更新计时，仅 B 开内建亲和。预期收益受 CPU 发射份额上限约束，可能因可用 CPU 缩小或数据线程拥塞变慢；只有总 cells/s、p95、启动与峰值内存整体证据才能决定是否采用。

## 2026-09-27：图边重复投影的针对性新假设

[Understanding GNN Computational Graph: A Coordinated Computation, IO, and Memory Perspective（MLSys 2022）](https://proceedings.mlsys.org/paper_files/paper/2022/hash/b559156047e50cf316207249d0b5a6c5-Abstract.html) 将“先在每条边重复做神经变换，再传播”改为可交换时“先对节点变换、后传播”，并结合融合和选择性重算。**论文结果不是本项目速度承诺。** 本项目实测 6,506 个图节点却产生 253,371 条合法目标—邻居边，单个基因平均参与约38.95个目标邻域；`RelayDeltaAttention.graph` 当前先 gather 邻居，加四位边来源投影，再逐边执行 key/value/decay/write 投影。因而重复投影是有明确规模的下一候选，而非泛泛尝试另一个开关。

可研究的因式分解仅限线性层：若 `selected_{ij}=h_j+W_s s_{ij}`，则数学上 `W selected_{ij}=W h_j+W W_s s_{ij}`，可以对 6,506 个节点和至多16种来源组合先投影，再在边上索引/相加；非线性归一化、`decay_down→decay_up` 及 KDA 随机次序仍须维持原位置，不能把有顺序的状态写入变为可交换聚合。预期主要减少重复 GEMM/发射，代价是边上 gather/add、节点投影中间存储和 backward scatter；浮点结合顺序可能改变输出和梯度，故它**尚非已验证等价优化**。先用真实 BF16 图层微基准检查各投影/梯度误差与节省，再用既定双卡完整更新门槛检验 loss、KoLeo 最近邻、optimizer/EMA/center；若越界立即淘汰，不放宽误差或混入 B0。是否实施取决于当前常量缓存 ABBA 与更细的图投影成本归因。

[PyTorch CUDA Graph 官方约束](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)与[CUDA Graph Trees 的动态形状说明](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)支持另一个**不同于失败的整段序列捕获**的候选：只捕获图 KDA 内大量重复、形状稳定的块，随机邻域排列与视图长度在捕获区外作为数据传入，变长尾块留 eager；保留相同初始状态、输出和反向。依据是当前一完整更新约130万GPU kernel、约116万次CUDA launch，而整段序列因随机长度在64次重编译后失败。若 64 行×固定邻域长度的核心图块覆盖绝大多数目标，可用少量 capture 分摊发射成本；但图裁剪长度、静态地址、checkpoint 的重算、RNG 与额外 graph pool 显存均是验收风险。先统计真实块形状及覆盖率、给单个无随机块做含 backward 的区域捕获小探针，记录编译/捕获次数和池显存；若不满足有限形状或完整更新校验，停止此路，不通过提高重编译上限回避失败。官方文档提供的是约束和机制，不构成本项目收益证明。


## 2026-09-27：按当前完整更新负载重排研究问题

已重新核对 [Kimi Linear 原始技术报告](https://arxiv.org/abs/2510.26692)、[DeltaNet 的 NeurIPS 2024 并行算法](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d13a3eae72366e61dfdc7eea82eeb685-Abstract-Conference.html)、[FLA 固定提交的 KDA 接口](https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/ops/kda/chunk.py)、[FlashAttention 的 IO 分块论文](https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)、[Checkmate](https://proceedings.mlsys.org/paper_files/paper/2020/hash/0b816ae8f06f8dd3543dc3d9ef196cab-Abstract.html)、[GNNAdvisor](https://www.usenix.org/conference/osdi21/presentation/wang-yuke)、[SALIENT](https://proceedings.mlsys.org/paper_files/paper/2022/hash/afacc5db3e0e85b446e6c7727cd7dca5-Abstract.html)以及[PyTorch DDP系统论文](https://arxiv.org/abs/2006.15704)。这是原始论文/作者代码的机制借鉴，不是移植性能结论。当前源码681d4fb的 m64 双卡持续128步已通过，旧38af3ce trace 仅显示图/KDA细粒度发射线索；新完整更新 profile 尚需确认关键路径。

已进一步检查[FLA 同一冻结提交的 fused recurrent KDA](https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/ops/kda/fused_recurrent.py)：该入口有短序列逐token递推、可返回最终状态，但其文件只提供前向入口且仍分配逐token输出；**不能直接替代**本项目只用最终S的训练算子。本轮自行根据本模型的状态方程推导短邻域解析反向，并保留现有图分块边界以限制重建状态的生命周期。来源是成熟递推/融合思想的适配，尚未声明直接复现或新方法贡献。新 `681d4fb` trace 证明图路径及其反向有大量细粒度调用；合成局部探针通过数值容差且前后向有局部收益，但全更新收益未证实。基于通信kernel约0.31秒、数据等待6.62%、旧预取ABBA负结果，调度/通信维持低优先级，若之后瓶颈转移再测。

|研究问题|论文提供的可借鉴机制|与本模型的精确边界|首个证伪实验|
|---|---|---|---|
|图 KDA 是否消耗主要发射与HBM预算？|KDA/DeltaNet 块解、FlashAttention 片上复用、GNNAdvisor 形状感知任务组织|每个目标邻域各自维护 S；顺序随机且不可交换；只读最终 S。不能替换为普通可交换图聚合、不能直接复用逐token输出内核|681d4fb双卡 profile 的图scope launch/访存、邻域长短分布；分块/融合候选随后做严格完整更新和ABBA|
|显存换重算是否优于当前全层checkpoint？|Checkmate的按子图时间/显存成本配置|当前m64峰分配30.65GB/卡；粗暴关闭重算可能OOM；随机视图/Teacher需保持原RNG时点|记录每子图重算次数、峰值生命周期及 m32/m64余量；仅重排保存策略不碰模型数学|
|GPU空档能否由CPU流水或通信重叠消除？|SALIENT CPU采样/搬运流水、CUDA pinned memory/streams、DDP梯度桶|旧项目预取ABBA整体慢2.45%；KoLeo跨卡候选集合、同步梯度及center更新时点固定|对齐CPU发射、H2D、GPU kernel、NCCL时间线；若可重叠窗口不足，淘汰调度候选而非追求利用率|

实验若采用算法思想是**针对最终状态/短图邻域的适配**，不是直接复现论文。增量是否构成论文新贡献需独立新颖性审查；当前目标只认真实完整训练吞吐、容量与等价验证。

2026-09-26 首轮检索。以下区分已核对的原始论文/作者仓库入口与尚未逐函数审阅的实现。
这是后续假设的依据，不是把已完成的工程工作追认为论文创新。不得仅因某方法在语言模型上快就移植。

|原始来源|机制与假设|本项目对应证据|兼容边界与代价|下一验证|
|---|---|---|---|---|
|[DeltaNet, NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/d13a3eae72366e61dfdc7eea82eeb685-Abstract-Conference.html)|用紧凑矩阵表示并行化delta更新，避免逐token串行训练|现有final-state block包含下三角求解和通道Gram临时张量|我们的通道衰减不能直接替换成无衰减DeltaNet；只能借鉴分块推导|冻结作者实现版本，核对门控位置、初始状态与final-state梯度，再考虑完整块优化|
|[Kimi Linear 原作者报告 §3.1](https://yzhang.site/assets/pubs/techreport/2025/kda.pdf)|对角门控delta recurrence的chunk表示；比普通DeltaNet更贴近当前式子|每层只需要最终S，图邻域较短，表达视图较长|不能加入短卷积或改变门控；论文逐token输出路径不等于我们的最终状态读取|检查完整chunk算子而非继续独立Gram替换；证明哪些输出可以删除且反向仍完整|
|[GLA, ICML 2024 作者单位页面](https://research.ibm.com/publications/gated-linear-attention-transformers-with-hardware-efficient-training)|硬件友好的门控线性注意力训练|不同chunk/邻域规模导致计算密度差异|GLA不是delta更新，不替换算子；只借鉴分块和数值布局|对比形状与门控数值范围，建立chunk开销模型|
|[FlashAttention, NeurIPS 2022](https://arxiv.org/abs/2205.14135)|分块减少HBM访问，反向重计算换存储|大量pointwise临时张量和launch；mHC融合已有局部到整体证据|借鉴IO成本模型，不声称采用softmax attention可直接优化KDA|分别测读写字节、launch数、重算时间；与原生端到端比较|
|[Checkmate, MLSys 2020](https://proceedings.mlsys.org/paper_files/paper/2020/hash/0b816ae8f06f8dd3543dc3d9ef196cab-Abstract.html)|根据计算图和profile成本选择保留/重计算，在显存约束下优化时间|关闭全部sequence checkpoint虽更新一致但显存约3.27→13.55GiB；这不是最佳选择的证明|应按子图选择，保存RNG和全更新语义；求解/建模本身有成本|优先核对当前checkpoint边界与重计算trace，考虑选择性保留昂贵子图，不再仅全开/全关|
|[GNNAdvisor, OSDI 2021](https://www.usenix.org/conference/osdi21/presentation/wang-yuke)|结合图和模型特征的二维任务组织及内存层级适配|graph CPU耗时/邻域padding和读取；图KDA短序列批处理|其邻居聚合通常可交换，我们KDA写入有顺序；可分组查询，不得重排每个邻居序列或图边|统计邻域长度、padding、分组成本，保持ID/随机顺序后再提分桶方案|
|[Lynx 原始论文](https://arxiv.org/abs/2406.08756)|使重计算与pipeline通信重叠，依赖可用通信窗口|现有profile尚未证明双卡通信窗口足够长|我们的双卡数据并行不同于论文大模型pipeline；强行套用可能增加争用|先测通信关键路径与可重叠窗口，再决定是否值得实现|
|[ByteScheduler, SOSP 2019 原文](https://people.computing.clemson.edu/~jmarty/projects/lowLatencyNetworking/papers/AGenericCommunicationSchedulerForDistributedDNNTrainingAcceleration.pdf)|通信调度的候选来源，具体调度约束待原文逐节核对|CPU预取在本项目ABBA慢2.45%，证明利用率不能替代有效吞吐|不能引入异步参数更新、改变累积或KoLeo集合|先核验同步梯度聚合依赖与通信分块成本；未确认论文细节前不实现|

## 官方/作者实现核对状态

- [FLA](https://github.com/fla-org/flash-linear-attention)：已核对仓库入口及KDA存在；下一步冻结commit、读取具体KDA chunk/final-state/backward文件和许可。不能将README的平台支持声明当成本项目跨设备通过。
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)、[Checkmate](https://github.com/parasj/checkmate)：核对作者仓库入口，尚未逐函数复现，不引入运行时依赖。
- [GNNAdvisor作者实现](https://github.com/YukeWang96/GNNAdvisor_OSDI21/blob/master/README.md)：核对artifact描述；其训练计时范围不直接对应我们含数据等待的指标。
- [BytePS](https://github.com/bytedance/byteps)：仅为通信系统参考入口，不声称已复现ByteScheduler。

## 本轮优先级与停止条件

1. 将已经通过完整更新与ABBA的mHC融合做可移植显式执行选项；CPU/MPS/无Triton保持原生，内核错误不能静默掩盖。
2. 在新机制实验前，优先完成KDA官方chunk与Checkmate式选择性重计算的具体实现审阅；用当前trace给候选成本排序，记录预期节省、额外存储和启动成本。
3. 图查询分桶只在padding/调度代价占关键路径时实施；通信重叠只在实际可用窗口存在时实施。负结果保留，不为了利用率100%强行并发。
4. 每个候选独立做输出/所有梯度/非零LR多步/EMA/center，再做代表性batch ABBA和显存；不放宽已冻结容差，不自动扩大设备资源。
5. 基于证据确定最终执行组合后做持续最大batch、恢复与300-control评估，再完整B0五轮best/last。当前Gram候选淘汰，无整体训练加速结论。

跨设备矩阵至少包含当前5090与另一代/带宽显存等级GPU；后者无现成授权设备时标注待验证。
明确区分直接复现、机制适配和未证实的新贡献。数学等价、跨设备可运行、跨设备加速、论文新颖性是四项不同结论。

## 首个冻结实现检查：FLA KDA

远端HEAD已固定为 `954438d1fcb5e1bb05c22f9908de9c5c2df74ae5`；读取
[chunk.py](https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/ops/kda/chunk.py)。
该文件头声明MIT许可。接口同时输出逐token结果和可选final_state；backward接收do及dht，
因此必须沿final_state梯度检查，而不是只比逐token输出。它有显式disable_recompute，
提示保存/重算策略应独立比较。safe_gate/lower_bound涉及门控改变，本项目不能为了速度打开。
use_gate_in_kernel=false允许输入既有log-decay，但这尚不证明数值、状态布局和完整更新兼容。
下一步读取同commit的chunk_fwd.py/chunk_bwd.py，确认只要final_state时哪些工作仍执行。
未导入FLA运行时、未复制源代码、未声称复现；此处是实现接口层的可核验观察。

## 完整KDA调用图检查与下一候选

已读取同一冻结commit的chunk_fwd.py、chunk_bwd.py（临时副本仅在/tmp，不导入模型）。
前向在状态传播后仍调用chunk_gla_fwd_o_gk；disable_recompute=false时清除w/u/qg/kg/v_new及h，
反向重建它们，再合并do和dht路径。当前实现已经只读最终S，因此“直接换成FLA”不自动省算。
完整块重写需重新证明final-state-only反向及浮点累加；鉴于Gram局部替换的整步失败，暂不直接迁移。

下一具体假设：按梯度活性消除Teacher Sinkhorn无用中间存储。
依据是FlashAttention的IO视角及Checkmate的张量生命周期视角，而非声称新算法。
当前融合kernel每token写20×2×4×4个FP32概率，即2560 bytes，供反向使用。
Teacher处于no_grad，反向不会消费这些值。候选在无梯度调用中只保留最终输出，
全部40次log归一化与浮点结合顺序不变；Student需梯度的路径保持现状。
理论上减少这部分HBM写出与临时存储，不改变渐近计算复杂度；编译器删除无用exp是否
改变最终输出仍须实测。收益取决于Teacher调用占比、tokens、内核是否带宽受限，不能按字节比外推速度。
验证：多形状输出exact→完整两卡多步更新/EMA/center→相同配置ABBA→峰值显存与启动成本。
此为尚未实现的独立候选。先完成当前显式auto配置的完整校验；不混合Gram候选。
