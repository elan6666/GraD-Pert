# 性能工程文献驱动矩阵

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
