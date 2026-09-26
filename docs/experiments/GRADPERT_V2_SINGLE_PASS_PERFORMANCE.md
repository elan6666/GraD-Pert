# GraD-Pert v2 单向最终状态读取：性能工程记录

更新：2026-09-26。工程测试，尚无当前方法正式五轮结果。

## 固定方法与验收边界

单向 KDA 写入后所有 query 读取最终 S；图分支每个合法邻域独立建状态、
仅查询目标基因。随机顺序、视图裁剪、完整预测＋SSL1＋SSL2 不变。
当前诊断使用双 RTX 5090、每卡 microbatch2、累积2、全局8；这是保守工程
探针，不是已选择的正式 batch。所有下表收据暴露源码 SHA、配置和数据身份。

## 实测与取舍

| 项目 | 完整更新/吞吐证据 | 当前决定 |
| --- | --- | --- |
| Eager＋激活重计算 | 12次更新，3预热9计时；更新中位数19.5796秒，峰值分配3.265GiB；含数据等待0.3876细胞/秒。[原始收据](single-c073a37-preflight/baseline/receipt.json)，源码c073a37c3c038561fb3d167b2a8f53f8650544bd | 当前默认；未认证持续容量 |
| 关闭 Cell/Response 重计算，保留图重计算 | 双卡两次完整更新与默认逐项差值0、输入/RNG一致。[一致性收据](single-473466d-checkpoint-parity/rank-0-receipt.json)；一次12步参考测量中位数17.1801秒，峰值13.552GiB，[速度收据](single-3a2980d-replay-abba/A1/receipt.json)，源码3a2980d2eee30faa1ad99fb4481262701879d262 | 显存代价明显，未采用；跨源码单次速度比较仅描述性 |
| 整段序列 CUDA Graph replay | 禁用序列重计算时双卡两步完全一致，[收据](single-a63ac39-replay-no-sequence-checkpoint/rank-0-receipt.json)；随机长度持续测试第11步触及重编译64次上限，10/12失败，[失败收据](single-3a2980d-replay-abba/B1/receipt.json) | 淘汰当前整段静态形状方案；不提高缓存上限来掩盖问题 |
| 一批 CPU 数据/视图预取 | 双卡两次完整更新输入/RNG一致，损失、梯度、模型/Teacher/center及优化器比较差值0，[rank0](single-ec2384a-prefetch-parity/rank-0-receipt.json)、[rank1](single-ec2384a-prefetch-parity/rank-1-receipt.json)，源码ec2384ade9fa66458eac2ff40c2dbff2fd067dc8 | ABBA全部完成，预取含等待平均步时增加2.4476%，不采用；[审计报告](single-ec2384a-prefetch-abba/comparison.json) |

关闭重计算的一致性覆盖两次更新（含非零 LR），不能代替长程科学结果。
上述显存是 PyTorch peak allocated；与 nvidia-smi 已保留显存不可混用。

## 时间线给出的瓶颈

[Profiler小结](single-c073a37-preflight/profile/rank-0-trace-summary.json)显示
大量细粒度 kernel 启动；该次约141万次 kernel。图编码、反向和 CPU 发射
值得优先优化；通信在同步诊断中较小。Profiler 的时间与利用率受到采样开销
影响，不可当作普通训练吞吐。原始大 trace 保留服务器。

基准数据/视图等待约1.27秒/步，占含等待时间约6.13%。CPU预取旨在把这部分
与当前GPU计算重叠：工作线程仅执行CPU数据/视图准备，独立视图RNG提前计算；
主线程消费时才提交公开RNG状态并转移张量。提前关闭或异常不提交未消费批次。
没有修改视图长度、采样分布、损失或模型结构。完整本地v2测试312项通过。

## CPU预取四轮结果

同源码ec2384ade9fa66458eac2ff40c2dbff2fd067dc8，完整身份审计通过。
A1/B1/B2/A2含等待平均步时分别为21.6529/22.3103/21.8072/21.4106秒。
预取等待占比下降但整体步时增加2.4476%；两组重复是描述性证据，非显著性
检验。维持同步默认，不把局部等待改善称为训练提速。原始收据与每卡计时见
[审计报告及其输入](single-ec2384a-prefetch-abba/comparison.json)。

## 尚需完成

1. 已完成CPU预取ABBA审计，淘汰当前线程预取候选。当前进入双卡容量扫描。
2. 按证据选择执行方式。已准备micro8/16/32/48/64双卡、累积2容量候选；
   若最高仍通过则继续扩展，失败则细化边界。先单步定位，再128+持续更新、
   checkpoint续跑及300-control验证。旧模型batch192不构成当前方法的证据。
3. 在选定较大batch重新检查预取收益，锁定干净发布版本与自包含配置。
4. 仅启动已授权完整B0五轮，完成best/last测试与最终指标/身份核验；不启动
   prediction_only、其他消融或恢复旧停止运行。

当前进程及路径以 [Byte状态](../../.byte-os/STATE.md) 和实时服务器收据为准。

## 机制级优化路线（用户补充后的优先级）

开关尝试不是机制优化完成。当前单步容量证据：micro72 passed，micro80 OOM，
88未启动；源码3ba9a960d79d207ce89dd938b672be051f327234，[收据](single-3ba9a96-capacity-integration/sweep.json)。
未进行128步稳定性认证；先完成主要机制优化，之后重新验证容量与正式B0。

### 优先1：避免显式构造衰减加权 Gram 的五维临时量

现有delta块先形成累积衰减g，再计算
`P[i,j] = sum_d(k[i,d] * k[j,d] * exp(g[i,d]-g[j,d]))`，仅j<i进入
`I + tril(beta * P,-1)`的三角求解。当前广播实现显式产生[B,H,L,L,D]的
衰减与乘积。以B64/H4/L32/D64为例，单个FP32临时量64MiB；这只是
形状推导，不是已测峰值归因。图KDA的多邻域、多层、多视图及重计算重复此路径。

候选：融合因果pair/channel归约，直接输出[B,H,L,L]，不把通道展开的
临时量写入全局内存。三角求解、其后的状态更新、随机数和模型参数保持原样。
解析反向对每个k/g元素累加它作为pair左/右端的贡献；避免原子写入，保留
确定性归约。数学公式等价，有限精度归约次序仍须独立验证，不能预先称位级等价。
需要测forward/backward和saved tensors；实现代价中等，风险是exp重算、寄存器
压力或反向归约抵消收益。Amdahl上限由该区域占完整更新的实测比例决定，当前
尚不能给出端到端加速百分比。先从既有trace提取kernel/CPU发射成本再选tile。

### 优先2：图邻域执行组织

审计padding比例、每64个query分块的发射开销、邻域来源相关投影是否重复。
候选只在证据支持时提出具体实现：重排必须可逆并保留基因ID/来源/随机顺序；
改变chunk大小可能改变dropout RNG，不能当作无条件等价开关。批处理粒度变大
还可能降低occupancy或抬高活跃显存。先计数浪费比例，再确定收益上限。

### 后续调度与验证

线程预取已无收益、整段静态CUDA Graph因随机长度重编译失败，不盲目重复。
不围绕DSec、沙箱或某篇论文定方向；其他技术仅作为适用性参考，不宣称原创。
新机制必须通过输出/所有输入梯度、非零LR多步optimizer/EMA/center与RNG验证，
再测双卡完整更新的启动成本、稳态吞吐、峰值显存和稳定性，并在代表性batch
复测。Profiler嵌套inclusive时间不能相加成关键路径；GPU union与kernel总时长
分开报告，不能用瞬时利用率证明效果。若改变模型数学或目标，转为待讨论方法
提案，不混入等价优化。机制通过后再做持续容量与完整B0五轮/best/last。

### 2026-09-26：既有 trace 的流式成本分解

原始 rank0 trace 已在服务器完成有界解析，13,518,242 个事件；小结果见
`single-38af3ce-profile-replay/costs/rank0.json`（含 trace 与分析脚本 SHA256）。
全窗口 kernel 1,414,510 次，kernel duration sum 4.8895s；CUDA runtime
launch API 1,281,076 次、5.5947s。CPU `logsumexp` 43,200 次、嵌套累计4.3420s，
对应 mHC Sinkhorn 每次20轮双轴归一化；不能直接把累计时间作为可节约墙钟时间。
最大单类 GPU float multiply kernel 累计1.1495s，亦不能全部归给 KDA。

因此先补充 launch correlation 与 CPU scope 的归因，比较 mHC 归一化、
图前向、dropout backward、三角求解的 kernel 数量/时间；不预设 Gram 融合
一定是第一优先级。三遍流式扫描只保留选定区间与关联ID，核验线程/进程，
避免将 GPU annotation 与 CPU scope 混合；范围可能嵌套，不相加。
采集不会再占 GPU，不改变任何模型、训练语义或性能默认配置。

#### mHC融合候选：先独立验证，尚未采用

关联归因结果见 costs/attribution-rank0.json：logsumexp CPU区间去重后1.5996s，
launch API累计0.5293s，129600个GPU kernel累计0.1660s。GPU算术量小而发射多，
验证融合20轮Sinkhorn的成本较低，故先检验此机制；KDA大临时量仍是后续候选。
这些来自带profiling的窗口、仅覆盖logsumexp前向，不提供整模型加速比例。
只消除此CPU区间的理想上限约31.74/(31.74-1.60)=1.053倍（假定区间全部位于
关键路径；真实可能更小）；完整融合还包含减法和反向，但不能用该数外推。

每步归一化 z'=z-logsumexp_axis(z)，反向为
u'=u-exp(z')*sum_axis(u)。保存40次归一化后的概率，按逆序应用这个精确导数，
不减少20轮、不改变4streams，不改随机数或目标。独立Triton前向/反向各一个kernel，
避免每次归一化分解成许多小kernel；代价是保存每token40×16个FP32概率（2560bytes），
后续要测其对checkpoint重计算、实际batch和内存的影响，不能只看微基准。
FP32 libdevice exp/log，关闭FMA合并；CPU公式输出精确一致、解析梯度10项测试通过。
代码独立opt-in且未接入ManifoldResidual。GPU严格数值、完整更新及吞吐尚未验证。

独立GPU探针首版4f855f9因Triton循环临时量作用域编译失败，0有效case，失败保留。
修正版59305eecb4ed68636dcdebae5a20625fadc717db，在两张5090分别检验
17/256/4096 token、logit scale .2/4/20，共18组，全部输出/梯度通过既定
atol3e-6/rtol3e-5。最大输出差4.92e-7、梯度差3.88e-7。
局部前向+反向eager4.33–5.04ms、fused0.514–0.656ms；4096token额外峰值
14,221,312→11,010,048bytes。冷调用耗时和各组原值保留在
`single-59305ee-sinkhorn-probe/receipt.json`。此为合成算子微基准，不是完整双卡
训练吞吐，不代表正式采用。下一步candidate-only完整更新校验，正式默认仍eager。
