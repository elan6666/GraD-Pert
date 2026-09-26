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

完整更新67fd8e1已失败：第1次candidate更新输入/RNG完全一致，但gradient最大差
1.6681e-4，ssl2_koleo约0.0060016与0.0060389，超出既定3e-5/3e-4标准。
两rank失败收据均保留，未进入ABBA，不采用此版本。微基准通过不能替代此门槛。

后续仅做有新依据的精度修复：审阅服务器Torch2.13安装头文件
ATen/native/cuda/Reduce.cuh:562–631及635–670，四元素strided归约在线程内
逐项结合，连续轴采用warp下降offset归约。原Triton对两轴都用树归约，
新增候选显式按(((a+b)+c)+d)实现行轴求和，保留连续轴树、libdevice与禁用FMA。
先增加40个归一化阶段的逐元素及BF16舍入差异计数；先用独立GPU探针验证假设，
不重复运行未修复版本、不提高容差、不因局部速度收益跳过完整梯度校验。

修正7ae9a8b0259dd31b79f3b9d0a39e1c8430b1005a独立双卡18组全部通过，
40个归一化阶段、最终输出与输入梯度全部逐元素相等，BF16舍入差异0。
这证实行轴归约结合顺序是已测微基准误差来源；尚不能代替整模型校验。
修正后完整双卡校验已另起single-7ae9a8b-sinkhorn-parity，不覆盖失败版本。
另外正式采用前要将kernel的N从constexpr改为运行时边界（只保留4stream、20轮常量），
防止随机视图token数导致每个长度一次编译；需要新的精确数值与冷/稳态验证。

7ae9a8b完整双卡校验exit0，两rank两更新均通过；302参数梯度、620模型状态张量、
739优化器状态张量比较均最大差0，包含Teacher/center的objective state_dict。
loss标量比较通过，但旧max_absolute字段仅统计张量，不能凭它宣称loss逐位相等；
新诊断已补标量差值与显式learning_rate记录。30个ManifoldResidual启用融合。
原始收据见single-7ae9a8b-sinkhorn-parity/。

后续候选将N改为runtime参数并禁止值/对齐特化，保持STEPS=20/BLOCK=8常量。
微基准扩展到1/17/256/4096/32768 token，记录Triton3.7.1两kernel每卡缓存数。
先验证变长无需逐长度编译且输出/梯度精确，再执行同版本完整更新与端到端ABBA。

动态N候选cab78a1在1–32768 token的双卡30组合成probe通过，40阶段/输出/梯度
全部exact，且每卡forward/backward各仅1个编译缓存。完整更新校验仍在运行。
吞吐工具补启动至训练、runtime准备、各rank热身update/data-wait时间；原稳定段
计时规则不变。后续ABBA源码若仅该工具元数据变动，会明确与parity源码不同。

cab78a1最终完整更新exit0：两rank两步输入/RNG一致，loss标量、全部gradients、
objective(Teacher+centers)、optimizer最大绝对差均0。第二步LR明确为
7.796055196070788e-8。源cab78a1与ABBA源b64745b的src/configs diff为空，
后者仅补工具冷启动计时。正式吞吐ABBA使用global128、micro32、accum2，
各12steps/3warmup，保证与实际容量相比有代表性；不把此12步称为持续容量验证。

### 待独立验证：KDA weighted-Gram前向融合

在mHC ABBA计时期间仅做本地CPU工作，没有另占GPU干扰吞吐。实现独立候选，
未接入delta_final_block，也未更改默认：
P_ij = 1[j<i] sum_d k_id k_jd exp(g_id-g_jd)。每个query直接归约64channel，
仅输出[B,H,L,L]，不创建前向[B,H,L,L,64]的decay/product。
输入连续化只在kernel入口；autograd保存原始stride的key/gate，避免改变PyTorch
反向归约结合顺序。长度N为运行时参数，固定最大chunk32和channel64，不按N编译。

第一候选反向保留PyTorch计算/归约顺序（重建decay），而非一次同时改所有算子。
U=dP，dproduct=U*decay；dk由两个广播乘法分别沿j/i归约相加；
dg=sum_j(U*(k_i*k_j)*decay)+sum_i(-(U*(k_i*k_j)*decay))，上三角先mask再exp。
key梯度完成后释放dproduct，以缩短大临时量共存时间。代价是反向重建decay，
可能抵消前向收益；只有端到端测量能决定是否采用。三角solve与final S均不动。
CPU10项测试覆盖float32/64、1/3/17/32长度、非连续输入、原block下三角及溢出mask，
输出/解析梯度逐元素相等。Triton kernel尚未在GPU执行，不能宣称编译或数值通过。
待ABBA收尾后使用scripts/v2/weighted_gram_probe.py独立测数值、冷/稳态和峰值显存。


### 已完成：mHC 完整双卡 ABBA 与 KDA 独立微基准

mHC ABBA 四组均12/12，父进程exit0，比较收据status=comparable。
训练源码 b64745b7cc2a752ab992a32c62a234193a676cad；配置 SHA256
c03f83643f208af475765d2491401fd2de0b6b0091d96c7243712b21152b5a8f。
完整预测+SSL1+SSL2，global128=m32×2累积×2卡，每组3热身、9计时。

|组|update中位秒|含数据等待平均秒|cells/s|峰值allocated bytes|
|---|---:|---:|---:|---:|
|A1|19.9003|21.3293|6.0011|16286307328|
|B1|19.0998|20.4358|6.2635|16231134208|
|B2|18.9844|20.2642|6.3166|16231134208|
|A2|19.9185|21.3046|6.0081|16286307328|

吞吐比1.047519，含等待步耗时下降4.5363%，两次B均快于两次A。
两重复不足以声明统计显著性；12步不等于持续容量。源码、配置、数据、
有序batch、RNG和执行因素均经比较工具审计。收据见
[single-b64745b-sinkhorn-abba-m32/comparison.json](single-b64745b-sinkhorn-abba-m32/comparison.json)。
启动/准备/热身按rank单独记录；未清空磁盘编译缓存，不称为空缓存冷编译。
结合cab78a1完整非零LR更新精确校验，保留该优化用于最终优化版本；
目前仍由诊断标志启用，正式配置接入尚待完成。

独立 weighted-Gram GPU probe 源码 c7e89e8525a149026bdb17354e289e42fda0a28a，
18组全部在既定atol3e-6/rtol3e-5内，exit0。两输入梯度逐元素一致，前向
最大差4.47e-8，尚非逐元素相同。B64/L32局部前反向约1.013ms降至0.916–0.932ms，
额外峰值338691072降至209716224bytes；较小形状大多变慢。
这不是整模型收益，不采用为默认。下一步核对64通道归约结合顺序，避免微小误差
经BF16放大；有新依据后才做新的候选和完整更新验证，不放宽容差。
[独立微基准收据](single-c7e89e8-gram-probe/receipt.json)保留所有形状与差异。
两卡当前空闲，未启动持续容量或正式B0。


### Gram 归约顺序假设检验 b3978a2

依据安装Torch Reduce.cuh:1105–1111，64通道不触发输入向量化（阈值128）；
635–670规定下降offset归约。候选显式先配对前后32通道，再归约32通道。
源码b3978a27b36fb7020376380603aca81474e3b044，干净发布收据SHA256
8523d8306d46a4874965fcc56d6a5553e8ea6620d359ed749c0385feaf43b3b3。
本地10测试通过；独立双卡18组probe exit0、均在原容差内，梯度exact。
但前向仍有差异，最大5.96e-8，未证明此修改消除了误差，不能据此采用。
小收据15463bytes经dry-run归档至single-b3978a2-gram-probe/receipt.json。
下一步分离exp、乘积、求和的中间值比较，定位误差后才提出下一候选；
禁止无新证据重复相同修改，不放宽容差。当前所有GPU任务终止，未启动B0。


### Gram 精度定位完成：全部显式归约树通过微基准

分阶段源码ac67ce8f03958754c45dcd65b0eb3b5e982b1ce2，双卡长度17/32结果：
exp、weighted乘积与Torch逐元素相同；Torch显式32→16→8→4→2→1配对
与其sum也相同。证据single-ac67ce8-gram-stages/receipt.json，exit0。
由此继续保留指数和乘法，实现每级显式下降offset归约而非剩余tl.sum。
新候选3784c7fbafa444ff128b7cdd90b0c7f67e91a05c，发布hash
0e5d7c118bfedb9085b44ddb0eb3872067637646aaad1a8a82357c4f5a37a65d。
双卡18组合成微基准全部前向及两输入梯度差0，exit0；原有容差未修改。
两个小收据共17303bytes dry-run复核后入库。已定位被测误差为归约路径，
不能外推至整模型。尚未接入模型默认；下一步candidate-only完整非零LR更新
验证所有梯度、优化器、EMA/center，再真实双卡吞吐/内存测量决定采用。


### Gram完整更新验证接入

候选源df9b3d79dc95a57ab3258c1be098c17e46ed0a4a；新增显式fused_gram参数，
图/self/cross/末尾CLS均通过RelayDeltaAttention的candidate-only标志进入，
Student/Teacher同时启用，默认false。禁止叠加其他执行因素。
新增单chunk/跨chunk初始状态及五项输入梯度测试，CPU替身验证分派等价；
13 Gram+43原方法/更新测试通过，算子mypy通过。CUDA完整更新另行测，
不能把CPU替身测试当成CUDA等价证据。服务器single-df9b3d7-gram-parity启动。


Gram df9b3d7完整更新失败exit1：两rank第一步输入/RNG完全一致，gradient最大差
0.00011191517114639282，超过3e-5/3e-4；未进入第二步非零LR。
objective在LR0相同不构成等价通过；不启动此版本ABBA或正式训练。
两rank失败收据44556bytes dry-run后保存single-df9b3d7-gram-parity/。
下一步在真实调用逐项比较前向，定位最早差异，不放宽容差。
已准备benchmark-only Gram开关与ABBA单因素审计（11测试通过），但未运行。


真实前向诊断e58d96688677497e685175e576aaa96b0d826490已终止exit1。
两rank收据candidate_gram_forward_audit=true，fused_gram_module_count=18；
未触发逐调用前向差异，第一步完成后仍失败于gradient最大差0.00011191517114639282。
输入/RNG相同。仅能证明该次已执行Gram调用前向一致，不能证明整模型等价。
两小收据共44636bytes经dry-run保存single-e58d966-gram-audit/。
下一步检查实际upstream下反向精度和输出布局是否改变后续算子路径；
不再无证据修改前向公式。当前无活跃GPU任务，未开始候选ABBA/正式B0。


### Gram候选暂不采用，进入最终配置与容量阶段

源fd07aeef20c888b6c732c961a555727fcdd86985的真实前向+反向逐调用诊断
终止exit1；两rank明确启用两种audit，18模块，未触发局部不一致，
第一步完整更新仍超差。输入/RNG一致；不放宽原标准。
证据single-fd07aee-gram-audit/两receipt共44718bytes，dry-run后归档。
局部与整体精度边界不同；可能涉及自动求导图的合并顺序等，但尚未证明具体原因，
不把推测写成根因。现阶段淘汰Gram候选，不跑其ABBA/容量/正式训练，
停止无明确新机制依据的迭代。默认继续原Gram，保存候选与失败可复现证据。
最终优化采用已通过严格完整更新及ABBA的mHC融合（吞吐+4.75%）；
接下来将其做显式配置选项，完成发布/同配置校验，再持续双卡容量与完整B0五轮。
此决定不是宣布整体目标完成，正式best/last科学结果仍待产出。


## 可移植性与论文证据约束（用户新增要求）

5090是验证平台，不是方法前提。优化应围绕计算图、final-state KDA、图邻域、
临时张量消除、内存生命周期/重计算、CPU-GPU-通信关键路径形成机制，
说明公式、复杂度及适用边界；设备内核仅为实现。保持公式/loss/采样/更新协议。
正式配置必须保留原生回退；按形状、精度、设备能力/显存选择执行策略，
不得将5090设备参数当成通用假设。调度优化以端到端有效吞吐和关键路径为准。

验证矩阵：当前双5090作为已测平台；另一代或不同显存/带宽GPU待可用资源验证，
不购买或扩大算力。区分跨设备可运行、跨设备加速和论文新颖性。
每个平台覆盖batch、序列长度、邻域大小；分别报告初始化/编译、稳态、峰值显存、
重计算开销和收支平衡点。当前mHC的4.75%仅适用于已测设备配置。
验证包括输出/所有梯度/非零LR多步完整更新/EMA/center和训练结果；
不同设备不要求逐位相同，但必须预先规定数值标准，不能为失败事后放宽标准。
需与既有final-state/chunk/fusion/recomputation工作对照再讨论新颖性，
不得把工程融合直接声称为论文创新。Gram候选未通过完整更新，不算已验证成果。

上述约束叠加于原流程：已验证机制正式接入→持续容量→完整B0五轮best/last，
不替换原目标，不启动其他消融。无额外设备时跨设备加速结论明确列为待验证。


## 文献驱动后续实验

用户新增要求：先广泛检索原始论文及官方实现，再形成优化假设，不做闭门反复微调。
首轮8类来源、实测瓶颈映射、兼容边界、实现核对状态和后续门槛见
[性能工程文献矩阵](GRADPERT_V2_PERFORMANCE_LITERATURE.md)。
下一步优先深入官方KDA chunk和选择性重计算，完成成本判断后再决定新的实验。


## 正式执行选项：Sinkhorn 后端

新增sinkhorn_backend=native/auto/triton；历史缺省仍native且省略旧payload字段，
保留历史配置/检查点身份。新自包含optimized_single_pass_jurkat系列仅新增auto，
原单向KDA、所有模型参数/loss/采样/训练协议不变。auto在NVIDIA CUDA、
四流且Triton存在时选择融合；CPU/MPS/无Triton/其他流数走原生。
显式triton遇不支持环境报错；选中内核后的编译或运行失败不静默回退。
这一规则是可运行策略，不是所有设备数值或加速已验证的声明。
capacity收据记录实际调用的后端模块数；Student与EMA Teacher同构选择。
CPU回退输出/梯度精确、历史身份和十份新配置单因素比较已有定向测试，
完整v2回归及正式配置双卡校验完成前不称为可正式运行版本。
