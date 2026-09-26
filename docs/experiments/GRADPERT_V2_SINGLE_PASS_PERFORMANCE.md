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
| 一批 CPU 数据/视图预取 | 双卡两次完整更新输入/RNG一致，损失、梯度、模型/Teacher/center及优化器比较差值0，[rank0](single-ec2384a-prefetch-parity/rank-0-receipt.json)、[rank1](single-ec2384a-prefetch-parity/rank-1-receipt.json)，源码ec2384ade9fa66458eac2ff40c2dbff2fd067dc8 | ABBA吞吐对照进行中；尚未启用默认 |

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

## 尚需完成

1. 核对CPU预取ABBA四份完整收据：同配置、数据/样本/视图RNG、线程与设备；
   比较含等待总耗时、更新耗时、显存和两组重复，不能只比较kernel片段。
2. 按证据选择执行方式。已准备micro8/16/32/48/64双卡、累积2容量候选；
   若最高仍通过则继续扩展，失败则细化边界。先单步定位，再128+持续更新、
   checkpoint续跑及300-control验证。旧模型batch192不构成当前方法的证据。
3. 在选定较大batch重新检查预取收益，锁定干净发布版本与自包含配置。
4. 仅启动已授权完整B0五轮，完成best/last测试与最终指标/身份核验；不启动
   prediction_only、其他消融或恢复旧停止运行。

当前进程及路径以 [Byte状态](../../.byte-os/STATE.md) 和实时服务器收据为准。
