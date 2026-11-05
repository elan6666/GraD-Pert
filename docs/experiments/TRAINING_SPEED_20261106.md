# GraD-Pert 完整训练更新提速（2026-11-06）

本机物理GPU1、小Jurkat R3-RI / TRAIN40 的完整更新短测中，开启整层邻域校验和固定矩阵复用后，含数据等待吞吐的两次配对速度比分别为 **1.0332640405309328 / 1.0858251139172412**，几何均值 **1.05921860091111（+5.92186%）**。主已独立复算并验收。

采纳的两个执行开关是 `relay_validate_once=true` 和 `cache_kda_constants=true`：前者在完整图层入口保留全部邻域合法性检查，减少分块/重计算中的重复检查；后者按长度、设备、dtype复用不可训练遮罩和单位矩阵，保留原块求解算术与随机流。CPU与内存均保留原来的非绑定策略。

## 可复现范围与结果

计时源为已发布的 `aee9c3a4da1ecbe5c9892724bcca7104340c4bd9`，与起点 `fb2adf48747abb56bf85965c37b790338103271e` 的800个 modeling/training/data/evaluation/config Git blob相同；新增的是诊断入口及启动守卫修复。后续文档/配置打包提交不能回填为本次计时源。

既有R3-RI结构、损失和TRAIN40选择保持；工程拓扑明确为micro80 × accumulation4 × world1 = global320。以下均为随机初始化开始的有限完整更新探针；历史训练权重及其双卡结果分别保留。物理GPU1，Torch 2.13.0+cu130，线程数8；A/C均保持128逻辑CPU可用范围。

| 顺序 | 执行配方 | 含等待 cells/s | 更新中位秒数 | 计时样本 | 计时 Git SHA |
| --- | --- | ---: | ---: | ---: | --- |
| A1 | 基线 | 9.30673341611744 | 33.360627540852875 | 3 | aee9c3a4da1ecbe5c9892724bcca7104340c4bd9 |
| C1 | 两个执行开关 | 9.616312973681756 | 32.21402705507353 | 3 | aee9c3a4da1ecbe5c9892724bcca7104340c4bd9 |
| C2 | 两个执行开关 | 9.786282261335318 | 31.708945791004226 | 3 | aee9c3a4da1ecbe5c9892724bcca7104340c4bd9 |
| A2 | 基线 | 9.012761020078232 | 34.52430435293354 | 3 | aee9c3a4da1ecbe5c9892724bcca7104340c4bd9 |

每臂4步、预热1步，其余3步计时，每步320行。A1↔C1、A2↔C2采用相同有序batch及view RNG。主指标为计时行数除以包含数据等待的总秒数；两次比值的几何均值用于配对汇总。计时前已封存门槛：两个比值均大于1且几何均值至少1.01。该短测是同机同配方的描述性工程结果，未估计统计显著性或完整训练周期收益。

自包含配置：

- [基线](../../configs/v2/training_speed_20261106/baseline/gradpert_v2/nadig_jurkat.yaml)，SHA256 `f01d25b3509960416c8bfd1e0893a27716f48957c3abed0aad01dbe06b9a1eb1`。
- [采纳配置](../../configs/v2/training_speed_20261106/candidate/gradpert_v2/nadig_jurkat.yaml)，SHA256 `43d18b069845571e483a322ff5d1ab085835074734ee134e1a3fdf72f9f6dd32`。两份配置除两个执行字段及其明确来源元数据外逐项相同。

它们是本次 benchmark-only 性能工程复现入口。所有作业均未新增val/test/DEG，也没有恢复旧研究循环。

## 数值与运行证据

- 单卡NCCL完整更新对比：`relay_validate_once` 单因子两步，LR均0.001；预测/CLS输出、损失、全部Student梯度、模型（含Teacher/center）和optimizer最大差均0，输入/RNG一致。`update_parity.py` 仍保留每次仅比较单因子的规则，不能将组合配置直接冒充单因子CLI输入。
- 组合配置另用确定性的原生 `capacity_probe.py --record-update-state`：每臂3步，复用已封存A参考；0..3四个完整状态及每步terms逐字相同，含Student/Teacher/梯度/optimizer/各RNG/缓冲区/温度计数和模式。复制哈希阶段不作为速度证据。
- 独立计时关闭profile、状态复制和额外阶段同步。主独立核对全部12个计时样本、总秒数=更新+等待、960计时行/臂、source/config/环境、样本顺序、view RNG、亲和性及零PKL。
- 本地24项启动/设备分配测试、服务器237项CPU回归通过；图表生产脚本另经实际PNG/SVG渲染检查。

原生计时comparison SHA256为 `91c548028411c20e332a71d45423c6c96c26d79279868f7beba0234db8d7886d`；主验收为 `abe2b3c17fab37918de95857e6482233eb03364be49c5b84f6cc6ea1a0b59d0a`。
服务器原件在 `/data/yilangliu/GraD-Pert/development/training-speed-20261106-config-only-a1/`。

## 已淘汰方案与测量依据

借鉴 [Databricks 的完整路径测量与专用内核思路](https://www.databricks.com/blog/reducing-qwen-training-time-6x-ai-runtime)。本轮基线插桩窗口约59.791秒、2434954次GPU kernel，CUDA活动区间并集约16.445秒；27.50%的区间占比包含profiler开销，不标作未插桩GPU利用率或MFU。

短图融合核在当前配方的完整首步仍未过既定数值门，梯度最大差0.00601743，输出最大差0.81632018；未放宽阈值或采纳。NUMA node3 CPU+内存绑定与两开关组合的另一套6步/预热2步四臂试验，比值0.8052813081537633 / 1.049009365727188，几何均值0.9191015364465123，未通过预设门槛，全部慢样本保留。两个计划分别记录，不跨计划比较绝对耗时，也不以一次空闲GPU快照解释原因。

## 工程接口与边界

`update_parity.py` 支持torchrun单rank或双rank，记录实际world size和物理GPU；单rank仍初始化NCCL并保留原集体通信/完整更新比较路径，单卡显存上限90%。
启动器对exec期间短暂空 `/proc/cmdline` 做最多20次、每次50ms且受原资源门保护的重读；仍不可识别时只清理未回收的直接Popen子进程，已识别路径仍按完整身份清理其后代，早退不无界等待。

所有CUDA作业保持原物理GPU非阻塞锁和selected-device shared-gpu-admission-3政策：UUID、foreign程序限制、90%自有配额、2GiB空闲、owner/后代身份、`PYTORCH_ALLOC_CONF=expandable_segments:True`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`不变。等待由同任务只读监督接管；主独立验收。无跨会话Bridge交接声明。

永久中文报告、两套全部曲线、原值CSV、配置和SHA校验的小收据位于 `experiment-analysis/training-speed-20261106/`；大trace、矩阵和权重留服务器。
图表可由 `scripts/v2/report_training_speed.py <已归档小汇总目录> <输出.png> --title <标题>` 从冻结aggregate复现。
