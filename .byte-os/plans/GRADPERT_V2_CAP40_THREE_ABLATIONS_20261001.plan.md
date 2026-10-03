## 2026-10-03: terminal collector repair and remaining queue

Original queue attempt1 reached E1 formal exit0/six epochs/tests then failed because collect_results did not recognize the already-supported v2_fixed_6 policy. E2/E3 preflights completed in the original attempt; their formal runs never started. Supervisor returned ownership, both monitors paused, main began repair. E1 terminal identity still needs independent server acceptance when connectivity returns.

Repair adds fixed6 acceptance with existing strict history/checkpoint/test/zeroPKL gates. Regression traverses six-epoch joint-only completion through next_action=skip_complete; corrupted budget/history/checkpoint/test remain rejected. New --completed-e1 input imports only validated pinned E1 config/full terminal evidence with identical data/selection/seed and unchanged native src Git tree. Old E1 is never rewritten/retrained. New queue creates fresh E2/E3 plans/run IDs, redoes same-source E2 sustained128 and E3 integration, then their fresh6epoch formals. Original failed controller/source/results remain immutable. No batch/weights/method changes.

110 targeted local tests and Ruff passed. Pending: published clean repair source/runtime; server collector acceptance of old E1/B0; same-source server checks; durable remaining queue launch/verification; explicit supervisor ACK+ACTIVE20min monitor for new queue. SSH reset and Mac locked prevent VPN UI recovery; user asked only to unlock Mac, recovery itself remains agent-owned. No new run launched during connectivity block.

# Jurkat cap40：当前 B0 后的三组独立对照

用户 2026-10-01 指定三组实验；已确认实验②指四个投影头的原型数翻倍，
并纠正实验③的蒸馏② DINO/iBOT 权重为 **1/1**，不是 2/2。
用户最新指令“继续推进消融实验”已授权执行三组：预检、六轮训练、best/last冻结评估及汇总。三组尚未启动，预检准备中。

2026-10-01：前置cap40+mHC B0六轮及best/last已由主会话独立验收；证据见 `.byte-os/evidence/v2-mhc-cap40-six-acceptance-20261001/receipt.json`。两个旧监控均PAUSED。下一实验阶段仍未启动，本次终态验收不扩大为消融运行授权。
当前基线已完成，原run/源码/产物保持历史身份。

## 参照与版本

- 修改前 HEAD 与 GitHub main 已核对同为
  `9b8fd666e8d91b5ca5f79f1ec385775ff594f377`。
- B0 训练源码：`aeac5fd94123af0b73810259e5e2985228b11d65`。
- B0 run：`nadig_jurkat-seed1-20260930T112531Z-f7ca7a77fa6848bb811f706d00e5daad`。
- B0 配置：`configs/v2/mhc_cap40_jurkat/six_epoch_m68_a2/gradpert_v2/nadig_jurkat.yaml`，
  SHA256 `1401174171770bb5c7ae580db55f9dd9d4907d8da6a37a42b86a64d747f412d0`。
- 三组配置复制以上完整配置，不使用继承。E1/E2/E3 各自直接对照 B0，
  不顺次叠加改动，也不从 B0 checkpoint 微调。沿用原模型、训练和评估入口。

## 实验矩阵

两组蒸馏的总系数均为 1。括号中为各项的实际 loss 权重：
蒸馏①按 condition/node/spread 排列，蒸馏②按 DINO/iBOT/KoLeo 排列。

| 组别 | mHC streams | 四个头各自原型数 | 蒸馏① | 蒸馏② | 所回答的问题 |
| --- | ---: | ---: | --- | --- | --- |
| 当前 B0 | 4 | 8192 | (0.8, 0.4, 0.1) | (0.8, 0.4, 0.1) | 完整参照 |
| E1 | 1 | 8192 | (0.8, 0.4, 0.1) | (0.8, 0.4, 0.1) | 关闭 Cell/Response 的 mHC，保留普通残差 |
| E2 | 4 | 16384 | (0.8, 0.4, 0.1) | (0.8, 0.4, 0.1) | 蒸馏原型容量翻倍的影响 |
| E3 | 4 | 8192 | (1, 1, 0) | (1, 1, 0) | 无 spread/KoLeo 且四项蒸馏等权的整套配方 |

E2 的 MLP hidden=2048、bottleneck=256 不变。四个独立原型线性层无 bias，
因此 Student 精确增加 `4×256×(16384−8192)=8,388,608` 个参数；
同构 EMA Teacher 也增加同样数量。所有中心按新原型维度初始化，不能加载旧头。
这只给出参数增量，不据此推断显存峰值或可持续 batch。

E3 是用户指定的**组合配方对照**：同时删除两个正则并修改四项权重。
结果不能单独归因为去掉 KoLeo、去掉 spread 或某一个权重。
关闭损失项不删除编码器和投影头；Student/Teacher 仍同构。
本轮不擅自追加单独的正则/权重拆分实验。

## 固定协议

- Jurkat 同一固定 cap40 清单：47836 训练行、1335 条件，每条件最多40；
  selection SHA256 `aa0916dfb91fff78d7a43f7d7325879a11c1a904e9bf31586420ca11cce84132`。
- 全部 control/validation/test、图先验、GenePT、表达基因轴与测试靶点表达剔除规则不变。
- 从头训练6 epoch、seed1；不早停。目标为每卡微批68×累积2×双卡=全局272，
  每轮176次优化，总1056次；**E2 的该容量尚未验证**。
- 图 scan32/target rows64、序列 scan16；单向写入、所有 query 读取最终 S；
  合法邻域、随机顺序、图来源门、2 Global+2 Local 与遮蔽规则不变。
- GLM5MuonSplit_v2、原 warmup+cosine、weight decay0、dropout0.1 不变。
- E1/E2 的 KoLeo 仍按原全局有效样本语义，排除同扰动条件候选；E3 权重为0。
  其余适用项统一 row_mean；图 node/spread 例外规则保持原定义。
- 每轮 joint_only 验证，Teacher/center 不更新；各组以自己的验证 joint loss 选 best。
  最后一轮 last；最终 best/last 使用冻结300-control测试，三种 Pearson 的
  all/统一DEG版本及已有 seen/unseen 表达分组，完整版本身份与 zero-PKL 收据。
- 原型数和损失权重会改变联合损失的数值尺度。跨组不以 joint 的绝对大小排名；
  报告各分项及实际权重，并比较同协议的 best/last Pearson。
  单 seed 结果是初步对照，不报告跨 seed 显著性，也不以测试结果调参。

## 配置与本地准备证据

配置根目录为 `configs/v2/cap40_ablations_jurkat/`：

| 子目录 | 配置 SHA256 |
| --- | --- |
| `E1_no_mhc` | `5718b2c298aaed31ae796205be5f8597a060da4106e4b0017e35f605dff68686` |
| `E2_prototypes16384` | `5197a3e3d7bfe57a7545b4cc1dad8ff7b5c666dd521bb298cec1ed2d9dabe64f` |
| `E3_unit_distillation_no_spread_koleo` | `6982785ffac1b5ac3f681956ddf8ad51af2aa505e3f02e22572876cb7d34cb2a` |

每个子目录中的完整文件为 `gradpert_v2/nadig_jurkat.yaml`。
本地 schema、V2Options、差异范围核对见
`.byte-os/evidence/v2-cap40-three-ablations-20261001/config-validation.json`。
此收据不代表 CUDA 更新、容量或训练已经通过。

## 下一阶段依赖与验收

1. 原监督会话先完成当前 B0：六轮 journal/history、best/last 真测试、源码/配置/
   checkpoint 哈希、exit0、COMPLETE 和 zero-PKL；终态停止旧监控并交回主会话。
2. 主会话核对终态，按本计划准备 E1→E2→E3 的发布版本与独立运行 ID。
   记录基线与新配置发布 SHA；如只有配置/文档变化，仍核对 src 树一致性，
   明确报告不同发布版本，不能把旧 B0 的 training SHA 改成新版本。
3. 正式开训前分别做同配置双卡完整更新与 checkpoint 重载预检。
   E2 额外做持续容量验证，原型 logits 和 optimizer 状态增大可能导致 OOM。
   测试使用独立 ID、不可变发布源码、GPU0/1 和
   `PYTORCH_ALLOC_CONF=expandable_segments:True`，保留失败证据。
4. 若 E2 无法持续使用 m68，不默默单独降低其 batch、缩小头或删 loss。
   交回主会话制定共同可比的 batch 协议；需要改变参照协议时保留当前 B0，
   单独说明新增匹配参照的必要性和范围，不据单步结果宣称容量通过。
5. 2026-10-01新授权覆盖原设计-only边界，执行三组时沿用已有
   runner/launcher 与 Byte 交接规则。短预检可用同任务 Luna 只读监督；
   长正式训练交 `监督查询对话`，仅有活动作业时启用20分钟检查。
6. 每组完整验收要求六轮、正确 best/last、真实测试结果、完整身份和终态；
   汇总训练/验证各 loss、测试指标、训练/验证/测试墙时、吞吐与峰值显存。
   配置存在或预检通过均不等于实验完成。

当前状态：执行准备；主会话拥有实现与发布。既有B0和暂停的监控保留历史，后续长队列重新明确交接。

## 执行机制与资源等待

代码基线 `7d3b0fe091a7231a0b8106da0cf3fe596d6b0dc4` 已核对GitHub main后开始修改。
新增 `scripts/v2/run_cap40_ablations.py` 仅做CPU控制与编排，复用原resolve_plan、capacity_probe、execute_plan和preflight/completion检查；src训练与模型代码不改。
固定三份批准配置SHA，准备新的run ID；先E1一步、E2持续128步（含checkpoint续跑与单条件300-control验证）、E3一步，再E1→E2→E3正式六轮。
所有预检成功才进入正式队列，E2失败立即停止，不降batch、不删loss。

每个GPU阶段开始前均等待GPU0/1≤512MiB，并取得既有project GPU leases后复核；CPU等待期间每60秒仅采集显存，不干扰其他程序。
每阶段独立log/exit，queue state记录资源等待/子PID/run ID，重复队列目录拒绝覆盖。
allocator固定expandable_segments:True。整个已启动控制队列超过短等待范围时交指定监督会话20分钟检查，避免主会话与Luna重复监督。
只有控制队列实际启动并存活后启用监控；正式组结束后自动下一组，错误立即停止由主会话诊断。
主返回提醒保持pending精确状态，不能再用pending_fallback_active破坏guard。
