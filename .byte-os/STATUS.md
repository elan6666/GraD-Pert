## 2026-09-29：新 v2 默认 joint 验证、三组 best/last 测试与实时进度已发布

源码 `10193b6edffca802410fef837fd918844d4cc6e3` 已推送 `main`。后续**新** v2 运行按固定 validation 视图计算完整预测＋SSL1＋SSL2 的 joint loss，并据此选择 best；原有 300-control prediction loss 和三种 Pearson 仍独立保存。每次成功训练更新累计真实数值表达基因，best/last 分别据各自 epoch 把同一次全基因测试推理汇总为全基因、见过表达、未见表达三组，每组各有 TxPert/TriShift/Systema Pearson 和有效条件数。实时文件/查看脚本显示 epoch 内与总 step、百分比、cells/s、step/s、joint 验证批次及 best/last 测试进度。协议与边界见 `docs/experiments/GRADPERT_V2_JOINT_VALIDATION_EXPOSURE_DEFAULT.md` 和 `docs/experiments/GRADPERT_V2_LIVE_PROGRESS.md`。v1、R50、旧 v2 运行和旧选择收据不追改。

隔离服务器副本 v2 回归 434 通过；数据/曲线/选择相关 48 通过（另有 1 个既有配置配对测试因缺少旧配置而跳过）；最终改动定向 37 通过。本地 Ruff check/format 全仓通过。`mypy src` 尚有 29 个既有错误，均在本次未改动的 Triton、reductions 或旧训练 step 文件；本次新增的 2 个类型错误已修复。未启动新训练，当前没有此任务的 GPU 作业；后续若执行正式运行，须以本新源码、独立配置身份和新运行 ID 重新预检，不覆盖旧产物。

## 2026-09-28：best.pt 表达可见性评估完成；v2 实时进度已发布

三轮 B0 的 best.pt 分组评估已 exit0，完整结果与身份见 `docs/experiments/GRADPERT_V2_EXPRESSION_EXPOSURE_20260928.md`。训练期见过表达的 4,775 个基因 TxPert/TriShift/Systema Pearson 为 0.422760/0.432257/0.257070；未见表达的 225 个基因为 0.138621/0.172251/0.119023，后两项仅 137/592 个条件有效。服务器结果 SHA256 `603ffa7e2cba7f78e53f0566a73b56fdceaab0af00eb99ba1fe791bc84e8f840`；原始全基因 best 指标逐项保持一致。旧训练/评估运行和产物不改，监督心跳已暂停。

后续新 v2 运行的实时进度功能已发布为源码 `87b44c8c1e8840c876bb63cdf783486aa40f7364`：每步写 `fit/live_progress.json`，记录 epoch 内步数、loss、吞吐，验证及 best/last 测试写条件数；查看命令见 `docs/experiments/GRADPERT_V2_LIVE_PROGRESS.md`。隔离服务器副本相关 17 项测试通过，v2 全套为 429 通过、1 个既有配置配对测试失败（当前基线提交缺少其引用的旧 `single_pass_jurkat/one_epoch_m66_a2` 配置），与进度功能无关。未启动新训练。

## 2026-09-28T04:39:59+08:00：m74 正式三轮 OOM 后的容量回退

m74 已在首轮中途 OOM，128 步通过不能证明整轮安全；旧 run 和 FAILURE 保留，不恢复。当前用已发布来源门版本 `8dfb267adb25591393602066ef1226e7cbb76174`，对 m68／图目标行64／序列chunk16 启动独立双卡128步持续探针，服务器根 `/data/yilangliu/GraD-Pert/development/v2-gate-m68-r64-s16-128-after-b0-oom-r2`；r1 因启动环境缺少 PYTHONPATH 在导入阶段失败，r2 已补齐。准备了新的三轮 m68 自包含配置，但须以探针终态、显存余量与吞吐决定是否采用；不能把进行中视为通过。本次只改变物理/全局 batch，完整预测＋SSL1＋SSL2、累积2和既定图/序列chunk不变。


## 2026-09-28T04:34:09+08:00：三轮 B0 attempt1 因 OOM 失败，交回主会话

精确运行 `nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143` 退出码1，`fit/epoch_state.json`停在0/3（434 updates/epoch），无epoch-0001、history、COMPLETE、best/last测试或test root；全run根扫描0个PKL。rank0 在GPU0上 dropout 申请128 MiB时OOM，报告仅余9.62 MiB，进程占31.34 GiB；rank1随后在NCCL ALLREDUCE seq5573超时300027 ms并SIGABRT。训练进程已全部退出，双卡回到0%/2 MiB。源码与配置身份匹配。failure JSON原件保存在服务器且19,977字节已核验SHA；小型副本和日志哈希索引记录在 `.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-terminal-failure-20260928.json`。本run不重启、不评估、不修改配置；等待主会话决定新的有界步骤。监督心跳暂停并交回。详细handoff：`.byte-os/coordination/handoffs/2026-09-28-b0-m74-three-epoch-oom-return.md`。


## 2026-09-28T04:12:23+08:00：服务器连接恢复，三轮 B0 仍在训练

TCP/22与SSH已恢复。精确进程 wrapper3829919、torchrun3830019、rank3830095/3830096均存活；`epoch_state`仍0/3（434 updates/epoch），history为空，只有epoch-0000初始checkpoint；无exit、COMPLETE或failure，测试根尚未创建。源码干净且提交/config SHA匹配。GPU0 29%/31944 MiB，GPU1 29%/32084 MiB，compute PID与两个rank对应。有限日志无错误；继续按20分钟监督。收据：`.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-supervise-check-20260928T2011Z.json`。


## 2026-09-28T03:53:52+08:00：服务器连接暂不可用

当前 EasyConnect 服务门户已登录且列出 `10.24.1.91`，但该资源显示 L3VPN 服务启动失败；辅助检查 TCP/22 与 SSH 均不可达。执行一次常规恢复并从已填表单登录后，页面仍报告启动失败；未做路由/DNS/代理更改。训练状态未知；上次可达观察为 epoch 0/3、进程存活且无 exit/COMPLETE/failure，不可视为当前状态。连接证据：`.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-connectivity-incident-20260928.json`。主会话已被通知；监控保持 ACTIVE，下一周期再做有限连接核验。

## 2026-09-28：三轮完整 B0 已启动，交监督会话

已发布源码 `e821173b4d11268b621ac8299e45b8d6d272b7e7`，服务器干净checkout和发布收据校验一致；86项相关测试、同三轮配置双卡单步通过。正式运行 `nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143` 已在双RTX5090启动：预测＋SSL1＋SSL2、每卡微批74×累积2＝全局296、3 epoch；运行根 `/data/yilangliu/GraD-Pert/runs-v2-b0-gate-m74-3ep-e821173/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143`，日志 `/data/yilangliu/GraD-Pert/development/v2-b0-gate-m74-3ep-e821173-formal-r1.log`，父PID3829919。启动核对见 `.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-launch.json`。当前只有启动证据，尚无完成、best/last结果；旧B0不恢复。后续20分钟精确运行监督与终态交回主会话，手册 `.byte-os/coordination/handoffs/2026-09-28-codex-b0-m74-three-epoch-supervise.md`。以下选型段落为训练启动前状态，由本节覆盖。

首次监督确认（2026-09-28T03:12:05+08:00）：Byte 所有权与 run ID/attempt 门禁匹配；服务器 wrapper、torchrun 与两个 rank 均存活，epoch 0/3；服务器源码干净且提交/配置哈希匹配。GPU0 为18%/30584 MiB，GPU1 为0%/30564 MiB。未见 COMPLETE、failure 或 exit 标记。20分钟心跳已启用。记录：`.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-supervise-check-20260928.json`。


## 2026-09-28：来源门 B0 选型完成，准备独立三轮正式运行

已发布源码 `8dfb267adb25591393602066ef1226e7cbb76174` 的双卡完整损失容量测试中，m74／图行64／序列chunk16完成128/128步、checkpoint续跑和一个验证条件300 control推理；峰值预留32.621GB/卡，含数据等待13.450 cells/s。更大m76/78/80分别在13/6/1步后OOM。12步同协议吞吐对照中m74两次13.794/13.791 cells/s，旧默认m66／序列32为12.687，约快8.7%；m70／序列32在第2步OOM。选定每卡微批74×累积2×双卡＝全局296，图扫描32、图行64、序列扫描16。用户授权下一步完整B0独立3 epoch，best/last测试；三轮策略与配置已在本地准备，待干净提交推送、服务器同源码测试和启动预检后运行。当前没有由本任务运行的GPU作业。旧一轮／五轮B0停训收据不改。详见 `docs/experiments/GRADPERT_V2_GATE_BATCH_CHUNK_20260928.md`。下方旧状态由本节覆盖。

## 2026-09-28：来源门新版本最大 batch／chunk 配对测试准备中

本阶段固定双卡、累积2次和完整预测＋SSL1＋SSL2，查每卡物理微批的持续容量，不靠累积次数虚增。锚点是新来源门＋原 scan32/序列32/图行64、m66（全局264）单步通过；旧版本 m66 128步通过、m68 第22步OOM只作先验。干净源码`f090b726ba0df0e02a2fb058e727236c9d9dfcef`第一组九项单步预检均通过：m70 seq32峰值预留32.35GB/卡，m70 seq16降至28.90GB；行块32没减峰值且明显减速，行块96/128在m66首步更快。已追加seq16的m72/74/76/77/78/80及seq32的m72查上界。当前只有单步证据，下一阶段须经128步持续验证及同batch稳态吞吐对照；配置、收据和边界见 `docs/experiments/GRADPERT_V2_GATE_BATCH_CHUNK_20260928.md`。本任务不启动正式B0。下方“后续默认”是进入本容量阶段前的状态。

## 2026-09-28：后续 v2 默认恢复原 chunk，新图来源门开启

新自包含 Jurkat 默认配置 `configs/v2/source_key_gate_jurkat/one_epoch_m66_a2/gradpert_v2/nadig_jurkat.yaml` 只比先前一轮 m66 配置增加 `graph_source_key_gate=true`；图扫描/序列扫描沿用32，图目标行块64。旧配置与大chunk工程对照不改。此前 batch 为每卡66×累积2×双卡，即全局264；新门控在同一batch双卡单步优化器更新和checkpoint重载通过，峰值预留31.43GB/卡，但尚未做128步持续容量。发布源码 `d2410efdc81a9690644ef2d165c2dea374ba0fbb`，服务器81项定向测试通过，单步收据 `/data/yilangliu/GraD-Pert/development/v2-gate-default-d2410ef-m66-integration-r1/receipt.json`。未启动正式一轮B0；先前已停B0不恢复。方法默认与容量证据边界见 `docs/design/GRADPERT_V2_RELAY_METHOD_PLAN.md`。

## 2026-09-28：v2 第四层来源门与 chunk 工程测试完成

图第 4 层稀疏 MLA 已支持零初始化的四位来源逐维 key 门，指定图扫描64、序列扫描256及图目标行块64/96/128均有自包含配置。代码与 m16 对照配置已分两次提交推送 main；最终测试源码 `2043e09a4f4cb29317b9879da68e32e8d953ee1c`，服务器80项定向测试通过，双卡4步完整损失测速均通过。m32×累积2时指定chunk反向OOM；统一 m16×累积2×双卡（全局64）下，参考/完整组合分别为3.373/3.201 cells/s，row128组合为3.963 cells/s；反向顺序复测参考/row128为3.376/3.978 cells/s。指定大chunk显存约26.39 GB/卡，参考约9.14 GB/卡。行块128短程有速度收益，但尚未通过逐目标随机语义、严格数值和持续容量验收，不能默认作为等价优化；指定扫描chunk也未产生端到端收益。未启动B0或其他消融，旧B0保持停止。完整身份、失败和通过收据及采用边界见 `docs/experiments/GRADPERT_V2_SOURCE_GATE_CHUNK_20260928.md`，下方旧chunk决定仍为历史对照。

## 2026-09-27：v2 chunk 对照完成；默认分块不变

旧完整 B0 已按用户立即停止：epoch 0/5、无 best/last 或 COMPLETE，原训练 SHA、配置和 run 保留。后续新一轮配置已把蒸馏②总系数改为 λ₂=1，内部 DINO/iBOT/KoLeo 为 0.8/0.4/0.1，不追改旧运行。独立版本 `d61af46d9e4282c8e5b69e9514d1134407d2b8e5`（干净服务器源码和发布收据齐全）通过 82 项定向测试。双卡完整更新下，KDA 扫描 48/64（全路径或仅长序列）均未过固定梯度/optimizer 容差；确定性 32/32 重复各项差值为 0，确定性 48/64 仍失败。图行 96/128 在 m32 的短程端到端 ABBA 有描述性收益，但同种子随机分配和 BF16 输出未过严格语义验收。因此没有采用新 chunk，默认扫描32、图行64；没有做不合格候选的持续容量或重启 B0。完整数字、源码/config 身份和服务器收据见 `docs/experiments/GRADPERT_V2_CHUNK_SWEEP_20260927.md`。当前无活动 GPU 任务，旧跨会话监控保持暂停。下方记录为历史快照，本节优先。

## 2026-09-27：m68 持续容量失败，回退测试 m66

干净发布源码 `28f447a24a400659ea09f9c0f6d735ea7e4ec8f1` 的 NUMA 本地 m68×累积2×双卡（全局272）单步预检通过，但128步持续测试的 rank1 在第22步反向传播 OOM（需再分配120 MiB，卡上仅余53.62 MiB）；`rank-1-failure.json` 记录 failed/22，顶层 `receipt.json` 是较早落盘的 running/16，不能当作通过。退出码1，两卡已空闲；原 run ID、日志、收据保留于 `/data/yilangliu/GraD-Pert/development/v2-numa-m68-28f447a-capacity-20260927T0929Z`。下一步只改 batch 建立 m66×累积2×双卡（全局264）自包含配置，发布新 Git SHA 和不可变服务器源，以新 ID 做单步预检及128步持续容量；若失败，回退已在旧版通过的 m64，并在选定新发布源码上重验。完整 B0 五轮及 best/last 等容量通过后再以全新 ID 启动。此条覆盖下方 m68 进行中状态。

## 2026-09-27：m68 双卡持续容量正在运行

干净发布 `28f447a24a400659ea09f9c0f6d735ea7e4ec8f1` 的自包含 m68×累积2×双卡、全局272配置 SHA256 `ab8c358ecdd7d6ee398120ebcaafd048bb25814263eb5b2e8271f479662735a9` 已通过新 ID 单步预检1/1、exit0（仅工程预检，不含300-control），收据 SHA256 `99f32bfe4ca9fde658977903fbf99e6b32f7e82922571a8c3135b3cdb38364c7`。当前新 ID `/data/yilangliu/GraD-Pert/development/v2-numa-m68-28f447a-capacity-20260927T0929Z` 执行128步持续测试，父PID文件同stem `.pid`（启动为3689686），NUMA本地绑定；需要exit0、128/128、checkpoint续跑与300-control才可接受。Luna子代理只读监督短时任务。旧B0保持中止，正式五轮尚未启动。此条覆盖下面旧状态。

## 2026-09-27：NUMA 调度验证通过，进入新容量边界测试

双卡 m64×累积2、全局256的 A1/B1/B2/A2 各12/12完成且队列 exit0；B 只使用 PyTorch 内建 `torchrun --numa-binding=node`。两次配对含数据等待吞吐 B/A 为 1.05240 和 1.04927，几何均值 1.05083；A1/B1/B2/A2 为 11.7784/12.3956/12.3440/11.7644 cells/s。峰值预留显存四组均 28,974,252,032 bytes；启动分别 8.35/7.99/7.59/8.12 秒。比较收据 `/data/yilangliu/GraD-Pert/development/v2-numa-32b2bfd-abba-20260927T0903Z/comparison.json`，SHA256 `813bd55bef4715ceb430944d251cc7ee7e319fbd29043996405d0cb3c2886ed4`。两次计时是描述性证据，未证明长期模型效果或逐位更新相等。采用 NUMA 绑定作为这台双5090服务器的执行策略；不改模型数学与训练配置。下一步发布独立 m68（全局272）容量配置，完成128步持续、checkpoint续跑与300-control；若 OOM 保存收据并回退。旧 B0 已按用户要求停止，不恢复；后续只用新 ID 完整预测＋SSL1＋SSL2 五轮及 best/last，不启动其他消融。此条覆盖下方历史快照。

## 2026-09-27：短图融合完整更新失败；转入原算术路径常量复用

`27aec3643cc613b17f3b87eecebf05d1453743ba` 已推送 GitHub main，并在独立干净服务器checkout及哈希封存发布收据后做双卡确定性完整更新。单步损失、图梯度、中心和优化器差异超出固定阈值，失败两rank收据 `/data/yilangliu/GraD-Pert/development/v2-short-27aec36-parity-m2-datafix-20260927/` 保留；此短图融合候选淘汰，不做吞吐/容量/B0。新 opt-in 的 KDA 形状常量复用保持原块求解，相关60项严格输出/梯度测试通过，局部BF16前后向2.738→2.632ms；仍未完成双卡完整更新和端到端吞吐。旧 B0 仍中止。Byte计划和性能报告记录失败与下一门槛。

## 2026-09-27：用户重新授权性能工程，原 B0 保持中止

当前阶段仅做性能调研、关键路径实测和等价机制优化；之后按新源码/配置进行双卡持续容量测试，再用全新运行 ID 执行已授权的完整 B0 五轮和 best/last，不启动其他消融。已开有界 Goal；短时 GPU 测试由本会话 Luna 子代理只读监督，不重新启用旧跨会话定时任务。路线和验收见 `.byte-os/plans/GRADPERT_V2_MECHANISM_PERFORMANCE_20260927.plan.md`。旧 `681d4fb609644d51c22f6b4e51dd61256adc3310` 的 m64 全局256容量已有128/128通过收据（20.183秒/步中位，11.8629 cells/s含等待，数据等待6.23%），m72在2/128 OOM。新 m32 双卡完整更新 profile `/data/yilangliu/GraD-Pert/development/v2-mech-681d4fb-m32-profile-20260927T0727Z` 已 exit0、6/6 passed；配置 SHA256 `d4f5b9cdb6279e6ed74c2777cf942a95b79a048154d6678ecc4fa9e77c5e477c`，普通更新中位17.550秒、含等待6.812细胞/秒、峰值分配/预留15.324/15.576GB每卡；末步profile29.609秒已排除在稳态外。rank0窗口30.839秒、GPU kernel并集7.182秒、1305663个kernel，图前向和反向细粒度发射明显；GPU并集不是未profile利用率。图邻域6506节点/253371条边，长度中位44、最大47，统一填充17.14%。短邻域融合最终状态机制的FP32合成单块前向0.540→0.0366ms、前后向3.636→0.804ms；首版BF16 value梯度有少量舍入边界失败，保留该证据。BF16 value梯度暂沿用原块求解的混合候选在服务器相关测试59项通过，局部前后向2.803→1.782ms；只是局部探针。当前在独立不洁临时checkout执行全部v2测试，未发布、未应用到正式训练，双卡完整更新及吞吐尚未验证。

## 2026-09-27：用户要求停止 B0 训练与定时监督

完整 B0 消融基线运行 `nadig_jurkat-seed1-20260927T041339Z-e35f5edb8e6248479462589ff3cef41f` 已按用户要求停止；仅向核对过命令行的 torchrun PID 3613061 发送 SIGTERM，其父进程及双 rank 均已退出，双卡现为 0% 利用率、各 2 MiB。退出码 1 是主动中止的结果，不是模型故障。`fit/epoch_state.json` 仍为 epoch 0/5，仅有初始 `epoch-0000.pt`；没有 `COMPLETE.json`、best/last 检查点或测试结果。服务器运行目录、日志与检查点原样保留。中止收据：`.byte-os/coordination/receipts/b0-681d4fb-user-stop.json`，SHA256 `9e792365f37ec635483eb1da5a61dc993e65fa2bb30298e1bc68f09153af5451`。

监督会话已停止查询并将 Byte 所有权交回主会话；`grad-pert-v2` 与 `grad-pert-v2-b0-return` 两条定时任务均为 `PAUSED`。此状态覆盖下方“已启动／交监督”历史记录。不要自动恢复该运行、启动新实验或重新启用定时监督，除非用户重新要求。

## 2026-09-27：m64 容量通过；完整 B0 五轮已启动

m64 双卡持续容量已 `exit=0`、`passed 128/128`，完成 checkpoint continuation 与 300-control `inference_shape=[300,5000]`，validation prediction loss `0.0009303691`；容量 receipt SHA256 `31a0798737a60bd318a861fd5e4d947a51f4d943ac0f613d0328ca4a3a7bfc04`。同一干净发布源码 `681d4fb609644d51c22f6b4e51dd61256adc3310` 和配置 SHA256 `23a2942804b73cdf92871e36808846f50382ded41b1588a68da76ebc0e69985b` 的完整 B0（预测+SSL1+SSL2）已启动，run ID `nadig_jurkat-seed1-20260927T041339Z-e35f5edb8e6248479462589ff3cef41f`，父 PID `3613045`；双卡、micro64×accum2、全局batch256、5 epoch、502更新/epoch。首次实查父进程、torchrun、双 rank、run manifest 和 `fit/epoch_state.json` 均存在，epoch 为0/5；训练及 best/last 尚未完成。长时监督交接和收据见 `.byte-os/coordination/handoffs/2026-09-27-b0-681d4fb-supervise.md`。本条覆盖下方容量测试进行中的旧状态。

## 2026-09-27：m64 容量测试 32/128，已交回主会话监督

m72 运行 `v2-teacher-681d4fb-m72-capacity-20260927` 因双 rank `torch.OutOfMemoryError` 在 2/128 失败，原运行、收据与日志保留。授权的 m64 新 ID `v2-teacher-681d4fb-m64-preflight-oom-20260927-1201` 双卡 integration-only 预检通过 1/1、exit 0，峰值 allocated/reserved 27,632,761,856/27,927,773,184 bytes。持续测试新 ID `v2-teacher-681d4fb-m64-capacity-oomfallback-20260927-1203`，PID 3602712；2026-09-27T04:17:04Z 实查进程及双 rank 仍运行，receipt 为 running 32/128、无 failure/error、无 exit 文件，两卡显存 29,796/29,878 MiB。源码 SHA `681d4fb609644d51c22f6b4e51dd61256adc3310` clean，配置 SHA256 `23a2942804b73cdf92871e36808846f50382ded41b1588a68da76ebc0e69985b`。按用户新流程（预计约1小时的测试由主会话和子代理监督）已把 owner 交回主会话，并暂停监督会话心跳以避免重复监督。详情见 `.byte-os/coordination/handoffs/2026-09-27-teacher-m64-capacity-return-main.md` 与容量收据。未启动 B0。

工作流更新：短时测试使用本会话子代理监督；长时正式训练/消融才用跨会话监督。主会话在监督报告终态后直接执行已授权的下一步；跨会话定时心跳完成交接后暂停或删除，需要新长任务时再启用。当前心跳 `grad-pert-v2` 已暂停，只有本会话子代理监督 m64；B0 完整五轮的启动计划已封存，等待本次容量 acceptance。

## 2026-09-27：m72 双卡128步容量测试已启动

Teacher Sinkhorn 优化双卡ABBA comparable，描述性总耗时下降3.84%，保留；m72×2×2=全局288单步预检1/1 passed，峰值预留显存31,279,022,080 bytes。已在干净发布源码681d4fb609644d51c22f6b4e51dd61256adc3310上启动m72的128步持续容量工程测试，PID3599854、运行根`/data/yilangliu/GraD-Pert/development/v2-teacher-681d4fb-m72-capacity-20260927`；当前只确认进程和0/128初始收据。监督交接见`.byte-os/coordination/handoffs/2026-09-27-teacher-capacity.md`。容量/B0仍未完成；此节覆盖下方历史空闲记录。

## 2026-09-27：Teacher Sinkhorn ABBA 已完成，交回主会话

队列 PID 3531359 于 `2026-09-26T20:56:37Z` 结束，`queue.phase=complete`、`queue.exit=0`；A1/B1/B2/A2 四份收据均为 12/12 passed（每组预热3步），运行根 `/data/yilangliu/GraD-Pert/development/v2-teacher-681d4fb-abba-20260927`。四组源码均为干净提交 `681d4fb609644d51c22f6b4e51dd61256adc3310`，配置 SHA256 `85d4c571727950bd8af0e3e34dd2612d25b20f4b16e578ebe478bb70cbaeab14`；有序 batch 计划、batch schedule hashes、view RNG 起点、数据根和双卡配置一致。唯一切换因子 `save_no_grad_sinkhorn_diagnostic_only` 为 A1/A2=true、B1/B2=false；`comparison.json` 标记 `status=comparable`、`execution_factor=teacher_no_grad_elision`、两组 B 均快于配对 A。

两组 A 的含等待总步中位数为 20.7562/19.4486 秒，两组 B 为 19.2542/19.2487 秒；A/B 总耗时比 1.0399、总耗时下降 3.84%，峰值 allocated/reserved 显存相同（3,505,913,344/3,867,148,288 bytes）。每种只有两次运行，属于描述性性能观察，不代表显著性、数值等价、持续容量或模型效果。它是 benchmark-only ABBA，不是正式训练完成；持续容量和 B0 五轮均未启动。完成证据与限制见 `.byte-os/coordination/handoffs/2026-09-27-teacher-abba-complete.md`。GPU 当前空闲；所有权已交回主会话决定下一阶段。

## 2026-09-27：主工程／监督查询双会话交接

用户指定主会话 `01a0c01a-0611-7a90-b3e8-8ad7e017748b` 做设计、实现、验证和启动，监督会话 `01a0df0b-4142-7df1-86c0-d959471d80a1` 在后台等待阶段每 20 分钟检查，并在发现问题或完成时立即唤起主会话。心跳 ID `grad-pert-v2` 已建立；当前无活动 GPU 任务、无交接中的后台运行。Goal 模式关闭。具体交接合同与下一步见 STATE.md；旧记录中的 Goal/监控状态为历史快照。

## 2026-09-26：正式auto配置双卡完整更新精确通过


### 显式后端配置完整更新通过

9d9bad86e16286b590b2bb31c5df5103ecfbbccb干净发布服务器source-v2-opt-9d9bad8，
publication SHA256 da83f52334078f742997eb4d2e3c850abba1b5234c13e09bfaceaa7688977294。
同profiling_m2_a2协议，仅candidate配置从native改auto；无诊断融合开关。
两rank两步全部比较最大差0，输入/RNG相同，第二步LR7.796055196070788e-8。
exit0，收据single-9d9bad8-config-parity/共42414bytes经dry-run归档。
这验证正式配置接入等价；不是新的吞吐或持续容量结果。当前GPU无活动任务。
下一文献指导候选是Teacher无梯度Sinkhorn的无用中间存储消除，详见文献矩阵。

## 2026-09-26：可移植执行配置通过349测试；文献驱动后续机制

mHC新增显式后端及原生回退，十份新配置仅切执行选项；349v2测试、mypy/ruff通过。
文献矩阵与FLA冻结接口检查已记录；新配置GPU校验和后续机制筛选尚待完成。
未启动容量长测或B0，原目标保持。

## 2026-09-26：Gram候选不采用，推进已验证优化与容量


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

## 2026-09-26：前向诊断缩小故障范围


真实前向诊断e58d96688677497e685175e576aaa96b0d826490已终止exit1。
两rank收据candidate_gram_forward_audit=true，fused_gram_module_count=18；
未触发逐调用前向差异，第一步完成后仍失败于gradient最大差0.00011191517114639282。
输入/RNG相同。仅能证明该次已执行Gram调用前向一致，不能证明整模型等价。
两小收据共44636bytes经dry-run保存single-e58d966-gram-audit/。
下一步检查实际upstream下反向精度和输出布局是否改变后续算子路径；
不再无证据修改前向公式。当前无活跃GPU任务，未开始候选ABBA/正式B0。

## 2026-09-26：真实输入前向首差异诊断启动

e58d966独立版本、父PID3455848，parity阶段存活。只诊断不跑吞吐。
失败版本已保存，下一步据实际形状/误差定位；详见STATE。

## 2026-09-26：Gram完整校验失败，不采用


Gram df9b3d7完整更新失败exit1：两rank第一步输入/RNG完全一致，gradient最大差
0.00011191517114639282，超过3e-5/3e-4；未进入第二步非零LR。
objective在LR0相同不构成等价通过；不启动此版本ABBA或正式训练。
两rank失败收据44556bytes dry-run后保存single-df9b3d7-gram-parity/。
下一步在真实调用逐项比较前向，定位最早差异，不放宽容差。
已准备benchmark-only Gram开关与ABBA单因素审计（11测试通过），但未运行。

## 2026-09-26：Gram候选接入完整更新校验

默认路径未改。56本地定向测试和类型检查通过，独立源df9b3d7已发布核验，
双卡完整更新校验PID3453868启动；只变更Gram融合，未启动正式B0。
当前路径、收据及下一步见STATE。

## 2026-09-26：Gram 精度差异已定位，18组前向与梯度精确通过

显式全部下降offset求和树后，两卡18组全部前向及两输入梯度差0。
尚待candidate-only整模型更新与吞吐验证，不将算子微基准当作正式采用。
无活跃GPU任务；完整任务目标继续。证据与后续见STATE和性能报告末节。

## 2026-09-26：Gram 归约修正假设完成独立检验

b3978a2双卡18组通过既定容差但未达到精确前向（最大5.96e-8），不采用。
需继续定位中间值差异；mHC已证实收益保持，未启动容量长测或B0。

## 2026-09-26：mHC ABBA 完成，吞吐提高4.75%；KDA微基准完成

四组12/12、exit0且完整比较审计通过；含等待步耗时下降4.54%。
保留mHC融合优化，正式配置接入待完成。原始小收据已归档。
KDA Gram双卡18微基准通过既定容差，输入梯度exact但前向最大差4.47e-8；
小形状偏慢、大形状局部改善，尚未接入模型或采用。下一步核对归约顺序。
所有GPU任务已终止，两卡空闲；持续容量、完整B0五轮与best/last尚待完成。
详见性能文档最新节；此记录覆盖下面历史“运行中/尚未GPU测试”状态。

## 2026-09-26：ABBA进入A2，本地回归通过

A1/B1/B2均完成12步，update中位19.9003/19.0998/18.9844秒；A2运行中。
已核对A1/B1源码、config、data、ordered batch及视图RNG起点一致，完整审计待四组。
本地v2回归340passed；补齐新CUDA候选的类型接口，20定向测试与4文件mypy通过。
KDA候选未执行GPU测试、未接入默认；当前GPU仍只用于ABBA。

## 2026-09-26：动态N完整更新精确通过，代表性batch吞吐对照启动

cab78a1两rank两更新（含非零LR）loss/gradient/objective/optimizer全部差0，
30微基准全部阶段exact且每GPU只有1forward/1backward编译variant。
全局128的ABBA已启动，父PID3442646，single-b64745b-sinkhorn-abba-m32。
每组12步3热身9计时、完整双蒸馏，同配置只切换融合；新增冷/稳态分开记录。
还没有真实训练加速结论或正式B0启动。路径/hash/下一步见STATE.md。

## 2026-09-26：完整校验否决首版，定位归约精度并重测

67fd8e1完整更新失败，损失/梯度超原容差，输入RNG一致；不采用、不跑其吞吐。
根据Torch原生strided reduction实现修正行轴求和结合顺序，7ae9a8b独立18组
全部40阶段/输出/梯度逐元素相等。修正后完整校验PID3439468进行中，未宣布通过。
当前路径/后续动态N编译问题见STATE.md。保持原eager默认，无B0或其他消融启动。

## 2026-09-26：mHC融合微基准通过，完整更新校验运行中

单向KDA/final-state读取保持；新融合候选仅针对mHC 20轮Sinkhorn执行，默认未改。
两张5090共18合成case输出与梯度通过，局部前反向约7–8倍提速，不能外推整模型。
61本地相关测试及18服务器定向测试通过。首版Triton scope失败保留，已修。
完整双卡校验已启动：PID3436999，single-67fd8e1-sinkhorn-parity，
源码67fd8e1883ecb5d1145fe179d797347057f29394，两边同config，全局8，完整双蒸馏。
先比对两次完整更新再端到端ABBA；状态/路径见STATE.md。未启动B0或其他消融。

## 2026-09-26：单向方法继续性能目标；关联归因进行中

全量trace统计完成并入库：1414510次GPU kernel、43200次logsumexp，
后者对应mHC Sinkhorn。累计耗时非互斥关键路径，未宣称任何新加速。
已发布301be18 CPU scope/launch/kernel三遍关联分析工具，6定向测试通过；
服务器CPU任务PID3432732，900s有界，stem single-301be18-trace-attribution-rank0。
无GPU任务；持续容量/B0待机制优化验证。当前详情见STATE.md与性能报告。

---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: single_pass_performance_preflight
current_workflow: byte-auto
next_workflow: byte-auto
review_verdict: single_pass_locally_validated
hard_blocked: false
updated_at: 2026-09-26
---

# Current state

## Resumed single-pass performance engineering — 2026-09-26

User steering prioritizes mechanism-depth KDA optimization before sustained capacity/B0.
Sweep3429104 exit1:72passed/80OOM/88notrun; no activeGPU job. Receipts saved.
Route in performance report: quantify existing trace, fused decay-Gram fwd/bwd
without expanded pair-channel temporary; not yet implemented. Byte state corrected.

CPU prefetch ABBA complete exit0, all12/12 and comparable identity audit.
Mean inclusive step increases2.4476%; do not adopt. Saved reviewed356,094byte
receipts and comparison under docs/experiments/single-ec2384a-prefetch-abba/.
Immediately launched current-method capacity integration sweep PID3427171,
stem development/single-a1d55fa-capacity-integration, source/publication below.
Both GPUs verified idle, locks held,1800s timeout, micro8/16/32/48/64, accum2.
Only one-step boundary search; continue sustained validation after terminal evidence.

CPU prefetch B2 passed12/12, median21.5217s/inclusive0.36685cells/s, wait2.2775%.
A2 now running; both prefetch repeats so far do not exceed A1 throughput.
Capacity integration wrapper prepared/uploaded (.sh at planned stem), not launched.

## 容量测试发布就绪（尚未运行）

源码 `/data/yilangliu/GraD-Pert/development/source-v2-capacity-a1d55fa`，
SHA `a1d55faff4a55d624e5061b7eb70e76d93f24c9b`，干净身份已核验。
发布 `development/gradpert-capacity-publication-a1d55fa.json` SHA256
`d00729ca8d66fe5dede305af80a94aa22fcba3298f0424154dd9f11cdf10e67c`。
单步扫描dry-run `development/single-a1d55fa-capacity-integration.plan.json`，
SHA256 `53e743e31d453259ed1136afbb4854b4b67ccb72e6566c667ea6be35f1ad678d`。
候选micro8/16/32/48/64，eager/重计算/同步数据路径，未加--execute。
待当前ABBA完成和取舍后再启动；如果启用预取则需要先正式配置化并重新发布，
不可让benchmark-only override冒充容量/正式设置。


CPU prefetch B1 passed12/12, median21.8183s, wait fraction2.5462%,
inclusive0.35858cells/s versus A1 .36947 (~2.95% slower). Waiting decreased
but update grew; CPU contention is a hypothesis, not established cause.
B2 now running in same ABBA wrapper3421214. Do not adopt based on wait fraction;
await B2/A2 for paired throughput conclusion. No reruns or other GPU job.

CPU prefetch ABBA A1 passed12/12, median20.6005s, inclusive0.36947cells/s;
B1 running at latest check. compare_benchmarks.py now accepts explicit
--execution-factor cpu_prefetch, requiring identical configs and A/B/B/A flags
without checkpoint override;9 audit tests and scoped mypy/ruff passed.

Prepared self-contained current-method capacity candidates under
configs/v2/single_pass_jurkat/capacity_m{8,16,32,48,64}_a2/gradpert_v2/.
All parsed and validate_profiles passed: only micro/global batch differ from
profiling_m2_a2; single-pass final-state eager, checkpointed, fullSSL1+SSL2,
world2 accumulation2. No GPU capacity launched while prefetch ABBA is active.
After optimization decision: publish selected execution configuration, run
ascending one-step integration to bound memory (expand if64 passes, refine if
failure); then128+ updates/checkpoint continuation/300-control inference near
boundary with fresh run IDs. Never call one-step boundary stable maximum.
Recheck prefetch benefit at selected larger batch; current ABBA only global8.

CPU prefetch parity PASSED two updates on both ranks: exact inputs/RNG and
max_abs0 gradients/loss/objective/optimizer. Exit0; reviewed41,682byte receipts
saved docs/experiments/single-ec2384a-prefetch-parity/. Complete local v2 suite
312passed in85.78s (two existing TorchScript deprecation warnings).
Immediately launched throughput ABBA PID3421214, stem
`/data/yilangliu/GraD-Pert/development/single-ec2384a-prefetch-abba`.
Source ec2384ade9fa66458eac2ff40c2dbff2fd067dc8/publication as below. A1/B1/B2/A2
all identical checkpointed eager profiling_m2_a2 configs; B alone --cpu-prefetch.
12updates each,3warmup9timed,1200s timeouteach,bothGPUlocks. Both parity receipts
are checked before benchmark. .stage/.pid/.exit; stem-label.log and
stem-label/receipt.json. Compare schedule/RNG/data/identity plus inclusive
throughput and memory after all complete. Existing compare_benchmarks.py CLI
is specific to validation-hoist; do not mislabel this as that experiment.
Default remains synchronous pending speed evidence; no capacity/formal yet.

CPU prefetch full-update parity now running PID3420187, stem
`/data/yilangliu/GraD-Pert/development/single-ec2384a-prefetch-parity`;
.stage/.run.log/.exit/.pid and rank receipts at stem/, bothGPUlocks,1200s.
Source `development/source-v2-prefetch-ec2384a`, fullSHA
ec2384ade9fa66458eac2ff40c2dbff2fd067dc8, clean identity verified; publication
`development/gradpert-prefetch-publication-ec2384a.json` SHA256
43f6146774258e336d6777e4fef1b439d9380e49af75a310bb9b3a090442d2fa.
Identical eager checkpointed profiling_m2_a2 config, only candidate CPU lookahead
changed; --deterministic --candidate-cpu-prefetch. Inspect terminal result before
throughput benchmark. Default remains synchronous; no formal/capacity launch.

Checkpoint-removal parity PASSED both ranks, exit0, two full updates with exact
inputs/RNG and max_abs0 losses/gradients/objective/optimizer. Reviewed41,616byte
receipts copied to docs/experiments/single-473466d-checkpoint-parity/. Not adopted:
A1 suggests ~12% faster update but ~4x memory; need sustained capacity tradeoff.
New opt-in CPU view/data lookahead implemented, default disabled. One worker,
one batch ahead; private NumPy generator, public state committed only at yield;
CPU-only assembly, consumer-thread device transfer. Early close/error preserves
consumed RNG, worker joined. Diagnostic flags isolate prefetch from other changes.
20 relevant tests, scoped mypy/ruff passed. Next publish and run full two-update
parity (--candidate-cpu-prefetch, identical eager checkpointed configs), then
benchmark if passed. No current GPU job; no timer/formal B0/capacity yet.

ABBA stopped terminal exit1: A1 passed12/12, B1 failed10/12 on both ranks,
Dynamo recompile_limit64 exceeded at chunk_delta_final_state; no B2/A2 launched.
Full-sequence static replay is rejected for current random-length views; do not
raise cache limit and repeat or adopt synthetic speedups as training evidence.
Small reviewed receipts saved docs/experiments/single-3a2980d-replay-abba/{A1,B1}/
(88,989+75,379bytes); failure traceback remains server *-B1.log. GPUs cleared.
Immediately launched independent checkpoint-removal parity PID3418239, stem
`/data/yilangliu/GraD-Pert/development/single-473466d-checkpoint-parity`;
source/publication473466d-standalone as below, bothGPUlocks,1200s timeout.
.stage/.run.log/.exit/.pid and rank receipts at stem/. Both configs identical
eager profiling_m2_a2, only candidate sequence checkpoints disabled; graph
checkpoint retained. Deterministic two-update check, no throughput claim.
Default eager/checkpointed unchanged. Inspect this result before new GPU job.

Next diagnostic source ready (not launched while ABBA owns GPUs):
`/data/yilangliu/GraD-Pert/development/source-v2-replay-473466d-standalone`,
SHA473466d0b373b5486ed024adab69e538d18b6e16, publication
`development/gradpert-replay-publication-473466d.json` SHA256
d6b28c717edb24081ec1dafb2cde5bbc3a71a11d5b871c133511a55765fdf796.
Clean identity verified. Uses --candidate-no-sequence-checkpoint with identical
profiling_m2_a2 configs, --deterministic; reference checkpointing retained.
Previous attempted shared clone source-v2-replay-473466d failed checkout because
Git alternates nesting exceeded depth; preserved unused, never launched. Repaired
by24MiB full source-only bundle and independent clone. No active source altered.

ABBA live at B1 after A1 passed12/12. A1 median17.1801s, inclusive0.43615cells/s,
peakallocated14,551,056,384bytes (13.55GiB), default checkpointed priorbaseline
19.5796s /3,505,913,344bytes. This is one reference observation, not ABBA outcome.
Prepared separate candidate-only checkpoint-disable parity flag, requiring
identical eager architecture/config on both sides and rejecting reference-repeat
or both-sides overrides. This checks against the real checkpointed default;
the previous replay parity disabled checkpointing on both sides.8 helper/CLI
tests plus scoped lint/typecheck passed. GPU validation of this new diagnostic
must wait for current ABBA to release GPUs; no duplicate job launched.

ABBA throughput diagnostic launched PID3413600, stem
`/data/yilangliu/GraD-Pert/development/single-3a2980d-replay-abba`.
Immutable source `development/source-v2-replay-3a2980d`, fullSHA
3a2980d2eee30faa1ad99fb4481262701879d262, publication
`development/gradpert-replay-publication-3a2980d.json` SHA256
a19bc42a1af60768863314c2ba640a8847f8bf1d905f9825d387d752682650f6.
A1/B1/B2/A2: eager/replay/replay/eager, all no-sequence-checkpoint override;
graph checkpoint remains. Global8=micro2*accum2*2GPUs. Each12 updates,
3warmup+9timed, timeout1200s each, bothGPUlocks. Logs/receipts: stem-label.log
and stem-label/receipt.json; wrapper .stage/.pid/.exit. Stops on failure.
14 targeted tests passed locally and scoped mypy/ruff passed. Inspect live
result, compare schedule/view hashes, timings and peak memory before adoption.
This diagnostic is not sustained capacity or formal training. Goal stays active.

Checkpoint-isolation diagnostic PASSED on both ranks, exit0, GPUs released.
Source a63ac39a3b9237cf57c5782bcc7f5939fa6e2edb; two full updates (second
nonzero LR), exact input/RNG identity, all compared losses/gradients/objective/
optimizer max_abs0. Each rank captured10 graphs, skipped0. Receipts:
docs/experiments/single-a63ac39-replay-no-sequence-checkpoint/ (41,598byte
reviewed transfer). Scope: BOTH sides disable Cell/Response checkpointing;
graph checkpoint remains. This isolates the failed checkpoint/replay combination,
not proof of a production speedup or capacity. Default remains eager.
Next: bounded throughput measurements with identical checkpoint override on
reference/candidate, then compare memory/throughput to checkpointed baseline.
Probe override restricted to benchmark-only and explicitly receipted; no formal
capacity/training can silently use it. Goal active, no timer or formal B0 yet.

Checkpoint-isolation diagnostic now running, PID3411374, stem
`development/single-a63ac39-replay-no-sequence-checkpoint`; same .stage/.run.log/
.pid/.exit and rank-receipt layout,1200s timeout, bothGPUlocks. Clean source
`development/source-v2-replay-a63ac39` at a63ac39a3b9237cf57c5782bcc7f5939fa6e2edb.
Publication development/gradpert-replay-publication-a63ac39.json SHA256
c2652c2080543178f443eb07ca98f66acf17dc2599dd7a505f84401956e8f8a0.
Both reference/candidate sequence checkpoints disabled by explicit diagnostic
flag; graph checkpointing retained. Formal configs unchanged. Six parity-helper
tests passed locally (corrects preceding count typo). Outcome pending.

835a9ad donation-disabled candidate also FAILED on both ranks with exactly
same cudagraph lifetime invariant in first cell backward; exit1, no candidate
update passed. Receipts saved under docs/experiments/single-835a9ad-replay-parity/
after35,700byte dry-run. Buffer donation alone is not the fix.
Next diagnostic isolates checkpoint interaction: --no-sequence-checkpoint on
update_parity disables Cell/Response checkpointing on BOTH reference and
candidate, graph checkpointing unchanged. Explicit diagnostic receipt flag;
formal configs/runners untouched. This is fault isolation, not adoption or a
claim of faster training.6 parity helper tests and scoped mypy/ruff pass.

Ownership-repair hypothesis published835a9ad663ca0546b2cf36da02d1df69a1dc9521,
clean immutable `development/source-v2-replay-835a9ad`, tree
b3103a1d8457898db791584716a8abcefbdcf1ab6acfa0f5f9184b932eb74ccd;
publication `development/gradpert-replay-publication-835a9ad.json` SHA256
64ed51fa9535eaf91f48d984b7dd728b408bd1ad12df6e26d5b404532e12a735.
New parity wrapper PID3409783, stem `development/single-835a9ad-replay-parity`;
same .stage/.run.log/.pid/.exit and rank receipt layout,1200s timeout/bothGPUlocks.
Only donation policy changes relative to failed replay candidate. Reference
still eager, checkpointing unchanged. Inspect terminal result before deciding
next optimization; do not repeat unchanged failing configuration. No model
optimization adopted and no capacity/formal job yet.

3c308dd sequence replay full-update candidate FAILED on both ranks during
first accumulated cell backward, before any candidate update comparison:
"graph recording observed an input tensor deallocate ... did not occur during
replay" in PyTorch cudagraph_trees.check_invariants. Exit1, processes gone,
GPUs clear. Both failure receipts dry-run35,700bytes then copied to
`docs/experiments/single-3c308dd-replay-parity/`. Do not adopt or call parity passed.
All303 local v2 tests passed; that does not cover CUDA storage lifetimes.
Inspected installed PyTorch runtime and functorch donated_buffer config.
Next bounded repair hypothesis: disable saved-buffer donation during regional
capture to stabilize checkpoint-recomputation ownership. Same method/tolerances,
checkpoint setting/views/losses unchanged.9 targeted tests plus scoped mypy pass;
new test asserts policy restoration and independent differentiable output copy.
GPU rerun still required; disabling donation is not yet established as a fix.

Sequence replay candidate published `3c308ddb5d163659cfbd87e325a92d1ea191bd64`;
clean immutable server `development/source-v2-replay-3c308dd` verified against
clean local publication. Publication `development/gradpert-replay-publication-3c308dd.json`,
SHA256 d7fbdbd15c8cd2e6dde21e09ba1b328c091e51d95ea67390561eff18de011e26.
Live bounded parity wrapper PID3407939, stem
`development/single-3c308dd-replay-parity`; `.sh/.pid/.stage/.log/.exit`, diagnostic
log `.run.log`, per-rank receipts inside stem directory. Both GPU locks, timeout
1200s; target3routing tests then deterministic2update reference/candidate check.
Reference profiling_m2_a2 and candidate replay_m2_a2, same single-pass method,
global8. Comparison includes nonzero-LR update2, all gradients/optimizer/EMA/
centers, inputs and RNG. No throughput queue or formal training launched until
candidate terminal evidence is inspected. Recompilation/cache/OOM failure is
candidate evidence, never grounds to silently relax precision or change views.

Opt-in sequence CUDA replay candidate prepared (relay_kernel=cudagraphs),
self/cross gene scans and CLS write only; graph neighborhoods remain eager.
Unchanged eager arithmetic inside compiled cudagraphs backend, no fusion.
Optimizer-boundary step markers only, cloned final outputs retain storage
ownership; finite64 shape-recompile budget, capture skips rejected. Variable
view shape capture costs/memory remain unvalidated and may reject the candidate.
Default remains eager; config single_pass_jurkat/replay_m2_a2 is engineering-only.
38 targeted tests passed, scoped mypy4files and ruff passed. CPU tests validate
dispatch/eager equivalence; actual replay proof requires target GPU parity.
Next publish clean source and run deterministic two-update dualGPU parity,
including nonzero-LR step2 and actual graph/skip counters. No throughput or
formal adoption before full-update and sustained memory evidence.

Replacement queue3404789 completed exit0: profile4/4 passed; synthetic
CUDA Graph12/12 cases passed, no skipped graphs or recaptures during timing.
FP32/BF16,grad/no-grad,live carried states and all five input gradients checked
at unchanged3e-5/3e-4 tolerance; max observed absolute error2.33e-10, RNG exact.
B2,T257 speedup4.83–5.87x; B64,T94 only1.008–1.046x; CLS-likeT1 2.43–3.08x.
These are synthetic paired-call timings, NOT model/update speedup or proof of
full-model parity. Small receipts/summaries dry-run (111900+21401bytes) then
copied to `docs/experiments/single-38af3ce-profile-replay/{profile,replay}/`.
Both GPUs verified no compute apps after completion. No monitor or GPU job now.
Next active engineering: opt-in model replay candidate, preserving eager default;
handle variable view shapes, live output/state lifetimes, checkpoint recomputation,
full nonzero-LR update/EMA/center/RNG parity and graph cache memory before any
capacity/formal run. Graph-neighborhood replay alone has weak speed evidence;
prioritize sequence KDA and ensure capture overhead does not erase gains.

Supersedes pending c073a37 capture/replay queue: explicitly stopped diagnostic
postprocessing (raw traces+summaries preserved, no terminal profile pass) and
its waiting successor. Both process trees gone, GPUs empty; stop evidence
`development/single-c073a37-postprocess-interrupted.json`. Integration and
12-step baseline remain passed. No training checkpoint/run was restarted.
Replacement immutable source `development/source-v2-profile-38af3ce`, SHA
`38af3ce7a9b00ba4f8876af6e9521c49cb555edb`, clean local/server/publication identity
verified. Model/config tree identical to c073a37, a7d78edc8a11dc3f39a469247382dac0e612139bd7ed1ee47f25d57bfec52ca8;
only diagnostic script behavior changed. Publication
`development/gradpert-profile-publication-38af3ce.json`, SHA256
`79ecb1e787512e32f78eae81f1b7a4ed5fa61096fa8ae0be0a9d930e1443f19a`.
New bounded queue PID3404789: `development/single-38af3ce-profile-replay`,
`.sh/.pid/.stage/.log/.exit`. Target12tests → dualGPU4step profile (900s timeout)
→ synthetic GPU0 replay (1800s timeout), both GPU locks held. Child outputs
append `-profile`/`-replay`; no detailed operator aggregation requested.
Inspect this queue, not stopped predecessors. Goal active, no automation.

Profiler postprocessing issue observed: after ~4.77GB/rank trace and small
summary had been saved, lazy key_averages aggregation ran >5minutes at one CPU
core/rank and ~49millionKiB RSS/rank. Host still had ample available RAM;
original processes kept intact under2400s wrapper timeout. Do not label hung
or terminal solely because .stage/receipt has not advanced.
Prepared diagnostic-only repair: detailed operator table is now opt-in via
`--profile-operator-table` (requires profile-last-update). Default still exports
raw trace, region summary and GPU interval union, without key_averages. No
model, loss or optimizer change. Tests5profile+7probe-policy pass; profile test
checks complete update/RNG equality, default forbids expensive aggregation,
explicit mode exports table. Scoped mypy2files and ruff pass. Applies only to
future immutable source; current c073a37 capture is not modified.

Both single-pass trace summaries have been generated and copied after bounded
dry-run (12,618bytes), `docs/experiments/single-c073a37-preflight/profile/`.
Rank0/1 kernel counts1,414,510/1,414,379 in ~31.51s profiled windows;
GPU interval-union fractions0.16365/0.20411 include profiler overhead, not normal
utilization. Graph forwards ~5.19/5.01s student plus3.18/3.01s teacher;
backward18.52/17.74s CPU-inclusive; gradient reduction14–16ms. Nested times
must not be summed. Evidence prioritizes launch overhead/KDA regional replay.
Raw traces (~4.77GB/rank) remain server-only. Profile workers3401379/3401380
are still live in CPU summary/operator aggregation, not terminal success;
queue retains locks and the synthetic replay successor remains gated.

Single-pass baseline passed12/12 (3warmup,9timed), source c073a37 unchanged.
Median update19.57964s; p9519.83892s; measured cells/s including data0.387596;
peak allocated3,505,913,344bytes. Global8 is diagnostic only. Data wait1.09–1.45s
per update is a modest fraction; prioritize KDA CPU launch/synchronization
investigation before assuming prefetch alone solves the bottleneck. Method
change from dual to single pass is not an equivalent-implementation speedup.
Receipt dry-run88,894bytes then copied to
`docs/experiments/single-c073a37-preflight/baseline/receipt.json`.
Queue has transitioned to final-update profiling; wrapper3399679 live. Replay
candidate wrapper3400405 still waits on GPU locks and verified predecessor gates.

Single-pass integration passed1/1 complete update on both GPUs with checkpoint
save/reload; receipt dry-run reviewed79,212bytes then copied to
`docs/experiments/single-c073a37-preflight/integration/receipt.json`.
Baseline12updates now running, profile follows. Finite successor synthetic
CUDA Graph probe queued behind both GPU locks: PID3400405, stem
`development/single-c073a37-cudagraph-check` with `.stage/.log/.pid/.exit`.
It requires all three preflight receipts passed and queue exit0 before GPU use.
This unchanged generic kernel tool checks two live final-state invocations and
carried-state gradients; it is not a model backend adoption or a two-pass model
training run. Final-state write implementation and model default stay eager.
Synthetic success alone will not establish end-to-end speed or update parity.

Current live bounded queue: `/data/yilangliu/GraD-Pert/development/single-c073a37-preflight`,
PID3399679; same stem `.sh/.pid/.log/.exit/.stage`. Stages: target30tests →
dual-GPU full-loss integration+checkpoint reload → 12update baseline (3warmup)
→ 4update bounded final-update profile. All stage success receipts gate the next.
Training SHA `c073a37c3c038561fb3d167b2a8f53f8650544bd`, immutable
`development/source-v2-single-c073a37`; publication
`development/gradpert-single-clean-publication-c073a37.json`, SHA256
`3287827f59329ea3be6185493c5e816e24d527027101cbad0277fbc1e7bf3ace`.
Fresh target tests30passed; integration processes verified alive. GPUs0/1 locked.
The first local publication receipt was rejected: ignored editable-install
`src/gradpert.egg-info` altered the local tree hash. Fresh clean local clone and
server now agree on tree a7d78edc8a11dc3f39a469247382dac0e612139bd7ed1ee47f25d57bfec52ca8.
Rejected receipt retained; no identity checks bypassed. Use clean publication
staging for subsequent releases, not the editable test worktree.

User explicitly resumed the original Goal after the single-pass correction.
The pause below is historical and superseded. Current method commit
`f6d84bebb3519a5cf94c91e38ea0b90854e1a952`: single random-order writing pass,
unchanged final-state readout. Resume only new run IDs; old dual-pass ABBA
and CUDA Graph queue remain stopped. SSH works; both5090s verified idle.
Next: publish the same method with a conservative micro2×accum2×world2=batch8
profiling config; verify fresh immutable server checkout and full-loss integration
with checkpoint reload. Then collect single-pass throughput/CPU-GPU timeline,
optimize evidenced bottlenecks with complete-update parity, and validate sustained
capacity before full B0 five epochs and best/last tests. No other ablations.
Goal active; no enabled automation. Old batch192 is not a certified new-method
capacity; base config batch values are placeholders until capacity acceptance.


## Latest user override — performance work paused; single-pass KDA

2026-09-26: Goal paused at user request. Stopped both verified process trees:
ABBA wrapper3386002 (including active B2 ranks) and waiting CUDA Graph
wrapper3389345, stopping the successor before freeing GPU locks. Post-stop
process inspection found no owned processes and nvidia-smi no compute apps.
Server evidence: `/data/yilangliu/GraD-Pert/development/user-pause-single-pass-20260926.json`.
A1/B1 completed receipts remain historical; ABBA is incomplete, B2 interrupted,
A2 and CUDA Graph probe have not been accepted as completed.

Only method change: `relay_passes: 1`, new self-contained current config
`configs/v2/single_pass_jurkat/gradpert_v2/nadig_jurkat.yaml`. All graph, cell,
response self/cross KDA write once, then queries read the final state. Self CLS
writes once at the end; graph target-only readout and cross control K/V remain.
Historical missing setting means two passes; old checkpoints/config identities
remain valid. Student parameter count unchanged at27,279,662; EMA same structure.
Local validation: 298 v2 tests passed (83.78s), plus the subsequently added
single/dual joint-loss parameterization passed both cases. Ruff and scoped mypy
passed. Receipt: `docs/experiments/single-pass-method-validation.json`.
New single-pass CUDA performance/capacity remains untested while paused.
Pre-change clean published baseline: f5267dfa3c34f3bf5993848d315ed4bbaf19c02b.
No new GPU test, capacity sweep, training or monitor is authorized to resume
by this correction. Earlier active/running statements below are historical.


Method published as `7e4c669e5043986209339a0c8a613b02645356d3`; Student27,279,662,
same frozen EMA Teacher. Full B0 retains prediction+SSL1+SSL2. Local acceptance
and disclosed historical failures: [receipt](../docs/experiments/relay-stage-a.json).

User restored server access and requested Goal supervision. Goal active; SSH
works again. Diagnostics published `d9c1fbfb4a7864be316426fd2a7b26797ad7b138` and verified
on clean server source. Supported server environment passed263 v2 tests;
dual-card complete update + resume passed at micro2/accum2/global8. Short final-
update CPU/GPU traces were exported, but summary parsing failed on CUDA-mirrored
annotations. Repair5186003 passed4 tests; both original traces successfully
reanalyzed with separate analysis SHA. About1.972million kernels/update suggests
small-op/chunk fusion as next diagnostic direction; profiled timings are not
normal throughput evidence. A1 passed40/40: median26.63s/update at global8. Candidate569ae1b passed
FP32 kernel checks but failed strict BF16 output tolerance; failure preserved,
default stays eager. Intermediate-rounding retry also failed; compilation is not adopted.
Independent per-layer graph validation candidate passes275 local v2 tests
including bitwise checkpoint/dropout gradient/RNG checks; full two-card
update parity exposed GPU repeatability differences. Published4898de3 reference-repeat
also fails strict gradient tolerance (max1.3163e-4), with exact inputs/RNG.
Deterministic reference/candidate diagnostic passed two complete updates on
both GPUs with bitwise-equal losses/gradients/model/optimizer and exact RNG.
Same-source ABBA throughput queue runs40updates each at global8. A1 passed,
median26.323s/update. B1 passed at25.983s/update; B2 automatically started.
Single-pair throughput differs by only1.33%; B2/A2 remain pending;
no throughput improvement or default adoption claimed. Next CUDA Graph replay
probe is published279696a, CPU helper checks pass, and queued behind ABBA with
explicit successful-receipt/resource gates; its GPU behavior remains unverified. No formal training or timer; Goal continues. STATE.md has receipts
and remaining gates. GPU0 idle100%
telemetry is anomalous but both cards passed CUDA arithmetic; analyze traces. [Plan](../docs/design/GRADPERT_V2_RELAY_METHOD_PLAN.md) and STATE.md carry
remaining diagnostics→optimization→capacity→five-epoch-B0 dependencies. No new
CUDA throughput, capacity, or scientific results claimed.

The material below is historical status for earlier variants and stopped runs.

2026-09-25 method update: new Jurkat v2 config
`configs/v2/hamiltonian_mla_jurkat/gradpert_v2/nadig_jurkat.yaml` selects
three fixed bidirectional Hamiltonian expander cycles, two graph layers with
updated-neighbor propagation, and 2 KDA + 1 full MLA block in each Cell and
Response Encoder, with no DSA in the new profile. Old configs and stopped B0
remain unchanged. Full formulas and review issues are in
`docs/design/GRADPERT_V2_HAMILTONIAN_MLA_METHOD.md`. The inherited global
batch192 is **not capacity-certified for this changed architecture**; do not
start formal training before new two-card sustained capacity evidence, clean
source publication and method review. No training was restarted by this edit.
Local v2 regression: 230 tests passed; Ruff lint/format and wheel/sdist build
passed. This does not establish two-GPU throughput or capacity.
The four-fold fixed-axis cross-cell **design contract** now points to the same
new mechanism; it still lacks axis-specific prior/graph artifacts and a
validated training adapter, so no cross-cell GPU run is implied.

The old five-epoch Nadig Jurkat v2 ablation baseline was stopped at the user's
request on 2026-09-24. The subsequent compact v2 B0 formal run was also
**stopped at the user's request on 2026-09-25 02:28 +08**, before the first
epoch committed. It used the complete prediction + SSL1 + SSL2 model;
`prediction_only` was never launched. The compact-model capacity retest remains
valid engineering evidence, not a scientific five-epoch result.
The earlier stopped four-layer Top500/GenePT-PCA256/SwiGLU model, global batch
128, and two-GPU row-mean protocol are fixed at training source
`536458333e437252178ffe493c5c50c9064e7615` and config SHA256
`8471f5ea68f4801406497985291a0088116ff5a8435b82f85e55c611548b06c6`.
The exact-source two-GPU integration check passed before launch.

- Historical stopped run ID:
  `nadig_jurkat-seed1-20260923T194747Z-1b7eda2abd6441f592d0834e1e275e88`.
- Server run root: `/data/yilangliu/GraD-Pert/runs-v2-glm53-current/`
  followed by that ID. The training log is
  `/data/yilangliu/GraD-Pert/development/v2-jurkat-baseline-5364583.log`.
- Stopped state: all training parent/rank processes exited and both GPUs
  released this run's memory. **1/5 epochs committed**. Epoch 1 completed
  1,081 updates and selected `epoch-0001.pt` as provisional best/last by
  validation prediction loss `0.0052692545`. Validation Pearson values:
  TxPert `0.142970`, TriShift `0.187574`, Systema `0.067022`. These are
  validation metrics, not test results. `COMPLETE.json` and best/last test
  receipts remain absent. The interrupted epoch 2 has no committed receipt;
  do not report this run as a completed five-epoch baseline.
- The `grad-pert-v2-batch128` heartbeat was deleted after the requested stop.

New design: both Cell and Response Encoders have two KDA layers followed by one
DSA/MLA layer; all four Student/Teacher distillation heads use 8192 prototypes.
The Student count is 23,123,847. An isolated server CPU snapshot passed 227 v2
tests; lint, format, and package build passed. The scoped change is published
as `974c5eadf35a95fbc3ba46cffe6d9ab34cdd4e94`; its clean immutable server
checkout and publication receipt passed the formal source-identity check.
The default global-batch-128 two-GPU one-step integration and checkpoint reload
passed. Short probes passed global batches 192, 208, and 224; global 240 OOMed
at step 3. The full 128-step probes at 224 and 208 OOMed after five and 23
completed steps, respectively; neither is a sustained-capacity result. The
128-step dual-GPU probe at global 192 **passed** with checkpoint continuation
and 300×5000 validation inference. Its clean receipt is
`/data/yilangliu/GraD-Pert/development/v2-compact-974c5ea-capacity-m48-128/receipt.json`,
SHA256 `71d952f90466052a52855a819090d9c5e03ba3e96081eadc03ac2cd3c896529d`.
Measured throughput was 8.486 cells/s after warmup, 7.992 cells/s end to end;
peak allocated memory was 31.01/31.05 GB on GPUs 0/1. Thus 192 is the highest
**sustained-validated** batch under this protocol, while the exact physical
maximum between 192 and 208 is unmeasured. The user selected the passed
**global batch 192**, micro48/rank × accumulation2 × two ranks, for this new B0
formal run. The older stopped run remains historical evidence only.

Stopped compact B0 run ID:
`nadig_jurkat-seed1-20260924T134139Z-e5df6111138747e08ba5a582e37c1ed2`.
Run root:
`/data/yilangliu/GraD-Pert/runs-v2-b0-compact192-fc90a0d/` plus the ID.
Training source is clean published commit
`fc90a0d992373e619976504c82bd6f94c930d7b2`; config SHA256
`674ea9ab4160e10e85de7f6da78137257efee510c97e2f9852a474a55e81acf5`.
Exact-source two-GPU one-step integration passed before launch. At the user
stop, the epoch journal and history were **0/5** (749 planned updates per
epoch), with only the epoch-0000 initial checkpoint and no finite validation
selection. The targeted parent/torchrun processes received SIGTERM; parent,
wrapper, torchrun, and both ranks exited, and GPUs 0/1 returned to 2 MiB each.
No `COMPLETE.json` or best/last test receipts exist. The two-hour heartbeat
`grad-pert-v2-b0-compact192-formal` was deleted. Do not resume or overwrite
this run ID; no other ablation row was launched. The detailed ledger is
[STATE.md](STATE.md); capacity and throughput evidence is in
[GRADPERT_V2_GLM53_DUAL_GPU_PERFORMANCE.md](../docs/experiments/GRADPERT_V2_GLM53_DUAL_GPU_PERFORMANCE.md).

Historical R50 batch-1024 status from 2026-09-14 remains in Git history and
its dedicated experiment documents. It does not describe this v2 run.
## 2026-09-27：KDA 常量复用 ABBA 完成；转测图分块发射机制

干净发布源码 `32b2bfd88f36b9a7fad435c533dc825ef81942b4` 的常量复用 A1/B1/B2/A2 均 `passed 12/12`，队列 `exit=0`。四组有序 batch 及视图随机数起点哈希相同；两次配对的 B/A 含数据等待吞吐比为 1.01051、1.01361，几何均值 1.01206。启动成本分别记录，峰值显存未降低；这是小幅描述性收益，保留 opt-in，但不把它当作主要性能突破或默认 B0 执行方案。比较收据 `/data/yilangliu/GraD-Pert/development/v2-kda-constants-32b2bfd-abba-20260927T0826Z/comparison.json`，SHA256 `aa594a8ab33ce2032ac81960ccfae92db2d31f9b77f7d33464604680f545aeef`。短时子代理已停止监督，双卡空闲。主会话现验证图 KDA 固定 64 目标分块的区域 CUDA Graph 候选，先局部输出/梯度，再双卡完整更新与整步吞吐；不改变默认、容量或 B0，直到验证通过。

区域图分块候选随后在隔离测试副本中被淘汰：Inductor CUDA Graph 的 BF16 输入梯度在长度15/32均超原阈值（`v2-graph-replay-scratch-first-20260927.log`）；`emulate_precision_casts` 复测仍失败（`v2-graph-replay-scratch-precision-20260927.log`）。原算术 `make_graphed_callables` 单调用精确，但两次调用保持前次输出供反向时产生非有限梯度；前向捕获、原算术重算反向在 BF16 两块链上也超原阈值。未推送、未启动整步/容量/B0。图拓扑 253371 边对 27202 个不同 `(邻居基因, 四位边来源)` 组合，理论重复 9.31 倍；去重投影在128目标 BF16 局部前向四项完全一致，但多个参数/输入梯度超原阈值（某投影参数梯度最大差达2.0），故同样未接入。下一优先级转向实测双卡 CPU/GPU NUMA 调度和现有原算术路径，整步吞吐门槛不变。

## 2026-09-27：KDA 常量复用双卡完整更新严格一致；ABBA 计时进行中（历史）

干净发布源码 `32b2bfd88f36b9a7fad435c533dc825ef81942b4` 的 opt-in 形状常量复用，在双 RTX 5090 两步确定性完整更新上通过：两 rank 输入与随机状态一致，损失、参数梯度、目标、优化器、Teacher 和 center 状态差值均为零；第二步学习率 `7.796055196070788e-08` 非零。收据 `/data/yilangliu/GraD-Pert/development/v2-kda-constants-32b2bfd-parity-m2-20260927/rank-{0,1}-receipt.json`。这仍不是吞吐或长期模型效果证据。当前同一发布源码的 m64、全局256 串行 A1/B1/B2/A2 benchmark-only 队列在 `/data/yilangliu/GraD-Pert/development/v2-kda-constants-32b2bfd-abba-20260927T0826Z`，PID `3668186`；A 为原路径，B 只开启常量复用，各12步、预热3步。短时由本会话子代理只读监督；主会话完成收据/身份/吞吐审计后决定采用或淘汰，再推进容量与新 ID B0。
