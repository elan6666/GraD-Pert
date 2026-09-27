## 当前所有权／动作（2026-09-27）

主会话负责当前性能构建；无活动双卡任务或长时监督。短图融合 `27aec3643cc613b17f3b87eecebf05d1453743ba` 的双卡确定性完整更新失败，输入/RNG对齐而梯度/损失/center/optimizer不对齐，证据保留于 `/data/yilangliu/GraD-Pert/development/v2-short-27aec36-parity-m2-datafix-20260927/`；不进入ABBA/容量/B0。当前下一步：原算术路径的 KDA 形状常量复用通过相关60项严格测试和局部计时，封存新 opt-in 候选→双卡完整更新→同配置ABBA，如无整体收益则淘汰；随后按用户目标继续机制评估、持续最大batch及新ID完整B0五轮。旧已停止B0不恢复，旧跨会话监督保持暂停。当前正在运行的任务以实时进程/收据为准；下方旧状态是历史快照。

## 当前执行：主会话性能工程（2026-09-27，新授权）

用户重新授权性能调研、双卡实测、等价机制优化及其后容量与全新 B0 正式运行；旧 B0 仍中止且不可覆盖。主会话拥有 build 阶段，当前无跨会话监督/定时监控；短时测试用 Luna 子代理只读监督。当前源码 `681d4fb609644d51c22f6b4e51dd61256adc3310` 的 m64 128步容量已通过，m72 OOM 保留。m32 完整更新 profile 已 exit0、6/6 passed，收据及 trace summary 位于 `/data/yilangliu/GraD-Pert/development/v2-mech-681d4fb-m32-profile-20260927T0727Z`；普通更新中位17.550秒，GPU/kernel与图发射详情已记录在 `docs/experiments/GRADPERT_V2_SINGLE_PASS_PERFORMANCE.md`。第一候选是独立短图邻域的融合最终状态递推：FP32局部加速约4.5倍，首版BF16 value梯度在极少量舍入边界失败；保留失败证据并采用原块求解计算该梯度的混合候选，相关服务器测试59项通过，BF16局部前后向2.803→1.782ms。尚未证明真实双卡完整更新等价或整体加速。临时checkout `/data/yilangliu/GraD-Pert/development/source-v2-short-provisional-20260927T0750Z` 仅用于不洁诊断测试，不能正式运行。下一动作：发布仅opt-in候选→双卡完整非零LR多步等价→同配置ABBA端到端，未通过则淘汰/修复。之后持续容量与新ID完整B0。细则见 `.byte-os/plans/GRADPERT_V2_MECHANISM_PERFORMANCE_20260927.plan.md`；若启动超过一小时的正式运行，再建立可返回主会话的耐久监控，不使用旧暂停心跳。

## 当前交接：用户暂停实验及监督（2026-09-27）

主会话 `01a0c01a-0611-7a90-b3e8-8ad7e017748b` 已接回所有权。B0 运行 `nadig_jurkat-seed1-20260927T041339Z-e35f5edb8e6248479462589ff3cef41f` 按用户要求 SIGTERM 中止，全部训练进程退出，双卡空闲；已提交 epoch 0/5，仅保留初始 `epoch-0000.pt`，无 COMPLETE、best/last 或测试收据。中止证据见 `.byte-os/coordination/receipts/b0-681d4fb-user-stop.json`（SHA256 `9e792365f37ec635483eb1da5a61dc993e65fa2bb30298e1bc68f09153af5451`）。监督会话停止工作，`grad-pert-v2`、`grad-pert-v2-b0-return` 两条自动化均暂停。下一步等待用户明确重新指示；不得自动恢复或启动实验、重启监督。下方运行中状态均为历史快照。

## 当前交接：完整 B0 五轮双卡训练已启动，交监督会话

m64 持续容量接受条件全部通过（exit0、128/128、checkpoint continuation、300-control、源码/配置身份），容量 receipt SHA256 `31a0798737a60bd318a861fd5e4d947a51f4d943ac0f613d0328ca4a3a7bfc04`。完整 B0 run ID `nadig_jurkat-seed1-20260927T041339Z-e35f5edb8e6248479462589ff3cef41f` 已启动，父 PID 3613045、双 rank 存活、`fit/epoch_state.json` 初始 epoch0/5；五轮及 best/last 未完成。源码 `681d4fb609644d51c22f6b4e51dd61256adc3310`，配置 SHA256 `23a2942804b73cdf92871e36808846f50382ded41b1588a68da76ebc0e69985b`。长时监督边界、路径与返程心跳见 `.byte-os/coordination/handoffs/2026-09-27-b0-681d4fb-supervise.md`。监督会话接手后才开启本运行的 20 分钟心跳；终态即暂停/删除。此节覆盖下方 m64 进行中记录。

## 当前交接：m64 容量测试主会话接手监督

活动 run `v2-teacher-681d4fb-m64-capacity-oomfallback-20260927-1203`，PID 3602712；2026-09-27T04:17:04Z 实查 32/128、双 rank 存活、无 exit/failure。主会话与子代理按新流程负责约1小时内测试的有界监督；跨对话监督心跳已暂停。m72 OOM 与 m64 fallback 证据见 `.byte-os/coordination/handoffs/2026-09-27-teacher-m64-capacity-return-main.md`。容量 acceptance 仍需 128/128、checkpoint reload 与300-control；未启动 B0。通过后主会话立即核验并启动此前已授权的完整 B0 五轮；失败则保存证据并决定修复/回退，不能把一次报告当作阶段终点。

监督规则：预计约一小时以内的有界测试，由本会话子代理只读监督并向主会话回报；更长的训练/消融才交指定跨会话监督。跨会话定时心跳仅在其负责的活动后台任务期间启用；终态或交回主会话时暂停/删除，需要新长任务时再开启。主会话收到任何监督终态后，在同一轮推进已授权的下一依赖。规则已同步到两份项目 `AGENTS.md`。

## 2026-09-27：m72 双卡128步容量测试已启动

Teacher Sinkhorn 优化双卡ABBA comparable，描述性总耗时下降3.84%，保留；m72×2×2=全局288单步预检1/1 passed，峰值预留显存31,279,022,080 bytes。已在干净发布源码681d4fb609644d51c22f6b4e51dd61256adc3310上启动m72的128步持续容量工程测试，PID3599854、运行根`/data/yilangliu/GraD-Pert/development/v2-teacher-681d4fb-m72-capacity-20260927`；当前只确认进程和0/128初始收据。监督交接见`.byte-os/coordination/handoffs/2026-09-27-teacher-capacity.md`。容量/B0仍未完成；此节覆盖下方历史空闲记录。

## 2026-09-27：Teacher Sinkhorn ABBA 已启动，交监督会话

源码 681d4fb609644d51c22f6b4e51dd61256adc3310 的本地 v2 351 测试、服务器 CUDA 16 测试通过；双卡确定性两步完整更新精确通过。非确定性校验与同路径重复对照均未过严格梯度阈值，保留收据，不混作候选失效证据。A1/B1/B2/A2 全模型 12 步串行队列已启动，PID 3531359，A1 收据真实存在；运行根 `/data/yilangliu/GraD-Pert/development/v2-teacher-681d4fb-abba-20260927`。按 `.byte-os/coordination/handoffs/2026-09-27-teacher-abba.md` 交由指定监督会话观察，完成或失败立即交回主会话。容量持续测试与 B0 五轮未启动。此节覆盖下面历史“无活动任务”记录。

## 最新状态：主会话负责工程，监督会话负责后台等待（2026-09-27）

主会话 `01a0c01a-0611-7a90-b3e8-8ad7e017748b` 负责模型设计、论文核对、性能实验设计、实现、验证、发布与启动；监督会话 `01a0df0b-4142-7df1-86c0-d959471d80a1` 只在有明确交接的后台阶段接管检查。Teacher 无梯度 Sinkhorn 存储消除候选的双卡 ABBA 已于 2026-09-27 完成：四组 12/12、`queue.exit=0`、比较为 comparable，描述性总耗时下降 3.84%；这不是正式训练或科学结果。当前没有活动 GPU 任务，所有权已交回主会话，由主会话审阅并决定下一有界阶段。持续容量与完整 B0 五轮及 best/last 仍未完成；不自动启动其他消融。

用户指定不用 Goal 模式；工具当前也报告无活动 Goal。已建立监督会话的 20 分钟心跳 `grad-pert-v2`，启动后立即触发一次。没有交接中的后台运行时仅核对状态并保持安静。主会话每次交接必须写清阶段、运行 ID/PID/日志、不可变源码与配置哈希、GPU/时间边界、预期完成收据、失败处理和下一步；监督会话以服务器实时证据核对。发现故障或里程碑后立即保存证据并消息唤起主会话，不等下一轮；主会话接回后监督会话停止对该任务重复操作。20 分钟是周期检查上界，单靠定时器无法在两次检查间瞬时发现故障；新后台任务如需故障发生即唤起，应在启动时加独立退出/失败事件通知，并先验证其可靠性。用户暂停/停止指令优先；全部任务完成时删除心跳。

上面的双会话安排覆盖下文旧的“Goal active／无定时监控”和阶段 F 的旧检查间隔，不改动历史实验事实。

## 显式mHC配置完整更新通过；下一步无梯度存储消除候选


### 显式后端配置完整更新通过

9d9bad86e16286b590b2bb31c5df5103ecfbbccb干净发布服务器source-v2-opt-9d9bad8，
publication SHA256 da83f52334078f742997eb4d2e3c850abba1b5234c13e09bfaceaa7688977294。
同profiling_m2_a2协议，仅candidate配置从native改auto；无诊断融合开关。
两rank两步全部比较最大差0，输入/RNG相同，第二步LR7.796055196070788e-8。
exit0，收据single-9d9bad8-config-parity/共42414bytes经dry-run归档。
这验证正式配置接入等价；不是新的吞吐或持续容量结果。当前GPU无活动任务。
下一文献指导候选是Teacher无梯度Sinkhorn的无用中间存储消除，详见文献矩阵。

FLA冻结完整前后向已读，直接替换会保留逐token输出，不能宣称省算。
文献矩阵新增具体候选：Teacher无梯度时每token2560bytes反向概率无需保存，
保持40归一化与Student梯度路径；先独立数值/完整更新再ABBA，未实现。
原目标仍为最终执行组合→持续双卡最大batch→完整B0五轮best/last。

# 当前状态 — 2026-09-26

## 授权与目标

历史状态：当时的目标为新方法升级与性能工程 → 双卡持续容量 → 完整B0五轮及best/last；2026-09-27 起改用上方双会话工作流。
不启动其他消融；不恢复旧停止运行。默认单向KDA、最终S统一读取，完整双蒸馏。
此处“无定时监控”为当时记录；现以顶部状态为准。保留历史来源，禁止修改活动服务器源码或覆盖runID。

## 当前阶段：融合mHC Sinkhorn完整双卡更新校验

trace两轮分析均完成，结果在docs/experiments/single-38af3ce-profile-replay/costs/。
1414510 kernels；logsumexp去重CPU区间1.5996s、129600 kernel累计0.166s。
不把嵌套时间相加；主要机制还包括图邻域/重计算/KDA临时量。先验证mHC融合，
保持20轮与解析梯度，不减少迭代、不改模型数学。KDA Gram候选尚未实施。

独立Triton融合算子18/18双设备合成case通过，原始收据
`docs/experiments/single-59305ee-sinkhorn-probe/receipt.json`。
最大输出/梯度差4.92e-7/3.88e-7，原4.33–5.04ms→0.514–0.656ms；
4096token临时峰值14,221,312→11,010,048bytes。非完整训练吞吐，不作采用依据。
首版4f855f9编译scope错误已修，失败收据保留；未改活动源码。
本地61测试通过，服务器18定向测试通过。

前次完整更新67fd8e1失败，第1candidate更新gradient最大差1.6681e-4，
ssl2_koleo亦超容差，输入/RNG完全一致。两rank收据已入库，不采用，不跑其ABBA。
经安装Torch Reduce.cuh证明strided四元素和逐项结合，修正Triton行轴归约；
7ae9a8b微基准18/18组、全部40阶段、输出及梯度逐元素相等，BF16差异0。
证据docs/experiments/single-7ae9a8b-sinkhorn-probe/receipt.json。

动态N cab78a1已通过双卡30组微基准：全部40阶段/输出/梯度exact，
每卡forward/backward各1编译variant。完整两rank两更新也全部max差0（含loss标量），
第二步明确非零LR7.796055196070788e-8。收据已dry-run复核281810bytes入库
`docs/experiments/single-cab78a1-sinkhorn-parity/`，未将第一步LR0冒充非零更新。

本地完整v2回归340passed（113b96a阶段）；接口类型修复后20定向测试及4文件mypy通过。
ABBA已终止exit0，四组12/12、完整审计comparable。
A1/B1/B2/A2含等待均值21.3293/20.4358/20.2642/21.3046秒；吞吐+4.7519%。
收据docs/experiments/single-b64745b-sinkhorn-abba-m32/；源码
b64745b7cc2a752ab992a32c62a234193a676cad，config SHA256
c03f83643f208af475765d2491401fd2de0b6b0091d96c7243712b21152b5a8f。
mHC通过完整精确更新和代表性batch性能门槛，保留；正式配置接入待完成。
独立Gram probe源码c7e89e8525a149026bdb17354e289e42fda0a28a，exit0、18组通过容差，
梯度exact、前向最大差4.47e-8；小形状偏慢，不采用默认。
收据docs/experiments/single-c7e89e8-gram-probe/receipt.json。
当前无活跃GPU任务，两卡各2MiB/0%；下一步核对Gram归约顺序并据新假设优化，
再完整更新/真实吞吐验证。不得放宽容差，不重复无新假设的失败候选。
保持单向KDA统一final-state读取；持续容量和完整B0尚待完成。

先前容量短检查全部终止：a1d55fa micro8/16/32/48/64 passed1/1，
3ba9a96 micro72 passed1/1+restore、micro80 OOM、88未启动。
各原始收据在对应docs/experiments/single-*-capacity-integration目录。
单步不证明持续容量；机制优化验证前不启动容量长测/正式B0。

## 已完成的性能取舍

CPU预取ABBA已终止exit0，四组12/12通过且配置/数据/RNG/源码审计通过。
A1/B1/B2/A2含等待均值21.6529/22.3103/21.8072/21.4106秒。
预取平均慢2.4476%，故不采用；同步路径默认保持。两次双卡更新数值完全
一致但不构成速度收益。详见 `docs/experiments/single-ec2384a-prefetch-abba/comparison.json`
及同目录原始收据（已dry-run356,094bytes后同步）。

性能总结 `docs/experiments/GRADPERT_V2_SINGLE_PASS_PERFORMANCE.md`。
全序列CUDA Graph候选10/12触发重编译64上限，淘汰当前路径，不增加上限重跑。
无序列重计算两次更新精确一致但显存约3.27→13.55GiB，暂不采用。
默认单向final-state eager+重计算；旧batch192不能证明当前方法容量。
完整本地v2测试312passed；新增benchmark因子审计9passed。

## 运维

本地工作树 `/Users/elan/code/grad-pert-v2-build`，定向提交推送HEAD:main。
服务器Python `/data/yilangliu/GraD-Pert/source/.venv/bin/python`。
SSH `ssh -S none -o BatchMode=yes -o ConnectTimeout=10 10.24.1.91`，当前可用。
服务器无rg，可用Python筛选输出。数据/权重/大trace只留服务器；小收据先dry-run。
共享Git克隆不可无限叠加；独立基底source-v2-replay-473466d-standalone可用。
干净发布应从无ignored egg-info的独立本地克隆生成，保持源码身份核验。
旧运行细节及已停止PID仅在STATUS/history/Git保留，不当作活动任务。
