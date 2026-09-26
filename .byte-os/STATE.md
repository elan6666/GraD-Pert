# 当前状态 — 2026-09-26

## 授权与目标

Goal active：当前新方法升级与性能工程 → 双卡持续容量 → 完整B0五轮及best/last。
不启动其他消融；不恢复旧停止运行。默认单向KDA、最终S统一读取，完整双蒸馏。
无定时监控；本Goal持续推进。保留历史来源，禁止修改活动服务器源码或覆盖runID。

## 当前阶段：机制级KDA优化；GPU短检查已收尾

用户补充：开关/预取/batch探针不等于深度优化完成。先实施和验证主要机制，
再持续容量与完整B0；不套用DSec或侧边固定方案。路线详见性能报告末节。
优先审计衰减加权Gram的[B,H,L,L,D]临时量，再融合归约及解析反向；尚未实现。
现有trace只有嵌套时间，先流式提取CPU发射/主要kernel成本，不能猜端到端收益。

CPU全量统计已完成：13,518,242事件，1414510 kernel，43,200 logsumexp。
小结果docs/experiments/single-38af3ce-profile-replay/costs/rank0.json；trace留服务器。
尚不能把float multiply或嵌套CPU耗时全部归给KDA，先与mHC Sinkhorn比较。
CPU关联归因正在运行：timeout父PID3432732，stem development/single-301be18-trace-attribution-rank0
(.pid/.log，完成.json，无.exit)，900s上限；不占GPU。
脚本scripts/v2/trace_attribution.py已发布301be18，三遍流式扫描CPU scopes→launch correlation→kernel。
6定向测试通过，覆盖嵌套区间、线程隔离、GPU annotation排除、乱序与多进程拒绝。
下一步检查PID/日志/完整JSON，按实测mHC、图层、三角求解等归因调整机制优先级。

无活动GPU任务。上探wrapper3429104已exit1：micro72 passed1/1及checkpoint恢复，
peakallocated31,009,318,400bytes；micro80 OOM，88未启动。单步不证明持续容量。
stem `/data/yilangliu/GraD-Pert/development/single-3ba9a96-capacity-integration`。
源码 `/data/yilangliu/GraD-Pert/development/source-v2-capacity-3ba9a96`，
SHA `3ba9a960d79d207ce89dd938b672be051f327234`，发布身份已核验。
发布 `development/gradpert-capacity-publication-3ba9a96.json` SHA256
`54e8f9df74e539e51001c74b1a0b51bc7b162b3cb89fe2fb5a746a35bffc0c2c`。
收据review158,510bytes后保存docs/experiments/single-3ba9a96-capacity-integration/。
前次micro8/16/32/48/64均passed，review403,935bytes后保存
`docs/experiments/single-a1d55fa-capacity-integration/`。

下一动作：有界流式解析既有大trace（无需重新占GPU）：服务器development/
single-38af3ce-profile-replay-profile/rank-{0,1}-trace.json，各约4.77GB。
避免key_averages超大内存问题。定量结果指导融合Gram前后向候选；保持随机数、
合法邻域、final-state读取、梯度及optimizer/EMA/center协议，不放宽误差掩盖失败。
优化完成前不启动72的长测/正式B0；最终仍需持续容量和完整B0五轮best/last。

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
