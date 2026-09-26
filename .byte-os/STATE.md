# 当前状态 — 2026-09-26

## 授权与目标

Goal active：当前新方法升级与性能工程 → 双卡持续容量 → 完整B0五轮及best/last。
不启动其他消融；不恢复旧停止运行。默认单向KDA、最终S统一读取，完整双蒸馏。
无定时监控；本Goal持续推进。保留历史来源，禁止修改活动服务器源码或覆盖runID。

## 正在运行：当前方法双卡单步容量扫描

wrapper PID3427171，启动前双卡均2MiB空闲。
stem `/data/yilangliu/GraD-Pert/development/single-a1d55fa-capacity-integration`。
`.stage/.pid/.exit/.run.log`，运行根同stem，`sweep.json`及`micro8/receipt.json`等。
源码 `/data/yilangliu/GraD-Pert/development/source-v2-capacity-a1d55fa`，
SHA `a1d55faff4a55d624e5061b7eb70e76d93f24c9b`，干净发布已核验。
发布收据 `development/gradpert-capacity-publication-a1d55fa.json` SHA256
`d00729ca8d66fe5dede305af80a94aa22fcba3298f0424154dd9f11cdf10e67c`。
计划 `.plan.json` SHA256 `53e743e31d453259ed1136afbb4854b4b67ccb72e6566c667ea6be35f1ad678d`。

依次micro8/16/32/48/64，world2、accum2、eager/重计算/同步数据，完整SSL1/2。
配置 `configs/v2/single_pass_jurkat/capacity_m{8,16,32,48,64}_a2/gradpert_v2/`。
双GPU锁，wrapper timeout1800s，失败停止，不得覆盖旧ID或修改活动源码。
单步含checkpoint保存/恢复，仅定位边界，不能证明持续容量或正式效果。
实时核对进程及每点收据；进程消失不代表成功。64通过则扩展、失败细化；
随后近边界做128+更新、checkpoint续跑、300-control验证，并测稳态吞吐。
后续仅完整B0五轮与best/last测试；不启动其他消融、不恢复旧停止运行。

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
