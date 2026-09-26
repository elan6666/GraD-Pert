# 当前状态 — 2026-09-26

## 授权与目标

Goal active：当前新方法升级与性能工程 → 双卡持续容量 → 完整B0五轮及best/last。
不启动其他消融；不恢复旧停止运行。默认单向KDA、最终S统一读取，完整双蒸馏。
无定时监控；本Goal持续推进。保留历史来源，禁止修改活动服务器源码或覆盖runID。

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

当前活跃父PID3442646，stem `/data/yilangliu/GraD-Pert/development/single-b64745b-sinkhorn-abba-m32`
(.pid/.log/.stage/.tests.log/.exit；每组stem-A1/B1/B2/A2目录receipt.json及同名.log)。
完整双蒸馏ABBA：global128=m32×accum2×2卡，每组12步，3warmup+9timed，
仅B启用--fused-sinkhorn；不改checkpoint/CPU预取/模型参数/损失权重。
服务器源码`/data/yilangliu/GraD-Pert/development/source-v2-sinkhorn-b64745b`，
SHAb64745b7cc2a752ab992a32c62a234193a676cad，干净发布核验。
publication `development/gradpert-sinkhorn-publication-b64745b.json` SHA256
`cce6250a6583e1903509f95e513c057f4f8fdfa678d907ab0108ac70c6c54a91`。
config single_pass_jurkat/capacity_m32_a2/gradpert_v2/nadig_jurkat.yaml SHA256
`c03f83643f208af475765d2491401fd2de0b6b0091d96c7243712b21152b5a8f`。
parity来源cab78a1，与b64745b的src/configs diff为空；后者只补启动/准备/热身计时。
启动门槛核验前次exit0、每rank2步通过、第二步非零LR、全比较max0。
下一步检查该PID/日志/组收据，4组完成用compare_benchmarks --execution-factor fused_sinkhorn
及同一config分析，dry-run后仅拉小JSON。不得将局部7–8倍直接说成模型收益。
仍未正式采用融合或启动B0；持续batch/恢复/300control/完整B0五轮best/last待完成。
KDA Gram临时量/图邻域工作组织仍待按剩余瓶颈决定，不重复失败的静态sequence graph/预取。

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
