# 当前状态 — 2026-09-26

## 授权与目标

Goal active：当前新方法升级与性能工程 → 双卡持续容量 → 完整B0五轮及best/last。
不启动其他消融；不恢复旧停止运行。默认单向KDA、最终S统一读取，完整双蒸馏。
无定时监控；本Goal持续推进。保留历史来源，禁止修改活动服务器源码或覆盖runID。

## 正在运行：CPU预取 ABBA

- wrapper PID3421214；实时核对该进程及子进程，进程消失不等于成功。
- stem `/data/yilangliu/GraD-Pert/development/single-ec2384a-prefetch-abba`
- `.stage/.pid/.exit/.log`；每组 `stem-A1.log` 等和 `stem-A1/receipt.json` 等。
- A1/B1/B2/A2 = 同步/预取/预取/同步。所有组12更新、3预热9计时。
- 同一eager/checkpointed配置 `configs/v2/single_pass_jurkat/profiling_m2_a2/gradpert_v2/nadig_jurkat.yaml`。
  全局8=每卡2×累积2×双卡；B仅加`--cpu-prefetch`，不关闭重计算。
- 源码 `/data/yilangliu/GraD-Pert/development/source-v2-prefetch-ec2384a`
  SHA `ec2384ade9fa66458eac2ff40c2dbff2fd067dc8`，干净发布已核实。
- 发布收据 `development/gradpert-prefetch-publication-ec2384a.json`
  SHA256 `43f6146774258e336d6777e4fef1b439d9380e49af75a310bb9b3a090442d2fa`。
- 每组timeout1200s、双GPU锁、allocator expandable_segments:True，失败停止队列。
- 最新A1 passed12/12，中位20.6005s、含等待0.36947cells/s、等待占5.8767%；
  B1 passed12/12，中位21.8183s、含等待0.35858cells/s、等待占2.5462%；
  B2 passed12/12，中位21.5217s、含等待0.36685cells/s、等待占2.2775%；A2运行中。
  务必实时查验，不据此推断后续组状态。

## 已完成证据与下一动作

CPU预取两次双卡完整更新精确一致、输入/RNG一致；本地v2测试312passed。
收据 `docs/experiments/single-ec2384a-prefetch-parity/`。
预取默认尚未启用，正式runner未改变；GPU结果通过后才测吞吐。
目前即上述吞吐阶段，完成后核对同配置/输入/RNG和含等待时间，再决定采用。
`compare_benchmarks.py --execution-factor cpu_prefetch`已支持本次审计；要求同配置、
A/B/B/A预取flag、无重计算override，并核对完整收据/输入/RNG/计时。9项测试通过。
待四组完成后使用该模式生成分析收据（记录分析脚本hash）；不要分析未完成组。


性能总结：`docs/experiments/GRADPERT_V2_SINGLE_PASS_PERFORMANCE.md`。
- 全序列CUDA Graph候选10/12失败，重编译64上限；不采用、不提高上限盲试。
- 无序列重计算两步差值0，但显存从约3.27到13.55GiB，暂不采用。
- 默认保持eager和激活重计算；旧batch192不代表当前方法容量。

容量候选已提交：`configs/v2/single_pass_jurkat/capacity_m{8,16,32,48,64}_a2/`。
仅micro/globalbatch不同，双卡、accum2、完整SSL1/SSL2均校验。尚未执行。
优化取舍后发布最终选定配置：先单步定位边界（64通过则扩展、失败细化），
再128+更新、checkpoint续跑、300-control验证，较大batch复测吞吐。
最后按新不可变源码/新runID启动完整B0五轮，核对best/last真正测试收据。

## 容量测试发布就绪（尚未运行）

源码 `/data/yilangliu/GraD-Pert/development/source-v2-capacity-a1d55fa`，
SHA `a1d55faff4a55d624e5061b7eb70e76d93f24c9b`，干净身份已核验。
发布 `development/gradpert-capacity-publication-a1d55fa.json` SHA256
`d00729ca8d66fe5dede305af80a94aa22fcba3298f0424154dd9f11cdf10e67c`。
单步扫描dry-run `development/single-a1d55fa-capacity-integration.plan.json`，
SHA256 `53e743e31d453259ed1136afbb4854b4b67ccb72e6566c667ea6be35f1ad678d`。
候选micro8/16/32/48/64，eager/重计算/同步数据路径；dry-run未执行。
服务器同stem的.sh启动器已准备（双GPU锁、1800s超时、失败停止），尚未启动。
待当前ABBA完成和取舍后再启动；如果启用预取则需要先正式配置化并重新发布，
不可让benchmark-only override冒充容量/正式设置。

## 运维

本地工作树 `/Users/elan/code/grad-pert-v2-build`，定向提交推送HEAD:main。
服务器Python `/data/yilangliu/GraD-Pert/source/.venv/bin/python`。
SSH `ssh -S none -o BatchMode=yes -o ConnectTimeout=10 10.24.1.91`，当前可用。
服务器无rg，可用Python筛选输出。数据/权重/大trace只留服务器；小收据先dry-run。
共享Git克隆不可无限叠加；独立基底source-v2-replay-473466d-standalone可用。
干净发布应从无ignored egg-info的独立本地克隆生成，保持源码身份核验。
旧运行细节及已停止PID仅在STATUS/history/Git保留，不当作活动任务。
