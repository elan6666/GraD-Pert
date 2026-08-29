# Objective and Key Results

## Objective

交付一个不依赖上游模型运行时、实验边界明确且可从原始数据追溯到论文级结果的 GraD-Pert 五数据集基准工程。

当前扩展目标：在不破坏历史 v1 证据的前提下，交付同一主入口、完全由
自包含配置驱动的 B2-vNext 单图/多图消融系统，并完成 Nadig Jurkat
固定 split、seed 1、10 epoch 的预注册服务器实验。

## Key Results

1. 干净环境安装、静态检查、单元测试和合成端到端 smoke 全部通过，并记录命令与环境 hash。
2. Replogle K562、Replogle RPE1、Nadig Jurkat、Nadig HepG2、Norman 五个数据集均通过下载校验、预处理/QC 和 canonical split 门禁；所有模型加载后的 train/validation/test condition IDs 与 `split_hash` 完全一致。
3. 唯一 GraD-Pert B2 路线完成完整训练 step、EMA/center 更新、checkpoint 恢复和 Scouter-style 每条件 300-control 推理；服务器容量门锁定 batch 256、16,384 prototypes 和 expandable allocator。当前默认使用完整图和七项系统优化的 B2；Nadig Jurkat seed 1 进入最多 200 epochs、validation-only patience 10 的正式训练。
4. GraD-Pert、官方包调用的 GEARS、官方包调用的 TxPert 在五数据集上均完成 1 epoch 集成门禁并进入统一 evaluator；三个 nonlearned baselines 完成统一评估；每个适用指标具有逐 condition 值、macro 汇总、availability 和来源 commit。
5. 每次正式运行在服务器保存完整 manifest、最佳 checkpoint、history、推理配方、精确 control/truth IDs 与 notebook provenance；默认不保存 PKL，需要逐细胞下游时才显式生成一个去重 `result.pkl`。本地仅同步小型指标、日志、receipt 和 server-artifact pointer。
6. 每个正式服务器任务的 preflight receipt 证明本地 HEAD、GitHub commit 与服务器 HEAD 完全一致，且服务器 worktree 干净；任何不一致均中止任务。
7. GraD-Pert、GEARS、TxPert 与每个 nonlearned baseline 的五数据集配置均为独立、自包含文件，配置矩阵完整性和禁止隐藏全局默认由测试验证。
8. B2-vNext 默认配置、HVG512+targets、RingInduced、运行时图节点覆盖比例
   `1/2`、8 个 local、mask-view 比例 `0/1`、native multi-source sparse
   graph Transformer、四项损失权重、decoder 和 GenePT coverage contracts
   均通过合成黄金测试、服务器 CUDA gate 与 config/receipt identity 验证。
9. H 图规模与 L 局部图两个 Nadig 10-epoch 模块在运行前冻结；每个可用坐标只训练一次，
   test 只访问一次，零持久化 PKL；不可用 GenePT 坐标保存权威缺失-target
   receipt 而不启动训练。
10. 新 A0 先完成真实无 test-truth profiling；只有实测达到门槛的实现瓶颈
    才允许优化，并通过逐 view/union/RNG/梯度/状态精确等价与单 GPU ABBA
    计时后，才成为 H/L 正式实验的源代码谱系。

## Current baseline

- 独立 Python 包、30 个自包含配置、B2 模型/训练器、三种 nonlearned、
  官方 GEARS/TxPert 隔离适配器、统一 evaluator、artifact/catalog 和
  notebook 已实现。
- 五数据集 exact 15 learned one-epoch + 15 nonlearned 坐标已完成并通过
  combined audit；跨模型 protocol/canonical/split/ordered-300-control
  identity 全部一致。
- 所有模型和 nonlearned 默认 `metrics_only`，成功 run root 持久化 PKL
  数量为零；只有显式 `single_pkl` 才生成一个去重 `result.pkl`。
- B0/B1/B2/B3 Nadig Jurkat 速度 pilot 已完成。独立 B0 timing coordinate
  使用完整图、关闭七项系统优化、1 epoch、`metrics_only` 和零持久化
  PKL，历史 c240 B0 保持不变；四个单因素对比与组合对比均使用同一
  wall-time 口径，B3 以 5.813x 相对 B0 加速获选。三指标仍不用于声称
  效果不变。

## Evidence required

- 冻结代码与论文审计记录。
- 数据 checksum、QC、split/evaluation manifest 与 hash。
- 本地测试、lint/typecheck/build 记录。
- GPU fit-test 与正式训练/评估 receipts。
- 本地/GitHub/服务器同 commit 的 preflight receipt 与 server-only artifact pointer。
- 当前 review、三轮 iteration 和最终 delivery 记录。
