# Objective and Key Results

## Objective

交付一个不依赖上游模型运行时、实验边界明确且可从原始数据追溯到论文级结果的 GraD-Pert 五数据集基准工程。

## Key Results

1. 干净环境安装、静态检查、单元测试和合成端到端 smoke 全部通过，并记录命令与环境 hash。
2. Replogle K562、Replogle RPE1、Nadig Jurkat、Nadig HepG2、Norman 五个数据集均通过下载校验、预处理/QC 和 canonical split 门禁；所有模型加载后的 train/validation/test condition IDs 与 `split_hash` 完全一致。
3. 唯一 GraD-Pert B2 路线完成完整训练 step、EMA/center 更新、checkpoint 恢复和 Scouter-style 每条件 300-control 推理；`K_head` 由服务器最坏情况显存 fit-test 一次性锁定；默认 200 epochs 上限与 validation-only early stopping 可验证生效。
4. GraD-Pert、GEARS、TxPert 与三个 nonlearned baselines 都能进入统一 evaluator；每个适用指标具有逐 condition 值、macro 汇总、availability 和来源 commit。
5. 每次正式运行在服务器保存完整 manifest、checkpoint、history、prediction PKL、Parquet/JSON/H5AD 与 notebook provenance，并能从 prediction payload 独立重算已发布指标；本地仅同步小型指标、日志、receipt 和 server-artifact pointer。
6. 每个正式服务器任务的 preflight receipt 证明本地 HEAD、GitHub commit 与服务器 HEAD 完全一致，且服务器 worktree 干净；任何不一致均中止任务。
7. GraD-Pert、GEARS、TxPert 与每个 nonlearned baseline 的五数据集配置均为独立、自包含文件，配置矩阵完整性和禁止隐藏全局默认由测试验证。

## Current baseline

- 设计讨论和参考代码已迁入当前工作区。
- 当前没有正式 Python 包、测试、canonical data 或训练结果。
- 2026-08-24 已重新观测服务器为 2 张 RTX 5090（每张 32607 MiB），两张卡各空闲约 32110 MiB，`/data/yilangliu` 可用约 1.6 TiB；正式运行前仍须再次执行 preflight。

## Evidence required

- 冻结代码与论文审计记录。
- 数据 checksum、QC、split/evaluation manifest 与 hash。
- 本地测试、lint/typecheck/build 记录。
- GPU fit-test 与正式训练/评估 receipts。
- 本地/GitHub/服务器同 commit 的 preflight receipt 与 server-only artifact pointer。
- 当前 review、三轮 iteration 和最终 delivery 记录。
