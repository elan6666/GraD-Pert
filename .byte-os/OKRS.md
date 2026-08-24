# Objective and Key Results

## Objective

交付一个不依赖上游模型运行时、实验边界明确且可从原始数据追溯到论文级结果的 GraD-Pert 五数据集基准工程。

## Key Results

1. 干净环境安装、静态检查、单元测试和合成端到端 smoke 全部通过，并记录命令与环境 hash。
2. Replogle K562、Replogle RPE1、Nadig Jurkat、Nadig HepG2、Norman 五个数据集均通过下载校验、预处理/QC 和 canonical split 门禁；所有模型加载后的 train/validation/test condition IDs 与 `split_hash` 完全一致。
3. 唯一 GraD-Pert B2 路线完成完整训练 step、EMA/center 更新、checkpoint 恢复和 Scouter-style 每条件 300-control 推理；`K_head` 由服务器最坏情况显存 fit-test 一次性锁定；默认 100 epochs 上限与 validation-only early stopping 可验证生效。
4. GraD-Pert、官方包调用的 GEARS、官方包调用的 TxPert 在五数据集上均通过 1 epoch 集成门禁并进入统一 evaluator；只对 GraD-Pert 跑完整训练，三个 nonlearned baselines 做完整推理评估；每个适用指标具有逐 condition 值、macro 汇总、availability 和来源 commit。
5. 每次正式运行在服务器保存完整 manifest、checkpoint、history、prediction PKL、Parquet/JSON/H5AD 与 notebook provenance，并能从 prediction payload 独立重算已发布指标；本地仅同步小型指标、日志、receipt 和 server-artifact pointer。
6. 每个正式服务器任务的 preflight receipt 证明本地 HEAD、GitHub commit 与服务器 HEAD 完全一致，且服务器 worktree 干净；任何不一致均中止任务。
7. GraD-Pert、GEARS、TxPert 与每个 nonlearned baseline 的五数据集配置均为独立、自包含文件，配置矩阵完整性和禁止隐藏全局默认由测试验证。

## Current baseline

- 独立 Python 包、30 个配置、B2 模型/训练器、三种 nonlearned、官方
  GEARS/TxPert 隔离适配器、统一 evaluator、artifact/catalog 和 notebook
  已实现；本地 137 tests pass，9 项因可选依赖/待同步收据跳过。
- 五个数据集已在服务器完成 datasets-v2 canonical/QC/split/control/graph/
  evaluator 门禁；共享可表示性过滤后的划分已冻结，小收据待回传。
- 服务器持续 128-step 容量门已选择全数据集统一 `K_head=16384`。
- 尚无完整的 15 个 learned one-epoch 结果、seed-1 nonlearned 正式矩阵或
  GraD-Pert full 结果；不得据此作性能或交付完成声明。
- 当前外部执行受服务器工具额度窗口和首次公开 push 批准约束；恢复后
  仍须重新执行 GPU/磁盘/job/source preflight。

## Evidence required

- 冻结代码与论文审计记录。
- 数据 checksum、QC、split/evaluation manifest 与 hash。
- 本地测试、lint/typecheck/build 记录。
- GPU fit-test 与正式训练/评估 receipts。
- 本地/GitHub/服务器同 commit 的 preflight receipt 与 server-only artifact pointer。
- 当前 review、三轮 iteration 和最终 delivery 记录。
