# GraD-Pert

## Product concept

一个可复现、独立安装的单细胞基因扰动响应预测研究包：实现唯一锁定的 GraD-Pert B2 路线，在五个数据集上用完全相同的条件切分和统一评价器比较 GraD-Pert、GEARS、TxPert 与 nonlearned baselines，并产出可重算、可供 notebook 和下游实验复用的结果。

## Target users

- 主要用户：GraD-Pert 研究与论文实验团队。
- 次要用户：需要复现实验、审查 split/指标公平性或复用预测产物的合作者。

## Core problem

现有设计散布于 GraD-Pert 讨论文档和多个上游代码库；数据预处理、条件切分、control access、评价公式与输出格式若由不同模型各自处理，会产生不可比较或不可复现的结果。

## Delivery format

- 独立 Python 包与 CLI。
- 五数据集 registry、下载/QC/canonicalization/split 工具。
- GraD-Pert B2 训练与推理实现。
- 隔离的 GEARS、TxPert benchmark runners 和中立 prediction artifact 边界。
- nonlearned baselines、统一 evaluator、PKL/Parquet/JSON/H5AD 产物与分析 notebooks。
- 正式计算部署在 `/data/yilangliu` 服务器；GitHub 作为本地与服务器之间唯一源码同步面，大数据和大产物不离开服务器。

## Current stage

Byte Auto 启动；正在进行设计、论文、冻结参考代码、数据源和 benchmark 接口审计，尚未开始产品代码实现。

## Success criteria

1. 五个数据集都能进入可验证的 `canonical_ready` 状态，并冻结数据、基因顺序、split、control-access 与 evaluation hashes。
2. 单一 GraD-Pert B2 配置可安装、训练、恢复、推理并通过组件与端到端测试。
3. GEARS、TxPert 和 nonlearned baselines 在相同条件切分上输出统一 artifact。
4. 主榜同时报告 TxPert macro Pearson delta、TriShift Pearson delta、Systema Pearson 及所有适用的冻结指标并集。
5. prediction payload 能独立重算指标；notebook 仅消费冻结 artifact。
6. 正式任务以三端 commit 一致性为硬门禁，本地只保留允许回传的小型结果和服务器产物指针。
