# User and Workflow Assumptions

## Confirmed

- 用户需要真正完成五个数据集，而不是只交付合成 smoke。
- 第一阶段只做一个 B2 GraD-Pert 模型版本，不设计或运行消融。
- 公平比较要求各模型在同一数据集上使用相同的训练、验证、测试扰动条件和评价采样。
- 输出必须方便复现、notebook 分析和下游实验。
- 正式训练及其数据准备、推理、指标物化在服务器完成。
- 代码仓库为 `elan6666/GraD-Pert`，本地、GitHub 与服务器代码必须同步到同一 commit。
- 数据集和checkpoint留在服务器；默认不生成PKL，显式生成的单一
  `result.pkl`也不回传，本地只取小型结果文件。
- GraD-Pert full runs 默认使用完整图和七项系统优化的 B2，训练上限为
  200 epochs，并启用 validation-only early stopping `patience=10`。

## Inferred and to validate

- 首要工作流是研究工程师通过 CLI 准备数据、训练/评估模型，再由 notebook 读取冻结产物生成表格和图。
- 正式实验使用已授权服务器与现有 SSH 身份，但每次执行前必须重新核对连接、磁盘、GPU、运行环境和三端 commit。
- 五数据集下载许可、公开链接与发布 split 是否完整仍需真实审计。
- `早停 epoch=10` 解释为连续 10 次 validation 无改善即停止，即 `patience=10`；`min_delta=0`。若用户指的是最少训练 10 epochs，需要在正式 test 前改写该合同。

## Explicitly excluded

- 第一阶段消融、cross-cell leave-one-out、额外数据集、额外 learned baselines 和性能最优宣称。
- notebook 内隐藏唯一的数据切分、训练或指标实现。
- 依赖 TxPert、GEARS 或 DINO checkout 才能运行 GraD-Pert 原生包。
