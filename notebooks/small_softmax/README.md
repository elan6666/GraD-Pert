# 小模型全量 Jurkat：训练后自动分析

从当前小模型工作流启动入口运行。训练成功 → 冻结末轮 checkpoint 验证 → 平台标准 CPU 训练分析 → 执行三个 notebook → 平台导入并下载核验附件。沿用原有的 445 个验证条件、每条件 300 个冻结对照和三套 Pearson。原训练配置、选模、优化器、检查点和测试集不参与这次改动。

| Notebook | 检查的问题 | 主要图 |
| --- | --- | --- |
| [01 训练过程](01_training_process.ipynb) | 优化、资源与表征是否异常 | 损失和耗时曲线、原型和共享梯度、输出误差、各层方差和有效秩 |
| [02 未见扰动](02_unseen_perturbations.ipynb) | 响应偏小、方向偏差、还是只在弱响应条件上失效 | 九组 Pearson 分布与有效数、按真值细胞数和效应强弱分层、幅度对角图、相对复制对照与训练均值的增益 |
| [03 下游诊断](03_downstream_tasks.ipynb) | 模型能否找对非靶强效基因，以及预测是否保留扰动身份 | Top-20/50 精确率、召回率和方向准确率；扰动身份 Top-1/5、MRR 和秩分布；复制对照、训练均值、随机参考 |

这三本都产出带内嵌图的 `.ipynb` 和独立 PNG。平台以同一实验同一标准分析条目保存，图可在“全部原始附件”打开；平台只允许 JSON 扩展名的文本附件，因此已执行 notebook 可从同名附件下载并改回 `.ipynb`。历史回执没有逐条件响应汇总，能回补训练与九组 Pearson 等图；下游 notebook 显示未采集。

## 使用方式

安装一次笔记本环境，例如：

```sh
uv venv ~/.local/share/gradpert/notebook-analysis
uv pip install --python ~/.local/share/gradpert/notebook-analysis/bin/python \
  -r notebooks/small_softmax/requirements.txt
```

新训练沿用已验收的两轮小模型计划、10 步独立预检、训练卡 UUID、单卡冻结验证启动器及独立核验器。新的启动器在 `resolve_evaluation_plan(Namespace(...))` 中显式传 `response_diagnostics=True`，生成的冻结验证计划也保留同名字段。将它和 `response_diagnostics: true` 的工作流 spec 一起封存，保持本地、GitHub 和服务器的正式训练源一致。spec 还需给出 `experiment_id`、`run_id`、`training_plan`、`plan_sha256`、`expected_updates`、`training_gpu_uuids`、`evaluation_gpu`、`evaluation_launcher_sha256`、`verifier_sha256`。

```sh
python scripts/v2/run_small_softmax_workflow.py \
  --spec /absolute/path/to/SPEC.json --spec-sha256 <sealed-sha256> \
  --notebook-python ~/.local/share/gradpert/notebook-analysis/bin/python \
  --platform-root /absolute/path/to/experiment-platform \
  --output /absolute/path/to/workflow-artifacts
```

该入口拥有训练至平台导入的顺序控制。远端在计划目录的 `platform-workflow-v2` 保留 PID 身份与阶段状态；本机断线后用同一 spec 和输出目录重接。历史实验用 `--collect-only`：读取已通过独立核验的服务端回执，不启动 GPU 任务。仅本机复画可运行 `execute_small_softmax_notebooks.py --inputs <artifacts> --output <notebooks>`，其输入只包括已核验训练汇总、验证回执和逐条件标量，不包含表达矩阵、预测、细胞 ID 或模型权重。

监督读取远端 `state.json` 以及本机输出的 `STATUS.json`、`READY.json` 或 `FAILED.json`。只有 `READY.json` 且平台 readback 已核验，才算走完全流程；`FAILED.json` 的 stage 区分训练、评估、报告、图形与归档，排查后使用原 spec 和输出目录恢复。已有训练终态、验证终态或失败运行不会被静默重启。已有监督会话沿用当前定时任务，只在终态或异常汇报。

## 解释边界

- 未见扰动是验证条件，seen/unseen expression 是输出基因暴露，两条轴分开。
- 按冻结配对对照估计条件均值响应，用真值强弱等频分组来诊断；分组不用于训练、选模或改动测试集。
- Top-K 是绝对效应大小的描述性代理，排除直接靶基因，不等同于显著 DEG；方向准确率把零预测视为未恢复非零真值方向。
- 扰动检索在冻结条件库内部比较；每个条件用自己的冻结对照池定中心，零真值向量不进库，并列按随机排列的期望计分。训练均值来自训练条件等权参考。CPU 候选库留在工作进程内，分片均使用同一候选库。
