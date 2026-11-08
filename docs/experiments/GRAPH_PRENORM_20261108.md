# 图分支前置归一化两组实验

用户于2026-11-08授权实现、准备和运行两组对照，明确每组完整2轮，
沿用B0的R3-RI基线。当前先完成代码与启动准备；长训练等待交给已有监督会话。
监督未接入、资源准入未通过时不启动长训练。

## 问题与两组设计

现有主干已采用前置归一化：图分支为LayerNorm，细胞与响应分支为RMSNorm。
本实验询问：仅将图注意力和图前馈入口换成RMSNorm，是否能改善表达预测或运行效率。

- 对照组：`graph_input_norm: historical`，保持图层现有LayerNorm。
- 候选组：`graph_input_norm: rmsnorm`，只替换每个图层的`norm1`和`norm2`。
- 两组均保留GELU、原有残差/mHC、参数初始化、先验映射、输出归一化、
  MLA内部KV归一化、预测/蒸馏头、损失、优化器、精度和学习率日程。
  用户追加要求将已有吞吐优化用于后续训练，因此两组均显式开启
  `relay_validate_once`与`cache_kda_constants`；两组之间这两个设置完全相同。
- 每个替换保留该位置的eps和scale初始化。标准RMSNorm不含beta，
  因而候选少8×256=2048个图LayerNorm偏置参数；这属于本次机制差异，
  不能称为严格参数匹配或数值等价实验。
- 不启用全局`ffn_input_norm`，不加入Q低秩压缩，不给无残差映射额外加Norm。

参考DeepSeek-V3的注意力/前馈入口RMSNorm布局；这是位置设计参考，
不是完整架构对齐。原生实现不导入或调用外部模型。
参考源码：https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

## 冻结协议

复用`configs/v2/training_speed_20261106/baseline/gradpert_v2/nadig_jurkat.yaml`，
其模型配方来自原R3-RI，单卡micro80×accum4、全局batch320、seed1，
TRAIN40选择清单和原条件/表达可见性划分保持。共同执行默认见
`TRAINING_EXECUTION_DEFAULTS.md`。配置自包含，两组之间只允许
`graph_input_norm`这一模型参数不同；相对历史母本还共同增加两个显式执行开关。
这是单独命名的2轮诊断实验，不属于R50正式50轮矩阵。

两组从头训练，各完整2个数据epoch；不加载旧权重，不覆盖历史run。
相同初始随机种子不保证机制不同后增强RNG始终相同；记录实际样本顺序和RNG证据。
尽量在同一物理GPU顺序执行，使用原GPU0资源准入、锁、allocator及CUBLAS合同。
记录训练源码SHA、配置SHA、真实更新数/样本数、检查点SHA、环境与硬件身份。

训练后评估epoch2 last：使用原445条件验证集及冻结300-control配方，
报告TxPert、TriShift、Systema三种表达Pearson及MSE，保留有效条件数和NA，
并分列训练见过/未见表达基因。复用冻结TRAIN参考，不做DEG，不读取test调参。
训练期间不依靠验证结果提前停止或调整超参数。

同时报告训练总耗时、稳定阶段更新/样本吞吐和峰值显存。
若只有速度收益，必须说明预测指标是否退化；若指标方向冲突，逐项保留。
单seed的微小差异只能作为描述性证据，不能宣称稳定提升。

## 验证与启动边界

合成测试需证明：历史payload不改变；只有8个图入口Norm改变类型；
其余权重初始化与RNG不变；所有非目标Norm保持；候选能前后向更新、Teacher EMA，
且checkpoint恢复后的下一步与连续执行相同。
正式运行前须完成本地/服务器适当测试、review、main发布与匹配clean源码核验。
容量检查只证明集成和资源，不当作2轮完成证据。

监督会话只检查明确run及终态，健康时安静，完成或异常回报主会话，
主会话独立验收并继续表达评估和结果归档。当前宿主禁止为单个任务创建
完成追踪型应用自动化；不能把未启用的定时任务或未ACK的连接写成已交接。
