# 共享 GenePT 投影的非线性版本

用户于2026-10-10要求在共享 GenePT 投影后加入激活函数。新路径为

\[
h_g^{(0)}=\operatorname{LayerNorm}\bigl(
  \operatorname{GELU}(Wv_g+b)\bigr),
\quad v_g\in\mathbb R^{2048},\quad W\in\mathbb R^{256\times2048}.
\]

原始6506×2048 GenePT表保持冻结；所有基因共享同一个随机 Xavier
初始化的线性层，bias初始化为0。GELU使用PyTorch默认精确形式
`x * Phi(x)`，没有新增第二个线性层、PCA、重建任务或独立可训练基因表。
投影先激活，再做既有LayerNorm；随后沿用图编码器和完整预测＋双蒸馏。

## 配置与版本边界

新自包含配置：
`configs/v2/cap40_genept_gelu_20261010/gradpert_v2/nadig_jurkat.yaml`。
它在历史U4自包含配置上只增加
`genept_projection_activation: gelu`。原U4的配置、manifest、运行及结果不改。
此配置未加入任何已完成队列，也未启动训练。

代码支持`none`与`gelu`；缺省`none`保持历史线性U4的输出、随机数、
参数名称及checkpoint架构身份。启用GELU必须同时启用
`learned_genept_projection=true`，激活类型写入新checkpoint架构及数据身份。
完整状态恢复拒绝在线性与GELU架构间混用；修改方法不等于原实验续训。

GELU没有参数。因此Student参数量仍为47,852,590，其中可训练
34,528,302、冻结原始GenePT 13,324,288；共享Linear仍为524,544参数。
Teacher采用同构激活路径并沿用EMA更新，不新增优化器参数或更新时点。

## 验证与结果解释

验证覆盖实际图前向中LayerNorm前的GELU输入、冻结原始先验、共享投影梯度、
公共骨干初始化/RNG与参数量不变、历史配置兼容及跨激活checkpoint拒绝，
并复用三步非零学习率optimizer/Teacher/center/RNG恢复测试。

这是方法更新，会改变表示和梯度；不是数学等价提速。旧U4的六轮指标只代表
线性版本。新版本尚无真实数据CUDA容量、吞吐或科学评估证据，不据局部测试
声称效果改善。后续训练需要新源码、配置身份及独立运行ID。

本地验收：98项相关测试通过；改动文件Ruff与format通过；三个改动源码在
`mypy --follow-imports=silent`下通过。普通mypy的17项导入模块错误在改动前
基线亦复现，本次未修改这些模块。紧凑验收记录见
`.byte-os/receipts/genept-gelu-local-20261010.json`。
