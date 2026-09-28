# 后续 GraD-Pert v2 运行：joint 验证与三组测试指标

此协议仅作用于使用本实现启动的**新 v2 运行**；历史 checkpoint、验证结果和
best/last 选择依据不追改，v1/R50 不变。正式运行仍须发布干净源码、使用新的
运行 ID 并记录配置和源码哈希。

每轮结束后，验证只读取冻结的 validation 扰动细胞及其匹配 control，按
`run_seed + 0x5A11D` 构造固定的配对、批次和图/表达视图；不消耗训练用的
NumPy 或 PyTorch 随机状态。验证对每个批次执行完整 `JointObjective`：

\[
L_{\rm joint}=L_{\rm pred}
+\lambda_1(0.8L_{1,\rm condition}+0.4L_{1,\rm node}+0.1L_{1,\rm spread})
+\lambda_2(0.8L_{2,\rm DINO}+0.4L_{2,\rm iBOT}+0.1L_{2,\rm KoLeo}).
\]

系数以每份自包含配置的实际 `lambda1/lambda2` 为准。批次 joint loss 按该
批次细胞数加权汇总；KoLeo 的近邻集合是该验证批次中的有效样本。Teacher、
center 与优化器在验证时不更新，`pending` 统计清空。固定 validation
`joint_loss` 是新运行选择 `best.pt` 的唯一标量；独立保留既有 300-control
`prediction_loss` 和三种 Pearson 作为诊断，测试指标不用于选模型。
固定视图不表示与训练时随机视图逐步同值；批次平均也不等于把整个
validation 集合一次性输入网络。

训练在每次**成功的 optimizer 更新**之后记录该更新中作为真实 control 和
扰动表达输入/监督的基因索引，逐轮累计写入 `fit/history.json`。best 与 last
分别按自己对应 epoch 的累计记录划组，而不是把“允许采样”误记为“已采样”：

- 全表达轴：原有顶层 `metrics`，保持既有输出路径。
- `seen_expression`：到该 checkpoint 为止训练更新中实际出现数值表达的基因。
- `unseen_expression`：同一表达轴中从未作为训练数值表达出现的基因；包括被
  测试靶点剔除协议排除的基因及允许采样但尚未抽中的基因。

每个 checkpoint 对每个条件只执行一次 300-control 完整表达轴推理；同一批
预测用于全轴和两组的 TxPert、TriShift、Systema Pearson。分组与各指标的
基因轴取交集；若交集不足，保留 `null` 和有效条件数，不填零。若某组没有
基因，显式记 `empty_expression_group`。GenePT 身份和图上下文可见不算
“见过表达”。两组基因数、身份哈希、checkpoint epoch 与每条件指标写入
best/last 测试收据。

可用 `python scripts/v2/show_progress.py --run-root <服务器运行根> --follow`
查看训练、joint 验证、预测验证与 best/last 测试阶段。实时进度是诊断文件；
完整完成仍须检查 `COMPLETE.json`、best/last checkpoint 和测试收据。
