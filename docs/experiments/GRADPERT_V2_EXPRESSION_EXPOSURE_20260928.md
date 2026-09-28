# B0 best.pt：按训练期表达可见性分组评估

Jurkat 完整 B0（三轮，seed 1）的 `best.pt` 在冻结 test split 上重新推理，
以训练期**是否作为数值表达 token 出现**划分 5,000 个评估基因。
训练采样允许集合的 4,775 个基因归入“见过表达”；因 test 扰动靶点
剔除协议而不允许采样的 225 个基因归入“未见表达”。CPU 回放了训练期
44 个 batch，证实允许集合全部被覆盖。未见表达并不意味着未见过该基因
的 GenePT 身份或图先验。

| 评估基因组 | 基因数 | TxPert Pearson（有效/总条件） | TriShift Pearson（有效/总条件） | Systema Pearson（有效/总条件） |
| --- | ---: | ---: | ---: | ---: |
| 见过表达 | 4,775 | 0.422760（592/592） | 0.432257（590/592） | 0.257070（590/592） |
| 未见表达 | 225 | 0.138621（592/592） | 0.172251（137/592） | 0.119023（137/592） |
| 原始全基因 best test | 5,000 | 0.212616（592/592） | 0.326909（590/592） | 0.193016（590/592） |

三个数均为各自协议下的条件宏平均 Pearson。DE 限定的 TriShift 和
Systema 在未见表达组只有 137 个有效条件，主要因为 DE 与该组交集不足；
不能把它们直接当作与 590 个条件同覆盖的性能差。全基因 Pearson 也
不是两个分组 Pearson 的基因数加权平均。该分析只测一个 best checkpoint，
并非跨种子或模型对照的科学结论。

**可核对身份。** 训练源码 `7713158cd3be4c0b65585f6500be234fc7197635`；
评估源码 `9ea0815c30fdac4d6300463dcafe68913c9d9218`；best epoch 3、
checkpoint SHA256 `8edafbebeaa4d3a19224c5fec6c146f0ba34ba64bc770dc334016f14c6518b1e`；
评估 JSON `/data/yilangliu/GraD-Pert/analysis-v2-exposure-9ea0815/best-stratified-test.json`
的 SHA256 为 `603ffa7e2cba7f78e53f0566a73b56fdceaab0af00eb99ba1fe791bc84e8f840`。
该 JSON 包含原测试收据、训练身份、数据/分割/控制清单哈希和每条件结果；
原始全基因指标与既存 best-test 收据逐项一致。表达可见性 CPU 回放收据
为 `/data/yilangliu/GraD-Pert/analysis-v2-exposure-f0f8246/best-exposure.json`，
SHA256 `51a528ce79f94a8fd02b99437212c8a617d767095c8d93f6faec5c88c369a31a`。
大矩阵、checkpoint 与逐条件产物仍留在服务器。
