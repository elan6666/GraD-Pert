# Jurkat 训练集 cap40 下采样与实测重复性

## 已确认范围

仅对冻结 split 的训练扰动条件，按每条件 `min(40, N_p)` 无放回采样；
control、验证、测试、未纳入 split 的行全部保留。保留全部 6506 个变量，
不重新归一化、不筛基因、不改变已有表达值。统计使用原来的 5000 个表达基因。
采样种子为 42，生成一份固定样本；100 次指该样本的重复拆半，不是 100 份 cap40 数据集。
采样文件是分析衍生数据，标为 `analysis_derivative_not_canonical_ready`，
不替换原 canonical 数据或训练默认配置。本次不启动模型训练。

每个条件内，按其 batch 细胞数比例分配 40 个名额：先向下取整，再按小数余数
分配剩余名额，余数相同随机打破平局；batch 内均匀抽样。稀有 batch 可能得到
零名额，逐条件记录采样前后 batch 数，不能称所有 batch 都完整保留。

原训练集为 1335 条件、128266 行，预计保留 47836 行；原总行数 238977，
衍生文件预计为 158547 行。control 12013 行、验证 41235 行、测试 54658 行、
其他排除行 2805 行均保持不变。这些计数来自本次服务器原数据核验，实际结果另行验收。

## 对比统计及解释

对条件 p 的细胞，重复 100 次生成两个互不重叠的子集 A、B。每个 batch 内以及
条件总体，两侧行数之差不超过 1；奇数 batch 的额外一行随机分配并平衡总体。
对两侧分别先计算基因表达均值，再计算 Pearson：

\[
\mu_{p,A}=|A|^{-1}\sum_{i\in A}x_i,\quad
\mu_{p,B}=|B|^{-1}\sum_{i\in B}x_i,\quad
r_{p,k}^{\Delta}=\operatorname{corr}_{g=1}^{5000}
(\mu_{p,A}-c_p,\mu_{p,B}-c_p).
\]

其中 `c_p` 为**原条件 batch 上下文中全部匹配 control 行的均值**，两组使用
相同原始 control 池，不随采样改变。这里不运行模型，也不用预测评估的 300-control
输入清单；那两份冻结清单只做前后哈希核验。只包含 Jurkat，因此 batch 即本次上下文。
Pearson 为对基因轴去均值后的点积除以两向量长度之积；全程用 float64 做统计。
另保存不减 control 的原表达 Pearson，用来区分基线表达相似与扰动变化重复性。

先对每个条件的有效 100 次结果取平均，再对有效条件等权平均。
只有一个细胞或常数向量导致相关性无定义时，保存缺失原因所需计数并排除分母，
不记作零。n≤40 的条件原样保留且拆半种子一致，两版本结果应完全相同。
对条件均值差值做 1000 次条件 bootstrap，给出描述性 95% 区间；
它不表示独立生物重复的置信区间。另按原条件细胞数分为 1–40、41–80、81–160、>160。

辅助指标仅在 N_p>40 的条件汇总：保留40行 vs 原完整均值（有样本重叠，偏乐观），
以及保留40行 vs 被移除行均值（互不重叠）。不能将前者当作独立重复性。
减去共同 control 参考本身也可能引入共享噪声，因此这一统计不是模型分数、
不是数学上的预测上限，也不能替代批次或生物重复验证。

参考 [TxPert 的实验重复性说明](https://github.com/valence-labs/TxPert/tree/08d82eea86746b044cf7531f4ec8c5f60e1cb73f#experimental-reproducibility)
及冻结实现 `gspp/models/baselines.py:349–515`：真实细胞子集均值与未使用细胞比较。
本次是该思路的分层拆半适配，不声称原样复现官方 ExperimentalAccuracy，
也不是其原始 counts 的 multinomial 采样估计。现有 canonical log 表达不转回 counts。

cap40 改变条件间细胞数，因此即使以后保留 `row_mean`，条件的训练总权重也会变化；
训练步数理论上可减少约 62.7%，但 joint 验证和终态评估规模保持原样，
总训练时长及模型性能必须另做实验，不能从数据量直接推断。

## 版本与运行

改动前 main 已推送并核验：`5d602a02158af679a4d384f677176ec910fa238a`。
原数据根：`/data/yilangliu/GraD-Pert/data/nadig_jurkat/within_cell_unseen_single`。
原 H5AD SHA256：`65b32637b24e6ee6d3399b3280d914dcedc34bf787098c3e9e14a47ebe80cbb5`。
原 split SHA256：`d8807af3f03f4ba6c57ddb4f77acdb915b3b180592bca4ba1619f3e49cbe6890`。
发布实现后，以干净同 SHA 服务器 checkout 在 `/data/yilangliu` 上执行：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
  /data/yilangliu/GraD-Pert/source/.venv/bin/python \
  scripts/analysis/train_downsample_reproducibility.py \
  --data-root /data/yilangliu/GraD-Pert/data \
  --output /data/yilangliu/GraD-Pert/development/UNIQUE_RUN_ID \
  --repeats 100 --seed 42
```

CPU-only，最多4个数值库线程。输出目录须为新目录，不能覆盖旧运行。
原始/采样细胞、完整表达、H5AD、行 ID 清单、逐次统计留在服务器；
本地只回传经大小清单核对的小型结果、收据和图。

产物清单：`cap40.h5ad`、`selection.json`（固定行 ID 和条件清单）、
`repeat_scores.csv`（逐条件/版本/重复的两个 Pearson）、`conditions.json`、
`summary.json`、`comparison.png`、`comparison.pdf`、`receipt.json`、
`COMPLETE.json`，以及实时 `progress.json`。完成需退出0、终态收据、
原文件/四份 manifest 哈希不变、衍生 obs/var 精确一致且所有保留表达逐值一致。

## 实现验证

本地定向检查：8项通过，覆盖采样配额与不重复、少细胞原样保留、
batch/整体拆半平衡和 singleton 随机化、向量化统计与直接计算一致、
全量 retained 表达和元数据导出一致，以及完整合成数据分析。
服务器同源码复核和实际数据结果待发布后执行；不把合成检查当作数据集分析完成。

## 实际数据接口修复

首轮 `jurkat-cap40-1af4195-20260930T093216Z` 以源码
`1af419582884519c8c24ed66aa42de7126d79d7b` 启动，退出1。
在采样前的基因轴校验停止，原始文件无写入：Jurkat H5AD 的 var index 是 ENSG ID，
canonical expression_gene_ids 是 gene_name 符号，脚本错误地把两者直接比较。
修复按已冻结 registry 的 gene_symbol_column 对齐表达列，并核对符号/观测顺序哈希；
保留原 ENSG index 和 gene_name 元数据，不改数据值或统计方法。
合成端到端测试也使用 ENSG index 与 gene_name 两种身份，8项定向检查重新通过。
旧失败收据、日志和目录保留，新源码和新 run ID 另行执行。

服务器直接 GitHub 克隆因 TCP/443 超时失败；保留传输失败收据，改用已推送提交的
完整 Git bundle 经 SSH 传输并核验 SHA256，不修改既有服务器 checkout。

第二轮 `jurkat-cap40-af41538-20260930T093845Z` 的1335条件统计完成，
但 native AnnData 切片在导出时删除未使用的 obs 分类水平，严格元数据检查退出1。
源码 `af415382232832fb455316bc210c1ee84e765473` 与失败目录保持不变；
不把已有图或衍生 H5AD 当作验收完成。修复在切片后恢复原 obs/var 的分类字典，
不放宽断言；回归用仅存在于删除行及从未出现的类别、非默认类别顺序验证。
重新发布并使用独立运行 ID，仍需全矩阵逐值检查、原文件哈希及终态验收。
