# GraD-Pert 项目讨论

日期：2026-08-24

## 用户请求

读取 Codex 任务 `GraD-Pert`（`01a00ab3-9864-7032-98d9-45f6d0016838`）及 `/Users/elan/code/扰动预测/TxPert` 中已经形成的设计，在 `/Users/elan/code/grad-pert` 建立可复现的 GraD-Pert 独立项目。第一阶段只完成一个锁定的 GraD-Pert 模型版本，不设计或运行消融；五个数据集同时接入 GraD-Pert、GEARS、TxPert 和若干 nonlearned baselines，并使用完全一致的数据、split、control access 与统一评价器。

本轮使用 `byte-discuss`，只固化需求与风险，不写产品/实验代码，也不把下面内容当作已经批准的最终执行计划。

2026-08-23 追加确认：五个数据集都要实际完成数据准备和正式实验；旧设计文档迁入新仓库并改为唯一维护源；projection head 容量根据服务器真实 GPU 与显存校准结果决定。

2026-08-24 追加收敛：MVP 只保留一个 GraD-Pert 配置；所有 backbone、训练路线、local、loss、temperature、decoder 和图权重消融全部延期。新增 GEARS、TxPert 两个 learned baseline 和三个 nonlearned baseline，并以同一冻结 split 和统一 evaluator 做公平比较。

2026-08-24 训练路线决定：MVP 暂定采用 B2，从随机初始化直接联合优化表达预测与图自蒸馏；第一阶段不实现 B3 的独立 SSL 预训练、stage handoff 或 B2-versus-B3 对比。若无新的明确决定，后续计划和实现均按 B2 展开。

2026-08-24 评价与产物决定：统一 evaluator 的指标集合取 TxPert 与 TriShift 当前冻结代码的并集。Pearson 主面板明确并列报告三个不同定义：TxPert macro Pearson delta、TriShift Pearson delta 和 Systema Pearson；不得因名称接近而合并。所有模型在同一数据集上必须使用完全相同的 train/validation/test 扰动条件、gene order、control access 和评价抽样；仅 seed 数值相同不构成公平性证明，必须由同一 canonical split/evaluation manifest 及其 hash 约束。实验产物同时服务命令行复现、指标重算、notebook 分析和下游实验。

2026-08-24 control 评价决定：采用 Scouter 的 population inference 设计。对每个测试扰动条件从协议允许的 control 池有放回抽取 300 个 control，生成 300 个预测细胞；真实集合是该条件下全部测试 perturbed cells，不是给每个真实细胞各抽 300 个 control，也不把 truth 强制重采样为 300。共享 evaluation draw manifest 固定实际 control row IDs，确保所有可接收 control 输入的模型使用相同多重集合。

## 当前理解

- `/Users/elan/code/grad-pert` 应成为新的、独立的 GraD-Pert Python 项目；当前目录为空且尚未初始化 Git。
- GraD-Pert 原生模型必须逐模块核对冻结的参考代码后在 `gradpert`
  命名空间独立实现；`src/gradpert` 正式路径不得导入或调用上游模型。
  GEARS 与 TxPert 是例外的隔离 benchmark：其 runner 必须直接调用冻结
  官方 checkout/package 与官方配置，不得在本仓库重写它们的模型。
- v1 的普通图上下文严格使用数据集预处理后的 HVG，唯一强制补入的是已知候选扰动靶基因：`V_master = H_data ∪ P_known`。
- Teacher 看两个完整 global；Student 看相同拓扑的两个 global 和八个扰动中心 local。Student 的一个 global 做 node-iBOT mask，八个 local 中固定四个做 anchor mask。
- 第一阶段只实现一个模型版本，暂定锁定 B2：从随机初始化直接联合训练表达预测与图自蒸馏；不同时实现 B3。
- v1 主数据集固定为 Replogle K562、Replogle RPE1、Nadig Jurkat、Nadig HepG2、Norman；当前本地状态是 `not_downloaded`，不是已准备完成。
- 交付范围不止是工程骨架：五个数据集都要下载、准入、训练、评估并形成可追溯结果；不能只用合成 smoke 代替正式数据运行。

## 已确认决定

### 1. 工程与复现边界

- 包名 `gradpert`，公开类名使用 `GraDPert...`。
- 标准 `pyproject.toml`、冻结 lockfile、`src/gradpert/` 布局。
- 必须提供：
  - `gradpert data prepare --all`
  - `gradpert data status --all`
  - `gradpert verify --all`
  - `gradpert run --dataset ... --config ...`
- 数据准备必须支持断点续传、checksum、安全解压、幂等重跑、manifest、QC、split 和 readiness 状态机。
- 正式运行必须记录 Git commit、lockfile hash、数据 checksum、`preprocessing_hash`、`split_hash`、完整解析后配置、随机种子和环境信息。
- 建立单一 implementation provenance registry，记录上游仓库、冻结 commit、文件/函数、张量形状、更新顺序、许可证、GraD-Pert 对应模块、黄金测试和主动差异。
- 上游代码只允许在隔离参考环境生成黄金张量；GraD-Pert 测试读取冻结结果，不运行时依赖上游包。

### 2. 数据集与 protocol

| dataset_id | 内容 | v1 protocol 要点 | 当前状态 |
|---|---|---|---|
| `replogle_k562_essential` | K562 CRISPRi 单扰动 | TxPert within-cell；normalize 4000、log1p、Top-5000 HVG | `not_downloaded` |
| `replogle_rpe1_essential` | RPE1 CRISPRi 单扰动 | within-cell 与 cross-cell processed view 必须分开 | `not_downloaded` |
| `nadig_jurkat` | Jurkat CRISPRi 单扰动 | within-cell 与 cross-cell processed view 必须分开 | `not_downloaded` |
| `nadig_hepg2` | HepG2 CRISPRi 单扰动 | within-cell 与 cross-cell processed view 必须分开 | `not_downloaded` |
| `norman` | K562 CRISPRa 单/双扰动 | 自有 downloader 获取 GEARS 指向的同一 Dataverse artifact；冻结 doubles split | `not_downloaded` |

数据状态固定为：`not_downloaded → downloaded/upstream_processed → audited → canonical_ready`。只有 `canonical_ready` 可训练。每个 canonical protocol 至少包含 `adata.h5ad`、source/preprocessing manifest、QC、`gene_ids.txt`、split manifest 和 checksum。

Adamson 只允许作为明确声明的 cross-cell 辅助来源；Adamson、Dixit、PBMC 都不进入 v1 五数据集主结果。

### 3. MVP 唯一模型与比较对象

| 组别 | MVP 设计 |
|---|---|
| GraD-Pert | 一个锁定的 B2 + GAT-Hybrid + STRING/GO + Local-RingInduced + D0 additive decoder |
| Learned baseline 1 | GEARS，独立隔离环境运行，读取同一 canonical data 和 split |
| Learned baseline 2 | TxPert，冻结官方实现与配置，读取同一 canonical data 和 split |
| Nonlearned 1 | `matched_control_mean`：只预测目标 cell line/batch 的合法 control 均值 |
| Nonlearned 2 | `global_train_delta`：control 均值 + 训练集全部扰动的全局平均 delta |
| Nonlearned 3 | `additive_seen_singles`：Norman 未见 double 使用两个训练中已见 single delta 相加 |

同一次 EMA 训练的 Teacher 与 Student 必须同构。Teacher 只包含 gene embedding、graph encoder、multi-graph fusion 和共享 DINO/iBOT projection head；不包含 basal encoder、expression decoder，也不读取 control 或真实 perturbed expression。

### 4. 锁定的 MVP 模型值

| 配置 | 默认值 |
|---|---|
| 图来源 | STRING + GO，各自独立 Top-20 |
| 边权重 | W0：仅用于 Top-20 selection |
| global | 2 个；每张来源图独立 DropEdge `p=0.1` |
| local | 固定 8 个 Local-RingInduced |
| local budget | 固定 512 |
| Student global mask | 恰好 1/2 global；ratio `U(0.1, 0.5)` |
| local anchor mask | 恰好 4/8 local；与 global 共用 mask token |
| projection head | `d_pert → 2048 → 2048 → 256 → K_head` |
| `K_head` | 服务器显存 fit-test 后一次性锁定，不作为性能消融 |
| 温度与中心 | `tau_s=0.1`、`tau_t=0.04`、`center_momentum=0.9` |
| Teacher EMA | B2 单一联合阶段内从 0.996 余弦增加到 1 |
| SSL loss | DINO + 1.0 node-iBOT + 0.1 KoLeo |
| 外层权重 | 固定 `lambda_ssl=0.1` |
| prediction view | 完整确定性 Top-20 图；无 DropEdge/local/mask/head |
| inference | 每个扰动条件从合法 control 池有放回采 300 个 control，生成 300 个预测细胞；所有模型共用物化后的 control row IDs |
| decoder | 固定 D0 additive：`Decoder(BasalEncoder(control) + z_pert)` |

### 5. 第一阶段不做消融

第一阶段不得启动 B2-versus-B3、backbone、单图/多图、local 构造、local budget、W0-W3、mask、KoLeo、`K_head` 性能、temperature、decoder 或 transductive/inductive 对比。旧设计文档可以保留这些未来研究问题，但 MVP 配置与实验 manifest 不暴露可批量扫参的消融矩阵；正式训练入口只暴露已锁定的 B2 路线。

GPU fit-test 只用于判断配置能否在 2 x RTX 5090 32 GB 上运行，不读取 validation/test 性能，因此不算性能消融。

### 6. 超参数与实验状态管理需求

建议采用分层配置，但最终工具在 `byte-plan` 阶段确认：

```text
configs/
├── dataset/
├── protocol/
├── graph/
├── model/
├── views/
├── loss/
├── trainer/
├── evaluation/
└── experiment/
```

要求：

- 配置必须有严格 schema 和组合合法性校验，例如 multi-graph backbone 不能只给单一 graph source。
- 所有默认值只有一个来源；实验 YAML 只覆盖差异项。
- 每次运行保存原始配置、解析后配置、配置 hash、环境 lock、数据/split/graph hash。
- 第一阶段不做 sweep。仅允许按 validation `txpert_macro_pearson_delta` 早停或选择 checkpoint；测试集不参与超参数、训练轮数、容量或 checkpoint 选择。
- 预注册固定种子集合，多种子输出均值和标准差。
- B2 联合训练按未见扰动 validation `txpert_macro_pearson_delta` 早停；不存在独立 SSL 预训练 epoch 或 stage handoff 超参数。
- 非法、未知或不可用指标写入 `metric_availability.json`，不得将 NaN 改成 0 或伪造逐细胞配对。

### 7. 验收标准

- 干净环境可安装，并能在离线合成微型数据上完成 `verify --all` 与 smoke run。
- 五数据集 registry、下载器、准入状态机和 protocol-specific preprocessing 有单元测试；没有数据时报告真实状态。
- graph coverage 分别生成 summary、missing genes 和 isolated genes 报告。
- 模型组件有 shape、数值、一次 optimizer step、一次 EMA step 的黄金测试。
- B2 checkpoint 能完整保存与恢复 Student、Teacher、optimizer、scheduler、centers 和 resolved config。
- 训练与评估 artifact 按 `dataset/protocol/split/model/run` 分层，并能追溯全部 hash。
- 代码库中无 GraD-Pert 正式路径对 TxPert/DINO/GEARS 的 import 或 CLI 调用。

### 8. 2026-08-23 追加确认

- **五个数据集全部做**：`replogle_k562_essential`、`replogle_rpe1_essential`、`nadig_jurkat`、`nadig_hepg2`、`norman` 都必须完成下载、checksum、预处理/QC、split、`canonical_ready`、训练、评估和结果 manifest。单个数据集的 smoke 成功不构成总交付完成。
- **设计文档迁移**：把 `/Users/elan/code/扰动预测/TxPert/grad-pert` 下的总设计与讨论文档迁入新仓库 `docs/design/`，保留来源和迁移时的内容哈希；迁移后新仓库是唯一维护源，旧路径只作为历史副本，不再双写。
- **GPU 决定 projection 容量**：不预先硬编码 65,536。计划阶段先在服务器读取 GPU 型号、数量、总显存和当前可用状态，再用五数据集中最坏显存配置执行完整训练 step 校准。
- 容量候选仍为 `K_low`、`2K_low`、65,536；从大到小测试，在包含 Student、Teacher、optimizer、两个 global、八个 local、backward 和 AMP 的真实 step 中，选择峰值显存不超过单卡总显存 85% 的最大候选。
- 为保证五数据集和全部主模型公平比较，容量按最坏情况一次性锁定，不能为每个数据集或模型临时选择不同 `K_head`。任何因硬件采用的降档都必须写入 experiment manifest，而不能被描述为性能调参结果。

### 9. TxPert 与 TriShift 代码核对结果

- 论文规定单扰动数据按 perturbation 分组，目标 train/validation/test 比例为 `0.5625/0.1875/0.25`；validation 和 test 都是互斥的未见扰动。
- Norman doubles 是例外：加载 GEARS setup 的预定义 split；MVP 主 test subgroup 固定为 `combo_seen2`，即 double 的两个 single 组成成分都在训练中见过。
- 论文规定每个模型只用 validation Pearson delta 选择超参数，锁定后才计算 test。
- 论文和官方 `baselines.py` 都以 cell line/batch matched control 为中心计算 delta；任何 baseline statistic 只能来自 train。
- 官方 `datamodule.py` 实际读取冻结的 `splits/train_test_split.pkl` 与 `splits/subgroup.pkl`，这应成为复现实验的首选来源。
- 发现论文/代码默认值冲突：论文目标 validation 比例是 `0.1875`，但当前 commit `08d82eea86746b044cf7531f4ec8c5f60e1cb73f` 中 `define_splits_singles(..., val_size=0.1)` 的默认调用会得到约 `0.65/0.10/0.25`。因此不得直接调用默认参数重新切分。
- TxPert 指标依据本地官方仓库 commit `08d82eea86746b044cf7531f4ec8c5f60e1cb73f` 的 `gspp/metrics.py`；prediction artifact 依据 `gspp/predictor.py` 的 results PKL 和 prediction/ground-truth/control H5AD 输出。
- TriShift 指标与产物依据本地干净 checkout commit `87ac2c51c3c266391093f71a8bce2e6beaa81518` 的 `src/trishift/_external_metrics.py`、`src/trishift/TriShift.py`、`scripts/trishift/analysis/recompute_metrics_from_pkl.py` 和 `_result_adapter.py`。
- TriShift 的可复用模式是：每个 condition 的 PKL 同时保存 `Pred/Ctrl/Truth`、full-gene arrays、DE index/name、gene order、export metadata 和 full metric summary；指标 CSV 可以从 PKL 独立重算，notebook 通过统一 result adapter 读取，而不承担训练逻辑。
- Scouter 官方 commit `0cfddd000e19b72ff033ba67c8315f7bc3304932` 的 `Scouter.pred` 使用 `n_pred=300`，从训练 control 池有放回抽样，并为每个扰动条件返回 300 个预测表达；`Scouter.evaluate` 再将预测均值与该条件全部真实细胞的均值比较。它不是“每个真实扰动细胞配 300 个 control”。

### 10. 公平 split 与评价合同

1. 每个 dataset/protocol 只生成一份 canonical split manifest；GraD-Pert、GEARS、TxPert 和 nonlearned baselines 必须读取同一份 condition 列表，禁止各自重新随机划分。
2. 四个单扰动数据集优先使用 TxPert 发布缓存中的原始 split 文件，并验证实际比例、互斥性和 hash。若发布物确实缺失，才用显式 `test_size=0.25`、`val_size=0.1875`、冻结 seed 生成一次并封存。
3. Norman 优先使用 TxPert 实验发布物所包含的 GEARS predefined split 与 subgroup；禁止用普通单扰动比例替代。
4. 所有模型读取相同表达尺度、有序 gene IDs、condition 编码、batch/cell-line 字段和允许的 control 池。模型适配只能改变容器/张量格式，不能改变样本集合或基因语义。
5. GEARS 使用其官方 `custom` split 能力加载同一 split dictionary；不得调用 `simulation` 重新生成另一套划分。
6. TxPert 直接加载同一 `train_test_split.pkl`/`subgroup.pkl`；GraD-Pert 与 nonlearned baselines 读取同一 manifest 的中立序列化版本。
7. 随机性分为三个独立 namespace：`split_seed` 只用于一次性生成 canonical condition split；`run_seed` 使用同一组 4 个值配对 GraD-Pert、GEARS、TxPert 的训练重复；`eval_seed` 控制 control sampling、分布指标 subsampling 和其他评价随机过程。nonlearned baseline 若无随机过程只运行一次；若涉及抽样则读取相同的 evaluation draw manifest。
8. 公平性的硬门禁不是“各模型配置里的 seed 数字相同”，而是每个模型实际加载后的 train/validation/test condition 集合、顺序规范化结果和 `split_hash` 完全相同。计划优先把具体 condition 列表直接注入三个模型；不得让 TxPert、TriShift 风格 splitter、GEARS 或 GraD-Pert 各自凭同一个 seed 重切，因为它们的排序、RNG 和取整语义不同。
9. 评价抽样优先物化为共享 manifest：按 dataset/protocol/split/condition 从合法 control pool 有放回记录恰好 300 个 `input_control_row_ids`，并固定全部 `truth_row_ids`、DE gene IDs 和分布指标 subsample row IDs。重复的 control row ID 必须保留，不能去重。只在无法物化时才依赖 `eval_seed`，并记录 RNG 算法与版本。
10. 对条件 `p`，统一预测对象为 `InputCtrl_p[300,G] → Pred_p[300,G]`；真实对象为 `Truth_p[N_p,G]`，其中 `N_p` 是该测试条件实际拥有的全部 perturbed cells，通常不等于 300。population mean、variance 和 Wasserstein 可在样本数不同的两组上计算；没有真实一一对应关系，因此 `pearson_sample`、`r2_sample`、`pearson_delta_sample`、`r2_delta_sample` 在主 Scouter-style 协议中标记 `not_applicable:no_paired_truth`。
11. 主榜全部预测进入 GraD-Pert 的统一 evaluator。即使上游仓库自带评价，MVP 主榜也不接受模型自报分数；上游原生分数只作为兼容性核对，并必须注明 evaluator/source commit。
12. Pearson 主面板固定为下列三个指标；每个都先得到逐 condition 值，再按测试 condition 等权 macro 聚合。排行榜和结果表必须显示规范 `metric_id`，同时保留上游 source key：

| 展示名称 | `metric_id` | 冻结 source key | condition 内定义 |
|---|---|---|---|
| TxPert macro Pearson delta | `txpert_macro_pearson_delta` | TxPert `pearson_delta` | 在完整评价 gene set 上计算 `corr(mean(pred-input_control), mean(truth)-mean(input_control))` |
| TriShift Pearson delta | `trishift_pearson_delta` | TriShift `pearson` | 在 condition-specific DE mask 上计算 `corr(mean(truth)-metric_control_pool_mean, mean(pred)-metric_control_pool_mean)` |
| Systema Pearson | `systema_pearson` | TriShift `systema_corr_20de_allpert` | 在 condition-specific Top-DE mask 上，分别将 truth/pred 均值减去 Systema reference 后计算 Pearson |

13. `txpert_macro_pearson_delta` 是 validation 早停、checkpoint 选择和主排序指标。这里 `input_control` 是生成该条件 300 个预测所用的同一组 control；在有限且非退化输入上应与冻结 TxPert 公式数值对齐。TxPert 将未定义相关性改为 0 的行为不进入主实现，主实现保留 NaN 并写明 unavailable reason。
14. `trishift_pearson_delta` 使用该 condition 的 DE gene IDs；按当前 TriShift 代码，优先读取 `top20_degs_non_dropout` 并排除被直接扰动的 target genes。artifact 必须另存 `metric_control_pool_mean`，即该协议完整合法 control pool 的均值，不能与 300 个 `InputCtrl` 的样本均值混用。
15. `systema_pearson` 使用当前 TriShift 兼容实现的 Systema reference：分别计算 train+validation 中每个非 control 扰动条件的表达质心，再对这些条件质心等权平均；随后在当前 test condition 的 Top-DE mask 上比较 truth delta 与 prediction delta。reference condition IDs、reference hash 和 DE mask 必须进入 metric manifest，test truth 不得参与 reference。
16. TxPert 其余指标面按冻结 registry 纳入：`pearson`、`pearson_sample`、`r2`、`r2_sample`、`mse`、`spearman_delta`、`pearson_delta_sample`、`r2_delta`、`r2_delta_sample`、`pearson_delta_de20`、`pearson_delta_de100`、`pearson_delta_kl0..3`、`normed_retrieval`、`fast_retrieval`。需要逐细胞配对、Pharos 分组、DE 或 retrieval reference 的指标若前提不满足，标记 unavailable，不删除条件或伪造数值。
17. TriShift 其余指标面按冻结输出纳入：`nmse`、`mse_pred`、`mse_ctrl`、`deg_mean_r2`、`systema_corr_deg_r2`、`systema_corr_20de_allpert_dist`、`systema_corr_deg_r2_dist`、四个 `scpram_r2_*` mean/variance 指标及 `scpram_wasserstein_degs_sum`。TriShift 的 `r2_all_*` legacy aliases 不重复作为独立科学指标。
18. 若某模型因图覆盖无法预测某个 test perturbation，不得静默删除。主报告显示 full-protocol coverage，同时另报 common-coverage subset。

### 11. 可复现产物与 notebook 合同

- 训练、评价和 notebook 分层：可复现逻辑必须位于包内 CLI/模块；notebook 只读取冻结 artifact 做表格、统计和画图，不隐藏数据切分、训练或指标实现。
- 每个 `dataset/protocol/split_hash/model_id/run_seed` 目录至少保存：原始与 resolved config、环境与代码 provenance、data/preprocessing/gene-order/split/control-access/evaluation hashes、checkpoint、逐 epoch train/validation history、最佳 checkpoint 选择记录、test prediction payload、逐 condition metrics、macro summary 和 metric availability。
- 主 prediction payload 采用版本化、模型中立的 PKL schema，学习 TriShift 的 condition-keyed 结构。每个 condition 至少包含 `Pred[300,G_de]`、`InputCtrl[300,G_de]`、`Truth[N_p,G_de]`、相应 full-gene arrays、`MetricCtrlPoolMean[G]`、`DE_idx`、`DE_name`、`gene_name_full`、全部 sample/row IDs、`export_metadata` 和 `full_summary`；PKL 文件名不带上游模型类名。用 `InputCtrl` 与 `MetricCtrlPoolMean` 分名，禁止继续用含义不清的单一 `Ctrl` 字段。
- `Truth` 仅由锁定后的 evaluator 在 validation/test artifact 阶段合并，训练 runner 不能读取 test payload。下游 notebook 可以读取完整 payload，但必须显示 split、model、run seed 和 artifact hash。
- 同时输出机器友好的 `metrics_by_condition.parquet`、`metrics_summary.json`、`metric_availability.json`、`run_manifest.json` 和必要的 H5AD。PKL 用于 NumPy/condition payload 兼容，Parquet/JSON 用于安全查询和跨语言消费；不得只保存一个不可审计的 pickle。
- 必须提供从 prediction PKL 独立重算全部适用指标的命令，并用测试验证重算结果与已发布 metrics artifact 一致。notebook 统一通过 result adapter 按 dataset/model/split/seed 定位产物，不拼接硬编码绝对路径。
- 发布 notebook 时同时保存源 `.ipynb`、无输出或清理版、执行日志和依赖的 artifact manifest；图表必须能追溯到具体 prediction/metric hash。

### 12. Nonlearned baseline 精确定义

- `matched_control_mean`：仅使用协议允许的目标 cell line/batch control 均值，是图中 `Batch only` 的实现。
- `global_train_delta`：在 matched control 上加所有训练扰动、按 batch-matched control 居中后的全局平均 delta。
- `additive_seen_singles`：只用于 Norman `combo_seen2`，预测 `control_mean + delta(gene_a) + delta(gene_b)`；两个 delta 均只能从 train singles 计算。
- TxPert 的 general baseline 对未见 single 会退化为 `global_train_delta`，因此在四个严格 unseen-single test 上不再重复列一个数值等价的 baseline。
- nearest-cell-line 与按 cell-line 样本数加权的 baseline 只适用于 cross-cell protocol；MVP 不做 cross-cell，因此延期。
- split-half reproducibility 会读取 test labels，只能作为实验可重复性参照，不能作为可部署预测 baseline，也不能参与模型选择。

### 13. `AGENTS.md` 规则草案

下一阶段在仓库根目录创建标准文件 `AGENTS.md`，不是小写 `agent.md`。至少写入以下强制规则：

- 原生包、类、配置和公开 API 统一使用 `gradpert` / `GraDPert...` 命名；不得创建带 `TxPert`、`DINO`、`GEARS` 名称的原生模型类。
- `src/gradpert` 中禁止 `import txpert`、`import gspp`、`import gears`、导入 DINO 上游模块、调用其 CLI 或依赖上游 checkout。
- 每个核心行为实现前必须检查冻结官方 commit 的具体文件、函数、张量形状与更新顺序；不知道时标记 blocked，不得根据论文术语或聊天记忆猜实现。
- 使用参考实现引导的独立重写，不逐文件复制或通过改名伪装上游源码；许可证、NOTICE、论文 Related Work 和内部 provenance registry 必须保留真实来源。
- 对外描述 GraD-Pert 时使用自身问题定义、模块名称和机制，不把它写成某上游项目的 wrapper、fork 或改名版；但不得用命名规则掩盖应有的学术引用和实现溯源。
- GEARS 与 TxPert 作为外部 benchmark 只能在隔离环境运行。核心包通过中立 prediction artifact 接入，不直接 import；benchmark manifest 必须保留真实 `model_id`，不能把外部结果伪装成 GraD-Pert 输出。
- 全部模型必须读取同一 split hash、gene-order hash、preprocessing hash 和 control-access manifest；任何 runner 不得自行重切数据。
- 数值 seed 相同不能替代条件集合核验；运行前必须比较实际 train/validation/test condition IDs 与共享 manifest，任何差异立即失败。
- 训练/评价代码负责生成版本化 artifact；notebook 只消费 artifact，不得包含唯一的数据处理、split、评价或模型逻辑。
- 禁止使用 test 数据调参、早停、选 checkpoint、构建 response-derived graph 或估计 baseline delta。
- 禁止在结果未产生时声称 GraD-Pert 优于任何模型；不适用或失败的指标必须如实标记。

## 未明确或高风险点

1. TxPert 论文与当前 onboarding split 函数默认比例不一致；必须先核对发布 split artifact，不能假定任一默认值就是论文实际运行划分。
2. TxPert 本地官方仓库已冻结在 commit `08d82eea86746b044cf7531f4ec8c5f60e1cb73f`；DINO/DINOv2、GEARS、STRING、GO 的版本/commit/checksum/许可证仍需在实现前冻结。
3. 五个数据集仍未下载；数据来源、许可、表达尺度和官方 split 文件是否齐全必须由下载 registry 的真实审计确认。
4. GEARS 官方说明它不支持跨 cell-type transfer；MVP 仅对五个数据集逐个独立训练 GEARS，不把它扩展成 cross-cell 模型。
5. 外部 learned baselines 与“核心包不得调用上游”存在表面冲突，必须用隔离运行环境 + 中立 prediction artifact 边界解决，不能把 upstream import 带入 `src/gradpert`。
6. 五数据集全量实验可能需要较长下载和 GPU 周期；执行计划必须定义按数据集的 readiness/训练/评估门禁和可恢复 checkpoint。
7. TxPert macro Pearson delta、TriShift Pearson delta 和 Systema Pearson 使用不同 gene mask、control/reference 与语义；若只按 `pearson` 显示名合并，会产生错误排行榜，必须用语义 ID、公式、来源 commit 和 applicability 区分。
8. Pickle 方便复用但不适合不可信输入且可能体积很大；只加载本项目 checksum 验证通过的 PKL，并同时保留 Parquet/JSON/H5AD 的可检查产物。
9. 若把 Scouter 的 condition-level `n_pred=300` 误实现为“每个真实细胞各采 300 个 control”，会把计算规模扩成 `N_truth × 300` 并制造不存在的逐细胞配对；shape contract 和 golden test 必须阻止这种实现。

## 必须确认的问题

当前没有阻塞进入计划阶段的问题；唯一训练路线已暂定为 B2。若后续改回 B3，必须作为新的明确设计决定更新计划，不能在实现中同时保留两套路线。

## 可以稍后确认的问题

- 技术栈和调度器不再阻塞需求确认；默认采用 Python 3.12 + PyTorch 2.6 + PyG/Lightning + Hydra/OmegaConf + uv，并先支持单机多 GPU，若服务器实际提供 Slurm 再启用 launcher。
- 默认把 MVP 限定为四个 within-cell unseen-single 协议 + Norman `combo_seen2`，暂不做 cross-cell leave-one-out。

## 建议默认值

- `/Users/elan/code/grad-pert` 作为唯一正式仓库；将旧设计文档复制进 `docs/design/` 并记录来源，不在两个位置继续双写。
- 第一里程碑完成可安装包、合成数据端到端 smoke、五数据集 registry/downloader/QC gate、唯一 B2 模型和统一 evaluator；随后继续完成五数据集的 GraD-Pert、GEARS、TxPert 与 nonlearned 正式运行。
- 核心依赖保持最小；图、训练调度、外部评估模型使用 extras，避免复制 TxPert 的整套环境。
- 使用 Hydra 分层组合配置 + Pydantic/dataclass 强校验；每次运行保存 resolved config 和 hash。
- 先支持单机多 GPU；Slurm launcher 作为可选扩展。
- smoke 配置使用小 prototype 容量与微型图；正式 `K_head` 由服务器完整训练 step 的 85% 显存门槛选择并全实验锁定。

## 非目标

- 不恢复 E1–E5、DEG 节点路线、Top-100 Teacher/Top-20 Student 或 B1。
- 不恢复已经删除的 local 随机删边与边重建辅助任务；v1 使用 global node-iBOT 和 local anchor mask。
- 第一阶段不实现或运行任何消融矩阵，不实现或比较 B3、其他 backbone、其他 local、其他 decoder、其他损失权重或 temperature。
- 第一阶段不做 cross-cell leave-one-out；nearest-cell-line baseline 延期。
- 不把 Adamson、Dixit、PBMC 加入 v1 五数据集主结果。
- 不在正式包中 vendor、import 或调用 TxPert/DINO/GEARS。
- 不在数据未通过 `canonical_ready` 时启动正式训练。
- 不把尚未运行的模型或消融写成性能最优结论。
- 第一阶段只报告锁定的传导式 unseen-response 协议，不同时引入 target-inductive 分支。

## 风险

- 图结构或 per-gene lookup 可能形成扰动 ID 捷径，必须依赖 node/anchor mask 与结构负对照共同审计。
- 65,536 prototype head、两个全图 global 和八个 local 可能成为显存瓶颈。
- 数据源的“processed”命名可能掩盖表达尺度、HVG、split 或 gene ordering 不一致。
- B2 从随机初始化同时接收预测与 SSL 梯度，存在早期目标竞争或训练不稳定风险；只能通过训练日志、梯度/损失健康检查和验证集早停监控，不能据此临时切换到 B3。
- 若后续代理重新启用旧消融面，会违背本次 MVP 收敛；`AGENTS.md` 和 experiment schema 都应阻止这一点。
- 复制上游依赖或实现细节会带来许可、复现和维护风险，因此必须先冻结 provenance 再独立实现。

## 推荐下一步

需求讨论已足够进入 `byte-plan`。下一步按唯一 B2 路线，把“官方实现审计与冻结 → `AGENTS.md`/设计文档迁移 → canonical split 与公平评价合同 → 唯一 GraD-Pert 模型 → 三个 nonlearned baseline → 隔离 GEARS/TxPert → 五数据集正式训练评估”拆成可恢复、可验证的执行阶段；计划确认后再使用 `$byte-build` 写代码。
