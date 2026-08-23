# Decision Log

## 2026-08-24

- MVP 只实现一个 GraD-Pert 版本：B2 从随机初始化联合优化表达预测与图自蒸馏；不实现 B3 和消融矩阵。
- 主数据集固定为五个：Replogle K562 essential、Replogle RPE1 essential、Nadig Jurkat、Nadig HepG2、Norman。
- learned baselines 固定为 GEARS 与 TxPert，均在隔离环境运行；原生 `gradpert` 包不 import 或调用上游模型包。
- nonlearned baselines 包含 matched-control mean、global train delta 和 Norman-only additive seen singles。
- 每个数据集只允许一份 canonical condition split；模型加载后的实际 train/validation/test condition IDs 和 hash 必须一致，不能仅依赖相同 seed。
- 推理采用 Scouter-style population protocol：每个扰动条件从合法 control pool 有放回抽取 300 个 control，生成 300 个预测细胞；truth 保留该条件全部真实测试细胞。
- Pearson 主面板分开报告 TxPert macro Pearson delta、TriShift Pearson delta 与 Systema Pearson。
- 产物使用版本化中立 schema，支持 PKL、Parquet、JSON、必要 H5AD、独立指标重算和 notebook 消费。
- 实现前必须检查冻结官方代码的实际文件、函数、张量形状和更新顺序；不能凭名称或记忆猜实现。
- 原生包完全采用 GraD-Pert 命名与独立实现；上游名称只允许出现在隔离 baseline、引用、许可证和私有实现溯源中，不能成为原生类名、checkpoint key 或运行时依赖。
- 正式数据准备、图构建、GPU fit-test、训练、推理和指标物化只在服务器执行；本地仅做开发、单元测试和合成 smoke。
- 代码唯一远端为 `https://github.com/elan6666/GraD-Pert.git`；每个服务器任务必须证明本地、GitHub、服务器三端为同一 commit，且服务器 worktree 干净。
- 数据集、H5AD、prediction PKL、checkpoint、逐细胞矩阵等大产物只留服务器；回传本地采用小型结果白名单，并保存服务器路径与 checksum 指针。
- Learned models 默认最多训练 200 epochs 并启用早停：每 epoch 检查 `val/txpert_macro_pearson_delta`，`mode=max`、`patience=10`、`min_delta=0`；测试只在 checkpoint 锁定后运行一次。
- 学习率、optimizer、weight decay、batch size 和模型数值配置先逐模型、逐数据集沿用冻结官方代码；官方若有数据集差异则保留差异，未公开值必须标记为 `project_preregistered`，不能冒充官方参数。
- 已核对 TxPert 当前公开配置：所有发布 YAML 的 batch size 都是 64，训练模块默认 AdamW、LR `1e-3`、weight decay `0`，但仓库没有五数据集的完整训练配置。已核对 GEARS 冻结 commit：官方示例 train/test batch 为 32/128，训练默认 Adam、LR `1e-3`、weight decay `5e-4`；未发布五数据集差异。
- 配置采用 `configs/experiments/<model_id>/<dataset_id>.yaml` 模型×数据集矩阵；每份文件自包含全部有效值。禁止单一总配置、隐藏 defaults 链和跨文件继承；共享内容仅限 Python schema/validator。
- Active v1 数值规格冻结在 `docs/design/GRADPERT_V1.md`：共享 128 维基因 embedding、每图独立四层双头 GATv2、64 维输出、节点自适应双图融合、TxPert 数值起点的 basal/decoder，以及严格的 optimizer→Teacher EMA→center 更新顺序。
- 四个 learned-model run seeds 固定为 `[1,2,3,4]`，split seed 为 42，evaluation seed 为 20260824；每个随机流仍使用独立命名和派生种子。
- Runner artifact 不携带 truth；公共 evaluator 在 prediction 封存后才连接完整 truth 形成 EvaluationBundle。
