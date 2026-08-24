# Decision Log

## 2026-08-25

- GraD-Pert full-run 上限由 200 改为 100 epochs，仍采用 validation-only early stopping `patience=10`。
- GraD-Pert 五数据集的 train/evaluation batch size 统一由 64 改为用户锁定的 256；必须经过服务器真实显存 smoke，不能静默回退。
- 训练性能优化必须保持 B2 损失、view、梯度归属、optimizer→Teacher EMA→center 顺序及公平评估协议不变。允许批处理互不相连的 graph views，并复用实际 prediction/SSL 反向梯度计算诊断指标。
- 启动长时间训练后暂停持续目标执行，改由定时任务检查进程、receipt 与失败证据，不进行前台 busy polling。
- batch-256 首轮证据显示 allocated 约 18--22GB、reserved 约 27--29GB，失败主因是 allocator 碎片/缓存而非真实张量 OOM；原生 CUDA 进程固定使用 `PYTORCH_ALLOC_CONF=expandable_segments:True` 并强校验。
- 性能优先后 prototype 容量选择上限保持 16,384；即使 allocator 节省使 32K/65K 可装入，也不自动升级为更慢的 head。

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
- Learned models 原始约定最多训练 200 epochs；该上限已由 2026-08-25 决策改为 GraD-Pert 100 epochs，早停和测试门禁不变。
- 学习率、optimizer、weight decay、batch size 和模型数值配置先逐模型、逐数据集沿用冻结官方代码；官方若有数据集差异则保留差异，未公开值必须标记为 `project_preregistered`，不能冒充官方参数。
- 已核对 TxPert 当前公开配置：所有发布 YAML 的 batch size 都是 64，训练模块默认 AdamW、LR `1e-3`、weight decay `0`，但仓库没有五数据集的完整训练配置。已核对 GEARS 冻结 commit：官方示例 train/test batch 为 32/128，训练默认 Adam、LR `1e-3`、weight decay `5e-4`；未发布五数据集差异。
- 配置采用 `configs/experiments/<model_id>/<dataset_id>.yaml` 模型×数据集矩阵；每份文件自包含全部有效值。禁止单一总配置、隐藏 defaults 链和跨文件继承；共享内容仅限 Python schema/validator。
- Active v1 数值规格冻结在 `docs/design/GRADPERT_V1.md`：共享 128 维基因 embedding、每图独立四层双头 GATv2、64 维输出、节点自适应双图融合、TxPert 数值起点的 basal/decoder，以及严格的 optimizer→Teacher EMA→center 更新顺序。
- 四个 learned-model run seeds 固定为 `[1,2,3,4]`，split seed 为 42，evaluation seed 为 20260824；每个随机流仍使用独立命名和派生种子。
- Runner artifact 不携带 truth；公共 evaluator 在 prediction 封存后才连接完整 truth 形成 EvaluationBundle。
