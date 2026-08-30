# Decision Log

## 2026-08-31

- Replace the successor A0 scientific coordinate from eight to four
  RingInduced local views. Advance the matrix identity to
  `nadig_jurkat_vnext_ratio_graph_v3`; old hashes, receipts and interrupted
  runs remain immutable eight-local evidence and cannot satisfy the new A0.
- Preserve single-variable L design by changing L2 from four to eight locals.
  L1 remains Fanout at four locals, L3 remains quarter local-node coverage at
  four locals, and L4/L5 retain proportional masks that resolve to `2/4` and
  `1/4`. Every H/M/W/D/E/O row inherits the four-local A0.
- Stop all active experiments and pause L execution. After a fresh measured
  performance pass, run only A0/H1/H2/H3 sequentially on one physical GPU.
- Treat the accepted eight-local sparse-union ABBA result as historical
  implementation evidence, not current-coordinate timing. Reprofile the new
  four-local A0 before selecting another optimization; GPU utilization is a
  diagnostic, while exact state, step wall, tails, memory and zero OOM remain
  acceptance gates.
- Use Hugging Face Trackio only as a separate formal-run telemetry sidecar.
  Mirror an explicit scalar allowlist for loss/validation/performance curves,
  manually sample the one selected GPU, keep the 64-step native log buffer,
  and never upload test metrics, predictions, row/gene identities, data or
  checkpoints. Because Trackio step replay is not idempotent, every live run
  uses a fresh locked lineage with `resume="never"` and remains provisional;
  native receipts are authoritative.

## 2026-08-30

- Accept the ordered CPU-vectorized sparse-union implementation after exact
  deterministic CUDA equality and serial same-GPU ABBA. The paired median
  optimized/reference ratio is 0.437470 (56.253% reduction, 2.286x), both p90
  comparisons improve, and peak allocated/reserved GPU memory rises only
  0.017/0.041%. This is implementation evidence only; formal H/L remains a
  separate ten-epoch scientific lineage.

- Use an exact eight-row performance sentinel rather than running bounded
  performance census stages for all 25 scientific rows: A0, H3, L1, L2, M4,
  W1, D2 and E2. The source matrix remains 25 rows, and the sentinel cannot
  satisfy scientific completion.
- Use M4 rather than M1 because M1 and W1 overlap on the single-source
  STRING-GAT family. Use L2 rather than an objective-weight row because the
  current native step computes auxiliary loss tensors before applying their
  weights, while L2 exercises the distinct four-local implementation path.
- Run one complete P1 capacity step for all eight. Use A0 for profiler-led
  optimization selection and add short timing only where it improves a
  measured decision or regression gate. Formal A0/H/L remains fixed at 10
  epochs after exact-effect and ABBA performance acceptance.
- On Linux, resource preflight uses `/proc/meminfo` `MemAvailable`; free-page
  counts are fallback-only because immutable-input hashing can move hundreds
  of GiB into reclaimable page cache.

## 2026-08-29

- Performance optimization compares reference and optimized implementations of
  the same frozen A0 scientific coordinate. GAT versus Exphormer-MG,
  RingInduced versus Fanout, different budgets, fewer views, AMP/TF32 or
  changed validation schedules are scientific changes and cannot support an
  unchanged-effect claim.
- The first implementation target is the measured Fanout candidate-index
  rebuild: precompute the exact immutable source-ordered mapping once per
  engine and preserve every candidate order, PCG64 draw, view edge/order/mask
  and warning byte exactly.
- Do not concatenate or reorder training graph forwards. Exphormer-MG
  BatchNorm running statistics and dropout RNG are per-view scientific state.
  Sparse-union optimization may share only immutable topology tensors after
  exact union/output/gradient/state gates pass.
- Use real monotonic timing windows and serial single-physical-GPU ABBA
  replicates. Concurrent two-GPU runs sharing host resources and the existing
  overlapping serial stage sum are not primary speed evidence.

## 2026-08-26

- The user superseded the earlier 100-epoch ceiling and B3 speed-only pilot
  selection for default execution: use full-graph systems-only B2 by default,
  with `max_epochs=200`, validation-only early stopping `patience=10`, and
  `metrics_only` zero-PKL output. The subsequent "4 split" request is executed
  as the already registered run seeds 1--4 on the single frozen canonical
  Nadig Jurkat split, not four newly generated data splits. Preserve every
  completed B0/B1/B2/B3 coordinate.
- The user explicitly superseded the earlier no-rerun limitation for timing:
  execute one fresh B0 performance coordinate with `metrics_only` and zero
  persistent PKL so artifact persistence cannot contaminate its timing. Keep
  the historical c240 B0 immutable and separate; do not overwrite or relabel
  it.
- The fresh B0 timing coordinate is now the performance baseline: 2,951.487 s
  and 43.46 cells/s on the full graph with all seven systems groups disabled.
  It passed 44 strict checks and did not change the historical B0 manifest.
- Select B3 as the Nadig Jurkat performance-pilot winner on actual one-epoch
  training wall time: 507.718 s versus B0 2,951.487 s, B1 844.180 s, and B2
  718.681 s. Keep the frozen Top-5000 expression/output/evaluation axes while
  using the directly recomputed Top-500-HVG-plus-target graph and all seven
  systems groups.
- Interpret B0→B1 and B2→B3 as direct graph-axis comparisons without and with
  systems, respectively. Interpret B0→B2 and B1→B3 as direct systems
  comparisons on the full and reduced graph, respectively. B0→B3 is the
  combined context comparison.
- Use monotonic full-epoch training wall as the primary speed evidence. The
  original warmup-excluded receipt sum is retained, but for prefetch-enabled
  runs it is not actual wall throughput because background data-read time can
  overlap GPU step time and is then summed again.
- Do not infer effect equivalence from the three one-epoch metrics. A longer
  controlled run is required before claiming unchanged predictive effect.
- Keep the historical c240 B0 immutable. The separate, user-authorized
  metrics-only B0 timing coordinate must not be relabeled as that historical
  result.

## 2026-08-25

- Large result output defaults to `metrics_only`: keep the best checkpoint,
  complete inference recipe, ordered control/truth row IDs, hashes, and small
  metrics, but produce no persistent PKL. When explicitly enabled, emit one
  `result.pkl` only, deduplicating selected control expression into a shared
  pool while preserving each condition's exact ordered 300 indices/IDs.
- The zero-PKL rule covers runner/framework metadata as well as result arrays.
  Successful GEARS runs retain only `model.pt`; the frozen official
  `config.pkl` and adapter `custom_split.pkl` are hashed for receipts and
  removed because their contents are reconstructable from the self-contained
  config, canonical split, and frozen official graph preparation.
- GraD-Pert full-run 上限由 200 改为 100 epochs，仍采用 validation-only early stopping `patience=10`。
- GraD-Pert 五数据集的 train/evaluation batch size 统一由 64 改为用户锁定的 256；必须经过服务器真实显存 smoke，不能静默回退。
- 训练性能优化必须保持 B2 损失、view、梯度归属、optimizer→Teacher EMA→center 顺序及公平评估协议不变。允许批处理互不相连的 graph views，并复用实际 prediction/SSL 反向梯度计算诊断指标。
- 启动长时间训练后暂停持续目标执行，改由定时任务检查进程、receipt 与失败证据，不进行前台 busy polling。
- batch-256 首轮证据显示 allocated 约 18--22GB、reserved 约 27--29GB，失败主因是 allocator 碎片/缓存而非真实张量 OOM；原生 CUDA 进程固定使用 `PYTORCH_ALLOC_CONF=expandable_segments:True` 并强校验。
- 性能优先后 prototype 容量选择上限保持 16,384；即使 allocator 节省使 32K/65K 可装入，也不自动升级为更慢的 head。
- 最终 batch size 不预设：在同一实现和 allocator 契约下分别对 64、256 执行 128 个真实 step，按容量、steps/s、cells/s 与估算 epoch 时间选择；256 不稳定或收益不足时正式回到 64。
- 实测结论选择 batch 256：五数据集均通过容量门禁，最小估算 epoch 加速 3.223×，最小阈值显存余量 24.55%，cell throughput 为 batch 64 的 3.84--4.04×；选择不使用验证/测试指标。

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
