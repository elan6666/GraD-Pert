# 后续新训练的执行默认项

2026-11-08用户明确要求将已经测过的吞吐优化用于以后的训练。
适用范围是原生v2 `relay_full`训练；外部基线和不支持的注意力路径不套用。

所有新训练配方显式写入：

```yaml
relay_validate_once:
  value: true
  source: user_locked
  reference: docs/experiments/TRAINING_EXECUTION_DEFAULTS.md
cache_kda_constants:
  value: true
  source: user_locked
  reference: docs/experiments/TRAINING_EXECUTION_DEFAULTS.md
```

生成新实验时使用`gradpert.config.training_defaults.with_training_throughput_defaults`，
或运行`PYTHONPATH=src python scripts/v2/prepare_training_config.py --input <母本> --output <新配置>`。
工具只创建新文件，不覆写旧配置。图归一化两组生成器已调用同一函数，两组同时开启。
将来专门比较执行开关时，另立明确命名的对照配置；不暗改历史运行。

这两个开关分别复用合法图邻域的校验结果、缓存运行所需KDA常量。
它们不会替代进程级资源准入、GPU锁、所有权核验、CUDA allocator或CUBLAS合同。
旧配置加载、旧检查点payload和历史实验保持原语义，不在加载时补入隐式默认。

已完成的Jurkat R3-RI完整2轮对比中，吞吐约提高10.3918%、总训练耗时约降低9.4135%。
该单seed实验的最终权重未保持逐位一致；部分Pearson上升而Systema和MSE略退化。
因此该设置是用户选择的后续执行默认，不代表已证明数值等价或预测质量全面提高。
永久原始证据位于`experiment-analysis/training-speed-20261106/full-2epoch-a1/REPORT_CN.md`
及其`pearson-val-a1/REPORT_CN.md`，训练commit为
`1831d7d3c8617030b430b0f6827ed31a99758963`。
