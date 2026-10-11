# BR完成后的双卡batch容量测试

用户要求当前BR结束后尝试最大双卡batch，给其他任务每卡至少2GiB物理空闲。现有owned总显存<=85%、固定GPU UUID和UID1003/精确ScButterfly规则继续生效，因此测的是该资源边界内的最高十步验证batch，不声称独占硬件绝对上限或长期稳定容量。

原BR科学源b37b171fd68cb29e5c8142c66778aac6da9a6f06，四轮/1000步训练真实完成；last SHA f679adf48082e4e334eb18601e487fb4a316ebc6e80365934e2c414c621fb9d2。独立holder4c8a7ff保持原5秒资源守卫、仅暂停CPU派发器，等待BR真实exit0/TRAIN_COMPLETE/最后checkpoint再退休CPU派发；F1没有启动，GM1没有预检。BR测试18Pearson仍待补，训练完成不代表科学完成。

容量候选复制当前BR完整配置，只有microbatch与train_batch_size及对应试验出处条目变化。world2、accum2、完整预测+双SSL、四prototype16384、原GenePT/图/Cell/Response/FFN、视图、随机顺序、loss聚合、teacher/center、数据划分保持原配置。每档fresh工程运行ID；不加载BR权重，不选择科学模型或用测试指标挑参数。

每档10个完整优化更新，2个warmup；非零LR、多步optimizer/EMA/center与checkpoint恢复，单条件冻结300control推理。先48同版本锚点→56→64→80→96→112→128→160→192→224→256，首个真实OOM后在已过/未过间按4-grid收窄。来源/配置/数据/publication精确绑定；非OOM错误、身份漂移或资源守卫失败停止队列，不能当OOM边界。256若通过只报告上界未找到，回主扩展，不能称物理最大。每个失败保留日志/收据及失败步数。

报告最高通过micro/global及首失败边界，同时单列细胞/秒最快配置、含等待步时、启动成本、双rank峰值allocated/reserved、每卡最小空闲与守卫结果。更大batch不预设更快。工程收据不算科学评估；本批BR/F1/N1/GM1仍batch192，不因容量测试自动改训练协议。

启动/终态/数字由不可变运行收据补充。接续：容量测试→BR epoch4 last18指标→F1/N1及GM1原授权流程。活动server源码、原source/config/checkpoints/runIDs不覆盖。

## Actual launch acceptance, search still running

Source `e3c9183dc24e6413efc75a3df29bfe564def826f`; queue `v2-br-capacity-e3c9183-20261011T202754Z`; queueSHA `6f2f37148f9cba8b4377f220f6cd14ddac3b4b6b545ca10174a41e663f3fd7b6`; runtimeSHA `534adb0e0ffa71eaed60ab15d10c3e44774b7e23bb1497cd0a72736f1f278b91`; publicationSHA `7ae9665127405ce6d88fff92ef6fae2d55941dad9ac3be4e5ab97b4f493b14f3`. BR training source remains b37b171fd68cb29e5c8142c66778aac6da9a6f06.

| Anchor | micro/GPU | accum | global | updates | training-wall cells/s | measured cells/s | peak allocated/reserved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m48 |48|2|192|10/10|8.063|8.257|13.42/13.65GiB|

Checkpoint restore/hash, nonzeroLR, finite gradients,300x5000 single-condition inference passed. Active-rank telemetry free approximately16.9GiB/each, owned approximately14.44GiB/each; snapshots are not full-lifecycle peaks. Controller automatically advanced m56/global224. Same free>=2048MiB and aggregate85% policy. Native m48receipt SHA `9d3e80cf7f2b2876e3507924a8ec58b71b5d4a08087093235b354bf56efb6549`; server main acceptanceSHA `9987a7283ff12d4f834e49b9a7f545bdf12d711c76b5d0499d801ebb276e3756`.20s deployment-confirmation failure retained separately; live matching controller and10-update pass reconciled, never relaunched.

Local/server29 CPU tests and review passed. Earlier22d inventedruntime.source_root fixed to actualrepository_root with deployed-shape regression; failedsource/runtime/publication preserved. Remaining: complete capacity/acceptance -> BR epoch4last18 metrics -> F1/N1/GM1 originalbatch192.
