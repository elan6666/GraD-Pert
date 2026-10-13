# GEARS十步预检的数据准备修复

用户询问耗时并要求预检十步，核实max_updates=10本来就是现协议。实际26/1781是CPU条件构图，非优化器步数；多数条件48sec且有长停顿，尚未进入GPU优化器。原预检已耗1h51，照现速率数据准备可能20h以上。scDFM保持原worker继续2400/3988，过去104min推进800updates，剩余训练约3.4h、评估另计。主已从监督接回，monitor PAUSED，无进程变化。

预改动baseline3a3e98987f47176f028d380059f1ad6799f94e33 clean且GitHub main一致。冻结官方GEARS f374e43e197b295016d80395d7a54ddb81cc6769 create_cell_graph_dataset已逐行核实；它只读control/target AnnData，采样器uniform replacement保持。根因候选为适配器逐cell重复Pandas控制池验证及整池/target深拷贝；原排序已完成并写成原preflight/fit/official_data/expression_holdout/perturb_processed.h5ad。

新适配只缓存同ctx同候选序列的控制池，官方采样/graph builder仍逐row调用，顺序不变。验证graph张量、配对audit、np/torch RNG、原数据不变。缓存官方排序须cache/parent RUN/data receipt/config/env/method SHA匹配，obs/var与sanitize X严格相等且所有排序/非零索引仅allowed基因；允许H5AD字符串categorical编码差异，禁止按是否非零推断allowed轴。无新排名/随机/标签规则。

阶段验收：本地synthetic/contract及服务器CPU检查；真实小队列原/新官方graph张量与RNG一致且实测耗时；同源官方CPU十步/恢复/zeroPKL。之后只终止旧GEARS预处理和CPU调度器，scDFM原PID/tick/模型源码不变。新GEARS独立run源码SHA/原配置/数据协议/四轮/last18指标，先十步真实GPU预检。混合source队列逐row核验科学源码，不伪称scDFM使用新源码；native依赖新两模型实际终态，并仍需主验收native预检。旧日志/run/cache/源码全部保存。不放宽GPU90%/2GiB/foreign1GiB门，不重跑BR/Scouter，不改活动源码。

预处理parity probe含加载/校验时间单列，sample8 cells/2control draws；它不是GPU吞吐、全数据长期稳定性或科学结果。Luna监督短探针，主不重复查。新实际预检启动后再交唯一长监督20min，失败保存证据交主。
