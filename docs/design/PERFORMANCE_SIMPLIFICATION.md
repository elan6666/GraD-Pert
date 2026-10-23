# 代码简化与性能优化（2026-10-23）

本轮以已发布 `02cb70ca05e0b8285a94cae2ef307f36f5261942` 为修改前版本。
当前训练和诊断保持各自冻结源码；本页记录独立工程改进，不重训基线。

## 规模与判断

按 Python/YAML 文件全文行数统计（含注释与空行），`src/gradpert` 有141个Python文件、35,801行；
`scripts/v2` 有103个Python文件、28,986行；`tests/v2` 有130个Python文件、28,444行；
`configs/v2` 有415份YAML、141,880行。测试和独立配置不是每步都会执行的代码。

复杂度主要集中在 `run_unified_ddp.py`（2,002行）、`cap40_posthoc.py`（1,707行）
和 `run_br400_group.py`（1,127行）。后续优先按训练执行、资源检查、收据验证、报告生成划分职责，
沿用现有公共模块；不为减少行数引入隐藏配置继承，也不合并语义不同的科学验收条件。

## 已完成的有界优化

`trace_costs.events` 原先在每个事件上复制剩余缓冲区；现在用游标推进，只有补读时才整理缓冲区。
仍保持流式读取、事件顺序和内容、Unicode空白处理、大小限制、错误类型及已有边界行为。
`trace_attribution` 的三次扫描和 `update_costs` 的五次扫描复用此解析器，无须增加分叉实现。

合成CPU测试：100,000事件、5,147,617字符、65,536字符读块，7次交替顺序。
中位解析耗时0.180826秒降到0.083503秒，吞吐比2.1655，耗时减少53.82%。
每次聚合输出与旧实现精确相同。该计时包含列表物化、不包含磁盘I/O和聚合计算，
不是服务器端到端诊断耗时或GPU训练吞吐结论。

复现：

```sh
python3 scripts/performance/benchmark_trace_parser.py --output /tmp/trace-parser-result.json
PYTHONPATH=src python -m pytest -q tests/v2/test_trace_costs.py tests/v2/test_trace_attribution.py tests/v2/test_update_costs.py
```

主会话在含PyTorch环境执行全部86项相关测试通过（含真实profiler事件测试）；
独立审查另外比较1,254组输入/分块组合，新旧输出或异常一致。
实现代理原环境缺少torch的一项测试已由主会话完整执行，未把跳过视为通过。
原始计时、环境、源码哈希、审查记录保存在永久实验分析目录的 `engineering-cleanup-20261023`。

## 训练性能的证据与优先级

以下是已有历史profile，来源 `17f2194bd03f0c1f4a9cae6ac031c756104c5618`，
不能当成当前R3-AUS的最新测量：

| 部分 | 已有观测 | 工程优先级 |
|---|---|---|
| KDA扫描 | rank0累计kernel时间57.91%–62.18% | 优先研究块内中间量、发射与反向生命周期 |
| 重计算 | SSL/joint累计kernel时间约18.3%/18.4% | 按存储成本选择重计算，先验证梯度与RNG |
| 原型投影、原型层、CE | SSL/joint合计不足1% | 暂不作为主要提速对象 |
| 编排/诊断工具 | 存在超大模块与重复包装 | 在运行边界外抽取明确职责，保持收据兼容 |

这些累计时间可重叠，不是可直接节省的整步比例。
既有原生Gram IO ABBA在 `772075cf7f8d59dd6a1653f8ee490e2fe9ae79a8`
记录14.24715→14.49583 cells/s（+1.745%）、peak allocated减少15.85%。
这是历史原生实测；诊断入口曾有更大增益，不外推为当前训练提升。

后续GPU候选必须在独立资源窗口验证完整更新的loss/梯度/optimizer/EMA/center/RNG与身份，
再做无profiler整步ABBA、显存和冷启动测量。没有整步收益则不采用；
不将CPU微基准、减少代码行数或单算子收益当成整体训练加速。
