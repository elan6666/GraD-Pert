# GraD-Pert 完整训练更新提速

本轮从已发布 `fb2adf48747abb56bf85965c37b790338103271e` 开始，目标是
减少相同训练工作量的端到端耗时。借鉴 [Databricks 的测量与专用内核思路](https://www.databricks.com/blog/reducing-qwen-training-time-6x-ai-runtime)，
先定位完整更新的计算和发射开销，再验收候选。其 Qwen 的提速数字不是本项目的预期收益。

使用既有 Jurkat TRAIN40 的 R3-RI 配方；单卡工程对比明确改为
micro80 × accumulation4 × world1 = global320。该单卡对比不冒充历史双卡性能，
不读取 test 调参，不新增 DEG 或科学训练，不启动旧研究循环。
旧的失败候选、配置、训练源、报告和结果不覆盖。

`scripts/v2/update_parity.py` 支持 torchrun 单 rank 或双 rank。单卡仍初始化
NCCL process group，走原 collective、完整损失、梯度、optimizer、Teacher/center
和 RNG 比较路径；单卡显存配额为 90%。收据分别记录 world size 和物理 GPU。
两个配置的 world size 必须与实际 worker 数一致，不能拿不同拓扑比较数值。

单卡示例（路径均替换为该次封存输入）：

```sh
PYTORCH_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONPATH=src:. python -m torch.distributed.run --standalone --nproc_per_node 1 \
  scripts/v2/update_parity.py --gpu 1 \
  --config /data/yilangliu/FAMILY/baseline/gradpert_v2/nadig_jurkat.yaml \
  --candidate-config /data/yilangliu/FAMILY/candidate/gradpert_v2/nadig_jurkat.yaml \
  --data-root /data/yilangliu/GraD-Pert/data-vnext-a942114 \
  --output /data/yilangliu/FAMILY/parity-a1 \
  --publication /data/yilangliu/FAMILY/PUBLICATION.json \
  --publication-sha256 EXACT_SHA256 --deterministic --diagnostic-learning-rate 0.001
```

该命令必须由持有对应物理卡非阻塞锁的 controller 执行；在启动及存活期间使用
原 shared-gpu-admission-3 的 selected-device 准入，保留 UUID、外部进程限制、
90% 自有配额、2GiB 空闲及 owner/后代身份检查。另一张卡的无关任务不参与本卡分配。

Profiler、状态/梯度复制和恢复验证不计作速度证据。数值比较通过后，另用不插桩
完整更新做相同 batch/view/RNG 的配对计时，分别保留启动、预热、数据等待和训练耗时。
数学/训练源码没有因单卡诊断入口而改变；诊断工具支持不代表候选已通过或已提速。

永久收据、中文报告及 aggregate 位于
`experiment-analysis/training-speed-20261106/`；原始 trace、矩阵和权重留服务器。
