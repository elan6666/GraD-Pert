# 下一批优先：v2 loss 消融

记录日期：2026-10-07。
用户指令：在六组设计交付后，用户要求“等下优先跑这些实验”。
最新指令：**“改为等现在这一个6epoch做完，立马做这个消融”**，替代上一版等待安排。
状态：**父队列已完成并独立验收；loss 原生代码已实现，权重1本地验证通过：39定向测试、540回归测试通过，11硬件/依赖skip、1父版本缺文件测试排除，尚未发布/启动。**

最新权重：M1 gene=1/CLS=0，M2 gene=1/CLS=1，mask20%；保留L0–L3与原SSL设置。
分工：main=`codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b`；executor=`codex:01a10391-22a3-78e1-9799-93ea9d6ffdd9`，reports_to/最终return=main；supervisor=`codex:01a0df0b-4142-7df1-86c0-d959471d80a1`。

父运行已独立核对：A2 完整6epoch、last-only测试和COMPLETE通过；验收记录 `.byte-os/coordination/receipts/loss-parent-acceptance-20261007.json`。
精确队列：`v2-functional-b0-ka-3d3f5ad-20261005T193340Z-881a1ef0`，attempt1。
当前 run：`nadig_jurkat-seed1-20261005T193538Z-f6364ab30a2d4b668590e2594c5c81bb`。
这是原队列最后一组；不等待任何其他未来实验，不中断它当前的6epoch。

## 范围与顺序

采用 [六组设计](../../docs/experiments/GRADPERT_V2_LOSS_ABLATIONS_20261007.md) 中的 L0、L1、L2、L3、M1、M2。
优先安排：**L0 → L1 → L2 → M2 → L3 → M1**。
前四组优先得到基线、误差形式、条件权重和完整双头辅助方案的对照；随后补齐交互与基因头拆解。
不增加多 seed、强度扩展组或其他数据集。

## 下一阶段

1. 核实已发布 B0 父版本，隔离无关本地修改；实现六组原生 loss、辅助路径、配置、日志和恢复合同。
2. 完成设计规定的公式、全局 reduction、SSL 不变性、mask 防泄漏、随机流隔离和恢复检查；发布干净新版本。
3. 当前 A2 按原协议完成6epoch及final last测试并验收后，立即进入本六组的实现/验证/发布/容量检查/启动，不再等待其他未来实验。只有前置门槛通过才能启动正式loss训练，不把尚未实现称为已启动。
4. 新家族保持拟定 Jurkat cap40、seed1、六完整 epoch、固定 batch272、无 validation、final last 测试；逐组落实新 run ID 和版本收据。
5. 六组正式训练前通过同版本/配置容量门槛；执行顺序按本记录。完整结果报告，不能按 test 调参。

## 边界

现有 `grad-pert-functional-b0-k-a-six` 监控已更新为每5分钟核验当前队列，终态返回目标改为本会话 `01a10391-22a3-78e1-9799-93ea9d6ffdd9`。
现有 `grad-pert-e23-twenty-return` 已重定向本会话并更新为loss续接任务；当前保持PAUSED，由supervisor终态返回时激活，接手后暂停以防重复执行。
没有创建重复监控；现有训练进程、源码和配置未改变。自动续接依赖当前每5分钟监控，不声称零延迟事件触发。
这些更新不是新loss训练启动收据，也不是模型已通过验收的证据。
未来实现/运行须遵循项目的发布一致性、服务器计算、容量、可恢复与 zero-PKL 合同。
