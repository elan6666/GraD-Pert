# 每个消融完成后的自动训练诊断分析

用户要求：每组真实训练及last测试结束后，不只汇报loss/Pearson，自动给出训练表征、原型分布、输出及梯度分析。当前五臂已保存初始化/每轮的固定训练探针，无需额外GPU推理。

## 执行和输出

独立发布的 `scripts/v2/analyze_completed_ablation.py` 仅读取完成运行的JSON，检查训练/测试/最后checkpoint身份与哈希、所有epoch、固定input/view哈希和零PKL。还会读取checkpoint字节核对hash，但不反序列化、不加载模型。CPU绘图与报告输出在独立服务器目录，原科学运行不写入。

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/yilangliu/GraD-Pert/source/.venv/bin/python \
  /data/yilangliu/GraD-Pert/development/source-ablation-analysis-SHA/scripts/v2/analyze_completed_ablation.py \
  --run-root /data/yilangliu/GraD-Pert/runs-native-ffn-evalchunk-420d3a1/RUN_ID \
  --name DP --output /data/yilangliu/GraD-Pert/development/ablation-analysis-SHA/DP
```

输出中文report.md、4张PNG、summary.json、training/heads/pairs/features/gradients/outputs/metrics七个CSV及receipt.json。包含全部Student/Teacher逐层聚合统计、原型熵/JS/argmax占比、实际CE=H+KL、输出相对control-copy改善及幅度、共享图梯度范数/余弦、18个最终Pearson和有效条件数。analysis SHA与旧training/evaluation SHA分别记录；产物全部hash。现有成功receipt重复时核对其run/source/input/hash后跳过；不覆盖已存在输出。报告失败单独记录并返回修复，不把科学训练追认为失败或重训。

## 当前运行自动化

不hotpatch活动420d3a1源码或controller。唯一监督会话在现有20分钟检查发现某臂真正完成时，调用新发布脚本一次，核验报告receipt并通知主会话分析；自动controller继续下一已授权臂。DP在本轮主动补做，后续FN/FG/FB/PO同规则。20分钟是当前核验发现的上限周期，不声称即时进程事件已验证。

新消融配置应启用已有process_diagnostics；历史缺少探针不自动追加GPU实验或伪造数据。探针是rank0、4训练细胞、最多128特征行/前256维、相关视图的前缀；head分布与CE配对行不相同。不存在UMAP/CKA或未见基因表征分组时明确写缺失。低秩、集中argmax、负梯度余弦不能独立认定坍塌、无效辅助任务或生物学收益。测试结果不用于选择超参数。

## 验证

7个合同检查：只读且保留不可定义指标；拒绝partial完成、变更checkpoint、固定probe漂移、错误CE分解、错误测试角色、残留PKL。实际DP绘图/报告服务器执行仍以独立receipt为准。当前训练源码420d3a1保持不变，后处理代码单独发布。
