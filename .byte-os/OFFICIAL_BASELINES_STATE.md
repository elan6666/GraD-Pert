## 2026-10-13：官方 Scouter/scDFM/GEARS 表达剔除适配验收与依赖队列

三组官方适配/config完成，140本地测试通过、1旧官方checkout测试跳过，Ruff通过；服务器真实官方包均10次优化/恢复预测一致/零PKL。同源2241c46ed3d320b3e194a1f7ba989637b4587a24两数值环境的真实数据合同一致：cap40训练47836/验证41235/control12013，4775可见表达/225剔除表达，excluded非零数0；仅官方训练数据对象，不含test扰动细胞。冻结300controls与DEG/Systema参考完全匹配历史BR。

后续官方queue已实际启动CPU等待：official-expression-queue-2241c46-20261013，SHA049e6fcf8d8277c19fb8c1bddb7f1cb9fa52b6c7e9ffdfce8764a16aff186bd0；仅父F1→N1→GM1真终态成功才GPU10-update预检，再scDFM→GEARS官方训练+18项last评估。Scouter47aliases已审核，唯一缺CCDC130/YJU2B官方向量，正式运行blocked，不删除条件或补零；用户额外先验授权待答。没有新的GPU训练/科学结果，当前原生队列与源码不改。现有监督会话保持唯一20min监控，下一阶段handoff已接纳，详情独立official-baselines-build.json及报告。下面为历史状态。


Main build scope independent of old native supervision; published code source2241c46.
Full runtime budget: official Scouter up to40epochs/patience5, scDFM200000updates, GEARS20epochs.
Native v1/model code unchanged; current GPU jobs preserved.
Report:docs/experiments/GRADPERT_V2_OFFICIAL_EXPRESSION_BASELINES_20261013.md
