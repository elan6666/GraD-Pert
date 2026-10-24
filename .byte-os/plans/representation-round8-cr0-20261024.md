# R8：快速Teacher下，检验全局coding-rate项的必要性

沿用持续表征优化、2epoch/中等分析/20分钟监督的明确授权；旧消融暂停，不重训任何基线。编辑前本地clean与GitHub main均为3d61cb0b50f66535023bf276a7cd04f69adafc3b；历史R6训练源码8a78218424802f1fab01be2b8b134d120c91dd71与checkpoint610eb0c6058870e13eb226cfe037d1899af5841ec452fc2fbbf813eb6ae0ceab保留。

R7降低lambda1确实使SSL1有效剂量回到R5量级，但445条件同有序control/truth验证MSE=.006258086035934465，比R6高.5033%，14/18 Pearson低于R6、15/18低于R5。不能凭降低总loss或增秩宣称成功。R6完整320零更新单批的CR/MSE graph cosine=-.372404；移除CR的固定state总梯度cosine由.262946升.343493，但R5同样移除反而由.965333降.570374，R6 response分组也没有改善。保留这些反证：这是必要性消融，不保证任务收益，不把原1/8观察冒称完整八批。

唯一新臂R8-CR0，科学父臂**原R6-EMA99**，只改model.parameters.cr_weight 4e-5→0。lambda1=1，lambda2=1 user_locked，EMA恒定.99，A=.05，U=0，epsilon²=.05，warmup30，作用点仍现有SSL2 CLS pre-L2瓶颈；架构、RI残差、GELU/GenePT/Softplus、view/RNG/center/temperature/prototype、seed1/global320=80×双卡×累积2/2epoch全部不变。R7不是该臂单因素父臂，不能称R7→R8仅一项变化。CR仍保留可观测定义但训练关闭，不伪造缺失CR历史为0；全人口小探针缺失仍NA。研究比较不读test。

文献依据：Wang & Isola ICML2020 https://proceedings.mlr.press/v119/wang20k.html 区分alignment与uniformity；Yu等NeurIPS2020 https://proceedings.neurips.cc/paper/2020/hash/6ad4174eba19ecb5fed17411a34ff5e6-Abstract.html 的MCR²为whole与within-class差分，现有global negative CR并非完整MCR²。论文支持分析目标，不证明取消CR会改善细胞预测。完整实测证据保存在永久R7/NEXT_RESEARCH_EVIDENCE.md。

实现所有权：研究实现agent只写scripts/v2/generate_representation_round8.py、configs/v2/representation_round8_20261024/**、tests/v2/test_representation_round8.py；main只写该计划及原有runner注册/报告。复用representation_family和现有训练、preflight、双卡eval128/reference128、中等及固定panel，禁止新增训练入口、改变src/gradpert或历史实验。完整配置独立落盘；冻结验证器须拒绝重hash后的预算/parent/weight/lambda2/EMA/space/test修改。相同native zero-update/state/RNG与2e-6门保留。

完成本地定向测试、独立只读审查、clean commit/push并绑定独立server checkout后，先10step+restore和双GPU eval128/reference128全部门；main精确验收后同一controller启动300update。正式job交既有监督对话、20分钟检查、主回交fallback；终态停监控，主继续比较和下一轮设计。

交付：同445验证18/有效N与NA、medium1335条件5317行/deep128条件506行/固定panel128、同control异condition与同condition异control、GG/GL/LL、Student/Teacher/raw与bottleneck谱、token与预测差异、CE=H+KL及正确/错误匹配gap、原始和加权loss。若MSE仍比RI差>5%，或几何/正确匹配/条件区分退化，不认定问题解决；单seed描述不冒称显著性。平台新增该臂，精确旧版本保留，默认科学父R6可选RI/R5/R7。性能候选不混入本轮源码。
