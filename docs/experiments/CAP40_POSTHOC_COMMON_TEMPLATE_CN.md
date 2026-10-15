# 消融共同分析模板 v3：明确标注全量或中等规模

当前用户改用中等规模；旧全量清单/未完成产物保留，不能并入中等结果。
共同清单从原cap40清单按context分层取每条件最多4细胞，保留全部1335条件；
深度为固定128条件（按context集合及原条件细胞数分层），梯度32条件为其子集。
最多5340基础行/512深度行/128梯度行，实际数以封存清单为准，singleton不复制。
同一五臂共享原始truth/control/query/view种子和新实际mask/edge recipes。
先PO，再DP/FN/FG/FB；不重训、不重跑旧正式测试。真实18指标复用旧last收据。
基础条件等权统计覆盖所有条件，row_mean只代表平衡抽样，不代表原全量行均值；
深度条件的未加权汇总只描述该分层清单，不能声称总体无偏估计。
CLS组间比较固定query panel+context；control treatment与paired perturbation分开。
每条件最多4细胞使条件内CLS秩上限很低，逐项报告实际独立control数，不补零。
以下原v2条目若写全量/64条件，均为历史全量方案；当前报告必须按actual scope/count覆盖。

适用 DP/FN/FG/FB/PO 的既有 last.pt。以同一 sealed manifest 为比较锚点。
更新于 2026-10-15；覆盖旧模板的四细胞/固定前缀探针限制，但不改写历史证据。
不重训，不更新 Student、Teacher、center、optimizer、梯度槽或任何 RNG 状态。

1. **结论与证据范围**：全量是 cap40 训练全集；历史训练轨迹、冻结 last 补测、正式测试三种证据分开。
2. **版本与校验**：run、checkpoint epoch/role/SHA、training SHA、analysis SHA、resolved config SHA、data/split/control/truth与清单hash；不可变源码。
3. **实验与配置**：解释本臂的单变量、损失权重、归一化/激活/bias/原型规则，引用原冻结配置；不按实验名推断配置。
4. **共同样本与覆盖**：47,836 truth 行、1,335 conditions；独立control去重数、每条件/context数、深度/梯度数、表达可见性、查询panel和ID/视图/mask映射。细胞系、细胞类型、batch分开。
5. **统计定义**：中心化协方差、有效秩/上限、相对方差、各统计的gene/cell/condition/view轴；条件等权及行加权分开；以条件bootstrap置信区间。多个view不是独立细胞。
6. **历史训练过程**：全部已保存epoch分项loss、权重、LR/EMA、梯度、耗时/显存。未保存写未采集，不能从last反推过去。
7. **原型与实际配对**：四头×Student/Teacher×视图；H_sample/H_marginal/JS/N_eff/top1/argmax；真实配对CE/H_pair/KL；center只读。全局JS用累计概率重算，不能平均分块JS。
8. **分组表征**：GenePT、全图、各主干组件/CLS、各视图、投影输入/瓶颈；全局混合秩、同细胞基因秩、同基因同context跨control方差、同条件CLS与不同条件均值、同细胞跨view分别列出。mHC stream均值须明示。
9. **输出**：预测/复制MSE、负值/零值/极值、效应RMS、幅度斜率/方向；训练query子集与正式全轴测试不可混称。剔除表达的基因不在训练诊断恢复标签。
10. **梯度与效率**：64固定条件×最多4cells，graph各共享参数组prediction/SSL1/SSL2 norm/cosine；无更新；启动+数据/诊断实际耗时、峰值显存。不是整个训练梯度的总体估计。
11. **正式测试**：复用原last18项（3Pearson×全基因/DEG×all/seen/unseen），每项finite/total、checkpoint与测试receipt。不得因诊断重跑测试或选参。
12. **共同四问**：表征是否区分、低loss来自H还是KL、输出/泛化如何、梯度/效率是否支持；每问分已观测、解释、未验证假设、限制。
13. **完整交付索引**：报告实际嵌入所有产生的图，逐图解释轴/分组/独立N/读法；列出所有CSV/JSON表的路径、粒度、大小和SHA。正文呈现共同汇总表，完整逐行明细提供CSV，不以摘要代替完整数据索引。

## 固定新旧图目录

- 01–04：原loss/prototype/representation/output-gradient图原字节保留，明确为历史小probe。
- 05：全量/深度/梯度覆盖；06：分角色视图的全量原型；07：全局表征。
- 08：逐行输出误差与效应幅度；09：分条件梯度。
- 新增加的图也必须完整列入，不限于九张。不得用旧图冒称全量图。

## 边界

同一Jurkat不能估计跨cell_line变化。当前补测使用原方法eval模式的固定顺序，不是训练随机排列敏感性检验。
单细胞query预算1,000不变；panel在允许的4,775基因内覆盖，所有臂共用实际清单，不仅共用seed。
最多128个哈希选中基因计算单细胞基因秩，秩上限127；全量细胞不等于每个细胞每层全基因协方差。
冻结checkpoint的全量训练输入诊断不等于训练期间全量激活历史；两者必须分表。
