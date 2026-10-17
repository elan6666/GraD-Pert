# PT2：蒸馏2轮，再预测微调4轮，双端点中等分析

用户于2026-10-17授权：性能优化后新增此模式，并分析蒸馏2轮后及微调4轮后。
这里的“蒸馏2”是两轮SSL预训练，仍同时包含SSL1 condition和SSL2 DINO/iBOT；
不是只开启编号为②的蒸馏。PT2是新独立实验，原PT1(2+2)/B0/DP2均不修改。

## 顺序与当前状态

当前只准备配置与验收，**prepared_not_dispatched**。训练性能优化的真实双卡完整更新、
无profile整步吞吐/显存/稳定性验证结束后，再启动PT2，之后继续原S12-L4等队列。
不能让新消融抢占性能验证窗口。DP2活动源码、配置、队列与唯一监督不动。
主CPU准备范围不接管监督会话的当前租约；长任务启动后仍交指定监督会话20min。
PT1未完成的可比中等分析仍保留待办，不能被此新实验覆盖。

## 配置与训练

- 起点：从头训练，沿用PT1/B0架构、Jurkat cap40、seed1、表达剔除、图/顺序/视图及mHC。
- 总6轮，global320=micro80×双卡×累积2；当前150更新/轮时共900更新。
- 第1–2轮(更新1–300)：仅SSL1 condition1＋SSL2 DINO1/iBOT1；node/spread/KoLeo0。
  不读取扰动后表达真值，不执行/更新预测头。沿PT1的300步warmup16%+cosine日程。
- 第3–6轮(更新301–900)：保留Student全量微调，包括图编码器、共享GenePT映射、
  Cell/Response与预测头；仅row-mean预测MSE。切换处仅一次重置Muon/AdamW动量，
  恒定LR2e-4、不重新warmup；Teacher/center/温度计数冻结。原始GenePT与图拓扑固定。
- 无逐轮验证/选best；epoch6 last做原18项测试：三Pearson×all/seen/unseen×全基因/统一DEG。
- 预测监督4轮/600步，与当前B0相同；PT2额外有2轮/300步自监督，**总计算预算并不相同**。
  阶段日程/动量重置仍是整套策略的一部分，不宣称严格单变量因果。
- 性能采用项需在正式源、配置与收据中明确绑定；只改变经验证的执行实现，不暗改数学/采样。
  自包含配置：configs/v2/pt2_staged_2plus4_20261017/PT2/gradpert_v2/nadig_jurkat.yaml。
  manifest.json是准备目录，**不是已可执行的四轮PT1队列**；不能绕过其family/budget检查。

## 两个分析端点

|端点|checkpoint|阶段与语义|
|---|---|---|
|A：蒸馏2轮后|epoch2/pretrain.pt|Student与Teacher有效SSL诊断；预测头尚未受监督，预测性能不作为正式末轮结果|
|B：微调4轮后|epoch6/last.pt|Student预测与表征；Teacher/center是阶段A冻结副本，不能冒充新蒸馏目标|

两个checkpoint均保留。沿用共同**中等**范围：1335条件/5317行，deep128条件/506行、
每细胞128基因；gradient32条件/125行。两端点使用同一份冻结row/control/gene/query/
Global/Local/mask清单和seed，逐基因ID对齐；同条件/异条件及context/batch分层，不扩大成全量。
固定诊断视图在训练之外只读产生，不改变该阶段训练输入、损失、RNG或Teacher更新时点。

共同报告包含：逐层GenePT/图/Cell/Response基因token与CLS的有效秩/相对方差；
Global–Global/Global–Local/Local–Local的raw和centered余弦/距离，分同细胞、
同扰动不同control、异扰动及可用的同control异扰动；原型样本/边际熵、JS、使用率；
分项loss/梯度活性与方向；末端预测输出和18项结果。报告真实行数/条件/细胞/基因/配对N，
缺失组标N/A，不把视图重复数当独立细胞。图/表/数据及来源完整归档。

端点A没有训练预测项，其训练预测loss/梯度不可填0；端点B没有训练SSL项，
Teacher CE/center更新记N/A。需要时对端点B作Student只读跨视图探针，明确标为diagnostic-only，
不能伪造实际蒸馏loss。不能把两个阶段的训练joint绝对值当同一目标直接比较。

## 实施与验收

现有native staged lifecycle已支持任意合法2+4预算并保留pretrain与last；配置先经schema/
V2Options解析和与PT1逐字段差分验证。现有run_fgpo_medium仅绑定last，后续必须先补
**显式epoch2/epoch6 checkpoint角色、哈希与阶段身份**的独立分析编排，不能把最后一轮的
journal/收据重标成epoch2。两端点复用同一封存分析清单，允许训练结束后只读补测epoch2，
减少训练中间的GPU阻塞。完整交付需要两个端点的medium COMPLETE与配图中文报告，不仅是末轮。

正式启动前：已发布clean源/配置哈希匹配；真实10更新门覆盖SSL→MSE切换与恢复（工程短边界
单独记录，正式仍300），双卡原300-control/eval128身份与数值验证；中间/最终分析checkpoint
绑定、state不变与共同采样检查。训练/分析/评估版本分开记录，新run ID，不覆盖原实验。
原生模型不改，旧四轮PT1控制器不扩大范围；新编排未验证前不宣称可直接启动完整流程。

永久交付位置：/Users/elan/code/grad-pert/experiment-analysis/，新run独立目录；
包含A/B两份报告、阶段对照、全部图/CSV/小JSON和源码/config/checkpoint/清单哈希。

## 编排实施进展（2026-10-17，未启动）

新增generate_pt2.py与pt2_native_io_20261017独立执行schema，保留原preparation目录。run_br400_group对六轮配置严格匹配PT2生成值，旧family仍四轮；PT2预检必须有staged_training_gate。run_pt2_medium按pretrain/last选真实epoch2/6文件，stage与SHA核验，共用一个封存视图manifest，每端点有原1e-8 observer parity及0更新/state proof。总COMPLETE仅在两端均成功后生成。CPU saved报告先验证完整终态/history，再裁取epoch2，故不会隐藏后半段身份漂移；不借用epoch6 test18。中间预测头尚未训练，表征探针若调用数值读出是diagnostic-only；不变成额外训练监督。当前本地回归中，正式native IO采用还需原生ABBA，正式PT2启动还需同新发布源的阶段10步/恢复和两GPU eval128原2e-6门。

## 已通过工程门并启动正式训练

f1799a089365f1722e73b676b3a338237c370c86，原生ABBA实际+1.745%/allocated−15.85%，有限漂移已披露；CPU118/实际阶段10/双GPU eval128门通过。run nadig_jurkat-seed1-20261017T145243Z-d84c7470867a4c5cbd658beb3e29a96a 已真实epoch1 step7/150/累计7/900，2+4与双端点中等分析按上述配置执行。长期监督只交指定监督查询对话；全交付未完成、无测试指标。不再将其标为仅prepared，不将文档提交当作训练源码。
