# U4＋GELU：表达泛化、顺序与监督对齐消融

状态：仅设计。用户2026-10-10请求设计新组；本计划不授权启动训练、
修改旧运行或开启监控。当前实现基底为已发布
`7a5018ad3d07025327547b0d1d1dacbc1d6f9049`；新监督、诊断和配置矩阵尚未实现。
后续实现须另行发布同一干净版本，不能直接拿本基底宣称全部实验可运行。

## 1. 基线与公共协议

实验族建议 `u4_gelu_mechanisms_20261010`，各编号只在这个族内解释。
N0是新完整基线：固定6506×2048 GenePT；共享Linear2048→256→精确GELU→
LayerNorm；图3层单向合法邻域KDA＋1层来源门控稀疏MLA；Cell2 KDA＋1 MLA；
Response2 self/cross KDA＋1 self/cross MLA，逐层注入扰动。所有query读最终S，
随机读取顺序，mHC4，宽256/4头/秩64，FFN clipped SwiGLU，dropout0.1，
四个16384原型头；不启用DSA、U1、U2、U3或control数值重建。

父配置：`configs/v2/cap40_genept_gelu_20261010/gradpert_v2/nadig_jurkat.yaml`。
Student可训练34,528,302，总参数47,852,590；额外13,324,288为冻结先验。
新GELU路径只有本地98项验证，尚无训练/效果/速度证据。

- 同一Jurkat train-only cap40、冻结canonical split、seed1；全部从头6轮，
  不逐轮计算验证loss/Pearson，不早停，不选best；末轮last-only冻结测试。
- 候选共同batch272=每卡68×双卡×累积2；测试阶段各组10次完整更新、
  checkpoint恢复及300-control推理。10步只是工程门，不宣称长期容量。
  若主矩阵任一组不适配，正式启动前统一冻结新的共同batch及配置身份。
- Muon分流优化器，峰值LR1e-3/16% warmup/cosine末值2e-4、WD0，
  图扫描chunk32/图目标64/序列16，原预取和checkpoint设置不变。
- 主损失行平均MSE＋SSL1 condition/node＋SSL2 DINO/iBOT，五项系数1；
  spread/KoLeo0。不在本组复活它们，不重复mHC开关或原型数量消融。
- 原225个测试靶基因的表达始终不进训练输入、预测标签或新增监督/缓存；
  身份和图先验保留，测试恢复control表达。主矩阵不新增基因留出。
- 同一原始300-control有序清单、全部truth与原有三种reference；不修改
  Systema训练＋验证非control条件均值等权reference。
- 相同主数据/视图/顺序/辅助任务独立随机流，记录ID和RNG来源。
  新辅助支路不推进Teacher/center第二次，也不访问val/test训练目标。

## 2. 第一批11组：一项机制一个问题

以下全部相对N0独立变化，不叠加。旧功能组与旧U4只有历史参照作用，
不作为新基线的同协议运行，不覆盖其配置、SHA或指标。

| 组 | 改动 | 主要对照/问题 |
|---|---|---|
| N0 | 新U4＋GELU完整基线 | 所有新组的共同锚点 |
| Nlin | 仅GELU改Identity | 与N0隔离激活的作用；原始先验、共享投影及初始化不变 |
| MR0 | 新增未遮蔽响应辅助监督，系数1 | masked-response的额外监督/计算匹配对照 |
| MR1 | 新增masked-response监督，系数1 | 与MR0隔离control表达缺失训练的作用；与N0报告净变化 |
| P1 | 新增训练条件群体均值监督，系数1 | 单细胞随机配对监督与群体评估之间的错位 |
| D1 | lambda1=0 | SSL1是否帮助响应预测；预测图编码器仍保留 |
| D2 | lambda2=0 | SSL2是否帮助响应预测；Response和CLS仍保留 |
| C1 | 仅Response第一层前注入ep | 后两层去掉重复注入；全部control cross、输出拼接保持 |
| O1 | 所有KDA采用固定基因ID顺序 | 图邻域/Cell/self/cross一起切换，身份与采样集合不变 |
| G1 | 图第四层关闭来源key逐维门 | 保留来源偏置、合法边、value及前三层来源信息 |
| G2 | 删除随机Hamiltonian环边 | 保留GO、STRING、self，不用其他随机边补齐 |

MR0/MR1使用同一每细胞25%主查询基因子集M（1000中约250），同一辅助随机流，
同一当前真实扰动标签、同一辅助前向数量、同一系数1。MR0不遮蔽输入与残差；
MR1遮蔽它们。因此MR1−MR0检验遮蔽，MR0−N0检验额外监督的影响。

D1/D2保留模块和初始化，避免同时删骨干/换参数量；关闭loss的图/表达视图
不必执行无用前向。Teacher仍同构、EMA每次optimizer后一次；有teacher logits
的活动head正常更新center，关闭head的center保持不动，完整记录差异。

## 3. Masked-response：禁止直接残差及cross泄漏

先抽取M，只遮蔽control表达嵌入，保留基因ID/图先验/扰动身份。必须重新生成
masked Cell输出Bc，Response初始化和每层cross都读取这个Bc，不能复用未遮蔽Bc。
原始主视图与四个SSL2视图照旧，不把新辅助mask混入既有蒸馏视图。

\[
\tilde b_{c,g}^{in}=h_g^{graph}+
 \begin{cases}e_{mask},&m_{c,g}=1,\\\phi(x_{c,g}),&m_{c,g}=0.\end{cases}
\]

现有实现`encode_response`即使传expression_mask也返回`control+delta`，所以
仅传mask不符合新设计。新辅助预测须显式使用

\[
\hat y^{aux}_{c,p,g}=(1-m_{c,g})x_{c,g}
       +D(r^{masked}_{c,p,g},e_p),
\qquad
L_{MR}=\frac1B\sum_c\frac1{|M_c|}\sum_{g\in M_c}
 (\hat y^{aux}_{c,p,g}-y_{p,g})^2.
\]

遮蔽位置学习绝对扰动后表达；mask=0的主预测仍是原残差任务。两个模式由
mask token区分，不另加预测头。MR0取同一个M上的普通`x+delta`辅助误差。
这里新增的是方法监督，不是等价优化；随机mask不等于整段训练无基因表达监督。

关键单测：保持可见表达、mask、seed与目标相同，仅改变被遮蔽位置的canonical输入x，
辅助预测必须不变；原始x对此输出的导数必须为0。原主支路保持原语义。
标签y只能进入loss，不能进入token/cross/图/基线缓存。

## 4. P1不是“condition_mean MSE”

基线预测仍row_mean，SSL仍row_mean及原例外。P1额外每步均匀抽4个训练条件，
每条件独立抽8个训练control（32个辅助细胞，不跑新增SSL），匹配该条件训练
真值的cell_type::batch分布。训练真值均值从实际cap40清单构造，按条件/context
计数混合，缓存只含训练可用表达轴；不得读验证/测试或被排除表达列。

\[
L_{pop}=\frac1{|C_{step}|}\sum_{p\in C_{step}}
 \frac1G\left\|\frac1K\sum_{k=1}^K\hat y_{c_k,p}
              -\mu^{train}_{p}\right\|_2^2,
\quad K=8,\ |C_{step}|=4.
\]

这是“先平均再比较”，不是“每个细胞MSE按条件平均”。主batch随机混合时
许多条件只有1行，直接对这些单行取均值无法消除配对噪声，所以使用明确的
独立多control辅助样本。有限K仍有均值采样噪声，不能称消除了全部噪声。
跨rank条件均值、计数与梯度正确汇总；不会各rank先求平方再简单平均。

## 5. 先做诊断，再决定补充组

所有新诊断均是训练结束后单独执行，不恢复逐轮验证。冻结一个非test的小型
probe面板及输入/query/control/target/RNG；先在训练侧probe使用，不依据test
结果选择组或系数。eval模式关闭dropout；诊断不更新EMA/centers。

1. 顺序：同一面板8个排列，加固定ID/反序对照；分别干预图邻居顺序、
   Cell/Response共同顺序、两者一起。图边按ID保持，不重采样视图/表达。
   输出逆排列归位后记录预测和delta的方差、相对L2、P95、群体均值波动。
   不混淆token存储重排与KDA扫描重排；对softmax/retention做纯ID重排数值对照。
2. Control：固定p/context/query，使用成对不同control记录y、delta、CLS变化，
   并检查delta变化是否近似抵消x变化。只置换同一条件的同一300-control集合
   不改变预测集合/均值，因此群体Pearson不下降不是“忽略control”的证据。
   替换集合要单独报告抽样变动，不冒充相同正式reference指标。
3. 扰动：固定control替换ep/已知target，记录delta与CLS变化；既有CLS→gene
   阻断仅作eval诊断，不变成训练消融。
4. 蒸馏梯度：N0 epoch1/3/6 checkpoint各固定8个训练probe batch，分块统计
   pred与condition/node/DINO/iBOT在共享投影、图、Cell、Response的余弦及范数比。
   无共同梯度的模块记NA，稀疏grad正确展开，不把不存在的grad当冲突。
   不增加optimizer/EMA/center提交，也不据负余弦自动改权重。
5. 图覆盖：原225基因的GO/STRING度、直接/最多4跳训练表达覆盖、来源重叠、
   环边占比、先验范数；按训练侧确定的分箱和匹配表达量/变异度报告误差关联。
   相关性不是因果，G1/G2作为结构对照；删环仍保留self，零生物边如实记录。

## 6. 真正未监督基因的训练侧诊断：独立两组

V0=N0，V1=MR1，共用固定J：从原4775训练可用基因按基因ID哈希seed20261010
抽floor(10%×4775)=477个，不利用测试表达/DEG选J。J在整个训练中不进入输入、
预测标签、新增MR/pop监督、SSL2表达或目标缓存，身份/图先验仍可见。
主矩阵原225剔除集不变，V系列明确额外剔除J；不能与主矩阵直接同口径排名。

probe truth来自原训练条件中未被cap40选中的剩余训练partition细胞，清单与
cap40训练严格不交；control使用训练control来源并冻结300个ID。若没有足够
剩余truth，该条件标NA，不重复训练行、不使用test补足。运行前核验可行性。
结束后一次性检验J的响应，单独报告原225、J477、其余4298三个集合。
这验证从未被数值监督的基因迁移；不以随机mask结果代替。V结果用于方法诊断，
不将J的高分解读为原225测试效果，冻结统计方案后才看正式test。

## 7. 第二批补充组（不自动加入第一批）

| 组 | 相对N0的变化 | 使用目的 |
|---|---|---|
| K1 | 全部KDA正反状态接力 | 新先验下是否降低顺序波动；不宣称排列不变 |
| K2 | 仅Cell/Response self-KDA读St | 末态压缩相对逐位置读出的影响 |
| A1 | 所有图/self/cross KDA和MLA替换softmax | 完整注意力对照，图仍限合法邻域 |
| A2 | 同范围替换CellFM式ReLU retention核心 | 顺序无关汇总对照，保持原来源适配定义 |
| C3 | 删除所有control cross | control回读是否有贡献，仍由Cell输出初始化 |
| R1 | 增加两排列delta一致性，系数0.1 | 仅当训练侧诊断显示明显顺序波动时再考虑 |
| MP1 | MR1＋P1 | 二者主效应验证后才研究交互；已有N0/MR1/P1成2×2其余三格 |

R1仅改变扫描顺序，两支共享同一基因/表达/图/扰动/视图与dropout掩码，
loss为mean((delta_pi−delta_pi2)^2)，两支均有梯度；不多更新Teacher/center。
额外随机流隔离，训练掉落噪声不作为排列差异惩罚。新增顺序无关分支、
control条件图门控、OT/MMD分布训练等仍是未冻结方法提案，不混入当前矩阵。
C2（只末层cross）及H1（关mHC）维持用户此前移除决定。

## 8. 评估、seed与交付

- 保留三个Pearson×all/统一DEG×整体/见过/未见，全部先细胞均值再相关、
  条件等权。各项给finite/total/NA原因，不假定新组天然都是147有效DEG。
- 同一control参考定义的效应RMSE/MAE、向量cosine、幅度比||pred_delta||/
  ||true_delta||及alpha=(true_delta·pred_delta)/||true_delta||²；零分母记NA。
  上/下调方向只在冻结truth-DEG上报告，并附严格非零与训练侧噪声阈值敏感性。
- 未见DEG另报告交集2、3–4、≥5的条件数/相关及≥5子集。保留原147口径；
  两个非恒定基因时Pearson±1的退化情况不能误当强泛化证据。
- 无训练模型的control-copy与train-only条件等权平均效应基线，使用同一
  control/context/基因分组。后者只用实际cap40训练可用基因；未监督列回退
  零效应并报告。Pearson遇到常量向量为NA，不伪造0。未实现线性基线不冒称有结果。
- 初批seed1；预先指定N0/Nlin/MR0/MR1/P1补seed2、3，不依据test冠军选重复组。
  种子重复与条件配对bootstrap分开解释；bootstrap条件不能代替训练seed。
- 记录训练分项loss、输出/梯度/完整更新验证、参数量、optimizer步数、纯训练与
  推理墙时、细胞吞吐和峰值显存。新增监督与D1/D2的joint尺度不可直接排名。

建议次序：基底/诊断支持实现与测试→各组10步工程门→N0及训练侧probe→
Nlin→MR0→MR1→P1→D1→D2→C1→O1→G1→G2。V0/V1作为单独迁移诊断轨。
这是规划次序；执行、seed补充和补充组须有明确授权后进入队列。
短时门由Luna只读监督，主不重复轮询；长队列交既有监督会话，终态后撤监控。
按先前偏好可在上一组postfit期间推进下一组fit，但必须通过资源与总吞吐门，
不能为了并发使两任务总体更慢或侵占未授权资源。

接受标准：配置逐行差异可审核；全部新方法保持基因ID/剔除/Teacher/center
规则，完整非零LR更新与checkpoint恢复正确；新源码/config/data/ordered
control/truth及checkpoint/eval各SHA可追踪；末轮真实测试与分项结果齐全、零PKL。
计划完成不等于代码完成、工程门通过或训练完成。

## 9. 来源与解释边界

- [当前U4及L0/U1/U2/U3结果](../../docs/experiments/GRADPERT_V2_U1_U4_RESULTS_20261010.md)：
  单seed、不同先验机制；不用于据test调参或证明新GELU收益。
- [Kimi Linear](https://arxiv.org/abs/2510.26692)：KDA有限状态门控的原始来源；
  我们的图邻域/最终状态读出为项目设计，不能从语言任务结果推断本任务收益。
- [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)：
  集合对称性动机；随机顺序KDA不满足其结构性不变性保证。
- [CellOT](https://www.nature.com/articles/s41592-023-01969-x)：非配对细胞分布与
  均值/异质性区分的动机。这里P1只做群体均值监督，不实现CellOT或真实反事实配对。
- [Ahlmann-Eltze et al. 2025](https://pubmed.ncbi.nlm.nih.gov/40759747/)及
  [作者代码/数据](https://zenodo.org/records/16092690)：简单基线对照动机。
- [Wei et al. 2026](https://www.nature.com/articles/s41592-025-02980-0)：多场景
  泛化基准。此次Nature原文直读受认证跳转限制，标题/出处由检索与PubMed核对；
  不声称复现其具体实验。上述新loss/系数/样本数均为项目设计建议，不是论文默认。
