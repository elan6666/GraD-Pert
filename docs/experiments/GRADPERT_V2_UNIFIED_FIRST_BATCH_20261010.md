# 统一 MLP 第一批：方法、实现与运行协议

用户授权只执行 N0、U24、MR1、P1、C1、O1、VH、S1-L4、CG1、S12-L4。
历史基底为40fb032365a7d61b8de6caae7e4fb15b78c2ee24。新行为使用独立配置
`configs/v2/unified_first_20261010/`，不改历史v1/v2配置或结果。

## 基线 N0

统一非线性为 pre-RMSNorm(eps1e-6)、带bias的gate/up/down：

\[
F(x)=W_d[\operatorname{SiLU}(\min(W_g\operatorname{RMSNorm}(x)+b_g,10))
\odot\operatorname{clip}(W_u\operatorname{RMSNorm}(x)+b_u,-10,10)]+b_d.
\]

标量表达先Linear1→256；原始冻结GenePT先共享Linear2048→256；随后各用
统一256→1024→256模块。扰动注入/预测特征为512→1024→256；预测末层线性。
四个蒸馏特征头256→2048→256、L2归一化、16384原型线性保持无bias。
FFN外已有一次RMSNorm时不再加内部Norm；图仅替换FFN前的Norm，不改变注意力
Q/K/V归一化、KDA投影、源门、mHC路由或输出激活。原有各角色dropout位置保留。

图3邻域单向KDA＋1来源门控MLA，Cell/Response各2KDA＋1MLA；Response每层
先注入扰动再self/cross/FFN。所有query读最终S；图每目标只写合法邻域，只读
该目标query。随机排列依赖身份ID，没有声称严格排列不变。mHC4/宽256/头4/秩64。

CPU完整实例计数（不含Teacher，冻结先验13,324,288）：

| 配置 | Student可训练参数 |
|---|---:|
| N0 | 42,699,822 |
| U24 | 42,980,782 |
| CG1 | 45,596,478 |

这些是实例计数，非CUDA容量或科学效果证据。

## 十个独立配置

| ID | 相对N0的唯一机制改动 |
|---|---|
| N0 | 共同完整基线 |
| U24 | 同一冻结原始GenePT的2048/64/256共享读出校正，down层权重/bias零初始化，不另存先验副本 |
| MR1 | 25%训练control查询表达遮蔽，预测扰动后表达辅助loss权重1；遮蔽原始残差及cross，不复用未遮蔽Cell |
| P1 | 同条件control/perturbed群体，均值MSE＋MMD替代随机逐细胞MSE；两个蒸馏保留 |
| C1 | 仅第一层前注入ep；cross及输出拼接保留 |
| O1 | 所有9个KDA改固定ID读取顺序 |
| VH | 两套蒸馏同时Global .8–1、Local .4–.65；数量2＋2不变 |
| S1-L4 | SSL1 Local为1GO＋1STRING＋2随机节点；SSL2仍2Local |
| CG1 | control摘要条件化图第四层的来源key门；前3层共享，value/拓扑不变 |
| S12-L4 | SSL1为2GO＋2STRING Local；SSL2为4随机表达Local |

所有预测/condition/node/DINO/iBOT系数1，spread/KoLeo0。有效CLS配对等权，
2Global+2Local为6项，2Global+4Local为10项。分支独立随机流防止增加图Local
推进表达视图的RNG。S12-L4与S1-L4不是隔离SSL2数量的单变量对照。

CG1使用共享feature/weight/summary MLP，Sigmoid逐维贡献、加权求和而非均值；
summary随后RMSNorm会弱化共同尺度，不能声称保留所有基因数/表达总量信息。
来源门为1+sum_s source_s(E_s+eta_{c,s}U_s)，eta=Tanh(Wcond*z+bcond)。Wcond/bcond
零初始化、U正常初始化；无图虚拟节点。SSL1匹配(control,p)/(control,gene)，SSL2
每个遮蔽/裁剪视图分别生成摘要，不复用未遮蔽主摘要。前三层图结果在输入相同
时复用；第四层按control计算，query分块，masked-node原型loss按256节点分块重算，
Teacher统计只在原forward记录一次。浮点求和顺序可能产生有限精度差异。

## P1的群体监督

\[
L_{pred}=\|\operatorname{mean}\hat Y-\operatorname{mean}Y\|_2^2/G+MMD_u^2(\hat Y,Y).
\]

多尺度RBF核的方差为训练truth非对角平方距离median乘(.5,1,2,4)；带宽detach，
核exp(-distance/(2*variance+1e-12))；四个无偏估计均值，排除各自集合对角，
保留有限负估计。MMD系数1为本项目预设，并非官方默认或测试调参结果。少于2
样本/退化带宽跳过MMD并记录有效项，仍计算均值MSE。每轮真实训练行恰好一次，
不采用官方每条件1000次的人工epoch长度；control有放回抽样并匹配truth的
cell_type::batch组成，没有声称实际单细胞前后配对。

参考：[scDFM固定实现](https://github.com/AI4Science-WestlakeU/scDFM/blob/8de47e7d3d443939f93f768f7902c66e0e82eda2/src/script/run.py)、
[DINOv2配对归一化](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/train/ssl_meta_arch.py)。
原生实现无上游运行依赖；这里的CG1/U24/MR1是项目方法定义，不声称复现对应上游模型。

## 运行、验证与证据

同一Jurkat train-only cap40/split/225表达剔除/seed1，从头6轮，无每轮验证/早停/
best选择；最后last.pt依原300-control清单评估。三Pearson×all/DEG×全部/seen/unseen
共18项，保留实际有效条件数/NA和三套原reference。科学数据/checkpoint不离服务器。
每卡一个独立实验，world1、accumulation1；batch32仅候选，须所有组10步非零LR、
恢复及300-control推理预检，保留失败证据并按同一batch整体回退。10步不证明长期容量。

队列先并行最重预检，再核对全部十组收据，然后自动进入正式训练；两个lane
按原清单偶/奇行运行，先N0/U24。源版本/配置/runtime/发布收据逐子进程校验；
失败停止后续启动、不杀活动peer、不静默重跑。长队列交既有监督会话20min检查；
终态返回主会话验收，停止旧监控；不启动第二批或V0/V1。

当前本地验证：完整模型十组各3步optimizer/EMA/center与精确checkpoint续跑；
核心MLP、条件图输出/梯度、MMD、采样预算及队列身份/失败规则测试。
全v2回归737通过、11环境跳过、1既有失败：旧sinkhorn测试引用不存在的
single_pass_jurkat/one_epoch_m66_a2配置，发布前40fb032树同样缺失；不补造历史配置。
CUDA、吞吐与正式结果待真实收据，不以本地合成测试宣称科学完成。
