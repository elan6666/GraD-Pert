# 未见表达基因：三个独立机制消融（讨论记录）

状态：用户已授权执行；三个机制与独立自包含配置已实现，新增20个测试通过，正在发布和准备容量队列；尚无新正式实验结果。原讨论记录保留，执行状态见计划和收据。三个独立消融在本稿中定义为各自仅开启一项，与共同L0参照比较；不是三项全开后分别关闭。当前主模型默认值不在讨论阶段修改。

用户原话：“讨论：可以，加上这三个改，并且生成三组独立的消融实验，并且移除L2，L3，M1，M2同时做”。

## 范围修订

- L0已有完整六轮与final-last结果，保留。
- L1未被取消，保持原六轮及final-last测试；若已终态则保留结果。
- 撤销旧待跑L2、L3、M1、M2的启动授权；已有代码、容量探针、运行与历史证据不删除。若其中一组已启动，由当前监督所有者保存证据并安全停止。
- 主会话已向指定监督会话01a0df0b-4142-7df1-86c0-d959471d80a1发送执行范围收缩通知，要求不改活动源码/封存配置，安全阻止旧controller继续推进。实际停止措施及验收以监督收据为准，本稿不冒称已完成。
- 监督返回并经主会话校验的收据：`/Users/elan/.codex/worktrees/v2-gene-exposure-eval/grad-pert/.byte-os/coordination/receipts/loss-ab022ca-scope-contraction-connectivity-20261008T1517Z.json`，SHA256 `35cc35b04fde08a37559e3844aa185a705954262ca5f447758b9e52c6f3d0d49`。本地监控范围已收缩，但TCP/22与SSH不可达，实际没有停止controller/rank，远端是否已进入被撤销组未知；后续自动调度尚未被核实阻断，待原监督所有者恢复连接后执行。不得把修改监控prompt描述为已取消服务器队列。
- 后续执行收据已覆盖上条“尚未阻断”的当前状态，但保留历史证据：北京时间2026-10-08 23:23，学校网络直连SSH恢复；监督核对精确queue/attempt/owner后仅SIGSTOP调度控制器，postcheck=T，L1 worker和双rank仍存活，测试491->493/592。L2/L3/M1/M2无formal log/exit或匹配进程，尚未启动。主会话核对收据SHA256为`a8113fece96f86b44fb875c6d792b056772c6437e821c8d7c94eb2ebdde57677`，文件`/Users/elan/.codex/worktrees/v2-gene-exposure-eval/grad-pert/.byte-os/coordination/receipts/loss-ab022ca-scope-contraction-enforcement-20261008T1523Z.json`。当前旧后续自动推进已阻断；原监督继续核对L1 final-last，随后终止并释放挂起controller、停止旧监控并交回主会话。新U组仍处于讨论，无产品代码或实验启动。
- M3 masked perturbation prediction只保留讨论，不加入本三组。
- “同时做”按同一批设计/交付解释；正式组默认各独占双卡顺序训练。不预先承诺三组并发；若要并发，需要相同训练语义、实际峰值内存和总吞吐证据。

## 独立开关与实验

| 行 | prior_shared_adapter | gene_conditioned_readout | direct_target_flag | 与共同参照的差异 |
|---|---|---|---|---|
| U0共同参考 | false | false | false | 原L0模型 |
| U1 | true | false | false | GenePT底座冻结，以共享适配器替代独立可训练表 |
| U2 | false | true | false | 图表保持可训练，仅用冻结初始先验生成最后预测层的权重修正 |
| U3 | false | false | true | 仅给表达变化量预测头追加“当前输出基因是否为直接靶点”标志 |

U0不是第四项新机制实验。建议在统一新源码版本复跑一次共同参考，以避免将已有跨版本L0差异当作新机制效应；旧ab022caa57a3b45dc5a14c6ae38bc280702112e4的L0仍保留历史参照。这个基线重复是推荐设计，不在本次讨论中自动启动。若决定仅复用旧L0，结果必须标为跨版本比较，并附关闭三开关的输出、初始化、随机流、多步更新与评估兼容性验证。

### U1：稳定基因先验

E0由既定GenePT确定性降维得到，仅含先验；不使用任何训练/测试表达生成。
E_g=E0_g+A_theta(E0_g)，E0固定；A建议256->64->256、GELU，末层零初始化，随后沿用现有graph norm及图层。
不同时加入每基因独立可训练残差。宽度256、图KDA/MLA、来源机制和其他编码器保持原定义。
U1最开始的基因表示与L0一致，之后梯度结构有意不同，不能要求训练轨迹等价。

### U2：先验条件化的变化量解码

当前head前半段phi([r_g;e_p])沿用。w_g=w0+H_theta(E0_g)，Delta_g=w_g^T phi([r_g;e_p])+b0。
H建议256->64->256、GELU，末层零初始化；冻结E0副本仅作head条件，不能通过这个副本读取表达。
原图输入基因表仍可训练，避免把U1暗中并入U2。保留原预测头bias；不同时添加其他门控或正则。
初始Delta与原head一致，之后是方法改变。预测仍为x_control+Delta。

### U3：直接靶点身份

t_pg=1[g属于T_p]，D([r_g;e_p;t_pg])输出Delta；输入从512增为513，隐藏维度仍256。
追加列权重零初始化，其余旧列与bias保持对应初始化。标志由gene ID和已知扰动ID计算，不读取表达或DEG。
只在最终预测头加入，不在Cell Encoder改变control表征，不额外强制query包含靶点。
不强制负方向，不把CRISPRi假设套在CRISPRa/药物任务上。

## 共同训练与评估

拟沿用当前L0：Jurkat cap40、seed1、从头六轮、global272=micro68*accum2*world2、无validation、无早停、final epoch6 last测试、best=null。
Graph3单向KDA+1来源感知稀疏MLA，Cell2KDA+1MLA，Response2self/cross KDA+1self/cross MLA；final-S读取、随机顺序、mHC4、宽度256、4头、16384原型、完整预测+SSL1+SSL2。
预测row-mean MSE；lambda1=lambda2=1；SSL1condition/node=1/1，SSL2DINO/iBOT=1/1，spread/KoLeo=0。M1/M2辅助重建关闭。
LR1e-3->2e-4，项目warmup+cosine；既有optimizer/center/EMA更新规则、图邻域、视图、cap40选择、split、300-control有序manifest保持原协议。
测试靶基因表达剔除仍true；225未见基因的表达不得成为训练输入或监督。原三Pearson定义不在本次改动中调整。
各组各自同构Teacher；固定先验作为buffer保持一致，trainable新增模块EMA沿用既定时点。

总体5000、已见表达4775、未见表达225各输出三Pearson all/统一DEG。补充把未见组按当前条件的direct target与other unseen拆分；direct target常只有1基因，不能在单条件内硬算Pearson。可报告每条件目标基因误差/方向，其他未见基因Pearson按有效>=2规则；若跨条件目标表达/变化量相关则单独命名，不冒充既有条件宏Pearson。

## 后续实现路线（待执行）

1. 在独立worktree核对并保留已发布前版本；实现三个config开关，off路径不改变既有模型/初始化与随机流。
2. 每个开关核对方程、梯度、冻结参数、gene/target ID对齐、多步非零LR、optimizer/Teacher/center和checkpoint恢复；新方法不混作等价性能优化。
3. 全部新增模块用独立初始化随机流；报告实际trainable/Student/Teacher参数与计算成本。
4. 发布同一干净新SHA，自包含U0/U1/U2/U3配置与新run IDs。
5. 三组同配置双卡持续128更新容量/恢复/300-control推理验证。新容量证据未得到前，不声称batch272已支持新模型；失败不私自降某一组batch。
6. 正式六轮和final-last评估后由既定监督会话交回主会话。只启动后续明确进入执行范围的组。

参考机制：scPRINT的ESM2先验保留；DeepSpot-M的基因嵌入条件化输出投影。U1/U2/U3是GraD-Pert原生适配提案，不声称复现或原创已有通用机制。
