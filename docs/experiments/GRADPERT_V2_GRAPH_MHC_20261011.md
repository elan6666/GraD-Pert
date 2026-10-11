# GM1：仅给图编码器加入 mHC

GM1与本轮BR比较。BR训练源码b37b171fd68cb29e5c8142c66778aac6da9a6f06、配置SHA256075331f51d2ff5699ce0f22386d756ed5b4ee80cce29ee10727cf676c47e62eb；GM1配置只多一个 `graph_mhc: true`。这是残差架构消融，不是等价性能优化。默认仍关闭图mHC；当前BR/F1/N1活动队列与源码保持原样。

## 方法边界

图仍是3层单向KDA＋1层边来源感知稀疏MLA。合法邻域GO Top20∪STRING Top20∪固定Hamiltonian环∪self、来源逐维key门和偏置、随机读取顺序、每个query的独立合法邻域状态、最终状态读出均保留；没有新增虚拟节点或合法邻域外读取。

每个图基因维护4条流 `X_i ∈ R^(4×256)`。起点为同一GenePT图身份表示重复4份。每层注意力、FFN分别有独立共享mHC路由器；4层共8个。路由参数由所有基因共享，不是每个基因新增一套参数。

对任一子层，沿用native mHC路由：从当前节点4条流的RMS归一化拼接生成预读门 `a_i`、输出门 `b_i` 和Sinkhorn双随机残差矩阵 `R_i`。先聚合 `u_i = sum_s a_(i,s) X_(i,s)`；图注意力仍只读取该节点的合法邻居，FFN则逐节点执行。设原子层输出为 `f_i`，流更新为

`X'_(i,s) = sum_t R_(i,s,t) X_(i,t) + b_(i,s) Dropout(f_i)`。

注意力后再以另一套路由执行原FFN。前三层KDA及第四层MLA一路保留4条流；仅第四层完成后对流平均，恢复 `6506×256` 或当前图视图的节点轴，再按原query positions输出所需基因。

保留每个core的norm1/norm2、MLA低秩KV归一化、FFN/激活/bias/dropout。图mHC路由器不额外插入输入RMSNorm，以避免把“残差机制”和“LN换RMSNorm”混在一起；路由器自身的拼接RMSNorm是mHC映射的一部分。前三层沿用原graph row chunk及checkpoint；MLA维持原完整节点轴计算。4条流混合只在同一节点内部，不改变图跳数或基因身份。

Teacher与Student同构；新增路由在共同模块全部初始化后使用CPU私有随机流构造。共同参数和原CPU/CUDA随机流保留。单流时直接执行原core，输出及参数/input梯度逐bit一致。`graph_mhc=False`不进入旧architecture payload，保持旧checkpoint身份。当前不支持与control条件图同时开启：配置显式拒绝，避免静默绕过条件分支。

## 配置和参数

|项目|BR|GM1|
|---|---|---|
|图残差|普通残差|4流mHC，注意力与FFN各一套|
|Cell/Response残差|mHC4|相同|
|图层数、宽度、头|4、256、4×64|相同|
|GenePT|冻结原2048先验，共享2048→4096→1024→256、GELU|相同|
|Student可训练参数|46,854,190|47,051,014|
|新增参数|—|196,824，约0.42%|
|主训练配置|micro48×2GPU×accum2=global192|相同|
|训练与测试|fresh4epoch，不验证，epoch4 last-only 18Pearson/zeroPKL|相同|

计数来自CPU合成形状6506×2048，无真实数据或模型权重下载；不含冻结GenePT表与EMA Teacher。

## 执行与验收

先完成native单元/回归、输出梯度/非零LR多步optimizer/EMA/center及checkpoint续接、sourceclean和Git发布。新服务器checkout与runtime/queue独立身份，不修改活动BR/F1/N1。GM1控制器先以CPU等待密封的现有三组队列；等待时不获取GPU lease、不调用CUDA或新子训练。父BR/F1/N1真正科学COMPLETE及训练/测试/checkpoint/source/config哈希、epoch4和zeroPKL验收全部通过后，才自动执行GM1十步预检/恢复/单条件300-control真实推理。

GM1预检成功后必须停在主验收屏障，监督会话返主核验；主接受精确source/queue/receipt后才正式fresh4训练与epoch4 last评估。失败保存证据并返主，不静默改变batch、模型、运行ID或当前父任务。用户授权的micro40/global160是必要时独立版本的统一回退候选，尚未运行。

当前实现CPU测试与审查通过；GPU预检和科学结果待父队列完成，不能宣称新图mHC已改善效果、速度或持续容量。

## 部署中断及恢复（2026-10-11）

GM1首个已发布源码为`93ed39d02fa76730630a718d79c489b6adf64f48`。服务器CPU合成检查248通过（56.07秒），但其checkpoint RNG保存意外初始化CUDA：我们自己的pytest PID2337434占用498MiB，不属于活动BR控制器树，导致严格共享GPU保护中止BR。此错误不属于GM1模型容量失败，也不应归因于其他用户。GM1模型CUDA预检/正式训练尚未启动；依赖封存失败，未创建GM1队列。旧发布、测试和BR失败收据原样保留。

后续CPU测试使用显式隐藏CUDA的独立子进程；CPU配置解析与checkpoint核验也隐藏CUDA，不放宽资源门。BR原科学源码`b37b171fd68cb29e5c8142c66778aac6da9a6f06`不变，完整恢复epoch1 checkpoint（SHA256 `b9584aa54883d231e7a64ec5a305eeab950f450de7f4a3d4e866128b786f2130`），包括Student、Teacher、center、optimizer和两rank RNG，继续原4轮1000更新日程；重做未提交的epoch2前16步。新控制器和attempt单独版本化，不覆盖旧队列或原launch。F1/N1未启动，仍使用原科学配置并解析新ID。三组十步预检原证据可重用，因为科学源码/配置均未改变；只验收恢复执行本身。GM1绑定新恢复队列的真终态，不绑定失败队列。

恢复控制器首次发布`3102fcd40a04980e22339349973961893d5791d0`服务器CPU隔离复核289通过（60.63秒）。其新恢复队列`v2-ffn-recovery-3102fcd-20261011T123239Z`在无BR子进程时被新ScButterfly PID2337671、UID1003（GPU0 720MiB）拒绝；此进程不属于原pinned白名单。没有BR新更新、没有GM1模型预检，旧队列与备份保留。后续仅改控制器无活动子任务时的资源等待，保留原白名单、内存预算和实际训练5秒保护；不让新PID继承旧许可，不触碰他人的进程。真实训练恢复仍由资源门决定。

## 实际新队列

发布`430a1326aeabdb2e326d840295f6b31d2d9bc3c2`，服务器隐藏CUDA检查295通过（60.69秒）。主与Luna核验两控制器/源码/配置/publication/queue/checkpoint备份：`v2-ffn-recovery-430a132-20261011T124207Z`CPU waiting_shared_resources，`v2-graph-mhc-430a132-20261011T124207Z`CPU waiting_dependency，均无GPU子任务。BR仍只提交epoch1，尚无新恢复更新；新ScButterfly PID不继承旧白名单，资源门保持严格。

资源可用后BR用原b37科学源码及完整状态/原4轮日程恢复，再F1/N1原配置新ID；三组真epoch4评估完成后自动GM1十步预检，主验收后fresh4训练及last评估。GM1仅图mHC4（四层各attention/FFN共8routes），Student47,051,014，+196,824（0.42%）；其他方法/损失/数据/批次不变，无新效果或速度结果。

已准备handoff给监督查询对话，每20min；转交后须验证ACK与ACTIVE。完整小收据`.byte-os/evidence/graph-mhc-20261011/graph-mhc-430a132-startup-20261011.json`；机制文档`docs/experiments/GRADPERT_V2_GRAPH_MHC_20261011.md`。旧失败队列全部保留。

