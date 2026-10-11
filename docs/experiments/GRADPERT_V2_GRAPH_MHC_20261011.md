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
