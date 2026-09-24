# GraD-Pert v2 新方法：Hamiltonian 图传播与全量 MLA

更新：2026-09-25。新运行配置为 `configs/v2/hamiltonian_mla_jurkat/gradpert_v2/nadig_jurkat.yaml`。历史 B0 的源码、配置和结果不追溯修改。“TxPert 式 expander”仅指固定随机双向 Hamiltonian 环拓扑，不等于其完整 Exphormer 或训练目标。参照的冻结上游证据为 [TxPert 的环生成与图边并集实现](https://github.com/valence-labs/TxPert/blob/08d82eea86746b044cf7531f4ec8c5f60e1cb73f/gspp/models/pert_models/exphormer.py) 和 [Exphormer 配置](https://github.com/valence-labs/TxPert/blob/08d82eea86746b044cf7531f4ec8c5f60e1cb73f/configs/config-exphormer-mg.yaml)；本项目独立实现，且使用项目种子 1，不声称与上游默认随机图逐边相同。

## 输入与图

固定全图顺序含 6,506 个基因。GenePT 向量经确定性 PCA 初始化可训练矩阵 \(E\in\mathbb R^{6506\times256}\)，随后 \(M=\mathrm{LN}(E)\)。表达分支每行抽样 1,000 个基因；训练抽样池默认排除测试扰动靶基因的**表达**，但保留其图节点和 GenePT 身份。

对每个节点保留 GO 和 STRING 各自的入边 Top20。固定种子 1 抽取 3 个随机 Hamiltonian 环，环上相邻基因连成双向边；无重边时每节点从 expander 至多得到 6 个邻居。再加入 self 边。四种来源取并集，同一条边只保留一次，并携带 \(s_{ij}\in\{0,1\}^4\) 来源标志。因此每行邻居数至多 \(20+20+6+1=47\)。局部视图使用诱导子图；随机边 dropout 不删除 self 边。

## 两层逐层图传播

图宽度 \(d=256\)，4 头，每头 64 维。单层 sparse read 对查询基因 \(i\) 和邻居 \(j\) 计算

\[
q_i^h=W_Q^h\mathrm{LN}(x_i),\quad k_j^h=W_K^hm_j,\quad v_j^h=W_V^hm_j,
\quad a_{ij}^h=\mathrm{softmax}_{j\in N(i)}\bigl(q_i^h\cdot k_j^h/\sqrt{64}+s_{ij}^{\top}b^h\bigr).
\]

令 \(r_i=W_O[\sum_j a_{ij}^1v_j^1;\ldots;\sum_j a_{ij}^4v_j^4]\)，则 \(y_i=x_i+\mathrm{Dropout}(r_i)\)，\(G(x_i,m)=y_i+\mathrm{Dropout}(\mathrm{FFN}_{4d,\mathrm{GELU}}(\mathrm{LN}(y_i)))\)。新模式先算 \(H^{(1)}=G_1(M,M)\)，再算 \(H^{(2)}=G_2(H^{(1)},H^{(1)})\)：第二层确实读取更新过的邻居。实际执行只为目标基因 \(S\) 及一跳邻居 \(S\cup N(S)\) 计算第一层，再为 \(S\) 计算第二层；在同一视图和边采样下，这与全图同步传播两层后截取 \(S\) 数值相同。旧 `static` 模式仍可复现历史配置。

## 控制细胞与响应

表达映射 \(\phi(x)=\mathrm{LN}(W_2\mathrm{GELU}(W_1x))\)，控制 token **相加**为 \(u_{cg}=H_g^{(2)}+\phi(x_{cg})\)。表达遮蔽时在 token 交互前用独立可训练 mask 向量替换 \(\phi(x_{cg})\)。末尾附可训练 control CLS；Cell Encoder 得到基因 token \(b_{cg}\) 和控制 CLS \(s_c\)。扰动 \(p\) 的靶点表示是 \(e_p=|T_p|^{-1}\sum_{g\in T_p}H_g^{(2)}\)。响应输入是 \(v_{cpg}=W_F[b_{cg};e_p]\)，末尾附可训练 response CLS；Response Encoder 得到基因 token \(r_{cpg}\) 和响应 CLS \(z_{cp}\)。预测以原始控制表达为残差：\(\hat y_{cpg}=x_{cg}+W_{D2}\mathrm{GELU}(W_{D1}r_{cpg})\)。当前 \(s_c\) 被计算但不直接输入响应分支；SSL2 的 CLS 是 \(z_{cp}\)，不是 \(s_c\)。

Cell 和 Response **各 3 个注意力+FFN 块**：2 个 KDA 后接 1 个**无因果遮罩的全量 MLA**。宽度 256、4 头、KV 共享压缩秩 64、4 个 residual streams、SwiGLU FFN、dropout 0.1；没有 RoPE。末层 MLA 对当前视图内全部 token 做注意力，不使用 DSA/Top500 内容索引。图 sparse read 与 token 编码器 MLA 是不同算子。

## 目标函数与教师

主视图的真实表达监督是每细胞先跨预测基因平均 MSE，再在双卡有效全局 batch 中按行平均。图/表达蒸馏视图不额外承担真实表达 MSE。总损失：

\[
L=L_{\rm pred}+1.0(0.8L_{1,\rm condition}+0.4L_{1,\rm node}+0.1L_{1,\rm spread})+0.1(0.8L_{2,\rm DINO}+0.4L_{2,\rm iBOT}+0.1L_{2,\rm KoLeo}).
\]

SSL1 有 2 个全局图视图和 4 个局部诱导图视图：学生图节点按视图遮蔽，教师不遮蔽；condition 为跨视图条件表示蒸馏，node 为遮蔽图节点蒸馏，spread 为去重条件的最近邻扩散。SSL2 有 2 个全局和 4 个局部表达视图：DINO 蒸馏响应 CLS，iBOT 蒸馏有效遮蔽 token，KoLeo 对响应 CLS 找近邻。4 个投影头均为 \(256\to2048\to256\to8192\)（prototype 层无偏置）。教师同构、EMA 动量 0.99→1.0；center 独立统计更新。默认 `row_mean`：细胞行等权，双卡联合计算条件计数和 KoLeo 邻域；iBOT 先按每个细胞的有效遮蔽 token 平均。SSL1 node 按有效图节点、spread 按去重条件、center 更新是统一归约开关的例外。

## 冻结参数和运行边界

| 参数 | 新配置 |
|---|---:|
| GenePT 表 / 总基因数 | 6506×256 可训练 / 6506 |
| GO、STRING / expander / self | 各入边 Top20 / 3 个双向环、seed 1 / 每节点 1 |
| 图层 | 2 层传播；4 头；图 FFN 4d GELU |
| Cell、Response | 各 2 KDA + 1 全量 MLA；每块后 1 SwiGLU FFN |
| 宽度、头数、KV 秩、residual streams、dropout | 256、4、64、4、0.1 |
| 投影头 | 4 个；hidden 2048、bottleneck 256、prototypes 8192 |
| Student 参数 | **22,927,109**（以当前代码及 6506×256 表实例化计数） |
| Teacher | 同构 EMA 副本；另有 4 个 center buffer |
| 训练/评估表达上下文 | 1000/1000 基因 |
| 双卡 batch 候选 | 每卡 micro 48 × 累积 2 × 2 卡 = 192 |
| 轮数/seed | 5/1；不早停；验证 MSE 选 best，保留 last |
| 优化器/LR/weight decay | GLM5MuonSplit_v2 / 0.001 / 0 |
| LR 日程 | 项目 warmup+cosine：16% warmup，终点 0.0002 |

**容量边界：** batch192 继承旧紧凑配置，只是候选；图传播与全量 MLA 改变显存和速度，旧架构容量收据不能批准新架构的正式运行。用户已停止旧 B0；本次改动不自动重启训练。

**待审机制问题：** (1) 前 2 个 KDA 仍沿冻结基因顺序因果扫描，移除 DSA 不会使整体编码器排列不变；应做基因顺序敏感性分析。(2) 控制 CLS \(s_c\) 目前不直接影响响应预测，不能宣称其有独立预测作用。(3) 图的双层邻居传播可能抵消去掉 DSA 的速度收益，必须重新持续测双卡吞吐和容量。(4) 新拓扑借鉴 TxPert/Exphormer 的环连接，图算子、编码器和训练损失仍是 GraD-Pert 自身定义。
