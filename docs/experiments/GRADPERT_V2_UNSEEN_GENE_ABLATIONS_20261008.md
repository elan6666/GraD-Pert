# 未见表达基因：U1–U4 独立机制消融

执行负责人为原主会话 `codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b`；长任务监督使用 `codex:01a0df0b-4142-7df1-86c0-d959471d80a1`。实现基于已发布 `47c36faf08505898aa7d6ef2bcf65be387673db8`；其模型、训练和 v2 测试代码与旧 L0 的 `ab022caa57a3b45dc5a14c6ae38bc280702112e4` 相同。新代码发布 SHA、容量和正式结果另见对应运行收据，不用当前 HEAD 倒填旧实验。

## 2026-10-09 U4 修订（覆盖旧三组排程）

用户授权新增独立 U4，并要求 U1→U2→U4→U3。U1 的旧六轮结果、U2 的活动六轮运行及其不可变源码/config 原样保留；暂停并退役旧 controller，未启动的旧 U3 正式 run 不再启动。新 follow-up queue 等待原 U2 的完整六轮末轮测试终态，再用新发布源码和新运行 ID 执行 U4→U3。U1/U2 不重跑，所有比较逐组暴露源码版本。

U4：`learned_genept_projection=true`，其余三个机制关闭。冻结原始 GenePT `v_g∈R2048`，学习 `E_g=W v_g+b∈R256`；W 随机 Xavier uniform 初始化、b=0，无 PCA、无均值拟合、无重建目标、无独立可训练 gene-ID 表。投影后沿用既有 LayerNorm 与图层。原始先验源 `/data/yilangliu/DinoGenePT/data/embeddings/seed-go-protein-pathway-master-aligned.npz`，SHA256 `34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`；按既定图轴选择6506×2048，轴 SHA256 `c1d0c76822d5c7ac1f561457010065fa3a195d27954f3c959d6732d107124e10`，矩阵 SHA256 `f02af4fb230ae4ab5fe5c2333059f95c668b11493ae70d72625d6514e6fa0937`，零缺失/零空向量。不会进入 PCA 预处理。

共享 Linear 参数524,544。Student 总参数47,852,590，其中可训练34,528,302、冻结原始先验13,324,288；Teacher 同构。公共骨干初始化与随机流通过合成测试保持相同，但 U4 初始基因表示与 PCA 基线不同：属于方法消融，不是等价提速，也不预先断言泛化改善。固定表随 checkpoint 保存，原始科学先验与 checkpoint 留服务器。

新自包含设计：`configs/v2/cap40_unseen_u4_20261009/manifest.json`；其中 U1/U2 配置仅完整描述设计，不用于重跑或倒填旧运行。`generate_unseen_group.py --with-u4` 生成四组；`run_unseen_followup.py` 只计划新 U4、U3，绑定旧 U2 的完整 launch/config/source 与进程身份，等待并核验完整终态；缺失 PID 不能称成功。复用原 controller、训练、测试入口，不另写模型训练主函数。

新 U4/U3 均先10更新+checkpoint恢复+300-control推理，同一global272；全部新短程门通过才正式运行U4再U3。6epoch、cap40、不验证、末轮last测试以及所有损失、随机顺序、图、表达剔除与指标协议保持固定。失败保留证据并交回主，不静默调整batch或重启。现阶段已实现并进入验证，正式结果仍以各run收据为准。

## 原三组定义（历史配置保持不变）

| 实验 | prior_shared_adapter | gene_conditioned_readout | direct_target_flag |
|---|---|---|---|
| 已完成 L0 参照 | false | false | false |
| U1 | true | false | false |
| U2 | false | true | false |
| U3 | false | false | true |

只启动三个独立新组，顺序 U1→U2→U3，各使用双卡。不启动三项组合、额外 U0 重跑、已取消 L2/L3/M1/M2 或 M3。旧 L0 是跨源码版本参照，必须明示版本区别；关闭所有开关的初始化、输出、RNG、三次非零学习率完整 optimizer/Teacher/center 更新已与改动前快照核对。CPU 短轨迹一致不能证明双卡长期训练逐位相同，也不是新方法效应已验证。

### U1：固定先验与共享适配器

既定 GenePT 确定性降维得到 E0（6506×256），不使用表达数据：

`E_g = E0_g + Aθ(E0_g)`。

E0 作为原 embedding 的冻结参数保存，排除 optimizer 路由；不存在额外每基因可训练残差。A 是 256→64→256 GELU MLP，末层权重和 bias 零初始化。随后沿用原图归一化和合法邻域图编码。图、Cell、Response 的公共宽度均保持256。合成小模型使用同比例宽度/4瓶颈。

### U2：先验条件化读出

原图基因表保持可训练；独立固定 E0 buffer 只供读出，不参与梯度。φ是原预测头的 Linear→GELU：

`Δ_g = (w0 + Hθ(E0_g))ᵀ φ([r_g; ep]) + b0`。

实现为原最后线性层结果加 `sum(φ * H(E0_g))`，保留零修正时的基准输出。H是256→64→256 GELU MLP，末层零初始化。不加入 value 门控、额外bias或正则。由显式 graph-axis gene ID 查询 E0，不能从 token 位置或浮点嵌入推断身份。

### U3：最终预测头的直接靶点标志

`t_pg = 1[g∈T_p]`，`Δ_g = D([r_g; ep; t_pg])`。

输入512→513；额外列零初始化，已有列/bias保留相同初始化，隐藏层仍256。多靶点和padding按身份/valid mask处理。标志只在最终预测头出现，不改变Cell、query采样、图边或任何表达可见性；不假定靶点响应方向。

所有新增模块在公共骨干初始化结束后使用独立 RNG 流。启用功能不消耗公共训练/视图随机流。各Teacher同构，新增可训练参数沿用原optimizer之后的EMA；固定底座/副本始终固定。checkpoint architecture记录开启项并拒绝不同架构的完整状态恢复；关闭项省略，兼容旧checkpoint。

## 固定协议

- Jurkat cap40，原selection/split/300-control有序manifest；seed1，从头6 epoch，无验证、无早停，best=null，末轮last冻结测试。
- global272 = 每卡micro68×累积2×双卡。四个16384原型头、mHC4、完整预测+SSL1+SSL2。
- 图3单向KDA+1来源感知稀疏MLA；Cell2KDA+1MLA；Response2self/crossKDA+1self/crossMLA；逐层ep，随机顺序，所有query读取最终S。
- 预测row-mean MSE；λ1=λ2=1；SSL1 condition/node=1/1，spread=0；SSL2 DINO/iBOT=1/1，KoLeo=0；辅助control重建关闭。
- GLM5MuonSplit_v2，原project warmup+cosine与LR1e-3→2e-4、weight_decay0；Teacher/center时点不变。
- 225个测试扰动靶基因表达仍从训练输入与预测监督排除；身份和先验可见；测试control表达恢复。总体5000、见过表达4775、未见表达225分别输出三种Pearson的all与统一DEG版本。
- 新机制不是数学等价性能优化，不预先断言对未见基因有效。直接靶点常只有一个，不在单条件上强算单基因Pearson。

## 参数实测（Student）

| 模型 | 总参数 | 可训练参数 | 额外固定先验buffer元素 |
|---|---:|---:|---:|
| L0 | 35,669,294 | 35,669,294 | 0 |
| U1 | 35,702,382 | 34,036,846 | 0（原表冻结） |
| U2 | 35,702,382 | 35,702,382 | 1,665,536 |
| U3 | 35,669,550 | 35,669,550 | 0 |

每组Teacher持有同构副本，均不训练。参数数字由真实配置、6506×256合成种子表实例化计数，不包含优化器状态；正式capacity收据再核对真实轴和显存。

## 启动与验收

自包含配置位于 `configs/v2/cap40_unseen_20261008/`，manifest逐项固定父配置SHA、单变量开关、epoch和batch。`generate_unseen_group.py`只生成这三项；`run_unseen_group.py`校验manifest和已完成L0，再复用已有capacity/training controller，不另写训练主函数。

2026-10-09用户修订：每组默认用同一发布源码、同一配置做10次双卡完整更新、固定train诊断、checkpoint恢复及300-control推理检查。所有短程预检门通过后才按U1→U2→U3正式训练，使用新run ID、不可变servercheckout和allocator `expandable_segments:True`。OOM/数值失败封存证据，停止队列并交回主；不私自只给某组降batch，不恢复旧取消队列。真实速度/显存和实验指标以收据为准。

短程预检队列与正式队列由指定监督每20分钟检查并汇报，终态交回主验收后暂停监控。仅定时检查被验证，未验证的外部事件通知不称为故障发生瞬间唤醒。

## 机制来源与边界

[scPRINT](https://www.nature.com/articles/s41467-025-58699-1) 提供保留生物基因先验的相关动机。[DeepSpot-M原始预印本](https://www.medrxiv.org/content/10.64898/2026.06.19.26356060v1.full) 与[官方实现](https://github.com/ratschlab/DeepSpotM)使用基因条件化输出权重的相关思路；其输入是组织图像，不能当作本扰动任务的效果证据。这里的方程是GraD-Pert原生适配，不导入上游代码，不声称复现该模型或创造这些通用机制。

10步收据标为preflight_only，包含checkpoint恢复和300-control推理，不宣称最大batch或长期稳定性。旧da310a7的U1预检在56/128时按用户指令停止，恢复/推理未完成，不能改记为10步通过；已保存的训练证据保留，但完整短程预检需要在新版本补齐。
