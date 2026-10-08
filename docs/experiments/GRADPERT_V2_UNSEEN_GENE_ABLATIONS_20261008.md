# 未见表达基因：U1–U3 独立机制消融

执行负责人为原主会话 `codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b`；长任务监督使用 `codex:01a0df0b-4142-7df1-86c0-d959471d80a1`。实现基于已发布 `47c36faf08505898aa7d6ef2bcf65be387673db8`；其模型、训练和 v2 测试代码与旧 L0 的 `ab022caa57a3b45dc5a14c6ae38bc280702112e4` 相同。新代码发布 SHA、容量和正式结果另见对应运行收据，不用当前 HEAD 倒填旧实验。

## 三个开关

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

每组先用同一发布源码、同一配置做128次双卡完整更新、固定train诊断、checkpoint恢复及300-control推理检查。所有容量门通过后才按U1→U2→U3正式训练，使用新run ID、不可变servercheckout和allocator `expandable_segments:True`。OOM/数值失败封存证据，停止队列并交回主；不私自只给某组降batch，不恢复旧取消队列。真实速度/显存和实验指标以收据为准。

长capacity队列与正式队列由指定监督每20分钟检查并汇报，终态交回主验收后暂停监控。仅定时检查被验证，未验证的外部事件通知不称为故障发生瞬间唤醒。

## 机制来源与边界

[scPRINT](https://www.nature.com/articles/s41467-025-58699-1) 提供保留生物基因先验的相关动机。[DeepSpot-M原始预印本](https://www.medrxiv.org/content/10.64898/2026.06.19.26356060v1.full) 与[官方实现](https://github.com/ratschlab/DeepSpotM)使用基因条件化输出权重的相关思路；其输入是组织图像，不能当作本扰动任务的效果证据。这里的方程是GraD-Pert原生适配，不导入上游代码，不声称复现该模型或创造这些通用机制。
