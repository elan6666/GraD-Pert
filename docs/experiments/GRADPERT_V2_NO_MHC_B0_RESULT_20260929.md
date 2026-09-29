# GraD-Pert v2 no-mHC B0：三轮训练和测试结果（2026-09-29）

这次 Jurkat B0 使用普通单流残差路由（`streams=1`，关闭 mHC），保留预测损失、SSL1 和 SSL2，按**验证集完整 joint loss** 选择 best。训练完成 3/3 epoch、1302 次优化器更新，退出码 0；best 与 last 均完成独立测试。Systema 的参考向量规则沿用既定实现，未在本次修改。训练期剔除测试扰动靶基因的表达输入与表达监督，测试时使用它们的 control 表达。

## 身份和验收

- Run ID：`nadig_jurkat-seed1-20260928T184412Z-2d6425f1e28e46b0b285af6096d03bd3`，attempt 1；服务器根：`/data/yilangliu/GraD-Pert/runs-v2-no-mhc-ca7884e/` 加该 ID。
- 训练与评估源码：干净 Git 提交 `ca7884e4e9b70bb55a61d97442467ff2531337b3`；配置 SHA256：`4c5ba9b39689b9d7993e023694cdedd0268668aa0f9e2f922e6c82e5cb25887e`；runtime SHA256：`62b19c5701981d5039810ac196f37c5a3d542da476d695659910074d479e651e`。
- 双 RTX 5090；每卡微批 74，梯度累积 2，等效全局 batch 296；seed 1。两个 checkpoint 与评估使用相同的测试 split、300-control 清单和参考向量哈希。
- 主会话独立读取服务器原始 `COMPLETE.json`、`fit/epoch_state.json`、三条 `fit/history.json` 记录及 `fit/best-test.json`/`fit/last-test.json`。三个终态 JSON 的 SHA256 分别为 `ea2a9ce95d3313003ec961db66aff1471ef9136296e933a1ac486c47400f6c12`、`bf4a86b998483e7533e56debf43a3879d43c17c04342a0db8d9fced6878ef2a3`、`f90c189f5e1fb04f9728a6fb1a7aa7839a14d9b7bbf4bbd103b5a3bc0b5d716a`，与监督终态收据一致。整个成功 run 根递归 PKL 数为 0。

## 验证和测试

| Epoch | 训练 joint loss | 验证 joint loss | 验证预测 loss | 角色 |
|---:|---:|---:|---:|---|
| 1 | 10.808477 | 4.800583 | 0.003915 | — |
| 2 | 0.526505 | **4.007639** | 0.003006 | **best** |
| 3 | 0.321231 | 4.051964 | **0.002812** | last |

best checkpoint `epoch-0002.pt` SHA256 为 `08e931b0e69e45805193a5fdc08297cdde4113876520d9df66cc5ddc0926611d`；last `epoch-0003.pt` SHA256 为 `5daef1bac40c19a20ba4ee54a2637e1f187e0d1f502010ba06708fc45a303978`。以下是测试扰动条件的 Pearson 宏平均；全基因指标 592/592 个条件有效，统一 DEG 指标 590/592 有效（另 2 个条件真实细胞各仅 1 个，无法进行 DEG 检验）。DEG 使用同一套目标基因不排除的 Top20 列表。

| 测试 Pearson | best：全基因 | best：DEG | last：全基因 | last：DEG |
|---|---:|---:|---:|---:|
| TxPert | 0.194231 | 0.351815 | 0.205355 | 0.383110 |
| TriShift | 0.144541 | 0.323934 | 0.157874 | 0.363699 |
| Systema | 0.073219 | 0.193800 | 0.074393 | 0.241011 |

按训练更新中**实际出现过数值表达**划分，见过表达 4,775 个基因，未见表达 225 个基因。下面每格依次是全基因 / DEG Pearson：

| 子集 | TxPert best | TxPert last | TriShift best | TriShift last | Systema best | Systema last |
|---|---:|---:|---:|---:|---:|---:|
| 见过表达 | 0.367829 / 0.436403 | 0.417571 / 0.491709 | 0.255445 / 0.400113 | 0.301251 / 0.470844 | 0.196816 / 0.221620 | 0.226705 / 0.307539 |
| 未见表达 | 0.138894 / 0.160998 | 0.147895 / 0.168010 | 0.131143 / 0.125876 | 0.141475 / 0.147179 | -0.021998 / 0.026212 | -0.020832 / 0.048288 |

未见表达子集的全基因指标仍有 592/592 个条件有效，但 DEG 指标仅 147/592 有效：许多条件的 Top20 DEG 与这 225 个基因交集不足两个，不能计算 Pearson。因此 DEG 分组列不能直接按同一条件总体与全基因列比较。

## 解读边界

last 的六个总体测试 Pearson 都高于 best，但 **best 必须保持为预设验证 joint loss 最低的 epoch 2**，不能因测试结果改选 epoch。未见表达子集明显更难，尤其 Systema 全基因 Pearson 接近零或为负；这是该运行的描述性结果，不能单凭一个 seed 判定机制无效。关闭 mHC、改用 joint 验证以及评估协议升级在这次新版 B0 中同时存在；本次不能单独归因 mHC 的效果，也不能与旧协议结果直接作因果对比。下一步若要判断结构贡献，需要同源码、同拆分、同训练与评估协议的配对消融；本次未启动其他组。

监督终态收据：`.byte-os/coordination/receipts/nadig_jurkat-seed1-20260928T184412Z-2d6425f1e28e46b0b285af6096d03bd3-terminal-20260929T1032Z.json`。服务器原始 `COMPLETE.json`、best/last 测试 JSON 和 checkpoint 是最终可复核依据；监督会话已交回主会话并暂停这次监控。
