# GraD-Pert v2 no-mHC B0：3＋3 全状态续训终态（2026-09-30）

三轮 Jurkat B0 从 `last.pt` 开启独立全状态续训，再训练三轮，累计达到 6/6 epoch。服务器运行退出码为 0，`COMPLETE.json` 与 best/last 测试收据齐全，`FAILURE.json` 不存在，成功运行根下无 PKL。这里的“6 epoch”表示实际三轮训练加三轮新阶段；不是从第零步预设六轮的同一学习率和 Teacher EMA 日程。

## 身份与验收

- 父运行：`nadig_jurkat-seed1-20260928T184412Z-2d6425f1e28e46b0b285af6096d03bd3`，`last.pt` SHA256 `5daef1bac40c19a20ba4ee54a2637e1f187e0d1f502010ba06708fc45a303978`，已完成 3/3 epoch。
- 续训运行：`nadig_jurkat-seed1-20260929T143029Z-d83c2632bba14bddb595f8b8c0734805`，服务器根 `/data/yilangliu/GraD-Pert/runs-v2-continue6-c47f84e/` 加运行 ID；attempt 1。
- 干净训练源码：`c47f84e796aed994fdd060a9afbe14a95fa64069`；自包含配置 SHA256 `66b6abdd0070d111a8275ba85739c6279cd5373559464b62eb0bb2be8473f956`。双 RTX 5090，GPU 0/1，每卡微批 74、累积 2，全局 batch 296，seed 1，每轮 434 次优化。
- 全状态恢复包括 Student、EMA Teacher、蒸馏 centers、优化器和各 rank 随机数。后续阶段不重新 warmup，学习率恒定 `2×10⁻⁴`；训练目标仍是预测＋SSL1＋SSL2。Teacher 动量在新阶段按累计六轮的总步数计算，而父三轮按三轮总步数计算；故它也不同于从头预设六轮。
- 主会话独立读取服务器 `COMPLETE.json`、`fit/epoch_state.json`、`fit/history.json`、best/last 测试收据和退出码文件，核对运行/源码/配置、检查点哈希、累计 6/6、joint-loss 选择、测试 split、六项指标与零 PKL。`COMPLETE.json` SHA256 `6536bb43482e5f825b59196d9a45fb7e7c4406373147948d4f5d1ff4f612a36e`；best/last 测试收据 SHA256 分别为 `74ac43e101dd7619b40ecb63f3ac037c96b1ff2b3d11bfc4829995a6cbf299a2`、`823486bf1278cfeced28b364a5b2e1d184fa8652ca3a6aea5b5cccf7e52c1d3f`。监督终态收据见 `.byte-os/coordination/receipts/v2-continue6-c47f84e-terminal-20260930.json`。

## 验证与测试结果

验证 joint loss 按 epoch 1→6 依次为 `4.800583、4.007639、4.051964、4.137487、4.178619、4.360646`。预注册选择规则选出 **epoch 2 best**（检查点 SHA256 `08e931b0e69e45805193a5fdc08297cdde4113876520d9df66cc5ddc0926611d`），与父三轮运行相同；`last` 为 epoch 6（SHA256 `6a7d9dbac7b9bfb20981adbde881b7d79493b2a7d6535e5d82fcbbd98da418da`）。

| 测试 Pearson | best epoch 2 | 父 last epoch 3 | 续训 last epoch 6 |
| --- | ---: | ---: | ---: |
| TxPert，全部基因 | 0.194231 | 0.205355 | 0.217347 |
| TxPert，DEG | 0.351815 | 0.383110 | 0.372661 |
| TriShift，全部基因 | 0.144541 | 0.157874 | 0.170832 |
| TriShift，DEG | 0.323934 | 0.363699 | 0.334027 |
| Systema，全部基因 | 0.073219 | 0.074393 | 0.093160 |
| Systema，DEG | 0.193800 | 0.241011 | 0.231196 |

三种全部基因指标均有 592/592 个有效测试条件；三种 DEG 指标均为 590/592，另外两个条件的真实扰动细胞数只有 1，无法构造所需 DEG。父 last 与续训 last 同一评测协议，但它们不是独立随机种子重复。第 6 轮全部基因 Pearson 较第 3 轮高，DEG Pearson 较第 3 轮低；同时第 4–6 轮验证 joint loss 均高于第 2 轮。因此本次结果支持“继续训练改变了表现”，不能据测试 Pearson 将第 6 轮改选为 best，也不能把 3＋3 的结果解释为从头六轮训练效果。

本任务的长时监控 `grad-pert-v2` 已在终态暂停；后续如需新实验，使用独立运行 ID 和新的明确协议。
