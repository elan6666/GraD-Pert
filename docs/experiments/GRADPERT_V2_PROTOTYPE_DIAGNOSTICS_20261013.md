# BR400／F1／N1／GM1 原型与熵诊断（2026-10-13）

结论：N1 的低交叉熵主要来自 Teacher 目标熵下降；本探针不支持所有样本集中到同一原型的坍塌。BR400/GM1 的 Teacher 较接近均匀分布，Student 已拟合这些高熵目标。F1 尤其图 condition 头具有很强的近均匀输出与表征退化迹象，不能把它扩大成整个预测模型坍塌。

## 口径和不可变证据

每组第4轮 last checkpoint；原训练源码与自包含配置严格匹配。相同4批×16训练细胞、1,000查询表达基因、完整图和真实视图配方；eval/BF16、不启用dropout；不更新optimizer、Teacher或center。输入哈希覆盖细胞/control行、查询位置、各图视图节点集合及表达视图位置/遮蔽；图mask与边dropout按相同私有种子/配方生成，但未另存其逐边哈希。四组 ordered input SHA 相同：`18be00133a764435abec130b4674c17706bea3226a7cf13df5ea9363b897304e`。每头 Student读取全部视图，Teacher读取两个Global；node按实际有效遮蔽位置。分布统计混合其有效调用样本，不声称等于条件等权或完整epoch loss；KL表按实际调用的sample-pair数量平均，不是历史训练row权重重放。

工具发布 SHA：`285a1fe1de9cdf4bc7d925677cefbadb8855d6ae`。工具统计不改原始模型计算；5项统计不变量测试通过。四组全部exit0；checkpoint/Student/Teacher/center逐张量状态指纹及checkpoint文件哈希不变。峰值显存约2.0–2.1GiB。完整小聚合收据SHA256：`354354b0ef8446f56c153eabc461e830cfb526b7d47f3c35d186c358c1a8c4d1`。所有科学矩阵和checkpoint留服务器。

服务器根：`/data/yilangliu/GraD-Pert/development/prototype-br400-285a1fe-20261013T051929Z`。

## 公式与读法

`p_s=softmax(logits_s/0.1)`；`p_t=softmax((logits_t-center)/0.04)`。

`CE=H(p_t)+KL(p_t||p_s)`；`H(p)=-sum(p log p)`；`JS=H(mean p)-mean H(p)`（等权经验样本的广义Jensen-Shannon差异）。所有熵单位nats。16384原型均匀熵为9.7040605。

高H、低JS：不同输入给出近乎均匀分布；低H、低JS：不同输入给出同一个尖峰/固定分布；低H、高JS：个别样本置信高，跨样本仍有差异。小KL只证明Student跟随Teacher，不能单独证明无坍塌或生物学效用。原型argmax数量在近均匀时受微小数值波动影响，必须结合概率与JS。

## 各头分解

| 实验 | 头 | CE | Teacher H | KL | Teacher JS | Teacher平均最大概率 |
|---|---|---:|---:|---:|---:|---:|
| BR400 | 蒸馏① condition | 9.64523 | 9.64425 | 0.000979 | 0.058578 | 0.0262% |
| BR400 | 蒸馏① node | 9.64112 | 9.63333 | 0.007788 | 0.069071 | 0.0366% |
| BR400 | 蒸馏② DINO | 9.68321 | 9.68316 | 0.000050 | 0.020626 | 0.0136% |
| BR400 | 蒸馏② iBOT | 9.68318 | 9.68273 | 0.000447 | 0.020863 | 0.0135% |
| F1 | 蒸馏① condition | 9.70405 | 9.70404 | 0.000007 | 0.000016 | 0.0062% |
| F1 | 蒸馏① node | 9.70342 | 9.70320 | 0.000220 | 0.000863 | 0.0070% |
| F1 | 蒸馏② DINO | 9.70024 | 9.70006 | 0.000177 | 0.003661 | 0.0082% |
| F1 | 蒸馏② iBOT | 9.69293 | 9.68309 | 0.009845 | 0.020172 | 0.0133% |
| N1 | 蒸馏① condition | 0.47115 | 0.45405 | 0.017098 | 4.148085 | 85.4697% |
| N1 | 蒸馏① node | 6.59383 | 6.39210 | 0.201727 | 2.463891 | 11.9013% |
| N1 | 蒸馏② DINO | 0.47723 | 0.46345 | 0.013774 | 4.026081 | 84.8790% |
| N1 | 蒸馏② iBOT | 0.68074 | 0.65799 | 0.022751 | 3.774459 | 78.0406% |
| GM1 | 蒸馏① condition | 9.60440 | 9.60327 | 0.001133 | 0.096402 | 0.0390% |
| GM1 | 蒸馏① node | 9.62215 | 9.61363 | 0.008517 | 0.086550 | 0.0484% |
| GM1 | 蒸馏② DINO | 9.66759 | 9.66754 | 0.000050 | 0.035557 | 0.0243% |
| GM1 | 蒸馏② iBOT | 9.66905 | 9.66882 | 0.000225 | 0.034235 | 0.0180% |

## 是否坍塌

**N1**：Teacher图condition／DINO／iBOT平均最大概率约85.47%／84.88%／78.04%，对应JS4.148／4.026／3.774；图node JS2.464。condition/DINO有效argmax原型数64/60，最大占比1.56%/4.69%，与全部样本集中一个原型不同。四头Teacher投影瓶颈相对中心化方差为0.872–0.975，协方差有效秩约24–38。有限样本下保留显著输入差异，不是全局常量表示。

**BR400／GM1**：Teacher entropy约9.60–9.68、JS约0.02–0.10；Student KL约0.00005–0.0085。它们并非无法拟合Teacher，而是目标本身很软、差异较弱。四头表示方差与有效秩非零，不宜称整个编码器完全坍塌；接近均匀的原型目标有提供弱监督的风险。

**F1**：图condition Teacher H9.704043、JS0.000016，与均匀熵9.704061几乎相同。图condition Teacher投影瓶颈相对方差0.000154，有效秩2.16；图node相对方差0.002842、秩3.26。明显比BR/GM1退化；支持图蒸馏头的近均匀坍塌/严重退化，而不是只用loss猜测。SSL2 DINO瓶颈秩约1.09，仍有较弱变化；SSL2 iBOT不能据此称完全坍塌。

## 为什么N1不同，以及不能推断什么

N1 在图/Cell/Response FFN的输入Norm及GenePT、表达升维后、扰动融合、预测、四个蒸馏头输入等位置采用RMSNorm；不是只改一个头。代码保持bias及Linear初始化规则。观测与归一化改变尺度、保留跨样本方向差异的机制一致，但尚未用只改head Norm等对照分离因果。不能把跨组差异全部归于某一个Norm位置。

N1低CE主要由Teacher目标熵降低解释，KL甚至大于BR/GM1，因此不能说N1仅因Student更好地拟合Teacher而loss更低。跨样本区分也可能是基因身份、扰动条件或细胞编码，不自动等于正确表达响应；N1历史测试Pearson并未因此最好。

仅诊断第4轮固定64训练细胞及对应图节点，不证明全训练集/所有epoch/多seed的结论。视图间JS混合输入与裁剪变化，不直接衡量生物学语义。原型数16384远多于64细胞，用不到所有argmax原型在该probe中不是坍塌证据。未启动改温度、补正则、改头、重训或其他新消融。

## 原始来源

[DINO原论文§5.3](https://arxiv.org/html/2104.14294v2#S5.SS3)讨论均匀/单维坍塌及CE=entropy+KL；本诊断另用跨样本JS和表征方差/秩排除仅凭小KL判坍塌的误读。

[DINO官方投影头](https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)使用L2瓶颈与weight-normalized原型层；当前GraD-Pert原型层是无bias的普通Linear。这是已核对的差别，尚不能断言它导致本次退化，未做方法修改。

## 训练来源

- BR400: training/method SHA `6a8c612bd7733d2f03b5c041aeff63c5f6ec8339`；checkpoint SHA `85ca4a88f392dc29cca651eab5801e252d96d823bbe3209b15498842c6671975`；config SHA `d7877c03839645901f930817d5258638ca8738c650bd23fc42e5da33415e6f50`；diagnostic receipt SHA `2d3a672a9126f7d5bc27d4ee7a672800c943351eea15b99d8275073bcee1cdd5`。
- F1: training/method SHA `4bfa1f13b95c73b482954e4efd396cfc82cc6afe`；checkpoint SHA `559994939dd394154449ae95019a5ac4261608c34b641c10f2c3a6872bd13aac`；config SHA `0d0074b831cf97564b1b06a1f60527051f33541e10a648c34721c9fed784d9c3`；diagnostic receipt SHA `5e478675f853878353d74a84dad92a8034c01518c0049ddfa56b9d72004f3620`。
- N1: training/method SHA `4bfa1f13b95c73b482954e4efd396cfc82cc6afe`；checkpoint SHA `8652674e7e323eb65d7db66baf1baf0b7bde912baf37d82edc02b59658c208fa`；config SHA `de7f7a4e1c4bfdf6c91ef8a0bed77bec74ca17886728b9de32b527b5ae6e0e8f`；diagnostic receipt SHA `3f73347b5ae962452c5d3facc239503f7736ec34f07d06155edb28d025f7fd73`。
- GM1: training/method SHA `4bfa1f13b95c73b482954e4efd396cfc82cc6afe`；checkpoint SHA `e8ed8d8b382a77a9bd02a583df8942449f057d1b54f8355b895e0b2940d32828`；config SHA `98c7ab0bc5d8d55a3835a9b4d7faa78ba8aeb048b732c508f4a80cc5bfb811b5`；diagnostic receipt SHA `32cce077f29e1a336defc19c4f144fe73df59871d1bc100807033fd4fb90e9c8`。
