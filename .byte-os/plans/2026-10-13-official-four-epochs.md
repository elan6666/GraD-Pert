## 2026-10-13：用户更新为四轮及两个 Scouter 先验版本（覆盖旧预算）

本轮预算统一4 epoch、last-only；模型、官方优化器/LR等设置保留。scDFM沿用条件/群体抽样，一轮定义ceil(Ntrain_pert/48)，当前47836行→997步/轮、总3988步；不是逐细胞遍历，cosine终点为新3988步。Scouter4轮/patience5，GEARS4轮。官方内部验证保留，公共test只last。

自有GenePT-Seed raw文件seed-go-protein-pathway-master-aligned.npz，SHA34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318，17730×2048，doubao-embedding-vision；实际全2372扰动靶基因覆盖，含CCDC130，全部有限。Scouter用整份项目原始先验替换，不用PCA/训练后256表，不混合1536/2048空间；官方架构输入prior维度随数据变化，hidden配置不变。

此前仅缺CCDC130/YJU2B的是官方Scouter使用的scELMo1536先验；现有独立原始GenePT文件经已审核aliases仍缺至少17个靶基因。用户授权忽略一个缺失条件，但先验来源已提请澄清；scouter_official配置仅支持训练CCDC130+ctrl的显式移除，暂不派发，不能把scELMo标成原始GenePT，也不额外删除16个条件。CCDC130属于train，完整592测试与DEG/Systema/300-control引用不变。原225测试表达剔除不因train遗漏改变。

旧官方queue2241c46已由监督仅终止等待controller，无GPUchild；stop收据coordination/receipts/BR400-official-expression-protocol-stop-20261012T1846Z.json，SHA7b07f69c832f9b3f81651414f412d0ae7f11f3168eef3602719962d1a66fcde1。旧source/config/run证据保留，不恢复200000步/20轮。父F1/N1/GM1与原资源策略不改。

完成本地163passed/1历史checkout跳过、Ruff/format；准备发布新版本、4组实际CPU10/restore和真实canonical检查；GPU仍等父真科学终态后预检。新queue优先自有Scouter→scDFM→GEARS，官方对照在先验来源澄清后追加独立ID/配置。此阶段不等于训练或科学完成。

