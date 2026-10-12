## 2026-10-13：四轮官方基线发布、真实测试及新依赖队列

用户预算更新覆盖旧official2241等待队列：已精确停止controller/wrapper，未启动外部GPU。新source43ccca4fd14dfbdf451d5673da3f5d1b96df833f，163localpassed/1历史skip、Ruff；四actualofficialCPU10/restore/zeroPKL及三canonical审计通过（官方alias清单遗漏SHA一次失败保留，独立attempt2修复）。自有GenePTraw2048全2372，CCDC130存在；project与GEARS数据完全一致；ScELMo候选只减40训练行且592test完全保留。

新CPU等待queue official-four-queue-43ccca4-20261013，SHA56aae90f62de7aa26cb92987080fd48e88691aecab52fc17972194c8cdd89fe7，controller2674076/start723504303/cmd8d3b5fcc5f5859d8dc8923884043e43b43ea821b59c061ce9835998c986771bc，已实核waiting_dependency。顺序projectScouter→scDFM→GEARS，4epochlast统一18指标，scDFM4ceil47836/48=3988updates（等效sample预算）。父F1/N1/GM1真科学完成后自动GPU10gate+formal+last评估；不修改父/旧source/config。官方GenePT对照来源仍待澄清：此前单缺CCDC130的是ScELMo，原始GenePT缺17；该arm未派发，不静默删更多。细节handoffs/2026-10-13-official-four-dependent-supervise.md，small evidence official-four-20261013。

## 2026-10-13：用户更新为四轮及两个 Scouter 先验版本（覆盖旧预算）

本轮预算统一4 epoch、last-only；模型、官方优化器/LR等设置保留。scDFM沿用条件/群体抽样，一轮定义ceil(Ntrain_pert/48)，当前47836行→997步/轮、总3988步；不是逐细胞遍历，cosine终点为新3988步。Scouter4轮/patience5，GEARS4轮。官方内部验证保留，公共test只last。

自有GenePT-Seed raw文件seed-go-protein-pathway-master-aligned.npz，SHA34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318，17730×2048，doubao-embedding-vision；实际全2372扰动靶基因覆盖，含CCDC130，全部有限。Scouter用整份项目原始先验替换，不用PCA/训练后256表，不混合1536/2048空间；官方架构输入prior维度随数据变化，hidden配置不变。

此前仅缺CCDC130/YJU2B的是官方Scouter使用的scELMo1536先验；现有独立原始GenePT文件经已审核aliases仍缺至少17个靶基因。用户授权忽略一个缺失条件，但先验来源已提请澄清；scouter_official配置仅支持训练CCDC130+ctrl的显式移除，暂不派发，不能把scELMo标成原始GenePT，也不额外删除16个条件。CCDC130属于train，完整592测试与DEG/Systema/300-control引用不变。原225测试表达剔除不因train遗漏改变。

旧官方queue2241c46已由监督仅终止等待controller，无GPUchild；stop收据coordination/receipts/BR400-official-expression-protocol-stop-20261012T1846Z.json，SHA7b07f69c832f9b3f81651414f412d0ae7f11f3168eef3602719962d1a66fcde1。旧source/config/run证据保留，不恢复200000步/20轮。父F1/N1/GM1与原资源策略不改。

完成本地163passed/1历史checkout跳过、Ruff/format；准备发布新版本、4组实际CPU10/restore和真实canonical检查；GPU仍等父真科学终态后预检。新queue优先自有Scouter→scDFM→GEARS，官方对照在先验来源澄清后追加独立ID/配置。此阶段不等于训练或科学完成。

## 2026-10-13：官方 Scouter/scDFM/GEARS 表达剔除适配验收与依赖队列

三组官方适配/config完成，140本地测试通过、1旧官方checkout测试跳过，Ruff通过；服务器真实官方包均10次优化/恢复预测一致/零PKL。同源2241c46ed3d320b3e194a1f7ba989637b4587a24两数值环境的真实数据合同一致：cap40训练47836/验证41235/control12013，4775可见表达/225剔除表达，excluded非零数0；仅官方训练数据对象，不含test扰动细胞。冻结300controls与DEG/Systema参考完全匹配历史BR。

后续官方queue已实际启动CPU等待：official-expression-queue-2241c46-20261013，SHA049e6fcf8d8277c19fb8c1bddb7f1cb9fa52b6c7e9ffdfce8764a16aff186bd0；仅父F1→N1→GM1真终态成功才GPU10-update预检，再scDFM→GEARS官方训练+18项last评估。Scouter47aliases已审核，唯一缺CCDC130/YJU2B官方向量，正式运行blocked，不删除条件或补零；用户额外先验授权待答。没有新的GPU训练/科学结果，当前原生队列与源码不改。现有监督会话保持唯一20min监控，下一阶段handoff已接纳，详情独立official-baselines-build.json及报告。下面为历史状态。


Main build scope independent of old native supervision; published code source2241c46.
Full runtime budget: official Scouter up to40epochs/patience5, scDFM200000updates, GEARS20epochs.
Native v1/model code unchanged; current GPU jobs preserved.
Report:docs/experiments/GRADPERT_V2_OFFICIAL_EXPRESSION_BASELINES_20261013.md
