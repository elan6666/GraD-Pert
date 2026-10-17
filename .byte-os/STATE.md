## 2026-10-17：PT1真实CPU报告补齐阶段缺失梯度

0be7a1c报告修复r1仅运行CPU即退出1：SSL阶段没有prediction梯度和prediction/SSL余弦，绘图仍以必有键读取。FAILURE SHA c855ebe1d95ba96493012100ba6b9f6589d75a06d3e6989973b7df94309c21b3、日志SHA389e1facf126ee73ba5cc190b97f46d17b471c19821c539c7ace9ee6be7fac09均保留；未启动GPU seal/probe/medium。改为缺失值断线，不填零；用真实缺失字段结构补回归，training.csv也明确阶段。下一新报告源/独立输出先验收真实CPU整份报告，再继续原中等流程。训练/测试/数值方法源ec32671不变。主owner，短监督Luna已终态，旧定时仍PAUSED。

## 2026-10-17：PT1四轮及last18已核验，中等报告适配修复

主已接回exact main/build并ACK，旧监督与return均PAUSED。PT1 fit四轮600更新及last18已独立核验，last checkpoint SHA520b71f4b5f49139724e7d0639df8676f74a3379727f7d523b88a7e63e668b0f；无PKL。原队列因CPU saved-report的跨阶段input hash校验失败而终态failed，失败root不改、不重训。

原探针epochs0/1/2固定SSL输入、不加载真值；epochs3/4固定MSE输入、读真值且无SSL视图，因此两个阶段不是相同view recipe。新CPU报告器按sealed配置核对阶段顺序、阶段内hash以及cells/query/seed/budget，禁止跨阶段同探针对比；曲线断开、无效MSE/Teacher项N/A。40项本地回归通过，原numerical method/medium脚本与配置不变。新独立修复源发布后执行real CPU报告及原reference/fast门、相同中等清单。训练源仍ec32671，不能将新报告源重标成训练源。

计划PT1_ANALYSIS_REPAIR_20261017.plan.md；完整中等验收后继续已准备profile，再推进DP2→S12-L4及其余已授权消融。下方旧“PT1运行中”均为历史。

## 2026-10-17：完整训练 profile 构建与消融并行推进

用户新增授权为当前完整更新做 CPU/CUDA 归因后再优化，且保持消融推进。主在独立工作树基于已发布 a0acaee6b05959093de2ae645fa54fa6ef3f35f3 仅构建诊断工具；监督会话仍持有 PT1/supervise，20min 监控不改，活动源 ec32671f4c37bc3d1d0f935ddca87fb89a55fa3e 与配置不动。监督已确认记录该独立 build 范围；此条不接管它的训练租约。

计划 `.byte-os/plans/TRAIN_UPDATE_PROFILE_20261017.plan.md`：单独标记 KDA/FFN/原型/CE，组合 checkpoint 前向与重计算上下文；用 fwdbwd flow 桥接 OS 与 autograd 线程 ID，再按 thread+sequence 及 CUDA correlation 做不重复归因。Trace 流式留服务器，未关联项明确 unknown。短采集涵盖 PT1 的 SSL/MSE 和后续 joint 工作负载，profile_only 收据不能充当正式容量验收。本地44项相关回归及22项最终严格零容差定向检查通过；src/configs相对活动ec32671无差异。诊断源17f2194bd03f0c1f4a9cae6ac031c756104c5618已clean发布；服务器GPU隐藏CPU50项通过。BUILD_READY SHAed63bb362099a2e106c409da126ec4c8ae96caec28f2a5b32e9d924766173867；未启动GPU计划SHAfc388457fb91fca9d3d9d8fc4a5fb00e17b061819a3e5d1113816e75e4e0dfad。CUDA采集等PT1完整终态回交，尚无新CUDA归因或训练提速结论。

PT1 全终态回主后插入短 GPU 采集，随后立即继续 DP2→S12-L4→CG1→U24→MR1→P1→C1→O1→K2-fixed-prefix；CPU trace 分析和候选实现与消融训练并行。不得为了优化重训/中断当前 PT1，也不让探针代替最终整步 ABBA 吞吐与完整更新语义验证。永久报告 `experiment-analysis/train-update-profile-20261017/REPORT_CN.md`。下方旧状态为历史。

## 2026-10-17：PT1与代码优化通过工程验收，正式训练已启动

训练源ec32671f4c37bc3d1d0f935ddca87fb89a55fa3e，clean/pushed/immutable；配置SHA7202d4f16e7dd83547294e21144500be3a971b9ffe934b3106b2f44bc9d52ac5。新增PT1两轮SSL后两轮MSE、阶段恢复和同/异条件Global/Local中等诊断；拓扑打包与评估静态图复用、无用SSL decoder/I/O移除。CPU组件12.49倍/1.65倍；双卡单条件eval约1.59倍、输出误差0；整步训练未证明稳定加速。独立GPU state差异原样保留，受控同checkpoint/RNG两rank×2SSL+2MSE的loss/梯度/完整objective/optimizer逐值相同；10步/恢复及两卡eval128原2e-6门再次通过。

正式run nadig_jurkat-seed1-20261017T052036Z-051a3d13c19044a4bbb53d6b3ef02678，主核验epoch1 step2/150、累计2/600、SSL非零LR更新。global320=80×双卡×累积2，4轮、无验证、last18+同中等清单；300步切换、单次动量重置、Teacher/center冻结。工程验收evidence/pt1-20261017/ENGINEERING_ACCEPTANCE.json，永久性能报告experiment-analysis/pt1-staged-20261017/performance-9dc871f/REPORT_CN.md。科学结果未完成。唯一长监督交接与20min激活由权威coordination/state.json维护；下方均为历史，不恢复旧队列。

## 2026-10-17：D0完整交付；官方DINOv2参考亦已完成

D0四轮600更新、epoch4 last18及同中等清单完整；此次仅补分析，无重训。原110a932 failed queue、nullable报告失败和r1导入失败均保留，r2独立目录成功。26项server CPU回归通过；主独立调用完成验证器、重算全部完成输出字节/哈希、确认数值源110a932与report/controller源7ae9a7e938d58e24993d3acf79cb09daa02652df分开。MEDIUM_COMPLETE SHAad2302e6576e792ef0140e33497cf8d1bd16cc95519d8663a8c09a37f2c61a53；1335条件5317行、deep128条件506行、gradient32条件125行；0更新/state unchanged，821个浮点统计maxabs7.958078640513122e-13<=1e-8。

完成前dry-run审查并仅拉45个小图/汇总（5925920 bytes）及验收元数据，逐本地哈希核对；9张模板图与2张比较图已逐张检查。永久中文报告experiment-analysis/no-ssl-d0-20261016/D0_MEDIUM_REVIEW_CN.md，18项测试/有效N/训练曲线与时间见同目录B0_D0_TEST18_CN.md。数据/split/剔除ID/300controls/reference/query/groups精确相同，resolved config仅lambda1/lambda2及来源注释不同。单seed D0 18项Pearson均低于B0；Response CLS秩13.824→4.269，Response混合token秩6.376→3.323，不能仅凭秩宣称质量或因果。D0关闭SSL，因此H/JS/零范数余弦不适用，空图不填0。

官方DINOv2参考分析已交主完成，报告experiment-analysis/dinov2-reference/REPORT_REVIEW_CN.md；公开backbone不含原SSL头/center，不能给原型熵/JS。两个任务无活动后台进程，Luna短监督已终态；旧监督及return fallback保持PAUSED。主已完成当前回交，不启动DP2或其他臂。本条覆盖下面旧“当前运行”叙述，原历史完整保留。

## 2026-10-16：主会话完成官方DINOv2中等参考分析；D0独立监督继续

官方DINOv2任务按用户新指令由主会话执行，独立于D0 relay。分析源码bcb78ddcbe5e581966c35736ab7d2275ef902073，官方7764ea0f912e53c92e82eb78a2a1631e92725fc8，ViT-S/14原权重strict load。服务器15项回归、真实输入probe及正式1335图/10类、deep506×128patch均通过，CPU4线程105.657秒，0optimizer更新/0梯度/state unchanged；COMPLETE SHAeec44aac3195314dea19e76b5cb0831c6b675eeac62ff02eb0d271f112bed522。主独立核验15服务器输出，dry-run后拉取14小图/表/报告+完成收据，3图已读；原图、权重、特征矩阵均留服务器。

完整验证集3925图逐文件ID/bytes/SHA与独立inventory一致；同一S3对象首两个已完成块已含全部validation，不等未使用train尾部，不宣称完整归档CRC。旧partial和慢下载收据保留。最终CLS rank113.071、混合patch rank189.452是视觉参考，不能作为GraD-Pert合格阈值；原型熵/JS因公开权重无原SSL头而不可用。

交付 /Users/elan/code/grad-pert/experiment-analysis/dinov2-reference/REPORT_REVIEW_CN.md，全部图数据及per-run SHA在同目录；MAIN_ACCEPTANCE.json和TASK_STATE.json已完成。此文档提交与分析bcb78dd分别记录。

D0源码110a93212acf71a67383d7334ee39f62edb91b08不受本分析影响，监督查询对话唯一20min ACTIVE已主读回。supervisor ACK a455cf51df488442c506501e98d978223ef3c22b92ac2e131b643b098ba2217e，main return PAUSED；最后208/600为监督15:38 UTC的快照，不是本段重新查询。D0四轮/last18/同中等仍未完成；DP2尚未启动并延后。精确当前训练状态以权威 /Users/elan/.codex/worktrees/v2-gene-exposure-eval/grad-pert/.byte-os/coordination/state.json 为准。主不重复轮询，不启动其他组。本条覆盖下列旧当前状态文字。

## 2026-10-16：DINO两组正式启动，旧VH终态保留

VH已独立验收4轮last及同中等分析；其旧controller因用户改优先级后的租约边界受控退出，不是科学失败，S12未启动。新DINO两组最初439efcb预检因eval leaf缺少精确family白名单在推理前失败；原根/ID/日志/收据保留，模型与配置未改。

修复版本cfe641bc89bcd879cbac5540c74f052bf97d2a4c已push并与独立server checkout一致；仅补精确两组mode/order/10update入口及回归。测试先复现失败、修后本地33通过、服务器CPU116通过；两个10update/restore和四份双GPUeval128/reference128全部通过。主重新运行原生共同数据/controls/source/config门，MAIN_ACCEPTANCE SHA28b2d74f55b1fd038cee73dc2852ab7ac3a3553682e1a183edcce86b98f849a3；PREFLIGHT_COMPLETE SHA461eeb947437bce38471a01adbead7862c8025ef57e5dfb5f978f045b6956e5f。

当前queue /data/yilangliu/GraD-Pert/development/dino-pair-cfe641b-b320-20261016/queue.json SHA75e0fbc72589a7db5af1ef505b4cba8c879219c0f5f63249887a8bd1007ca65e；训练源/data/yilangliu/GraD-Pert/development/source-dino-pair-cfe641b保持不可变。B0-DP→DP2各4轮/global320=m80×2GPU×accum2/seed1/no-validation/last18/同中等，不启动其他臂。B0-DP实际manifest与源码/config匹配，双rank/allocator合规，主核对6/600优化步；这是启动，不是科学完成。原型参数与温度差异见DINO_PAIR_20261016计划，DP2不是官方DINOv2复现。

Luna短监督已结束，主不并行轮询；实际fit交唯一“监督查询对话”每20min。当前handoff /Users/elan/.codex/worktrees/v2-gene-exposure-eval/grad-pert/.byte-os/coordination/handoffs/2026-10-16-dino-pair-cfe641b-formal-supervise.md SHAbca1cb1c15ee130dea65a69d13c1053f18a50e29ad2bd9be52071350eac034cf；ACK/ACTIVE及owner只以权威coordination state为准。故障保存证据回主，B0-DP完成后controller自动推进DP2，两组完成再回主比较结果与表征。本文档提交与训练cfe641b分开记录；不更新活动服务器checkout。后续旧状态均为历史。

## 2026-10-16：CG1首个更新触发显存保护，共同batch回退320

3397408十臂queue在CG1完成0/10更新时终态failed，FAILURE SHA4cd70fdd702baae3704acf01dd28a9b898ec363fde4f44d7d215a8b02ae37178。rank0/1实际CUDA进程显存29590/29508MiB，超过32607MiB×90%=29346.3MiB上限；余量2512/2594MiB且无foreign，首因是own allowance，并非foreign或后续SIGTERM。初始固定训练诊断已完成，正式训练未启动。失败源码/queue/日志/收据不覆盖，两个monitor保持PAUSED。

按已授权共同容量回退，新十臂目录configs/v2/fgpo_node0_20261016_b320：共同m80×2GPU×accum2=320，其余模型、损失、视图、采样、4轮、无验证、last18、eval128和中等分析均不变；逐臂与400配置逐字段核对仅microbatch/train_batch_size两项变动。新B0也从头使用同一320，不与旧400锚点混用。新clean发布/独立ID/全十臂10step+restore+双GPUeval重新验证后主验收。分析优化已验收15.079%测量区间缩短的结论保持；历史文字如下。

## 2026-10-16：完整中等提速已验收，推进新家族预检

同一DP权重、同一1335条件/5317行、128深度条件/506细胞、32梯度条件/125细胞的完整配对已独立验收：reference 3060.477s → fast 2598.974s，测量区间缩短15.079%；159654浮点统计最大差4.91e-11（门限1e-8），预测/CLS CSV一致、状态更新0，峰值allocated均8211.816MiB。该区间含进程内数据准备/模型加载/分析，未含解释器导入和最终绘图，不称整个交付流程提速。服务器COMPLETE SHA aeccac2f41c3d112d5a5b9917c8d399bcc688d1c5ffed4ed98fb0854d47b2460；主验收收据SHA d3fe46262065336f3f6c7aeb396e19db3d4247619d11bf7b4c0ede6f553ecc7c。

已实质接回main/build/in_progress，两旧monitor PAUSED。041ca06新十臂源码服务器CPU检查250通过、4失败：观察器测试误调用不存在的remove()，导致首个测试未清理全局CE hook，后续测试受污染；生产清理接口为close()。修正测试并保留原失败root，发布新源后重验。B0→VH→S12-L4→CG1/U24/MR1/P1/C1/O1/K2-fixed-prefix，四轮/global400候选/无验证/last18及相同中等分析不变，正式消融尚未启动。资源共存仅新增用户明确授权的UID1003 SciPENN exact脚本，上限与未知身份拒绝规则保持。下列运行叙述为历史。

## 2026-10-15：共用策略已授权，恢复探针通过，完整中等配对测速继续

旧fb01436全中等参考在1206/1335条件被外部资源门中止；2765fd6恢复写入变量遮蔽错误已修复。d37c645参考三条件完成后，恢复段因新SciPENN脚本被原名单拒绝。失败源码、ID、产物全部保留。用户现明确允许UID1003 `run_scipenn_strict_b_s2_smoke.py` 共用，仍每卡外部≤1GiB/最多1个、物理余量≥2GiB、我们≤90%，未知身份仍拒绝。

新分析源c9efc242ace04ea709723d001a8be28e256eeaa6 clean/pushed，与服务器一致；78资源/恢复检查通过。独立运行 `/data/yilangliu/GraD-Pert/development/medium-observer-c9efc24-20261015`：恢复与连续分析1927统计最大差0；fast对reference最大差2.55e-11/1e-8，47.42s→40.41s（短探针14.77%下降）。PROBE_COMPLETE SHA501dea148e8c703f3a7297b36f81c275531dffd9d118d4183df34b8cd7a62dd2。当前reference-medium→fast-medium完整配对；完整收益尚未验收。原1335条件/5317行/128深度/32梯度/128基因与科学source420d3a1保持；每100条件无损保存恢复点。

Luna短监督已结束；主继续新family调度和诊断兼容测试，之后交唯一长监督。新FG+PO/node0 B0→VH→S12-L4优先，4轮/global400候选/无验证/last18/只做同中等诊断。没有正式消融启动；权威owner/run见v2-gene-exposure-eval coordination。下列运行状态是历史。

## 2026-10-15：中等诊断性能优化，新消融尚未启动

原共同中等范围1335条件/5317行、深度128条件/506细胞、梯度32条件/125细胞、每细胞128基因保持。新GPU充分统计、身份选择前移、小Gram有效秩优化已通过确定性短探针：1927浮点数最大差2.55e-11；42.88s→40.85s，深度统计7.11s→4.73s。完整配对 `/data/yilangliu/GraD-Pert/development/medium-observer-fb01436-20261015` 正在运行，分析源码fb01436e544328e5e56effff51dd7c74f1d54dd8、科学源码420d3a1；尚未验收完整收益。旧失败/重复参考证据保留。

后续FG+PO/node0新B0→VH→S12-L4优先，再CG1/U24/MR1/P1/C1/O1/K2-fixed-prefix，4轮/global400候选/无验证/last18。十组模型配置与14项新增、64项相关回归已保存；family调度器/新方法与视图分析绑定、真实双卡10step预检尚待完成。短Luna监督结束，完整测速转唯一监督查询对话20min，终态主继续实施。权威run/owner见v2-gene-exposure-eval工作树coordination/state.json。下列旧全量计划已被中等范围覆盖。

## 2026-10-15：cap40五臂全量只读补测构建

DP/FN/FG/FB/PO四轮last均已有终态；PO原有CPU分析收据184b7236...已生成。主会话已接权build/in_progress，旧监督与return均PAUSED。用户授权全量cap40分析、不重训：47836truth/1335conditions，共同context-matched control/query/view/mask清单；全部行输出/CLS、全条件最多8cells深度诊断、64conditions×最多4梯度。原方法420d3a1不变，分析独立发布；12个身份/分组/流式统计测试通过，Ruff通过。当前尚未跑新GPU补测，先seal和五臂只读parity/state门，成功才正式分析。共同模板docs/experiments/CAP40_POSTHOC_COMMON_TEMPLATE_CN.md与计划CAP40_FULL_POSTHOC_20261015.plan.md。scDFM仍延期、不重训BR/其他模型。

## 2026-10-14: DP representation/prototype/output report accepted

Analysis source ae92f9db7f4f511b0f9d88238d707129bdda20d8 clean/published; 7 local and 7 server contract checks passed. Training/evaluation source420d3a1 unchanged. Actual CPU report receipt643e4944..., model_loads0/updates0/GPUfalse; 4PNG/7CSV/summary/report hashes verified; dry-run-reviewed transfer795811 bytes. Graph token rank7.97->48.28, Cellgene4.85->44.14, Responsegene final2.23; TeacherSSL1condition JS3.7586, SSL2node prefix concentrated argmax/JS0.00972, with small-probe caveats. Interpretation docs/experiments/DP_TRAINING_PROCESS_ANALYSIS_20261014.md. AGENTS adds per-arm completion report rule. Next: new formal+analysis handoff and unique20min supervisor ACK/ACTIVE; process FN/FG/FB/PO saved aggregates on CPU without changing controller or adding GPU work. scDFM checkpoint preserved/evaluation deferred.

## 2026-10-14：DP科学完成；每臂自动训练诊断报告

当前native-ffn-five-eval128-420d3a1-20261014，source420d3a1五臂正常自动推进。DP四轮/480更新/last18评估/zeroPKL已独立核对，checkpoint f0800c5e...，last-test a8edfbb8...。后续FN在训练，现有controller/source不动。用户新增每臂终态自动分析：独立CPU仅读小聚合JSON工具，中文报告+4图+7CSV/原型/逐层表征/输出/梯度/最终18指标，analysis与training SHA分开。7合同检查通过；DP实际报告生成和新监督自动化绑定待服务器收据。权威coordination已短回主/build/in_progress，旧monitor暂PAUSED；完成后立即新handoff恢复唯一20min监督。不改旧训练、不增加GPU诊断、不重跑官方/BR；scDFM评估延期原pt保留。方法/限制见 docs/experiments/GRADPERT_V2_AUTOMATIC_ABLATION_ANALYSIS_20261014.md。

## 2026-10-14: GEARS audit accepted; native FB eval300 gate failed

GEARS source7798605/checkpoint624ca968 preserved. Read-only diagnostic21ac1c9 (receipt5f2c9a37ae8218038dee8131442ecb6c98a34c3889b31b75cd1f478527ab8d1d) replays3fixed conditions with original300controls on CPU:225unseen mean deltas exactly0, truth std nonzero, manual/official/common Pearson constant_vector consistent; final independent decoder weight/bias~1e-41, exact masked gradients0. Coupled Adam weight_decay and no output supervision explains collapse toward control-copy. No scientific result overwrite/optimizer update. Report docs/experiments/GEARS_UNSEEN_ZERO_DELTA_AUDIT_20261014.md; small receipts evidence/gears-unseen-audit-20261014/. GPU audit old runtime policy denied current previously-authorized TotalVI-s2; no policy relaxed; initialCPUdiagnostic reference-order assertion corrected to condition-ID lookup, official eval alreadycorrect.

Native3ee8ef3 failed atFB eval300/reference128 fixed numerical tolerance. DP/FN/FG gates and FB10steps pass, noformaltraining. Failure/oldqueue remainimmutable; precisehandback/mainACK complete, bothmonitorsPAUSED. Main now prepares eval-only engineering at smaller common inference chunks256,192,128 under original2e-6 tolerances. All300control IDs/gene/DEG/aggregation, train4epochs/global400/m100*2*accum2/no-validation/last diagnostics unchanged. Tests109passed; all candidateconfigs selfcontained. Largest passing candidate still needs full five-arm dual-card gates/mainacceptance before formal launch. No active new native job yet.

## 2026-10-14: DP gates accepted, FN running; long handoff prepared

DP ten updates/restore and GPU0/GPU1 eval300-vs128 receipts passed. Latest Luna snapshot SHA9f24f2e6cf7fe3471fced0418890fa7738375fba3250228db3e847d9c84c8754 reviewed; FN preflight running. Luna ended, no duplicate polling. Queue native-ffn-five-leaf-3ee8ef3-20261014/source3ee8ef3 remains immutable. Handoff79941c7b28f81e77545dfce8b8b6ae34cbd61d1a48c01ba4033a4be86ee75230 prepared for exact supervisor01a0df0b-4142-7df1-86c0-d959471d80a1. Monitor/return fallback staged PAUSED; pending transfer/ACK/ACTIVE20min verification. At PREFLIGHT_COMPLETE supervisor returns main for independent exact MAIN_ACCEPTANCE, then formal DP/FN/FG/FB/PO four epochs/last. No formal fit yet; scDFM evaluation remains deferred and its original last.pt preserved. Small startup/handoff evidence: evidence/native-ffn-five-20261014/.

## 2026-10-14：新五组队列真实启动，短监督预检

发布源3ee8ef3b343efa5223f0be4f7dd570e839651f88，本地和服务器101检查通过。新queue native-ffn-five-leaf-3ee8ef3-20261014 SHA01a9f4239cf0412446f1497f58b2d5e46d35b8fc63163997d03fb14f9e48a7a9；runtime5f76311010cb773632170f31c89fd7f0eee323a2d2132a60ec5c525062acb2c7。startup a26da90737bfcc04e0aecdc1d5f2de13fafb960e681753a7f12c50039d7d0516：controller3046624/start735652620/真实cmd892d5f2f5da774c7dd44106cdc6fc8161e7bd03dcf2a48e48a8ab70bd21a675b；DP torchrun3046781/start735653056，CUDA0,1/allocator合规；真实进度8/10。main核对小receipt SHA，Luna唯一只读监督后续45min预检，主不重复查。仍需五臂10update/restore及双卡300vs128 eval全部门、主MAIN_ACCEPTANCE，正式训练未开始。scDFM评估延期/原pt保留，旧failed队列保留、没有重跑BR。长监督/return monitors PAUSED，待确需长监督再唯一交接。

## 2026-10-14：DP十步通过，eval队列范围检查修复

1fa8cd4 queue已终态failed：DP十步/恢复通过，receipt275b27b94c2a6cf5954d86a355b57ef0675bcf7c2d377e08d494f378314ede40；eval-capacity-g0在计算前被旧BR family guard拒绝。失败监督收据0cd174237b4d1d4c89e89384430bd6459c5bf5a5fae6cf4bbfe3eb63e5dc6561保留，无正式臂启动。main独立确认state失败，Luna已结束。本地补精确FFN五臂/order/mode/10steps及原leaf绑定测试，101检查通过、Ruff/diff通过；模型/config/数值容差/资源策略不改。下一步发布新源/新ID并重验预检，scDFM原pt不动、评估延期。下文启动状态为历史。

## 2026-10-14：五组预检实际启动

新source1fa8cd48d89b1bf590a9fdf73a633505d2e522a8，本地/服务器134 checks通过。queue native-ffn-five-deferred-1fa8cd4-20261014 SHA b911097330eb914bf0eefc311c3a1552fd4f65dda1463eb1549f92a603c92360，controller3042404/start735567114，DP child3042493/start735567519，CUDA0,1/allocator匹配，容量收据running。Luna只读监督短门，主不重复查；正式训练尚未开始，待五组10步/restore及双卡eval容量门、主MAIN_ACCEPTANCE。scDFM评估延期且原pt保留，GEARS终态验证；旧CPU等待器精确退休。仍DP→FN→FG→FB→PO四轮、global400、accum2、无验证、last18指标/zeroPKL及原过程诊断。监督/返回monitor现PAUSED。

## 2026-10-14：用户授权五组优先，scDFM评估延期

主已持权build；先DP→FN→FG→FB→PO，原四轮/global400/双卡累积2/no-validation/last18metrics和诊断保持。新增显式调度policy验证原scDFM训练完成+原pt保留及GEARS真正完成，scDFM评估仍延期而非完成；原strict默认不变。旧native r2仍CPU等依赖，无正式消融；新源码测试/发布/独立预检准备中。每卡约800MiB的UID1003 TotalVI source-validation在既有许可内，资源门不改。scDFM fit/last.pt SHA9dff79f7e2836482ea6f272dd619c9df967e73f8e5b16f911679c3d557c62a8c保持，禁止重启评估。详见更新后的plans/OFFICIAL_RECOVERY_FFN_DIAGNOSTICS_20261013.plan.md。

## 2026-10-14：用户停止scDFM评估，checkpoint保留

监督精确SIGTERM仅scDFM评估worker2823360，主已独立SSH核实进程退出、fit/last.pt存在且SHA2569dff79f7e2836482ea6f272dd619c9df967e73f8e5b16f911679c3d557c62a8c前后相同（702057381字节）。训练3988/3988、四轮预算完成；评估48/592中断，缺last-test/COMPLETE，不能标为官方基线完成。原运行/权重/已有产物保留，GEARS未改且已完成；native未启动、仍等待scDFM。官方controller因用户中断终态失败，禁止重启。两个监督/返回monitor均PAUSED，owner已回主build并实质ACK；停止请求已完成，等待用户下一决策。验收 .byte-os/evidence/gears-preparation-20261014/scdfm-user-stop-main-acceptance.json，详细停止收据在权威coordination目录；此状态覆盖此前运行/监督中的叙述。

## 2026-10-14：启动核对通过，准备交唯一长监督

Luna短startup已结束：GEARS新源7798605构图1781/1781用5:49，仍preflight，尚无GPU十步通过收据；scDFM原worker继续3000/3988。新native r1启动因缺少显式CUDA_VISIBLE_DEVICES空值被CPU门拒绝，旧queue/log保留；fresh r2 PID2897522/start731623535已核实running/waiting_official_baselines、空CUDA、queue SHA583496f99cba00325055d34ed16d61c08a8bde8393ba199d2377740e18728e72。之前dd02…/PID2895684不再有效。两startup收据见handoff；当前main持权，唯一监督handoff SHA672da021bc8c50789af3fd4bcb9e38c482694342c160bd8eee77129a15c177af已准备，两个monitor现PAUSED；需relay、精确监督ACK和ACTIVE20min验证后结束main轮询。新native待两官方真终态后只启动预检，MAIN_ACCEPTANCE仍由主验收。

## 2026-10-14：GEARS准备修复验证并启动独立接续

7798605源：服务器196检查、官方CPU10updates/restore/zeroPKL、真实8cell graph/pair/RNG精确parity通过。CSR重映射2.583s，原转换>5min；局部graph10.674→0.037s，不能称端到端285x。原parallel已foreign资源门终态，scDFM原worker保留；GPU1重新空闲后生成新terminal-adoption mixed source schedule，controller2894437，GEARS fresh ID/source7798605，仍10步→四轮last18metrics。新native CPU依赖waiter2895684、五组顺序/主验收屏障不变，旧waiter精确退休。Luna唯一短startup核验进行中；长监督尚未ACK/重新ACTIVE，主owner/build。报告docs/experiments/GEARS_PREPARATION_PERFORMANCE_20261014.md及小receipts；旧失败和探针中断均保留。

## 2026-10-14：GEARS准备第二阶段（十步不变）

透明阶段探针定位build_prepared的CSR→COO转换：初始化0.802s、读取12.017s，后续转换多分钟仍无完成；旧21min探针保存中断证据。新canonical CSR索引重映射和context候选tuple共享已实现；数值/顺序/输入不变检查进行中，真实输入parity与新CPU十步待验证。scDFM原worker继续，旧GEARS准备仍运行，未启动新正式GEARS。主owner/build，长monitor PAUSED；Luna唯一监督短stage1探针。

## 2026-10-14：GEARS十步预检的CPU准备瓶颈接回

max_updates一直为10；26/1781是CPU条件构图数，不是训练step。现原预检1h51未进GPU、多数条件48sec，主已接回owner/build，旧monitor PAUSED，scDFM原worker继续。修复仅缓存同context/control候选池和只读target视图；官方构图/随机采样保持。已完成官方排序H5AD仅在文件/原RUN/data/config/env/method及完整sanitize输入严格核对后复用。新的独立GEARS源码/运行记录和保留scDFM旧源码的混合source调度准备中，真实数据耗时/原新graph RNG对照及新GPU十步尚未完成。计划plans/GEARS_PREPARATION_REPAIR_20261014.plan.md。

## 2026-10-13：GEARS已在GPU1实际启动预检，scDFM原训练连续保留

调度发布4bf0db46f6398b4241df76b447ae2303c0d7415d，本地/服务器75检查通过。仅旧CPUcontroller2818852安全退休；原scDFM worker2823360/start729241036不变，实际日志1600/3988。新并行controller2853681，schedule SHA e43f560e3c735b002c8bbe1289a986227b5679eb6e2105313315bba807669b77，GEARS preflight2853707实际CUDA_VISIBLE_DEVICES=1，当前CPU数据准备，尚无GPU/训练更新证据。Luna短监督结束，主独立核对，无FAILURE。两runner科学源/配置/数据/评估仍9607981，GEARS原尚未启动run保持；native依赖不变，须两真终态strictforward后才推进五组门。长监督新handoff待ACK/ACTIVE，不重跑BR/Scouter；旧controller消失不能当成scDFM停训。小收据evidence/official-ffn-recovery-20261013/parallel-4bf0db4-main-startup.json。

## 2026-10-13：用户授权空闲GPU1并行GEARS，调度迁移准备

GPU1实测无CUDA进程（2MiB/0%），GPU0 scDFM full仍运行。旧顺序派发器占双卡锁且保护会拒绝独立GEARS，不能直接另起。主已ACK监督handoff，旧monitor PAUSED。新调度器只接管CPU派发，不重启scDFM：精确PID/tick/cmd树参与原资源门；GEARS保留原source/config/runtime/run，物理GPU改为1。科学src/benchmarks逐文件哈希不变是硬门，训练SHA仍9607981，调度新SHA另记。75项检查及lint/format通过，实际部署/迁移/GEARS启动尚未完成。计划plans/OFFICIAL_PARALLEL_GEARS_20261013.plan.md。

## 2026-10-13：GPU数值门通过，官方恢复与后续五组依赖已启动

训练发布SHA `9607981fd7d44b3982574167132d06f739bc1b7e`，服务器独立干净checkout已核对。最终scDFM/GEARS各自冻结环境CPU十更新、恢复预测精确和zero-PKL均通过。scDFM保留布局的激活卸载GPU三更新对照通过：两组各1065张量最大绝对差0，原rtol1e-6/atol1e-7未变。真实官方batch48十步fit已完成，300-control预检收据尚待核对；这不是正式训练或科学结果。

官方新attempt2 `official-recovery-9607981-20261013-r2` controller2818852已运行；先scDFM预检/3988更新及last评估，再GEARS四轮及last评估。依赖native队列 `native-ffn-five-9607981-20261013-r2` controller2820803仅CPU等待两组实际18指标终态，不占GPU锁。然后DP→FN→FG→FB→PO十步及评估容量门，PREFLIGHT_COMPLETE后须主会话验收并生成MAIN_ACCEPTANCE才训练，不能由监督绕过。BR/Scouter不重跑。旧missing-parent失败和CLI/Python启动失败原样保留。机器资源门仅新增用户刚授权的UID1003 TotalVI s2脚本，上限不变。精确队列SHA/数值证据/阶段见evidence/official-ffn-recovery-20261013/launch-9607981-r2.json；长监督ACK/定时激活待单独收据，当前仍主owner。

## 2026-10-13：新TotalVI共用授权；官方环境按臂选择

用户明确允许UID1003 run_totalvi_s2_formal.py按相同限制共用（单卡外部≤1GiB/最多1个、物理≥2GiB、我们≤90%）。新增精确authorization_reference，旧策略不能借此扩大脚本范围；66资源门检查通过。387abab scDFM真实CPU10更新/恢复精确/zeroPKL成功；GEARS首次手动启动误用scDFM Python，被anndata版本门拦住，失败保留。后续按runtime[python]选择各自冻结环境，最终发布版本重新执行两组CPU门。没有正式训练/科学完成；五组native须等待两官方真实last终态，BR不重跑。

## 2026-10-13：恢复官方基线，准备五组FFN及训练诊断

用户授权仅恢复scDFM→GEARS测试/四轮训练/last18指标评估；BR和Scouter不重跑。五组DP→FN→FG→FB→PO已自包含配置，global400/m100×双卡×accum2/四轮/无验证/仅last，须等待两官方真实终态。原始clean基线141989e已与GitHub main一致。原型/逐层token/输出及共享graph梯度探针已实现，训练前后RNG/Student/Teacher/center/optimizer/grad等合成多步精确检查；155+6本地检查通过。

scDFM新增保留stride与storage offset的CPU保存激活卸载；普通repeat及原容差GPUparity是正式preflight之前的硬门，不改模型/MMD/FP32/48batch。旧OOM/容差失败保留。当前GPU0有UID1003 run_totalvi_s2_formal.py约796MiB，现白名单不覆盖，新GPU探针被门拦住；授权问题已发，未扩白名单。新controller只允许CPU等待资源，不批准未知进程。尚无新GPUparity/训练/科学完成。详细计划plans/OFFICIAL_RECOVERY_FFN_DIAGNOSTICS_20261013.plan.md，报告docs/experiments/GRADPERT_V2_FFN_FIVE_DIAGNOSTICS_20261013.md。下一步发布clean新源码、真实CPU10/恢复、启动带numeric门的独立官方队列，再准备依赖native队列并交唯一监督。

## 2026-10-13：BR400/F1/N1/GM1 原型诊断完成

用户要求检查蒸馏9.7与N1低loss差异；诊断工具285a1fe，原模型分别保持训练6a8c612/4bfa1f1。相同4×16真实训练输入与完整视图，4组input SHA一致、eval/BF16、无optimizer/EMA/center更新，原checkpoint与逐张量模型状态不变；四组exit0。N1目标尖锐且跨样本JS2.46–4.15，不支持常量/单原型坍塌；低CE主要由Teacher entropy下降解释。BR/GM1目标近均匀且KL小；F1图condition Teacher JS1.6e-5、瓶颈相对方差1.54e-4/秩2.16，严重退化迹象。报告docs/experiments/GRADPERT_V2_PROTOTYPE_DIAGNOSTICS_20261013.md与evidence/prototype-diagnostics-20261013/summary.json。仅有限输入诊断，不判定生物学效用或重训；Luna短监督已终态，未建新定时监控。此前官方scDFM修复仍未正式启动：88卸载GPU多步对照失败，旧失败证据保留；本诊断不覆盖该未完成事项。

## 2026-10-13：scDFM batch48 显存修复（保存激活卸载）

设备别名修复f83a44e542c0245b296ed4f29e9f74df91a54841两组真实CPU10-update/恢复通过，但新queue official-remaining-f83a44e-20261013T042152Z的scDFM GPU preflight在官方differential attention前向OOM：27.30GiB allocated，28.22GiB allowed，下一临时张量需1.43GiB。无正式fit，无GEARS启动；旧失败root/源码/config保持。本段覆盖“等待GPU预检”的状态。

保留官方batch48、FP32、完整batch MMD、Adam/cosine/3988updates与所有数据/评估规则。仅适配器在官方train_step前向外包PyTorch save_on_cpu(pin_memory=True)，将反向保存张量转存CPU，反向取回；不改官方模型/attention函数、不切半精度、不分批近似MMD。新增显式training.activation_offload=cpu及适配记录；默认none与CPU路径无改动。89本地检查通过；Ruff/format通过。需要同源真实GPU输出/梯度/多步optimizer及checkpoint预测对照、CPU10/恢复、再真实batch48十步/300-control预检。CPU卸载可能增加传输时间，不宣称提速或科学完成。主保持owner，旧跨会话监控PAUSED；短诊断交Luna，不重复查询。

## 2026-10-13：scDFM 设备别名误判修复，接续剩余两组

本段覆盖此前“四臂等待/运行”状态：BR400/F1/N1/GM1及两组Scouter均已完成各自last评估。官方四臂queue official-four-all-queue-43ccca4-20261013在scDFM GPU preflight失败，controller exit1，GEARS未启动；原queue、失败收据与已完成Scouter产物保留，不重跑。主会话已接回，两个旧monitor保持PAUSED；authority见 /Users/elan/.codex/worktrees/v2-gene-exposure-eval/grad-pert/.byte-os/coordination/state.json。

根因已实复现：服务器Accelerate1.9.0返回torch.device('cuda')，与配置cuda:0直接比较为False，即使current CUDA index是0。适配器现在只把未编号CUDA设备解析为current_device，再严格比较设备类型/编号；CPU、错误GPU、多进程拒绝规则保持。仅benchmarks/expression_holdout/scdfm.py改动，native模型、上游checkout、数据/loss/预算/推理规则不变。预修改基线5159085e4ef3b84098864eaca6f65de318f89e6e已确认clean且GitHub main一致。本地80个适配/配置检查及6个dispatch检查通过，新增10个设备guard检查含在前80中；Ruff/format通过。新source发布、同源真实官方CPU10-update/恢复及scDFM GPU预检尚待执行。

下一步：独立发布源码及新ID，只推进scDFM→GEARS，各4epoch，scDFM3988optimizerupdates；GPU十步/恢复/300-control预检后自动正式训练和18指标last评估。完成启动门后交监督查询对话20min监督，旧失败queue绝不续写。实现计划 plans/SCDFM_DEVICE_ALIAS_REPAIR_20261013.plan.md。

## 2026-10-13：用户选定官方 ScELMo；四组依赖队列已启动等待

用户明确选此前官方Scouter的scELMo，只忽略训练CCDC130+ctrl（40行），不选原始rawGenePT。实际新queue official-four-all-queue-43ccca4-20261013，SHA7a741792d5626d5df51c9eca6a2bd296f434118fe1e5553528c9bcf3b93d67f8，source43ccca4fd14dfbdf451d5673da3f5d1b96df833f，controller2719988/start725165201/cmdSHAa52fe0b4c2d93ccd8bab542ac1b7134fdd2a93279151c4a95b741a09c0399e27。已实核waiting_dependency。顺序自有GenePT Scouter→官方ScELMo Scouter→scDFM→GEARS，各4epochlast-only；scDFM3988updates。沿用已验证4CPU10/恢复/zeroPKL和完整canonical/prior审计，代码/config不变，不重测同一CPU工作。

只有ScELMo臂训练47796行，其余47836；val41235/control12013相同；完整592tests/frozen300controls/三Pearson×全DEG×allseenunseen仍相同，225测试表达剔除维持。先验种类/维度1536vs2048及40训练行差别明确记录，不能称纯同输入先验比较。旧3arm56aae等待controller/wrapper已精确SIGTERM退出，stopreceiptSHAa504d162439fa4af608f7720557baac0c1d8973209b036b6f0f6176aeecfefbe保留；旧2241和旧3arm都不恢复。父当前消融不触碰，其真终态后自动GPU10gate+正式4轮+last评估。同一监督20minmonitor更新中；handoff coordination/handoffs/2026-10-13-official-four-all-dependent-supervise.md，结果未科学完成。

## 2026-10-13：四轮官方基线发布、真实测试及新依赖队列

用户预算更新覆盖旧official2241等待队列：已精确停止controller/wrapper，未启动外部GPU。新source43ccca4fd14dfbdf451d5673da3f5d1b96df833f，163localpassed/1历史skip、Ruff；四actualofficialCPU10/restore/zeroPKL及三canonical审计通过（官方alias清单遗漏SHA一次失败保留，独立attempt2修复）。自有GenePTraw2048全2372，CCDC130存在；project与GEARS数据完全一致；ScELMo候选只减40训练行且592test完全保留。

新CPU等待queue official-four-queue-43ccca4-20261013，SHA56aae90f62de7aa26cb92987080fd48e88691aecab52fc17972194c8cdd89fe7，controller2674076/start723504303/cmd8d3b5fcc5f5859d8dc8923884043e43b43ea821b59c061ce9835998c986771bc，已实核waiting_dependency。顺序projectScouter→scDFM→GEARS，4epochlast统一18指标，scDFM4ceil47836/48=3988updates（等效sample预算）。父F1/N1/GM1真科学完成后自动GPU10gate+formal+last评估；不修改父/旧source/config。官方GenePT对照来源仍待澄清：此前单缺CCDC130的是ScELMo，原始GenePT缺17；该arm未派发，不静默删更多。细节handoffs/2026-10-13-official-four-dependent-supervise.md，small evidence official-four-20261013。

## 2026-10-13：官方 Scouter/scDFM/GEARS 表达剔除适配验收与依赖队列

三组官方适配/config完成，140本地测试通过、1旧官方checkout测试跳过，Ruff通过；服务器真实官方包均10次优化/恢复预测一致/零PKL。同源2241c46ed3d320b3e194a1f7ba989637b4587a24两数值环境的真实数据合同一致：cap40训练47836/验证41235/control12013，4775可见表达/225剔除表达，excluded非零数0；仅官方训练数据对象，不含test扰动细胞。冻结300controls与DEG/Systema参考完全匹配历史BR。

后续官方queue已实际启动CPU等待：official-expression-queue-2241c46-20261013，SHA049e6fcf8d8277c19fb8c1bddb7f1cb9fa52b6c7e9ffdfce8764a16aff186bd0；仅父F1→N1→GM1真终态成功才GPU10-update预检，再scDFM→GEARS官方训练+18项last评估。Scouter47aliases已审核，唯一缺CCDC130/YJU2B官方向量，正式运行blocked，不删除条件或补零；用户额外先验授权待答。没有新的GPU训练/科学结果，当前原生队列与源码不改。现有监督会话保持唯一20min监控，下一阶段handoff已接纳，详情独立official-baselines-build.json及报告。下面为历史状态。

## 2026-10-12：BR400 epoch4 last 双卡评估独立验收；剩余三臂协议升级

BR run nadig_jurkat-seed1-20261012T061448Z-cba17ce53a6b426cb8367b767d453cd5 的第4轮last成功评估592条件，18个Pearson/有效条件/hash/zeroPKL已核验；双卡各batch300，约39.3分钟。TxPert整体全/DEG0.228570/0.383056，TriShift0.177556/0.345992，Systema0.095442/0.213948。旧bestlast队列按用户停止保留，不说全队列成功。后续F1→N1→GM1四轮/global400，禁逐轮验证、不选best、只测last，需要新发布源码/config/ID；实现中，尚未训练。当前owner主，Luna本次监督终态结束，无活动定时监控。报告docs/experiments/GRADPERT_V2_BR400_LAST_ONLY_SWITCH_20261012.md；新计划BR400_REMAINING_LAST_ONLY_20261012.plan.md。以下旧阶段历史。

## Active BR400 best/last stage (2026-10-12)

User approved validationjointbest plus final4last, all BR/F1/N1/GM1 fresh400, common evaluationmax probe. Publishedprechange832a926; old192peercontinuation superseded before dispatch. Build/gates/mainacceptance/launch pending; exact roles and resource boundaries in planGRADPERT_V2_BR400_BEST_LAST_20261012.plan.md.

## 2026-10-12: BR epoch4 scientific completion accepted; continue peers

Main independentlyverified nativevalidate_completed and18metrics/source/config/checkpoint/data/control/environment/zeroPKL. BRsourceb37, checkpointf679...9d2; TxPertall/DEG0.218508/0.389176, TriShift0.164770/0.356504, Systema0.103747/0.211304. Overallfinite592all/590DEG, unseen225genesDEG147conditions. All592processed, noimputation. Original10ad CPUqueue retired, F1/N1untouched withsource/config/data10-stepcheckedpreflightsandoriginal.65fitplans. NewcompletedBRimportmodebuildownedmain, freshF1N1IDs/common192/4epochs/sourceb37, noBRfit/evalrepeat. GM1requiresfreshparentdependency afterremainingpeerqueuecreated; originalGM1retired. Oldevalmonitor deleted, returnfallbackpausedaftermainACK. Evidencepost-br-capacity-20261012/gradpert-BR-epoch4-terminal-main-acceptance.json; nextplanGRADPERT_V2_POST_BR_CAPACITY_20261012.plan.md.

## 2026-10-12: BR last evaluation actually running after capacity acceptance

Acceptedfour-grid global400capacity10updates, minimumfree5004/5024MiB,16.376training-wall cells/s. OriginalBRscientifictraining/evaluation sourceb37 unchanged. FreshCPUevalcontroller `v2-br-eval-057a428-20261011T212820Z` source057a428585dbd7fb61128c3928ef2b3d51365ebd, local/server23CPUchecks+reviewpassed; main independentlyverified exactqueue/source/controller/child/checkpoint/config hashes and actualprogress10/592. NativeGPU0 attempt1/25percentcap, last epoch4/frozen300controls/18metrics, same2GiB/85percent/schema3guard, nootherfit. F1/N1/GM1remainpendingbatch192, cap400notauto-adopted. Lunafinishedshortlaunchwatch; handoff exactsupervisor20min pendingACK/ACTIVE. Evidenceevidence/post-br-capacity-20261012/gradpert-br-eval-main-launch-acceptance.json; scientificcompletionnotyet.

## 2026-10-12: post-BR capacity accepted, BR evaluation next

Highestconfirmed4-grid dualGPU capacityglobal400=m100x2GPUsxaccum2,10updates/restore/300x5000inferencepassed. Same2GiBfree/85percentbudget; minimumfree5004/5024MiB, sampledowned27098/27078MiB. Trainingwall16.376cells/s vsbaseline1928.063, +103.1percent boundedprobe; startup12.399s, nativecheckpointrestoreincluded. m104/global416guardfailure andm112/global448OOMpreserved. Source1255a936369fb2085b6da279046ed6558ad412be, native src identicale3c. Tenstepcapacity notlongstability/physicalexactmaximum. ExistingBR/F1/N1/GM1remainbatch192.

BR evaluatorCPUguardwrapperpublished23ca5c4dcfd7598049db2eef792125196cc34d99;23localtests+reviewpassed. Main independentlyacceptedm100terminal, Lunafinished; nextserverCPUcheck/freshevalroot/realoriginalb37epoch4last18metrics, thenlongsupervisor20min. Evaluationnotyetstarted. OldmonitorsPAUSED, nootherfitstarted. Compactevidence/post-br-capacity-20261012/.

## 2026-10-12: first capacity search terminal; single missing-grid refinement

Independent main verification: m48/56/64/80/96 ten-update passed; global384 fastest15.832training-wall cells/s, sampled free6940MiB eachGPU. m112/global448OOM after1update; m104/global416 stopped at85percentbudget27836MiB, free4266MiB. OldrootFAILED/source/receipts immutable; allworkersgone. Main substantiveACK andbotholdmonitorsPAUSED. Remaining4-grid m100/global400 freshsealed singleprobe implementation/verification inprogress, noresourcepolicyrelaxation/no priorpointrepeat. BR epoch4 last18eval pending; F1/N1/GM1 notstarted andretainoriginalbatch192. Detailed report/receipts: docs/experiments/GRADPERT_V2_BR_CAPACITY_20261012.md andevidence/post-br-capacity-20261012/.

## 2026-10-12: BR training complete, guarded capacity search running

BR epoch4/1000 updates complete; epoch4 last18 metrics pending, F1/N1/GM1 not started. Capacity source `e3c9183dc24e6413efc75a3df29bfe564def826f` published and immutable. Local/server29 CPU regressions and independent review passed. Queue `v2-br-capacity-e3c9183-20261011T202754Z` attempt1: matching controller2436572/start715413203/cmd, queue/runtime/publication independently accepted; m48/global192 passed10 updates, nonzeroLR/finite gradients, checkpoint restoration,300x5000 inference. Training-wall throughput8.063cells/s (measured updates8.257), peak reserved13.65GiB. Only bounded capacity evidence; controller advanced m56/global224. World2/accum2 unchanged, physical free>=2GiB perGPU and aggregate owned<=85%, exactUID1003 script allowance unchanged.

Deployment wrapper exceeded20s startup confirmation and exited1; durable controller remained alive and passed native probe. Failed confirmation preserved, no duplicate launch. Earlier22d failed before GPU due wrong runtime field; hotfix uses actualrepository_root. Temporary publication reverseforward canceled; SSH retained. Highest passingbatch and fastest throughput reported separately; remaining ablations keep192. Exact supervisor20min handoff being prepared, ACK/ACTIVE unverified until receipt. Evidence `.byte-os/evidence/post-br-capacity-20261012/gradpert-capacity-e3c-main-acceptance.json`.

## 2026-10-12：BR四轮训练完成，容量测试优先

独立holder source4c8a7ff67b087d306fcd6753a987809bdd8baca2已发布，本地11/服务器65隐藏CUDA检查和复核通过。精确STOP仅两个CPU派发器，BR双rank自然完成，原5秒schema3守卫接力；真实epoch4 TRAIN_COMPLETE+原进程exit0及last SHA f679adf48082e4e334eb18601e487fb4a316ebc6e80365934e2c414c621fb9d2验证。旧CPU派发器已退休，F1未启动，旧queue/state/source保留。BR仍未评估，不是科学完成。用户新增post-BR最大双卡batch十步测试，每卡free>=2GiB且维持owned<=85%；先容量，再补BR last18指标，再接F1/N1/GM1，当前不改变消融batch192。主已实质接续构建容量探针控制器；holder小收据evidence/post-br-capacity-20261012/。

## 2026-10-12：动态共用修复发布，BR真实恢复

新controller/GM1 source10ad0aae9a586377cda2212f29db6e6fe9a38a1f，187本地/366服务器隐藏CUDA检查及独立复核通过。用户批准schema3仅UID1003/精确ScButterfly脚本的新PID共用，其余1GiB/2GiB/85%/UUID限制不变。新queue v2-ffn-recovery-10ad0aa-20261011T163726Z attempt3，BR科学源仍b37，原epoch2/500完整状态恢复后已实测epoch3/502更新、双rank身份匹配；旁边UID1003任务720MiB仍存活，所有旧失败证据/备份保留，无模型/config/batch/LR改动。F1/N1freshID顺序，GM1新依赖queue仅CPU等待、未预检。现在无科学完成结果。主验收收据evidence/graph-mhc-20261012/2026-10-12-ffn-recovery-10ad0aa-and-GM1-main-acceptance.json，长监督handoff已准备，20min激活与ACK待核验。

## 2026-10-12：BR第二次真实保护停训；严格恢复准备中

原430a132恢复队列与GM1都已真实failed/failed且有FAILURE.json；监督收据漏报了该marker，主实时核验纠正。最后遥测在原双卡BR进程之外出现UID1003 ScButterfly PID2381683（GPU0 498MiB），不是模型OOM。所有旧queue/source/配置/日志保留，两监控均暂停；主已接owner/build并实质ACK。

BR在epoch3 step71、571/1000时中断，但可恢复边界为已提交epoch2/500steps。checkpoint SHA256 3c3e772d4a39a7b393b9360e8b59e170e79f113fd21951c6f2fdf686578a1ccc；服务器隐藏CUDA核验payload/hash/原b37身份/两rank RNG/StudentTeacher四中心/完整优化器均通过，尚未执行新nativeDDPrestore。新的严格恢复控制器将支持已恢复队列的再次恢复与已提交epoch2/3，原科学源码/4×250日程/runlaunch不变；F1/N1新ID、GM1新密封依赖，原失败记录/备份不覆盖。用户已明确批准同UID1003/同ScButterfly脚本的新PID共用，每卡最多1个≤1GiB、剩余≥2GiB、owned≤85%；新schema3仅在新队列启用，旧策略原样保留。重复恢复及动态共用代码已完成；187项隐藏CUDA CPU检查、Ruff/format/diff和独立复核通过。支持epoch2完整状态、封存祖先失败/备份、保留原科学b37配置和日程。下一步发布新控制器、服务器CPU复核并真实恢复BR；GM1尚未预检，无科学结果。证据evidence/graph-mhc-20261012/。

## 2026-10-11：GM1图mHC排队；BR严格恢复等待资源

发布`430a1326aeabdb2e326d840295f6b31d2d9bc3c2`，服务器隐藏CUDA检查295通过（60.69秒）。主与Luna核验两控制器/源码/配置/publication/queue/checkpoint备份：`v2-ffn-recovery-430a132-20261011T124207Z`CPU waiting_shared_resources，`v2-graph-mhc-430a132-20261011T124207Z`CPU waiting_dependency，均无GPU子任务。BR仍只提交epoch1，尚无新恢复更新；新ScButterfly PID不继承旧白名单，资源门保持严格。

资源可用后BR用原b37科学源码及完整状态/原4轮日程恢复，再F1/N1原配置新ID；三组真epoch4评估完成后自动GM1十步预检，主验收后fresh4训练及last评估。GM1仅图mHC4（四层各attention/FFN共8routes），Student47,051,014，+196,824（0.42%）；其他方法/损失/数据/批次不变，无新效果或速度结果。

已准备handoff给监督查询对话，每20min；转交后须验证ACK与ACTIVE。完整小收据`.byte-os/evidence/graph-mhc-20261011/graph-mhc-430a132-startup-20261011.json`；机制文档`docs/experiments/GRADPERT_V2_GRAPH_MHC_20261011.md`。旧失败队列全部保留。

## 2026-10-11：恢复控制器CPU复核通过，新增资源等待门

修复3102fcd40a04980e22339349973961893d5791d0服务器隐藏CUDA检查289通过（60.63秒）。恢复队列v2-ffn-recovery-3102fcd-20261011T123239Z在BR启动前被新UID1003 ScButterfly PID2337671（GPU0约720MiB）拦住；原共用白名单仅绑定已退出的旧PID，不能把新PID静默继承进去。该恢复队列events为空、BR没有执行新更新，失败证据及epoch1备份保留。

主正在修复无活动CUDA子任务时的资源等待：未知占用保持CPU等待、资源门不放宽；实际训练中仍5秒检查、违规立即停止自身子任务。GPU身份漂移仍终态失败。新恢复队列将保持原b37科学源码/协议/预检，GM1绑定新等待队列。监督尚未重新开启，不能宣称BR已恢复训练。

## 2026-10-11：GM1 已发布；BR 停训原因及恢复修复

GM1图mHC实现已发布93ed39d02fa76730630a718d79c489b6adf64f48；配置仅相对BR打开graph_mhc，Student47,051,014（+196,824）。本地输出/梯度、多步optimizer/EMA/center及checkpoint检查通过；服务器248 CPU检查通过，但测试没有隐藏CUDA，checkpoint RNG抓取创建498MiB上下文并触发BR共享卡保护停训。PID2337434属于我们，不是他人任务；科学源码与资源门不改。旧失败队列/日志保留，GM1尚未形成活动队列或模型GPU预检。

BR最后提交epoch1、250/1000更新，epoch2有16步未提交，恢复会重做这些步。主会话已接回并暂停旧监控，正在实施CPU测试隔离及严格完整状态恢复控制器：BR用原b37源码/原4轮warmup+cosine及checkpoint恢复，F1/N1同原配置新ID。GM1将绑定恢复队列，父三组真终态后才十步预检，主验收后fresh4训练/epoch4评估。不能声称恢复已启动或新mHC已有科学收益。详细证据见plans/GRADPERT_V2_GRAPH_MHC_20261011.plan.md及docs/experiments/GRADPERT_V2_GRAPH_MHC_20261011.md。

## 2026-10-11：GM1 图 mHC 单变量组实施完成

用户新增GM1：仅相对BR开启图mHC4，四层各attention/FFN独立route，保留原norm、KDA末态/邻域/source、GenePT、Cell/Response和训练/评估协议。Student47,051,014比BR多196,824参数；Teacher同构，新增CPU私有初始化避免CUDA RNG污染。默认关闭保持旧payload。247受影响检查及3源码mypy通过，独立审查 finding已修复；后续发布/服务器CPU复核与CPU依赖等待控制器准备。仅父BR/F1/N1真4轮last18指标及哈希/zeroPKL完成后执行GM1十步预检，仍停主验收屏障；现在不占GPU或修改活动b37。计划plans/GRADPERT_V2_GRAPH_MHC_20261011.plan.md，设计docs/experiments/GRADPERT_V2_GRAPH_MHC_20261011.md。

## 2026-10-11：正式BR长监督ACK和定时注册已核验

指定“监督查询对话”独立ACK：controller及fit身份/source/queue/preflight哈希匹配，无FAILURE，BR epoch1 step9/250。supervisor ACTIVE20min、exacttarget/currentqueue已主从saved automation核验；returnfallback PAUSED，准备真实终态handback才激活。owner监督、repairbudget0，主结束短监督且不重复查GPU。主保存ACK证据evidence/ffn-dual-coexist-20261011/b37-supervisor-ack.json(SHAab8dabcba03e2ced214312429071cfe160461e3bfdc6cb8c116d86b0b6739a30)。训练仍在进行，非四轮科学完成。

## 2026-10-11：三组共存预检验收通过，BR正式四轮启动

训练SHA b37b171fd68cb29e5c8142c66778aac6da9a6f06；F1/BR/N1均10/10、完整checkpoint恢复、实际300×5000推理、有限梯度、精确source/config哈希通过；两个指定foreign任务在各组全部运行采样均存在。主已独立验收并写精确acceptance；同queue自动启动fresh BR，真实epoch1 step1/250、目标4轮1000更新。随后仅F1→N1各4轮，无验证、last-only18Pearson/zeroPKL，三组共用m48/world2/accum2/global192。降低m40候选未运行，当前不因显存需要降低。性能报告docs/experiments/GRADPERT_V2_FFN_COEXISTENCE_20261011.md；短测试非持续容量/科学结果。Luna终态已退出，主不重复监督；指定长监督交接与20min激活正在完成。旧失败完整保留，绝不修改活动b37服务器源码。

## 2026-10-11：修复后新共存预检真实启动

训练源码 b37b171fd68cb29e5c8142c66778aac6da9a6f06 干净已发布；本地89/服务器89检查通过。新queue v2-ffn-coexist-b37b171-20261011T100325Z；两张卡被指定外部任务占用时启动，先F1→BR→N1十步/恢复/300control。已修复原始GenePT输入绑定及子进程身份竞态；模型src未改。每卡微48×累积2×双卡=192，四轮协议不变。仅预检运行，未放行正式训练；Luna唯一短监督，长monitor仍PAUSED，主不重复查询。旧091失败保留。详见evidence/ffn-dual-coexist-20261011/b37-startup.json；权威状态在canonical coordination/state.json。

## 2026-10-11：真实共存启动后修复 GenePT 配置

0916536 队列 v2-ffn-coexist-0916536-20261011T095120Z 首个 F1 预检在0/10更新失败，原始2048先验守卫拒绝继承的PCA256路径；不是容量失败，不代表共存通过。两张卡指定 ScButterfly 进程在启动/退出遥测均存在。旧运行、源码、收据完整保留；BR/N1及正式训练均未启动。三组自包含配置明确改用已核验SHA34d4c81…的原始17730×2048先验，仍按冻结图ID选择；不改变投影模型、batch或其他训练协议。同时修复子进程启动身份采集竞态，防止身份None进入资源回收路径。新版本重新发布及三个十步/恢复/300control实际推理待完成。长监控PAUSED，主拥有执行权。

## 2026-10-11：双卡各约700MiB任务共存预检已获授权

旧exclusive等待controller已身份核对停止，无CUDA/正式训练启动，旧队列保留。新双GPU身份密封policy与测试→主验收→正式训练屏障实施；每卡foreign<=1GiB、ours<=85%、physicalfree>=2GiB、5s守卫，方法/batch/BR-F1-N1四轮配置不变。73本地守卫/队列检查通过，待新版本发布、真实双卡十步/恢复/300control推理。测试用Luna，正式运行才交长监督。详见FFN_FOUR计划追加范围。

## 2026-10-11：BR/F1/N1 四轮替换任务实施中

用户授权仅三组，原bias、无截断、恢复功能B0骨架，保留batch192及共享GenePT(4096/1024隐藏，BR GELU)。旧N0完成、CG1用户停止、未启动其余组，source与产物保留；主已ACK返回owner，两个旧monitor PAUSED。新native FFN/config/四轮lifecycle已实施，107定向检查通过；回归/发布/服务器三组10步与实际300-control推理/正式启动待完成。计划：plans/GRADPERT_V2_FFN_FOUR_20261011.plan.md。最新权威运行状态以canonical coordination/state.json为准。

## 2026-10-11 02:39：m48预检通过，N0正式fit启动

CG1/N0各10/10＋checkpoint恢复＋实际300control推理已独立验收；训练源码bf4218557210ba266a98d6cc2d1ac15a976c5c91，world2/m48/accum2/global192。只N0→CG1→U24→MR1→P1各6轮无验证、epoch6 last-only。旧m68/.85cap OOM保留；无外部进程，不声称真实共存收益。Luna已结束，canonical Byte owner转指定监督会话，现有20min监控待ACK激活。详细实时状态以canonical coordination/state.json为准。见UNIFIED_DDP报告及shared-gpu-20261011/m48-main-acceptance.json；科学结果未完成。

## 2026-10-11：m68共存上限预检失败，统一m48候选准备中

新SHA6df3bdd…的CG1在0/10更新CUDA OOM，触达85% allocator上限（26.65GiB），物理显存仍余3.87GiB；外部CUDA进程为空。未启动N0或正式训练；失败队列/日志/收据保留。五组统一回退候选micro48×2卡×accum2=global192，不放宽共存门槛。新的十组自包含配置仅改microbatch和effective batch，已逐组深比对，原方法/六轮/数据/评估不变。待发布新源码和fresh CG1/N0预检；Luna短监督终态已退出，长期监控PAUSED。科学任务未完成。

## 2026-10-11：有边界的GPU1共存策略，本地验证完成

主会话已接回执行权，旧等待控制器2096296已停止且从未启动CUDA子任务；旧队列/源码保留，监督监控PAUSED。用户授权与低显存ScButterfly共存测试；新增可选密封策略：仅指定GPU1/UID/脚本、最多一个外部进程且<=1GiB，双卡实际空闲>=2GiB，我们allocator<=85%，5秒检查，越界只停我们自己的身份匹配进程。本地45定向测试/Ruff/diff通过，待新源码发布和服务器预检。ScButterfly在旧控制器停止后仍存活，18:08 UTC已退出；当时我们没有CUDA任务，因此不能声称它失败或认定退出原因，也不能将后续空闲预检当作实际共存证据。科学配置与五组顺序不变，先CG1/N0各10步+restore+300-control，再N0→CG1→U24→MR1→P1各6轮last-only。详见GRADPERT_V2_UNIFIED_DDP_20261010.plan.md。

## 2026-10-10：统一 MLP 恢复双卡累积（当前覆盖单卡计划）

旧单卡控制器/N0/U24已按用户指令停止，停止核验SHA256 eb4f0e46a67c0618ed6671a0a40e01a98344af70944b3e0501cab55159af9fa6。旧产物保留。新十组配置：world2、每卡微68、accum2/global272；先N0→CG1→U24→MR1→P1，用户明确先只跑这五组；其余五组不派发。CG1 rank-local control对齐及P1完整群体MMD适配已完成，本地验证/发布阶段；CUDA10步预检及正式训练尚未开始。Owner主会话，旧监督/返回监控PAUSED。详见主实现工作树计划 GRADPERT_V2_UNIFIED_DDP_20261010.plan.md。

## 2026-10-05：唯一合并E23从头20轮队列已交监督

训练SHA e27f49166dae688eb35444331a1ef1916238efe7，固定config7214e3…c08df8。合并四头16384和SSL1/SSL2各(1,1,0)，mHC4/global272/cap40；本地114、服务器133检查通过。队列 v2-cap40-e23-twenty-e27f491-20261004T203648Z-5d34d6ae attempt1 已真实启动，监督ACK时容量预检24/128，尚非正式epoch。通过后自动fresh20及best/last测试。指定监督会话 owner；20min监控ACTIVE与exacttarget/run已由主会话复核，返回fallback PAUSED。无需主会话重复查服务器。当前共享权威状态位于 /Users/elan/.codex/worktrees/v2-gene-exposure-eval/grad-pert/.byte-os/coordination/state.json；发布状态只是交接快照。证据 .byte-os/evidence/v2-cap40-e23-twenty-20261005/、ACK .byte-os/coordination/receipts/v2-cap40-e23-twenty-supervisor-ack-20261005T0447Z.json。历史六轮结果不变，旧分别两臂20轮解释已覆盖。

## 2026-10-05：用户澄清为②＋③合并、从头20轮

仅一个E23配置：prototypes16384，SSL1/SSL2各(1,1,0)，mHC4；其余cap40/global272/joint-only/frozen tests保持。此前两条20轮请求解释已被覆盖。准备合并配置和资源门槛队列，尚未启动新作业。

## 2026-10-05：E2/E3 20轮新请求准备中

用户授权②③20epoch；从头20轮或完整状态6→20的起点待用户选择。fixed20 schema/planner/native/collector已发布，代码SHA f3620b0d439778a7f1ed21d894e3d06bcdea0b3f；本地112、服务器119定向检查及Ruff通过，服务器源码前后干净。旧六轮结果冻结，新配置/双卡预检/正式队列待起点确定后准备。当前GPU1有其他任务，继续既有空闲门槛。没有新训练或监控启动。计划 `.byte-os/plans/GRADPERT_V2_CAP40_E2_E3_TWENTY_EPOCH_20261005.plan.md`。

## 2026-10-05：cap40 三组消融全部完成并独立验收

B0/E1/E2/E3均6/6、1056更新、min-joint best=last epoch6、真实best/last测试、相同冻结数据/300-control有序清单与表达分组、checkpoint哈希/来源核验、zeroPKL。替换队列COMPLETE；E1未重跑。最终比较和loss分项已归档：`docs/experiments/GRADPERT_V2_CAP40_THREE_ABLATIONS_20261001.md`；独立证据 `.byte-os/evidence/v2-cap40-three-ablations-20261001/final-comparison-20261005/independent-acceptance.json`。单seed描述性结果，E3为组合消融，不按测试调参。正式墙时分解/峰值缺失已明确记录；E2预检128通过。Owner为主会话，整体任务已完成，无后续作业；监督及返回监控均DELETED。此记录覆盖以下未完成/等待/监督状态。

## 2026-10-03: collector repaired, E1 accepted, E2/E3 queue running

Repair SHA2871c4e79a70b7d0d4f900866ce019316b68670f published and immutable server source matches;110 local/server directed tests/Ruff passed. Native src tree unchanged; collector fixed6 bug repaired without changing scientific protocol. Main independently accepted original E1sixepochs/bestlast true tests/hashes/zeroPKL; it remains train/evalSHAac220f5. VPN/SSH connection is verified after Mac unlock, superseding prior connection block. New queue v2-cap40-remaining-2871c4e-20261003T072407Z attempt1/controller742925 only E2/E3 freshIDs plus validated imported E1; old failed queue preserved. Luna startup verification finished. Exact supervisor independently ACKed live E2 capacity preflight24/128, source/runtime/E1 linkage and ACTIVE20min monitor; formal E2/E3 has not started at ACK. Return fallback remains PAUSED. Owner supervisor; main does not duplicate queries. Automatic E2capacity->E3integration->E2/E3fresh6epoch/tests; failures return to main. ACK .byte-os/coordination/receipts/v2-cap40-remaining-supervisor-ack-20261003T0735Z.json; E1 acceptance .byte-os/evidence/v2-cap40-three-ablations-20261001/e1-independent-acceptance-20261003.json. Overall three-ablation experiment unfinished; prior waiting/failed/blocking snapshots below are historical.

## 2026-10-03: E1 terminal collector repair in progress

Original cap40 ablation queue attempt1 failed only after E1 six-epoch child exit0, during collect_run missing fixed6 support; E2/E3 formal not started. Main acknowledged exact handback; both monitors PAUSED. Collector repair and safe completed-E1 import now pass110 tests/Ruff, preserving native src/configs and old run identities. New remaining E2/E3 queue requires server acceptance/immutable release and launch/ACK gates. SSH reset plus locked Mac currently prevents VPN recovery; user asked to unlock only, no duplicate run launched. Evidence: .byte-os/evidence/v2-cap40-three-ablations-20261001/collector-repair-validation.json. This supersedes old queue-running snapshots below; overall ablations remain incomplete.

# 2026-10-01: active approved cap40 three-ablation queue

Owner is exact supervisor codex:01a0df0b-4142-7df1-86c0-d959471d80a1. Queue v2-cap40-ablations-ac220f5-20261001T060835Z attempt1/controller368656, source ac220f5e90e9e5c0c9900cdb2c285a629297c93d clean. CPU waiting_for_idle_gpus is verified; no GPU child. Native preflights then E1/E2/E3 fresh6epochs and best/last tests advance automatically. No E2 capacity claim or silent fallback. Supervisor-owned ACK saved and future monitor grad-pert-v2 ACTIVE20min confirmed; return fallback PAUSED until terminal handback. Queue/source/config identities and exact row roots in coordination/state.json and launch receipt. Main accepts terminal outcomes, fixes core failures, compares final results to accepted B0. No other experiments authorized; historical stopped/completed states below remain historical.

# 2026-10-01: execute approved cap40 three-ablation stage

Main owns build/publish/launch. Existing B0 is accepted; E1/E2/E3 execution now authorized. CPU-only resource wait controller will preserve occupied GPUs and automatically advance exact preflights then formal rows. E2 capacity failure stops without changing batch or losses. New row/run identities and long-run supervision must be verified after launch. No Goal requested.

# 2026-10-01: cap40+mHC terminal independently accepted by main

Read-only server audit verified6/6,1056updates,minimum-joint best=last=epoch6,real best/last tests,source/config/data/selection/checkpoint identities,exit0,COMPLETE,noFAILURE,zeroPKL,and stopped target PIDs. Receipt:.byte-os/evidence/v2-mhc-cap40-six-acceptance-20261001/receipt.json. Both monitors PAUSED. No ablation launched; three configs remain design-only with E2capacity unverified. Earlier pending independent review and heartbeat guard mismatch are superseded by this explicit main acceptance.

# Planned 2026-10-01: three independent cap40 B0 ablations

E1 streams1; E2 four independent prototype heads16384 with hidden2048/bottleneck256;
E3 SSL1(1,1,0), SSL2(1,1,0), both stage multipliers1. User corrected SSL2 DINO/iBOT
from2 to1. Each compares directly to currentcap40+mHC B0; six fresh epochs,seed1,
targetm68×accum2×world2=272,existing val/test/selection. No new runs or monitors.
Three self-contained configs passed schema/options/full-parity checks and2 existing
cap40/mHC config tests. GPU integration and E2 sustained capacity remain unverified.
Current run/owner/20-minute monitor remain unchanged; after its terminal acceptance,
return to main for the next stage. Plan:.byte-os/plans/GRADPERT_V2_CAP40_THREE_ABLATIONS_20261001.plan.md.
Local evidence:.byte-os/evidence/v2-cap40-three-ablations-20261001/config-validation.json.

# Live cap40 mHC six-epoch B0 2026-09-30

Run nadig_jurkat-seed1-20260930T112531Z-f7ca7a77fa6848bb811f706d00e5daad, attempt1, source aeac5fd94123af0b73810259e5e2985228b11d65.
Root /data/yilangliu/GraD-Pert/runs-v2-mhc-cap40-six-aeac5fd/nadig_jurkat-seed1-20260930T112531Z-f7ca7a77fa6848bb811f706d00e5daad.
Source /data/yilangliu/GraD-Pert/development/source-v2-mhc-cap40-aeac5fd, configSHA 1401174171770bb5c7ae580db55f9dd9d4907d8da6a37a42b86a64d747f412d0.
RuntimeSHA 19b87960e43dcd36bda67a87645c9b205e1ec55a4bf6b2817407034534e7b89c, publicationSHA 095782e81d633c3336ab7b2823d95f65e0be67fef82ab2ec569376bfaeb2b213.
PID225555, log /data/yilangliu/GraD-Pert/development/v2-mhc-cap40-aeac5fd-six-formal-r1.log, same stem .pid/.exit.json.
Dual RTX5090 GPU0,1: m68 x accumulation2 = global272. mHC4, full loss, seed1,
6fresh epochs/176updates each; original val/test/reference, joint_only validation.
95local/server checks and dual complete-objective integration/checkpoint reload pass.
Start confirmation: epoch1 step2, ranks225585/225586 alive, no terminal markers.
Luna verification ended. Main prepares exact supervisor20min lease; final result
requires6/6,best/last true tests, hashes, COMPLETE/exit0 andzeroPKL.
Handoff .byte-os/coordination/handoffs/2026-09-30-v2-mhc-cap40-six-supervise.md.
Old no-auto-launch and old3+3 monitor states below are superseded history.

# Active 2026-09-30: cap40 mHC fresh six-epoch B0

User authorized cap40 and mHC. Implementation is v2-only frozen train row IDs,
47836/1335 conditions; canonical val/test/control and evaluation references unchanged.
Exact X/obs/var parity verified between analysis/training canonical; uns differences
remain separately recorded. Bound selection SHA aa0916dfb91fff78d7a43f7d7325879a11c1a904e9bf31586420ca11cce84132.
Next: tests, scoped main publication, immutable server, dual m68 a2 integration,
fresh six-epoch full B0 and designated20-min supervisor handoff. No run launched
yet. Main owns build; old monitors paused. Plan GRADPERT_V2_MHC_CAP40_SIX_EPOCH_20260930.plan.md.
This new authorized stage supersedes prior no-auto-launch analysis-only status.

# Delivered 2026-09-30: Jurkat proportional sampling and four-version comparison

Analysis SHA8638c6557d779ed2290a00b0882964597ca81d3b, clean local/GitHub/server.
Server source: /data/yilangliu/GraD-Pert/development/source-jurkat-proportions-8638c65.
Queue development/jurkat-proportions-8638c65-20260930T103836Z, exit0.
Runs development/jurkat-proportion50-8638c65-20260930T103836Z and proportion25
same suffix; comparison jurkat-sampling-comparison-8638c65-20260930T103836Z.
50%64475rows,delta=.262513065;25%32246rows,delta=.166474252.
Original128266/.373931664;cap4047836/.261779657.100split repeats,1331valid,
4singletons missing;sameoriginal5000genes/control/reference assignments.
All1335conditions/nontrain110711rows/6506variables and retainedmatrix/metadata
exact.12tests local/server; exit0/COMPLETE/source/hash/parity/quotas/CSV accepted.
ConfigSHA50%=93201429299fb0bf8345f62d2adcb14f0286a5572526bd935672084b86367584.
ConfigSHA25%=7d4cb61931b6aa4094abe8b8bfe9d5f64ec5a3717068d8cc87f190931a9da0aa.
Acceptance and all complete artifact identity index:
docs/experiments/data/jurkat-sampling-comparison-8638c65-20260930T103836Z/.
Chinese report/PDF:docs/experiments/JURKAT_SAMPLING_COMPARISON_20260930.md.
All small transfers reviewed/hash-verified; scientific large files serveronly.
Luna supervisor ended; no active task/automation/training/defaultswitch.
Current requested analysis delivered; do not auto-launch mHC B0 or new dataset.
Prior states below are historical evidence, not active wait ownership.

# Live CPU analysis 2026-09-30T10:38Z

Source 8638c6557d779ed2290a00b0882964597ca81d3b, clean three-way identity.
Server source /data/yilangliu/GraD-Pert/development/source-jurkat-proportions-8638c65.
Queue /data/yilangliu/GraD-Pert/development/jurkat-proportions-8638c65-20260930T103836Z.
PID214600, queue .plan.json/.pid/.exit.json/.driver.log.
Runs: jurkat-proportion50-8638c65-20260930T103836Z, then proportion25 same suffix.
Comparison: jurkat-sampling-comparison-8638c65-20260930T103836Z, same development root.
Queue planSHA6778aa63bf88710ebb78a85ccc022c422eacd4535c7b27eab417b2e8030af8bd.
CPU-only4threads, seed42,100splits. Local/server12checks pass.
Luna cap40_cpu_watch owns bounded read-only supervision; main waits without
repeat server querying. Next terminal main acceptance -> four-version report.
No GPU training, default switch or scheduled monitor. Prior snapshots follow.

# Active stage 2026-09-30: Jurkat proportional sampling analysis

Additional methods recovered from prior discussion: condition-wise50% and25%.
Only training perturbations sampled, round-half-up and minimum2whenpossible;
all controls/nontrain rows and retained values remain exact. CPU-only4threads,
100balancedhalf-splits, same original control/gene reference as cap40.
12local checks passed. Next: scoped main publication, clean server tests,
sequential new analysis IDs with Luna read-only bounded waiting; terminal
acceptance and four-version Chinese report. No training/default change.
Plan: .byte-os/plans/GRADPERT_JURKAT_PROPORTIONS_20260930.plan.md.
Historical states below remain original evidence.

# Current 2026-09-30: Jurkat train-only cap40 analysis delivered

Accepted run: jurkat-cap40-5a8a7bb-20260930T094437Z.
Server root: /data/yilangliu/GraD-Pert/development/ plus that run ID.
Clean analysis SHA: 5a8a7bb6712bc4facdfeb13d8dd629bd0f250880.
Configuration SHA256: 1e692732fca3fad57a713ada9475be71a9bce785563d7a5d1b53ec01850cb80f.
Derived H5AD SHA256: 9cb901ead750d82f96cf85016e409b863643e340c043ff8c3aef48d5e48ef98b.
Source directory: /data/yilangliu/GraD-Pert/development/source-jurkat-cap40-5a8a7bb.
Acceptance receipt: docs/experiments/data/jurkat-cap40-5a8a7bb-20260930T094437Z/
jurkat-cap40-5a8a7bb-20260930T094437Z-acceptance.json.

128266→47836 training rows, all 1335 conditions; total158547×6506.
100 split repetitions, seed42, same original context-matched control pool;
original/cap condition-equal Pearson delta=.373931664/.261779657 (1331 valid).
Four singleton conditions omitted; 866 conditions lose at least one batch.
Full retained values and obs/var exact, original H5AD/manifests unchanged,
exit0/COMPLETE/source-clean/zero-PKL accepted. Main independently verified
identity, all rows/partitions/quotas/metadata and267000 score rows.
All small downloads had explicit size/hash checks; H5AD/selection/repeat CSV
remain on server. Report and full plot-data index:
docs/experiments/JURKAT_CAP40_RESULT_20260930.md (Chinese two-column PDF linked).

Luna bounded supervisor finished; no monitor, GPU training or automatic B0
launch remains assigned. No default dataset or model/training setting changed.
Analysis-only derivative is not canonical_ready. Next action is scientific
review of this tradeoff; the mHC joint-only B0 remains ready and unlaunched.
The prior two failed analysis directories/source and transport evidence remain
preserved; following sections are historical snapshots.

# Current repair: preserve categorical metadata on cap40 export

Run jurkat-cap40-af41538-20260930T093845Z computed all scores but failed strict
obs metadata equality after AnnData slicing pruned unused categories. Old outputs
are unaccepted and preserved. The repair restores original obs/var dictionaries
before native writing; no weakened assertions or changed expression/statistics.
Regression includes removed-only and unused ordered levels; 8 tests pass.
Next: publish, server tests, new CPU run and full terminal acceptance. No training.

# Current repair: cap40 canonical gene-symbol identity

The first CPU run jurkat-cap40-1af4195-20260930T093216Z failed before
sampling: H5AD var index uses ENSG while the canonical gene axis uses gene_name.
Fix uses the frozen registry symbol column and validates expression/observation
order hashes. The same 8 tests now include distinct ENSG/symbol metadata and pass.
Failure evidence remains under its old run ID. The server GitHub transport also
timed out; a hash-verified Git bundle restored publication without touching old
checkouts. Next: publish this repair, same-source server tests and a new CPU run.
No GPU training or default dataset change. Original cap40 scope stays intact.

# Current 2026-09-30: approved train-only Jurkat cap40 analysis

Main owns implementation/publication and terminal acceptance. A single bounded CPU
analysis will be supervised read-only by a Luna same-task agent; no cross-chat
monitor or new training is assigned. Cap40 selects min(40, Np) perturbation rows
per frozen training condition with proportional batch quotas, seed42, without
replacement. All other rows/genes/values stay exact. Compare original/fixed sample
with 100 disjoint batch-balanced half splits and condition-equal Pearson delta;
same original context-matched control pool for both. Full-expression Pearson,
retained/removed and overlapping sample/full diagnostics are auxiliary.

Pre-change published source: 5d602a02158af679a4d384f677176ec910fa238a.
8 targeted local tests pass. Next: publish scoped code/docs to main, immutable
server checkout and same-source tests, launch exactly one CPU analysis, inspect
COMPLETE/exit0 and exact data invariants, retrieve only small summary/figure files,
then deliver Chinese comparison report. No model/default data change or GPU run.
The mHC joint-only B0 implementation below remains ready but unlaunched.

# Current 2026-09-30: next fresh B0 restores mHC and uses joint-only validation

Owner: main chat. The requested bounded update is implemented and locally
verified and published on main as
`49456353927ea893f470247a9e4d13e547041dc8` (remote identity verified).
No training, server mutation or monitor was started by this update.
The next B0 config is
`configs/v2/mhc_joint_only_jurkat/three_epoch_m68_a2/gradpert_v2/nadig_jurkat.yaml`,
SHA256 `156c7a0314073fb2901f6d56f8667c657a1d4878c283a1d28ac67b3bdd1ba01c`.
It restores Student/EMA Teacher four-stream mHC, retains full prediction + SSL1
+ SSL2, and chooses `validation_mode=joint_only`: fixed-view joint validation
and its components only, joint-loss best selection, no val population prediction
or Pearson/reference materialization. Terminal best/last tests retain the frozen
300-control, six all/DEG Pearson, exposure groups and zero-PKL protocol. Global
batch272 is micro68 per GPU × accumulation2 × two GPUs; three epochs, seed1,
existing warmup/cosine and loss weights. The historical mHC m74 OOM and m68
three-epoch completion justify the config choice but do not prove new-source
capacity. Historical no-mHC configs/checkpoints and v1 remain unchanged.

Pre-change pushed main identity was independently verified as
`6d5cb7e4d05dcaeb7c8105dc04562a2627e9459a`. Targeted tests: 107 passed; scoped
Ruff/format and six-source-module mypy pass; wheel/sdist build pass. Expanded
checks: 538 passed, 11 CUDA/Triton skips, five unrelated failures reproduced
at the exact baseline. Details: `docs/experiments/GRADPERT_V2_MHC_JOINT_ONLY_B0.md`.
Next action is delivery of this update; any future formal run
needs a clean matching server release, same-config CUDA integration check and
new run ID. No supervision lease is needed while no job is assigned. Existing
unrelated coordination state/receipts are excluded from this code publication.

# Current 2026-09-30: full-state v2 B0 3→6 continuation accepted

The exact run `nadig_jurkat-seed1-20260929T143029Z-d83c2632bba14bddb595f8b8c0734805` completed attempt 1 with exit 0, cumulative 6/6 epochs, protocol-selected joint-loss best epoch 2, epoch-6 last, both six-Pearson test receipts, matching clean source/config/checkpoint identities, no failure marker and zero PKL. The main chat independently verified the server artifacts and hashes against `.byte-os/coordination/receipts/v2-continue6-c47f84e-terminal-20260930.json`. The run-specific `grad-pert-v2` monitor is PAUSED; no training or ablation was started on handback. Results and interpretation: `docs/experiments/GRADPERT_V2_NO_MHC_B0_CONTINUE6_RESULT_20260930.md`. The next action is scientific discussion or a separately authorized experiment, not further polling of this terminal run. The prior “running” section below is historical.

# Current 2026-09-29: full-state v2 B0 continuation from epoch 3 to 6 is running

Outcome: continue the completed no-mHC Jurkat B0 from its epoch-3 `last.pt` through epochs 4–6 at constant LR `2e-4`, then verify cumulative 6/6 epochs, joint-loss best selection, best/last test receipts, source/config/checkpoint/data identities, and zero PKL. This is a new 3+3 training stage, not a retroactive six-epoch cosine run. The immutable parent run and v1 remain untouched. Published implementation/source commit `c47f84e796aed994fdd060a9afbe14a95fa64069` adds the explicit six-epoch policy and self-contained config; server tests passed 79/79, dry-run passed, and same-config dual-GPU full-state one-update probe restored the parent, advanced optimizer step 1302→1303 at LR `0.0002`, changed trainable parameters, and saved/reloaded a checkpoint. Probe receipt: `/data/yilangliu/GraD-Pert/development/v2-continue6-c47f84e-full-probe-r1/COMPLETE.json`.

Active run ID `nadig_jurkat-seed1-20260929T143029Z-d83c2632bba14bddb595f8b8c0734805`, root `/data/yilangliu/GraD-Pert/runs-v2-continue6-c47f84e/` plus that ID; config SHA256 `66b6abdd0070d111a8275ba85739c6279cd5373559464b62eb0bb2be8473f956`; parent `last.pt` SHA256 `5daef1bac40c19a20ba4ee54a2637e1f187e0d1f502010ba06708fc45a303978`. Server checkout `/data/yilangliu/GraD-Pert/development/source-v2-continue6-c47f84e`, publication receipt SHA256 `e1001c1e4a962d729c2feb57b2ca17e96716978b586aff0f9d6dc9e473c6b23b`, runtime SHA256 `73d83138c6736337ffd25f1d495a3f9fd1a50fa2ace9f886b3016d48f4d1466f`. GPU 0,1; microbatch 74 per GPU × accumulation 2 = global batch 296; seed 1. Wrapper PID file, log, exit JSON share stem `/data/yilangliu/GraD-Pert/development/v2-continue6-c47f84e-formal-r1`; driver plan has exact run identity. Launch verification saw parent epoch 3 committed, epoch 4 live, wrapper/torchrun/both ranks alive, both GPUs computing, no exit/COMPLETE/FAILURE. The prior three-epoch parent is never resumed in place. Handoff: `.byte-os/coordination/handoffs/2026-09-29-codex-v2-continue6-supervise.md`; launch receipt: `.byte-os/coordination/receipts/v2-continue6-c47f84e-launch-20260929.json`. Supervision lease target is `codex:01a0df0b-4142-7df1-86c0-d959471d80a1` with monitor `grad-pert-v2` every 20 minutes only while this run is active. On terminal success or failure, supervisor returns the baton to `codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b`, pauses the monitor, and the main chat performs independent acceptance/repair. No other ablation is authorized by this run.

# Current 2026-09-29: v2 reusable training and evaluation interfaces delivered

Overall outcome: v2 single-/multi-GPU training and evaluation instructions, hash-bound independent checkpoint evaluation, and new full-state/LoRA training stages are implemented. Exact same-run resume remains distinct; old B0 and v1 behavior are preserved. Baseline `1b0f7064ff42311b8ac3615a13e84f5943ca43ba` was verified pushed before edits. Implementation `1cba4272f4d4c28281ab5efca0cf13057cb605d0` and evaluation progress/environment follow-up `5852acd46c9a1a8f1fa79827b76e4c502ad049fb` were published on main, with clean immutable server checkouts at `/data/yilangliu/GraD-Pert/development/source-v2-checkpoint-workflows-{1cba427,5852acd}`. Both 3+2 modes passed server dry-run. Actual dual-GPU one-update probes restored completed B0, changed trainable parameters at LR `2e-4`, saved/reloaded checkpoints, and reached optimizer step 1303; hash-verified `COMPLETE.json` files are under `/data/yilangliu/GraD-Pert/development/v2-checkpoint-workflows-1cba427-{full,lora}-probe-r1`. Single-GPU dry-run also passed. Full independent dual-GPU evaluation of the old `last.pt` under immutable `1cba427` completed at `/data/yilangliu/GraD-Pert/development/v2-checkpoint-workflows-1cba427-dual-eval-r1/COMPLETE.json` (SHA256 `21fab046917048aa5cb5aa55cec485bbd2d1aa3bf7b0342ddc4d5259738ae7e4`): all 592 conditions and all 9,920 numerical result values exactly match the original `fit/last-test.json`; frozen control/reference hashes, query recipe, and expression-exposure record also match. DEG metrics are finite for 590/592 conditions, as in the original. Entire evaluation root has zero PKLs; its processes exited and GPUs returned idle. The follow-up `5852acd` added progress and environment metadata and passed targeted server tests/dry-run; this metadata-only follow-up did not itself receive a full 592-condition GPU run. Expanded v2 suite had 452 passed, 11 skipped, and one unrelated pre-existing failure because `configs/v2/single_pass_jurkat/capacity_m66_a2/gradpert_v2/nadig_jurkat.yaml` is absent from baseline HEAD; targeted Ruff/format and wheel build passed. No formal 3+2 training or monitor is active. Do not stage unrelated historical `.byte-os/coordination` files.

## Historical completed B0 (2026-09-29)

The no-mHC single-stream/joint-validation run `nadig_jurkat-seed1-20260928T184412Z-2d6425f1e28e46b0b285af6096d03bd3` attempt 1 has completed three epochs (1302 optimizer updates), exit 0, with `COMPLETE.json` and both best/last tests. The main chat independently re-read the server's `COMPLETE.json`, `fit/epoch_state.json`, `fit/history.json`, and best/last test JSON and matched their SHA256 values to the supervisor's terminal receipt. The entire run root contains zero PKLs. The clean training/evaluation source is `ca7884e4e9b70bb55a61d97442467ff2531337b3`; the config SHA256 is `4c5ba9b39689b9d7993e023694cdedd0268668aa0f9e2f922e6c82e5cb25887e`. The protocol-selected best is epoch 2 by minimum validation joint loss 4.0076390792; last is epoch 3 with validation joint loss 4.0519638342. Full results and their interpretation are in `docs/experiments/GRADPERT_V2_NO_MHC_B0_RESULT_20260929.md`. The `grad-pert-v2` monitor is PAUSED because this run is terminal. No rerun or further ablation was started. The supervisor's terminal receipt is `.byte-os/coordination/receipts/nadig_jurkat-seed1-20260928T184412Z-2d6425f1e28e46b0b285af6096d03bd3-terminal-20260929T1032Z.json`; the completed-run handoff is `.byte-os/coordination/handoffs/2026-09-29-codex-v2-no-mhc-b0-complete.md`. Earlier sections below are historical snapshots, not the current run state.


## 2026-09-29：主会话完成 v2 验证/测试输出协议更新

主会话已发布 `10193b6edffca802410fef837fd918844d4cc6e3`：新 v2 运行的 checkpoint 选择改为固定视图完整 joint validation loss；best/last 默认分别输出全表达轴与按实际训练表达暴露划分的两组各三种 Pearson；进度显示总/轮内 step、速度、比例和验证/测试阶段。Teacher/center 在验证时只读，验证 RNG 不推进训练 RNG；一次条件推理同时形成三组指标。隔离服务器 v2 回归 434 通过，扩展相关测试 48 通过（既有缺配置测试 1 项跳过），最终定向 37 通过；Ruff 全仓通过。`mypy` 剩余 29 条位于未改旧模块，未宣称全仓 typecheck 通过。新源码已推送 main；**未启动训练或评估**。历史 B0 与已有分组结果保持原身份，不按新协议追认。若后续获授权启动新 B0，先使用新 ID、干净服务器 checkout、配置哈希与新 joint validation 容量预检；现无长时间监督租约。

## 2026-09-28：主会话接回结果；当前无活动训练交接

监督会话完成 `v2-best-expression-exposure-9ea0815-20260928` 终态核对并交回主会话：服务器结果 `best-stratified-test.json` SHA256 `603ffa7e2cba7f78e53f0566a73b56fdceaab0af00eb99ba1fe791bc84e8f840`，exit0、zero-PKL、GPU 空闲；`grad-pert-v2` 心跳 PAUSED。按先前授权已实现并发布后续 v2 运行的步数/阶段/损失/吞吐与验证、测试条件进度；源码 `87b44c8c1e8840c876bb63cdf783486aa40f7364`。当前没有新 GPU 任务或监督租约，旧 run 不重启。下一动作由主会话与用户讨论分组结果或安排明确授权的后续实验，不能以本分组指标调参测试集。

## 2026-09-28T04:39:59+08:00：m74 正式三轮 OOM 后的容量回退

m74 已在首轮中途 OOM，128 步通过不能证明整轮安全；旧 run 和 FAILURE 保留，不恢复。当前用已发布来源门版本 `8dfb267adb25591393602066ef1226e7cbb76174`，对 m68／图目标行64／序列chunk16 启动独立双卡128步持续探针，服务器根 `/data/yilangliu/GraD-Pert/development/v2-gate-m68-r64-s16-128-after-b0-oom-r2`；r1 因启动环境缺少 PYTHONPATH 在导入阶段失败，r2 已补齐。准备了新的三轮 m68 自包含配置，但须以探针终态、显存余量与吞吐决定是否采用；不能把进行中视为通过。本次只改变物理/全局 batch，完整预测＋SSL1＋SSL2、累积2和既定图/序列chunk不变。


## 2026-09-28T04:34:09+08:00：三轮 B0 attempt1 因 OOM 失败，交回主会话

精确运行 `nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143` 退出码1，`fit/epoch_state.json`停在0/3（434 updates/epoch），无epoch-0001、history、COMPLETE、best/last测试或test root；全run根扫描0个PKL。rank0 在GPU0上 dropout 申请128 MiB时OOM，报告仅余9.62 MiB，进程占31.34 GiB；rank1随后在NCCL ALLREDUCE seq5573超时300027 ms并SIGABRT。训练进程已全部退出，双卡回到0%/2 MiB。源码与配置身份匹配。failure JSON原件保存在服务器且19,977字节已核验SHA；小型副本和日志哈希索引记录在 `.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-terminal-failure-20260928.json`。本run不重启、不评估、不修改配置；等待主会话决定新的有界步骤。监督心跳暂停并交回。详细handoff：`.byte-os/coordination/handoffs/2026-09-28-b0-m74-three-epoch-oom-return.md`。


## 2026-09-28T04:12:23+08:00：服务器连接恢复，三轮 B0 仍在训练

TCP/22与SSH已恢复。精确进程 wrapper3829919、torchrun3830019、rank3830095/3830096均存活；`epoch_state`仍0/3（434 updates/epoch），history为空，只有epoch-0000初始checkpoint；无exit、COMPLETE或failure，测试根尚未创建。源码干净且提交/config SHA匹配。GPU0 29%/31944 MiB，GPU1 29%/32084 MiB，compute PID与两个rank对应。有限日志无错误；继续按20分钟监督。收据：`.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-supervise-check-20260928T2011Z.json`。


## 2026-09-28T03:53:52+08:00：服务器连接暂不可用

当前 EasyConnect 服务门户已登录且列出 `10.24.1.91`，但该资源显示 L3VPN 服务启动失败；辅助检查 TCP/22 与 SSH 均不可达。执行一次常规恢复并从已填表单登录后，页面仍报告启动失败；未做路由/DNS/代理更改。训练状态未知；上次可达观察为 epoch 0/3、进程存活且无 exit/COMPLETE/failure，不可视为当前状态。连接证据：`.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-connectivity-incident-20260928.json`。主会话已被通知；监控保持 ACTIVE，下一周期再做有限连接核验。

## 2026-09-28 首次监督确认

首次监督确认（2026-09-28T03:12:05+08:00）：Byte 所有权与 run ID/attempt 门禁匹配；服务器 wrapper、torchrun 与两个 rank 均存活，epoch 0/3；服务器源码干净且提交/配置哈希匹配。GPU0 为18%/30584 MiB，GPU1 为0%/30564 MiB。未见 COMPLETE、failure 或 exit 标记。20分钟心跳已启用。记录：`.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-supervise-check-20260928.json`。

## 当前运行：B0 m74/chunk16 三轮，监督交接中（2026-09-28）

运行ID `nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143`、尝试1、服务器根 `/data/yilangliu/GraD-Pert/runs-v2-b0-gate-m74-3ep-e821173/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143`，父PID3829919，源码 `e821173b4d11268b621ac8299e45b8d6d272b7e7`，配置SHA `3f1d3d9dde6bf151eeb3916809863eda0020fa89d8c82c74005d19c4d424dedd`，runtime SHA `988968f35f1e97558be2ec32deeb285e3a33781624da8a0407314083c8baab6c`，发布收据SHA `d5da951138794e9e5944754e40bc7536c8f32769f5e7e327060d2bf1c56e563b`。启动时launcher/torchrun/双rank存活，run manifest和计划已写，双卡占用；没有完成或测试收据。精确交接 `.byte-os/coordination/handoffs/2026-09-28-codex-b0-m74-three-epoch-supervise.md`，接下来须核对20分钟监控确实ACTIVE且绑定监督对话；终态要求3/3轮、COMPLETE、best/last、训练/评估身份及zero-PKL。任何失败保存证据，复杂修复交回主会话，不覆盖本运行。下方选型段落为历史。

## 当前阶段：m74/chunk16 选定，三轮完整B0发布预检（2026-09-28）

双RTX5090、完整预测＋SSL1＋SSL2、累积2下，来源门源码 `8dfb267adb25591393602066ef1226e7cbb76174` 的m74／图行64／序列16完成128/128步、checkpoint续跑和单条件300-control推理，exit0；峰值allocated/reserved 32,310,246,400/32,621,199,360 bytes。m76/78/80的持续测试OOM，失败原证据保留在服务器各独立ID。12步同路径吞吐排序m74两次13.794/13.791 cells/s，高于m66／序列32的12.687；m70／序列32短测OOM。选定全局batch296；收据与局限见 `docs/experiments/GRADPERT_V2_GATE_BATCH_CHUNK_20260928.md`。用户新授权独立3 epoch完整B0及best/last测试，本地已准备 `v2_fixed_3` 策略和 `configs/v2/source_key_gate_b0_jurkat/three_epoch_m74_a2/gradpert_v2/nadig_jurkat.yaml`；下一动作是定向测试、提交推送main、干净不可变服务器checkout、同配置预检，然后新运行ID启动。旧B0不恢复、不覆盖。当前无本任务GPU运行；下方更早阶段均为历史快照。

## 当前阶段：来源门容量与chunk配对（2026-09-28）

目标是双RTX5090、累积2、完整损失下，最大经过128步持续验证的物理微批，同时比较可用chunk组合的含数据等待吞吐与峰值显存。先前新来源门m66全局264只通过单步；旧图版m66持续通过、m68第22步OOM不可移作新证明。第一组九项双卡单步已在不可变源码`f090b726ba0df0e02a2fb058e727236c9d9dfcef`全部通过，原始receipt在服务器`/data/yilangliu/GraD-Pert/development/v2-gate-capacity-f090b72-{variant}-integration-r1/`；m70 seq16较seq32少用约3.45GB峰值预留，rows32未减峰值且较慢，故追加seq16 m72/74/76/77/78/80及seq32 m72。配置设计与序贯预检→持续容量→同batch吞吐门槛在`docs/experiments/GRADPERT_V2_GATE_BATCH_CHUNK_20260928.md`。下一步发布扩展配置的干净提交，使用新服务器checkout及独立ID查上界，随后选择候选做128步持续容量。训练等待交同任务子代理只读监督，主会话不重复轮询。旧B0保持停止，非容量测试不启动。

## 当前默认：图来源门开启、原分块、batch264（2026-09-28）

用户要求结束大chunk默认尝试，同时保留新图编码器。新版本 `d2410efdc81a9690644ef2d165c2dea374ba0fbb` 已推送 main；自包含配置 `configs/v2/source_key_gate_jurkat/one_epoch_m66_a2/gradpert_v2/nadig_jurkat.yaml` 只在先前 `optimized_single_pass_jurkat/one_epoch_m66_a2` 上增加 `graph_source_key_gate=true`。图邻域KDA scan32、序列继承scan32、图目标row64；旧配置和独立chunk对照保持可复现。每卡微批66×累积2×双卡，全局264；这是旧模型已通过128步容量的batch，**不是新门控已验证的持续容量**。

新配置 SHA256 `f0f69871fdd8e797645d85b2faf434740d685be7c9370b6f063da4ac3fb3e844`；干净服务器源码 `/data/yilangliu/GraD-Pert/development/source-v2-gate-default-d2410ef-from-local`，publication收据SHA256 `020b778d4db5aa78e8ee7e7b31eec057c3a3f81a7c8d84782e3f0b8a3afd6dc8`。双卡单步完整更新1/1通过并重载checkpoint，峰值 allocated/reserved 31,064,793,600 / 31,427,919,872 bytes；收据 `/data/yilangliu/GraD-Pert/development/v2-gate-default-d2410ef-m66-integration-r1/receipt.json`。服务器81项定向测试通过，实验进程已退出、GPU空闲。下一次正式训练前须做新门控版本的持续容量验证；本轮未启动B0或其他消融。下方大chunk工程测试结论为历史，不能再读作后续默认。

## 当前任务完成：第四层来源门与图64／序列256分块（2026-09-28）

用户指定的方法与工程配置已经实施：图第4层边来源逐维key门零初始化并保留来源偏置，图邻域KDA scan chunk64、Cell/Response序列KDA chunk256；图目标row chunk64/96/128各有独立配置。基线提交`9a820e1d2a2dfc38bdcfee9370ecb1e518733b2f`先验已推送；实现提交`657b63ad509f5511b3cb4a585be0dc44a92aff65`及回退微批配置提交`2043e09a4f4cb29317b9879da68e32e8d953ee1c`均推送main。最终干净不可变服务器源码`/data/yilangliu/GraD-Pert/development/source-v2-gate-chunks-2043e09`；publication收据`/data/yilangliu/GraD-Pert/development/gradpert-v2-gate-chunks-2043e09-publication.json`，SHA256`898df95cdfbcba7fb0e8ba54515ca317e4d08489d418a129c247d30d075b6734`。服务器定向80项测试通过。

双RTX5090全局batch128（m32×累积2）下，门控单独单步通过，chunk-only与完整组合反向OOM；失败ID/收据均保留。统一全局batch64（m16×累积2）下六组4步完整预测＋SSL1＋SSL2均通过，最后一步学习率非零，细胞批次顺序哈希相同。参考3.373 cells/s、9.14GB/卡；完整组合3.201 cells/s、26.39GB/卡；row96组合3.716 cells/s，row128组合3.963 cells/s，row128反序复测3.978对参考3.376。图行块扩大虽有重复短程速度收益，但此前严格语义审计未通过逐目标随机分配与有限精度等价；本次无持续128步和一轮效果证明，不改变正式性能默认或启动B0。完整配置/结果/收据/限制见`docs/experiments/GRADPERT_V2_SOURCE_GATE_CHUNK_20260928.md`。当前没有由本任务运行的GPU进程；旧B0/收据原样保留。下一步若用户授权正式采用某候选，先解决随机语义与容量，再以新run ID继续；下方历史计划不表示正在运行。

## 当前阶段：chunk 对照已结案，B0 保持停止（2026-09-27）

主会话完成双卡扫描32/48/64及图目标行64/96/128工程对照。没有通过“语义＋端到端收益＋容量”全部门槛的候选；维持扫描32、图行64和 `relay_sequence_chunk_size=None`。λ₂=1 及蒸馏②实际权重0.8/0.4/0.1已写入未来新配置；以后 v2 新实验按1 epoch，不覆盖旧B0。旧训练已停在0/5，不启动新B0或其他消融。报告 `docs/experiments/GRADPERT_V2_CHUNK_SWEEP_20260927.md`，工程测试源码 `d61af46d9e4282c8e5b69e9514d1134407d2b8e5`，收据均保留服务器。下方“训练运行中”等记述是历史快照，以本节为准；监控保持暂停。

## 当前执行：m68 OOM 后验证 m66（2026-09-27）

主会话拥有容量与后续 B0 启动。NUMA m68 全局272新 ID 128步测试在 rank1 第22步反向 OOM，exit1；`rank-1-failure.json` 是终态，顶层 receipt 停在16步旧状态。原始失败证据保留于 `/data/yilangliu/GraD-Pert/development/v2-numa-m68-28f447a-capacity-20260927T0929Z`；GPU0/1 已空闲。下一动作：从干净已推送 `28f447a24a400659ea09f9c0f6d735ea7e4ec8f1` 只变微批68→66、全局272→264，定向推送 main，建立干净不可变服务器源及发布收据，独立 ID 单步预检→128步持续、checkpoint 续跑及300-control。若 m66 OOM，保留失败并选 m64，在新源码复测。容量通过才启动新 ID 完整预测+SSL1+SSL2五轮及 best/last，长时训练交指定监督会话；旧停训 B0 不恢复。下方 m68 进行中快照是历史记录。

## 当前执行：m68 128步持续容量（2026-09-27 09:29 UTC）

主会话拥有容量阶段。干净发布源码 `28f447a24a400659ea09f9c0f6d735ea7e4ec8f1`、服务器 checkout `/data/yilangliu/GraD-Pert/development/source-v2-numa-m68-28f447a`、publication SHA256 `3e03290ea72f9d03fc05224413c07bbc012bc1213ebc47e44dcabb2a9aeee5e9`、配置 SHA256 `ab8c358ecdd7d6ee398120ebcaafd048bb25814263eb5b2e8271f479662735a9`。m68单步预检独立 run 1/1 exit0，持续 run `/data/yilangliu/GraD-Pert/development/v2-numa-m68-28f447a-capacity-20260927T0929Z` 已启动，父PID3689686；脚本 SHA256 `3c503b566a822048757ffb77e21d513512974f09b5e4d4b0097c746349ab8f42`。只读 Luna 短时监督，主会话收到失败/终态即推进：通过则核对128步、恢复、300-control及可用 batch 后启动新 ID 完整 B0；OOM则保留失败并评估m66或回退m64，不覆盖本 run。两张GPU由本测试占用，无其他消融；旧B0不恢复。下方是历史快照。

## 当前所有权／动作（2026-09-27 09:24 UTC）

主会话持有性能阶段；NUMA A1/B1/B2/A2 队列 `complete`/`exit0`，Luna 短时监督已结束，两卡空闲。比较收据 SHA256 `813bd55bef4715ceb430944d251cc7ee7e319fbd29043996405d0cb3c2886ed4`，同源码 `32b2bfd88f36b9a7fad435c533dc825ef81942b4`、同配置 SHA256 `23a2942804b73cdf92871e36808846f50382ded41b1588a68da76ebc0e69985b`，同有序 batch/视图 RNG 起点；只变 rank 本地 NUMA 绑定。配对吞吐几何比 1.05083，显存无增加，决定采用于当前服务器。下一动作：将文档及自包含 m68 容量候选定向提交推送 main，在新干净不可变服务器 checkout 发布收据并做双卡128步持续容量；失败保留，安全回退 m64。容量门槛通过后新运行 ID 完整 B0 五轮与 best/last。旧 B0 中止不恢复；跨会话旧监督仍暂停。下方章节是历史快照。

## 当前所有权／动作（2026-09-27）

主会话负责当前性能构建；无活动双卡任务或长时监督。短图融合 `27aec3643cc613b17f3b87eecebf05d1453743ba` 的双卡确定性完整更新失败，收据 `/data/yilangliu/GraD-Pert/development/v2-short-27aec36-parity-m2-datafix-20260927/` 保留，不进入容量/B0。KDA 形状常量复用在干净发布 `32b2bfd88f36b9a7fad435c533dc825ef81942b4` 上已双卡两步严格一致，A1/B1/B2/A2 各12步通过；含数据等待吞吐比 1.01051/1.01361，几何均值 1.01206，比较收据 `/data/yilangliu/GraD-Pert/development/v2-kda-constants-32b2bfd-abba-20260927T0826Z/comparison.json`。收益较小，保留 opt-in。后续区域 CUDA Graph 与图边相同基因/来源的预投影都已在局部梯度门槛失败，未推送/未用于正式更新；日志及诊断副本保留在服务器 development。当前下一步：针对双卡跨 NUMA 拓扑与约130万 kernel 发射检查各 rank CPU 亲和调度，做同源串行整步对照；若无收益即停止该方向。再决定持续最大batch与新ID完整B0五轮。旧已停止B0不恢复，旧跨会话监督保持暂停。当前正在运行的任务以实时进程/收据为准；下方旧状态是历史快照。

## 当前执行：主会话性能工程（2026-09-27，新授权）

用户重新授权性能调研、双卡实测、等价机制优化及其后容量与全新 B0 正式运行；旧 B0 仍中止且不可覆盖。主会话拥有 build 阶段，当前无跨会话监督/定时监控；短时测试用 Luna 子代理只读监督。当前源码 `681d4fb609644d51c22f6b4e51dd61256adc3310` 的 m64 128步容量已通过，m72 OOM 保留。m32 完整更新 profile 已 exit0、6/6 passed，收据及 trace summary 位于 `/data/yilangliu/GraD-Pert/development/v2-mech-681d4fb-m32-profile-20260927T0727Z`；普通更新中位17.550秒，GPU/kernel与图发射详情已记录在 `docs/experiments/GRADPERT_V2_SINGLE_PASS_PERFORMANCE.md`。第一候选是独立短图邻域的融合最终状态递推：FP32局部加速约4.5倍，首版BF16 value梯度在极少量舍入边界失败；保留失败证据并采用原块求解计算该梯度的混合候选，相关服务器测试59项通过，BF16局部前后向2.803→1.782ms。尚未证明真实双卡完整更新等价或整体加速。临时checkout `/data/yilangliu/GraD-Pert/development/source-v2-short-provisional-20260927T0750Z` 仅用于不洁诊断测试，不能正式运行。下一动作：发布仅opt-in候选→双卡完整非零LR多步等价→同配置ABBA端到端，未通过则淘汰/修复。之后持续容量与新ID完整B0。细则见 `.byte-os/plans/GRADPERT_V2_MECHANISM_PERFORMANCE_20260927.plan.md`；若启动超过一小时的正式运行，再建立可返回主会话的耐久监控，不使用旧暂停心跳。

## 当前交接：用户暂停实验及监督（2026-09-27）

主会话 `01a0c01a-0611-7a90-b3e8-8ad7e017748b` 已接回所有权。B0 运行 `nadig_jurkat-seed1-20260927T041339Z-e35f5edb8e6248479462589ff3cef41f` 按用户要求 SIGTERM 中止，全部训练进程退出，双卡空闲；已提交 epoch 0/5，仅保留初始 `epoch-0000.pt`，无 COMPLETE、best/last 或测试收据。中止证据见 `.byte-os/coordination/receipts/b0-681d4fb-user-stop.json`（SHA256 `9e792365f37ec635483eb1da5a61dc993e65fa2bb30298e1bc68f09153af5451`）。监督会话停止工作，`grad-pert-v2`、`grad-pert-v2-b0-return` 两条自动化均暂停。下一步等待用户明确重新指示；不得自动恢复或启动实验、重启监督。下方运行中状态均为历史快照。

## 当前交接：完整 B0 五轮双卡训练已启动，交监督会话

m64 持续容量接受条件全部通过（exit0、128/128、checkpoint continuation、300-control、源码/配置身份），容量 receipt SHA256 `31a0798737a60bd318a861fd5e4d947a51f4d943ac0f613d0328ca4a3a7bfc04`。完整 B0 run ID `nadig_jurkat-seed1-20260927T041339Z-e35f5edb8e6248479462589ff3cef41f` 已启动，父 PID 3613045、双 rank 存活、`fit/epoch_state.json` 初始 epoch0/5；五轮及 best/last 未完成。源码 `681d4fb609644d51c22f6b4e51dd61256adc3310`，配置 SHA256 `23a2942804b73cdf92871e36808846f50382ded41b1588a68da76ebc0e69985b`。长时监督边界、路径与返程心跳见 `.byte-os/coordination/handoffs/2026-09-27-b0-681d4fb-supervise.md`。监督会话接手后才开启本运行的 20 分钟心跳；终态即暂停/删除。此节覆盖下方 m64 进行中记录。

## 当前交接：m64 容量测试主会话接手监督

活动 run `v2-teacher-681d4fb-m64-capacity-oomfallback-20260927-1203`，PID 3602712；2026-09-27T04:17:04Z 实查 32/128、双 rank 存活、无 exit/failure。主会话与子代理按新流程负责约1小时内测试的有界监督；跨对话监督心跳已暂停。m72 OOM 与 m64 fallback 证据见 `.byte-os/coordination/handoffs/2026-09-27-teacher-m64-capacity-return-main.md`。容量 acceptance 仍需 128/128、checkpoint reload 与300-control；未启动 B0。通过后主会话立即核验并启动此前已授权的完整 B0 五轮；失败则保存证据并决定修复/回退，不能把一次报告当作阶段终点。

监督规则：预计约一小时以内的有界测试，由本会话子代理只读监督并向主会话回报；更长的训练/消融才交指定跨会话监督。跨会话定时心跳仅在其负责的活动后台任务期间启用；终态或交回主会话时暂停/删除，需要新长任务时再开启。主会话收到任何监督终态后，在同一轮推进已授权的下一依赖。规则已同步到两份项目 `AGENTS.md`。

## 2026-09-27：m72 双卡128步容量测试已启动

Teacher Sinkhorn 优化双卡ABBA comparable，描述性总耗时下降3.84%，保留；m72×2×2=全局288单步预检1/1 passed，峰值预留显存31,279,022,080 bytes。已在干净发布源码681d4fb609644d51c22f6b4e51dd61256adc3310上启动m72的128步持续容量工程测试，PID3599854、运行根`/data/yilangliu/GraD-Pert/development/v2-teacher-681d4fb-m72-capacity-20260927`；当前只确认进程和0/128初始收据。监督交接见`.byte-os/coordination/handoffs/2026-09-27-teacher-capacity.md`。容量/B0仍未完成；此节覆盖下方历史空闲记录。

## 2026-09-27：Teacher Sinkhorn ABBA 已启动，交监督会话

源码 681d4fb609644d51c22f6b4e51dd61256adc3310 的本地 v2 351 测试、服务器 CUDA 16 测试通过；双卡确定性两步完整更新精确通过。非确定性校验与同路径重复对照均未过严格梯度阈值，保留收据，不混作候选失效证据。A1/B1/B2/A2 全模型 12 步串行队列已启动，PID 3531359，A1 收据真实存在；运行根 `/data/yilangliu/GraD-Pert/development/v2-teacher-681d4fb-abba-20260927`。按 `.byte-os/coordination/handoffs/2026-09-27-teacher-abba.md` 交由指定监督会话观察，完成或失败立即交回主会话。容量持续测试与 B0 五轮未启动。此节覆盖下面历史“无活动任务”记录。

## 最新状态：主会话负责工程，监督会话负责后台等待（2026-09-27）

主会话 `01a0c01a-0611-7a90-b3e8-8ad7e017748b` 负责模型设计、论文核对、性能实验设计、实现、验证、发布与启动；监督会话 `01a0df0b-4142-7df1-86c0-d959471d80a1` 只在有明确交接的后台阶段接管检查。Teacher 无梯度 Sinkhorn 存储消除候选的双卡 ABBA 已于 2026-09-27 完成：四组 12/12、`queue.exit=0`、比较为 comparable，描述性总耗时下降 3.84%；这不是正式训练或科学结果。当前没有活动 GPU 任务，所有权已交回主会话，由主会话审阅并决定下一有界阶段。持续容量与完整 B0 五轮及 best/last 仍未完成；不自动启动其他消融。

用户指定不用 Goal 模式；工具当前也报告无活动 Goal。已建立监督会话的 20 分钟心跳 `grad-pert-v2`，启动后立即触发一次。没有交接中的后台运行时仅核对状态并保持安静。主会话每次交接必须写清阶段、运行 ID/PID/日志、不可变源码与配置哈希、GPU/时间边界、预期完成收据、失败处理和下一步；监督会话以服务器实时证据核对。发现故障或里程碑后立即保存证据并消息唤起主会话，不等下一轮；主会话接回后监督会话停止对该任务重复操作。20 分钟是周期检查上界，单靠定时器无法在两次检查间瞬时发现故障；新后台任务如需故障发生即唤起，应在启动时加独立退出/失败事件通知，并先验证其可靠性。用户暂停/停止指令优先；全部任务完成时删除心跳。

上面的双会话安排覆盖下文旧的“Goal active／无定时监控”和阶段 F 的旧检查间隔，不改动历史实验事实。

## 显式mHC配置完整更新通过；下一步无梯度存储消除候选


### 显式后端配置完整更新通过

9d9bad86e16286b590b2bb31c5df5103ecfbbccb干净发布服务器source-v2-opt-9d9bad8，
publication SHA256 da83f52334078f742997eb4d2e3c850abba1b5234c13e09bfaceaa7688977294。
同profiling_m2_a2协议，仅candidate配置从native改auto；无诊断融合开关。
两rank两步全部比较最大差0，输入/RNG相同，第二步LR7.796055196070788e-8。
exit0，收据single-9d9bad8-config-parity/共42414bytes经dry-run归档。
这验证正式配置接入等价；不是新的吞吐或持续容量结果。当前GPU无活动任务。
下一文献指导候选是Teacher无梯度Sinkhorn的无用中间存储消除，详见文献矩阵。

FLA冻结完整前后向已读，直接替换会保留逐token输出，不能宣称省算。
文献矩阵新增具体候选：Teacher无梯度时每token2560bytes反向概率无需保存，
保持40归一化与Student梯度路径；先独立数值/完整更新再ABBA，未实现。
原目标仍为最终执行组合→持续双卡最大batch→完整B0五轮best/last。

# 当前状态 — 2026-09-26

## 授权与目标

历史状态：当时的目标为新方法升级与性能工程 → 双卡持续容量 → 完整B0五轮及best/last；2026-09-27 起改用上方双会话工作流。
不启动其他消融；不恢复旧停止运行。默认单向KDA、最终S统一读取，完整双蒸馏。
此处“无定时监控”为当时记录；现以顶部状态为准。保留历史来源，禁止修改活动服务器源码或覆盖runID。

## 当前阶段：融合mHC Sinkhorn完整双卡更新校验

trace两轮分析均完成，结果在docs/experiments/single-38af3ce-profile-replay/costs/。
1414510 kernels；logsumexp去重CPU区间1.5996s、129600 kernel累计0.166s。
不把嵌套时间相加；主要机制还包括图邻域/重计算/KDA临时量。先验证mHC融合，
保持20轮与解析梯度，不减少迭代、不改模型数学。KDA Gram候选尚未实施。

独立Triton融合算子18/18双设备合成case通过，原始收据
`docs/experiments/single-59305ee-sinkhorn-probe/receipt.json`。
最大输出/梯度差4.92e-7/3.88e-7，原4.33–5.04ms→0.514–0.656ms；
4096token临时峰值14,221,312→11,010,048bytes。非完整训练吞吐，不作采用依据。
首版4f855f9编译scope错误已修，失败收据保留；未改活动源码。
本地61测试通过，服务器18定向测试通过。

前次完整更新67fd8e1失败，第1candidate更新gradient最大差1.6681e-4，
ssl2_koleo亦超容差，输入/RNG完全一致。两rank收据已入库，不采用，不跑其ABBA。
经安装Torch Reduce.cuh证明strided四元素和逐项结合，修正Triton行轴归约；
7ae9a8b微基准18/18组、全部40阶段、输出及梯度逐元素相等，BF16差异0。
证据docs/experiments/single-7ae9a8b-sinkhorn-probe/receipt.json。

动态N cab78a1已通过双卡30组微基准：全部40阶段/输出/梯度exact，
每卡forward/backward各1编译variant。完整两rank两更新也全部max差0（含loss标量），
第二步明确非零LR7.796055196070788e-8。收据已dry-run复核281810bytes入库
`docs/experiments/single-cab78a1-sinkhorn-parity/`，未将第一步LR0冒充非零更新。

本地完整v2回归340passed（113b96a阶段）；接口类型修复后20定向测试及4文件mypy通过。
ABBA已终止exit0，四组12/12、完整审计comparable。
A1/B1/B2/A2含等待均值21.3293/20.4358/20.2642/21.3046秒；吞吐+4.7519%。
收据docs/experiments/single-b64745b-sinkhorn-abba-m32/；源码
b64745b7cc2a752ab992a32c62a234193a676cad，config SHA256
c03f83643f208af475765d2491401fd2de0b6b0091d96c7243712b21152b5a8f。
mHC通过完整精确更新和代表性batch性能门槛，保留；正式配置接入待完成。
独立Gram probe源码c7e89e8525a149026bdb17354e289e42fda0a28a，exit0、18组通过容差，
梯度exact、前向最大差4.47e-8；小形状偏慢，不采用默认。
收据docs/experiments/single-c7e89e8-gram-probe/receipt.json。
当前无活跃GPU任务，两卡各2MiB/0%；下一步核对Gram归约顺序并据新假设优化，
再完整更新/真实吞吐验证。不得放宽容差，不重复无新假设的失败候选。
保持单向KDA统一final-state读取；持续容量和完整B0尚待完成。

先前容量短检查全部终止：a1d55fa micro8/16/32/48/64 passed1/1，
3ba9a96 micro72 passed1/1+restore、micro80 OOM、88未启动。
各原始收据在对应docs/experiments/single-*-capacity-integration目录。
单步不证明持续容量；机制优化验证前不启动容量长测/正式B0。

## 已完成的性能取舍

CPU预取ABBA已终止exit0，四组12/12通过且配置/数据/RNG/源码审计通过。
A1/B1/B2/A2含等待均值21.6529/22.3103/21.8072/21.4106秒。
预取平均慢2.4476%，故不采用；同步路径默认保持。两次双卡更新数值完全
一致但不构成速度收益。详见 `docs/experiments/single-ec2384a-prefetch-abba/comparison.json`
及同目录原始收据（已dry-run356,094bytes后同步）。

性能总结 `docs/experiments/GRADPERT_V2_SINGLE_PASS_PERFORMANCE.md`。
全序列CUDA Graph候选10/12触发重编译64上限，淘汰当前路径，不增加上限重跑。
无序列重计算两次更新精确一致但显存约3.27→13.55GiB，暂不采用。
默认单向final-state eager+重计算；旧batch192不能证明当前方法容量。
完整本地v2测试312passed；新增benchmark因子审计9passed。

## 运维

本地工作树 `/Users/elan/code/grad-pert-v2-build`，定向提交推送HEAD:main。
服务器Python `/data/yilangliu/GraD-Pert/source/.venv/bin/python`。
SSH `ssh -S none -o BatchMode=yes -o ConnectTimeout=10 10.24.1.91`，当前可用。
服务器无rg，可用Python筛选输出。数据/权重/大trace只留服务器；小收据先dry-run。
共享Git克隆不可无限叠加；独立基底source-v2-replay-473466d-standalone可用。
干净发布应从无ignored egg-info的独立本地克隆生成，保持源码身份核验。
旧运行细节及已停止PID仅在STATUS/history/Git保留，不当作活动任务。
## 当前所有权／动作（2026-09-27 08:27 UTC）

主会话拥有当前性能阶段；已完成的双卡完整更新 parity 为 `32b2bfd88f36b9a7fad435c533dc825ef81942b4`、两 rank 两步 passed、输入/RNG/全部损失/梯度/optimizer/Teacher/center 差值零，第二步 LR 非零。短时 ABBA 队列 `/data/yilangliu/GraD-Pert/development/v2-kda-constants-32b2bfd-abba-20260927T0826Z` 正运行，PID `3668186`，顺序 A1/B1/B2/A2；仅本会话 Luna 子代理只读监督，旧跨会话心跳保持暂停。原 B0 中止且不恢复。队列终态后主会话立即审计总步时、含等待吞吐、峰值显存、启动成本与哈希；有效才进入新源码128步持续容量，否则淘汰并依据关键路径选择下一机制。用户授权范围仍含新 ID 完整 B0 五轮及 best/last，不启动其他消融。下方章节为历史快照。


## 2026-10-10 Unified MLP first batch implementation

Only N0/U24/MR1/P1/C1/O1/VH/S1-L4/CG1/S12-L4 are authorized. New common MLP, masked response, population mean/MMD, conditional graph, raw shared readout and independent Local views implemented. Target tests133 pass; full v2 regression737 pass/11 environment skips/1 missing historical fixture failure reproduced on immutable40fb032. Scoped Ruff/format/mypy pass, no first-batch review blocker. Student trainable counts N0/U24/CG1=42699822/42980782/45596478; frozen prior13324288. Candidate batch32/world1/accum1 requires CUDA10-update restore and300-control gates. No scientific completion or capacity claim yet.

Plan: plans/GRADPERT_V2_UNIFIED_MLP_FIRST_BATCH_20261010.plan.md. Report: docs/experiments/GRADPERT_V2_UNIFIED_FIRST_BATCH_20261010.md. Source implemented in a415d8c; next publish final snapshot, immutable server checkout, worst-first gates, then automatic six-epoch two-lane first batch only. Main owns build; old U4/U3 monitors remain deleted. VPN UI/TCP/SSH recovered and GPUs idle.

## BR400 GenePT activation recompute repair

Source2898882 GM1 gate passed, F1 exceeded85% owned memory (28108MiB vs27715.95MiB); no formal fit. Failure preserved. Equivalent Student GenePT mapping activation checkpoint now passes151 CPUchecks+read-onlyREADY, common to BR/F1/N1/GM1. New checkpoint family preflightF1→GM1→BR→N1, formalBR→F1→N1→GM1. Await fresh immutable publication and actual dual-GPU10step+joint+eval300 gates; then main acceptance/formal4epoch bestjoint/last. Batch400/resource85%/free2GiB unchanged.

## TotalVI coexistence now authorized

User approved UID1003 run_totalvi_source_validation_development.py under unchanged limits. SharedGPU schema3 optionally adds only this exact program with explicit new authorization; ScButterfly retained; combined1foreign/GPU≤1GiB/free≥2GiB/owned≤85%. CPU127passed. Prior6e2b93c attempt stopped before GPUchildren; preserve terminal evidence. Fresh published-source preflight pending; then BR/F1/N1/GM1 global400 four epochs bestjoint+last tests.

## User supersedes strict85% with90% for fresh BR400

Latest user asks less strict memory guard: own90% with physicalfree2GiB, UID1003 exact ScButterfly/TotalVI oneperGPU≤1GiB unchanged; newreference seals policy. Old85%runs remainhistorical.130CPUchecks+samefit/testfraction regressionpassed; model/loss unchanged, no iBOTremat adopted. Freshpublication/allfour10step+joint/eval300 preflight pending before formalBR/F1/N1/GM1 four-epoch bestjoint/last.

## BR fourth-epoch validation completed before safe user cancellation

Epoch4last SHA85ca4a88f392dc29cca651eab5801e252d96d823bbe3209b15498842c6671975 saved+validated; oldcontroller2524465 ownedtree stopped, allowned exited, allpendingarms unstarted. Oldroot userstop preserved. Publish independent guardedlast-onlyeval withscienceSHA6a8c612, GPU0,1 eachbatch300/owned90/free2GiB. Evaluatefirst, then newnone-validation/last-onlyremainingF1/N1/GM1. No current CUDAjob/noactiveoldsupervisormonitor; mainownsbuild.


## 2026-10-16 next priority: B0-DP → DP2

User requests both arms immediately after preserved active VH fit/last/medium. Old S12-L4 and later arms deferred; frozen old source/queue/run IDs untouched. New parent is FG+PO/node0 B0, global320, four epochs, last18 and same-medium diagnostics. B0-DP retains historical DP freeze semantics; DP2 adds temperature0.04→0.03 in epoch1, EMA0.996→1, fixed prototype gain and first-epoch no-gradient freeze. Plan: plans/DINO_PAIR_20261016.plan.md. Current stage implementation/static checks; no new formal GPU run yet. Mutable owner/controller/boundary-waiter/monitor identities remain in authoritative coordination state, not this note.
