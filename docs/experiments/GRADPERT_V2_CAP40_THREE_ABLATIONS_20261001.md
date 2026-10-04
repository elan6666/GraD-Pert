# 2026-10-05：三组消融已独立验收，最终对照

B0/E1/E2/E3 均完成六轮、1056 次优化；每轮只做 joint loss 验证，best 按本组最小验证 joint 选择，四组 best=last=epoch6。best/last 均有真实冻结测试收据，逐一重算 checkpoint SHA256；source/config/data/selection/environment lock 与评估身份已核对。四组有序 condition/control/truth 哈希相同，每条件均为原定300 control。无失败标记、成功运行根无 PKL；E1 沿用已完成旧运行，没有重跑。替换队列于北京时间2026-10-04 21:46:26 COMPLETE，SHA256 `4c2ff13cee70a74b5e19359541ab556dfdb8fbe99d3326b145c71d42937892e1`。监督监控及主返回提醒均已删除；没有启动其他实验。

## 总体测试指标

下表是相同5000表达基因轴上的条件宏平均；all 有592个有效条件，统一DEG有590个（另2个真值细胞数为1，不能做既定差异检验）。best/last 内容与指标相同，仍分别保留两个角色收据。

| 配置 | TxPert all / DEG | TriShift all / DEG | Systema all / DEG |
|---|---:|---:|---:|
| B0：mHC 完整基线 | 0.212492 / 0.380273 | 0.166488 / 0.351775 | 0.079602 / 0.232675 |
| E1：关闭 mHC | 0.218007 / 0.381337 | 0.169879 / 0.353931 | 0.075271 / 0.214566 |
| E2：原型数 16384 | 0.209193 / 0.355427 | 0.164478 / 0.319862 | 0.081327 / 0.201001 |
| E3：蒸馏 1/1，关闭 spread/KoLeo | 0.217075 / 0.381580 | 0.168728 / 0.346391 | 0.088424 / 0.217086 |

| 配置 | 测试预测 MSE | 第6轮验证 joint |
|---|---:|---:|
| B0 | 0.006509533 | 4.058603 |
| E1 | 0.006163277 | 4.235659 |
| E2 | 0.006657580 | 4.218107 |
| E3 | 0.006096135 | 5.493816 |

原型数量、权重及删除负值正则会改变 joint 的尺度；上列 joint 用于检查本组 checkpoint 选择，不能用来跨组评价预测质量。

## 按训练表达可见性分组

“见过”指完成的训练更新中实际输入/监督过该基因的表达数值，不是图节点身份是否可见。四组分组 ID/hash 完全一致：seen=4775，unseen=225；先取同一套 truth-defined DEG，再与各组基因集合求交。两组 all 均有592个有效条件；seen DEG有590个，unseen DEG仅147个。unseen 的多数条件DEG交集不足2个基因，跳过并保留原因，不能当作零相关纳入分母。

### 已见表达的4775个基因

| 配置 | TxPert all / DEG | TriShift all / DEG | Systema all / DEG |
|---|---:|---:|---:|
| B0：mHC 完整基线 | 0.418995 / 0.467207 | 0.311421 / 0.431089 | 0.216256 / 0.273443 |
| E1：关闭 mHC | 0.412188 / 0.480475 | 0.304615 / 0.448660 | 0.204864 / 0.263426 |
| E2：原型数 16384 | 0.415944 / 0.453574 | 0.304656 / 0.408301 | 0.228912 / 0.253869 |
| E3：蒸馏 1/1，关闭 spread/KoLeo | 0.420603 / 0.458886 | 0.307178 / 0.417023 | 0.247550 / 0.265329 |

### 未见表达的225个基因

| 配置 | TxPert all / DEG | TriShift all / DEG | Systema all / DEG |
|---|---:|---:|---:|
| B0：mHC 完整基线 | 0.113987 / 0.181124 | 0.105858 / 0.131780 | -0.019707 / 0.073539 |
| E1：关闭 mHC | 0.134265 / 0.171988 | 0.126095 / 0.118031 | -0.024319 / 0.032462 |
| E2：原型数 16384 | 0.126889 / 0.119297 | 0.119852 / 0.095778 | -0.022012 / 0.008502 |
| E3：蒸馏 1/1，关闭 spread/KoLeo | 0.141498 / 0.237078 | 0.133446 / 0.184957 | -0.028283 / 0.106016 |

## 解释边界

- E1 关闭 mHC 后总体 TxPert/TriShift 上升、Systema下降，方向混合；不能据单seed结果断言 mHC普遍有效或无效。
- E2 原型翻倍后总体6项Pearson中的5项低于B0，MSE略高；当前六轮预算下没有普遍收益证据。不同输出维度还改变初始化随机数消耗，此对照代表完整配置效果，不是逐更新同随机路径的算子等价测试。
- E3 总体 all 三项及 TxPert DEG 高于B0，TriShift/Systema DEG低于B0，MSE低约6.35%。在unseen组，TxPert/TriShift all与DEG、Systema DEG高于B0，但Systema all更负。它同时改四项权重并删除spread/KoLeo，无法单独识别任何一项的作用。
- 所有结果来自seed1；best/last是同一checkpoint，不能当作两个独立重复。未做跨seed显著性结论，未依据测试分数选择配置或启动追加实验。

## Loss、性能与完整身份

所有24个epoch的训练/验证分项见 [epoch-loss-components.csv](../../.byte-os/evidence/v2-cap40-three-ablations-20261001/final-comparison-20261005/epoch-loss-components.csv)。所有144条best/last×总体/seen/unseen×六项指标及run/config/checkpoint/test收据SHA见 [pearson-all-groups-best-last.csv](../../.byte-os/evidence/v2-cap40-three-ablations-20261001/final-comparison-20261005/pearson-all-groups-best-last.csv)。原逐轮曲线继续保存在各服务器fit目录；无需下载权重或表达矩阵。

E2在相同正式配置m68×accum2×双卡（全局272）完成128/128持续预检、checkpoint重载和300-control验证。更新中位22.991秒，P95 24.114秒；预检训练墙时3114.840秒，对应端到端11.177细胞/秒；rank0/1峰值allocated分别31,972,887,040/31,722,863,104 bytes，reserved分别32,281,460,736/32,008,830,976 bytes。数据来自持续预检，不冒称正式六轮峰值或长期吞吐。E3的一步预检只证明完整更新/重载可运行，不适合与E2持续吞吐比较。正式日志未保存可可靠分离的训练、验证、测试墙时或全程峰值；这些分项标为缺失，不用进程间隔或预检外推代替。详情 [performance-evidence.json](../../.byte-os/evidence/v2-cap40-three-ablations-20261001/final-comparison-20261005/performance-evidence.json)。

| 配置 | 训练/评估 commit（两者同版本） |
|---|---|
| B0 | `aeac5fd94123af0b73810259e5e2985228b11d65` |
| E1 | `ac220f5e90e9e5c0c9900cdb2c285a629297c93d` |
| E2 | `2871c4e79a70b7d0d4f900866ce019316b68670f` |
| E3 | `2871c4e79a70b7d0d4f900866ce019316b68670f` |

不同发布版本明确保留。四者native `src` Git tree均为 `ed919ee6eaed01aa7a778c7fa276f87f2a496758`；E2/E3的2871c4e只修复收集器/队列，不是换模型后重新评估E1。完整身份及逐项核验见 [independent-acceptance.json](../../.byte-os/evidence/v2-cap40-three-ablations-20261001/final-comparison-20261005/independent-acceptance.json)。监督终态收据 `.byte-os/coordination/receipts/v2-cap40-remaining-terminal-20261004T1346Z.json`；主验收收据 `.byte-os/coordination/receipts/v2-cap40-three-ablations-main-acceptance-20261005.json`。

---

以下为历史执行记录，启动/等待/未知容量状态已被上述最终验收覆盖；旧失败队列和旧运行身份保持不变。

## 2026-10-03: remaining supervision armed and acknowledged

Queue/controller identity and source/runtime confirmed independently by Luna, then designated supervisor. E2 sustained128 capacity probe is active24/128 at independent ACK (formal training not yet begun). Its dual-GPU memory snapshot31712/31452MiB is an instantaneous observation, not peak capacity acceptance. Existing grad-pert-v2 monitor ACTIVE20min on exact supervisor/queue/attempt1; mainreturnfallback PAUSED. ACK .byte-os/coordination/receipts/v2-cap40-remaining-supervisor-ack-20261003T0735Z.json. This supersedes pending-ACK and connectivity-block notes below; no second long supervisor remains.

# 2026-10-03 repair and E1 accepted result

Old queue attempt1 failed after successful E1 because the terminal collector omitted v2_fixed_6. Its failure marker/log and all old identities remain intact. Main independently re-read server terminal/test receipts, exact data identity, zeroPKL and hashes. Repaired collector at2871c4e79a70b7d0d4f900866ce019316b68670f accepted E1; its training/evaluation SHA stays ac220f5e90e9e5c0c9900cdb2c285a629297c93d. Collector repair is not a new model evaluation. Native src tree ed919ee6eaed01aa7a778c7fa276f87f2a496758 is unchanged.

Best and last are both epoch6 (checkpoint9f18f995f16e9d7852c8fff377cc2ccad9ed845cce210b8be8c24b37041bb86d). Test prediction loss0.0061632775. All metrics592 finite conditions; DEG590 of592, matching B0's insufficient-cell DE omissions.

| Metric | B0 mHC all / DEG | E1 no-mHC all / DEG |
|---|---|---|
| TxPert | 0.212492 / 0.380273 | 0.218007 / 0.381337 |
| TriShift | 0.166488 / 0.351775 | 0.169879 / 0.353931 |
| Systema | 0.079602 / 0.232675 | 0.075271 / 0.214566 |

These are one-seed descriptive comparisons, not significance or a hyperparameter-selection rule. mHC/no-mHC differences are mixed across metrics. Test results do not change the remaining frozen designs. Identity and acceptance: .byte-os/evidence/v2-cap40-three-ablations-20261001/e1-independent-acceptance-20261003.json.

110 local/server checks passed for collector fixed6 and completed-E1 import; Ruff passed. New immutable server source /data/yilangliu/GraD-Pert/development/source-v2-cap40-repair-2871c4e, matching published repairSHA. Runtime SHA de87592aa86462e843dfa65ce0960ab78c4c1ea2e999dae144eaed1d3210ae60, publication SHA1956abd0f30b2a1895461bc34993ede55287a1c803d0194594d5029a258d367b.

Remaining queue v2-cap40-remaining-2871c4e-20261003T072407Z, controller742925, root/data/yilangliu/GraD-Pert/development/v2-cap40-remaining-2871c4e-20261003T072407Z. --completed-e1 references accepted old E1; no new E1 plan. E2/E3 get fresh IDs. Both original preflights passed (E2 sustained128 atpeak31972887040bytes); they are repeated under the repaired full source identity to satisfy existing exact-source gates. Formal budget/microbatch/losses stay unchanged. Queue launch exists, live startup verification and new supervisor lease pending at this record; later dated acknowledgement supersedes it.

---

# Jurkat cap40 three-ablation execution

Execution authorized on 2026-10-01 after independent acceptance of the cap40 mHC six-epoch B0. This is an execution record; new scientific results are not yet available.

| Row | mHC streams | Four prototype heads | SSL1 condition/node/spread | SSL2 DINO/iBOT/KoLeo |
|---|---:|---:|---|---|
| Accepted B0 reference | 4 | 8192 | 0.8/0.4/0.1 | 0.8/0.4/0.1 |
| E1 | 1 | 8192 | 0.8/0.4/0.1 | 0.8/0.4/0.1 |
| E2 | 4 | 16384 | 0.8/0.4/0.1 | 0.8/0.4/0.1 |
| E3 | 4 | 8192 | 1/1/0 | 1/1/0 |

Both distillation stage multipliers remain 1. E2 changes prototype count only, not MLP dimensions. E3 is a combined recipe; it cannot isolate each component's individual causal contribution. Absolute joint losses across different head counts/weights are not directly comparable as predictive quality.

All rows use fresh six-epoch training, seed1, GPU0/1, micro68 x accumulation2 x world2 = global272, 176 optimizer updates/epoch. Frozen cap40 selection retains 47836 training perturbation rows and all1335 conditions. Validation/test/control/reference manifests, test-target expression exclusion, full prediction/SSL1/SSL2, operator semantics and random-view distributions stay unchanged. Epoch validation is joint-only; minimum validation joint selects best, last is epoch6. Automatic terminal tests evaluate both roles under frozen300-control protocol, all/DEG three Pearson metrics and expression-exposure groups, with zeroPKL.

## Release and gates

Published immutable training source ac220f5e90e9e5c0c9900cdb2c285a629297c93d (model src unchanged from B0 aeac5fd94123af0b73810259e5e2985228b11d65). Server source /data/yilangliu/GraD-Pert/development/source-v2-cap40-ablations-ac220f5. Runtime SHA256 fb327a18d0f93cdc5dba968e38dc8031a64c18ff6040998bc71ae18b5ce84eeb; publication receipt SHA256 3e850573eba4c7d36d379b49326fa913bc27f6241d9389f1345a8edcc90073b7. Local and server directed suites passed78 checks. This does not establish CUDA capacity.

Controller runs E1 complete-update/reload integration, E2 sustained128 updates/reload/one-condition300-control validation, then E3 integration. Formal E1->E2->E3 runs start only after all preflights pass. E2 memory capacity remains unknown. Any failure stops the queue with evidence; no silent batch/loss/head fallback, run overwrite or automatic retry.

Controller PID368656 is a durable CPU process. Startup state is running/waiting_for_idle_gpus with no GPU child. Other scbutterfly jobs occupied both GPUs at728MiB during verification. Controller waits until both<=512MiB, acquires GPU/row leases and rechecks resources. It preserves unrelated jobs. GPU training has not yet been verified at publication of this record.

Queue root /data/yilangliu/GraD-Pert/development/v2-cap40-ablations-ac220f5-20261001T060835Z. Its state.json, queue.json, stage logs/exits/preflight-index and FAILURE/COMPLETE are authoritative. External .log/.pid/.launch.json record launch. Model outputs stay under /data/yilangliu.

## Exact rows

- {"name": "E1_no_mhc", "run_id": "nadig_jurkat-seed1-20261001T060836Z-725056771bb54902b4a82316cacbcdea", "run_root": "/data/yilangliu/GraD-Pert/runs-v2-cap40-ablations-ac220f5/nadig_jurkat-seed1-20261001T060836Z-725056771bb54902b4a82316cacbcdea", "config_sha256": "5718b2c298aaed31ae796205be5f8597a060da4106e4b0017e35f605dff68686"}
- {"name": "E2_prototypes16384", "run_id": "nadig_jurkat-seed1-20261001T060836Z-4abe0ca0b78a4130aaa4470a33461b13", "run_root": "/data/yilangliu/GraD-Pert/runs-v2-cap40-ablations-ac220f5/nadig_jurkat-seed1-20261001T060836Z-4abe0ca0b78a4130aaa4470a33461b13", "config_sha256": "5197a3e3d7bfe57a7545b4cc1dad8ff7b5c666dd521bb298cec1ed2d9dabe64f"}
- {"name": "E3_unit_distillation_no_spread_koleo", "run_id": "nadig_jurkat-seed1-20261001T060836Z-26ce673c50704a59aea53535363c4fd5", "run_root": "/data/yilangliu/GraD-Pert/runs-v2-cap40-ablations-ac220f5/nadig_jurkat-seed1-20261001T060836Z-26ce673c50704a59aea53535363c4fd5", "config_sha256": "6982785ffac1b5ac3f681956ddf8ad51af2aa505e3f02e22572876cb7d34cb2a"}

## Supervision and acceptance

Designated existing chat: 监督查询对话, native ID01a0df0b-4142-7df1-86c0-d959471d80a1. Existing grad-pert-v2 monitor retargeted to this queue/attempt1, ACTIVE every20min, verified tool and saved configuration. Main return fallback grad-pert-v2-b0-return retargeted and PAUSED during queue. On terminal handback, owner returns to main with main_continuation exactly pending; return fallback may activate if substantive acknowledgement is absent, then main pauses it. Immediate fault event-to-chat adapter is unverified;20min polling is verified detection path. Completed Luna startup verification is not a second ongoing supervisor.

Each formal acceptance requires6/6 and history, minimum-joint best and epoch6last, actual best/last frozen tests, source/config/data/selection/ordered control/truth/checkpoint/evaluation identity, noFAILURE and zeroPKL. Queue completion requires allthree accepted rows and queue COMPLETE. Native run IDs are distinct; old B0/source/checkpoints remain unchanged.

Evidence: .byte-os/evidence/v2-cap40-three-ablations-20261001/queue-local-validation.json, queue-launch-verification.json, monitor-activation.json. Handoff: .byte-os/coordination/handoffs/2026-10-01-v2-cap40-three-ablations-supervise.md. Supervisor acknowledgement is recorded separately after live verification.
