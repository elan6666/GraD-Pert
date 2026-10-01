# 已完成：Jurkat cap40＋mHC B0 六轮结果

本轮从头训练6epoch，完整预测＋SSL1＋SSL2、mHC4、全局batch272（每卡68×累积2×双卡）。主会话于 2026-10-01T13:47:29.668822+08:00 独立复核服务器原始证据。best和last都选中第6轮，但分别保留测试角色与收据。训练及评估SHA均为 `aeac5fd94123af0b73810259e5e2985228b11d65`，配置SHA `1401174171770bb5c7ae580db55f9dd9d4907d8da6a37a42b86a64d747f412d0`。

| Epoch | 更新累计 | 训练预测 | 训练joint | 验证预测 | 验证joint |
|---|---:|---:|---:|---:|---:|
| 1 | 176 | 0.099099 | 19.966110 | 0.064593 | 15.630725 |
| 2 | 352 | 0.059014 | 5.432248 | 0.059166 | 6.043361 |
| 3 | 528 | 0.056868 | 1.199991 | 0.056429 | 4.561331 |
| 4 | 704 | 0.055725 | 0.591937 | 0.054313 | 4.188529 |
| 5 | 880 | 0.055110 | 0.460473 | 0.054234 | 4.072661 |
| 6 | 1056 | 0.054934 | 0.422775 | 0.053954 | 4.058603 |

冻结300-control测试：592条件；预测MSE 0.0065095328161766134。

| Pearson口径 | 全基因 | DEG |
|---|---:|---:|
| TxPert | 0.212492 | 0.380273 |
| TriShift | 0.166488 | 0.351775 |
| Systema | 0.079602 | 0.232675 |

全基因均592/592有效；DEG590/592，两个条件因真实细胞数为1无法做DE检验。统一DEG集合与既有Systema参考未改变。

| 表达暴露组（全基因口径） | 基因数 | TxPert | TriShift | Systema |
|---|---:|---:|---:|---:|
| 训练出现表达 | 4775 | 0.418995 | 0.311421 | 0.216256 |
| 训练未出现表达 | 225 | 0.113987 | 0.105858 | -0.019707 |

未见表达组DEG仅147/592条件有效，不能与590条件的整体DEG均值当成相同条件集合直接比较。训练未见表达也不等于图身份/先验未见。cap40、更长从头训练和历史3+3续训协议不同，历史结果不能用于单独归因mHC或采样的效果。

实际best/last及epoch6文件SHA256均 `8a258885784f45d010c0d1c8fabce6c8ac46252bd0292de71a53ea69133b97e3`；exit0、COMPLETE、无FAILURE、全run零PKL，目标父/torchrun/rank进程退出。GPU1另有非本轮scbutterfly进程，不宣称服务器整体空闲。完整损失分项、测试/数据/配置哈希见[主会话小验收收据](../../.byte-os/evidence/v2-mhc-cap40-six-acceptance-20261001/receipt.json)。

三组后续对照已设计，DINO/iBOT实验③权重均1；尚未启动，E2容量未验证。

---

# Jurkat cap40 mHC B0: six fresh epochs

The user selected the existing fixed cap40 sample and enabled mHC. This new
self-contained configuration is
`configs/v2/mhc_cap40_jurkat/six_epoch_m68_a2/gradpert_v2/nadig_jurkat.yaml`.
It inherits no config: all settings are written explicitly. The old three-epoch
mHC config and historical no-mHC continuation remain unchanged.

## Data and lineage

Only training perturbation rows are selected:128266 ->47836, all1335 train
conditions retained, each at most40. This is the accepted seed42 proportional
batch-quota sample, fixed across all epochs; it is not resampled each epoch.
Control, validation, test and excluded rows remain canonical and unchanged.
No expression values, graph axes, test-target expression policy or Systema
train+validation reference are changed by the row selector. An optional v2-only
manifest restricts the existing train_row_indices; no derivative H5AD is
relabeled canonical-ready and v1 uses the original pipeline.

The analysis canonical H5AD65b32637b24e6ee6d3399b3280d914dcedc34bf787098c3e9e14a47ebe80cbb5
and training canonical f051343c191dcdb02cabfae66a6bfe10b2770752b6503802da33d4b74e845861
have identical238977x6506 float32 X (full512-row backed comparison, zero
value differences), and exact obs/var including categories. Their uns
preprocessing records differ, including graph_only_candidate_targets; these
provenance differences are preserved rather than declaring full-file equality.
The training graph remains the existing frozen v2 graph.

Server binding directory:
`/data/yilangliu/GraD-Pert/development/jurkat-cap40-v2-binding-20260930`.
Equivalence receipt SHA256:ca12b16967f77a06d96139ffaa23cd997af13aa1b2acbe7250ed1d8696b04f85.
Training selection SHA256:aa0916dfb91fff78d7a43f7d7325879a11c1a904e9bf31586420ca11cce84132.
Selected ordered row-ID hash:c8e79242336c194833ac329a807b7eeb76651707204b4ec1379f6a7483aa9a93.
Original accepted analysis selection SHA256:a609d5aa4087d209dc62c1e8ed70dc7cff89ed6952b2afb18d0972cb477c94f2;
analysis SHA5a8a7bb6712bc4facdfeb13d8dd629bd0f250880. Long row-ID lists stay on server.

## Training and evaluation

Student and EMA Teacher use four mHC streams; full prediction+SSL1+SSL2.
SSL stage multipliers are1; internal weights remain0.8,0.4,0.1 in each stage.
Single-pass KDA writes once and every query reads final S; graph source key
gates/random order/view and global KoLeo semantics are unchanged. Width256,
four heads, graph3KDA+1MLA, Cell/Response2KDA+1MLA; no DSA.

GPU0,1: each physical microbatch68, accumulation2, globalbatch272, seed1.
Graph scan32, target rows64, sequence scan16. m68 has earlier three-epoch mHC
success evidence; new-source one-update integration is required before launch.
Six fresh epochs use the project warmup/cosine schedule over this selected-row
budget. This is not a 3->6 continuation and does not initialize from old weights.
Validation runs only fixed-view joint loss and components; minimum joint loss
selects best.pt. After six epochs best and last automatically run the frozen
300-control evaluation, three Pearson definitions each all/DEG plus the existing
expression-exposure groups. Zero-PKL protocol applies to the whole run root.

## Implementation and acceptance

V2Options exposes optional paired train_selection_path/train_selection_sha256.
Absent fields preserve historical config serialization. Before model allocation,
loader verifies the file hash, canonical/split/axes, unique train-only ordered
IDs, exact min(cap,Np) quotas and all conditions, non-training row-axis hash,
and unchanged samples from the hash-pinned accepted analysis plus equivalence
proof. Only then it replaces train_row_indices. Receipt in runtime identity
seals the selection for training checkpoints and standalone evaluation.
Synthetic schedule tests cover all six epochs' fixed IDs, unchanged validation
and control pools, leakage, duplicates, bad hashes/parent/split/order/quota and
atomic failure. Execution still uses existing runtime/runner/evaluator.

Local verification:95 affected config/runtime/selection/joint-validation/data-contract tests pass; scoped Ruff check/format pass and config/selection typechecking pass. The server repeated the95checks and the real dual-GPU integration gate passed. Formal launch and final terminal acceptance are verified below.

## Verified launch

Training source `aeac5fd94123af0b73810259e5e2985228b11d65` was published on main and matched
clean local release/server source tree8f4137da3330e636a557fb8c68525dcfcfe2247a55db86b8ac368edac2259f9d.
95server tests and sameconfig dual-GPU one-update/checkpoint-reload passed.
Fresh run `nadig_jurkat-seed1-20260930T112531Z-f7ca7a77fa6848bb811f706d00e5daad` has started: epoch1 step2/176 observed, both ranks alive and
GPU identity matched. There are1056planned optimizer updates over six epochs.
This paragraph records launch-time evidence; all six epochs and final tests have now completed.

[Data preflight](data/v2-mhc-cap40-aeac5fd/v2-mhc-cap40-aeac5fd-data-preflight.json),
[canonical equivalence](data/v2-mhc-cap40-aeac5fd/equivalence.json),
[integration receipt](data/v2-mhc-cap40-aeac5fd/integration-receipt.json),
[launch identity](data/v2-mhc-cap40-aeac5fd/v2-mhc-cap40-aeac5fd-launch.json),
[start confirmation](data/v2-mhc-cap40-aeac5fd/v2-mhc-cap40-aeac5fd-start-confirmation.json).
Only reviewed small receipts were copied; scientific files and row-ID lists remain
on server. The designated supervisor handed back the successful terminal receipt. Both the supervisor monitor and main return fallback are now PAUSED; main independently verified final acceptance.
