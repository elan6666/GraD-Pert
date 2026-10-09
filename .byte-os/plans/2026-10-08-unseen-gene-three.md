# 2026-10-09 当前执行修订：新增 U4

用户明确要求U1→U2→U4→U3。U1已完成；U2原运行持续训练，旧controller已暂停并由主精准退役，旧U3从未正式启动。保留所有旧source/config/queue/收据和U3已完成短预检，不覆盖或重标记。

当前阶段：实现U4冻结原始2048维GenePT＋随机Xavier可训练Linear2048→256，无PCA或独立可训练基因表；核验公式/冻结/梯度/非零LR多步optimizer-EMA-center与resume/公共骨干RNG。发布后，新的不可变source与fresh queue等待旧U2完整终态，随后U4→U3各10步预检和6epoch末轮last评估，global272/cap40/无验证保持一致。新U3使用新run ID，逐组标注跨源码比较。

验收：实现与定向/回归测试通过；小型server CPU验证；新queue真实等待U2且不会重复启动；监督20min ACK和回主fallback验证后交接。预检不声称长期容量。见实验文档的U4修订；下方旧计划保留历史，不再作为当前排程。

---

# U1/U2/U3 execution plan

Main: codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b. Supervisor: codex:01a0df0b-4142-7df1-86c0-d959471d80a1. Canonical mutable coordination remains in `/Users/elan/.codex/worktrees/v2-gene-exposure-eval/grad-pert/.byte-os/coordination/state.json`.

The current request authorizes three native mechanisms and their priority independent ablations after the finishing job. L0 and L1 are terminal; original L2/L3/M1/M2 were never formally started and the old controller was terminated. The main verified L1's final receipts directly on server and accepted ownership. No training process was restarted.

1. Implement separate architecture flags, zero-initialized adapters/extra column, ID plumbing in training/views/inference/diagnostics; preserve legacy off-path and v1. **Implemented.**
2. Validate formula, frozen prior, identity, gradients, full optimizer/EMA/center/nonzero-LR multi-step and checkpoint restore, configuration/queue contracts. **Passed local tests.** Whole v2: 564 passed,11 unavailable CUDA/Triton skips,1 unchanged missing-config failure; 7 additional new queue tests passed. Read-only Luna review found no scoped issue. Changed-file lint/format pass. Mypy reports17 inherited issues in4 unrelated dependency files, no errors in the changed modules; do not call full mypy clean.
3. Commit/push scoped implementation and self-contained U1/U2/U3 configs on main; immutable server checkout and publication/config hashes. **Pending publication.**
4. Launch a fresh sequential capacity+formal queue. Each arm must pass10 dual-card complete updates (user revision2026-10-09; shortpreflight, notlongcapacity)+fixed-train loss diagnostics+resume+300-control inference before any formal arm. All gates must pass; failures preserve evidence and hand back, no automatic per-arm batch reduction. **Pending.**
5. U1→U2→U3 each from scratch6epochs, cap40, micro68×accum2×world2=272, validation disabled, epoch6last test. Reuse completed oldL0 as explicitly cross-version common reference; no extraU0 retraining or cancelled arms. **Pending gates.**
6. Transfer long queue to exact supervisor, arm20min recurring check and verify ACK+return wake. Main remains only designer/publisher/launcher; on terminal report independently verify and deliver all/DEG three Pearson for all/seen/unseen. **Pending launch.**

Pre-change baseline was already published:47c36faf08505898aa7d6ef2bcf65be387673db8; its modeling/training/config/tests-v2 matchedab022caa57a3b45dc5a14c6ae38bc280702112e4. Prior snapshots prove exact CPU state/RNG/initial prediction and three complete nonzero-LR updates with disabled flags. Scientific effectiveness remains untested.

Historical L0 root `/data/yilangliu/GraD-Pert/runs-v2-loss-diagnostic-ab022ca/nadig_jurkat-seed1-20261007T181232Z-580abf8834e0429ba15dc2f2359893bb`; COMPLETE SHA256 c2465e7d72444d4c6c54d3563f1c3974189190370706df385426fc17ac9b5b1e; final-last test SHA256473d21bf8084eb55127c311d6232be7daaaf8f52971e12726ff374aae30ea02a. L1 COMPLETE SHA25612f6bdff74a58beeb5eaf4e59093ab6ffa364a3511597c7ec41420ee3559ca14 and last-test13e0f016b97eae8e96b32ba5150222ca6e35f870b876ade082f27ad5d9833c77 independently matched supervisor receipt.
