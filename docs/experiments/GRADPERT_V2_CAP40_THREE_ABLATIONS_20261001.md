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
