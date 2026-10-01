# Approved cap40 E1/E2/E3 queue supervision

Main codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b -> supervisor codex:01a0df0b-4142-7df1-86c0-d959471d80a1, host local.
Project /Users/elan/.codex/worktrees/v2-gene-exposure-eval/grad-pert.
Queue v2-cap40-ablations-ac220f5-20261001T060835Z, attempt1, durable controller PID368656.
Root /data/yilangliu/GraD-Pert/development/v2-cap40-ablations-ac220f5-20261001T060835Z; external .log/.pid/.launch.json.
Source /data/yilangliu/GraD-Pert/development/source-v2-cap40-ablations-ac220f5; SHA ac220f5e90e9e5c0c9900cdb2c285a629297c93d, verified clean, model src identical to B0 aeac5fd.
Runtime development/v2-cap40-ablations-ac220f5.runtime.json SHA fb327a18d0f93cdc5dba968e38dc8031a64c18ff6040998bc71ae18b5ce84eeb.
Publication development/gradpert-cap40-ablations-ac220f5-publication.json SHA 3e850573eba4c7d36d379b49326fa913bc27f6241d9389f1345a8edcc90073b7.

User explicitly authorized execution. All rows fresh6epochs seed1, micro68 x accum2 x world2=272, fixed cap40 47836 rows/1335 conditions, original split/control/reference, joint-only epoch validation, final best/last frozen300-control tests with all/DEG/exposure metrics, zeroPKL.
E1 streams1, otherwise B0. E2 streams4/four heads16384, MLP unchanged. E3 streams4, SSL1(condition,node,spread)=(1,1,0), SSL2(DINO,iBOT,KoLeo)=(1,1,0), lambda1=lambda2=1.
Each config pinned under configs/v2/cap40_ablations_jurkat/<group>/gradpert_v2/nadig_jurkat.yaml.

Startup verified: controller alive, state running/waiting_for_idle_gpus, no child or terminal marker. Both GPUs728MiB, unrelated scbutterfly jobs preserved. Resource gate both<=512MiB; CPU controller checks60sec and leases/rechecks resources before GPU stages. No GPU training started at handoff. Luna finished bounded verification. Its E2/E3 path transcription was corrected by independent raw queue/runtime recheck; all three paths and hashes match, no server files edited. Receipt .byte-os/evidence/v2-cap40-three-ablations-20261001/queue-launch-verification.json. Local/server78 tests passed.

Controller advances automatically: E1 one-update integration/reload; E2 sustained128 updates/reload/one-condition300-control validation; E3 one-update integration/reload; only all passed then E1->E2->E3 formal6epochs with best/last tests. Reads queue.json exact native IDs/roots; no extra launch commands needed. Any failure stops with FAILURE.json. No batch fallback, extra retries, head/loss reduction, implicit resume, duplicate controller, or other ablations. Source is immutable. E2 capacity is still unknown. Missing PID never success; inspect state, bounded log, per-stage exit, preflight-index, receipt, per-formal epoch_state/history/COMPLETE/test identities/zeroPKL.

Supervise every20min via existing grad-pert-v2 targeting this supervisor, exact queue/attempt owner guard. Record acknowledgement and tool+saved ACTIVE wake before main ends. Main does not duplicate live queries. Routine unchanged healthy checks quiet; milestones/failure/completion report. Resource wait is healthy while unrelated GPU jobs exist. Stop/return for failure, dead controller, scientific decision, or all complete. Do not edit active server code or expand resources; routine VPN recovery is allowed.
Return: durable terminal evidence/handoff, pause grad-pert-v2 and verify inactive, owner exact main, main_continuation EXACTLY pending (never pending_fallback_active). Send actionable main continuation and observe bounded main acknowledgement. Main return heartbeat grad-pert-v2-b0-return is prepared PAUSED,5min; activate only if main does not substantively acknowledge terminal handback. Main pauses on acceptance. Native fallback wake previously observed; immediate event-to-chat adapter remains unverified,20min polling is verified detection route.
Final acceptance each6/6,best/last roles true frozen tests, exact source/config/selection/control/truth/checkpoint/eval identities,noFAILURE,zeroPKL; queue COMPLETE plus all three receipts. Then main compares same-protocol results to accepted B0, preserving different loss scales; no test-set tuning. Pause all monitors when terminal.

Exact rows:
{"name": "E1_no_mhc", "run_id": "nadig_jurkat-seed1-20261001T060836Z-725056771bb54902b4a82316cacbcdea", "run_root": "/data/yilangliu/GraD-Pert/runs-v2-cap40-ablations-ac220f5/nadig_jurkat-seed1-20261001T060836Z-725056771bb54902b4a82316cacbcdea", "config_sha256": "5718b2c298aaed31ae796205be5f8597a060da4106e4b0017e35f605dff68686"}
{"name": "E2_prototypes16384", "run_id": "nadig_jurkat-seed1-20261001T060836Z-4abe0ca0b78a4130aaa4470a33461b13", "run_root": "/data/yilangliu/GraD-Pert/runs-v2-cap40-ablations-ac220f5/nadig_jurkat-seed1-20261001T060836Z-4abe0ca0b78a4130aaa4470a33461b13", "config_sha256": "5197a3e3d7bfe57a7545b4cc1dad8ff7b5c666dd521bb298cec1ed2d9dabe64f"}
{"name": "E3_unit_distillation_no_spread_koleo", "run_id": "nadig_jurkat-seed1-20261001T060836Z-26ce673c50704a59aea53535363c4fd5", "run_root": "/data/yilangliu/GraD-Pert/runs-v2-cap40-ablations-ac220f5/nadig_jurkat-seed1-20261001T060836Z-26ce673c50704a59aea53535363c4fd5", "config_sha256": "6982785ffac1b5ac3f681956ddf8ad51af2aa505e3f02e22572876cb7d34cb2a"}
