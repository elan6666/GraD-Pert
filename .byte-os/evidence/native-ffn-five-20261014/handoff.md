# Native five FFN queue: preflight supervision and main return

Main: codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b.
Supervisor: codex:01a0df0b-4142-7df1-86c0-d959471d80a1 (host local).
Run/schedule: native-ffn-five-leaf-3ee8ef3-20261014, attempt1.
Authority: /Users/elan/.codex/worktrees/v2-gene-exposure-eval/grad-pert/.byte-os/coordination/state.json.
Editable source/docs: /Users/elan/.codex/worktrees/v2-official-expression/grad-pert.

## User scope and current evidence
User 2026-10-14 prioritizes DP -> FN -> FG -> FB -> PO. Preserve scDFM original
last.pt; its evaluation remains stopped/deferred. Do not restart scDFM, GEARS,
Scouter, BR, or old failed queues. Five arms unchanged: Jurkat cap40, expression
holdout, seed1,4epochs, global400=micro100*world2*accum2, disabled validation,
last-only common18 metrics (all/seen/unseen x three Pearson x all/DEG with valid
condition counts), zeroPKL and training-only diagnostics. Old scientific runs
remain separately sourced. Resource policy unchanged: approved UID1003 scripts,
max1 foreign/GPU <=1GiB, free>=2GiB, owned<=90%; allocator expandable_segments:True.

Server queue root: /data/yilangliu/GraD-Pert/development/native-ffn-five-leaf-3ee8ef3-20261014.
queue SHA256 01a9f4239cf0412446f1497f58b2d5e46d35b8fc63163997d03fb14f9e48a7a9.
Source /data/yilangliu/GraD-Pert/development/source-native-ffn-leaf-3ee8ef3.
Git SHA3ee8ef3b343efa5223f0be4f7dd570e839651f88, immutable clean published.
Runtime /data/yilangliu/GraD-Pert/development/native-ffn-leaf-3ee8ef3.runtime.json;
SHA5f76311010cb773632170f31c89fd7f0eee323a2d2132a60ec5c525062acb2c7.
Publication /data/yilangliu/GraD-Pert/development/gradpert-ffn-leaf-3ee8ef3-publication.json;
SHAcceeaa684f903b2b460c18745e33483070a9a53eb0a01bb27292fec6e1ebb31d.
Manifest configs/v2/ffn_five_20261013/manifest.json. Local/server101checks passed.

Controller3046624 UID1002/start735652620/cmd SHA892d5f2f5da774c7dd44106cdc6fc8161e7bd03dcf2a48e48a8ab70bd21a675b.
Controller CUDA_VISIBLE_DEVICES empty, children0,1; logs controller.log and
<arm>-preflight.log/<arm>-eval-capacity-g0.log/g1.log inside queue root.
Latest Luna snapshot startup-handoff.json SHA9f24f2e6cf7fe3471fced0418890fa7738375fba3250228db3e847d9c84c8754:
DP10/10 restore passed (4c245b6303d470cb7394bc5da3f20bc5253eab5c52e2ab8e675ad8445b8e1239),
DP eval GPU0/GPU1 passed batch300/reference128 under original2e-6 tolerances.
FN preflight running at0/10; child3049542/start735695108/cmd7061e9dec111ce425999682b73e2fdb5453a3986acf62a621006d9aa65b487b3.
Luna has ENDED; no duplicate short supervisor. Main reviewed snapshot hash,
queue/source identity. Formal training has NOT started; no MAIN_ACCEPTANCE exists.

Prior failed1fa8cd4 queue preserved: DP10/10 passed then eval family guard rejected;
repair only adds exact FFN family/order/mode/10steps eval binding and tests. No
math, model/config, tolerance or resource changes. Old failure receipt SHA
0cd174237b4d1d4c89e89384430bd6459c5bf5a5fae6cf4bbfe3eb63e5dc6561.
New dependency explicitly validates preserved scDFM completed fit + true GEARS
terminal, while scDFM eval remains deferred_by_user. No fake scientific COMPLETE.
scDFM checkpoint /data/yilangliu/GraD-Pert/runs-official-recovery-9607981/scdfm-jurkat-seed1-20261013T103748Z-1cba65fce8c941a49ea7cdf6a6f08455/fit/last.pt;
SHA9dff79f7e2836482ea6f272dd619c9df967e73f8e5b16f911679c3d557c62a8c.

## Supervisor actions and return barriers
Obtain exact lease/run/source/queue match, independently acknowledge live job.
Activate existing grad-pert-br400-four-arm-supervision for this run at20minutes,
record an ACK receipt and read back ACTIVE/exact target/interval/run. No other
monitor for this job. Keep healthy unchanged checks quiet; notify actionable
milestone/failure. Read bounded state/log/receipts/owned processes/GPU. Process
absence alone never implies completion. Never weaken policy or kill other users.

At PREFLIGHT_COMPLETE + waiting_main_acceptance, verify all five10-update/restore
and ten eval capacity receipts exist/hash/source/config/data consistent; collect
small evidence. You MUST NOT write MAIN_ACCEPTANCE or launch formal jobs. Pause
monitor, record return handoff, atomically return exact ownership to main/build,
main_continuation=pending, activate staged grad-pert-unified-first-return fallback
at20minutes and message main (explicit user authorized both directions). Main
must independently verify queue/gates/common identity/deferred official proof,
write exact {**PREFLIGHT_COMPLETE,approved:true} only after acceptance, then
controller automatically advances DP fit/test -> FN -> FG -> FB -> PO. New
long-training handoff/ACK required afterward; no standby supervisor polling.

On FAILURE, save compact terminal hashes/identity and return similarly; no hot
server edits, no retry/ID reuse, no capacity/tolerance changes. Main owns complex
repair. Do not launch other groups. At all actual terminal results, return for
scientific acceptance and report, not merely controller exit. Receipt source,
checkpoint/config/data/control/truth/DEG hashes and zeroPKL remain required.

Return-wake fallback staged PAUSED for this exact run; enable only when main
owns pending continuation. Pause supervision on every handback. Return monitor
is paused by main on substantive ACK, not on message transport. Immediate idle
notification has not been verified:20minute fallback is the proven wake route.

## Sealed formal row identities (not started)
- DP: nadig_jurkat-seed1-20261014T044125Z-9b3bae49aa1340b5bad2c3d7664b7f5b; config SHA826a1c9dd905045c4ca1a168a980e37cc1defc2bacffe3e460b97b4cde0cf19f; root/data/yilangliu/GraD-Pert/runs-native-ffn-leaf-3ee8ef3/nadig_jurkat-seed1-20261014T044125Z-9b3bae49aa1340b5bad2c3d7664b7f5b.
- FN: nadig_jurkat-seed1-20261014T044125Z-9f6148dc09cf45c1a71b05208bac72db; config SHA5281d7e5409ededb7488a6e06acc35cfd1e84dc037055830b3e0cbb88e37e06d; root/data/yilangliu/GraD-Pert/runs-native-ffn-leaf-3ee8ef3/nadig_jurkat-seed1-20261014T044125Z-9f6148dc09cf45c1a71b05208bac72db.
- FG: nadig_jurkat-seed1-20261014T044125Z-941e1600d83b456981c5f079cf425b40; config SHA7a48a09d6bfa4dae080d3bd5daf2bff7e20d2f3d914351e8719b40bc962ef3ee; root/data/yilangliu/GraD-Pert/runs-native-ffn-leaf-3ee8ef3/nadig_jurkat-seed1-20261014T044125Z-941e1600d83b456981c5f079cf425b40.
- FB: nadig_jurkat-seed1-20261014T044125Z-f857857908b34e178180ee44b7e188a8; config SHA27401644954cdef4703de827f2823f8abfc13734dfb2a7f376a6045ac0c33529; root/data/yilangliu/GraD-Pert/runs-native-ffn-leaf-3ee8ef3/nadig_jurkat-seed1-20261014T044125Z-f857857908b34e178180ee44b7e188a8.
- PO: nadig_jurkat-seed1-20261014T044125Z-6e3fa72233c24f79919dd4b923906762; config SHA7386c15de7536e28277c3008c916f9a253d9eeb772afbb221bfffc3eb1e14e27; root/data/yilangliu/GraD-Pert/runs-native-ffn-leaf-3ee8ef3/nadig_jurkat-seed1-20261014T044125Z-6e3fa72233c24f79919dd4b923906762.
