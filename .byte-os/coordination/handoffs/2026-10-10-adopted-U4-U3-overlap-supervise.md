# U4 evaluation beside adopted U3 training

Main: codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b.
Supervisor: codex:01a0df0b-4142-7df1-86c0-d959471d80a1 (local).
Run v2-overlap-bd2cfb2-20261009T172518Z, attempt1. Controller1941806; root /data/yilangliu/GraD-Pert/development/v2-overlap-bd2cfb2-20261009T172518Z.

The user requested train/eval overlap, with U4 evaluation before U3 as fallback.
U3 was already active. Published evaluator bd2cfb2ab764444b3482dc630f567d8273fc86dc
(local/server44checks). Original U4/U3 training remains1929cba76bee4c7f868083616a105d2507c7fa54.
Only old controller1866440 was retired; child1929837, torchrun1929841 and
ranks1929867/1929868 kept running unchanged. Do not query or restart old controllers.

## Actual launch and startup evidence

Server source /data/yilangliu/GraD-Pert/development/source-v2-overlap-bd2cfb2;
publication 83fa6192b9913b134fecec425d40176d0688a1de06c43195ee1f42887d51776c; runtime /data/yilangliu/GraD-Pert/development/v2-overlap-bd2cfb2.runtime.json,
SHA 83f6a17f833d6910767b296746830c31d0626f3509ee1faa7335d5f230bcc2ae.
Launch sibling .launch.json SHA 6d273929f32879aa8be7886aa30a2275b7fa94f0c3a913ce5ebb46b16aa1c813; input schedule sibling
.schedule.json SHA 78daac267e05f4131ddbf4821b2b38e8a128aeffb0e7a8e5e6b9022cd1349aa6. Copied root/schedule.json
SHA ed7b1d19632eb0a536a6ddc1806640370a44f2a421a4c9acaf0699f7b7922bfe (canonical JSON formatting differs; payloads must be equal).
Log sibling .log; PID sibling .pid. Root/state.json, COMPLETE.json, FAILURE.json.
Server STARTUP-VERIFIED.json SHA 05426ee32635518da473f69bfb71ed8f0d58e1a460aad1397f99c0ce2aab4a94.
Local receipt coordination/receipts/adopted-overlap-startup-20261010T0127Z.json.

U4 root /data/yilangliu/GraD-Pert/runs-v2-deferred-1929cba/nadig_jurkat-seed1-20261009T071927Z-b9c03b0d497c41c18d47edb33f63b566; trained6/6, checkpointlast827fb7a8b2a877a8a5d31b71ea1efb119f92c4820d8f4d490659a636dde583f0.
U4 evaluates full592 conditions on GPU0, original batch128 and frozen300controls,
query1000; CPU Teacher/optimizer state, CUDA allocator cap0.19, unchanged metrics.
Current startup12/592. Evaluation plan SHA 3060e1a3f6f2ce3ca5e4c0335102b6a1c65e92e1d76834d91861a9701e30054d.
U3 root /data/yilangliu/GraD-Pert/runs-v2-deferred-1929cba/nadig_jurkat-seed1-20261009T071928Z-2272edf0ab7c417fb0f6078564a619e6; same6epochs/global272=m68*a2*world2/no validation;
startup epoch2 step21/176,197/1056updates. Both ranks alive.

## Diagnostic evidence and limits

Probe32/32passed,341.33s, receiptSHA aae1a816ba6c347a84c2511eaf7ed0706335ac3c945d48a82cae5752a113a6b2; eval peakreserved4.77GiB.
U3 updated166→179,25.74s/update vs adjacent baseline21.39s (~20.3% slower).
Maximum sampled sharedGPU0memory25705MiB; no OOM/nonfinite.
This is short feasibility evidence. Full-run stability/makespan gain is unverified.
Report in docs/performance/2026-10-10-u4-u3-overlap.md; do not claim long-run speedup.

## Assigned supervision

Reactivate the SAME grad-pert-loss monitor20min for this exact run/attempt; report
on each check per user preference. No duplicate active short helpers remain.
Read-only repair_count0. Observe active schedulePID/log/state, U4 POSTFIT_STATE,
postfit-last/worker-0-progress.json, independent COMPLETE/FAILURE, canonical
fit/last-test and parentCOMPLETE; observe U3 live/epoch_state/TRAIN_COMPLETE,
child/ranks/GPU and FAILURE. No extra fit: existing U3 is adopted by cmdlineSHA.
After U4 evaluation finishes, controller waits original U3 fit exit and verifies
TRAIN_COMPLETE, then evaluates U3 exclusively on idle GPUs0,1 using disjoint frozen
conditions and same recipe. Completion requires both parentCOMPLETE, last roles,
training/eval/config/checkpoint/data/orderedcontrol/truth hashes and zeroPKL.

If OOM, failure, missing receipt or stalled job, preserve evidence and return to
main/build/pending immediately; pause thismonitor, activate existing exact-main
return fallback and message main. User fallback remains U4evaluation first then
same-source full-state U3 resume. Supervisor must NOT kill/restart U3, lowerbatch,
change methods, overwrite runs, or automatically retry shared allocation.
If stable-window measurements suggest poor overall throughput, return the evidence
for the same fallback decision; do not infer it solely from momentaryutilization.
Normal full completion also returns to main for metric comparison; no extra groups.

Monitor exists only while this assigned job is active. After handback pause/delete;
verify actual main ACK. Periodic20min wake is verified, instant event wake unverified.
