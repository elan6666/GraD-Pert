# Resume the already authorized DP2 dependency

User asked why training had not continued after D0 and the diagnostic/report work
completed. The next arm remains DP2, followed by S12-L4 and the previously approved
remaining family. A status-only reply and completed repair do not revoke that
authorization. Main owns the launch; the designated supervisor owns the active
long-running job after exact ACK and verified 20-minute registration.

Pre-change baseline c5a8e598c4cd3dbc5e1b3b9697badaf73693d7d1 was clean locally and
matched live GitHub main before edits. Preserve all old queues, failed receipts,
B0-DP/D0 runs and completed analyses. Do not replay completed arms.

The new singleton DP2 manifest uses precisely the previous DP2 YAML bytes/hash
e79224152e1d879fb3718694dc2b026e03b85404b6ccec419e3132624e68f374.
Only orchestration scope changes: new schema limits prepare/update/evaluation and
medium-analysis dispatch to DP2. Original two-arm manifests remain byte-compatible.
No modeling, objective, optimizer, schedule, data or metric implementation changes.

Publish scoped code/tests/config on main; create a clean independent server source,
publication receipt, runtime, queue and run ID. Keep the approved resource policy.
Require 10 successful dual-GPU updates with restore/inference diagnostics and both
eval128/reference128 leaves; main independently verifies all receipts before
writing exact MAIN_ACCEPTANCE. Never infer success from a vanished process.

Formal protocol: Jurkat cap40, seed1, from scratch four epochs, micro80 per GPU,
two GPUs, accumulation2/global320. No epoch validation/best selection; epoch4 last
only, same 300 controls and 18 Pearson variants. Follow with unchanged medium
population (1335 conditions/5317 rows; deep128/506; gradient32/125), parity and
zero-update/state-unchanged checks. Report renderer already recognizes DP2.

Bounded preflight waiting uses one authorized Luna subagent; main does not duplicate
its polling. After formal fit actually advances, record process/source/config/run
identity and hand off to the exact supervisor chat. On terminal return, supervisor
pauses its monitor and wakes main for independent acceptance, report archival and
the next authorized S12-L4 dependency. No reauthorization merely for that transition.
