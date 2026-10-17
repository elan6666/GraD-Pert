# PT1 saved-report repair; preserve fit/test and failed queue

Main received terminal receipt67b6c70ea8fd8d7535229b02dd2beb83c9cef95b34140403ba15c5fd20a57530
and handoffb705d29066f4c05d93445861879b2a8bb37f63893377eea67c59c76168b4d8b7;
exact owner/main/build ACK recorded, both old monitors PAUSED. Baseline884c6f7572a435d7a966b89fff729009a011316c
is clean/published before this scoped change. No retraining, checkpoint/source/queue mutation.

Evidence: original epochs0/1/2 process input hashes equal fdb25b40...cb78;
epochs3/4 equal ee59648f...f6b0. Runtime intentionally removes SSL views and loads
numeric truth only in MSE phase. A single cross-phase input hash is invalid.

Repair only CPU reporting: derive exact epoch/phase order from sealed staged
config, require one input hash per phase, unchanged diagnostic size/seed/budget,
and keep ordinary non-staged drift rejection unchanged. Record per-phase hashes;
cross-phase probes are explicitly different, not pooled as identical samples.
Break plotted trajectories at the boundary, show absent MSE/Teacher/CE as N/A,
and use same-phase endpoints for representation/entropy tables. No guard deletion.

Allow an independently published PT1 report-repair controller just as D0: exact
PT1 family only, fresh independent output, original method and numerical analysis
scripts byte-for-byte unchanged. CPU synthetic phase/drift/render tests, then
real CPU report. Following original reference/fast gate and fixed comparable
medium analysis preserves1335conditions/5317rows, deep128/506, gradients32/125.
Verify all18 last metrics and checkpoint/data identity independently.

After comparable-medium acceptance, run prepared bounded diagnostic captures
then advance DP2. Do not call the old failed queue complete or restart it. Append
repair receipts and archive figures/aggregates; no scientific matrices pulled.

Local directed40tests passed (20.94s); Ruff/format/diff pass. Frozen src and
numerical medium/parent generators unchanged relative to trainingec32671.
Main independently verified epoch1/2 SSL150/300 steps and epoch3/4 MSE450/600,
last checkpoint520b71f4...b0f, all18metrics/test SHA20bc22c7...d68, zeroPKL.
Real CPU saved report and fixed comparable medium still pending.
