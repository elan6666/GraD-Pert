# R1024 handoff

Requested: eleven independent 50-epoch rows, batch1024; two GPUs, two processes
per GPU when capacity allows. Retain E3, no prior comparisons/combinations/K1.
Baseline published main 7f31ba47af9f6ad54f3ff6d6d6f3fc0af514a6fc; isolated worktree
/tmp/gradpert-r50-e3 avoids unrelated primary checkout changes.

Implementation: configs and exact essential/matched-size locals, configurable
EMA start, two-layer Exphormer, one-step four-lane queue and automatic full50
best/last test runner. EMA config was previously not passed to the engine;
historical default .996 remains unchanged. New tests exercise actual update.

Before CUDA: publish new commit, identical clean server checkout, full gates,
live publication receipt, hash-pinned input plan, idle GPU/resource check.
Prepare plan for scripts.server.run_r1024_steps; it performs only one-step tests,
never formal50 automatically. Initial four slots S1/S2/U1/T1; remaining rows
stay assigned by ROWS[slot::4]. Each process has allocator fraction .4.
Inspect all terminal receipts and overlap before releasing formal rows.
Capacity failure preserves evidence and blocks that lane, not a scientific
configuration change or automatic retry. One-step is not sustained capacity.

Monitor grad-pert-r50 is paused during active preparation goal. After acceptance,
complete goal, check get_goal, launch fresh preflight root, restore same monitor
at 30 minutes with concise updates. On terminal results pause monitor, create
bounded review/next-launch stage; prioritize approved eleven coordinates.
Formal runner: scripts.server.run_r1024_full requires matching one-step SHA,
source/config/prior/publication and runs50 then both checkpoint tests.
No long goal during CUDA waiting. No existing jobs or evidence overwritten.
