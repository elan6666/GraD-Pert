# Scoped resource admission — approved, deployment pending

Observed 2026-10-15: both physical RTX5090 cards have one UID1003 process running
`run_scipenn_strict_b_s2_smoke.py`, approximately 704/702 MiB respectively.
The completed three-condition reference diagnostic peaked near 8 GiB in prior
measurements. Memory capacity permits coexistence; shared compute throughput
still requires measurement and is not guaranteed by free memory.

The proposed policy adds this exact script to the previously approved
ScButterfly/TotalVI programs. It preserves UID1003, at most one foreign process
per card, foreign memory <=1024 MiB, physical free memory >=2048 MiB, and our
per-card aggregate process memory <=90%. Unknown identities and other programs
remain rejected. A valid old policy continues to reject this script.

`scripts/v2/shared_gpu.py` contains the prepared validator;
`tests/v2/test_shared_gpu.py` verifies the new program and retained UID, script,
memory, headroom and explicit authorization boundaries. No queue, existing
policy, or server source has been changed yet. The user explicitly answered
"允许按相同上限共用" on 2026-10-15 for the current medium analysis and subsequent
already-authorized ablations. The exact authorization reference is
`2026-10-15-user-scipenn-s2-memory90-coexistence`. Runtime activation requires a
new published source, fresh run ID and policy receipt; failed runs stay intact.
This is a resource scheduling change, not a scientific method change.

Why confirmation is needed: the project's resource rule says
"Do not weaken resource gates without explicit scoped authorization."
Previous approval covered different exact script names.
