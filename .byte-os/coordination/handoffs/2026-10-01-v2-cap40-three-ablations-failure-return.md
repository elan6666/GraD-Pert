# E1 completed; ablation queue failed during result collection

Queue `v2-cap40-ablations-ac220f5-20261001T060835Z`, attempt 1, was supervised by `codex:01a0df0b-4142-7df1-86c0-d959471d80a1` and is returned to main `codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b` for diagnosis.

## Evidence

- The durable controller wrote queue `state.json` with `status=failed`, row `E1_no_mhc`, and `FAILURE.json` with `ValueError: run has no supported fixed-epoch v2 contract`. The traceback is in the queue controller log under `next_action -> collect_run`.
- E1 formal subprocess exited 0. E1 `COMPLETE.json` records 6 epochs, joint-loss best at epoch 6 (4.235658792155981), epoch-6 checkpoint SHA `9f18f995f16e9d7852c8fff377cc2ccad9ed845cce210b8be8c24b37041bb86d`, best/last test roles, and `zero_pkl=true`; `fit/best-test.json` and `fit/last-test.json` exist.
- The queue-level `COMPLETE` marker is absent. E2 and E3 formal training did not start. A scan of all three pinned run roots found zero `.pkl` files.
- Source commit `ac220f5e90e9e5c0c9900cdb2c285a629297c93d` remains immutable; no server source/config was changed, no retry or relaunch was attempted.

## Ownership and next action

The supervisor monitor `grad-pert-v2` is paused and verified. Ownership returns to main; `main_continuation` is exactly `pending`. Main should inspect the failure and E1 receipts, determine whether the collector contract issue can be safely fixed, then decide on any new attempt under a new run ID. Do not continue querying or relaunching this failed queue from the supervisor.

Receipt: `.byte-os/coordination/receipts/v2-cap40-three-ablations-queue-failure-20261001T2141Z.json`.
