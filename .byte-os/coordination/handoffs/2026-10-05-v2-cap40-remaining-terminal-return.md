# Terminal handoff: cap40 E1/E2/E3 queue complete

Queue `v2-cap40-remaining-2871c4e-20261003T072407Z`, attempt 1, is terminal `complete`. The controller exited after E3; queue COMPLETE SHA256 `4c2ff13cee70a74b5e19359541ab556dfdb8fbe99d3326b145c71d42937892e1`, no queue FAILURE, and queue-root PKL count is zero. GPU snapshot at completion was 2 MiB each on GPU 0/1. The source is published, clean commit `2871c4e79a70b7d0d4f900866ce019316b68670f` (runtime SHA `de87592aa86462e843dfa65ce0960ab78c4c1ea2e999dae144eaed1d3210ae60`, publication SHA `1956abd0f30b2a1895461bc34993ede55287a1c803d0194594d5029a258d367b`).

The replacement queue preserved previously main-accepted E1, commit `ac220f5e90e9e5c0c9900cdb2c285a629297c93d`, six epochs, best/last epoch 6, both true tests, zero PKL; it did not rerun E1. E2 and E3 each passed their preflight and formal stages (both exit 0), completed six epochs/history, selected minimum joint loss at epoch 6 (same checkpoint as last), produced genuine best/last frozen 300-control tests, seen/unseen-expression metric groups, no FAILURE, and zero PKL across each run. Training and evaluation commits match per E2/E3 and equal `2871c4e79a70b7d0d4f900866ce019316b68670f`; data/split/control/reference/selection identities are recorded and consistent. Details and SHA256s are in `.byte-os/coordination/receipts/v2-cap40-remaining-terminal-20261004T1346Z.json`.

Key all/DEG Pearson (best and last are identical):

| Arm | TxPert all / DEG | TriShift all / DEG | Systema all / DEG |
| --- | ---: | ---: | ---: |
| E1 no-mHC (accepted earlier) | 0.218007 / 0.381337 | 0.169879 / 0.353930 | 0.075271 / 0.214566 |
| E2 prototypes 16384 | 0.209193 / 0.355427 | 0.164478 / 0.319862 | 0.081327 / 0.201001 |
| E3 no spread/KoLeo | 0.217075 / 0.381580 | 0.168728 / 0.346391 | 0.088424 / 0.217086 |

The supervisor's current turn verified the terminal receipts over SSH. This handoff returns ownership to main session `codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b` for independent acceptance and cross-arm comparison only; do not tune on test metrics. `main_continuation` is set to the exact string `pending`. No GPU process remains for this queue; do not query the server as an idle monitor.
