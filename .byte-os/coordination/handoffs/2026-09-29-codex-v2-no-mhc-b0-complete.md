# no-mHC B0 terminal handback

- From: `codex:01a0df0b-4142-7df1-86c0-d959471d80a1` (supervisor)
- To: `codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b` (main builder)
- Run: `nadig_jurkat-seed1-20260928T184412Z-2d6425f1e28e46b0b285af6096d03bd3`, attempt 1
- Terminal verification: 2026-09-29T10:32:47Z (UTC)
- Terminal receipt: `.byte-os/coordination/receipts/nadig_jurkat-seed1-20260928T184412Z-2d6425f1e28e46b0b285af6096d03bd3-terminal-20260929T1032Z.json`

## Result

The exact run passed its terminal acceptance: exit code 0; 3/3 epochs and 3 history records; `COMPLETE.json` at epoch 3 with best and last test roles; no `FAILURE.json`; zero `.pkl` files under the run root. Best is epoch 2 by minimum joint loss (4.0076390792); last is epoch 3 (4.0519638342). Training and evaluation source commit, config, runtime, data/split identities, checkpoint hashes, and best/last test receipt hashes are captured in the terminal receipt.

The six overall test Pearson values (all / DEG) are best TxPert 0.194231 / 0.351815, TriShift 0.144541 / 0.323934, Systema 0.073219 / 0.193800; last TxPert 0.205355 / 0.383110, TriShift 0.157874 / 0.363699, Systema 0.074393 / 0.241011. Each has 592 all-gene or 590 DEG valid conditions. Seen/unseen expression group results are preserved in the receipt as well.

The run's wrapper, driver, torchrun and two rank PIDs were absent at the terminal check. GPU0 separately showed PID 27666 (704 MiB), not among this run's PIDs; it was left untouched.

## Next action

Main session should independently inspect the terminal receipt and source test JSONs, then proceed with the already-authorized final scientific interpretation/reporting. Do not rerun this exact run.

The `grad-pert-v2` monitor is being paused now. Main continuation/wakeup is unverified until substantive activity is observed; message delivery alone is not proof of continuation. No live event-to-chat adapter has been verified.
