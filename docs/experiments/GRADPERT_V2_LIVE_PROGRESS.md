# GraD-Pert v2 live progress

New v2 runs write `fit/live_progress.json` atomically from rank 0. Training
updates record the epoch, completed optimizer steps, latest loss terms, elapsed
time in the current process, and observed cells/second since the beginning of
the current epoch (including data preparation between updates). Validation and
best/last tests record completed and total perturbation conditions. The
diagnostic file does not participate in loss computation, checkpoint selection,
or formal result receipts.

To watch an active server run:

```bash
python scripts/v2/show_progress.py --run-root "$GRADPERT_RUN_ROOT" --follow
```

The viewer prints a new line when the atomic status changes. `--interval`
controls the polling interval in seconds (default: 2). A resumed run reports
the committed global optimizer step, while elapsed time and throughput restart
for the new process. Older runs do not have this file; their original logs and
receipts remain the authoritative evidence. Completion still requires the
run's `COMPLETE.json` and best/last test receipts.
