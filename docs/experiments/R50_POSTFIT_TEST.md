# R50 automatic best/final test delivery

User update 2026-09-10 supersedes validation-only delivery and best-only
retention for this program. Each full R50 coordinate still trains exactly50
epochs, validates each epoch and selects best by validation only. Afterwards
automatically evaluate best and the actual final-epoch checkpoint against the
same canonical test split, ordered300 controls, truth rows and three metrics.
Do not select hyperparameters or best/last reporting roles from test scores.
Integration smoke remains one validation-only epoch; it is not tested.

New R50 native runs retain best.pt and last.pt with hashes. The selection runner
seals training COMPLETE first, then calls the post-fit evaluator. Test artifacts
live in a separate sibling `<lineage>-test/<row>/{best,last}` root. Training
receipts remain unchanged and truthfully report zero test calls during selection;
post-fit receipts separately record actual test calls. Best and last with the
same SHA share one evaluation and explicit role aliases, never duplicate metrics
presented as independent results. No persistent PKL; temporary evaluator files
are cleaned by the existing metrics_only lifecycle.

## Historical migration (no training replay)

Training source f4bad63f4a4b8e2fd8803bb8fc18cf9f74029bbe remains immutable at
/data/yilangliu/GraD-Pert/development/source-r50-f4bad63-formal.
Training root /data/yilangliu/GraD-Pert/runs/r50-f4bad63-v1.
REF and LR-low are complete50; REF best is epoch50, so final can alias best.
LR-low best is epoch40 and last was already deleted: mark final unavailable,
as expressly accepted by the user; do not rerun it.

LR-mid uses an external Linux inotify checkpoint-preservation process. It links
only atomically published last.pt into a separate archive; replacing/deleting
the training pathname does not delete its archived inode. Never alter checkpoint
bytes or active source. If an event is missed, the evaluator must reject any
archive that is not exactly epoch50 with29100 steps and matching source/config/
data/split identity. An intermediate checkpoint cannot substitute for final.

Post-fit inference runs in a new clean published checkout. Receipts distinguish
training source from evaluation source. Native model/graph/inference/metric code
is unchanged; new code orchestrates existing APIs, strictly loads all checkpoint
state, verifies canonical/graph identity and creates test access claims before
opening test data. Partial/failed test attempts are never automatically retried.
A complete output is reusable only after its file hashes verify.

## Launch and monitoring handoff

After CPU tests/lint/types/build, clean main publication and matching server
checkout, complete the preparation goal before launching the CPU archive daemon
and GPU1 post-fit queue (REF, LR-low, then LR-mid when training completes).
Queue checks the chosen physical GPU UUID for other compute processes and waits
rather than evicting them. One GPU is used for inference while GPU0 continues
LR-mid. Do not introduce concurrent fits without capacity/throughput evidence.

Restore one hourly quiet heartbeat for the named archive/test/training sessions.
Check bounded logs, source/checkpoint/receipt identity, PKL/resources, terminal
receipts; notify only completion/failure/material change. Preserve failed roots
and stop scheduling new work on any failure. Never replay canceled old rows.
After all three full training and available-role tests complete, validate outputs,
freeze the LR decision using validation scores alone, disable the monitor before
the next bounded preparation goal. Prepare the next batch/scheduler stage and
bounded multi-process capacity/throughput probes on idle resources; no change to
an active source/config. Advance only reviewed new configs; no automatic full
retraining to fill missing historical weights.
