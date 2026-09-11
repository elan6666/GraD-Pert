# R50 immediate post-fit testing

2026-09-11 user decision: remove the whole-GPU-idle test queue gate.
Completed training proceeds directly to best/last testing; evaluator free-memory
reserve checks, canonical evaluation and zero-PKL policies remain unchanged.
Batch128/512 already use this direct chain. The legacy migration queue now does
likewise (regression test covers all three rows with idle polling forbidden).

Live handoff: terminate only the waiting legacy queue PID 3280734 after command
identity verification. Run its existing clean published 252c9b3 evaluator with
`--row lr_mid` on physical GPU0, preserving the same output root and archived
epoch-50 last checkpoint. This explicit row path bypasses the legacy outer wait;
no active server source is edited, no completed row is reevaluated. Fresh session
`gradpert-r50-mid-direct-gpu0`, log in the original contract as
`lr-mid-direct-gpu0.log`. Require absent LR-mid output before launch.

Monitor hourly with short progress each time. Verify terminal exit and
postfit.verify_existing, both checkpoint roles and canonical hashes; never
relaunch an incomplete evaluation automatically. Training failure must never
advance to testing. Tests are reporting only, not parameter selection.
