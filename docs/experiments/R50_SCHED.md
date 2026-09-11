# R50 schedule coordinate

Latest user decision 2026-09-11: use LR .001 and batch512 provisionally for the
next stages. New row `sched512` directly compares with completed `batch512`;
only the LR schedule changes. This is a user-chosen parent, not a claim that
batch512 won the still-incomplete batch comparison. Evaluation batch stays256.
The old `sched` batch256 contract at 3c30d1b was not confirmed launched and is
superseded: do not execute its launcher. Preserve it as unlaunched evidence.
Expected full budget is50*335=16750 steps; warmup8*335=2680 steps. Runtime meta
must confirm the actual steps_per_epoch. Original EMA is unchanged.
Network currently reports Can't assign requested address. On recovery first
check old session/root/log for unexpected activity, then create a fresh clean
published sched512 server checkout, run full gates, make a new hash-pinned
contract and launch on GPU0 only after preparation goal ends. GPU1 batch128
continues untouched. Do not reuse 3c30d1b config/publication hashes for sched512.

The following paragraph records the prior batch256 design, superseded above.

2026-09-11: LR-stage best validation scores REF .4069091270569533,
low .3428432147008625, mid .34808985119780694 select REF at .001.
The difference exceeds the preregistered .002 practical tie threshold.
Test results are not selection inputs. SCHED can proceed independently of the
still-running batch128 comparison because it keeps REF batch256.

Only scheduler changes: 50 epochs, 582 steps/epoch, 29100 steps; linear warmup
from zero to .001 over 4656 steps (8 epochs); cosine decay toward .000001 over
the remaining steps, no restarts. The last used LR is slightly above the floor;
the terminal schedule boundary equals the floor. Native LR-only schedule leaves
the original .996-to-1 teacher EMA path and endpoint arithmetic untouched.
All data, model, losses, views and training/evaluation batch settings match REF.

Fresh one-epoch integration then fresh50 with automatic best/last canonical
tests, no early stopping and zero persistent PKL. GPU0, reserve4GiB; GPU1 batch128
is not stopped or changed. New clean publication and full gates required before
launch; never edit an active server checkout. Hourly monitor records progress,
terminal validation and test receipts and advances the authorized matrix only.
