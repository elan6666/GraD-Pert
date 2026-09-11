# R50 schedule coordinate

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
