# GraD-Pert v2 acceptance ledger

Scope remains the full implementation, two-5090 capacity testing, grouped runnable
experiments and verified supervision handoff. This ledger does not reduce it.
Source inspected: `173f81b7cc7f083497d86398f94f10d658d7e049`.

| Requirement | Current evidence | Acceptance gap |
| --- | --- | --- |
| Config-selected v1/v2, historical v1 behavior | `config/v2.py`, `execution/train_entry.py`; native trainer/resume regression79 passed,1 CUDA-only skipped; final focused trainer6 passed | Final combined regression on delivery source and real server entry |
| V2 method | `modeling/v2`, `training/v2/objective.py`, native operator tests and joint-loss capacity probes | Final design-to-code audit; no scientific efficacy claim |
| Reuse v1 infrastructure | Canonical data/evaluation, metrics, entry, source checks, selection, RNG restoration, curves and `training/epoch.py` are shared | Review outer lifecycle/postfit differences; preserve necessary teacher/rank-state adapters |
| Small evaluation artifacts | Schema metrics_only, best/last retention, compact validation population, no-PKL completion check | Real complete-validation lifecycle with curves and best/last receipts on delivery source |
| Two-GPU engineering evidence | micro2/8/16/32,16×accum2 and two-rank micro8 completed; original effective32 receipts in `evidence/v2-capacity/` | micro64 and same-source micro32/micro64 repeats passed; final measured batch decision and per-variant checks remain |
| Hyperparameter-first groups | `experiments/v2/groups.json`; standalone generation, validation-only selection, follow-up dependency gates | H3 levels/configs depend on capacity; final complete generated matrix not frozen |
| Execution/recovery/collection | `scripts/v2/run_group.py`, `resume.py`, `collect_results.py`; tests and source gates | Actual server queue dry-run, per-row integration checks and source parity |
| Expression holdout/context generalization | Holdout manifest, training exclusion/poison tests, fixed-axis context-query API | G1 canonical runner implemented and tested; concrete task configs and real checks remain |
| CLS diagnostics | Response CLS→gene intervention and tests | D1 canonical runner implemented and tested; Norman server engineering check running |
| Five datasets and baselines | Existing canonical/shared infrastructure preserved | Five dataset single-update/reload probes passed at35c6c1a; official/v1 baseline startup preflight remains |
| ZCode supervision | Relay boundary remains Codex build/test first | Exact destination resolution and complete verified handoff package |

Do not use a single-condition capacity inference as evidence for full validation
metrics or50-epoch lifecycle completion. Do not use a one-step integration pass
as sustained memory evidence. Best/last test scores must never choose a parent.
Actual training/evaluation SHAs remain attached to each original receipt; later
code publication must not relabel historical evidence.

The v1 R50 task described in `STATUS.md` is separate historical/parallel work;
this ledger and the v2 plan track this implementation without rewriting its run
state. Mutable active probe details remain in the task's implementation state log.
