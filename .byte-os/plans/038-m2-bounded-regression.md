# M2 bounded regression

The sealed M2 formal attempt failed after 28 completed steps on a one-node
local STRING graph. P1 executes only one step and P2 only 25, so neither is
adequate evidence that this path is repaired.

Add a separate training-only `m2_regression` stage with exactly 32 consecutive
frozen batches (steps 0 through 31), heavy capacity instrumentation, no timing
acceptance, and the existing validation/test guards. Restrict it to M2 and
require a hash-pinned P1 from the same source, config, GPU and frozen inputs.
Do not change the P1/P2/P3 protocols or broaden the scientific matrix.

Acceptance also needs observed execution of the singleton normalization path:
32 steps alone are not proof if the frozen batch sequence does not encounter
the original fault. Add diagnostic-only singleton-call evidence and require a
positive count before declaring the regression passed. Preserve normal forward
order, RNG and normalization behavior. This coverage instrumentation and its
tests are implemented in the bounded worker. A diagnostic wrapper records only
successful singleton normalization calls and their step indices, returns the
original result unchanged, and restores the original method after execution.
Missing coverage fails closed. Synthetic tests verify step-28 coverage, exact
32-step termination, restoration, and rejection when coverage is absent.
Targeted worker/census tests: 83 passed; Ruff and diff checks pass. Full local
and server checks, final review and publication remain pending; this stage is
not yet ready to launch. Final local full suite passed 516 tests with 10
environment/reference skips; full Ruff, format and diff checks passed. Review
confirmed hook installation is inside the protected execution block and native
normalization is unchanged. Server runtime checks remain required because local
PyTorch is unavailable.

After local review/tests, publish on main only, run exact server gates, create
fresh source/publication/P0/batch/P1 contracts, and run the bounded regression.
Preserve existing 1bc5862 P1 receipts; no overwrite or relabeling. Formal M2/M4
remain unlaunched until regression and launch review pass. Existing M4 capacity
results do not establish safe same-GPU concurrency or a full-run memory peak.
