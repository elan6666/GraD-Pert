# Selected-device admission for dataset300 diagnostics

The user authorized the remaining dataset300 diagnostic analyses to use physical
GPU0 independently of unrelated work on GPU1 on 2026-11-03. This changes the
resource allocation of a new analysis attempt; it changes no training, inference,
model, data selection, statistic or checkpoint behavior.

A new sealed `dataset_transfer_medium` execution spec opts in with
`admission_scope: selected_device`. Its `gpu` and `selected_device` bind the exact
physical index and UUID inherited from the original TRAIN runtime. The original
`shared-gpu-admission-3` policy remains unchanged, including foreign UID/program
allowances, one foreign process per card, 1024MiB foreign allowance, the original
own-memory fraction and 2048MiB physical free-memory reserve. For this family the
original own-memory fraction is 0.90. No program is added to the whitelist.

Selected-device admission applies these checks only to the allocated card. It
rejects any owned descendant observed on another card. It retains the original
policy validation and shared budget checker; unrelated foreign work on the other
known card does not consume this allocation. Telemetry and identities are still
collected from both cards rather than being rewritten to pretend the other card
is idle. Launch admission, the controller's continuous 15-second checks, and the
worker's phase/condition checks all select the same scope from the sealed spec.
The existing nonblocking physical-card lock, owner and child identities,
`CUDA_VISIBLE_DEVICES`, expandable-segments allocator and cuBLAS contracts remain.
A guard failure terminates only the identity-checked owned child tree.

Specs without the new field retain the original whole-policy behavior. Existing
training controllers continue using whole-policy admission. Historical failed
attempts remain failed and immutable; an unlaunched old spec is not silently
converted. New behavior requires published source, a fresh output, a new runtime
publication binding, compatibility approval against the original training source,
and a new scope-bound completion receipt. Compatibility must independently check
unchanged modeling/training/data/evaluation/config blobs. It is not a new claim
of scientific improvement or completed analysis.

Synthetic tests cover off-card unknown/large work and low/missing reserve,
selected-card foreign identity/count/growth, physical UUID, missing telemetry,
reserve and summed descendant budgets, owned off-card escape, sealed scope
mutation, and continuous native-loop guard failure/owned-child cleanup. Actual
native completion still requires the complete schedule, used-prediction/token/
full-graph parity, zero updates and runtime-state restoration, original checkpoint
and population bindings, whole-family zero PKL, and independent main acceptance.
