# B2-vNext pre-formal review — 2026-08-28

## Verdict

Ship the implementation to one clean public commit and proceed to formal data
and one-epoch CUDA preflights. Do not launch the 10-epoch matrix until those
receipts pass.

## Scope reviewed

- HVG512-plus-target graph and dataset scale contracts
- Fanout/ring view builders and anchor masking
- Native single/multi graph encoders and STRING weight routes
- additive/MLP/control-conditioned decoders
- exact GenePT emb_b coverage and four feature routes
- common training/checkpoint/EMA path
- 22-row config matrix and queue orchestration
- metrics-only/zero-PKL and seven-systems contracts

## Findings resolved

- Transformer BatchNorm is computed independently per training view.
- W1/W2/W3/WS use one full-topology frozen STRING mapping across all crops.
- Sparse prediction union tensors are resident when the systems flag is on.
- Manifests reject unrelated, duplicate, or reordered graph-axis evidence.
- D2 and every GenePT route update trainable parameters, update the teacher by
  EMA, and round-trip the complete checkpoint/optimizer state.
- Missing GenePT perturbation targets stop the family before model construction
  and produce a sealed unavailable receipt.

## Verification

- Local: 253 passed, 11 honest skips; Ruff, format, diff, focused mypy, build.
- Server development: 315 passed, 3 pending-formal-receipt skips; Ruff, format,
  strict mypy on 72 files, isolated build.
- Review: no remaining P0/P1 in the model/weight/resident-cache scope.

## Remaining formal gates

1. Commit/push and synchronize the same clean commit locally/GitHub/server.
2. Materialize and review corrected five-dataset prepared receipt chains.
3. Materialize exact-commit Jurkat HVG512 and GenePT availability receipts.
4. Run and strictly validate A0 for exactly one epoch before queue launch.
