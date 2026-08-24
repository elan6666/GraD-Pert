# Iteration 1 — Shared Representability and Sustained Capacity

## Evidence trigger

The first official GEARS K562 integration built the official graph but exposed
conditions whose targets were absent from the frozen default graph. The first
65,536-prototype one-step probe also understated sustained CUDA reservation.

## Change

- Freeze one official-GEARS representability intersection and apply it to every
  model after the source split without reshuffling retained conditions.
- Rebuild all five datasets-v2 split/control/graph/evaluator receipts.
- Replace the one-step head probe with 128 consecutive real steps on every
  dataset and one global selection threshold at 85% initial free memory.

## Measured result

- Both frozen GEARS resource hashes are recorded; no retained target is missing
  from both graph sources.
- 65,536 failed K562, 32,768 failed Jurkat, and 16,384 passed all 128 steps on
  all five datasets. Worst peak reserved memory was 24,226,299,904 bytes versus
  a 28,168,037,990-byte threshold.
- Every native config now freezes `prototype_count=16384`.

## Regression evidence

Server data/graph/evaluator verification passed for all five datasets. Local
split regression reconstructs each datasets-v2 condition list and exact split
hash from the archived source assignment plus frozen exclusions.
