# Combined E2+E3 fresh twenty-epoch run

User correction: one combined E23 run, not separate E2/E3 runs. From scratch20epochs, seed1. Four prototypes16384, SSL1(condition,node,spread)=(1,1,0), SSL2(DINO,iBOT,KoLeo)=(1,1,0), lambda1=lambda2=1; mHC4 enabled. Existing six-epoch controls frozen.

Keep cap40 fixed selection, m68 x accumulation2 x two GPUs=global272, single-pass final-state KDA, legal graph neighborhoods, random order/source gates/views/expression exclusion and all scientific paths unchanged. Fresh warmup+cosine applies to3520updates: max1e-3,min2e-4,warmup fraction.16. Joint-only validation/min-joint best, last20; automatic frozen300-control best/last all/DEG and seen/unseen tests; zeroPKL.

Self-contained config configs/v2/cap40_combined_jurkat/E23_prototypes16384_unit_distillation/gradpert_v2/nadig_jurkat.yaml. Existing cap40 runner adds separately sealed --combined-twenty mode: exactly one row, no E1 import,128-update sustained capacity on exact source/config, then fresh20epochs/tests. No silent capacity fallback, resource gates unchanged. Separate source/config/run identity, no overwrite.

Main: implementation/checks/publication/immutable runtime/durable controller start. Short CPU checks use Luna; long resource wait/preflight/training uses designated supervisor20min with exact ownership/ACK and return wake. Main independently accepts terminal receipts and removes monitors.
