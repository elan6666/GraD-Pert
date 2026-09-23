# Current GraD-Pert v2 Jurkat baseline

`default/gradpert_v2/nadig_jurkat.yaml` is the parent for **new** Jurkat v2
ablations. It keeps the complete model and both distillation branches: trainable
GenePT-only PCA256 seed table, three KDA layers, one unordered-gene
content-indexed sparse MLA layer (Top500), clipped SwiGLU, global row-mean loss
reduction, and cross-rank KoLeo. The model width is 256. Training is the project
five-epoch warmup+cosine protocol on two GPUs; global batch 128 is per-rank
microbatch 32 with accumulation 2. Historical B0/B1 and earlier v2 group configs
remain pinned to their original source and must not be reinterpreted as this
baseline.

`capacity_m{40,48,56,64}_a2` are **engineering candidates only**. They change
just per-rank physical microbatch and effective global batch (160/192/224/256).
Each requires its own immutable two-GPU capacity receipt; a short integration or
throughput pass does not authorize a formal five-epoch run. The largest passing
profile is a measured capacity boundary, not automatically the scientific
ablation batch. Unless explicitly changed after capacity/throughput review, use
the fixed default global batch 128 for comparable ablations.

Generate new ablation rows with `scripts/v2/generate_group.py --group GROUP
--output NEW_DIRECTORY`. Its CLI default parent is this `default` config. Pass
`--parent` explicitly for another dataset or a selected later parent, and verify
the generated manifest before launch. Each row still needs published source,
exact-row preflight, dataset-specific capacity evidence, and a new run ID. The
legacy `preregistered_jurkat` matrices and `prepare_dataset.py` integration
templates are historical and are not the default parent for the new architecture.

Cross-cell folds and the other four within-cell datasets should inherit this
**architecture and loss definition**, but must receive their own GenePT-PCA256
artifact, canonical hashes, data configs, and two-GPU capacity receipts. A Jurkat
capacity result cannot certify another gene axis or dataset.
