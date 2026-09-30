# Jurkat cap40 mHC B0: six fresh epochs

The user selected the existing fixed cap40 sample and enabled mHC. This new
self-contained configuration is
`configs/v2/mhc_cap40_jurkat/six_epoch_m68_a2/gradpert_v2/nadig_jurkat.yaml`.
It inherits no config: all settings are written explicitly. The old three-epoch
mHC config and historical no-mHC continuation remain unchanged.

## Data and lineage

Only training perturbation rows are selected:128266 ->47836, all1335 train
conditions retained, each at most40. This is the accepted seed42 proportional
batch-quota sample, fixed across all epochs; it is not resampled each epoch.
Control, validation, test and excluded rows remain canonical and unchanged.
No expression values, graph axes, test-target expression policy or Systema
train+validation reference are changed by the row selector. An optional v2-only
manifest restricts the existing train_row_indices; no derivative H5AD is
relabeled canonical-ready and v1 uses the original pipeline.

The analysis canonical H5AD65b32637b24e6ee6d3399b3280d914dcedc34bf787098c3e9e14a47ebe80cbb5
and training canonical f051343c191dcdb02cabfae66a6bfe10b2770752b6503802da33d4b74e845861
have identical238977x6506 float32 X (full512-row backed comparison, zero
value differences), and exact obs/var including categories. Their uns
preprocessing records differ, including graph_only_candidate_targets; these
provenance differences are preserved rather than declaring full-file equality.
The training graph remains the existing frozen v2 graph.

Server binding directory:
`/data/yilangliu/GraD-Pert/development/jurkat-cap40-v2-binding-20260930`.
Equivalence receipt SHA256:ca12b16967f77a06d96139ffaa23cd997af13aa1b2acbe7250ed1d8696b04f85.
Training selection SHA256:aa0916dfb91fff78d7a43f7d7325879a11c1a904e9bf31586420ca11cce84132.
Selected ordered row-ID hash:c8e79242336c194833ac329a807b7eeb76651707204b4ec1379f6a7483aa9a93.
Original accepted analysis selection SHA256:a609d5aa4087d209dc62c1e8ed70dc7cff89ed6952b2afb18d0972cb477c94f2;
analysis SHA5a8a7bb6712bc4facdfeb13d8dd629bd0f250880. Long row-ID lists stay on server.

## Training and evaluation

Student and EMA Teacher use four mHC streams; full prediction+SSL1+SSL2.
SSL stage multipliers are1; internal weights remain0.8,0.4,0.1 in each stage.
Single-pass KDA writes once and every query reads final S; graph source key
gates/random order/view and global KoLeo semantics are unchanged. Width256,
four heads, graph3KDA+1MLA, Cell/Response2KDA+1MLA; no DSA.

GPU0,1: each physical microbatch68, accumulation2, globalbatch272, seed1.
Graph scan32, target rows64, sequence scan16. m68 has earlier three-epoch mHC
success evidence; new-source one-update integration is required before launch.
Six fresh epochs use the project warmup/cosine schedule over this selected-row
budget. This is not a 3->6 continuation and does not initialize from old weights.
Validation runs only fixed-view joint loss and components; minimum joint loss
selects best.pt. After six epochs best and last automatically run the frozen
300-control evaluation, three Pearson definitions each all/DEG plus the existing
expression-exposure groups. Zero-PKL protocol applies to the whole run root.

## Implementation and acceptance

V2Options exposes optional paired train_selection_path/train_selection_sha256.
Absent fields preserve historical config serialization. Before model allocation,
loader verifies the file hash, canonical/split/axes, unique train-only ordered
IDs, exact min(cap,Np) quotas and all conditions, non-training row-axis hash,
and unchanged samples from the hash-pinned accepted analysis plus equivalence
proof. Only then it replaces train_row_indices. Receipt in runtime identity
seals the selection for training checkpoints and standalone evaluation.
Synthetic schedule tests cover all six epochs' fixed IDs, unchanged validation
and control pools, leakage, duplicates, bad hashes/parent/split/order/quota and
atomic failure. Execution still uses existing runtime/runner/evaluator.

Local verification:95 affected config/runtime/selection/joint-validation/data-contract tests pass; scoped Ruff check/format pass and config/selection typechecking pass. Formal server preflight/launch are next; not yet claimed complete.
