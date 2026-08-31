# Four-local RingInduced index performance review

## Verdict

Pass for implementation delivery. The immutable incoming-edge index is an
exact-effect systems optimization at the four-local A0 coordinate. It is not
scientific ablation evidence and does not authorize L execution.

## Measured selection

The bounded real Python profile attributed 5.723 seconds to
`build_training_graph_views` and 5.173 seconds to 192 RingInduced builds across
six steps. Repeated base-edge/self-loop preparation consumed 3.039 seconds and
incident-node scans 1.840 seconds. This exceeded the preregistered 10% and
100-ms materiality boundary.

The selected implementation builds one immutable, source-aware incoming-edge
index per topology. The runtime visits only selected targets and derives
incident nodes in the same ordered pass. Graph scale, local ratio/count,
anchors, edges/weights, self-loop insertion, warnings, view order, model,
precision, loss and evaluation are unchanged.

## Evidence

- Real 2,809-node, eight-condition, four-local microbenchmark: reference
  983.661 ms median, indexed 833.647 ms, 15.251% and 150.014 ms lower, exact
  complete local-view SHA equality. Evidence SHA
  `f02277663cff420f01917d605ee97cfc0346ae40b54df08aa2a91edaa88fe60e`.
- Deterministic CUDA gate: six reference and six indexed steps; exact views,
  all non-timing metrics, CPU/CUDA RNG, losses, every gradient, Student,
  Teacher, optimizer, centers and predictions. Zero retry/OOM, zero PKL and no
  validation/test access. Evidence SHA
  `2b4f241aba137cdd138f576ca20a27b1f954f2b59dcbf874ed2a53b8f177f088`.
- Serial same-GPU ABBA: A1 reference, B1 indexed, B2 indexed, A2 reference;
  two warmups and ten measured steps per arm. Reference p50 was
  2,964.568/3,018.291 ms; indexed p50 was 2,554.524/2,523.254 ms. Paired ratio
  0.848836, 15.116% and 452.540 ms lower wall. Both p90 pairs improved,
  allocated/reserved memory was identical, minimum free memory was 22.812 GB,
  and every arm retained zero retry/OOM/PKL and no truth access. Evidence SHA
  `7e82fe419da63a2d85b8786af6751acafbc0ddfd1caf2bc1488cba7cda1ea579`.

The earlier checkpoint-count-two candidate remains rejected: it passed
capacity and deterministic equality but its paired ABBA ratio was 1.026487,
2.65% slower than count four.

## Delivery conditions

- Full local and exact-commit server pytest, Ruff, format, strict mypy and
  isolated build gates must pass on the reviewed source commit.
- A0/H1/H2/H3 may run at row level on both physical GPUs after a new clean
  local/GitHub/server identity is published. Every row remains single-GPU and
  H3 requires an explicit capacity gate.
- L remains paused.
- Trackio is formal-only. Private Bucket authentication is ready, but the
  private Space currently returns `402 Payment Required`; no public fallback
  is authorized.
