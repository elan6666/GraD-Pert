# GraD-Pert Context

Read `AGENTS.md` first. The authoritative state is `.byte-os/STATUS.md`, active
specs live in `.byte-os/` and `docs/design/`, and executable plans live in
`.byte-os/plans/`.

Hard boundaries:

- Native package is standalone `gradpert`; do not import/call upstream model
  repositories or introduce upstream-named native classes.
- v1 is B2 only on the five frozen datasets; no ablations or cross-cell work.
- Configs are self-contained model-by-dataset YAML files, never one global file.
- Formal compute is server-only and requires local/GitHub/server commit parity.
- Data, PKL/H5AD predictions, and checkpoints stay on the server.
- Never edit `TxPert/official-repo/**`; use it only as frozen evidence.

Local verification commands are listed in `.byte-os/HARNESS.md`.

