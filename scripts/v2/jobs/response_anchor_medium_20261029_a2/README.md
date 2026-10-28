# R14 medium/crossed submission, attempt 2

This versioned job wrapper runs two retained epoch-2 checkpoints through the
unchanged published analysis source `7ae8824c7e8c65bb34c89df302f57999cbf38cef`.
It is submission orchestration, not a new model or diagnostic implementation.

R3-RI uses physical GPU 0; R14-FRGA uses physical GPU 1. Each endpoint runs
`seal`, `medium`, `panel`, then `validate`. Source, checkpoint, metadata, owner,
allocator, population, resource, parity and whole-job zero-PKL checks remain
fixed. Logs stay outside the native scientific output closure.

Attempt 1 stopped before any native phase: the metadata sealer exited zero,
but the controller observed no child process identity. Attempt 2 retries only
missing identity observations while that child remains alive, for at most two
seconds. A nonempty mismatch, lookup error or exit without verified identity
still fails. Every observation is retained. This addresses a possible process
startup race; its precise cause was not reproduced by the bounded Linux probe.

The controller and sealer are copied byte-for-byte to the fixed fresh job
artifact paths in their constants. Record this wrapper commit separately from
analysis and both checkpoint training commits. Verify their tracked blob hashes
before launch. Preserve prior attempts; do not overwrite or resume this job ID.
The synthetic subprocess tests exercise orchestration only, not GPU correctness.
