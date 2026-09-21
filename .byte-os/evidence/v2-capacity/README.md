# Effective batch32 execution comparison

These are completed engineering probes on Nadig Jurkat, using clean published
training source `85f0f0867eba4c4d244cf912e721dd4770c7c226`. Both execute the full
prediction + SSL1 + SSL2 objective for128 updates, checkpoint continuation, and
one validation condition with300 controls ×5000 output genes. They are not
50-epoch results or complete validation-set evaluation.

| Execution | GPU | Effective batch | Measured cells/s | End-to-end training cells/s | Peak allocated bytes |
| --- | --- | --- | --- | --- | --- |
| micro32 × accumulation1 | 0 | 32 | 4.772242 | 3.927099 | 15080300544 |
| micro16 × accumulation2 | 1 | 32 | 2.490249 | 2.237420 | 8130809856 |

The32-row execution is faster in these observations but uses more memory.
The observations came from different cards while both probes ran concurrently;
this is not a repeated controlled hardware comparison. KoLeo uses physical
microbatch neighbors, so equal effective batch does not establish objective
or gradient equivalence. Neither row establishes the maximum feasible batch.
Batch64 testing and final execution-profile selection remain pending.

The adjacent JSON files are original small server receipts, transferred only
after an rsync dry run. No checkpoints or prediction matrices were transferred.
They include exact source/config/data/ordered-axis hashes, environment identity,
per-update timing, memory and checkpoint/inference evidence. Verify with
`scripts/v2/capacity_report.py` using the matching committed capacity configs.

Receipt SHA256:
- m32: `3e237ba13df260e4bfe221869a6069e11cb82bf9360253fb92a40a270b2f153d`
- m16accum2: `6870fc2810a1348183f2509a616b9bac7e387f65544e0c67c941493f4265f53b`
