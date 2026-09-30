# Jurkat cap40 mHC B0 six-epoch supervision

From main `codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b` build to existing supervisor
`codex:01a0df0b-4142-7df1-86c0-d959471d80a1`, host local. Project checkout:
`/Users/elan/.codex/worktrees/v2-gene-exposure-eval/grad-pert`.
Exact run `nadig_jurkat-seed1-20260930T112531Z-f7ca7a77fa6848bb811f706d00e5daad`, attempt1; this is fresh training, not continuation.
Launch receipt: `docs/experiments/data/v2-mhc-cap40-aeac5fd/v2-mhc-cap40-aeac5fd-launch.json`.
Root `/data/yilangliu/GraD-Pert/runs-v2-mhc-cap40-six-aeac5fd/nadig_jurkat-seed1-20260930T112531Z-f7ca7a77fa6848bb811f706d00e5daad`; test root is the same root plus `-test`.
Source `/data/yilangliu/GraD-Pert/development/source-v2-mhc-cap40-aeac5fd`, training SHA `aeac5fd94123af0b73810259e5e2985228b11d65`.
Config `configs/v2/mhc_cap40_jurkat/six_epoch_m68_a2/gradpert_v2/nadig_jurkat.yaml`,
SHA `1401174171770bb5c7ae580db55f9dd9d4907d8da6a37a42b86a64d747f412d0`. RuntimeSHA `19b87960e43dcd36bda67a87645c9b205e1ec55a4bf6b2817407034534e7b89c`;
publicationSHA `095782e81d633c3336ab7b2823d95f65e0be67fef82ab2ec569376bfaeb2b213`; planSHA `0cebf9e4d80e934069b58d3fbc3d9b878901d0c236f950dfab204253db5c86e3`.
Frozen selection SHA aa0916dfb91fff78d7a43f7d7325879a11c1a904e9bf31586420ca11cce84132.
Full prediction+SSL1+SSL2, mHC streams4 Student/Teacher, seed1, GPU0,1;
physical micro68 x accum2 = global272, exactly6epochs x176updates.
Only training perturbations cap40 (47836rows/1335conditions); original frozen
validation/test/control/Systema references unchanged. Epoch validation joint_only,
minimum val joint selects best; last is epoch6; terminal tests retain frozen300controls,
all/DEG Pearson and expression exposure groups. No other ablation is authorized.

95local/server tests and real dual-GPU one-update/checkpoint-reload passed.
Integration receipt SHA07bdc51e51670db1b66fecceab632bc7e6d5563c00cc5fb95cd0be926b0412dc.
Formal start confirmation SHA3cac41014536c9058eadefb182b707b0fe028b053aa934c5e9e98b3fd7d2369a:
epoch1 step2/176; wrapper225555, torchrun225569, ranks225585/225586 alive,
GPUrank memory28978/28838MiB, no exit/failure. Luna short verification is finished.
Log `/data/yilangliu/GraD-Pert/development/v2-mhc-cap40-aeac5fd-six-formal-r1.log`; PIDfile same stem.pid, exit same stem.exit.json.
Check fit/live_progress.json, epoch_state.json, history, processes/GPUownership,
exit/COMPLETE/FAILURE, best/last test receipts and recursive zeroPKL. PID loss is not success.

Monitor grad-pert-v2: reactivate for this exact run/attempt at20minutes, target
supervisor. Save verified ACTIVE acknowledgement/lease receipt before main ends.
Notify meaningful changes/failure/completion; routine healthy checks quiet.
No tested immediate event bridge; polling is the verified failure detection path.
On terminal success/failure, save evidence, pause monitor and verify PAUSED, return
baton to exact main with main_continuation=pending and send actionable next step.
Main-target grad-pert-v2-b0-return is currently PAUSED; if substantive main reentry
is absent, use the previously authorized run-scoped main heartbeat fallback,
retargeted to this run, then main must pause it on acknowledgement. Do not promise
an untested event wake. Supervisor model provenance remains unverified.
Routine VPN recovery allowed; source/scientific repairs return to main. Never edit
active checkout, overwrite IDs, blindly restart, expand compute or start other groups.
Acceptance: exit0,6/6/history6,min-joint best,epoch6last,both real test receipts,
consistent source/config/data/checkpoint/test hashes, no FAILURE and zeroPKL.
Launch is verified; overall experiment is unfinished. Next owner monitors this run.
