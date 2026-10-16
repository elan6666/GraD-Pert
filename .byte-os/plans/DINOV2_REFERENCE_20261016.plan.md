# 官方 DINOv2 参考分析

User moved execution from supervisor to main on 2026-10-16. This is a separate
main-owned analysis task; the D0 relay stays supervisor-owned and must not be
overwritten. Runtime state and all local deliverables live in the ignored
`/Users/elan/code/grad-pert/experiment-analysis/dinov2-reference/` directory.

- Official upstream frozen at 7764ea0f912e53c92e82eb78a2a1631e92725fc8,
  ViT-S/14 backbone weight SHA b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9.
- Inspect strict-load state and official forward/intermediate behavior; no
  upstream model reimplementation or native modeling import.
- Imagenette320 validation,1335 hash-selected class-balanced images;
  nested506 deep images×128patches; geometry-only 2G+2L probes.
- Same covariance effective-rank/relative-variance definitions as medium
  GraD-Pert; distinguish CLS, mixed patches, per-image patches, same-image
  views, within-class and between-class pairs, actual N and width ceiling.
- Pretrained backbone lacks original SSL heads/center/teacher: no prototype
  entropy,JS,CE/KL or training gradient claims. No downstream head substitute.
- CPU-only, explicit CUDA_VISIBLE_DEVICES='',max4threads, no GPU overlap with
  D0. Zero optimizer updates; strict state hash equality. Images/weights/raw
  features remain server-only. Publish clean source before execution, new
  output identity; preserve partial download and prior audit.
- Verify sampling/cosine/statistics/crop invariants, official accessor parity,
  server small real-input probe and measured CPU timing, then full medium.
- Chinese report lists every plot/table and how to read them. Do not treat
  image-class values as biological quality thresholds or independent patches
  as independent images. Only aggregate outputs transfer after dry review.

## Accepted result

Completed main-owned medium on bcb78ddcbe5e581966c35736ab7d2275ef902073:1335 images/10classes/deep506×128 patches;15 server tests, official accessor maxabs0, strict weights, CPU4 threads/105.657s,0updates/state unchanged. All15 server output hashes verified;14 small outputs plus receipt transferred and3figures opened. COMPLETE SHA eec44aac3195314dea19e76b5cb0831c6b675eeac62ff02eb0d271f112bed522. Full CN review and all figures/CSV/JSON are in ignored permanent archive dinov2-reference/REPORT_REVIEW_CN.md. Original SSL prototype targets unavailable and labelled accordingly. D0 background monitoring remains independent and active; no new monitor or formal ablation dispatched.
