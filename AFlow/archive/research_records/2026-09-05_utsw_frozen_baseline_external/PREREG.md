# UTSW362 frozen-baseline cross-stage external preregistration

Frozen before download of any UTSW subject outside BT0001/2/3 and before GPU loading either checkpoint on 2026-09-05.

## Frozen cohort/source

- TCIA UTSW-Glioma package1265; CC BY4.0; preoperative/SRI24.
- Metadata SHA `f7ff889fa7d197b2f589695d3e7ea8a2d1d198c4f9c0044fc71898a08aa24ed0`; inventory SHA `5f210c0a1e0c33f7b71a56d9bbce8b52a62f24e2a0b53655aa3a098dbde3f871`.
- Every and only362 `Manually Refined Segmentation=Yes` IDs; sorted ID SHA `e4809dd7f43439309bef71784b3accff95be1a6915e044f46de352e780ec525c`.
- Per case primary `brain_t1ce.nii.gz` plus full-grid `rtumorseg_manual_correction.nii.gz`;724 files;1,813,086,599 bytes/copy.
- Two fresh authorizations/single `--src-base` transfers; all relative paths, official size and copy A/B SHA exact before NIfTI load.

## Frozen models

1. Full972 random-init voxel-KD epoch168:
   - path `workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth`
   - file SHA `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`
   - state SHA `f24201eecc7bf4aaff92b8e0a50c85a31e6913f5c9701cca6ad7df3439618ea7`; schema `aflow.sammed3d.t1ce_wt_voxel_kd_fulltrajectory_final_checkpoint`; epoch168.
   - Prior access: fixed validation and user-authorized failed-gate reused test270, then mirror-uint8 UCSD75; weights unchanged. Primary UTSW view=Flip8.
2. GCSAM fit864 epoch184:
   - path `workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth`
   - file SHA `acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`
   - state SHA `eea33e1d3a1ab9505ba0b85aa915a9df59260b50a0991b49e1e9c77417690f02`; schema `aflow.sammed3d.t1ce_wt_gcsam_inner.checkpoint`; epoch184.
   - Prior access: train-inner selection, user-authorized reused test270, mirror-uint8 UCSD75; later fixed184 replication failed. Primary UTSW view=identity.
- Both use `AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=0.1)`,4,700,897 parameters. Training-source official archive parity remains unresolved.

## Frozen access order

1. Require ≥3h before deadline2026-09-06T14:02:39Z and ≥12GiB free `/tmp`; freeze preaccess with all source/model/code/protocol hashes and old-access flags.
2. Synthetic-only GPU smoke both models, batch2, identity+Flip8, finite/shape/strict load/24GiB gates; no UTSW payload.
3. Dual-download source;724/724 exact; no voxel access before receipt.
4. Build one canonical-N4 cache using copy A, SimpleITK thread1 per worker; all-positive label→WT; RAS; image-label affine≤1e-6; N4 positive mask/shrink2/fixed iterations; brain bbox/z-score/128³. Require all362 finite/binary/nonempty, exact bytes; fixed BT0001/2/3 output hashes equal canonical bridge QC `1cec4ba9...`/COPY_A `a1776d09...`.
5. On one idle GPU evaluate KD then GCSAM in fixed order; each separately runs identity1 plus Flip8 8 (identity is intentionally recomputed), BF16 autocast, batch8, threshold`>0`, point=argmax, no SAM/postprocessing. Total6516 single-example forwards.

## Frozen outputs/statistics

- Per model/view: Scan/Subject Dice (equal because one scan/person), Point hits/rate, prediction/target volume; per-case Dice/Point/mask SHA.
- 10,000 subject bootstrap, seed20261391: absolute CIs and same-view paired GCSAM−KD Dice/Point-rate deltas. Paired differences are descriptive, not causal/promotion evidence.
- Primary absolute views: KD Flip8; GCSAM identity. Reference gate per primary: Scan≥0.70, Subject≥0.70, Point≥326/362. Report all models/views regardless.
- One evaluation only. No case/model/view/threshold/epoch/preprocessing/label/bootstrap change and no downstream tuning/retest.
- On completion update archive, handoff, method catalog and leaderboard external cross-stage section; keep separate from post-treatment/UCSD/internal rankings and label grid/provenance caveats.

