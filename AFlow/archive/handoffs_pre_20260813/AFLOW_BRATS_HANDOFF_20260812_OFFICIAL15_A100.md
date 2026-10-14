# AFlow BraTS T1ce/WT official15 A100 continuation handoff

Status timestamp: 2026-08-12 06:40 UTC.

## Correction to the active research line

The official SAM-Med3D standard-preprocessing reproduction had previously only
been implemented; it had not been formally executed.  The prompt-local refiner
completed earlier on 2026-08-12 is a separate native-GT-crop/N4 development
experiment and must not be described as the official15 reproduction.

The active formal arm is now exactly:

1. paired T1ce/WT resample to 1.5 mm isotropic (linear image, nearest LabelMap);
2. TorchIO `ToCanonical`;
3. GT-label-centred `CropOrPad` to 128^3;
4. post-crop positive-image-voxel sample z-score applied to the whole patch;
5. no N4;
6. train-only `RandomFlip(axes=(0,1,2))`; no validation augmentation.

The task remains frozen: T1ce only, WT={1,2,3}, raw label 4 excluded, 972/108
subject-clean development split, three accumulated corrective clicks, recurrent
low-resolution mask, and raw-logit > 0.  The sealed 270-case test was not read.

## Why a new entry point was required

The historical official15 VA launcher was single-GPU and the byte-pinned base
trainer hard-coded the pre-migration manifest hashes.  In addition,
`T1ceWtPreprocessConfig.to_protocol_dict()` has a legacy `gt_crop + random_flip`
serialization bug (uninitialized geometry/normalization followed by flip being
overwritten as no augmentation).  This confirms the old formal path had not run.

Historical source files were left unchanged.  A fail-closed wrapper now reuses
the frozen optimization loop while requiring official15 cache provenance and
explicitly serializing the correct runtime protocol.

## Frozen code and inputs

- Builder: `scripts/build_t1ce_wt_official15_cache.py`
  - SHA-256 `ee882e0eed4704b398568dd2a36683ac1312cb429c5be85de1636816599114c9`
- Reused historical loop: `scripts/train_sammed3d_t1ce_wt_dpdsam_aligned.py`
  - SHA-256 `31c114fe24e6ad419c3eac29a50775a5ab9d7dedf758ae907c3d7f417829c20c`
- Official VA wrapper: `scripts/train_sammed3d_t1ce_wt_official15_va.py`
  - SHA-256 `4320b54f7b7e4d454a6a7eb44ce74a33554f3d6ecde43a7e8e5527cccb04421a`
- 2xA100 launcher: `scripts/run_t1ce_wt_official15_reproduction_va_a100_20260812.sh`
  - SHA-256 `53f755d01cc5b11c32d1902331e66cddb0d2336c7a438a50c63d167914d85eba`
- Train manifest SHA-256: `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`
- Validation manifest SHA-256: `f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb`
- Base checkpoint SHA-256: `899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`
- Preregistration:
  `workspace/BraTS/sammed3d_t1ce_wt/OFFICIAL15_VA_A100X2_PREREG_20260812.json`

## Preflight evidence

- Focused synthetic/runtime suite: 25 passed, 0 failed.
- Real train case `BraTS-GLI-00005-100` independently recreated the public
  TorchIO chain:
  - image max absolute difference: 0;
  - label voxelwise equality: true;
  - normalized positive mean: 8.31e-8;
  - normalized positive sample std: 1.0;
  - crop retention: 1.0.
- Real-case record:
  `workspace/BraTS/sammed3d_t1ce_wt/OFFICIAL15_REAL_TRAIN_CASE_PREFLIGHT_20260812.json`
- Full caches under `/tmp/aflow_t1ce_wt_official15_va_128`:
  - train images SHA `b2b9b2adaedd92eb85977ad45b20de359d9c2de1d9ba4f285fdc663efadd70bf`;
  - train labels SHA `2d6c25755de8639e107257f0dd339de3535b00a1b405f0ffc4816c1304ed5b21`;
  - validation images SHA `d29cd598020d5f1a6e121e7937ab1524c81b3bdd29fe7d480e26ecd6400056ae`;
  - validation labels SHA `e43fd0c3975379b6849c601d3207c5def31bc3076a968040cc17d71675cb261e`.
- Cache QC: 972 train and 108 validation; target spacing is only 1.5 mm;
  minimum/mean WT crop retention both 1.0; zero cases with crop loss.
- Two-GPU preflight passed with NCCL over NVLink.
- Two-GPU smoke passed forward/backward, finite metrics, checkpoint save, and
  strict bitwise reload; smoke selected val Dice 0.333824 (not scientific).

## Formal run now active

Worker: `1042986`, 2 x NVIDIA A100-SXM4-80GB.

Run identity:
`sammed3d_official15_va_gtcrop_fullunfreeze_a100x2_train972_val108_e200_trainseed20260805_r20260812`

Output:
`workspace/BraTS/sammed3d_t1ce_wt/sammed3d_official15_va_gtcrop_fullunfreeze_a100x2_train972_val108_e200_trainseed20260805_r20260812`

Persistent log:
`workspace/BraTS/logs/official15_va_a100x2_formal_20260812.log`

Frozen optimization: 2 GPUs, per-GPU batch 6, accumulation 1, effective batch
12, BF16, full unfreeze, AdamW encoder/head LR 1e-4/1e-5, weight decay 0.1,
MultiStepLR at 120/180 gamma 0.1, exactly 200 epochs, no early stop, train seed
20260805.

Observed through epoch 3:

| epoch | train Dice | validation Dice | train cases/s |
|---:|---:|---:|---:|
| 1 | 0.509432 | 0.507483 | 20.65 |
| 2 | 0.558172 | 0.528048 | 21.61 |
| 3 | 0.584289 | 0.534129 | 21.01 |

Peak allocated memory is 35,862,472,704 bytes per rank; observed board usage is
about 38.8 GiB per GPU.  Estimated completion at current throughput is roughly
three hours after the 06:37 UTC launch.

## Required continuation

1. Monitor `history.csv` and the persistent log; do not alter the frozen run.
2. Require `TRAIN_DONE.json` and `selected_checkpoint_reload.json`; the latter
   must report strict load and bitwise identity.
3. Summarize all 200 complete epochs, selected validation Dice/epoch,
   throughput, peak memory, cache QC, and checkpoint SHA.
4. Keep all interpretation development-only.  Do not open the sealed test
   without a new explicit operator authorization after reviewing the result.
