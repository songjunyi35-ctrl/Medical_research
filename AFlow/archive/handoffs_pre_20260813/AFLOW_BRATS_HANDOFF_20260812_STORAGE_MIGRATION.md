# AFLOW BraTS storage-migration handoff — 2026-08-12

This addendum records the completed storage cutover for the active T1ce/WT
mainline. It does not change the scientific protocol in
`AFLOW_BRATS_HANDOFF_20260807.md` or
`AFLOW_BRATS_HANDOFF_20260810_ADDENDUM.md`.

## Outcome

The active project no longer requires the useast1b `liveaim` NAS for raw T1ce
or segmentation input. The verified active copy is:

- HDFS URI:
  `hdfs://harunava/home/byte_data_tns/user/songjunyi/datasets/brats2024_gli_t1ce_seg_v1`
- expected worker mount:
  `/mnt/hdfs/sjy/datasets/brats2024_gli_t1ce_seg_v1`
- cases: 1350
- files: 1350 `*-t1c.nii` + 1350 `*-seg.nii` = 2700
- logical bytes: 98,091,449,088 (91.354781 GiB)
- three-replica HDFS storage: 294,274,347,264 bytes (about 274.1 GiB)

The former empty VA directory `/mnt/hdfs/sjy/AFlow_archive` was removed. No
NAS source file was deleted during migration.

## Integrity evidence

- Source and destination both contain exactly 2700 selected files and exactly
  98,091,449,088 logical bytes.
- All 2700 files passed a complete source-versus-destination byte comparison;
  the eight-way comparison completed in 225 seconds with zero mismatches.
- The destination contains 1350 case directories and no unexpected files.
- A real development case, `BraTS-GLI-00005-100`, was loaded from the new VA
  manifest with the unchanged mainline loader. It produced finite image and
  target tensors of shape `(1,128,128,128)` and 6663 WT voxels.
- The sealed-test NIfTI content was not opened by the manifest builder or the
  loader smoke test. The sealed-test authorization guard remains unchanged.

## Active manifests

New raw-data work must use
`data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/`:

| manifest | rows | SHA-256 |
|---|---:|---|
| `train_t1ce_wt.jsonl` | 972 | `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65` |
| `validation_t1ce_wt.jsonl` | 108 | `f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb` |
| `outer_train_t1ce.jsonl` | 1080 | `60f2f9134b19a070bbdb8f7a989e49eeb88d5c70023bcdd79ed9ebddda4ee430` |
| `sealed_test/test_cases_t1ce.jsonl` | 270 | `a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9` |
| `dpdsam_train_t1ce_binary.jsonl` | 872 | `b168b1bd74a20f78d9825b2c13174ab007d86f3b893695c8b6d83a1757bb9588` |
| `dpdsam_validation_t1ce_binary.jsonl` | 209 | `8df470753bbda1f5074f91999a51ae23a9873ac0009bf13ece975e3b7ab3459e` |

Machine-readable provenance is in `storage_manifest.json`, SHA-256
`f82cc9f062b47b16b0bc429e6d5e3ce82c6ab1c2af25c277f2520f38af981e3c`.
The deterministic builder is `scripts/build_t1ce_va_hdfs_manifests.py`,
SHA-256
`742b9383c5a493e88cc8004ecd2659ca2d0a204a8a9b94a3af5d7870f492f401`.

The original manifests were not modified. Their bytes and hashes remain valid
for completed experiments and the two existing local memmap caches. The new
outer manifests project `modalities` to T1ce only and replace every active data
path with the VA location.

The active `scripts/run_worker_research.sh` now supplies the VA train and
validation manifests by default for `sammed3d-t1ce-wt` and
`eval-sammed3d-t1ce-wt`. Its official-1.5-mm reproduction mode delegates to the
new storage-specific
`scripts/run_t1ce_wt_official15_reproduction_va_20260812.sh`. Older launchers
remain unchanged as historical records and must not be used for a new uncached
run after detaching NAS.

Storage-cutover tests passed: 64 targeted tests, shell syntax checks, manifest
path existence checks, and the real loader smoke. The repository-wide suite
also exposed an older independent freeze mismatch: the July v2 preflight binds
`sammed3d/t1ce_wt_training.py` SHA-256 `4d13...`, while the restored current
baseline file is `be2d...`. The storage migration does not rewrite that frozen
preflight or claim that historical arm is reproducible from the current source
tree.

## Worker requirement

Workers using these manifests must mount
`hdfs://harunava/home/byte_data_tns/user/songjunyi` at `/mnt/hdfs/sjy`.
Do not launch an uncached raw-data run until both the selected manifest and its
paths pass preflight on that worker. Cached train/validation and the residual
refiner remain self-contained on local project storage.

## NAS retirement boundary

It is safe for the active T1ce mainline to detach or omit the `liveaim` NAS
volume. That statement is narrower than permission to physically erase all NAS
content:

- the source still contains the retired `t1n` (48.475036 GiB), `t2f`
  (48.475036 GiB), and `t2w` (48.179131 GiB) modalities; they were deliberately
  not migrated;
- `/mnt/bn/liveaim/songjunyi/tns-experiments` is a separate approximately
  76 MiB repository and is unrelated to AFLOW BraTS;
- physically deleting those assets is irreversible and is not required to use
  VA-region GPU workers.

Therefore: detach NAS from the active project/worker configuration now if
desired. Permanently deleting the remaining NAS data requires a separate,
explicit decision accepting loss of the retired modalities and the independent
repository.

## Official-training semantic alignment — 2026-08-12

The first official15 run only aligned preprocessing and still used the frozen project
three-click trainer. It was stopped after completed epoch 61 and sealed with
`PROTOCOL_AUDIT_ABORTED.json`; its best development Dice was 0.5565408558 at
epoch 17. Its artifacts were preserved, but it is not evidence for the upstream
SAM-Med3D training recipe.

A source-audited replacement is now active:

- runner: `scripts/train_sammed3d_t1ce_wt_upstream_literal_va.py`
  (`fda8a458cc692289092038a64469aa4538d37e3750d13cc2737f537f088b7d19`);
- launcher: `scripts/run_t1ce_wt_upstream_literal_va_a100_20260812.sh`
  (`a9e4e3cc33b068ffd4e82cc2db626b381663f12d27bcce748776aabd7e824af2`);
- targeted test: `tests/test_upstream_literal_training.py`
  (`f94085b90159217a91138ab7935277dcf821bea0d6f88415fd24895992835cd4`);
- combined official cache/training regression: 29 passed.

The pinned upstream sources are `train.py` SHA-256 `474d9765...e1e3f`,
`inference.py` `5a31aff3...960`, and `utils/click_method.py`
`0838972f...92d7`. The audited official `train.sh` enables two GPUs. The formal
configuration therefore retains per-GPU batch 12, accumulation 20, FP16
autocast plus GradScaler, encoder/head LR 8e-4/8e-5, AdamW weight decay 0.1,
200 epochs, MultiStepLR milestones 120/180, seed 2023+rank, 12 workers/rank,
and the recommended untouched Turbo checkpoint SHA-256 `899a46d0...dd3c`.

The executed interaction is the upstream hard-coded 11-pass loop: current point
only, recurrent low-resolution mask, one random no-point pass in indices 2..8,
final no-point pass at index 10, raw-logit 0.5 error sampling/train Dice, and sum
of all 11 DiceCE losses. Cases with at most 1000 foreground voxels follow the
upstream recursive replacement behavior after RandomFlip.

Two explicit engineering corrections are recorded rather than hidden: gradients
are mean-all-reduced because the upstream DDP wrapper is bypassed by direct
submodule calls, and binary integer GT is cast to float only at the DiceCELoss
boundary for current MONAI compatibility. Validation is an RNG-isolated
official-style five-click diagnostic (RITM for clicks 1-2, uniform XOR for
clicks 3-5) reporting both final click and per-case best-of-five; stable per-case
seeding is an added monitoring extension. Upstream train.py itself has no
validation loop and selects by train loss/train Dice, so those primary
checkpoints are also retained.

The final 504-case formal-shape smoke completed with one applied and zero skipped
optimizer updates, GradScaler 65536, 67,628,254,208 bytes peak allocated per
GPU, 21.25 cases/s, strict bitwise checkpoint reloads, and
`scientific_result_eligible=false`.

The formal run is active on worker 1042986 (2 x A100-SXM-80GB):

- output: `workspace/BraTS/sammed3d_t1ce_wt/sammed3d_upstreamliteral11_officialval5_va_a100x2_train972_val108_e200_s2023`;
- log: the same basename with `.launch.log`;
- epoch 1: train Dice 0.33384, val best-of-5 0.30392, one of two FP16 updates
  skipped while the scale adapted from 65536 to 32768;
- epoch 2: train Dice 0.36904, val best-of-5 0.49236, final-click-5 0.47616,
  both optimizer updates applied.
- epoch 3: train Dice 0.48774, val best-of-5 0.50553, final-click-5 0.47967;
- epoch 4: train Dice 0.49223, val best-of-5 0.50959, final-click-5 0.47146;
  all four optimizer updates across epochs 3-4 were applied with zero skips.

The sealed test split was not read or evaluated.

## Baseline completion and authorized test — 2026-08-12/13

The statement immediately above records the state at formal-run launch and is
now superseded by this completion record. The upstream-literal run completed
all 200 epochs without a protocol change. `TRAIN_DONE.json` declares the result
scientifically eligible; strict reload records are bitwise identical.

The selected development checkpoint is the validation-best epoch 21 model:

- validation mean per-case best-of-five Dice: `0.5713061351995669`;
- validation final-click-five Dice: `0.5591851002518674`;
- checkpoint:
  `workspace/BraTS/sammed3d_t1ce_wt/sammed3d_upstreamliteral11_officialval5_va_a100x2_train972_val108_e200_s2023/sam_model_val_official_best5.pth`;
- checkpoint SHA-256:
  `c0c85a0fb90f6301344053f97cabe8a4972bd9febd4c6ae248f3a0908598fe31`.

The user then explicitly authorized the frozen 270-case test. Before any test
NIfTI was opened, the checkpoint, manifest, preprocessing implementation,
five-click prompt protocol, seeds, metric aggregation, and code hashes were
written to `TEST270_FREEZE_20260812.json`. The test contains 270 scans / 119
subjects and has zero case and subject overlap with the 972/108 development
split. Its official-1.5-mm cache reused the exact development voxel transform;
all 270 cases retained 100% of WT foreground after crop.

Two A100 ranks evaluated 135 cases each. The complete frozen result is:

| Metric | 270-scan macro | 119-subject balanced |
|---|---:|---:|
| per-case best over clicks 1-5 | **0.5942261964** | **0.5809701177** |
| final click 5 | **0.5857539438** | **0.5726994265** |

Scan-macro click-1 through click-5 Dice is
`0.5489002342 / 0.5669396817 / 0.5737304338 / 0.5790306286 / 0.5857539438`.
All 270 case records are unique, finite, in frozen-manifest order, and the
summary was independently recomputed after report creation.

The complete report is:
`workspace/BraTS/sammed3d_t1ce_wt/sammed3d_upstreamliteral11_officialval5_va_a100x2_train972_val108_e200_s2023/test270_epoch21_official15_upstreamliteral5_s2023_20260812/test_report.json`,
SHA-256
`7bd59843459876bd006230b24bc111cdfd726896cb3e6748b54b24c0a9659b63`.

Research interpretation is now frozen: this completes and establishes the
senior-line vanilla SAM-Med3D baseline under the project's explicit
subject-clean protocol. Its `0.5942` result is close to MSCL Table 2's `0.601`
vanilla baseline. It must not be called an exact replication of undisclosed
paper split/prompt details, and MSCL Table 1's `0.731` threshold-loss-enhanced
configuration is not the vanilla target.

The 270-case set is revealed and must not be used to tune the next method.
Future work should preserve this baseline, develop/select only on 972/108, and
freeze any future comparison before another test evaluation. Because this test
also appeared in earlier project arms, a pristine external/confirmatory set is
required for a strong untouched-test claim.
