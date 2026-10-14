# Preregistration - sammed3d_braincrop_fullunfreeze_ddp2_train972_val108_e200_s20260806

Created (UTC): 2026-08-06

## 1. Hypothesis

Inserting crop-to-nonzero before the 128^3 resize raises validation DSC relative to the whole-head-resize arm, because it roughly doubles the lesion's voxel count on the model grid and stops ~52% of the input (and ~267 of the image encoder's 512 tokens) from covering background.

## 2. Prior evidence

- Source: `workspace/BraTS/sammed3d_t1ce_wt/BOTTLENECK_DIAGNOSIS_20260806.json`
- Probe design: same whole-head-trained weights, geometry swapped at inference only
- epoch 64 delta (native 1mm): **+0.0274**, 95% CI [+0.0158, +0.0394], sign-test p = 0.00068
- epoch 24 delta: +0.0040 (NOT significant)
- Caveat: the probe fed brain-cropped input to whole-head-trained weights, so its delta is a LOWER bound on the trained effect, not an estimate

## 3. The single factor

**Changed:** spatial preprocessing only: whole_resize -> braincrop_resize

**Held identical to the reference arm:**

- base checkpoint and its SHA-256
- full image-encoder unfreeze, 100,509,232 trainable parameters
- AdamW, encoder LR 1e-4 / head LR 1e-5, weight decay 0.1
- per-GPU batch 2 x 2 GPUs x 3 accumulation steps = effective batch 12
- MultiStepLR milestones [120, 180], gamma 0.1
- 200 epoch cap, min 40 epochs before stop, patience 40
- bf16, explicit gradient all-reduce across 2 ranks
- train seed 20260805, validation seed 20260723
- affine + elastic + noise augmentation, train split only
- DiceCELoss(sigmoid=True, squared_pred=True)
- 3 accumulated corrective clicks with the recurrent low-res mask, raw logit > 0
- N4 bias field correction, shrink 2, 4x50 iterations, nonzero mask
- nonzero-population z-score
- the frozen 972 train / 108 validation subject-clean split

Unlike the reference arm -- which moved unfreeze policy, LR direction, N4, epochs, batch and DDP together and was therefore explicitly marked non-attributable -- this arm changes exactly one factor, so its delta against the reference arm IS a controlled contrast.

## 4. Geometry

- Name: `braincrop_resize`
- crop to the bounding box of nonzero voxels, then resize that box to 128^3. Does NOT use the label: the box comes from image intensity only, so localization is not oracle.
- Uses GT for localization: **False**
- Normalization: source nonzero-voxel population z-score before resize; unchanged by the crop because the crop removes only zero voxels
- Builder: `scripts/build_t1ce_wt_braincrop_cache.py` (SHA-256 `fcb1428300be5b376a82cb4413590d37204ac0b84ed2904ae891aa43e3b217c3`)

## 5. Data binding

| split | count | manifest SHA-256 | images SHA-256 | test_access |
|---|---|---|---|---|
| train | 972 | `ea21e2306321b71d...` | `f772520ca0d99fca...` | False |
| validation | 108 | `fc750ab15266b8a5...` | `19107e84aba65c5c...` | False |

## 6. Selection rule

- Metric: `val_dice`
- Rule: strict argmax over completed epochs; ties keep the earlier epoch
- Test access during selection: **False**

## 7. Test protocol

- Runs allowed: **1**
- Precondition: a frozen configuration record must be on disk BEFORE any test NIfTI is opened
- Evaluator: `scripts/evaluate_sammed3d_t1ce_wt_braincrop.py`
- Reference arm on the same 270 cases: scan C3 = 0.571109, subject C3 = 0.553484
- This will be the 6th consumption of the 270-case test by this project. The number is an internal held-out readout, not clean external validation.

## 8. Prohibitions still in force

- no threshold / TTA / morphology / connected-component / postprocessing search
- no mid-run change to LR, unfreeze depth, initialization, augmentation, epoch cap or patience
- no use of any test result to select an epoch, a configuration or a next arm
- no rewriting of a binding hash to make a check pass; fail closed instead

## 9. Known defects carried over deliberately

- **patience 40 is smaller than the first LR milestone 120, so MultiStepLR can never fire before early stopping**
  - Why kept: changing it would break the single-factor contrast with the reference arm, which ran under the same defect and stopped at epoch 64
- **the trainer constructs T1ceWtPreprocessConfig(spatial_preprocessing='whole_resize') while the cache is braincrop_resize**
  - Impact: none on behaviour: that object is consumed only by build_training_augmentation, which is geometry-independent. The geometry recorded in train_config.json is copied from the cache index and is therefore correct.
  - Why kept: the alternative is editing a trainer whose SHA is recorded in the completed arm's train_config.json

## 10. Prediction registered before the run

- Primary: validation scan-macro DSC at click 3 exceeds the reference arm's 0.5633
- Falsifiable: if validation DSC does not exceed 0.5633, the crop is recorded as not helping when trained with, and no test run is taken
- The agent previously predicted the inference-only probe would LOWER the score and was refuted by measurement. That miss is recorded so this prediction is read with appropriate scepticism.
