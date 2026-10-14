# CAPG-3D automatic prompt experiment preregistration — 2026-08-14

## Why this direction

The earlier `0.625359` brain-crop test score is not a fully automatic prompt
result: the crop box is image-only, but click 1 is sampled from GT foreground
and later clicks are sampled from GT/prediction error.  The completed DPD-style
global/domain prompt arms were also weak on this task.  This experiment therefore
targets the missing deployment step directly: **generate a spatial 3D prompt
from T1ce alone and pass it into frozen SAM-Med3D**.

## Frozen paired arms

- `plain` (GPU 0): one-channel MONAI SegResNet locator; its WT-logit argmax is
  the automatic positive point.
- `capg` (GPU 1): the same SegResNet and segmentation objective, with one extra
  centre-heatmap output and an auxiliary soft-Dice + 0.1 MSE centre loss.  The
  target is a Gaussian (`sigma=6`) centred on the foreground voxel nearest the
  WT centre of mass.  Its heatmap argmax is the automatic positive point.  The
  matched segmentation-head bias is zero in both arms; the sparse centre-map
  channel starts at bias `-4`.
- Centre auxiliary weight: `0.1`.  The output is called CAPG-3D (Centre-Aware
  Prompt Generator 3D).

Both arms use seed `20260819`, train/validation batch 8, BF16, AdamW (`lr=2e-4`, weight decay
`1e-5`), image/label flips, shift up to 8 voxels, intensity scale/offset/noise,
100 maximum epochs, 30 minimum epochs, and patience 20.  Selection is fixed-logit
threshold (`>0`) locator scan-macro Dice on the 108-case validation set.

## Data and leakage guard

- 972 train / 108 validation, subject-clean.
- T1ce only, WT=`{1,2,3}`.
- N4 followed by an **image-nonzero bounding-box crop** and resize to `128^3`.
  No label is used to place the crop.
- Both cache indices must declare `test_access: false` and
  `uses_gt_for_localization: false`.
- The 270-case test has no argument or code path in the trainer/evaluator and is
  not read during this experiment.

## Automatic SAM protocol

The frozen brain-crop SAM checkpoint is
`87dad82c67151128c7e99d64ae87f37c68547c769d2e610a56d432b554e0732a`.
For each validation image the locator produces, without GT:

1. one positive point from the preregistered arm-specific argmax;
2. one `32^3` dense prompt made from
   `5*tanh(locator_segmentation_logit/5)`.

Four fixed reports are produced at threshold zero: locator, SAM point-only,
SAM dense-only, and SAM point+dense.  The label is used only afterward for Dice
and point-hit diagnostics.

## Primary decision rule

Primary endpoint: `sam_point_dense` scan-macro Dice, CAPG minus plain.  CAPG-3D
becomes a positive architecture candidate only if this delta is at least
`+0.005`.  Subject-balanced Dice, a 10,000-replicate paired subject-bootstrap
95% CI, locator Dice, and automatic-point hit rate are secondary.  No prompt
threshold, component rule, logit scale, or checkpoint is tuned after seeing the
paired result.  A negative result will be recorded rather than rescued on the
same validation set.

Resource-smoke amendment, made before any real-data training or validation
result was observed: a synthetic CAPG backward step at batch 8 used 20.13 GiB
on one A100-80G and completed in 1.00 s.  The originally drafted batch 2 was
therefore increased symmetrically to batch 8 for both arms; no loss, learning
rate, epoch budget, selection rule, or endpoint changed.
