# T1ce binary preprocessing and metric audit (2026-07-22)

Status: completed at 2026-07-22 16:11 CST. This audit found and corrected a
material resize-coordinate bug. It did not open any outer-test NIfTI, run a
model, or modify the training runner/model.

## Scope and access boundary

- Audited frozen development manifests only: `872` train cases / `389`
  subjects and `209` validation cases / `97` subjects; subject overlap `0`.
- The outer `269` cases were not supplied to the diagnostic and no outer-test
  image or label was opened.
- Full-volume geometry, labels, empty targets, and resize round trips were
  checked for all `1081` development cases. T1ce voxel coverage was checked on
  a deterministic `64`-case sample (32 per split).
- Reproducer: `scripts/audit_t1ce_preprocess_metrics.py`. It refuses any row
  whose declared split is not `train` or `validation`.

## Finding: legacy nearest was a real alignment bug

The original implementation resized images with trilinear interpolation and
`align_corners=False`, but resized masks and restored predictions with PyTorch
legacy `mode="nearest"`. These modes use different coordinate conventions:
trilinear is voxel-center aligned while legacy nearest follows the historical
corner-biased rule. Thus image and target were systematically displaced on the
128-cube, and restore introduced another biased mapping.

An all-case oracle round trip (`source target -> 128^3 -> source`) measured:

| mode | DSC mean | median | p05 | min | IoU mean |
|---|---:|---:|---:|---:|---:|
| legacy `nearest` | 0.788933 | 0.799949 | 0.662055 | 0.275551 | 0.657152 |
| center-aligned `nearest-exact` | 0.924792 | 0.929385 | 0.876510 | 0.731407 | 0.861328 |

Volume was not the issue: restored/source volume ratio mean was `1.000216` for
legacy and `0.999451` for nearest-exact. The loss was spatial overlap. NumPy
index audit and direct PyTorch results were voxel-identical for both modes on
16 checked cases (`0` mismatches). A separate direct-PyTorch 16-case check gave
legacy mean DSC `0.771836` versus nearest-exact `0.920361`.

**Validity consequence:** every existing T1ce B0 checkpoint/metric produced by
a manifest that does not explicitly freeze `nearest-exact` for mask resize and
prediction restore is invalid for comparison or continuation. In particular,
the pre-fix `workspace/BraTS/dpdsam_t1ce_b0/*` pilots must not be resumed or
reported. They may remain only as failed-pipeline evidence.

## Geometry, axes, labels, and empty cases

- All `1081` image/seg pairs had identical shape/affine; mismatch count `0`.
- Every volume was `(182,218,182)`, spacing `(1,1,1)` mm, orientation `LAS`.
  The loader performs no transpose or crop, so no axis permutation was found.
- Foreground is exactly raw labels `{1,2,3}`. RC label `4` is background.
  Label `4` occurred in `921` cases and remained excluded.
- Source `{1,2,3}` target empty cases: `0`; legacy 128-grid empty cases: `0`;
  restored empty cases: `0`. Empty-target averaging cannot explain low DSC.
- Source target volume was median `39.795 ml`, mean `58.411 ml`, p05
  `8.255 ml`, min `0.801 ml`, max `308.285 ml`.
- In the 64-case T1ce-content sample, 6 cases had target voxels at zero-valued
  T1ce positions, totaling only 125 voxels. Targets are never clipped to the
  brain mask; this is an input-information caveat, not a scoring crop.

## Corrected contract and dual-grid metrics

`sammed3d/t1ce_binary.py` now uses `nearest-exact` for brain masks, binary GT,
and prediction restore. Image resize remains trilinear with
`align_corners=False`. Nonzero source-T1ce population z-score and foreground
`{1,2,3}` are unchanged.

The paper states whole-volume resize to `128^3` but does **not** state whether
DSC/IoU/HD95 are computed on the resized grid or after restoring predictions.
Therefore protocol v2 must report both, from one frozen thresholded prediction:

1. `model_grid`: prediction versus resized GT on `128^3`; spacing is
   `source_spacing * source_shape / (128,128,128)`, here
   `(1.421875, 1.703125, 1.421875)` mm.
2. `source_grid`: prediction restored with `nearest-exact` versus untouched
   source GT at `(1,1,1)` mm.

The reusable interface is `score_t1ce_dual_grid()`. Its output intentionally
sets headline metric to `unresolved_reconstruction_assumption`. Choosing the
paper-comparison headline (model-grid is plausible, but unverified) must be
declared in protocol v2; it cannot be inferred from the published draft.

DSC and IoU use ordinary binary voxel overlap. HD95 uses symmetric surface
distance in physical millimetres. Both-empty is DSC/IoU/HD95 `1/1/0`; one-empty
is `0/0/FOV-diagonal-mm`. Development GT is always present, so this policy only
matters when a model predicts empty.

## Verification and launch gates

- `tests/test_t1ce_binary.py`: `14 passed`; includes exact `{1,2,3}` mapping,
  RC exclusion, non-asymmetric axes, `nearest-exact != nearest`, source restore,
  dual-grid spacing, empty policy, and physical HD95.
- `py_compile` passed for the data module, audit script, and focused tests.
- Required before corrected pilot: versioned v2 implementation manifest must
  record `mask_resize=nearest-exact`, `prediction_restore=nearest-exact`, both
  metric grids, their spacings, and an explicit headline reconstruction choice.
- Run a fresh corrected smoke/pilot from the original SAM-Med3D checkpoint. Do
  not resume pre-fix optimizer/checkpoint state. Keep outer test sealed.

## Conclusion

The low pre-fix Dice was partly caused by a confirmed preprocessing defect, not
by direction, NIfTI axes, target mapping, or empty-case reduction. Correcting
nearest alignment raises the source-grid discretization ceiling from mean
`0.7889` to `0.9248`; remaining gap after a fresh run is model/training behavior.
