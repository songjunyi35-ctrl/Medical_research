# UTSW primary-FeTS no-N4 input bridge preregistration

Frozen before the first primary-FeTS input conversion on 2026-09-05.

## Frozen inputs

- Source transfer receipt SHA-256: `f37fa324c413ba7a9c91db0f80b764501c29d1b926501c9f89237d2f96111c90`; 15/15 official dual transfers exact.
- Source raw QC SHA-256: `0f31398be358b4c38d3c961adcf08b63eae4617002ea5b2814aa773b484ee80f`.
- Subjects: BT0001, BT0002, BT0003.
- Image: `brain_t1ce.nii.gz` only; `_ants` is forbidden in this implementation.
- Label: `rtumorseg_manual_correction.nii.gz` only.
- Copies: `/tmp/aflow-utsw-label-variant-20260905/copy_a` and `copy_b`.

## Frozen transformation

1. Load both files with `nibabel.as_closest_canonical`.
2. Require image and label shape exact and affine max absolute difference≤`1e-6`.
3. Require finite image, min≥0, at least one zero and positive voxel.
4. Require finite exactly integer label with zero and positive voxels; convert every label>0 to value1 before calling the shared training preprocessing so UTSW label4 remains part of WT.
5. Call the shared no-N4 `arrays` route: nonpositive→0, positive-support bbox, positive-population z-score, trilinear image and nearest-exact label resize to128³.
6. Compare copy A/B output arrays byte-exact and geometry dict exact for all three cases.

## Gates and access

- All stated gates must pass for all three cases; no tolerance, subject, image version or mapping changes.
- No N4, checkpoint, GPU, model forward, old validation/outer/confirmation/test270/UCSD/RHUH access.
- A pass is input-only and does not authorize downloading the remaining359 manual cases or any UTSW model evaluation.

