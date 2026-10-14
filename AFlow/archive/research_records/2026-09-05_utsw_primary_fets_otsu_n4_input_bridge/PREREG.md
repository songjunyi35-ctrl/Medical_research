# UTSW primary-FeTS Otsu-N4 input bridge preregistration

Frozen before first UTSW Otsu-N4 execution on 2026-09-05.

## Frozen evidence and inputs

- Source transfer receipt SHA `f37fa324c413ba7a9c91db0f80b764501c29d1b926501c9f89237d2f96111c90`.
- Source raw QC SHA `0f31398be358b4c38d3c961adcf08b63eae4617002ea5b2814aa773b484ee80f`.
- no-N4 INPUT_QC SHA `7573bed6d03f194a429d7eacb4ffb3b711feeda15e91a47539aaf025020deb59`.
- canonical INPUT_QC SHA `1cec4ba98fae05e54caca2a1eb0a14681758e79ed6ac35ec47b0a28a58353be0`; canonical COPY_A SHA `a1776d09d7e84a786e3fc27f114ec4119a6f02bf1cdb35102906a3e8876a3665`.
- BT0001/2/3; copy A/B; primary `brain_t1ce.nii.gz`; full-grid `rtumorseg_manual_correction.nii.gz`.

## Frozen execution

1. Two separate Python child processes, SimpleITK global threads1.
2. Canonical RAS and source geometry≤1e-6; finite nonnegative image with zero/positive support; integer label subset`{0,1,2,4}` with zero/positive; map label>0 to WT.
3. `n4_otsu_correct`: OtsuThreshold0/1/200 mask, nonempty and subset source-positive; shrink2; `[50,50,50,50]`; apply field to full source and retain exact positive support.
4. Shared no-N4 arrays stage only after correction: support bbox, source population z-score, trilinear image/nearest-exact label to128³.
5. Each child stores hashes/geometry/mask metadata; parent compares children, no-N4 and canonical archived rows.

## Frozen gates/access

- Cross-process output image/label/geometry3/3 exact.
- Versus no-N4 and canonical: label/geometry3/3 exact; image3/3 different from each.
- No rerun or parameter/case/tolerance change on failure.
- 0checkpoint/GPU/model/old cohort/remaining UTSW subject access; pass does not authorize full362.

