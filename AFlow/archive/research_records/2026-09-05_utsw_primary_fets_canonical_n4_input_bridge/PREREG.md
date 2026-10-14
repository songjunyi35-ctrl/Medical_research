# UTSW primary-FeTS canonical-N4 input bridge preregistration

Frozen before the first UTSW canonical N4 execution on 2026-09-05.

## Frozen sources

- UTSW source transfer receipt SHA-256: `f37fa324c413ba7a9c91db0f80b764501c29d1b926501c9f89237d2f96111c90`.
- UTSW source raw QC SHA-256: `0f31398be358b4c38d3c961adcf08b63eae4617002ea5b2814aa773b484ee80f`.
- Passed primary-FeTS no-N4 INPUT_QC SHA-256: `7573bed6d03f194a429d7eacb4ffb3b711feeda15e91a47539aaf025020deb59`.
- Subjects/image/label: BT0001/2/3; `brain_t1ce.nii.gz`; `rtumorseg_manual_correction.nii.gz`; copy A and B.

## Frozen execution

1. Parent freezes the implementation and confirms no prior per-copy/output reports.
2. Launch copy A and copy B as separate Python processes, sequentially to avoid CPU contention.
3. In each process set SimpleITK global threads=1; canonical RAS; require source image finite/nonnegative with zero/positive support and label integer/nonempty; map every label>0 to WT.
4. Call the historical canonical `arrays(..., use_n4=True)`: positive-mask N4, shrink2, unchanged four-level iteration schedule, support bbox, source z-score, trilinear/nearest-exact resize128³.
5. Each child stores only source/output hashes, finite/nonempty flags and geometry. Parent compares child reports and the archived no-N4 INPUT_QC.

## Frozen gates

- A versus B: output image SHA, output label SHA and geometry exact for all three.
- Canonical versus archived no-N4: output label SHA and geometry exact for all three; output image SHA different for all three.
- No rerun, N4 mask/threads/iterations/tolerance/case change after any failure.
- No model/GPU/validation/outer/confirmation/test270/UCSD/RHUH access; no remaining UTSW subject access.

