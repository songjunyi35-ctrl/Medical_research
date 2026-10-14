# UTSW fixed-three label-variant bridge preregistration

Frozen before any UTSW NIfTI payload transfer or pixel access on 2026-09-05.

## Frozen sources and cohort

- Official TCIA package ID: 1265; collection UTSW-Glioma; CC BY 4.0.
- Official full inventory SHA-256: `5f210c0a1e0c33f7b71a56d9bbce8b52a62f24e2a0b53655aa3a098dbde3f871`.
- Fixed subjects: BT0001, BT0002, BT0003, the lexicographically first three official TSV rows with `Manually Refined Segmentation=Yes`.
- Fixed files per subject: `brain_t1ce.nii.gz`, `brain_t1ce_ants.nii.gz`, `rtumorseg_manual_correction.nii.gz`, `tumorseg_manual_correction.nii.gz`, `tumorseg_FeTS.nii.gz`.
- Fixed 15-path/size text SHA-256: `0f7fd05936e9b216ae744961d45d33a602d4ea7ba5e7fcaa5d6a54c8bd2c9a3c`; total bytes per copy: 28,880,493.

## Frozen access order

1. Rehash the official inventory and IDEA/PREREG.
2. Obtain one transfer specification containing exactly the15 frozen paths and transfer to copy A.
3. Obtain a fresh public authorization/transfer specification for the same15 paths and transfer to copy B.
4. Require path set, official listed size, local size, and SHA-256 equality A versus B for all15 before loading any NIfTI.
5. Only then load headers/arrays and evaluate all four predeclared pairs:
   - `brain_t1ce` × `rtumorseg_manual_correction`
   - `brain_t1ce` × `tumorseg_manual_correction`
   - `brain_t1ce_ants` × `rtumorseg_manual_correction`
   - `brain_t1ce_ants` × `tumorseg_manual_correction`

## Frozen gates

- Geometry pass per pair/case: shape exact and affine elementwise max absolute difference ≤`1e-6`.
- Image validity: finite and at least one nonzero voxel.
- Label validity: finite, exactly integer-valued, includes0 and at least one positive voxel. Label values and overlap with the FeTS label are descriptive only and cannot select among geometry ties.
- Overall pass: all transfer gates and validity gates pass, and exactly one of the four pair definitions passes geometry for all three cases.
- Zero or multiple all-case geometry pairs closes this implementation as unresolved. No case substitution, tolerance change, resampling, orientation transform, label-union choice, or model-score tie-break is allowed.

## Forbidden access

- No checkpoint load, GPU, model forward, validation/outer/confirmation/test270/UCSD/RHUH access.
- No UTSW file outside the15 frozen paths.
- No full-package download or future external evaluation authorization.

