# CAPG-3D Dice+0.1CE replacement preregistration — 2026-08-14

## Why this is a replacement run

The first formal `DiceCE(1:1)` pair was stopped after six complete epochs and
preserved under `capg3d_pair_20260814/PROTOCOL_ABORT.json`.  Both arms converged
toward the same all-background solution: train/validation hard Dice reached
zero while loss approached 1 (soft Dice loss 1 plus CE approaching zero).
This diagnoses an objective imbalance for small WT foreground in a whole-brain
field, not an architecture contrast.

The replacement makes one matched change before either architecture has
produced a usable segmentation: segmentation loss becomes
`1.0 * soft Dice + 0.1 * CE`.  This preserves the foreground-overlap gradient
while reducing the voxel-count-dominated incentive to predict background.
Because the failure and correction were observed on the development split,
the eventual result remains development evidence and must not be described as
an untouched confirmatory result.

## Frozen remainder

Everything else remains as in `CAPG3D_PREREG_20260814.md`:

- 972 train / 108 validation, subject-clean; no 270-case test access;
- exact byte-matched N4 image-only brain-crop cache;
- GPU0 plain vs GPU1 CAPG, shared seed `20260819` and matched initialization;
- batch 8, BF16, AdamW `2e-4`, weight decay `1e-5`, max 100 epochs,
  minimum 30, patience 20;
- CAPG centre loss weight `0.1`, Gaussian sigma 6, centre bias -4;
- frozen brain-crop SAM checkpoint and fixed 32^3 dense transfer
  `5*tanh(locator_logit/5)`;
- fixed reports: locator, point-only, dense-only, point+dense;
- primary endpoint: CAPG minus plain `sam_point_dense` scan Dice;
- positive-candidate threshold: at least `+0.005`; subject-bootstrap CI and
  point-hit rate remain secondary; no threshold or component search.
