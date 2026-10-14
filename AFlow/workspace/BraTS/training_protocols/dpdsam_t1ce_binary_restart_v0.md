# DPD-SAM T1ce binary restart protocol v0

Date: 2026-07-22 14:55 CST
Status: direction freeze; target-label mapping audit in progress; GPU launch blocked

`authorizes_confirmatory_access=false`  
`authorizes_test_access=false`

## Superseding task contract

This protocol supersedes the four-modal five-class Paper1913/P0 performance
mainline at the user's explicit direction. The stopped P0 artifacts are retained
only as historical diagnostics and must not be resumed.

- input: T1ce only (`modalities.t1c` locally), tensor `[B,1,128,128,128]`
- output: one binary residual-tumor mask versus background
- prompt: automatic learned continuous domain-specific prompt; no manual point
  or box at inference
- geometry: preprocess the whole MRI volume and resize to `128^3`; do not use
  the P0 full-grid sliding-window protocol
- metrics: binary DSC, IoU and HD95 on the independent patient-level test set
- statistics: five independent runs sharing one patient-level split and using
  different random seeds; report mean and sample standard deviation
- split: patient-level 8:2 train/test; reserve approximately 20% of the training
  cohort for validation (approximately 64/16/20 percent overall)

The local five-class BraTS label set must not be collapsed by guesswork. The
DPD-SAM PDF calls the target a single-channel residual-tumor ground truth but
does not disclose whether this is ET, TC, WT, or another union. Exact mapping
must be recovered from author code/artifacts or explicitly frozen by the user
before any split generation, metric claim, or GPU training.

## Paper-declared reconstruction settings

Primary local source: `article/DPD-SAM.pdf`, SHA-256
`ce49d74a94dcd3529c394a39458ea8583ac8d8226aa0d02de099112171473b4d`.

- backbone: SAM-Med3D; frozen image encoder with an added learned CLS token
- offline LDM: frozen-encoder CLS features, K-means++ with `K=4`
- trainable path: rank-1 domain prompt generator, prompt adapter, SAM mask
  decoder, and conditional diffusion refinement network
- segmentation loss: weighted Dice + BCE (paper does not declare the two
  coefficients)
- total loss: segmentation + `alpha=0.1` diffusion + `beta=0.1` adapter CE
- warm-up: 50 epochs without diffusion, then joint refinement
- optimizer: AdamW, learning rate `4e-4`, weight decay `0.1`
- training: 200 epochs, batch size 2
- preprocessing: intensity normalization, bias-field correction, skull stripping,
  whole-volume resize to `128^3`
- augmentation: TorchIO affine (rotation up to 15 degrees, translation up to
  10%), elastic deformation, and Gaussian noise
- diffusion: training `T=1000`, DDIM inference `S=10`

Paper BraTS2024 five-run results: U-Net `51.9+/-2.1`, nnU-Net `72.8+/-1.2`,
SAM-Med3D `74.1+/-1.5`, CDSG-SAM `83.1+/-0.8`, and DPD-SAM
`86.8+/-0.6` DSC. These numbers are binary test-set results and are not
comparable to the stopped P0 WT/TC/ET macro D3.

## Launch gates

1. Recover and freeze the exact residual-tumor label mapping.
2. Rebuild a 1350-case patient-level 64/16/20 development/validation/test split
   with one frozen split shared by all five seeds.
3. Implement a strict T1ce-only binary loader, whole-volume `128^3` transform,
   binary DSC/IoU/HD95 evaluator, and GT firewall tests.
4. First reproduce the paper's SAM-Med3D `74.1+/-1.5` automatic-prompt baseline;
   only then attach MedTAWS to the same frozen binary inference contract.
5. Do not reuse four-modal D3/D6 scores as evidence for this task.
