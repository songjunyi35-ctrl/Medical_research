# DPD-SAM T1ce binary restart protocol v2

Date: 2026-07-22
Status: frozen geometry-corrected development protocol; supersedes v1 for new runs

`authorizes_development_training=true`  
`authorizes_confirmatory_access=false`  
`authorizes_test_access=false`

This protocol inherits the target, split, prompt firewall, optimizer defaults, and
five-run rules from `dpdsam_t1ce_binary_restart_v1.md`. It changes only the mask
resampling contract, dual-grid reporting, checkpoint selection, and validity of runs
created before this correction.

## 1. Fixed task

- Input: T1ce only (`modalities.t1c`), one channel.
- Target: binary union of raw labels `{1,2,3}`; raw label `4` is background.
- Geometry: whole volume resized to exactly `128 x 128 x 128`; no crop or sliding window.
- Prompt: image-only automatic continuous prompt; no label-derived point or box.
- Development split: 872 fit cases / 389 subjects and 209 validation cases / 97
  subjects, subject-disjoint. The historical outer 269 cases remain sealed.
- Threshold: fixed probability `>= 0.5`; no threshold, TTA, morphology, or
  post-processing search.

## 2. Geometry correction

Image interpolation remains trilinear with `align_corners=False`. Every categorical
spatial operation now uses PyTorch `nearest-exact`, including:

1. source brain mask to the model grid;
2. source `{1,2,3}` target to the model grid;
3. thresholded model prediction back to the source grid.

Legacy PyTorch `nearest` is forbidden. It uses a different coordinate convention and
was empirically shown on real development cases to misalign the mask with the
center-aligned image. Across all 1081 development cases, source-to-128-to-source target
self-DSC mean/median/p05/min was `0.788933/0.799949/0.662055/0.275551` with legacy
`nearest`, versus `0.924792/0.929385/0.876510/0.731407` with `nearest-exact`.
The complete statistics are recorded in
`workspace/BraTS/research_audits/t1ce_preprocess_metric_audit_20260722.md` and are not
inferred from model output.

## 3. Dual-grid metrics

Every validation event must emit both grids from the same thresholded prediction:

- `model_grid`: prediction versus the `nearest-exact` 128-cubed target. Physical
  spacing is `source_spacing * source_shape / model_shape`.
- `source_grid`: the same prediction restored with `nearest-exact` versus the untouched
  source target using source spacing.

For comparison with the local DPD-SAM draft, the pre-registered headline and best
checkpoint rule are macro `model_grid.dsc`: the draft states that all volumes are
resized to `128^3` and does not describe restoring predictions before scoring. This is
a transparent reconstruction assumption, not a recovered author-code fact.
`source_grid` DSC/IoU/HD95 is a mandatory stricter secondary report and may not be
suppressed when it is lower.

Empty-mask policy is unchanged on both grids: both empty gives DSC/IoU/HD95
`1/1/0`; one empty gives `0/0/FOV-diagonal-mm`. Report the empty counts per grid.

## 4. Invalidated runs

All T1ce B0/B1 results created before the `nearest-exact` correction are invalid for
scientific comparison because both the training target and source-grid restoration used
legacy `nearest`. This includes the real smoke, both 64/16 pilots, both 256/64 pilots,
and the interrupted full 872/209 e7/e8 attempts. Their files are retained only for
engineering provenance and are enumerated in
`workspace/BraTS/dpdsam_t1ce_b0/INVALID_LEGACY_NEAREST_ARTIFACTS_20260722.md`.

No old checkpoint may be resumed. Corrected runs must start from the frozen source
SAM-Med3D checkpoint in a new output directory.

## 5. Corrected development sequence

1. Pass CPU unit tests for target construction, `nearest-exact`, dual-grid scoring,
   prompt firewall, strict checkpoint binding, and RNG resume.
2. Run a paired corrected 256-fit/64-validation, 20-epoch gate on two B200s:
   - B1 engineering control: static target token plus encoder Q/V LoRA rank 4;
   - D1 paper-aligned coarse reconstruction: frozen image encoder, fit-only GAP
     descriptors, K-means++ `K=4`, rank-1 dynamic domain prompt, adapter CE beta 0.1.
3. Both arms use the same cases, seed, Dice+BCE, threshold, validation cadence, and
   dual-grid evaluator. D1 additionally uses its declared adapter CE.
4. Advance the stronger valid automatic arm to all 872/209 cases. D1 uses the paper's
   first 50-epoch no-CDR warm-up before any conditional diffusion implementation.
5. Only after a high-scoring automatic checkpoint and evaluation contract are frozen
   may MedTAWS search begin. The sealed outer set remains inaccessible until the final
   five-seed configuration is frozen.

## 6. Reproduction labels

- B0/B1 static-token arms are engineering controls, not the paper's reported
  SAM-Med3D baseline reproduction.
- D1 is a minimal DPD-SAM reconstruction, not bit-exact: frozen encoder-output GAP is
  an explicit CLS surrogate, prompt length defaults to one, and author code is absent.
- The local draft targets are SAM-Med3D DSC `0.741` and DPD-SAM DSC `0.868`. These
  targets are not considered reached until a valid corrected run uses the frozen
  development protocol; legacy-nearest values cannot be compared to them.
