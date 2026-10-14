# AFLOW BraTS handoff addendum — 2026-08-10

This addendum records the H100 reproduction performed after
`AFLOW_BRATS_HANDOFF_20260807.md`. It does not replace or relax any guardrail in
that handoff.

## Mainline remains frozen

- T1ce only.
- WT is raw labels `{1,2,3}`; raw label `4` is excluded.
- Input is `128 x 128 x 128`.
- Evaluation uses three accumulated corrective clicks: click 1 from GT
  foreground and clicks 2/3 from XOR error, with the previous low-resolution
  mask recurrently supplied.
- The decision rule is raw logit `> 0`.
- Development split is subject-clean: 972 train scans and 108 validation scans.
- The sealed 270-case test split was not opened.

## Same-H100 baseline closure

The frozen stage-2 GT-crop checkpoint was evaluated on a fresh H100 using both
the unmodified mainline evaluator and the new refiner-cache generator. Both
returned exactly `0.6359485301438967` scan-macro validation DSC. The residual
refiner was zero-initialized and its epoch-0 output was identical to the cached
coarse logits (`0.6359485293721298` after float16 cache storage).

The historical independent value `0.6380564026652955` is therefore not the
same-H100 numerical reference. The approximately `-0.002108` difference is
consistent with BF16-dependent corrective-click branching across GPU runs; it
is not a cache-runner mismatch.

## Full-resolution residual refiner result

The preregistered model consumes T1ce and the frozen SAM click-3 raw logit and
predicts an additive full-resolution residual. It completed all 60 epochs on an
H100 without NaNs, OOM, or protocol changes.

- selected epoch: 58
- frozen coarse validation DSC: `0.6359485293721298`
- refined validation DSC: `0.6683989912099032`
- absolute gain: `+0.032450461837773524`
- improved/worsened scans: 93/15 of 108
- subject-macro coarse/refined DSC: `0.6308633 -> 0.6647122`
- checkpoint SHA-256:
  `4513ea4544ae153064575ec2ed5c56e1b78adf5e89e9d094754227b3bd429f21`
- `TRAIN_DONE.json` SHA-256:
  `da16735a54f98817e72a0527ce72f04ee925c5bfaba93cf1d90f333da7f36ccb`

The preregistered development-success line was 0.68 and the reproduction target
was 0.70. Neither passed. Consequently the automatic-cascade gate remains
closed and the sealed test must not be accessed for this arm.

## Updated bottleneck

The refiner gives a broad real gain, but lesion size remains dominant:

| GT WT volume | n | coarse DSC | refined DSC | gain |
|---|---:|---:|---:|---:|
| `<8k` | 7 | 0.2408 | 0.2567 | +0.0159 |
| `8k-20k` | 18 | 0.5842 | 0.6181 | +0.0340 |
| `20k-40k` | 23 | 0.5911 | 0.6292 | +0.0381 |
| `>=40k` | 60 | 0.7148 | 0.7465 | +0.0318 |

The 101 scans with WT volume at least 8k voxels average `0.6969333` after
refinement. The seven sub-8k scans average only `0.2566896`; three of those are
longitudinal scans from subject `BraTS-GLI-02208` and remain near zero. This is
the clearest remaining within-mainline bottleneck.

The deterministic refiner gain (`+0.0325`) is also close in scale to the
DPD-SAM paper's reported CDR-only increment (`+0.041`). The larger unresolved
gap is upstream: the paper/colleague baseline is reported above 0.7, while the
same-H100, subject-clean, three-click local baseline is exactly 0.63595. No
local artifact was found that demonstrates a 0.7+ result under all of the
frozen mainline dimensions. Do not substitute the old approximately 0.745
artifact: that run used T2-FLAIR, WT1234, and five clicks.

## Artifacts

- Result ledger:
  `workspace/BraTS/sammed3d_t1ce_wt/RESIDUAL_REFINER_RESULT_20260810.json`
- Original preregistration:
  `workspace/BraTS/sammed3d_t1ce_wt/RESIDUAL_REFINER_PREREG_20260810.json`
- Same-H100 amendment:
  `workspace/BraTS/sammed3d_t1ce_wt/RESIDUAL_REFINER_PREREG_AMENDMENT_H100_20260810.json`
- Same-H100 evaluator recheck:
  `workspace/BraTS/sammed3d_t1ce_wt/H100_RECHECK_GTCROP_3CLICK_20260810.json`
- Full run:
  `workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_stage2_gtcrop_h100_train972_val108_e60_s20260810_v2/`
- Cache generator:
  `scripts/cache_sammed3d_t1ce_wt_refiner_inputs.py`
- Training runner:
  `scripts/train_sammed3d_t1ce_wt_residual_refiner.py`
- Reproduction runner:
  `scripts/run_t1ce_wt_residual_refiner_reproduction_20260810_v2.sh`

## Highest-value next evidence

Before spending another sealed-test read or claiming a failed reproduction, get
the colleagues' exact manifest, subject IDs, raw-label mapping, modality,
checkpoint hash, click sampler/seed, metric aggregation, and per-case output.
The local evidence now rules out H100 availability and the refiner stage as the
main source of the 0.7 discrepancy. If those dimensions are identical, the next
development-only arm should target prompt-centered coarse-to-fine treatment of
the sub-8k regime, preregistered as one fixed geometry rather than a validation
grid search.
