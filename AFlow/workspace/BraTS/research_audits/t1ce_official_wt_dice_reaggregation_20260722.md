# T1ce Official-WT Dice Reaggregation

Date: 2026-07-22 19:55 CST

## Question

The residual-tumor endpoint is now frozen as the BraTS 2024 post-treatment
official whole-tumor region:

`WT = raw labels {1,2,3}`.

The question was whether the existing 270-case Dice must be rerun or changes
after adopting this definition.

## Target identity

No inference rerun is required. The completed evaluator already constructs its
binary target with `np.isin(seg, (1,2,3))`. This is identical to the official WT
mapping in `sammed3d/brats_metrics.py` and to
`sammed3d/t1ce_binary.py::RESIDUAL_TUMOR_LABELS`.

The legacy target named `tumor_WT` in `scripts/prepare_brats2024_full.py` is not
the same endpoint: it uses T2-FLAIR and raw labels `{1,2,3,4}`. Its historical
approximately 0.7 Dice must not replace the current official-WT result.

## Bound evaluation

- Cases / subjects: 270 / 119.
- Manifest:
  `data/datasets/brats2024_random_subject_80_20_seed20260722_v1/sealed_test/test_cases.jsonl`.
- Input: T1ce only.
- Weights: untouched `sam_med3d_turbo.pth`.
- Geometry: `ToCanonical + GT-centered CropOrPad(128^3)`.
- Primary interaction: three accumulated uniform GT/XOR clicks plus recurrent
  low-resolution mask.
- Threshold: raw logit greater than zero.
- TTA / postprocessing: none / none.
- Report:
  `workspace/BraTS/t1ce_official_geometry3_test/full_270_single_b200_v2.json`.
- Report SHA-256:
  `18b6e245fe0a566ec7498d3f478b1d9e4405d4ae951b4b09e4031223ade37460`.

## Reaggregated Dice

Primary accumulated-click protocol:

| Endpoint | Click 1 | Click 2 | Click 3 | Best-of-3 |
|---|---:|---:|---:|---:|
| Scan-macro crop Dice | 0.243101 | 0.291748 | **0.330096** | 0.339958 |
| Subject-balanced crop Dice | 0.242981 | 0.292463 | **0.329269** | 0.338133 |

For click 3:

- scan population standard deviation: `0.166867`;
- scan median and IQR: `0.336744 [0.196602, 0.438494]`;
- subject population standard deviation: `0.136175`;
- subject median: `0.338313`;
- subject-bootstrap 95% interval: `[0.304634, 0.353668]`, 10,000
  subject-level resamples with seed 20260722;
- 48/270 scans reach Dice at least 0.5, 3/270 reach at least 0.7, and 27/270
  remain below 0.1.

Click 3 improves over click 1 by mean `+0.086996`. There are 219 improved and
51 worsened scans.

The crop retains all target voxels in 248/270 cases. If every official-WT voxel
outside the crop is counted as a false negative while predictions outside the
crop are fixed to background, click-3 Dice becomes `0.329909` scan-macro and
`0.329066` subject-balanced. The small change confirms that crop truncation is
not the cause of the remaining low score, although the GT-centered crop remains
an oracle localization assumption.

The upstream-exact RITM/current-point arm has click-3 scan Dice `0.330285`,
subject-balanced Dice `0.327565`, best-of-3 scan Dice `0.352253`, and
best-of-3 subject-balanced Dice `0.349954`.

## Conclusion

The current official-WT endpoint does not recover the historical approximately
0.7 value. The correct three-click result remains approximately 0.33. The old
value used a different modality, label union, click count, and best-of-five
aggregation. Future improvement must come from training/adaptation under the
frozen WT protocol, not from relabeling the existing result.

No model forward, GPU process, threshold search, TTA, postprocessing, or new
test access was used for this reaggregation.
