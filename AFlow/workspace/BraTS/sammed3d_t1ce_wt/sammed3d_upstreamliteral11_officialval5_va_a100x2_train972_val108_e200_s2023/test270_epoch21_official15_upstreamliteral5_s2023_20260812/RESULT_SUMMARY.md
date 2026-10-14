# BraTS 2024 T1ce/WT clean-baseline test result — 2026-08-12

## Binding

- Test: 270 scans / 119 subjects, zero case and subject overlap with development.
- Model: epoch-21 validation-best checkpoint, SHA-256
  `c0c85a0fb90f6301344053f97cabe8a4972bd9febd4c6ae248f3a0908598fe31`.
- Preprocessing: paired 1.5 mm resampling, canonical orientation, GT-centred
  CropOrPad 128^3, positive-voxel sample z-score, no N4, no test augmentation.
- Prompt protocol: five current-point/recurrent-mask interactions; RITM for
  clicks 1-2, uniform XOR for clicks 3-5; raw-logit threshold 0; stable
  SHA256(seed=2023, case ID) sampling.
- Test was not used for checkpoint selection or adaptation.

## Results

| Metric | Scan macro | Subject balanced |
|---|---:|---:|
| Per-case best of clicks 1-5 | **0.594226** | 0.580970 |
| Final click 5 | **0.585754** | 0.572699 |

Scan-macro Dice by click: 0.548900, 0.566940, 0.573730, 0.579031,
0.585754. The scan-macro best-of-five population standard deviation was
0.131231 and median was 0.600321.

The 0.5942 clean-baseline result is numerically close to MSCL Table 2's 0.601
baseline, but it must not be claimed as an exact reproduction because that
paper does not disclose an equivalent subject split and prompt/evaluation
protocol.

## Audit

- Complete report: `test_report.json`
- Report SHA-256:
  `7bd59843459876bd006230b24bc111cdfd726896cb3e6748b54b24c0a9659b63`
- Completion binding: `TEST_DONE.json`
- All 270 cases were unique, in frozen manifest order, finite, and independently
  recomputed after report creation.
- The completed evaluator hung only during final NCCL teardown; exact PIDs were
  terminated after both report hashes were proven unchanged. See
  `TEARDOWN_NOTE.json`.
