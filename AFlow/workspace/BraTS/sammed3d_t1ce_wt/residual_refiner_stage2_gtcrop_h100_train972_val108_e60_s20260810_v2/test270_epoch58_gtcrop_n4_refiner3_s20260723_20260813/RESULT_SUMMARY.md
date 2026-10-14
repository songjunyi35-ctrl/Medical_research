# GT-centered residual refiner: 270-case test

The frozen epoch-58 refiner completed its one authorized readout on 270 scans
from 119 subjects.

| Aggregation | Frozen SAM click-3 coarse | Refined | Paired gain |
|---|---:|---:|---:|
| Scan macro | 0.666658 | **0.703798** | **+0.037140** |
| Subject balanced | 0.648864 | **0.686637** | **+0.037773** |

The refiner improved 244 scans and worsened 26. The scan-macro validation
selection value was 0.668399 at epoch 58; validation, not test, selected the
checkpoint.

This is an oracle-localization result: preprocessing uses N4 followed by a
native-resolution GT-centered 128^3 crop, and inference uses three accumulated
GT-derived corrective clicks. It is not an automatic end-to-end score, is not
the official15/five-click branch, and must not replace the automatic brain-crop
test score of 0.625359. No threshold search, TTA, morphology, connected-component
filtering, or test-time adaptation was performed.

Audited machine-readable result: `test_report.json` (SHA-256
`f6ef9f1965f29502e1d190747f0363b89ecb42828ac58213dcd3e0fb674f8f44`).
