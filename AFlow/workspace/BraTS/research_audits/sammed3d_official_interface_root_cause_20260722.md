# SAM-Med3D T1ce three-click interface root-cause audit

Date: 2026-07-22 17:18 CST

## Finding

The earlier `0.011020` three-click result is not a valid SAM-Med3D baseline. It
used whole-volume resizing, preserved the source LAS array orientation, and did
not pass the previous low-resolution mask to the prompt encoder. It remains only
an invalid-interface diagnostic.

The untouched checkpoint itself is healthy. The repository's own
`phase0_reproduce.py` was rerun on one B200 for three stored `tumor_WT` cases.
Best-of-five Dice was `0.7777/0.6111/0.7694` (mean `0.7194`) versus the stored
`0.7911/0.5631/0.7612` (mean `0.7051`).

## Upstream code contract

`SAM-Med3D-main/validation.py` does all of the following:

1. `TorchIO ToCanonical`;
2. GT-label-centered `CropOrPad(128,128,128)`;
3. masked TorchIO z-normalization after cropping;
4. current corrective point as the sparse prompt;
5. previous decoder low-resolution logits as the recurrent dense mask prompt;
6. RITM clicks for steps 1-2, uniform XOR error sampling afterward;
7. crop-domain Dice and, for its reported endpoint, per-case maximum over clicks.

The user's frozen protocol instead requires three clicks, uniform first GT
foreground sampling, uniform FN/FP error sampling, and accumulation of all prior
points. The corrected primary arm combines that sparse-click contract with the
upstream recurrent mask and upstream oracle crop geometry.

## 270-case corrected results

Input is T1ce. The reconstructed residual target is raw labels `{1,2,3}`, with
label 4 excluded. All numbers are case-macro Dice on the oracle 128-cubed crop.

| Arm | Click 1 | Click 2 | Click 3 | Best-of-3 |
|---|---:|---:|---:|---:|
| Accumulated uniform clicks, no recurrent mask | 0.008997 | 0.011974 | 0.014095 | 0.014735 |
| Accumulated uniform clicks + recurrent mask | 0.243101 | 0.291748 | **0.330096** | 0.339958 |
| Current uniform click + recurrent mask | 0.243101 | 0.285019 | 0.310956 | 0.335998 |
| Upstream RITM/current click + recurrent mask | 0.257079 | 0.303683 | 0.330285 | 0.352253 |

Recurrent mask feedback contributes `+0.316002` click-3 Dice when holding the
accumulated-point arm and crop geometry fixed. Accumulating history contributes
`+0.019140` over current-point-only under the matched uniform-click seed.

Crop target coverage mean/min is `0.998466/0.932624`; 248/270 cases have complete
target coverage. This is an oracle crop metric and is not an automatic full-volume
endpoint.

## Interpretation boundary

The corrected `0.330096` is the valid measurement for the user's three-click
protocol under an upstream-compatible decoder interface, not a reproduction of
the DPD-SAM paper's `0.741` automatic baseline. The local paper explicitly omits
both the raw BraTS-label mapping for its residual target and the automatic prompt
protocol used for that baseline. The historical repository `~0.716` result is a
different task: `tumor_WT`, a selected T2-FLAIR input, labels `{1,2,3,4}`, five
clicks, and per-case best-of-five. It is not T1ce residual `{1,2,3}` click-3.

## Artifacts

- Full report: `workspace/BraTS/t1ce_official_geometry3_test/full_270_single_b200_v2.json`
- Report SHA-256: `18b6e245fe0a566ec7498d3f478b1d9e4405d4ae951b4b09e4031223ade37460`
- Evaluator SHA-256: `d4ed817832e0aca948b11a8024bf91c22bbec0fba0cf20f88f4682d604ed63c5`
- Phase-0 wrapper SHA-256: `805db6c4cde1d2d6224ad09bc3e4e5bd8c9f3ff78804cda8636591bd79f77a91`
- Runtime: one B200 (`1007515`), 173.15 seconds; no remaining compute process.
