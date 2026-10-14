# True-160³ epoch200 test270 result

The user later authorized one fixed epoch200 evaluation on the reused internal 270-case test set. This supersedes the earlier training-only access state recorded in `RESULT.md`; outer selector and confirmation remained unopened.

## Correct test preprocessing

The test input was rebuilt from the 270 raw T1ce/seg NIfTI pairs. It did not upscale or otherwise consume the historical 128³ test cache. The chain was canonical RAS → N4 shrink-2 → image-derived nonzero brain crop → source nonzero-population z-score → trilinear image / nearest-exact label resize to 160³.

- Test manifest SHA-256: `a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9`
- Cache index SHA-256: `62b9348726b3ad34078a2e18b5dcb6879c100dd41a6329033090d1b524438a1a`
- Image blob SHA-256: `5c726bc1c31880bf459f20b60d08ac3297e977f4e21e31531bc5a02c6ecdd3c3`
- Label blob SHA-256: `0f84e0184f33bf17d72cf088f246d7d90c9e81335c33ac76fc573da4b8d4de47`
- Four evenly spaced cases reproduced image and label bytes exactly from raw NIfTI.

## Absolute result

Fixed checkpoint: true-160³ full-756 epoch200, SHA-256 `a25e8519b3f9a66ee5c389ad94f0e62ac548dfd6339468637c8f2c3536139b67`.

| metric | true-160³ epoch200 | bootstrap 95% CI |
|---|---:|---:|
| scan Dice | `0.737060297` | `[0.717035, 0.756105]` |
| subject Dice | `0.719728630` | `[0.691519, 0.746955]` |
| point hit | `0.874074` | — |

The report covers 270 scans / 119 subjects. Threshold was raw logit `>0`; TTA, postprocessing, and SAM were disabled.

## Descriptive comparison with historical true-128³ epoch168

| metric | 128³ epoch168 | 160³ epoch200 | candidate minus baseline |
|---|---:|---:|---:|
| scan Dice | `0.741203949` | `0.737060297` | `-0.004143652` |
| subject Dice | `0.723179486` | `0.719728630` | `-0.003450856` |
| point hit | `0.929630` | `0.874074` | `-0.055556` |

The paired scan difference 95% CI was `[-0.010491, +0.001766]`; the paired subject difference CI was `[-0.012075, +0.004820]`. Both cross zero. At the case level, 150 improved, 119 worsened, and one tied; at the subject level, 65 improved and 54 worsened. The positive median but negative mean indicates that several larger regressions outweighed many small gains. Point transitions were 228 hit→hit, 23 hit→miss, 8 miss→hit, and 11 miss→miss.

The true-160³ epoch200 model therefore did not improve the reused internal test point estimate over true-128³ epoch168. Cross-resolution Dice is not perfectly apples-to-apples because nearest-exact labels were rasterized onto different 128³ and 160³ grids, but the point-hit degradation is not explained by Dice rasterization alone.

No checkpoint, threshold, scale, or postprocessing tuning will be triggered by this test result.

## Files

- Absolute report: `workspace/BraTS/sammed3d_t1ce_wt/plain_resolution_20260818/full756_160_seed20260829_epoch200/FINAL_TEST270_TRUE160_EPOCH200.json`
- Paired report: `workspace/BraTS/sammed3d_t1ce_wt/plain_resolution_20260818/full756_160_seed20260829_epoch200/TEST270_TRUE160_EPOCH200_VS_TRUE128_EPOCH168.json`
- Machine summary: `archive/research_records/2026-08-18_plain_resolution_160/TEST270_RESULT.json`
- Pre-access receipt: `archive/research_records/2026-08-18_plain_resolution_160/TEST270_TRUE160_EPOCH200_RECEIPT.json`

Access record: outer selector false, confirmation false, test true by explicit user authorization.
