# CAPG-3D automatic localization and prompt result — 2026-08-14

## Outcome first

This experiment solved an important protocol gap but did **not** validate the
proposed centre-head architecture as a Dice improvement.

- A fully image-only 3D locator reached validation scan/subject Dice
  `0.716125/0.706215` (plain) and `0.719090/0.710459` (CAPG).
- CAPG minus plain locator Dice was only `+0.002965` scan and `+0.004243`
  subject; the paired subject-bootstrap 95% CI was
  `[-0.003736,+0.013624]`.
- The preregistered primary endpoint, frozen-SAM point+dense Dice, was
  `0.599324/0.601878` for plain and `0.597910/0.601175` for CAPG
  (scan/subject). CAPG minus plain was `-0.001414/-0.000703`, subject 95% CI
  `[-0.004178,+0.002713]`. It failed the `+0.005` success threshold.
- Automatic point hit rate was `0.935185` for the plain segmentation-logit
  argmax but only `0.638889` for the CAPG centre-heatmap argmax. This is the
  direct mechanism for the failed prompt claim.

Therefore the defensible progress is a strong, lightweight **automatic
brain-crop locator** development result, not a positive CAPG/SAM architecture
claim and not a test result.

## Why this experiment was needed

The historical brain-crop score `0.625359` was often described informally as
automatic, but only its crop was image-derived. Its first point was sampled
from GT foreground and subsequent corrections from GT/prediction error. The
new path uses GT only as a training target and as a post-prediction metric. At
validation inference the locator receives T1ce alone and produces its own mask
and point.

## Data contract and exact cache recovery

- 972 train / 108 validation, subject-clean; the 270-case test was never read.
- T1ce only, WT=`{1,2,3}`, label 4 excluded.
- N4, nonzero-image bounding-box crop, resize to `128^3`; the crop placement
  never uses the label.
- Raw data came from the active VA/HDFS storage manifest after the historical
  `/mnt/bn/liveaim` path was found unavailable.
- The rebuilt cache matched the 2026-08-06 historical cache byte-for-byte:

| split/file | SHA-256 |
|---|---|
| train images | `f772520ca0d99fca0edc60aa04a7c73e0d7818eea9fa0174a29038f8334644b9` |
| train labels | `bf470d48a5f115f09407ad351bb6dee11ee36bdd0c2a4285ae0bd9cc22125724` |
| validation images | `19107e84aba65c5c257f6a50273547e1b74e761d72e0487ad6f48a5be16d4bad` |
| validation labels | `6ac3a3909f157eaa629c0fca0dbfb30bd78c23206317c62be2376443ce469e0e` |

Both split builds also passed four independent single-case N4/geometry/label
mass reprocess checks.

## Architecture and paired protocol

Both arms use a 4.70M-parameter MONAI SegResNet and identical initialization,
data order, augmentation, optimizer, batch 8, BF16 and seed `20260819`.

- plain: one WT segmentation logit; its global argmax is the positive point.
- CAPG-3D: the same segmentation output plus a centre heatmap trained on a
  Gaussian around the foreground voxel nearest the WT centre of mass; its
  heatmap argmax is the positive point.
- The frozen SAM receives the selected positive point and/or a fixed dense
  prompt `5*tanh(locator_logit/5)` resized to `32^3`. No threshold, component,
  logit-scale or checkpoint search was performed after evaluation.

The first `DiceCE(1:1)` formal attempt was correctly aborted after epoch 6:
both arms approached the all-background solution. Its artifacts and exact
numbers are preserved in `capg3d_pair_20260814/PROTOCOL_ABORT.json`. The
replacement reduced the matched CE weight to `0.1`, after which both arms
trained normally. Because this correction used development behavior, the
result is development evidence, not untouched confirmation.

## Formal results

| output | plain scan | CAPG scan | CAPG−plain scan | plain subject | CAPG subject | CAPG−plain subject |
|---|---:|---:|---:|---:|---:|---:|
| locator | `0.716125` | `0.719090` | `+0.002965` | `0.706215` | `0.710459` | `+0.004243` |
| SAM point-only | `0.000312` | `0.000276` | `-0.000036` | `0.000337` | `0.000288` | `-0.000049` |
| SAM dense-only | `0.596964` | `0.595425` | `-0.001540` | `0.600584` | `0.600045` | `-0.000539` |
| SAM point+dense | `0.599324` | `0.597910` | `-0.001414` | `0.601878` | `0.601175` | `-0.000703` |

The selected locator checkpoints were epoch 99 (plain) and epoch 94 (CAPG).
Same-batch strict reload differences were exactly zero. Batch-1 reporting
differs from embedded batch-8 BF16 thresholding by only `8.85e-6/2.43e-5`.

The frozen SAM is not a useful postprocessor for these strong locator masks:
point+dense drops scan Dice by `0.116801` for plain and `0.121180` for CAPG.
Point-only is near zero because this path uses SAM's no-mask embedding, whereas
the old interactive evaluator always supplied a recurrent zero mask even on
click 1; it must not be compared directly with the old GT-click score.

Training used about 20.1 GiB peak per A100 and completed 100 epochs in
`4273/4363 s`. Plain has 4,700,897 parameters and CAPG 4,700,914; both are
about 21 times smaller than the 100.5M-parameter SAM-Med3D model.

## Scientific interpretation and next step

1. Freeze the simpler plain locator as the current automatic deployment
   candidate. CAPG did not pass its primary endpoint, and its dedicated centre
   point was substantially less reliable than the segmentation argmax.
2. Do not cascade the current locator through the old frozen brain-crop SAM;
   report the direct locator mask. Any future SAM integration should train the
   dense-prompt/decoder path end-to-end against locator-produced prompts.
3. A foreground-constrained centre point is an obvious mechanism correction,
   but it must not be tuned and selected again on these same 108 cases. Freeze
   it before a new subject-clean development or external confirmatory set.
4. Do not evaluate the revealed 270-case test for this iteration. A new
   external/confirmatory set is needed before claiming generalization.

## Evidence

- preregistrations: `CAPG3D_PREREG_20260814.md`,
  `CAPG3D_DICE01_PREREG_20260814.md`;
- model: `sammed3d/t1ce_wt_capg3d.py`;
- trainer/evaluator/analyzer: `scripts/train_t1ce_wt_capg3d.py`,
  `scripts/evaluate_t1ce_wt_capg3d_auto_sam.py`,
  `scripts/analyze_t1ce_wt_capg3d_pair.py`;
- paired report:
  `workspace/BraTS/sammed3d_t1ce_wt/capg3d_dice01_pair_20260814/FINAL_COMPARISON.json`,
  SHA-256 `c08a68124a111b940ce8ba14e60dcb994d64ce4b159436009a3ad6b553055aef`;
- plain checkpoint SHA-256:
  `fee5f0255f821c39e0ea08224291041b2c2af8102876fe9a10544c617777c31d`;
- CAPG checkpoint SHA-256:
  `e2eb903237b351c20d07b08d29bf0bb6de687d2f5935f13fbc791ec71d1b3aa9`.

Final repository regression: `252 passed, 26 warnings`.
