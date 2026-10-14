# Plain locator subject-clean confirmation result — 2026-08-17

## Outcome first

The plain image-only locator protocol replicated on a newly frozen subject-clean cohort and passed all preregistered confirmation gates.

| role / metric | scan Dice | subject Dice | automatic point hit |
|---|---:|---:|---:|
| 108-case selector, selected epoch 99 | `0.720419` | `0.704617` | `0.935185` |
| 108-case untouched confirmation | **`0.716505`** | **`0.726813`** | **`0.925926`** |

Confirmation percentile-bootstrap 95% intervals were `[0.692522, 0.739225]` for scan-macro Dice and `[0.700117, 0.751749]` for subject-balanced Dice. The preregistered gates were scan Dice at least `0.68`, subject Dice at least `0.67`, and point hit at least `0.85`; all three passed.

This is strong internal protocol-replication evidence for the plain locator. It is not external-data confirmation, not an evaluation of the exact historical checkpoint, and not a test result.

## Isolation and frozen protocol

- The source was only the historical 972-case training manifest, SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- A split frozen before training assigned 756 cases / 346 subjects to train, 108 / 45 to selector, and 108 / 53 to confirmation.
- Every role pair had zero case overlap and zero subject overlap.
- The historical 108-case validation images/labels were not opened by this experiment.
- The revealed 270-case test manifest and NIfTI files were not opened.
- The old plain checkpoint was not evaluated on the new confirmation role because it trained on all 972 source cases and would leak those cases.
- The new checkpoint was selected only by selector scan Dice at the fixed raw-logit threshold 0. Confirmation was opened once and was not used for threshold, component, TTA, epoch, seed, or postprocessing selection.

Pre-registration: `PLAIN_LOCATOR_CONFIRMATION_PREREG_20260817.md`, SHA-256 `4a15f0ca273307349e0ee5943a83576aeaff322f34304e0af33a628a49820f1f`.

## Persistent cache recovery

The prior CAPG brain-crop cache lived under `/tmp` and was absent on the new H100 worker. It was rebuilt from the 972-case train manifest into persistent workspace storage with 48 CPU workers:

`workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train`

- preprocessing: N4 shrink 2, image-nonzero bounding-box crop, resize to `128^3`; `uses_gt_for_localization=false`;
- 972/972 cases completed in `2534.37 s`;
- images SHA-256 `f772520ca0d99fca0edc60aa04a7c73e0d7818eea9fa0174a29038f8334644b9`;
- labels SHA-256 `bf470d48a5f115f09407ad351bb6dee11ee36bdd0c2a4285ae0bd9cc22125724`;
- both hashes exactly match the 2026-08-14 byte-identical build;
- four independent single-case N4/geometry/label-mass checks passed;
- index SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.

## Training and strict reload

The model was the unchanged 4,700,897-parameter plain MONAI SegResNet locator: T1ce input, one WT logit, no CAPG centre head, and no SAM cascade. The CE weight remained `0.1`; batch size remained 8. One NVIDIA H100 80GB ran 100 epochs in `2209.94 s` with `20.07 GiB` peak allocated memory.

Selector best was epoch 99:

- scan Dice `0.7204192918`;
- subject Dice `0.7046169334`;
- automatic point hit `0.9351851852`.

An independent checkpoint reload reproduced the complete selector report exactly, including every per-case value; scan difference was exactly 0. The checkpoint SHA-256 is `0bb7ca52d7ac30068f752a4fc05a859f4dce42dce6ace40ba1f6b8b332889c2f`.

## One-time confirmation

The fixed epoch-99 checkpoint was then evaluated once on the sealed confirmation role with raw logit `>0` and no postprocessing:

- cases / subjects: 108 / 53;
- scan Dice mean / median / population SD: `0.716505 / 0.749297 / 0.122224`;
- subject-balanced Dice: `0.726813`;
- automatic segmentation-argmax point hit: `0.925926`;
- case and subject overlap with checkpoint train/selector roles: 0 / 0;
- all preregistered gates: passed.

The historical 108-validation plain result was scan/subject `0.716125/0.706215`. The new confirmation differs by approximately `+0.000380/+0.020598`, but these are different cohorts and the comparison is not paired. It supports reproducibility; it does not establish a Dice improvement.

## Scientific interpretation

1. The original `0.716125` plain-locator development result was not a split-specific accident: a separately trained model reached `0.716505` on subjects excluded from its train and selector roles.
2. The direct plain locator remains the current automatic candidate. CAPG's centre head and the old frozen-SAM postprocessor remain negative results; nothing here rehabilitates either path.
3. The evidence is internal because all roles come from the same BraTS 2024 source cohort. An external dataset is still required for a generalization claim.
4. Decoder adaptation remains scientifically reasonable, but the newly revealed confirmation role must not become a tuning set. Develop locator-conditioned dense-prompt decoder training on train/selector or a newly frozen inner split, then use genuinely new external data for final confirmation.

## Evidence and hashes

| artifact | SHA-256 |
|---|---|
| split manifest `data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1/split_manifest.json` | `934798b76aa17a5a82c32270d4ecbd2d03fe52ec293b30c381d464c766cb2555` |
| train receipt `workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/plain_seed20260819/TRAIN_DONE.json` | `8cb7aadd2dc18ba7358de738a96a2f9eea035ef370da44969c5959729caa1bf5` |
| selector reload `workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/SELECTOR_RELOAD_VALIDATION.json` | `9d1afe54a313a3cab370b56207cee9c0cf97d5bebbe4864a7d573a28c1983ee1` |
| confirmation start receipt `workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/FINAL_CONFIRMATION.json.started.json` | `9bb9992f8548ff338266c0d6e199e725a0b8c484aa3230ad9c458bd2fa47de48` |
| final confirmation `workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/FINAL_CONFIRMATION.json` | `1cec25a2a1e603b1403dbf2d430b992eabc35aeb5353e0191334ce33fbf75e14` |

New role-aware dataset support rejects missing cases, duplicate cases, case overlap, and subject overlap. The repository regression after implementation was `254 passed, 24 warnings`; warnings were existing SciPy deprecations.
