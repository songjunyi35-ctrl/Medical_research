# Plain locator subject-clean confirmation preregistration — 2026-08-17

Status: frozen before split construction, training, selector evaluation, or confirmation evaluation.

## Objective and evidence boundary

Replicate the fixed plain automatic locator protocol on a new internal subject-clean split made only from the historical 972-case training cohort. This is an internal protocol replication, not an external-data confirmation and not an evaluation of the exact historical checkpoint, because that checkpoint was trained on all 972 source cases.

The historical 108-case validation cohort is not used. The revealed 270-case test manifest and its NIfTI files are not read. The existing plain locator checkpoint is not evaluated on the new confirmation cases because it has already trained on them.

## Frozen split

- Source: `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl`, 972 cases, SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- Grouping unit: all longitudinal scans of one `subject_id` stay in one role.
- Roles: 756 train / 108 selector / 108 untouched confirmation.
- Selection algorithm: confirmation first by seeded shuffled-subject exact case subset-sum with seed `20260817`; selector second from the remainder with seed `20260818`.
- The split builder must report zero case and subject overlap across all role pairs. Any mismatch aborts.

## Frozen preprocessing and model

- T1ce only; WT is raw labels `{1,2,3}`, raw label 4 excluded.
- N4 with shrink factor 2, image-nonzero bounding-box crop, resize to `128^3`; crop placement is image-only.
- Expected rebuilt 972-case cache hashes from the 2026-08-14 byte-identical build: images `f772520ca0d99fca0edc60aa04a7c73e0d7818eea9fa0174a29038f8334644b9`, labels `bf470d48a5f115f09407ad351bb6dee11ee36bdd0c2a4285ae0bd9cc22125724`. A mismatch aborts before training.
- Model: plain `AutomaticPromptLocator`, MONAI SegResNet, init filters 16, dropout 0.1, one WT logit, approximately 4.70M parameters. No CAPG centre head and no SAM cascade.
- Training seed `20260819`; AdamW LR `2e-4`, weight decay `1e-5`; DiceCE weights Dice 1.0 / CE 0.1; weighted replacement sampling; batch 8; BF16.
- Augmentation is frozen to the prior locator protocol: independent axis flips, joint image/label shift up to 8 voxels per axis, intensity scale/offset, and Gaussian noise.
- Maximum 100 epochs; minimum 30; patience 20. Hardware is one NVIDIA H100 80GB rather than the prior two independent A100 arms.

## Selector and one-time confirmation

- Select exactly one checkpoint by highest 108-case selector scan-macro Dice at raw logit threshold 0.
- Confirmation runs once after training completes. It uses raw segmentation logit `>0`; there is no threshold, component, centre, TTA, or checkpoint search.
- Primary estimates: confirmation scan-macro Dice and subject-balanced Dice. Secondary diagnostic: automatic segmentation-argmax point hit rate.
- Report case and subject percentile-bootstrap 95% intervals with 10,000 resamples and seed `20260817`.
- Preregistered replication gates: scan Dice at least `0.68`, subject Dice at least `0.67`, and point hit rate at least `0.85`. All three must pass for the preregistered success label. Results are reported even if a gate fails.
- Seeing confirmation behavior cannot authorize a CE-weight correction, additional seed, epoch change, threshold change, or postprocessing search under this preregistration.

## Code binding

- Split builder SHA-256: `d11148ed0a0bd203ecc3e769374bd1d3552fc9f3568f5e168fb55ad1952f4c51`.
- Trainer SHA-256: `d74ff2f0cf190c8a5458a70d19125d9a30603894d5f1799f192f0e60dd088fdb`.
- Confirmation evaluator SHA-256: `9e2bf5a8953574489a62db014627c33232555b3bcf2928b4ba00fc6a2980d1b5`.
- Locator model SHA-256: `276ee7c131e413cf7943dc29e5372c1e5bbbb85c780e27f030343c5ef8bf4cd0`.
- Targeted regression at freeze time: `9 passed`.

Any code change affecting split roles, data selection, forward inference, thresholding, aggregation, or success gates requires a new preregistration. Logging-only fixes may be documented but must not change model predictions or role membership.
