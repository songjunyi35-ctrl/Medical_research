# UCSD-PTGBM patient-disjoint external confirmation preregistration

Status: frozen before downloading or reading any eligible external segmentation voxel

## Evidence roles and access boundary

- Existing 1,080-case train manifest: `data/datasets/brats2024_random_subject_80_20_seed20260722_v1/train_cases.jsonl`, SHA-256 `aa2e55915e2a4bc7c2af2e95d6e08f2199397694fe4a92ea66999b534a22f63d`.
- Existing reused internal test270 manifest: `data/datasets/brats2024_random_subject_80_20_seed20260722_v1/sealed_test/test_cases.jsonl`, SHA-256 `7b91690c63a5eb359be105b3cc7c4ad563d41fca2b2c84e521251b7ed006b8d2`.
- These two manifests union to 1,350 scans and define every previously exposed BraTS patient for this audit. Fixed validation, outer, confirmation, and every train-only inner role are subsets or views of this union.
- External metadata access before freeze: official release filenames, sizes, SHA-256 digests, URLs, and the official `participants.tsv` mapping have been read. Eligible image voxel values and segmentation voxel labels have not been downloaded/read.
- Eligible external cohort is a one-time confirmation/test role. It is never train, inner tune, validation, threshold selection, checkpoint selection, model selection, or postprocessing selection data.
- The separate official `BraTS-GLI 2024 Test Data` package is excluded. Only the official GitHub release IDs/tags below are accepted.

## Frozen official source identity

- Repository: `data-nih/tcia`.
- Accepted release `(tag, release_id)` pairs:
  - `ucsd-ptgbm`, `302631741`
  - `ucsd-ptgbm_1`, `302636016`
  - `ucsd-ptgbm_2`, `302639054`
  - `ucsd-ptgbm_3`, `302642815`
- `ucsd-ptgbm_4` contains no required pair and is not needed.
- Required image regex: `^sub-\d{4}_ses-\d{2}_run-post_T1w\.nii\.gz$`.
- Required label regex: `^sub-\d{4}_ses-\d{2}_BraTS_tumor_seg\.nii\.gz$`.
- `participants.tsv` official asset SHA-256: `718854757eba9332f0e2872e93b88809987aff204a3d01350e138f8ae0a5960e`.
- Expected official metadata: 178 UCSD subjects / 243 timepoints in participants; main releases 136 subjects / 184 paired studies.
- Expected patient-disjoint eligible cohort after subtracting all 1,350 existing BraTS patients: 53 subjects / 75 scans; main-release case overlap count 109, eligible case and subject overlap both 0.

## Frozen checkpoints and evaluation

- Primary: `workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth`, SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`.
- Secondary: `workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth`, SHA-256 `acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`.
- Model: existing 4,700,897-parameter plain SegResNet locator.
- Input: T1 post-contrast only; no other UCSD modality is downloaded or used.
- Target: WT=`{1,2,3}`; RC=`{4}` excluded.
- Preprocess: canonical RAS → N4 shrink2 → image-positive bounding box → source-positive population z-score → image trilinear / target nearest-exact resize to 128³.
- Prediction: raw logits `>0`, no test-time augmentation, no Segment Anything Model, no postprocessing, no ensemble, 1× forward/model.
- Automatic point: argmax raw logit; point hit denominator is only source-WT-positive scans. Empty-WT scans are retained and separately report predicted-empty and false-positive voxels.
- Empty-set Dice: 1 only when both target and prediction are empty; otherwise standard `2TP/(|pred|+|target|)`.
- Metrics and gates are exactly those in `IDEA.md`; bootstrap 10,000, seed `2026090201`.
- External results will not choose between the models for a subsequent run. Both are reported as independently frozen historical methods.

## Frozen stop and archival contract

1. Build and test a new isolated inventory/downloader/cache/evaluator; do not modify hash-bound historical entry points.
2. Before eligible label download, write `PREACCESS_RECEIPT.json` with hashes of this document, IDEA, source inventory, participants, manifests, scripts/tests, and checkpoints.
3. Hash every downloaded file against the official asset digest before reading NIfTI.
4. Stop before model inference if any expected count, overlap, name, digest, NIfTI geometry, finite, or label-domain gate fails.
5. Save raw model report and paired analysis, recompute all SHA-256, write `RESULT.md` and `RESULT.json`, and update current handoff/method catalog.
6. Because this is a named external confirmation cohort, update both leaderboard Markdown and JSON in a separate External section and run all three repository validators.
7. No result-dependent rerun, threshold/crop/N4/label/case change, ensemble, or extra checkpoint evaluation is authorized.

