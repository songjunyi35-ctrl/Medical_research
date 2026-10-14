# UTSW362 paired canonical/no-N4 external confirmation preregistration

Frozen before download of any UTSW subject outside BT0001/2/3 and before loading any model for UTSW on 2026-09-05.

## Eligibility and precedence

1. Recompute Stage A from archived raw E/F reports; `no_n4.performance_passed` must be true.
2. Require conditional Full972 formal complete with both `canonical` and `no_n4`, same seed20261343, epoch168, paired engineering gates passed.
3. Require `RHUH_PRECEDENCE.json` written only after the already-running RHUH priority queue is terminal. It must contain either an archived RHUH result SHA or an explicit pre-metric close boundary with access flags. UTSW cannot start first.
4. At source-download start require at least4 hours before deadline 2026-09-06T14:02:39Z and at least16GiB free under `/tmp`.
5. Freeze `EXTERNAL_PREACCESS.json` with Stage A raw, Full972 reports/checkpoints, RHUH precedence, source inventory, IDEA/PREREG and implementation hashes before GPU smoke/source download.

## Frozen cohort and source

- Official TCIA package1265, CC BY4.0.
- Official metadata SHA `f7ff889fa7d197b2f589695d3e7ea8a2d1d198c4f9c0044fc71898a08aa24ed0`; full inventory SHA `5f210c0a1e0c33f7b71a56d9bbce8b52a62f24e2a0b53655aa3a098dbde3f871`.
- Cohort: every and only the362 unique IDs whose official TSV field `Manually Refined Segmentation=Yes`; sorted ID-list SHA `e4809dd7f43439309bef71784b3accff95be1a6915e044f46de352e780ec525c`.
- Per subject exactly `brain_t1ce.nii.gz` and `rtumorseg_manual_correction.nii.gz`; 724 files; one-copy bytes1,813,086,599.
- Two fresh public authorizations and two single-transfer `--src-base` copies. Require each local relative path, official listed size, and copy A/B SHA exact for724/724 before any NIfTI load.

## Frozen cache build

- Source must be canonical RAS, image-label shape exact/affine≤1e-6, image finite/min≥0/contains0+positive, label finite/integer/contains0+positive.
- UTSW WT is `label>0`, including label4 ET.
- Shared bbox/z-score/trilinear image/nearest-exact label to128³.
- Canonical cache uses historical positive-mask N4, shrink2/fixed iterations, SimpleITK threads1 per worker; no-N4 cache only sets nonpositive to0.
- Counts362; image dtype float32, label uint8; each cache3,795,845,120 bytes.
- Require both cache labels byte-identical and per-case geometry identical. Fixed BT0001/2/3 output hashes must equal archived no-N4 QC `7573bed6...` and canonical QC `1cec4ba9...` as applicable.

## Frozen models and inference

- Models: conditional Full972 `formal/canonical` and `formal/no_n4`; seed20261343; epoch168; same architecture4,700,897 parameters.
- Each model reads only its matching cache.
- Views: identity and Flip8 axes `[(),(-1),(-2),(-2,-1),(-3),(-3,-1),(-3,-2),(-3,-2,-1)]`; average raw logits; threshold `>0`; no SAM/postprocessing.
- Total forward accounting: 6516 single-example forwards. Batch size8, num_workers0, BF16 autocast; one idle GPU after synthetic dual-model smoke.

## Frozen metrics and gates

- Cohort:362 scans/362 subjects, so Scan and Subject Dice are numerically equal but both fields remain explicit.
- Report per arm/view Scan Dice, Subject Dice, Point hits/rate, predicted/target voxels; per-case Dice/Point/prediction hash.
- Patient paired bootstrap10,000, seed20261381 for no-N4−canonical deltas and absolute intervals.
- Support: Flip8 Scan delta>0, delta95% CI lower≥−0.003, Point hits no-N4≥canonical−3. Identity is secondary.
- Report regardless of gate; no repeat/tuning or old-cohort access. Update external/cross-stage leaderboard section and canonical docs after any completed run, labeling preoperative/SRI24 and rasterization caveats.

## Stop and cost

- Any eligibility/source/cache/smoke/hash/forward-count gate failure stops at that boundary and is archived.
- No case filtering, file version, label mapping, N4, threshold, view, model, epoch, bootstrap or gate change.
- No old validation/outer/confirmation/test270/UCSD/RHUH metric reuse for selection.

