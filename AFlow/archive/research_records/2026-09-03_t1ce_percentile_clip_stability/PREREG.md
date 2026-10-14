# T1ce fixed 1–99 percentile clipping stability preregistration

Status: frozen before implementation, cache construction, split generation, model construction, training, or new train-only inner access

## Objective and evidence firewall

- Objective: isolate deterministic per-case brain-mask 1st/99th percentile clipping before z-score as the only preprocessing difference versus the current T1ce control.
- This is post-test/post-external development. Fixed validation108, outer, confirmation, reused test270, and UCSD External75 are already exposed and forbidden to all entry points in this experiment.
- Existing validation/test/external results motivate the bottleneck only; they do not choose clip percentiles, ordering, epoch, split, checkpoint, threshold, model, or gates.
- RHUH/MU NIfTI and any external asset are also forbidden. Stage A uses only the 972-scan BraTS train role and two new subject-disjoint inner splits.

## Frozen data and artifacts

- Full972 train-role manifest: `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl`, 972 scans/444 subjects, SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- Control brain-crop train cache index: `workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json`, SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.
- Candidate cache location: `/tmp/aflow_t1ce_percentile_clip_20260903/train`; this task-created, raw-rebuildable cache may disappear with `/tmp`, while code/index hash and all scientific reports remain archived.
- Native WT volume artifact SHA-256 `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`.
- Historical teacher checkpoint SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`.
- EMA teacher checkpoint SHA-256 `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`.
- IDEA SHA-256 `7504085184ea0e17d97268e9233ff4eb857f9f80c1b49e301aa1534416a24b0a`.
- New split/cache/train/evaluator scripts and their hashes will be frozen in `IMPLEMENTATION_RECEIPT.json` before cache construction or model construction. Historical hash-bound scripts remain unchanged.

## Frozen Stage A replications

- Replicate A: split seed `20261111`; exact 108 inner scans and remaining 864 fit scans; paired train seed `20261112`; bootstrap seed `20261115`.
- Replicate B: split seed `20261113`; exact 108 inner scans and remaining 864 fit scans; paired train seed `20261114`; bootstrap seed `20261116`.
- Both arms in both replications train to fixed global epoch168 before either inner metric is opened. Training never constructs an inner dataset or saves an inner-selected checkpoint.
- All four fixed-epoch checkpoints and training reports must exist, pass gates, and be frozen in `INNER_PREACCESS.json` before the evaluator can open either inner.

## Frozen preprocessing and method

- Shared order: canonical RAS → N4 shrink2 with 4×50 iterations and nonzero mask → image-nonzero bounding-box crop → arm transform → trilinear image/nearest-exact label resize to128³.
- Control arm transform: source nonzero population z-score.
- Candidate arm transform: NumPy linear percentile method at fixed 1.0 and99.0 over finite nonzero source voxels; clamp all brain voxels to those thresholds; compute population mean/std after clamp; z-score clamped nonzero voxels; background remains0.
- Candidate labels, case ordering, crop geometry, source/resized WT-positive counts and `labels.u8` must be byte-exact to Control. Candidate images must be finite and differ from Control.
- Model: plain `AutomaticPromptLocator`, init_filters16, dropout0.1, 4,700,897 parameters.
- Teacher: `0.75*history168 raw logits + 0.25*EMA168 raw logits`, no teacher gradient; each arm's teacher consumes that arm's image.
- Loss: MONAI sigmoid DiceCE `lambda_dice=1`, `lambda_ce=0.1` plus binary voxel KL weight0.3, temperature2.
- Sampling: frozen native WT-volume weights, replacement, 972 requested/968 consumed, 121 steps/epoch, batch8.
- Historical flip/shift/scale/offset/noise augmentation unchanged and driven by paired RNG.
- AdamW base LR`2e-4`, weight decay`1e-5`; epochs1–100 constant then historical 200-epoch-denominator cosine toward`1e-5`; exact stop at epoch168; gradient clip5.
- Output raw logit`>0`, point=argmax logit; no TTA, Segment Anything Model, connected components, postprocessing, ensemble, or threshold search; deployment1×.

## Engineering gates

- Synthetic tests must cover: fixed percentile definition/order; background0; finite output; constant/invalid source rejection; candidate label byte equality; source/candidate cache schema; exact split counts/no subject overlap; forbidden-role path rejection; fixed epoch168/no inner loader; paired initial/sampling/sampler/label/augmentation-RNG contract; evaluator requires four complete reports/checkpoints before access; bootstrap/gate direction.
- GPU smoke runs both arms with the same replicate/seed for two steps. Initial state, sampling weights, sampler indices, post-augmentation labels, and augmentation draw signature must be exact. Images are expected to differ only because preprocessing differs.
- Formal pairing per replicate requires exact initial state, native sampling weight, all168 epoch sampler indices, all epoch label-batch hashes, all augmentation parameter hashes, and exact checkpoint reload.
- Losses/gradients/parameters finite; teacher receives no gradient; parameter count exact; peak allocated≤24GiB; each arm≤6h.
- Candidate cache build requires source manifest/hash/count, exact N4/crop/percentiles/normalization metadata, all thresholds finite with low<high, labels SHA exact to Control, independent sample rebuild byte equality, and no evaluation-role access.
- Repository disk addition<300MB; candidate cache resides only on `/tmp`, whose pre-build free space is554GB.

## Frozen Stage A analysis and gates

- Metrics: Scan Dice over108 equal-weight cases; Subject Dice averages within subject then across subjects; Point hit over108; threshold0.
- Bootstrap: 10,000 percentile resamples; case-paired Scan and subject-paired Subject; seeds`20261115/20261116`.
- Per replication scientific gates: Candidate−Control Scan>0; Subject>0; Point hits delta≥−1; Scan and Subject paired95% CI lower bounds≥`-0.003`.
- Cross-replication gates: arithmetic mean of the two Scan deltas>0 and mean Subject deltas>0; all cache/pairing/engineering/access gates pass.
- Any failure stops the method: no third seed, no percentile/order/epoch/threshold changes, no Full972, and no access to fixed validation/outer/confirmation/test270/UCSD75.

## Conditional Stage B

- If and only if every Stage A gate passes, a separate pre-training receipt may authorize one paired Full972 Control/Candidate run from the same fresh initial state, seed`20261117`, to fixed epoch168.
- Stage B performs training only and freezes both checkpoints. It does not access fixed validation, outer, confirmation, test270, UCSD75, RHUH, or MU. The candidate waits for a future independent expert-label cohort.

## Expected cost and archival

- Stage A four runs: approximately13.6 GPU-hours /6.8h dual-GPU wall; Stage B if authorized adds approximately6.8 GPU-hours/3.4h wall.
- Candidate cache approximately9.5GB on `/tmp`; persistent workspace checkpoints/reports<300MB; no deletion of existing material.
- Archive cache index/hash, split manifest/hash, implementation/smoke/launch/preaccess receipts, raw inner result, decision, `RESULT.md`, and machine JSON. Update the method catalog and current handoff at proposal/start/finish.
- No leaderboard update because every named test/external cohort is forbidden.
