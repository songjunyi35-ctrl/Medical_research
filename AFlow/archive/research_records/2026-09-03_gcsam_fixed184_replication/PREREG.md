# GCSAM fixed-epoch184 two-replication preregistration

Status: frozen before split generation, implementation, model construction, training, or new train-only inner access

## Objective and evidence firewall

- Objective: replicate the already-frozen `rho=0.05`, non-adaptive, paper-norm GCSAM versus AdamW on two new subject-disjoint train-only splits and paired seeds.
- This is post-test/post-external development. Reused test270 and UCSD External75 have been accessed by historical checkpoints and are permanently closed for selection and evaluation here.
- Existing GCSAM test/external outcomes motivate replication only. They do not choose optimizer hyperparameters, epoch, split, model, threshold, preprocessing, or gates.
- No fixed validation, outer, confirmation, test270, UCSD75, RHUH NIfTI, or MU NIfTI path is accepted by Stage A entry points.

## Frozen data and artifacts

- Full972 train-role manifest: `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl`, 972 scans / 444 subjects, SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- Brain-crop train cache index: `workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json`, SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.
- Native WT volume artifact: `archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/NATIVE_WT_VOLUMES.json`, SHA-256 `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`.
- Historical teacher checkpoint SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`.
- EMA teacher checkpoint SHA-256 `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`.
- Current internal/external reports are archived context only and must not be read by split/training/evaluation code except the final result writer after Stage A decision is frozen.

## Frozen replications

- Replicate A: split seed `20261101`, exact 108 inner scans, remaining 864 fit scans; paired train seed `20261102`; bootstrap seed `20261105`.
- Replicate B: split seed `20261103`, exact 108 inner scans, remaining 864 fit scans; paired train seed `20261104`; bootstrap seed `20261106`.
- All four arms train before any inner metric access. Training reads only its fit role and split identity metadata; it does not construct or evaluate the inner dataset.
- Both arms stop at exact global epoch184. No checkpoint/epoch selection exists; only epoch184 checkpoint is saved.

## Frozen method

- Arms: `adamw_control` and `gcsam_candidate`.
- Candidate is the unchanged historical paper-form GCSAM: `rho=0.05`, non-adaptive; centralize every parameter gradient with `ndim>1` across dims `1..N-1`; normalize the centralized global L2 norm; reuse the same teacher logits and replay CPU/CUDA RNG for the second student pass; exact restore before the AdamW update.
- Model: plain `AutomaticPromptLocator`, init filters16, dropout0.1, 4,700,897 parameters.
- Teacher: `0.75*history168 raw logits + 0.25*EMA168 raw logits`, no teacher gradient.
- Loss: MONAI sigmoid DiceCE `lambda_dice=1`, `lambda_ce=0.1` plus binary voxel KL weight0.3, temperature2.
- Sampling: frozen native WT volume weights, replacement, 972 requested / 968 consumed, 121 optimizer steps/epoch, batch8.
- Augmentation: historical `max_shift=8` path unchanged.
- LR: AdamW base `2e-4`, weight decay `1e-5`; epochs1–100 constant then the historical 200-epoch denominator cosine toward`1e-5`; training stops at184.
- Gradient clip5 on the update gradient.
- Output: raw segmentation logit `>0`, point=argmax logit, no TTA, SAM, connected components, postprocessing, ensemble, or threshold search; deployment1×.

## Engineering gates

- New isolated split, training, paired runner, evaluator, and finalizer entry points; historical hash-bound files remain unchanged.
- Synthetic tests: exact split counts/no subject overlap; forbidden-role path rejection; fixed epoch184/no inner loader in training; GCSAM restore/RNG/teacher contract; paired initial/sampling/first-batch equality; evaluator requires all four completed checkpoints before access; bootstrap and gate direction.
- CUDA smoke for both arms on each replication or a contract-equivalent representative pair before formal.
- Within each replicate: exact initial model state SHA, native weight SHA, every epoch sampler index SHA, and first augmented image/label SHA.
- Checkpoint model-state SHA and strict reload exact; teacher gradients absent; loss/gradient/parameters finite; GCSAM perturbation norm within`2e-6` of0.05.
- Peak allocated ≤24GiB. Each control≤6h; each candidate≤9h. Disk addition<300MB; no new cache.

## Frozen analysis and gates

- Each inner is opened once only after four formal training reports/checkpoints exist and their hashes are frozen in `INNER_PREACCESS.json`.
- Metrics: Scan Dice equal-weight over108; Subject Dice mean within subject then subjects equal; Point hit over108. Threshold0.
- Bootstrap: 10,000 percentile resamples; case-paired for Scan, subject-paired for Subject; seeds20261105/20261106.
- Per replicate gates: Candidate−Control Scan>0; Subject>0; Point hits delta≥-1; Scan and Subject paired 95% CI lower bounds≥-0.003.
- Cross-replication gates: arithmetic mean of the two Scan deltas>0 and mean of Subject deltas>0; all pairing/engineering/access gates pass.
- Any failure stops before Full972/fixed validation and freezes a negative replication result. No post-result changes or third replication.
- If and only if every Stage A gate passes, Stage B is authorized with fresh seed20261107 and requires a separate pre-validation receipt. Stage B details and gates are those in IDEA.md; it never authorizes test270/UCSD75.

## Expected cost and archival

- Stage A approximately21–22 GPU-hours / 10–11h dual-GPU wall, <300MB, inference unchanged1×.
- Save raw reports, `INNER_PREACCESS.json`, decision JSON, checkpoint/report hashes, `RESULT.md` and machine JSON; update current handoff and method catalog at every status transition.
- No leaderboard update unless a named test cohort is run; this experiment forbids all named test/external cohorts.

