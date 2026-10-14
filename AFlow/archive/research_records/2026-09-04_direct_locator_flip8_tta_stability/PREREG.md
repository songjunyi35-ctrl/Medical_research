# Direct locator fixed flip8 TTA stability preregistration

Status: frozen before split generation, implementation, model construction, training, fresh-inner access, or new test access

## Objective and evidence status

- Objective: determine whether fixed eight-reflection test-time augmentation improves the automatic T1ce/WT direct locator across two fresh train-only replications, then—only if every gate passes—apply the identical formula once to the archived Full972 KD checkpoint on reused internal test270.
- This research occurs after fixed validation108 and test270 have been repeatedly exposed. Stage A is fresh train-only development; any conditional test is `user-authorized post-test development; reused internal test; not pristine blind; not external`.
- No existing test/validation number chooses a reflection subset, weight, threshold, model, or epoch. The complete group of all eight three-axis reflections, equal logit weights, threshold0, and fixed epoch168 are frozen now.
- Fixed validation108, outer, confirmation, UCSD75, RHUH, and MU are forbidden throughout. Test270 is forbidden unless every Stage A gate passes and a separate pre-test receipt is frozen.

## Frozen data and models

- Full972 train-role manifest: `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl`, 972 scans/444 subjects, SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- Train cache index: `workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json`, SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.
- Native WT volume artifact SHA-256 `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`.
- Historical teacher checkpoint SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`.
- EMA teacher checkpoint SHA-256 `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`.
- IDEA SHA-256 `87b53daa6ae91f22d00d16a5b6eef4294b1ee34d12e451060b14417cae4dd095`.
- Conditional test target: archived Full972 random-init KD epoch168 checkpoint SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`; archived identity test report SHA-256 `9827a75b88dc9239902c7e473a7d75e2d6058f4f515bfd2b4d93003ba7bd34cd`.
- Current reused-test Scan leader frozen for success comparison: GCSAM fit864 epoch184=`0.752003176`.

## Frozen Stage A replications

- Replicate A: split seed`20261201`, paired training seed`20261202`, bootstrap seed`20261205`.
- Replicate B: split seed`20261203`, paired training seed`20261204`, bootstrap seed`20261206`.
- Each split has exactly108 inner scans selected subject-disjointly and864 remaining fit scans. The two inner sets must differ; cross-replication overlap is reported, not optimized.
- One standard base model is trained per replication. Both models finish and freeze fixed epoch168 checkpoints before either inner manifest is opened by the evaluator.
- Training entry points accept only fit paths and never construct an inner loader, compute inner metrics, or select a checkpoint.

## Frozen training recipe

- Input/target: T1ce brain-crop true128³; WT raw labels`{1,2,3}`, label4 excluded.
- Model: plain `AutomaticPromptLocator`, init_filters16, dropout0.1, center_aware false, 4,700,897 parameters.
- Frozen teacher raw logits: `0.75*history168 + 0.25*EMA168`; teacher receives no gradient.
- Loss: MONAI sigmoid DiceCE with Dice1/CE0.1 plus binary voxel KL weight0.3, temperature2.
- Native-WT-volume weighted replacement;972 requested/968 consumed/121 steps per epoch; batch8.
- Historical flip/shift/scale/offset/noise augmentation; AdamW base LR`2e-4`, weight decay`1e-5`; epochs1–100 constant then historical 200-denominator cosine toward`1e-5`; exact stop epoch168; gradient clip5.
- Random initial state differs across replications but is frozen by each training seed.

## Frozen inference factor

- Identity Control: one BF16-autocast forward on the canonical128³ image, raw logit threshold`>0`, Point=argmax raw logit.
- Flip8 Candidate axes in this exact order: `()`, `(-1,)`, `(-2,)`, `(-2,-1)`, `(-3,)`, `(-3,-1)`, `(-3,-2)`, `(-3,-2,-1)`.
- For each view, flip the input, run the same checkpoint once, inverse-flip the output along the same axes, convert to float32, sum in the frozen order, and divide by8.
- Candidate mask is mean raw logit`>0`; Point=argmax mean raw logit.
- Identity Control is exactly the first Flip8 atom, so evaluation uses8 forwards total per case, not9.
- No reflection subset/weight search, no probability averaging, no threshold change, no TTA stochasticity, no SAM, connected components, morphology, calibration, ensemble, or postprocessing.

## Engineering gates

- New isolated split/training/evaluator/preaccess/test entry points; no historical hash-bound script is modified.
- Synthetic tests: eight combinations exact/order/uniqueness; flip→inverse identity; float32 logit mean; identity atom reuse; threshold/Point definitions; paired bootstrap/gate directions; split counts/no subject overlap; forbidden-role path rejection; training fixed168/no inner loader; evaluator requires both completed checkpoints and frozen preaccess.
- GPU smoke: one checkpoint over synthetic/real train-only batch must show identity atom bit-exact to direct identity, flip/inverse shapes exact, all outputs finite, and sequential peak≤24GiB.
- Formal training:168 epochs,121 steps/epoch,968 consumed draws, finite stats, reload exact, teacher no gradients, peak≤24GiB, elapsed≤6h/model, inner metrics absent.
- Evaluator writes a started receipt before constructing either inner dataset, refuses repeat output, and verifies split/checkpoint/cache/evaluator hashes.

## Frozen Stage A analysis and promotion

- Metrics: Scan Dice over108 equal-weight cases; Subject Dice mean within subject then equal-weight subjects; Point hit over108.
- Bootstrap:10,000 percentile resamples; case-paired Scan and subject-paired Subject; A/B seeds`20261205/20261206`.
- Per-replication gates: Flip8−Identity Scan>0, Subject>0, Point hits delta≥−1; Scan/Subject paired95% CI lower bounds≥`-0.003`.
- Cross-replication gates: arithmetic mean Scan delta>`0.000243638` and mean Subject delta>0; all engineering/access gates pass.
- Any failure stops before test: no third seed, no single-axis/four-view/weight/probability/threshold/epoch/checkpoint search.

## Conditional reused-test protocol

- Only every Stage A gate passing authorizes a separate pre-test receipt. It binds the archived Full972 checkpoint/report, test270 manifest/cache hashes, evaluator hash, exact Flip8 formula, threshold0, current leader, bootstrap, output path, and one-run limit.
- Test evaluator recomputes identity and Flip8 together and requires identity parity to archived Full972 test metrics/report before publishing Flip8 metrics.
- Test success gates: Flip8 Scan>`0.752003176`; Flip8 Scan>`0.751759537` identity; Flip8 Subject≥identity`0.737327107`; Point hits delta≥−1 from identity262/270. Confidence intervals are reported but point-estimate ranking is not called significant without support.
- Regardless of success, tie, or failure, update `AFLOW_BRATS_LEADERBOARD.md` and `.json`; label reused/post-test/non-blind/non-external and record8×cost.
- Test results cannot trigger any formula, view, weight, threshold, checkpoint, or postprocessing change/retest.

## Cost, storage, and archival

- Stage A training≈6.8 GPU-hours/3.4h dual-A100 wall; Stage A evaluation1,728 forwards. Conditional test adds2,160 forwards.
- Sequential views keep inference peak near1×; deployment latency/compute≈8×.
- Reuse existing cache; persistent additions expected<200MB. No deletion is required for this method.
- Archive split, implementation/smoke/launch/preaccess receipts, raw reports, hashes, `RESULT.md` and machine JSON; update handoff/catalog at each status. If test runs, update both leaderboard files and run all validators.
