# Flip-equivariance consistency training stability preregistration

Status: frozen before split generation, implementation, model construction, training, or fresh-inner access

## Objective and evidence boundary

- Objective: determine whether a small final-mask transformation-consistency loss can make a direct locator more accurate in both identity 1× and fixed Flip8 8× inference across two new paired replications.
- This is post-test development. Method selection is based on the two fresh Stage A Flip8 replications completed before test access, primary papers, and a train-only scale audit. No validation/test/external metric selects the loss form, weight, split, seed, epoch, view schedule, threshold, or gates.
- Fixed validation108, outer, confirmation, reused internal test270, UCSD75, RHUH, MU, and all other revealed evaluation cohorts are forbidden. No result from this experiment authorizes returning to test270.

## Frozen documents and inputs

- IDEA SHA-256: `a71b75ff8c7b775fc0606141c5b1c6258aeaf4670ec990c819dedb13f85e1175`.
- Train-only scale audit SHA-256: `88af69c0855728db60f937efc79b565b8a117ef6d86be74efa6acc5e86e5b4a4`.
- Full972 train manifest: `data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl`, 972 scans/444 subjects, SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`.
- Train cache index: `workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json`, SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.
- Native WT artifact SHA-256: `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`.
- Historical teacher SHA-256: `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`; EMA teacher SHA-256: `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`.
- Previously accessed Flip8 inner manifests used only to exclude subjects: A SHA-256 `a96abf516ae8f69dd02bc12906c0a381b7fafd991073fe500d3d49de36577e81`; B SHA-256 `0d85256e43d1de2ee6b71587a210f193af8bb50be2a9d378ab1f4b59ceee5b88`. Their metrics are not inputs to the new evaluator.

## Frozen fresh splits and seeds

- Replicate C: split seed`20261211`, paired training seed`20261212`, bootstrap seed`20261215`.
- Replicate D: split seed`20261213`, paired training seed`20261214`, bootstrap seed`20261216`.
- Candidate inner subjects are drawn only from Full972 subjects absent from both old A/B inner manifests.
- C inner is a deterministic seeded subject subset totaling exactly108 scans. D inner is drawn from the remaining eligible subjects, so C/D inner case and subject overlap are both zero.
- Each replicate fit is the exact Full972 complement of its own inner:864 scans, subject-disjoint from its inner. Subjects assigned to the other replicate's inner may appear in fit; within-replicate leakage is forbidden.
- Subject selection uses a deterministic seeded shuffle followed by subset-sum dynamic programming to reach108 scans; first discovered predecessor per subtotal is frozen. No label volume, model metric, test identity, or difficulty score enters selection.

## Frozen arms and training

- Arms per replicate: `control_w0` and `candidate_w01`.
- Both arms use identical seed, initial state, sampler indices, historical augmentation RNG, two student forwards, reflection cycle, teacher outputs, optimizer, and schedule.
- Input/target: T1ce image-only brain crop true128³; WT raw labels`{1,2,3}`, label4 excluded.
- Model: `AutomaticPromptLocator`, init_filters16, dropout0.1, center-aware false,4,700,897 parameters.
- Existing loss on the first/original branch: MONAI sigmoid DiceCE with Dice1/CE0.1 plus binary voxel KL knowledge distillation weight0.3, temperature2, from frozen `0.75*history168+0.25*EMA168` raw logits.
- Native-volume weighted replacement:972 requested/968 consumed/121 optimizer steps per epoch, batch8.
- Historical flip/shift/scale/offset/noise augmentation; AdamW LR`2e-4`, weight decay`1e-5`; epochs1–100 constant then historical denominator200 cosine toward`1e-5`; fixed stop epoch168; gradient clip5.
- After historical augmentation, both arms run the original batch and one reflected batch through the same student in training mode. The second raw logit is inverse-reflected before consistency computation.
- Seven nonidentity reflection axes in exact cycle: `(-1,)`, `(-2,)`, `(-2,-1)`, `(-3,)`, `(-3,-1)`, `(-3,-2)`, `(-3,-2,-1)`. Zero-based global optimizer step chooses `axes[step % 7]`; no reflection RNG.
- Per-volume consistency is `1-(2*sum(sigmoid(z1)*sigmoid(invflip(z2)))+1)/(sum(sigmoid(z1))+sum(sigmoid(invflip(z2)))+1)`, then batch mean.
- Control total loss=`existing_loss+0.0*consistency`; Candidate=`existing_loss+0.1*consistency`. Both retain/backpropagate the second graph, so the single experimental factor is consistency gradient weight.
- Teacher receives only the original augmented branch and no gradient. No EMA/student teacher, pseudo-label, feature projection, contrastive term, or unlabeled data is added.

## Frozen inference and analysis

- Every completed arm is evaluated twice in one pass per case: identity is the first Flip8 atom; fixed Flip8 axes are `()`, `(-1,)`, `(-2,)`, `(-2,-1)`, `(-3,)`, `(-3,-1)`, `(-3,-2)`, `(-3,-2,-1)`.
- Each view is inverse-reflected; raw logits are accumulated in this order in float32 and divided by8. Identity/Flip8 threshold is0; Point is argmax of the corresponding raw logit.
- No view subset, weighting, probability averaging, threshold, checkpoint, Test-Time Augmentation formula, Segment Anything Model, connected component, morphology, calibration, ensemble, or postprocessing search.
- Four fixed epoch168 checkpoints must complete and freeze before either C/D inner manifest is constructed by an evaluator.
- Metrics per108-case inner: Scan Dice, Subject Dice, Point hits. Bootstrap:10,000 percentile resamples, case-paired Scan and subject-paired Subject, seeds`20261215/20261216`.

## Frozen gates

- Engineering gates: within each pair exact initial model-state SHA, sampling-weight SHA, all168 sampler-index hashes, augmentation RNG-before/after hashes, first augmented batch tensors, reflection axes by step, two-forward count, first existing loss; finite stats;121 steps/968 draws each epoch; teacher no gradients; reload exact; no inner access; peak≤60GiB; elapsed≤9h/model.
- Per-replication primary gates on Candidate Flip8−Control Flip8: Scan>0, Subject>0, Point hits≥−1, paired Scan CI lower≥`-0.003`, paired Subject CI lower≥`-0.003`.
- Per-replication 1× gates on Candidate identity−Control identity: Scan>0, Subject>0, Point hits≥−1.
- Mechanism gate per replication: arithmetic mean of Candidate train-only consistency over epochs164–168 ≤`0.90` times the corresponding Control mean.
- Cross-replication gates: arithmetic mean Candidate−Control Flip8 Scan>`0.001`; mean Flip8 Subject>0; mean identity Scan>0; mean identity Subject>0; all engineering/access gates pass.
- Any failed gate closes the implementation: no weight/loss/view/schedule/epoch/seed tuning, no third replicate, and no revealed-cohort access.
- Passing all gates permits only a separately preregistered patient-independent external confirmation that has not been used for method selection; it never permits another test270 or UCSD75 run.

## Engineering tests and smoke

- Add isolated split/training/preaccess/evaluator/runner entry points; do not modify hash-bound historical files.
- Synthetic tests must cover old-subject exclusion; C/D exact counts and mutual subject/case disjointness; arm constants; seven-view order/cycle; flip/inverse identity; consistency formula; w0/w0.1 loss direction; exact pair RNG/sampler/first batch; fixed168; no inner loader; paired comparison/gate directions; forbidden-role rejection; evaluator requires all four complete checkpoints and preaccess.
- GPU smoke uses train-role data only: paired one/two optimizer steps, exact initial/batch/reflection hashes, finite gradients/loss, Candidate parameter divergence after update, Control/Candidate two student forwards, reload exact, and peak≤60GiB. No new inner/test/external access.

## Cost, storage, and next decision

- Four models in two dual-A100 waves; estimated10–14h wall and20–28 GPU-hours. Formal launch waits for proportional smoke.
- Expected peak35–50GiB with hard60GiB gate. Reuse existing cache; four checkpoints/logs/reports<150MB.
- Development inference reports both1× and8×; deployment parameter count unchanged. A valid success must improve both, so an 8×-only gain is not relabeled as learning the benefit into the model.
- Archive all split/implementation/smoke/launch/preaccess/raw/result evidence and hashes; update method catalog/handoff at every status. Leaderboard is unchanged because no named test cohort is accessed.
