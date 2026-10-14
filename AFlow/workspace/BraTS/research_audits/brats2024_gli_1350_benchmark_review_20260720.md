# BraTS 2024 GLI 1350-Scan External Benchmark Review

Review date: 2026-07-20  
Dataset audited locally: 1350 scans/timepoints from 613 longitudinal subjects.  
Scope: Work #2 research objective, external literature targets, and protocol comparability.  
Access discipline: no local sealed-test metric was read; no inference or training was run.

## 1. Executive verdict

**The current SAM-Med3D result has not exceeded published BraTS 2024 GLI methods.** It cannot be
reported as a BraTS leaderboard result because the local and official tasks are different. Even a
non-rigorous raw-number comparison is unfavorable: all six local Dice values are below the official
2024 winner.

The Work #2 primary objective is therefore frozen as:

> Develop a four-modality, fully automatic, multi-class SAM-Med3D adaptation for post-treatment
> glioma segmentation, evaluate it on every case with the official BraTS 2024 region definitions
> and lesion-wise metrics, and beat strong nnU-Net/MedNeXt baselines under the same subject-disjoint
> protocol before making any external SOTA claim.

The existing 5-click binary model remains an **interactive prompted delineation** result and an
auxiliary thesis experiment. It is no longer the headline route for claiming superiority over
automatic BraTS papers.

## 2. Official task and local task are not equivalent

The official BraTS 2024 post-treatment task uses T1n, T1c/T1Gd, T2w, and T2-FLAIR jointly and asks
for one automatic 3D multi-class mask. It evaluates ET, NETC, RC, SNFH, TC, and WT with lesion-wise
Dice and lesion-wise HD95. Official `TC={ET,NETC}` and `WT={ET,NETC,SNFH}`; RC is excluded from WT.

| Dimension | Official BraTS 2024 | Current local Work #2 |
|---|---|---|
| Inference | Fully automatic | Five GT-error corrective clicks |
| Input | Four modalities jointly | T1c or T2f single modality per binary target |
| Output | Four atomic classes in one volume | One target-specific binary mask per run |
| Cases scored | Every case, including absent classes | Foreground-positive rows only |
| Primary metrics | Lesion-wise Dice and lesion-wise HD95 | Standard binary voxel Dice |
| WT | Labels `{1,2,3}` | Legacy target `{1,2,3,4}`, including RC |
| Spatial protocol | Official 1 mm preprocessed volumes | Local 1.5 mm resampling and label-centered crop |
| Evaluation set | Hidden validation/test | Internal clean validation |

The five clicks are generated from the current prediction error against ground truth, so they are
oracle-like corrective prompts. Excluding target-absent rows also removes the automatic detection
false-positive burden. These advantages make it especially inappropriate to explain the numerical
gap as merely a harder local protocol.

## 3. Current local result

The selected partial-decoder candidate is the mean of training seeds 2026 and 2027 on the same
strict clean validation audit: 630 foreground-positive target rows, 133 scans, and 67 subjects.
The primary local summary below is subject-balanced within each target and then macro-averaged.

| Region | Local subject-balanced Dice | Local row-weighted target Dice |
|---|---:|---:|
| ET | 0.483358 | 0.490394 |
| NETC | 0.301040 | 0.329798 |
| RC | 0.575759 | 0.578878 |
| SNFH | 0.679483 | 0.679748 |
| TC | 0.530278 | 0.538759 |
| WT, legacy includes RC | 0.757572 | 0.762848 |
| Six-region macro | **0.554582** | **0.563404** |

The historical `label_RC=0.58932` must not be used here. It was RC-only, used a 16-scan validation
set from the old scan-level split, and had longitudinal subject leakage. The recent
`+0.005265` is a candidate-minus-control weak-target improvement, not an absolute Dice or SOTA
score.

## 4. Official 2024 winner and raw gap

The official BraTS tooling lists Ferreira et al. first and Kim et al. second for the 2024 adult
post-treatment glioma task. The winning system ensembles five-fold nnU-Net, Swin UNETR, and
MedNeXt models and uses 14,301 synthetic scans in addition to 1,350 real training scans.

| Region | Local current | Winner hidden-test Dice | Raw local minus winner |
|---|---:|---:|---:|
| ET | 0.4834 | 0.7900 | -0.3066 |
| NETC | 0.3010 | 0.8076 | -0.5066 |
| RC | 0.5758 | 0.7760 | -0.2002 |
| SNFH | 0.6795 | 0.8926 | -0.2131 |
| TC | 0.5303 | 0.7874 | -0.2571 |
| WT | 0.7576 | 0.8938 | -0.1362 |
| Descriptive six-region macro | **0.5546** | **0.8246** | **-0.2700** |

The winner's hidden-test HD95 values are ET 35.63, NETC 30.35, RC 44.58, SNFH 16.87, TC 38.19,
and WT 17.95. The Dice and HD95 macros here are descriptive only; official ranking aggregates
ranks across regions and metrics rather than ranking a simple Dice mean.

This table is deliberately labeled a raw comparison. It does not make the protocols equivalent.
Its only valid conclusion is that the current result is neither formally comparable nor
numerically above the winner.

## 5. Published 1350-scan reference points

| Work | Split and protocol | Reported Dice | Interpretation |
|---|---|---|---|
| Ferreira et al., official rank 1 | 1350 real plus 14,301 synthetic; four modalities; official hidden test; lesion-wise | ET .7900, NETC .8076, RC .7760, SNFH .8926, TC .7874, WT .8938; six-region macro .8246 | Primary external target, but the original queue is closed |
| Kim et al., official rank 2 | 1350; internal random 80/20; four modalities plus T1c-T1; ensemble/TTA; official public validation | ET .7332, NETC .7861, RC .6948, SNFH .8682, TC .7120, WT .8704; macro .7775 | Public-validation anchor; internal split is not documented as subject-grouped |
| LG UNETR | 1350 scans / 613 subjects; random scan-level 8:1:1; four modalities; standard voxel Dice | ET .8080, TC .7825, WT .8850; three-region macro .8251 | Not official six-region evaluation; random longitudinal scans may leak subjects |
| UKAN-EP | 1080/135/135 random scans; four modalities; standard voxel Dice | ET .6149, NETC .5197, RC .6924, SNFH .8887, WT .9001; five-region macro .7232 | Useful secondary baseline; no TC and no documented subject grouping |
| DCD stage-wise distillation | 1080/270; four modalities; standard Dice/HD95/NSD | nnU-Net teacher four-class mDice .7523; compressed DCD student .6851 | Useful efficiency target; split is not documented as subject-grouped |
| SwinCLNet | Patient-level five-fold CV on all 1350; four modalities; standard Dice/HD95 | WT/TC/ET mean Dice .8782, mean HD95 9.71 mm | Strong subject-level reference, but only three aggregate regions and not official lesion-wise scoring |

These values answer whether the current model has surpassed the literature: **no**. They do not
form a single sortable leaderboard because splits, targets, absent-class handling, and metric
implementations differ. A paper must compare methods again under one frozen local protocol.

## 6. Research plan required for a defensible claim

### Phase A: protocol and data identity

1. Verify whether the local Kaggle three-part extraction corresponds to the corrected official v2
   training archive. The official v2 MD5 is `1d910b17d6cd32e38aa6296b8dfb7c77`; v1 is
   `2a4d884cc365ae2ec4bf1d8c2795ce53` and contains 10-15 incorrect labels.
2. Build a four-modality 1 mm manifest grouped by the 613 subject identifiers.
3. Retain every scan and every class-absent case.
4. Generate the official atomic labels ET/NETC/SNFH/RC and derive TC/WT without RC in WT.
5. Integrate and regression-test the official BraTS 2024 metrics.

### Phase B: strong same-protocol baselines

1. Train nnU-Net v2 or ResEnc nnU-Net with subject-grouped cross-validation.
2. Add MedNeXt if compute permits.
3. Report lesion-wise Dice, lesion-wise HD95, standard Dice, subject-balanced summaries, parameters,
   memory, and inference time for every method on exactly the same folds.

Without these baselines, a SAM-only number cannot establish that the proposed adaptation is useful.

### Phase C: automatic SAM-Med3D method

1. Replace single-modality input with four-modality fusion, initially through modality-specific
   stems/adapters feeding a shared 3D encoder.
2. Replace GT-derived corrective clicks at evaluation with an automatic semantic head or an
   automatic proposal/prompt module that never consumes ground truth.
3. Predict the four mutually exclusive atomic labels jointly, then derive TC and official WT.
4. Compare full fine-tuning, parameter-efficient adapters, and the strong U-Net baselines under an
   equal protocol. The two B200 GPUs are suitable for running paired folds or baseline/method arms.

### Phase D: external confirmation

The current local sealed test is consumed and cannot be retried. Synapse now exposes an additional
BraTS 2024 GLI training-data archive; a peer-reviewed paper describes it as 271 newly labeled
post-challenge cases. Its exact relationship to the original hidden test is not confirmed by the
file page itself, so it may only be described as a candidate external holdout after provenance is
verified.

Before any access to those labels:

1. freeze architecture, training data, preprocessing, folds, checkpoints, metrics, and statistics;
2. create a one-shot access ledger;
3. evaluate all frozen baselines and the proposed method once;
4. make superiority claims only from paired, same-case results with confidence intervals.

The original BraTS 2024 queue no longer accepts post-challenge submissions. Therefore, an internal
split of the 1350 scans alone cannot prove that a method beat the original leaderboard winner.

## 7. Frozen success criteria

- Engineering validity: automatic four-modality inference, no GT-derived prompt, official regions,
  all cases, and official metric tests all pass.
- Method validity: the proposed SAM method beats its same-protocol nnU-Net/MedNeXt baseline on both
  mean lesion-wise Dice and the pre-registered rank-style aggregate, without a material HD95 loss.
- Replication: the direction holds across subject-grouped folds and at least two training seeds for
  the final comparison.
- External claim: superiority holds on a previously untouched external holdout after one-shot
  evaluation.
- Aspirational reference: official winner validation six-region Dice macro 0.7903 and hidden-test
  macro 0.8246. These are targets, not currently achieved results.

Until Phase A is complete, further small improvements to the current five-click clean validation
are not evidence toward the automatic BraTS SOTA objective.

## 8. Primary sources

- [BraTS 2024 post-treatment task and metrics](https://arxiv.org/abs/2405.18368)
- [Official top algorithms and 2024 ranking](https://brats.readthedocs.io/en/stable/)
- [Official winner and hidden-test table](https://arxiv.org/abs/2411.04632)
- [Official runner-up paper](https://arxiv.org/abs/2409.08143)
- [LG UNETR on BraTS 2024 GLI](https://www.mdpi.com/2076-3417/15/11/5981)
- [UKAN-EP on BraTS 2024 GLI](https://link.springer.com/article/10.1186/s12880-025-02053-w)
- [DCD stage-wise distillation](https://arxiv.org/abs/2605.26382)
- [SwinCLNet](https://www.nature.com/articles/s41598-025-31937-8)
- [Official GLI v2 label-correction changelog](https://www.synapse.org/Synapse:syn53708249/discussion/threadId=11204)
- [Additional GLI archive](https://www.synapse.org/Synapse:syn64314352)
- [Post-challenge submission status](https://www.synapse.org/Synapse:syn53708249/discussion/threadId=11237)
