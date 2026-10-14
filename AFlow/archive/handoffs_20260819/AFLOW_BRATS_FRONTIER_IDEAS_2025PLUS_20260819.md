# AFlow BraTS 2025–2026 frontier ideas — 2026-08-19

This synthesis strictly prioritizes work first released or formally published in 2025–2026. The project bottleneck is single-model integration of complementary functional knowledge: the fixed two-model logit ensemble improves all three internal-test metrics, while higher resolution, single EMA, and cross-seed parameter interpolation fail.

## Selected first experiment: 2025/2026 voxel-logit ensemble distillation

Teacher is fixed from the already validation-selected ensemble:

`z_teacher = 0.75 z_history168 + 0.25 z_EMA168`

Student is the same 128³ 4.70M SegResNet initialized from historical epoch168. Run a matched 32-epoch continuation control and KD arm with identical seed, sampler, augmentation, LR schedule, and supervised DiceCE. The only change is:

`L_KD = T² mean_voxel KL([1-pT,pT] || [1-pS,pS]), T=2`

`L_total = L_DiceCE + 0.3 L_KD`

This recipe is supported directly by a 2025 deep-ensemble-to-single segmentation study and a 2026 3D brain MRI/CT distillation study that uses voxel KL with `T=2`, `lambda=0.3`, then deploys only the student.

Primary sources:

- [Uncertainty-Aware Retinal Vessel Segmentation via Ensemble Distillation, 2025-09-15](https://arxiv.org/abs/2509.11689)
- [From Performance to Practice: 3D brain MRI/CT distillation, 2026](https://www.nature.com/articles/s44401-026-00108-w)
- [MICCAI 2025 multi-source medical segmentation distillation](https://papers.miccai.org/miccai-2025/0355-Paper2167.html)

Minimum protocol: continuation checkpoints after 8/16/24/32 epochs; validation-only selection; paired supervised continuation control. Promotion requires the student to beat both the matched control and historical checkpoint on scan and subject Dice, preserve point hit within one case of the ensemble, and absorb at least 80% of the ensemble's validation gain.

## Ranked 2025+ follow-ups

### 2. ReCo-KD region/context distillation — 2026

ReCo-KD targets background dilution and small 3D structures using scale-normalized region distillation and contextual affinity. The lightweight adaptation is foreground/boundary/background-normalized logit KD; only if useful should decoder affinity alignment be added.

- [ReCo-KD, 2026-01-13](https://arxiv.org/abs/2601.08301)

### 3. LoMix 3D learnable multi-scale logit supervision — NeurIPS 2025

LoMix learns training-time mixtures of decoder-stage logits and discards auxiliary heads at inference. Its appendix reports a 3D MedNeXt experiment and explicit Conv3d/trilinear adaptation. It is preferable to generic old deep supervision because mixture operators and loss weights are learned jointly.

- [NeurIPS 2025 paper](https://papers.nips.cc/paper_files/paper/2025/file/5f0e54e34ba4c30fbc6f229986dbffb4-Paper-Conference.pdf)
- [Official code](https://github.com/SLDGroup/LoMix)

### 4. DCD wavelet detail distillation — MICCAI 2026

DCD distills directional 3D wavelet detail bands at encoder stages and reports BraTS 2024 experiments. A project-specific extension would keep the successful ensemble logit KD and use EMA168 as a detail teacher through learned projections, retaining one-model inference.

- [DCD, 2026-05-25](https://arxiv.org/abs/2605.26382)
- [Official code](https://github.com/ClinicaAlpha/DCD-3D-MedSeg)

### 5. IAD infusion-to-assimilation distillation — CVPR 2026

IAD explicitly addresses KD overwriting a student's existing useful capability: first inject teacher knowledge, then reduce KD and let the student re-assimilate semantics. This is relevant if naive voxel KD improves teacher matching but damages the historical student's strong Dice basin.

- [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Hong_From_Infusion_to_Assimilation_Distillation_for_Medical_Image_Segmentation_CVPR_2026_paper.html)
- [Official code](https://github.com/hjklearn/IAD)

### 6. Calibration and hard-tail losses — 2025/2026

- Hard-binned mL1-ACE was tested with MONAI SegResNet, DiceCE, and BraTS; it preserves Dice better than soft-binned calibration losses while reducing unreliable confidence: [Average Calibration Losses, 2025](https://arxiv.org/abs/2506.03942), [official code](https://github.com/cai4cai/Average-Calibration-Losses).
- Pixel-wise Modulated Dice emphasizes detached high-error voxels with almost no extra compute: [PM Dice, 2025/2026](https://arxiv.org/abs/2506.15744).
- FairDRO's case-level KL-DRO addresses hidden high-loss cases in 3D radiotherapy segmentation: [FairDRO, 2026](https://arxiv.org/abs/2605.10521).

## Ruled out or deferred

- Cross-seed parameter soup was executed and catastrophically failed, so no further alpha search.
- 160³ is not a teacher candidate unless a development oracle analysis proves unique complementary cases; its direct model is weaker and storage-expensive.
- 2025 NMSW native-resolution patches and EfficientMedNeXt/DMRFB remain architecture-level options after distillation/LoMix, not first-line changes.

All future hyperparameters and checkpoints must be fixed on development validation. The reused test270 cannot select KD temperature, weight, continuation length, or architecture.
