# AFlow BraTS bottleneck literature synthesis — 2026-08-19

Three parallel research agents searched first-party papers and official implementations. Consensus: the project has reached a local single-model generalization/integration bottleneck, not a data-resolution information bottleneck. The successful `0.75 historical + 0.25 EMA168` logit ensemble proves useful complementary information exists; the highest-value goal is to compress that information into one model.

## Completed zero-cost probe: parameter soup

Model-soup-style parameter interpolation was the cheapest first check, motivated by [Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html) and its [official implementation](https://github.com/mlfoundations/model-soups). The fixed parameter ratio copied from the successful logit ensemble failed on every validation case, proving the two independently seeded checkpoints lie in different parameter basins. No further alpha search is justified.

## Ranked next ideas

### 1. Fixed-ensemble knowledge distillation into one SegResNet

Freeze the already successful teacher:

`z_teacher = 0.75 z_history168 + 0.25 z_EMA168`

Initialize a single 128³ student from the historical epoch168 checkpoint and continue for a short low-LR pilot. Preserve the existing DiceCE supervised loss and add teacher soft-target terms. The simplest first version is probability/logit regression; a stronger version adds a spatial distribution KL over voxels so the student learns the teacher's argmax ordering as well as mask shape.

Primary sources:

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Born-Again Neural Networks](https://arxiv.org/abs/1805.04770)
- [Structured Knowledge Distillation for Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.html)
- [Channel-wise Distillation for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/html/Shu_Channel-Wise_Knowledge_Distillation_for_Dense_Prediction_ICCV_2021_paper.html)

Recommended minimum experiment: historical checkpoint initialization; 32-epoch student and matched supervised-only continuation; teachers frozen/eval/no-grad; same geometric augmentation for teachers and student; fixed teacher weights and threshold. Promotion requires the student to exceed the historical validation scan/subject metrics, retain point hit within one case, and absorb at least half the teacher ensemble's validation gain. Final inference remains one model.

### 2. Reliability-selective ensemble distillation

If naive distillation copies teacher errors, weight KD by teacher agreement and confidence. Apply spatial-peak KD only when the teacher argmax lies inside the training GT; use GT-only loss on teacher-miss cases. This directly limits confirmation bias.

Primary sources:

- [A Good Student is Cooperative and Reliable](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_A_Good_Student_is_Cooperative_and_Reliable_CNN-Transformer_Collaborative_Learning_ICCV_2023_paper.html)
- [On the Efficacy of Knowledge Distillation](https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html)

### 3. TERM hard-case risk for the large-regression tail

Replace ordinary batch-mean reduction with tilted empirical risk over per-case DiceCE:

`L = logsumexp(t * L_i) / t - log(batch) / t`

Positive `t` emphasizes difficult cases smoothly and addresses the observed pattern in which many small gains are offset by fewer large regressions. First fixed pilot: `t=5`, batch8, all else unchanged.

Primary source and official code: [Tilted Empirical Risk Minimization](https://arxiv.org/abs/2007.01162), [TERM code](https://github.com/litian96/TERM).

### 4. DSC++ or structured argmax margin

DSC++ penalizes high-confidence FP/FN more strongly and may reduce incorrect global logit peaks; it adds no inference cost. A more direct point-specific option is a training-only margin `relu(m + max(logit outside GT) - max(logit inside GT))` on the same segmentation logits, avoiding the already-failed auxiliary center head.

Primary sources:

- [DSC++](https://arxiv.org/abs/2111.00528), [official implementation](https://github.com/mlyg/DicePlusPlus)
- [Heatmap Regression without Soft-Argmax](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Heatmap_Regression_without_Soft-Argmax_for_Facial_Landmark_Detection_ICCV_2025_paper.html)

### 5. Deep supervision, SWA, then SAM

Deep supervision changes decoder gradient delivery while preserving single-model inference; SWA explores and averages a basin differently from recent-weight EMA; original SAM directly optimizes local sharpness but roughly doubles training cost.

Primary sources:

- [MONAI SegResNetDS official source](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/segresnet_ds.py)
- [SWA](https://arxiv.org/abs/1803.05407)
- [SAM](https://arxiv.org/abs/2010.01412)
- [Medical-imaging sharpness optimizer comparison](https://arxiv.org/abs/2408.04065)

## Recommended execution order

1. Fixed-ensemble distillation versus a matched supervised-only continuation.
2. Reliability-selective KD only if naive KD incompletely absorbs the teacher.
3. TERM and DSC++ as independent single-factor experiments.
4. Deep supervision/SWA/SAM only after cheaper methods fail.

Do not use the revealed test270 to choose KD loss weights, epochs, TERM tilt, or DSC++ gamma. All future choices must freeze on development validation first.
