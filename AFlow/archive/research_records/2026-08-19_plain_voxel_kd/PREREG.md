# 2025/2026 voxel-logit ensemble distillation preregistration

The fixed successful teacher is `0.75 historical168 + 0.25 EMA168` in logit space. A single true-128³ SegResNet student starts from historical epoch168 and continues for 32 epochs. The matched control has identical initialization, seed, weighted sampling, augmentation, optimizer, and epoch169–200 LR schedule. KD changes only the loss by adding binary per-voxel KL at temperature `2` with weight `0.3`.

Checkpoints are fixed at global epochs176/184/192/200. Selection and all gates use only historical fixed validation108. KD must beat both its matched control and the historical checkpoint, absorb at least 80% of the fixed teacher ensemble gain, and retain point hit within one validation case of the teacher. Failure stops the direction without test or hyperparameter search.

Primary 2025+ evidence: [2025 ensemble segmentation distillation](https://arxiv.org/abs/2509.11689) and [2026 3D brain MRI/CT voxel-KL distillation](https://www.nature.com/articles/s44401-026-00108-w).
