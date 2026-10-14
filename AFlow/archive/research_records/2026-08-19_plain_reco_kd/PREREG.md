# 2026 ReCo-inspired region-normalized KD preregistration

This is a single-factor extension of the completed voxel-KD continuation. It keeps the same student initialization, fixed ensemble teacher, seed, sampler, augmentation, LR, `T=2`, and voxel-KL weight `0.3`. The only addition is `0.1` times per-case normalized SmoothL1 logit distillation across GT foreground, a two-voxel morphological boundary band, and nonboundary background; boundary receives weight two.

Selection uses fixed validation108. Relative to base voxel-KD, promotion requires scan gain at least `0.001` or one additional point hit, while subject Dice may not fall more than `0.001`; historical scan and subject must remain exceeded. Failure ends the region-KD direction without test or hyperparameter tuning.

Primary source: [ReCo-KD, 2026](https://arxiv.org/abs/2601.08301).
