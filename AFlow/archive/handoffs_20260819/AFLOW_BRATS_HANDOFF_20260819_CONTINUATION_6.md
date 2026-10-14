# AFlow BraTS continuation handoff 6 — 2026-08-19

本文件承接`AFLOW_BRATS_HANDOFF_20260819_CONTINUATION_5.md`。

## 34. 清理失败160³派生cache，恢复约20 GiB

用户明确授权删除可重建旧cache并继续科研。删除：失败full756 true-160 cache约15G、失败test270
true-160 cache约5.2G、128/160两套12-case smoke约356M。磁盘可用空间从1.3G恢复到22G。
所有160 checkpoint、test/paired报告、科研归档与重建脚本均保留；继续使用的true-128 train cache
7.4G和fixed validation cache 1.1G保留。

## 35. 2026 ReCo-KD区域归一蒸馏：小幅同epoch收益，但不超最佳base KD

在成功voxel-KD上只增加weight`0.1`的前景/2-voxel边界/背景分别归一SmoothL1 teacher-logit
loss，boundary multiplier为2；seed、teacher、KL T=2/weight0.3、数据与32轮continuation不变。
训练1202秒，final reload差0。

validation最佳ReCo为global epoch192：`0.732960/0.723072/0.944444`。相同epoch192上相对
base KD提升scan`+0.000441`、subject`+0.000688`、point恰好1例，但CI跨0；且绝对Dice低于
base KD已选epoch200的`0.734384/0.724434/0.935185`。因此不为point绕过整体最优模型，ReCo
判负、不进test、不调region权重/边界宽度。归档见
`archive/research_records/2026-08-19_plain_reco_kd/RESULT.md`。

当前推荐保持：最高分为w=0.25双模型ensemble；最高效新单模为2025/2026 voxel-KD epoch200。
下一候选应转向NeurIPS 2025 LoMix 3D，而不是继续缝ReCo loss。
