# GCSAM Africa95结果与一手方法主张的边界

状态：只读论文与既有报告综合；0新训练/模型/指标访问。

## 一手论文主张

GCSAM（Gradient-Centralized Sharpness-Aware Minimization，梯度中心化锐度感知最小化）论文把多维权重梯度减去均值，再用于SAM上升扰动；直觉是减少梯度噪声并走向对参数扰动更稳的平坦区域。作者报告在CIFAR-10、乳腺超声和COVID胸片分类上优于Adam/SAM，并把设备/人群差异下的generalization作为动机。

论文证据是2D分类，不是3D医学分割、Dice/Point或跨治疗阶段Whole Tumor。项目实现还预先选择论文公式的centralized-gradient norm，而非官方示例的pre-centralization norm；因此不能把本项目叫原论文复现。

## 项目证据对上了什么

- 原冻结GCSAM184在Africa95的Dice比Full972 identity高`0.068617`，配对患者CI整体为正，四个已观察中心均值均正。
- 提升主要是recall `+0.0809`，precision差近0且不确定；对训练罕见的中大型WT更明显。
- 这些现象与“某些未见域上有更稳表示”的论文直觉相容。

## 为什么不能归因给优化器

外部两checkpoint不是same-seed单因素实验：

- GCSAM：fit864，另留108 inner选择epoch184，GCSAM optimizer，独立seed；
- Full：fit972，固定epoch168，AdamW voxel-KD，另一seed。

训练病例数、选epoch流程、seed和optimizer同时不同。Africa95只保证同病例、同输入、同指标的**预测比较**，不保证训练因果配对。

真正只改optimizer的fixed184 A/B双复制已失败：一个fold Scan负、另一个Subject负，平均Scan负且CI门均未过。因此当前最严谨结论是：

1. GCSAM184这个具体checkpoint在Africa95的区域Dice/recall更好；
2. 当前GCSAM训练配方仍没有跨seed可复制证据；
3. Africa结果不能选择`rho`、epoch、seed或重新训练，也不能证明平坦性是原因。

最后输出bias还更负（`−0.4804` vs Full `−0.4551`），排除“只把logit整体上移”的简单解释，但也不证明平坦解机制。

## 未来怎样真正检验

需要新的、未见评测数据和same-fit/same-seed/same-epoch AdamW↔GCSAM checkpoint，唯一变化是optimizer；外部队列必须在checkpoint冻结后一次性评测。当前fresh内部容量只剩2人/3 scans，Africa/UCSD/test都已关闭，所以本轮不能完成该因果实验。

## 证据

- [GCSAM paper](https://arxiv.org/abs/2501.11584)，当前HTML SHA-256 `08a9e5d09c2eabb94785af344f459148e4c7b889cfa2b16d79c0bf2ecfb5a208`
- [原GCSAM结果](../../research_records/2026-08-31_gcsam_full972_stability/RESULT.md)
- [fixed184双复制](../../research_records/2026-09-03_gcsam_fixed184_replication/RESULT.md)
- [Africa95结果](../../research_records/2026-09-06_brats_africa95_frozen_baselines/RESULT.md)
- [机器边界](RESULT.json)
