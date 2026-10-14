# ELPGR-3D A2 双种子架构实验预注册（2026-08-14）

冻结时间：`2026-08-14T04:10:08Z`，早于任何本方案 GPU smoke 或正式训练。

## 研究问题

当前 residual refiner 把 T1ce 与 frozen SAM-Med3D coarse logit 只在输入层拼接，随后在全体素范围直接加 residual。它没有显式回答“哪里应该修、哪里应该保留”，因此可能破坏本来正确的 coarse 区域。

本轮检验 ELPGR-3D A2（Error-Localized Prompt-Gated 3D Residual Refiner）：

- 增加两个完全不依赖 GT 的 dense-prompt 输入：`4p(1-p)` coarse uncertainty，以及 `3x3x3` coarse-probability morphological gradient；
- residual 之外增加 correction-gate head；
- 输出为 `coarse + (1 + tanh(gate)) * residual`，允许局部抑制或放大修正；
- 训练时用膨胀半径 2 的 `XOR(coarse mask, GT)` 监督 gate；GT 不进入推理输入；
- 从旧 refiner 做结构桥接：旧两通道首层和 residual head 原样复制，新输入核及 gate head 置零，所以 epoch 0 必须严格复现旧 checkpoint。

这轮先检验 uncertainty/boundary dense prompt + correction gate（A2）。boundary prediction head（A3）、频率分支、LoRA 和 GRPO 均不在本轮加入，避免一次改变过多因素。

## 数据与隔离

- train：972 scans，subject-clean 主训练集；
- validation：108 scans / 50 subjects；只作开发验证；
- task：T1ce 单模态，WT=`{1,2,3}`；
- cache：GT-centered N4 `128^3` refiner cache，coarse 为 frozen SAM click-3 logit；
- 270-case test、sealed test manifest 和任何 test evaluator 在训练入口均不可达；
- 由于这 108 例已参与过此前开发，本轮无论结果多高都只称 development validation，不能称独立 test 泛化。

缓存证据：

- train index SHA-256：`34e9322a45b6501d841e2fab1dba43ef68353e767146ca547e5e8bb280aa95bf`
- validation index SHA-256：`72fddef3c7a8755f0712b864efbb7328e2d8ae97313ac7ca7a0306d4b612ce44`

## 初始化与配对控制

所有新 arm 从同一 epoch-58 legacy residual-refiner checkpoint 开始：

- SHA-256：`4513ea4544ae153064575ec2ed5c56e1b78adf5e89e9d094754227b3bd429f21`
- reload validation scan Dice：`0.6695482504071185`

冻结的历史 robust-shift4 控制使用相同初始化、数据、batch、学习率、训练 epoch、augmentation、独立 coarse shift 和 seed 编号：

| seed | control scan Dice | control subject Dice | checkpoint SHA-256 |
|---|---:|---:|---|
| 20260814 | 0.6963512466 | 0.6925460146 | `c94f32c04cb89f51e1b83ae157bbce1d89371aa21b4267164e1ed645fa4d4cbc` |
| 20260815 | 0.6990433552 | 0.6945672413 | `c5d2b48db2666f097cb0d86e78c20ea95d163c07f98401714d25d755e99aceb0` |
| mean | **0.6976973009** | **0.6935566279** | — |

旧控制与新网络的参数初始化消耗不同，不能声称两个训练过程使用逐步相同的 augmentation RNG；“同 seed”表示相同 seed 编号和训练协议，不表示 bitwise training trajectory。

## 固定训练协议

两张 A100 各跑一个 seed：GPU0=`20260814`，GPU1=`20260815`。

- 16 continuation epochs，patience 10；
- batch size 2，4 workers；
- AdamW，LR `5e-5`，weight decay `1e-5`；
- BF16 autocast，gradient clip 5；
- joint spatial max shift 12、RandomFlip、intensity scale/offset/noise，完全沿用旧 refiner；
- training-only independent coarse integer shift max 4，probability 1.0；
- segmentation loss：DiceCE，权重 1；
- gate loss：DiceCE，权重 0.1；
- gate error dilation radius：2 voxels；
- 每 seed 按 validation scan-macro Dice 选择 best epoch，与历史控制一致。

冻结代码：

- `sammed3d/t1ce_wt_error_gated_refiner.py`：`2527e591db3c836047ac24ab0daacb905f32e3e896ebbdc322101afe9a3f5e3c`
- `scripts/train_t1ce_wt_error_gated_refiner.py`：`2621bdc24d91181999997329f8c00e44e8706806e6d3d206d1749caed4b40c38`
- launcher：`58d2d229c0a61a25633e632c884e1532daf857fc1a4e34b491b3afb33a3f343d`

代码若因 smoke 暴露 bug 而改变，必须在正式训练前把变更及新 SHA 追加到本文；不得静默修改。

## 指标与判定

Primary：两 seed best validation scan-macro Dice 的均值。

Secondary：subject-macro Dice、每 seed 方向、improved/worsened cases、gate Dice、平均 residual scale。

预注册的“进入下一阶段”门槛同时满足：

1. 新方法两 seed scan-Dice mean `>= 0.6996973009`，即相对冻结控制 mean 至少 `+0.002`；
2. 任一 seed 相对对应控制不得下降超过 `0.002`；
3. 两个 seed 完整 reload 和证据校验通过。

未通过就记录为负结果；不因为某个单 seed 或某个 epoch 好看而改变门槛。通过后才允许做 A1/A2/A3 消融和新 inner-selector 协议。subject Dice 与边界指标即使改善，也不能替代未通过的 primary，但可作为后续假设来源。

## GRPO 决策

本轮不训练 mask-level GRPO。`128^3` mask 有 2,097,152 个体素，DiceCE 提供稠密可微监督，而整幅 Dice reward 只给一个高方差标量。历史本地 GRPO 还是 label-RC、存在 subject 泄漏且 importance ratio 退化的旧任务，不能迁移为当前 WT 证据。

GRPO 保留为后续独立贡献：先做监督冷启动的 `AutoPromptPolicy3D`，动作只选择候选提示点、正负类型和 STOP，再用真正的 group rollout/old log-prob/clipped ratio/reference KL 优化“更少提示达到相同 Dice”。它不替代 mask decoder 的 DiceCE。
