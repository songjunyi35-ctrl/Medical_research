# SAGE-3D frozen-base safe adapter 配对预注册（2026-08-14）

冻结时间：`2026-08-14T04:34:22Z`，早于本方案任何 GPU smoke/正式训练。

## 来自 ELPGR-A2 的顺序证据

上一份预注册实验 `ELPGR3D_A2_PREREG_20260814.md` 已完整结束且不达标：

| seed | robust control | ELPGR-A2 | 差值 |
|---|---:|---:|---:|
| 20260814 | 0.696351 | 0.694599 | -0.001752 |
| 20260815 | 0.699043 | 0.697235 | -0.001808 |
| mean | 0.697697 | 0.695917 | -0.001780 |

best epoch 时 correction gate 虽达到约 `0.58–0.69` Dice，但平均 residual scale 只有
`0.18–0.27`。因此下一实验不调 A2 的 gate-loss 权重，而是改变安全结构：冻结强基座，
让新模块以零 residual 开始，避免覆盖原有修正。

## 研究问题与两臂

共同结构 SAGE-3D（Safe Adaptive Gated Error adapter）：

- 基座固定为 robust seed-20260815 refiner，全部冻结且始终 eval；
- 小型 `init_filters=8` SegResNet adapter，输入五通道：T1ce、coarse logit、base
  logit、base uncertainty、base/coarse probability disagreement；
- adapter 输出 residual 与 gate 两个 head，末层全零初始化；
- epoch 0 必须严格复现 frozen base；
- 只有 segmentation DiceCE，不使用稀疏 error-map 辅助 loss。

配对臂：

| GPU | arm | 输出 |
|---|---|---|
| 0 | plain | `base + residual` |
| 1 | gated | `base + 2*sigmoid(gate)*residual` |

两臂均为总参数 `5,878,379`、可训练参数 `1,177,050`，使用完全相同的 adapter
结构、初始化、数据顺序和 seed `20260816`。plain 臂保留等容量 gate head，但不把它用于
输出；因此两臂唯一功能差异是 implicit gate 是否调制 residual。gate 无单独标签，仅从
最终 DiceCE 接受梯度。

## 冻结资产与隔离

- base checkpoint：`c5d2b48db2666f097cb0d86e78c20ea95d163c07f98401714d25d755e99aceb0`
- base validation scan Dice：`0.6990433552267926`
- base validation subject Dice：`0.6945672412819807`
- train/validation：972/108 scans，subject-clean；T1ce，WT=`{1,2,3}`；
- GT-centered N4 `128^3` refiner cache，属于 oracle-localization development 实验；
- train index：`34e9322a45b6501d841e2fab1dba43ef68353e767146ca547e5e8bb280aa95bf`
- validation index：`72fddef3c7a8755f0712b864efbb7328e2d8ae97313ac7ca7a0306d4b612ce44`
- 270-case test、sealed manifest 和 test evaluator 均不可达；108 validation 已有开发
  暴露，结果不能称独立 test 泛化。

## 固定训练协议

- 两臂 seed 均为 `20260816`；
- 16 epochs，patience 8；
- batch 2，workers 4，BF16；
- AdamW，LR `1e-4`，weight decay `1e-5`，gradient clip 5；
- 与 robust refiner 相同的 joint augmentation：RandomFlip、max shift 12、强度
  scale/offset/noise；
- training-only independent coarse integer shift：max 4，probability 1.0；
- base 冻结；adapter 单一 DiceCE objective；
- 每臂按 validation scan-macro Dice 选择 best epoch。

冻结代码：

- `sammed3d/t1ce_wt_safe_adapter.py`：`c671a966315af59b818cda1cdd0dc85427a98a5591d60687ef9009a22129b2e3`
- `scripts/train_t1ce_wt_safe_adapter.py`：`d6a2b67c94f4af9589e3564a238ed8f824ba3e35240e70e28133a974e7736e77`
- launcher：`bff4074e33f9d1dec9f1eb0797748911a55ed78b9928937663026cff7a465417`

若 smoke 暴露 bug，必须在正式训练前把修复和新 SHA 追加到本文，不得静默修改。

## 预注册判定

1. **Safe-adapter 有效**：plain 或 gated 至少一个 best scan Dice
   `>= 0.7010433552`，即相对 frozen base 至少 `+0.002`；同时 subject Dice 不低于
   base 超过 `0.001`。
2. **Implicit gate 有额外贡献**：gated 相对 plain 的 scan Dice 至少 `+0.001`，且
   subject Dice 不下降。
3. 若只达到 1 未达到 2，只能把收益归因于 frozen-base uncertainty adapter，不能归因
   于 gate。
4. 两项都未达到则记负结果；不按 epoch 曲线事后改 LR、loss 或阈值。

本轮仍是架构开发实验，不触发 270 test。即使超过 0.70，也要在新的外部确认集或重新
设计的未暴露确认协议上复核，才能作为最终泛化结论。
