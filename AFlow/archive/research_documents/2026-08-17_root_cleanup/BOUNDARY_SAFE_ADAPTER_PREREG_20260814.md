# SAGE-3D boundary auxiliary 全量配对预注册（2026-08-14）

冻结时间：`2026-08-14T05:24:53Z`，早于本方案任何 GPU smoke/正式训练。

## 目的

上一轮 seed-20260816 frozen-base safe adapter 得到正向 development signal：plain/gated
scan Dice 为 `0.701991/0.702090`，但 gate 没有相对 plain 的可证明收益。本轮同时回答：

1. plain safe adapter 的 `+0.0029` 能否在新 seed 的完整 972/108 训练中复现；
2. 不使用稀疏背景 CE 的 Dice-only 三维 boundary auxiliary，能否在同 seed 上进一步提高
   segmentation Dice。

## 严格配对臂

| GPU | arm | seed | boundary weight |
|---|---|---:|---:|
| 0 | control | 20260817 | 0.0 |
| 1 | boundary | 20260817 | 0.1 |

两臂完全相同：

- robust seed15 frozen base；
- SAGE plain adapter，总参数 `5,878,379`、可训练 `1,177,050`；
- residual/boundary 两输出 head、相同随机初始化、sampler、augmentation 和训练步；
- 最终 segmentation 均为 `base + residual`；boundary head 不参与推理输出；
- epoch 0 必须精确复现 frozen base。

唯一差异：boundary arm 增加 `0.1 * DiceLoss(boundary_head, GT_boundary_band)`。boundary
target 是 GT mask 半径 1 的对称三维形态学边界带。只使用 Dice，不加 BCE/CE，避免上一轮
error-map auxiliary 的大量背景把 residual 全局压小。

## 数据和隔离

- full train：972 scans；validation：108 scans / 50 subjects；subject-clean；
- T1ce，WT=`{1,2,3}`；GT-centered N4 `128^3` refiner cache；
- train index SHA-256：`34e9322a45b6501d841e2fab1dba43ef68353e767146ca547e5e8bb280aa95bf`；
- validation index SHA-256：`72fddef3c7a8755f0712b864efbb7328e2d8ae97313ac7ca7a0306d4b612ce44`；
- frozen base SHA-256：`c5d2b48db2666f097cb0d86e78c20ea95d163c07f98401714d25d755e99aceb0`；
- base scan/subject Dice：`0.699043355 / 0.694567241`；
- 270-case test、sealed manifest 和 test evaluator 不可达；
- 108 validation 已有开发暴露，本轮仍只能称 development，不是独立 test。

## 固定优化协议

- 16 epochs，patience 8；batch 2，workers 4，BF16；
- AdamW，LR `1e-4`，weight decay `1e-5`，gradient clip 5；
- RandomFlip、joint max shift 12、强度 scale/offset/noise；
- training-only independent coarse shift max 4，probability 1.0；
- 主损失为 segmentation DiceCE；
- 每臂按 validation scan-macro Dice 选 best epoch；同时报告 subject Dice、改善/恶化病例数
  和 boundary-head Dice。

显存未填满不是增加 batch 的充分理由：上一轮 GPU utilization 已约 99%。为保持与已取得
safe-adapter 结果的优化可比性，本轮不改变 batch/LR。

## 预注册判定

1. **复现成功**：control best scan Dice `>=0.701043355`（相对 base 至少 `+0.002`），
   且 subject Dice 不低于 base 超过 `0.001`。
2. **boundary 有额外贡献**：boundary − control scan Dice `>=+0.001`，且 subject Dice
   不下降；同时 boundary head 必须学到非零信号。
3. boundary arm 若只超过 base、但没有超过 matched control，只能说明 safe adapter 再次
   有效，不能说明 boundary 有效。
4. 不因某个 epoch、boundary Dice 或 train loss 好看而更改判据；不触发 test。

## 冻结代码

- `sammed3d/t1ce_wt_boundary_aux.py`：`57de2e1d604dfdb65a64e93c8bad44e61445e53d0af4e3903f76b4fae8c81682`
- `scripts/train_t1ce_wt_boundary_safe_adapter.py`：`5c49401ab7e5a50adc06364159e361cbb40e2165c5f6bf70c848b3a0564ff511`
- launcher：`f788d175448bc6ee8ba6e45e16e6f63abcdfc03ac7f3f64d63bbae6c6986ed01`

若 smoke 暴露实现 bug，必须在正式训练前记录修复和新 SHA，不得静默修改。
