# BraTS T1ce/WT residual refiner 鲁棒训练追加报告

日期：2026-08-13 UTC

## 结论

在不访问 270 例 test 的前提下，新分支在 108 例 subject-clean validation 上达到
scan-macro Dice **0.700419**，严格超过预设目标 0.70。该数值是
validation-selected、oracle GT-centered 的开发结果，不是自动定位或独立 test 成绩。

最终协议：三个 robustshift4 随机种子的 validation-best checkpoint 等权平均；对
三维三个轴执行全部 8 种 reflection，逆变换后对 24 份 float32 logits 等权平均；
应用先前在 validation 上锁定的 `logit > -1.0`；删除小于 500 voxels 的
26-connected components。

| 指标 | Dice |
|---|---:|
| epoch-58 refiner A100 replay | 0.669548 |
| 最佳单 seed，raw threshold 0 | 0.699043 |
| 三 seed ensemble，raw threshold 0 | 0.697902 |
| 三 seed × 8-flip TTA，raw threshold 0 | 0.698235 |
| 三 seed × 8-flip TTA + locked postprocess | **0.700419** |
| 最终 subject-balanced | 0.695372 |

最终协议相对 epoch-58 A100 replay 提升 `+0.030871`，108 例中 96 例改善、12 例
下降。按 GT target voxels 分层，`<8000`、`[8000,30000)`、`[30000,80000)`、
`>=80000` 四组的提升分别为 `+0.023455`、`+0.043810`、`+0.026148`、
`+0.026115`。

## 为什么有效

训练缓存上的 frozen SAM coarse Dice 为 `0.829776`，validation coarse 只有
`0.636588`。因此原 refiner 在训练时主要看到 in-sample 的容易 coarse，而验证时
面对 held-out 的困难 coarse，这是二阶段 stacker covariate shift。

有效的单因素是在训练期仅对 coarse logit 额外施加每轴最多 4 voxel 的随机整数
平移，边界零填充；image 和 label 不做这次额外平移。三个种子 `20260813/14/15`
的 raw validation-best Dice 分别为 `0.695636/0.696351/0.699043`，说明该主效应
可以跨种子复现。

## 灰度损失结论

MSCL 风格本地实现为：

`MSE(GT 区域 T1ce 均值, sigmoid(logit) 加权预测区域 T1ce 均值)`。

严格配对结果：

- clean 低 LR 续训：control `0.673363`，gray weight 1 为 `0.672401`，差值
  `-0.000962`；
- shift4 鲁棒训练：control `0.695636`，gray weight 10 为 `0.693882`，差值
  `-0.001754`。

灰度项能够降低 gray-mean MSE，但没有增加 Dice。工程上它可以与 residual refiner
结合，科研上本轮结论是否定的；**0.700419 不能归因给灰度 loss**。

## 边界与下一步

- 分支仍使用 oracle GT-centered N4 128^3 crop 和旧三点击 coarse；
- raw 三模型 × TTA 尚为 `0.698235`；最后越过 0.70 依赖在同一 validation 上
  选择过的阈值和连通域规则；
- 0.700419 因此不是无偏泛化估计；
- 应立即冻结整个推理协议，在新的外部/确认集上一次性评测；
- 已经揭示的 270 例 test 禁止用于选择 seed、阈值、TTA 或连通域规则。

机器可读最终记录：

`workspace/BraTS/sammed3d_t1ce_wt/FINAL_VALIDATION_RESULT_RESIDUAL_REFINER_ROBUST_ENSEMBLE_20260813.json`

核心证据：

- `workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_robustshift4_three_seed_ensemble_20260813/ENSEMBLE_VALIDATION.json`
- `workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_robustshift4_three_seed_ensemble_20260813/ENSEMBLE_TTA8_VALIDATION.json`
- `workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_robustshift4_graymean_paired_a100_20260813/robustshift4_control/POSTPROCESS_VALIDATION.json`
