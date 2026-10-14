# `AFLOW_BRATS_ROBUST_REFINER_20260813.md` 灰度结论补充

最后一个预注册的 `mixshift4p50_graymean_w1` arm 在主报告生成后完成。本文件只修正
主报告中的灰度结论；最终模型、0.700419 分数和所有边界均不变。

三组严格配对如下：

| 训练 regime | Control | Gray | Gray - control |
|---|---:|---:|---:|
| clean continuation，gray w1 | 0.673363 | 0.672401 | -0.000962 |
| shift4 p=1，gray w10 | 0.695636 | 0.693882 | -0.001754 |
| shift4 p=0.5，gray w1 | 0.683627 | 0.685182 | **+0.001554** |

因此不应写成“灰度在所有配对中都没有增益”。准确结论是：灰度项在较弱的 50%
shift 课程中有小幅正则收益，但跨 regime 不稳定，而且该 gray 模型仍比最佳无灰度
shift4 p=1 模型低 `0.010455`。最终 >0.70 协议没有使用灰度 loss。

以机器可读 V2 结果为准：

`workspace/BraTS/sammed3d_t1ce_wt/FINAL_VALIDATION_RESULT_RESIDUAL_REFINER_ROBUST_ENSEMBLE_20260813_V2.json`
