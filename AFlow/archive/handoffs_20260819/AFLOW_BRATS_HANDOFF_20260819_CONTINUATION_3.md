# AFlow BraTS continuation handoff 3 — 2026-08-19

本文件承接`AFLOW_BRATS_HANDOFF_20260819_CONTINUATION_2.md`。

## 31. 用户追加授权EMA168单模型test270

用户在固定w=0.25双模型ensemble刷新internal test点估计后，要求把EMA模型单独测test。按validation
中EMA候选最好的固定EMA168执行一次，EMA200未测试；raw logit阈值0、无TTA/后处理/SAM，test
不回流调EMA decay、起点、checkpoint或阈值。

| metric | historical epoch168 | EMA168 single | EMA-历史 | paired 95% CI |
|---|---:|---:|---:|---:|
| scan Dice | `0.741203949` | `0.740696468` | `-0.000507482` | `[-0.005142,+0.003969]` |
| subject Dice | `0.723179486` | `0.724904822` | `+0.001725336` | `[-0.005896,+0.009275]` |
| point hit | `0.929630` | `0.948148` | `+0.018519` | — |

128/270 cases提高、140下降、2持平；66/119 subjects提高、53下降。两项Dice CI均跨0，EMA168
单模型与历史模型Dice统计上相容，point明显更好，但没有刷新scan Dice。该结果进一步支持ensemble
机制：EMA提供互补point/误差信息，历史模型维持主Dice。当前推荐仍为固定
`0.75 * historical + 0.25 * EMA168`，其test为`0.745054/0.727345/0.955556`。

EMA168单模型test报告SHA-256为
`6e4515dc098143016b02b7c7121970c4bc4fcf5a61e1164b64d6863858c6e739`，归档见
`archive/research_records/2026-08-19_plain_ema/EMA168_TEST270_RESULT.md`。
