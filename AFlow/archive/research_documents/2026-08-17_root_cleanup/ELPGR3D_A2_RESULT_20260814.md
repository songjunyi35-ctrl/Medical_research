# ELPGR-3D A2 正式结果（2026-08-14）

## 结论

ELPGR-3D A2 没有超过冻结的 robust-shift4 控制，按
`ELPGR3D_A2_PREREG_20260814.md` 判为正式负结果。不能宣称 Dice 收益。

| seed | robust control scan Dice | ELPGR-A2 scan Dice | 差值 | ELPGR subject Dice | best epoch |
|---|---:|---:|---:|---:|---:|
| 20260814 | 0.696351247 | 0.694599239 | -0.001752008 | 0.691885258 | 8 |
| 20260815 | 0.699043355 | 0.697235247 | -0.001808109 | 0.693762779 | 6 |
| mean | **0.697697301** | **0.695917243** | **-0.001780058** | **0.692824019** | — |

预注册成功线要求新方法两 seed mean 至少 `0.699697301`，因此未通过。两个 seed
下降均未超过 `0.002`，但这不能替代 primary 失败。

## 实际实现与机制观察

相对旧 `4,701,329` 参数 residual refiner，新模型为 `4,702,210` 参数，只增加
`881` 个参数：

- label-free 输入增加 coarse uncertainty `4p(1-p)` 与 coarse probability 的局部
  三维形态学梯度；
- 输出 residual 和 correction gate；
- 最终 logit 为 `coarse + (1+tanh(gate))*residual`；
- gate 辅助目标为膨胀半径 2 的 `XOR(coarse prediction, GT)`；
- 结构桥接在 CPU FP32 单测中与旧模型逐体素零容差一致；A100 BF16 完整 validation
  reload 为 `0.669550`，与冻结 `0.669548250` anchor 差约 `1.7e-6`，在预注册容差内。

best epoch 的 gate 确实学到了 error map，但同时过度压低 residual：

| seed | gate Dice | gate target fraction | mean residual scale | improved/worsened vs coarse |
|---|---:|---:|---:|---:|
| 20260814 | 0.694851 | 0.061930 | 0.180236 | 103 / 5 |
| 20260815 | 0.584570 | 0.061930 | 0.266906 | 105 / 3 |

训练后期 train Dice 持续升高、gate loss 持续下降，而 validation 明显下降。合理解释是：
约 6.2% 的稀疏 error target 让 gate 把大部分体素的旧 residual 压小；网络随后在训练集
用更大的 residual 补偿，但破坏了 robust refiner 已学会的 held-out 修正。这是来自结果
的机制推断，不是新的独立因果实验。

因此不继续搜索 gate loss 权重。后续顺序实验改为 frozen robust base + zero-initialized
safe adapter，使 epoch 0 精确保留强基座，再让新模块只学习追加修正。

## 训练与验收

- 972 train / 108 validation，subject-clean；270 test 不可达；
- 两张 A100-SXM4-80GB 并行，各约 6.7 GiB training memory，GPU 利用率约 98–99%；
- 16 epochs，全程无 NaN/OOM；每 arm 约 1065 秒；
- 全仓测试：`232 passed, 26 warnings`。

证据：

| artifact | SHA-256 |
|---|---|
| seed20260814 `TRAIN_DONE.json` | `8bad3bc254a3d1e79b3d355d139bb0446dfe7a921509280d93b41ae0eb63349e` |
| seed20260815 `TRAIN_DONE.json` | `17a5a0dc4e5ae0c7dd2308374aabce55539c0b938eb7684716390a3764fe3c8f` |
| seed20260814 best checkpoint | `a007da2f48855193242e997abb2aa217679c1429f05c899c5ed6f7ece00c6240` |
| seed20260815 best checkpoint | `149b777097ecd8d4c0179955e35c235d67da3f5fd237fd2fa96d5b0918869ec5` |

所有 artifact 位于
`workspace/BraTS/sammed3d_t1ce_wt/error_gated_refiner_a2_two_seed_20260814/`，并明确记录
`test_access: false`。
