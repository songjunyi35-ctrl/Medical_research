# Hard mL1-ACE GPU smoke 与成本更新

- 状态：4-step、真实 train-cache、batch 8 的 A100 smoke 全门通过。
- 输入范围：只读取 756 例训练 cache；未访问 validation、outer selector、confirmation 或 test。
- 峰值显存：allocated 21,717,570,048 bytes（约 20.23 GiB），reserved 23,647,485,952 bytes（约 22.03 GiB）；低于预注册的 24 GiB allocated 停止门。
- 速度：3 个计时 step 的中位数 0.4344 秒；按 94 steps/epoch × 32 epochs 机械外推约 1,306.7 秒，即 21.8 分钟。
- 数值：监督损失 0.18666，hard mL1-ACE 0.20446，两者比值 1.095；总损失 0.39113；ACE 到 logit 的梯度有限且非零。
- 随机性：再次计算 ACE 前后的 CUDA RNG SHA 相同，说明这个无随机参数的 loss 不改变配对随机流。
- 决策：smoke 通过，保持预注册权重 1.0，不根据 smoke 调权；允许进入正式训练。

证据：`workspace/BraTS/sammed3d_t1ce_wt/plain_hard_ml1ace_20260820/smoke_seed20260860/GPU_SMOKE.json`，SHA-256 `275ae6e8d54dbbb75e0c4054220c0e7e7869542c2b4f5c004c31ab8520d89e8a`。
