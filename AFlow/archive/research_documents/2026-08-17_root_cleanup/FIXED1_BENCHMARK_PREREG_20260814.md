# Fixed 1-forward refiner-stage 在线 benchmark 预注册

冻结时间：2026-08-14 UTC，在运行 fixed1 benchmark 前冻结。

候选固定为 `FIXED1_DEPLOYMENT_CANDIDATE_20260814.json`：seed 20260815 identity、
threshold -0.75、CC500，共 1 member-view forward。reference 固定为 3 seeds × 8 views
的 24-forward locked workflow。

沿用 2026-08-13 deployment benchmark 的口径和 runner：

- validation 前 24 例；只测性能，不读取标签；
- 每个 receipt warmup 3 cases，24 cases × 2 repeats；
- run1：candidate GPU0、locked GPU1；run2 交换 GPU；
- 报告两个换卡方向和合并后的 latency、throughput、model allocated after load、peak
  allocated；
- 运行后验证 candidate/locked online output 与 exact atom cache mask parity；
- 计时范围只包含 refiner stage 的 H2D、forward、fusion、D2H、CC，不含上游 SAM coarse
  inference 和预处理，禁止称端到端加速。

不访问 270-case test，不使用 benchmark 结果改变 workflow。若某次 GPU 运行失败，只允许
原参数重跑并保留失败收据。
