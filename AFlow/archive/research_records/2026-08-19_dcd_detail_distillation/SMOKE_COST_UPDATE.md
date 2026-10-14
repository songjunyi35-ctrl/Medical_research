# DCD GPU smoke cost update

This updates execution cost only; the scientific protocol remains frozen.

- Real train-cache batch size: `8`.
- Smoke DCD weight: maximum frozen `mu=0.01`.
- One warmup and three CUDA-synchronized measured optimizer steps.
- Step seconds: `1.964520 / 1.959725 / 1.961886`; median `1.961886s`.
- Mechanical `median × 94 steps × 32 epochs`: `5901.35s`, or `98.36min`; below the frozen two-hour stop gate.
- Peak allocated/reserved GPU memory: `34,733,490,688 / 43,700,453,376` bytes (`32.35 / 40.70 GiB`), below the `72GiB` stop gate.
- At maximum mu, weighted DCD was `0.031835`, supervised loss `0.211305`, ratio `0.150661`.
- Formal early epochs use much lower or zero mu but still compute raw DCD for protocol logging; actual formal time remains to be measured.
- No new validation, outer selector, confirmation, or test data were accessed.

Evidence: `workspace/BraTS/sammed3d_t1ce_wt/plain_dcd_20260819/smoke_seed20260846/GPU_SMOKE.json`, SHA-256 `b984177804f64352abe7120fa8d5856fa59fbdc70dfee0c37869fd482620fa9e`.
