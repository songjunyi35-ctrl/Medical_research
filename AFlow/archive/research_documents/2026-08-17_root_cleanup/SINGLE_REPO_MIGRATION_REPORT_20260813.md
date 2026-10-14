# AFlow 单仓库迁移验收记录（2026-08-13）

权威科研交接见 `AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md`。本文件只记录本次
代码/资产迁移和验收，避免在 handoff 中堆积机械清单。

## 新增内容

- `third_party/SAM-Med3D/`：33 个上游源码/许可证/README 文件，以及
  `SNAPSHOT_PROVENANCE.md`、`SHA256SUMS`。
- `artifacts/checkpoints/sam_med3d_turbo.pth` 和同目录 README。
- `environment/README.md`、`requirements-medical.txt`、
  `sammed3d-venv.freeze.txt`。
- `sammed3d/paths.py`。
- `scripts/aflow_env.sh`、`doctor_brats_single_repo.py/.sh`、
  `verify_t1ce_wt_refiner_cache.py`、`rebuild_t1ce_wt_refiner_cache.sh`。
- `AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md`（唯一权威交接）。
- `workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813/` 持久缓存及
  `MIGRATION_RECEIPT.json`。
- `workspace/BraTS/sammed3d_t1ce_wt/SINGLE_REPO_DOCTOR_20260813.json`。

## 修改内容

路径/源码 import：

- `sammed3d/engine.py`
- `scripts/evaluate_sammed3d_t1ce_zero_shot.py`
- `scripts/train_sammed3d_t1ce_wt_dpdsam_aligned.py`
- `scripts/train_sammed3d_t1ce_wt_partial_unfreeze.py`
- `scripts/train_sammed3d_t1ce_wt_upstream_literal_va.py`
- `scripts/preflight_sammed3d_t1ce_rank0_matched.py`
- `scripts/make_dpdsam_aligned_prereg.py`
- `scripts/cache_sammed3d_t1ce_wt_refiner_inputs.py`
- `scripts/train_sammed3d_t1ce_wt_residual_refiner.py`
- `tests/test_sammed3d_peft.py`
- `tests/test_preflight_sammed3d_t1ce_rank0_matched_v2.py`

统一 source `scripts/aflow_env.sh` 的 shell 入口：

- `scripts/bootstrap_b200_env.sh`
- `scripts/finish_gtcrop_arm.sh`
- `scripts/launch_sammed3d_t1ce_fv_lora_a10.sh`
- `scripts/launch_sammed3d_t1ce_rank0_matched_a10.sh`
- `scripts/launch_sammed3d_t1ce_rank0_matched_a10_v2.sh`
- `scripts/launch_t1ce_fv_lora_auto_test_a10.sh`
- `scripts/run_sammed3d_t1ce_rank0_matched_foreground_v2.sh`
- `scripts/run_sammed3d_t1ce_rank0_selected_val_reload_v1.sh`
- `scripts/run_sammed3d_t1ce_wt_dpdsam_aligned.sh`
- `scripts/run_sammed3d_t1ce_wt_matched_val108_cpu_pair_v1.sh`
- `scripts/run_sammed3d_t1ce_wt_matched_val108_h100_pair_v1.sh`
- `scripts/run_sammed3d_t1ce_wt_partial_unfreeze_formal.sh`
- `scripts/run_sammed3d_t1ce_wt_partial_unfreeze_smoke.sh`
- `scripts/run_t1ce_wt_official15_reproduction_20260810.sh`
- `scripts/run_t1ce_wt_official15_reproduction_va_20260812.sh`
- `scripts/run_t1ce_wt_official15_reproduction_va_a100_20260812.sh`
- `scripts/run_t1ce_wt_prompt_local_refiner_a100_20260812.sh`
- `scripts/run_t1ce_wt_residual_refiner_graymean_paired_a100_20260813.sh`
- `scripts/run_t1ce_wt_residual_refiner_graymean_paired_a100_20260813_v2.sh`
- `scripts/run_t1ce_wt_residual_refiner_mixshift4p50_arm_20260813.sh`
- `scripts/run_t1ce_wt_residual_refiner_reproduction_20260810.sh`
- `scripts/run_t1ce_wt_residual_refiner_reproduction_20260810_v2.sh`
- `scripts/run_t1ce_wt_residual_refiner_robust_to_clean_arm_20260813.sh`
- `scripts/run_t1ce_wt_residual_refiner_robustshift4_graymean_paired_a100_20260813.sh`
- `scripts/run_t1ce_wt_residual_refiner_robustshift4_seed_arm_20260813.sh`
- `scripts/run_t1ce_wt_upstream_literal_test_a100_20260812.sh`
- `scripts/run_t1ce_wt_upstream_literal_va_a100_20260812.sh`
- `scripts/run_upstream_phase0_worker.sh`
- `scripts/run_worker_research.sh`

文档/忽略规则：`.gitignore`、`README.md`、`T1CE_MAINLINE.md`、
`AFLOW_BRATS_RESULTS.md`、`AFLOW_BRATS_WEEKLY_REPORT.md`。

## 扫描与测试

- 环境没有安装 `rg`，因此用等价的递归 `grep --include` 扫描
  `sammed3d/`、`scripts/`、`tests/`、`third_party/`。
- 活动 Python/shell 中旧 `SAM-Med3D-main` 绝对路径和 `../SAM-Med3D-main`：0。
- 活动 Python/shell 中旧 `sammed3d-venv` 绝对路径：0。
- 剩余 sibling venv 命中：`scripts/aflow_env.sh` 两行有意的相对兼容 fallback；
  正式优先级仍是 `AFLOW_PYTHON` 和 `AFlow/.venv`。
- `bash -n scripts/*.sh`：通过。
- 当前 Python 下所有 `sammed3d/*.py`、`scripts/*.py`、`tests/*.py` 编译：通过。
- vendored `sha256sum -c third_party/SAM-Med3D/SHA256SUMS`：全部通过。
- `pytest -q`：207 passed，26 warnings。
- 两例 A100 GPU cache smoke：通过。
- 972/108 完整 cache rebuild + full logits SHA：通过。
- 两张 A100 上无豁免 doctor：通过；报告 SHA-256
  `148422d80eedf3be505271462064c03e99286ace8059eeb72bafd8fc1ed8cf69`。

## 未改动边界

- 没有运行 `git reset`、`git clean`、`git checkout`。
- 没有删除、移动、重命名已有实验文件。
- 原 sibling `SAM-Med3D-main` 和 `sammed3d-venv` 保留。
- 没有启动新的训练；GPU 工作仅为固定 checkpoint 推理缓存重建和 smoke/doctor。
- 历史 JSON/日志中的旧绝对路径保留作为 provenance，没有批量改写。
