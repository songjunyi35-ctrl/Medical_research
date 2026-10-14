# BraTS T1ce 代码清理审计

审计时间：2026-08-13 UTC

## 当前结论

代码清理与 residual-refiner 测试共享同一文件系统，不是独立副本。远端 worker
`1042986` 使用独立的 2 张 A100，因此不会与本次只读审计或 CPU 文件操作争抢
GPU；但是它会从共享仓库读取源码、checkpoint 和 manifest，并通过
multiprocessing 重新导入模块。因此在测试生成 `TEST_DONE.json` 以前，禁止移动、
改写或删除其依赖闭包。

本轮已经完成的删除只包括可再生成或已确认失效的文件：

- `.pytest_cache/`、`scripts/__pycache__/`、`sammed3d/__pycache__/`、
  `tests/__pycache__/`；
- 两个零字节历史日志；
- 四个已经核验 `/proc/<pid>` 不存在的历史 PID 文件。

清理前共计约 1.2 MiB。未删除、移动或改写任何源码、测试、checkpoint、结果
JSON、manifest、数据、训练 cache 或当前测试输出。

## 当前测试保护区

直到下面的文件出现前，不得启动第二阶段代码归档：

`workspace/BraTS/sammed3d_t1ce_wt/residual_refiner_stage2_gtcrop_h100_train972_val108_e60_s20260810_v2/test270_epoch58_gtcrop_n4_refiner3_s20260723_20260813/TEST_DONE.json`

必须保护：

- 整个 `residual_refiner_stage2_gtcrop_h100_train972_val108_e60_s20260810_v2/`；
- base checkpoint `sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805/sam_model_val_dice_best.pth`；
- `workspace/BraTS/cache/t1ce_wt_gtcrop_n4_128/`；
- `/tmp/aflow_t1ce_wt_gtcrop_n4_128/test_authorized_refiner_20260813`；
- `RESIDUAL_REFINER_RESULT_20260810.json` 和三个固定 manifest；
- 当前 evaluator、test-cache builder、refiner trainer、SAM engine/PEFT、T1ce
  data/prompt/training 模块及其导入依赖。

冻结记录中的关键哈希：base checkpoint `42f96a...`，refiner checkpoint
`4513ea...`，refiner result `8fe3c0...`。

## 第二阶段：测试完成后无损归档

以下是历史/负结果代码族，约 48 个文件、412 KiB。它们没有必要继续平铺在活跃
目录，但 T1ce 源码大多未被 Git 跟踪，所以应先移动到
`archive/retired_t1ce_code_20260813/` 并生成完整 SHA-256 manifest，而不是直接
删除：

当前 48 个候选的逐文件路径与清理前 SHA-256 已固定在
`CODE_ARCHIVE_CANDIDATES_20260813.sha256`。

1. LoRA/rank-0 历史分支：launcher、watcher、preflight、paired analysis 及测试；
2. partial-unfreeze 负结果分支的入口、trainer 和专项测试；
3. DPD-SAM A2/B0/D1 重建分支及专项测试；
4. prompt-local 64^3 正式负结果分支及专项测试；
5. 已中止的三点击 official15/phase0 launcher；
6. residual-refiner v1、旧 GT-crop conclusion、LR-decay prereg；
7. 旧 DPD-aligned 启动和预注册入口。

归档前先同步 `README.md` 和 `scripts/run_worker_research.sh`，移除 DPD-SAM、
partial-unfreeze 等历史公开命令，避免产生悬空入口。

以下看起来像旧代码，但仍被 upstream-literal baseline 直接或间接导入，不能归档：

- `scripts/train_sammed3d_brats.py`
- `scripts/train_sammed3d_t1ce_wt.py`
- `scripts/train_sammed3d_t1ce_wt_dpdsam_aligned.py`
- `scripts/train_sammed3d_t1ce_wt_official15_va.py`
- `sammed3d/partial_unfreeze.py`

若要进一步精简这些文件，需要先把共享协议、dataset 和训练辅助函数抽到独立模块，
再修改 import 并运行完整回归测试，不能直接按文件名删除。

## 大文件候选，不在当前测试期间删除

- 两套 N4 开发 cache 合计约 21.1 GiB，均仍服务于 brain-crop/refiner 追溯；
- T1ce checkpoint/output 约 7.8 GiB；
- 明确带 `SMOKE_ONLY.json` 的大 checkpoint 约 3.9 GiB，是测试完成后优先确认
  删除的空间候选；
- 已中止的 official15 run 约 768 MiB；
- 当前正式 upstream baseline 约 1.5 GiB，其中非选中 checkpoint 是否删除要
  单独确认，不能当成字节重复文件处理。

空间清理必须先记录精确路径、用途和必要哈希，再取得明确删除授权。当前
residual-refiner test 结束以前，不删除任何 checkpoint 或 cache。
