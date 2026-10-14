# BraTS T1ce/WT 科研流水账

正式阅读起点：2026-08-13，单通道T1ce→WT，师兄路线对应的vanilla SAM-Med3D基础模型复现完成。

按用户2026-09-14要求，已从本账本正文移除这个起点之前的旧工作流搜索、四模态方向及未对齐的早期试跑。以下复现记录原文保留，不改实验数字、交互协议或既有测试访问事实；文档裁剪不代表测试集重新成为盲测。

本早期账本的留存记录截至2026-08-13。此后的工作记录继续见：

- [8月13日起的长交接](../../AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md)
- [方法尝试与成败记录](../../AFLOW_BRATS_METHOD_CATALOG.md)
- [当前交接](../../AFLOW_BRATS_HANDOFF_CURRENT.md)

本文件是历史复现记录，不是当前实验执行授权。清理详情及压缩回退备份见[清理收据](../reviews/2026-09-14_ledger_baseline_cleanup/RECEIPT.json)。

---

## 2026-08-13：官方预处理 vanilla SAM-Med3D baseline 复现完成

### 结论先行

本阶段已完成师兄路线对应的 **vanilla SAM-Med3D T1ce→WT baseline 复现**，并将其冻结为
下一阶段科研的统一对照。采用 subject-clean 972/108 development split，在两张 A100 80GB
上完成 200 epoch 全参数训练；按 validation 选择 epoch 21 后，在 270 例项目内 held-out
test 上取得：

- scan-macro per-case best-of-5 DSC：**0.594226**；
- scan-macro final-click-5 DSC：**0.585754**；
- 119-subject-balanced best-of-5 DSC：**0.580970**；
- 119-subject-balanced final-click-5 DSC：**0.572699**。

该结果与 MSCL Table 2 的 vanilla SAM-Med3D baseline `0.601` 相差 `-0.006774`，数值上处于
同一量级，可以作为“师兄 baseline 复现完成”的项目结论。严谨限定是：MSCL 未披露完全等价
的 subject split、病例数与 prompt 随机协议，因此这是 **under our explicit clean protocol**
的可审计复现，不是未公开细节的 exact reproduction。MSCL Table 1 的 `0.731` 对应含作者
灰度阈值损失的配置，不是本 vanilla baseline 的目标。

### 数据、任务与 subject 隔离

- 数据：BraTS 2024 post-treatment GLI，共 1,350 scans / 613 longitudinal subjects。
- 输入：T1ce 单模态。
- 目标：binary `WT={1,2,3}`；raw label `4` 明确排除并视为背景。
- train：972 scans / 444 subjects。
- validation：108 scans / 50 subjects。
- test：270 scans / 119 subjects。
- train/validation/test 两两之间 case ID 和 subject ID overlap 均为 0。
- 所有 active T1ce/seg 数据已迁移到 VA HDFS：
  `/mnt/hdfs/sjy/datasets/brats2024_gli_t1ce_seg_v1`；2,700 个文件通过源/目标字节级比较，
  zero mismatch。

冻结 manifest SHA-256：

| Split | SHA-256 |
|---|---|
| train | `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65` |
| validation | `f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb` |
| test | `a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9` |

### 官方 1.5 mm 预处理复现

train、validation 和 test 均调用同一 voxel transform 实现：

1. 从原始 segmentation 生成 `WT={1,2,3}` 二值 TorchIO `LabelMap`；
2. 图像和标签成对重采样到 `1.5×1.5×1.5 mm³`；图像线性插值，LabelMap 最近邻插值；
3. TorchIO `ToCanonical` 转 canonical orientation；
4. 根据 GT 标签中心 `CropOrPad(mask_name="label")` 到 `128×128×128`；
5. 在裁剪后的 T1ce patch 中，用正值图像体素计算 sample mean/std，再对整个 patch 做 z-score；
6. 不做 N4；
7. 仅训练时执行三个空间轴各 `p=0.5` 的 `RandomFlip`；validation/test 不增广。

train、validation、test 的 crop retention 最小值和均值均为 `1.0`，没有病例丢失 WT 前景。
缓存记录图像/标签 payload SHA，test 缓存只在用户明确授权且 freeze 文件写入后生成。

### 模型与上游训练语义

- 初始化 checkpoint：SAM-Med3D Turbo，SHA-256
  `899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`。
- 全参数解冻：`100,509,232 / 100,509,232` trainable。
- 训练循环逐行对齐实际执行的 upstream `train.py`：硬编码 11 次 interaction；每次只传当前
  sparse point，不累计 point history；上一轮 low-resolution logits 递归作为 dense mask prompt；
  零基 interaction index 2–8 中随机一次不传 sparse point，最后的 index 10 也不传点。
- click error 与 train Dice 按 upstream quirk 使用 raw logit `>0.5`；11 个 MONAI DiceCE loss
  求和而非取平均；前景体素 `<=1000` 的样本在 RandomFlip 后按 upstream recursive replacement
  行为替换。
- 双卡 correction：由于上游直接调用 model submodules、绕过 DDP wrapper forward，显式执行
  mean gradient all-reduce；binary integer GT 只在 DiceCE loss 边界转 float，以兼容当前
  MONAI。两项 correction 均在配置和测试中显式记录。

### 正式训练配置与过程

| 项目 | 配置 |
|---|---|
| GPU | 2 × A100-SXM4-80GB |
| epochs | 200，不 early stop、不在运行中改协议 |
| per-GPU batch | 12 |
| accumulation | 20；nominal effective batch 480 |
| optimizer | AdamW，weight decay 0.1 |
| encoder/head LR | `8e-4 / 8e-5` |
| scheduler | MultiStepLR milestones 120/180，gamma 0.1 |
| precision | FP16 autocast + GradScaler |
| seed | 2023 + rank |
| workers | 12 / rank |

upstream 的 step boundary quirk 使首个更新包含 21 个而不是 20 个 microbatches；该行为保留并
记录。epoch 1 的两个 optimizer steps 中有一个因 FP16 overflow 跳过，GradScaler 从 65,536
降到 32,768；epoch 2 起无 skipped step、无 NaN、无 OOM，训练完整生成 `TRAIN_DONE.json`。
每卡峰值 allocated memory 为约 68.43 GB，吞吐稳定在约 21–22 cases/s。

训练损失持续降低，train Dice 持续上升；最低 train loss `1.844121` 和最高 train Dice
`0.799562` 都出现在 epoch 199。validation 在早期达到峰值后回落，因此没有使用末期权重：

| Checkpoint/epoch | Train Dice | Val best-of-5 | Val final-click-5 |
|---|---:|---:|---:|
| selected epoch 21 | 0.609249 | **0.571306** | **0.559185** |
| final epoch 200 | 0.797926 | 0.520878 | 0.515272 |

### 五点击 validation/test 协议

- 共五次 current-point interaction，始终递归传入 low-resolution mask；
- click 1–2 使用 upstream RITM error-region sampler；
- click 3–5 使用 upstream uniform XOR error sampler；
- inference prediction 使用 sigmoid `>0.5`，等价 raw logit `>0`；
- 每例 seed 固定为 `SHA256(seed=2023, case_id)`，保证 validation 与 test 可重复；
- 主指标为每例五次点击中的最大 Dice 再对病例取宏平均，同时报告 final-click-5 和完整曲线。

test scan-macro click curve：

| Click | C1 | C2 | C3 | C4 | C5 |
|---|---:|---:|---:|---:|---:|
| Dice | 0.548900 | 0.566940 | 0.573730 | 0.579031 | **0.585754** |

best-of-5 scan median/std=`0.600321/0.131231`；best click 分布为 C1/C2/C3/C4/C5 =
`21/29/40/57/123` cases。

### 冻结、产物与后续边界

selected checkpoint：
`workspace/BraTS/sammed3d_t1ce_wt/sammed3d_upstreamliteral11_officialval5_va_a100x2_train972_val108_e200_s2023/sam_model_val_official_best5.pth`，
epoch 21，SHA-256
`c0c85a0fb90f6301344053f97cabe8a4972bd9febd4c6ae248f3a0908598fe31`。

完整 270-case report：同一 run 下
`test270_epoch21_official15_upstreamliteral5_s2023_20260812/test_report.json`，SHA-256
`7bd59843459876bd006230b24bc111cdfd726896cb3e6748b54b24c0a9659b63`。

270 cases 均唯一、按冻结 manifest 顺序排列、五个 Dice 全部 finite；汇总值已从逐病例记录
独立重算并与报告完全一致。该 test 没有参与 checkpoint 选择、训练、loss/architecture 搜索或
推理适配。但它在项目历史中曾用于其他 arm，故是 project-internal held-out test，而不是从未
访问的 external test。

从本节起，baseline 冻结，不再因 test 数值改变 batch、loss、prompt、学习率、预处理或
checkpoint。下一阶段方法研究只能使用 972/108 development split 迭代；若要形成强 untouched
test 结论，应另建 pristine external/confirmatory set，并在访问前冻结方法与统计方案。
