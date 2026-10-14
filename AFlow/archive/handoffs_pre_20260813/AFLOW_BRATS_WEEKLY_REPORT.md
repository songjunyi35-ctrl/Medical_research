# 周报：SAM-Med3D x BraTS 简版

> **历史周报（2026-08-10 起不再作为执行入口）。** 当前唯一主线是
> T1ce -> WT={1,2,3}，见 `T1CE_MAINLINE.md` 与 2026-08-10 handoff addendum。

## 2026-08-13 最新进展：师兄路线 vanilla baseline 复现完成

本阶段已经完成并冻结了一个可作为后续科研统一起点的 **vanilla SAM-Med3D
T1ce→WT baseline**。面向组内汇报或工作经历，可以表述为：

> 我完成了师兄路线对应的 SAM-Med3D baseline 复现：重建 BraTS 2024
> subject-clean 数据划分，对齐官方 1.5 mm 预处理和上游交互训练语义，在两张
> A100 80GB 上完成 200 epoch 全参数训练，并用 validation-best checkpoint 在
> 270 例 held-out test 上取得 best-of-5 DSC 0.5942。该结果与 MSCL Table 2
> 报告的 vanilla baseline 0.601 处于同一量级，现冻结为下一阶段方法研究的项目基线。

这里的“复现完成”是项目与师兄路线层面的复现结论：任务、数据清洗、官方预处理、模型、
训练循环和评测接口均已形成可审计闭环，结果与公开 vanilla baseline 的差值仅为
`-0.0068` DSC。论文没有披露完全相同的 train/validation/test 划分和 prompt 随机协议，
因此正式论文中应写“reproduced/established a clean vanilla baseline under our explicit
protocol”，不能写成逐项完全相同的 exact reproduction。MSCL Table 1 的 `0.731` 与其
灰度阈值损失增强配置一致，不再作为 vanilla SAM-Med3D 必须达到的目标。

### 我完成的工作

1. **纠正 baseline 目标。** 阅读并交叉核对 MSCL 表格后，将纯 SAM-Med3D 参照从此前
   误认为的约 `0.74` 修正为 Table 2 的 `0.601`；停止为了追 `0.74` 临时修改训练协议。
2. **重建数据与 subject-clean 划分。** 从 1,350 个 BraTS 2024 GLI scans、613 个
   longitudinal subjects 出发，按 `BraTS-GLI-xxxxx` subject 前缀隔离为：
   972 train scans / 444 subjects、108 validation scans / 50 subjects、270 test scans /
   119 subjects。三个 split 的 case overlap 和 subject overlap 均为 0。
3. **完成数据迁移和完整性验证。** 将当前 T1ce 与 segmentation 主线数据迁移到 VA HDFS，
   共 2,700 个 NIfTI 文件、约 91.35 GiB；源与目的端逐文件字节比较零 mismatch，训练不再
   依赖旧 NAS。
4. **对齐官方预处理。** 对 T1ce 图像与 WT 标签成对重采样到 `1.5×1.5×1.5 mm³`；图像
   使用线性插值、标签通过 TorchIO `LabelMap` 使用最近邻；随后 `ToCanonical`，以 GT 标签
   中心 `CropOrPad` 到 `128³`，最后按 patch 内正值图像体素的 sample mean/std 做 z-score
   并应用到整个 patch。不使用 N4；只在训练集使用三轴 `RandomFlip(p=0.5/axis)`，
   validation/test 无增强。train、validation、test 的 WT crop retention 最小值和均值均为
   1.0，没有病例因裁剪丢失前景。
5. **逐行审计并复现上游训练语义。** 初始化为官方推荐 SAM-Med3D Turbo checkpoint；
   image encoder、prompt encoder 和 mask decoder 共 `100,509,232/100,509,232` 参数全部
   解冻。实际训练使用上游硬编码的 11 次交互：每轮只传当前点并递归传入 low-resolution
   mask；零基索引 2–8 中随机一次不传 sparse point，索引 10 的最后一次也不传点；
   11 个 DiceCE loss 求和；
   小于等于 1,000 个前景体素的样本按上游逻辑递归替换。
6. **完成双卡正式训练。** 使用两张 A100-SXM4-80GB，per-GPU batch 12、gradient
   accumulation 20、AdamW、weight decay 0.1、encoder/head LR `8e-4/8e-5`、
   MultiStepLR milestones `120/180`、FP16 autocast + GradScaler、seed 2023、200 epochs。
   为保证双卡数学正确，显式执行 mean gradient all-reduce；为兼容当前 MONAI，只在
   DiceCE loss 边界把二值整数 GT 转成 float。epoch 1 有一次 GradScaler 溢出跳步，scale
   从 65,536 自动降至 32,768，此后所有更新正常，无 NaN 或异常退出。
7. **用 validation 选模而非看 test 调参。** validation 使用与上游 inference 对齐的五点击：
   click 1–2 为 RITM current point，click 3–5 为 uniform XOR current point，全程递归 mask，
   raw logit `>0` 输出；每个 case 使用 `SHA256(seed=2023, case_id)` 固定随机性。最佳
   validation best-of-5 为 `0.571306`（epoch 21），对应 final-click-5 为 `0.559185`。
   尽管 train Dice 最终达到 `0.799562`，仍严格选择 epoch 21 而非训练末期权重。
8. **完成一次性 270 例 test 评测与审计。** 在打开 test NIfTI 前冻结 manifest、checkpoint、
   预处理、五点击协议和代码 SHA；两张 A100 各处理 135 例，结果合并后验证 270/270 唯一、
   manifest 顺序一致、全部有限，且独立重算与报告汇总完全一致。test 没有参与训练、选模或
   推理适配。

### 最终 baseline 结果

| 指标 | 270-scan macro | 119-subject balanced |
|---|---:|---:|
| per-case best over clicks 1–5 | **0.594226** | **0.580970** |
| final click 5 | **0.585754** | **0.572699** |

Scan-macro 的 click 1→5 Dice 为：
`0.548900 → 0.566940 → 0.573730 → 0.579031 → 0.585754`。Best-of-5
的 median 为 `0.600321`，population std 为 `0.131231`。

### 当前结论与下一步边界

- **baseline 阶段已完成，不再调参。** 后续方法统一以 epoch-21 checkpoint 和上述结果为
  对照，不再改变本 baseline 的 batch、loss、split、prompt 或选模规则。
- 该结果说明标准预处理、全参数训练和官方交互接口已经正确形成闭环；它不是 `0.731`
  灰度阈值损失增强模型，也不是 label-free automatic segmentation。
- 270 例对本次 baseline 的训练和选择是 held-out，但该集合在项目更早历史中做过其他模型
  评估，因此应称为 **project-internal held-out test**，不能冒充从未访问的外部 test。
- 下一阶段的新方法只在 972/108 development 数据上设计与选型；在方法和比较协议冻结前，
  不再利用这 270 例作 architecture、loss、学习率或后处理选择。若需要强外部结论，应另建
  pristine external/confirmatory set。

正式逐病例报告位于：
`workspace/BraTS/sammed3d_t1ce_wt/sammed3d_upstreamliteral11_officialval5_va_a100x2_train972_val108_e200_s2023/test270_epoch21_official15_upstreamliteral5_s2023_20260812/test_report.json`，
SHA-256 为
`7bd59843459876bd006230b24bc111cdfd726896cb3e6748b54b24c0a9659b63`。
冻结的 epoch-21 checkpoint SHA-256 为
`c0c85a0fb90f6301344053f97cabe8a4972bd9febd4c6ae248f3a0908598fe31`。

更新日期：2026-08-13

## 一句话目标

本项目分成两条工作线。

**work #1 已即时重启并更名**：论文级方法名为 **MedTAWS（Medical Tree-guided Adaptive
Workflow Search，医学树引导自适应工作流搜索）**。stochastic UCB-guided 搜索算法保留，
但旧单模态、GT corrective-click、binary-target 的 BraTS 接入与旧 Dice 已退出论文主证据；
现在从四模态联合输入、无 GT prompt、自动五类协议重新验证。它不冒充标准 MCTS。

**work #2 当前主线已改为外部 SOTA 对标**：训练四模态、全自动、多类 SAM-Med3D，按
BraTS 2024 官方区域定义和 lesion-wise Dice/HD95 与强 nnU-Net/MedNeXt 基线同协议比较。
现有 5-click binary SAM 结果只保留为 interactive prompted delineation 支线，不再作为
“超过 BraTS 自动分割论文”的主指标。

> 下文较早的 round 18、threshold、TTA、morphology 和旧 Work #1 “完成”叙述只作历史账本；
> 不得据此恢复旧搜索或覆盖本页最新追加节。

**2026-07-17 口径更正**：下文截至 2026-07-07 的 `0.58932` 是旧 200-scan
scan-level 划分上的历史结果。旧数据实际只有 39 个纵向 subjects，存在 train-val 和
train-test subject 重叠；它不再是当前 subject-clean best，也不作为新协议晋级线。

详细流水账放在：

```text
AFLOW_BRATS_RESULTS.md
AFLOW_BRATS_HANDOFF.md
```

## work #1：搜索目标是什么

work #1 的问题不是“怎么训练一个更强的 SAM-Med3D”，而是更窄：

> 在固定 SAM-Med3D-turbo checkpoint 不变的前提下，能不能只通过更聪明的推理流程，把 zero-shot 分割 Dice 往上抬一点？

这里的 `workflow` 指一整套推理配方。它包括：同一个病例跑几次、用哪些seed、保留 click 过程里的哪张概率图、怎么融合概率图、用什么阈值变成 mask、做不做形态学清理、最后怎么从多个候选 mask 里选最终答案。

最初搜索目标是 `tumor_ET`，因为它是 BraTS 标准三目标 WT/TC/ET 中最弱的目标之一，zero-shot Dice 只有约 0.33。搜索完成后，再把搜到的 workflow放到 full target set 上评估：

```text
tumor_WT, tumor_TC, tumor_ET, label_SNFH, label_RC, label_NETC
```

搜索约束：

- 不训练、不微调 SAM-Med3D。
- 不加载额外学习到的分割权重。
- 不用 GT 选择阈值、候选 mask 或后处理参数。
- GT 只用于 SAM-Med3D 官方 click prompt 生成和最终 Dice 评估。

## work #1：搜索空间怎么理解

这次 AFlow 搜索的空间可以用大白话理解成 6 类旋钮：

| 搜索项 | 大白话解释 | 想解决的问题 |
|---|---|---|
| seed ensemble | 同一个病例用不同随机种子跑多遍，再把结果融合 | 降低单次推理的随机波动 |
| click trajectory | SAM-Med3D 是交互式分割，5 次 click 会产生一串中间结果 | 判断用最后一张图，还是中间图也有价值 |
| smoothing | 对概率图做轻微高斯平滑 | 去掉局部毛刺，让边界不要太锯齿 |
| gamma calibration | 对概率值做非线性校准，`gamma=0.9` 会稍微抬高中等置信度体素 | 减少边界区域被固定阈值误删 |
| threshold sweep | 不只用一个阈值，而是用 `0.40/0.45/0.50/0.55` 生成多张候选 mask | 避免赌单一阈值刚好最优 |
| morphology + consensus | 先清理小噪点、补洞、连小缝，再做体素级投票 | 保留稳定病灶，删除偶然假阳性 |

注意：搜索空间里包含 click trajectory 的中间概率图，但最后 `round_18`
收敛到的有效做法是使用每个 seed 的 final probability map。

## work #1：搜到的 round_18 workflow

主结果是 `round_18`，核心流程如下：

1. 使用 SAM-Med3D-turbo 固定 checkpoint。
2. 每个病例跑 3 个 seed：`2023, 2024, 2025`。
3. 每个 seed 使用默认 5-click prompt 策略。
4. 每个 seed 取第 5 次 click 后的 final probability map。
5. 对 3 张 final probability map 做平均，得到一张更稳定的平均概率图。
6. 对平均概率图做轻量校准：
   - Gaussian smoothing：`smooth_sigma=0.8`
   - gamma transform：`gamma=0.9`
7. 用多个阈值生成候选 mask：
   - `0.40, 0.45, 0.50, 0.55`
8. 对每张候选 mask 做轻量 morphology：
   - `min_component_voxels=8`
   - `close_iters=1`
   - `fill_holes=True`
9. 最后用 voxel-wise consensus voting 输出最终 mask。

这条 workflow 的核心思想是：不相信一次 seed、不相信一个阈值，也不直接相信
原始概率图。先用多 seed 平均降低随机性，再用轻量校准处理边界不确定性，
然后用多阈值产生“宽松到严格”的候选答案，最后通过投票保留多个候选都支持的
体素。

## work #1：一个 case 的直观例子

以一个有明显肿瘤主体和模糊边界的 BraTS 病例为例。SAM-Med3D 输出的不是
直接的 0/1 mask，而是一张 3D probability map：每个 voxel 都有一个 0 到 1
之间的分数，表示模型认为这个 voxel 属于目标区域的置信度。

假设某个肿瘤主体 voxel 在 3 个 seed 下的概率分别是：

```text
seed2023: 0.82
seed2024: 0.76
seed2025: 0.80
average : 0.79
```

这个 voxel 三次都很高，说明它是稳定信号。后面不管用 `0.40`、`0.45`、
`0.50` 还是 `0.55` 阈值，它都会进入候选 mask，最终投票时也会留下。

再看一个边界 voxel：

```text
seed2023: 0.48
seed2024: 0.39
seed2025: 0.52
average : 0.46
gamma=0.9 后约为 0.50
```

如果只用固定阈值 `0.50`，这个 voxel 很可能被删掉；但 `round_18` 会同时
看 `0.40/0.45/0.50/0.55` 多个候选。它在宽松阈值里会被保留，在严格阈值里
可能被删掉，最后由 consensus voting 根据多个候选的支持程度决定是否留下。
这比“只押一个阈值”更稳。

再看一个随机噪声 voxel：

```text
seed2023: 0.51
seed2024: 0.06
seed2025: 0.04
average : 0.20
```

这个 voxel 只有一次 seed 误报，平均以后概率会被压低，基本过不了阈值。
即使有一小片噪声刚好过了阈值，`min_component_voxels=8` 也会把太小的连通域
删掉；`close_iters=1` 会轻微连接真实区域里的小缝；`fill_holes=True` 会补掉
mask 内部的小空洞。

所以 `round_18` 的实际效果可以概括成一句话：

> 稳定出现的病灶被保留，单次 seed 造成的随机噪声被平均和形态学清理掉，边界上不确定的 voxel 通过多阈值和投票来处理。

## work #1：关键结果

`round_18` 在标准 WT/TC/ET headline metric 上超过了 zero-shot baseline：

| Metric | Baseline | round_18 | Delta |
|---|---:|---:|---:|
| WT/TC/ET average | 0.4699 | 0.4734 | +0.0035 |
| WT/TC/ET case-weighted | 0.4972 | 0.5017 | +0.0045 |

分目标看：

| Target | Baseline | round_18 | 结论 |
|---|---:|---:|---|
| `tumor_WT` | 0.71605 | 0.72810 | 提升 |
| `tumor_TC` | 0.35999 | 0.35661 | 基本持平略降 |
| `tumor_ET` | 0.33359 | 0.33552 | 小幅提升 |
| `label_SNFH` | 0.72191 | 0.72304 | 小幅提升 |
| `label_RC` | 0.45536 | 0.42985 | 明显不足 |
| `label_NETC` | 0.10411 | 0.10106 | 仍很弱 |

后续 target-aware 固定外壳：

```text
workspace/BraTS/workflows/round_31/graph.py   # WT 最好，full WT = 0.73482
workspace/BraTS/workflows/round_30/graph.py   # RC mirror TTA，full RC = 0.44957
workspace/BraTS/workflows/round_23/graph.py   # TC / NETC 的冻结分支参考
workspace/BraTS/workflows/round_18/graph.py   # full-target reported baseline
```

结论：  
work #1 证明了 no-training workflow 搜索能带来小幅提升，尤其是 WT / ET /headline metric。但对 `TC/ET/RC/NETC` 这些弱目标，单靠后处理和推理 workflow已经不够，必须进入训练权重的 work #2。

## work #2：当前训练目标

当前不再继续调后处理，headline 目标也不再是内部 5-click binary Dice。正式目标是：

- 输入 T1n/T1c/T2w/T2f 四模态，整卷自动输出 ET/NETC/SNFH/RC 四类；
- 派生 `TC={ET,NETC}` 和官方 `WT={ET,NETC,SNFH}`，WT 不含 RC；
- 对所有病例计分，包括 target 缺失病例，报告官方 lesion-wise Dice/HD95；
- 在完全相同的 subject-disjoint folds 上先训练 nnU-Net/MedNeXt 强基线；
- SAM 自动方案必须超过同协议强基线，才进入冻结后的独立外部 holdout；
- 当前 sealed test 已消费且禁止重试，不能再用于模型选择或结论。

现有 partial-decoder candidate 仍可作为 5-click interactive 支线结果：两 seed strict
clean-val subject-balanced 六目标 macro 为 `0.554582`。该结果使用 GT-error corrective
clicks、单模态、foreground-positive rows 和 legacy WT，因此不能与官方论文直接排名。

## work #2：历史 200-scan 基础设施（已被 2026-07-17 全量协议取代）

本节以下 140/20/40 patient 统计来自当时把完整 `case_id` 误当 patient id 的旧 split，
不能继续用于 subject-disjoint 结论。当前有效 split 见文末 2026-07-17 小节。

已建立 patient-level split：

```text
data/datasets/brats_patient_split_seed2026/
data/datasets/brats_patient_split_seed2026/split_manifest.json
```

划分无病人级泄漏：

| Split | Patients | label_NETC | label_RC | tumor_ET | tumor_TC | tumor_WT | label_SNFH |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 140 | 48 | 118 | 102 | 103 | 140 | 140 |
| val | 20 | 6 | 17 | 14 | 14 | 20 | 20 |
| test | 40 | 13 | 33 | 29 | 29 | 40 | 40 |

新增训练入口：

```text
scripts/train_sammed3d_brats.py
scripts/run_worker_research.sh train ...
```

当前默认训练策略：

- `decoder_only`：训练 `prompt_encoder + mask_decoder`，冻结 `image_encoder`。
- loss：MONAI `DiceCELoss`。
- click 协议：固定 5-click。
- batch 默认：`batch_size=1`。
- checkpoint 格式兼容现有 eval harness。

新增 checkpoint-aware eval：

```text
sammed3d/engine.py
scripts/evaluate_brats_file.py --checkpoint ...
scripts/evaluate_brats_multi_target.py --checkpoint ...
```

它只替换模型权重，不改变 fixed workflow 后处理外壳。

## work #2：M0 sanity

已完成一个小样本 overfit sanity：

| Run | Target | Cases | Strategy | Best Dice | Best Loss | Best Epoch |
|---|---|---:|---|---:|---:|---:|
| `m0_label_RC_overfit4` | `label_RC` | 4 | decoder + prompt encoder | 0.75736 | 0.19862 | 17 |

产物：

```text
workspace/BraTS/training/m0_label_RC_overfit4/
```

解释：训练链路已经打通，包括数据加载、forward、loss、反传、保存 checkpoint、
checkpoint-aware eval。这个 M0 还不是“Dice 接近 1”的强过拟合，所以 M1 前
建议继续把 M0 压得更扎实。

一个失败试验不要用：

```text
workspace/BraTS/training/m0_label_RC_overfit1/DO_NOT_USE.txt
```

该 run 是 `lr=0.001` 的 1-case overfit，训练不稳定，中途停止。

## checkpoint-aware eval smoke

同一个 1-case `label_RC` val subset，同一个 fixed `round_31` workflow：

| Model | Dice |
|---|---:|
| zero-shot SAM-Med3D turbo | 0.43958 |
| M0 trained checkpoint | 0.52461 |

这只是 smoke test，不是论文结论。正式结论必须在 clean test fold 上比较。

## 新方向：把 GRPO 思想迁移到分割训练

用户希望把 GRPO 这类强化学习训练范式迁移到 SAM-Med3D 权重训练。

初步迁移思路：

1. 不直接跳过 supervised PEFT。先拿到稳定的 M0/M1 supervised checkpoint。
2. 把一个 BraTS case + target 看成一个 prompt/query。
3. 对同一病例采样一组 segmentation rollouts：
   - 不同 click seed
   - 不同 prompt trajectory
   - 可选 dropout / stochastic decoder
4. 每个 rollout 得到一个 mask，用医学分割 reward 打分：
   - Dice / soft Dice
   - 小结构弱目标加权：`TC/ET/RC/NETC`
   - 空 mask、碎片化、体积异常惩罚
   - WT / SNFH 不退化惩罚
5. GRPO 的 group-relative advantage 可迁移为：
   - 同一病例 K 个 rollout 的 reward 减去 group mean
   - 不训练 critic
   - 用 normalized advantage 更新 prompt encoder / mask decoder
6. 加 KL / reference penalty，避免模型偏离 supervised checkpoint 太远。

关键边界：

- GRPO 不能变成新的后处理搜索，必须真的更新模型权重。
- 不能在 test fold 上调 reward 权重或选 checkpoint。
- 不能同时改 workflow 外壳和模型权重，否则无法判断提升来源。

## GRPO 方向：已完成的最小实现

已新增脚本：

```text
scripts/train_sammed3d_grpo.py
scripts/run_worker_research.sh grpo-train ...
```

当前实现不是照搬 LLM token 训练，而是迁移 GRPO 的核心结构：

1. 一个 BraTS case + target 作为一个 query。
2. 对同一病例采样 K 个 segmentation rollouts。
3. 每个 rollout 由不同 click seed / prompt trajectory 产生一个 mask。
4. 用 Dice、体积惩罚、空 mask 惩罚、碎片化惩罚构造 reward。
5. 对同一病例内部做 group-relative advantage。
6. 用 Bernoulli mask log-prob 的 PPO/GRPO-style surrogate 更新
   `prompt_encoder + mask_decoder`。
7. 加 reference KL penalty 和少量 supervised anchor，避免训练发散。

已完成 smoke：

| Run | 目的 | Target | Cases | Group size | 结果 |
|---|---|---|---:|---:|---|
| `audit_label_RC_2case` | 离线 reward audit | `label_RC` | 2 val | 4 | reward std = 0.130，有 group 信号 |
| `smoke_label_RC_2case` | 1 epoch GRPO-style 训练 | `label_RC` | 2 train / 2 val | 2 | 训练和保存 checkpoint 跑通 |
| `label_RC_m0init_8train_4val_e3` | M0 初始化后的 GRPO 短训练 | `label_RC` | 8 train / 4 val | 2 | val reward / Dice 有正向变化 |

同 seed、同 2 个 val case、group size = 2 的 audit 对比：

| Checkpoint | Reward | Dice |
|---|---:|---:|
| zero-shot SAM-Med3D turbo | 0.24888 | 0.24802 |
| GRPO smoke checkpoint | 0.43717 | 0.41253 |

同 seed、同 4 个 val case、group size = 2，M0 初始化前后对比：

| Checkpoint | Reward | Dice |
|---|---:|---:|
| M0 supervised checkpoint | 0.41310 | 0.35743 |
| GRPO 3-epoch checkpoint | 0.47504 | 0.40753 |

产物：

```text
workspace/BraTS/grpo/audit_label_RC_2case/
workspace/BraTS/grpo/audit_label_RC_2case_g2_zeroshot/
workspace/BraTS/grpo/smoke_label_RC_2case/
workspace/BraTS/grpo/audit_label_RC_2case_g2_trained/
workspace/BraTS/grpo/audit_label_RC_4val_m0init_g2/
workspace/BraTS/grpo/label_RC_m0init_8train_4val_e3/
workspace/BraTS/grpo/audit_label_RC_4val_grpo_e3_g2/
```

注意：这只是 GRPO 方向的工程 smoke 和小样本体检，不能作为论文结论。
下一步需要先扩大到 clean val，再决定是否进入正式 M1/M2-RL。

## 下一步

1. 继续加强 M0，让 1-4 个 `label_RC` / `tumor_TC` case 真正高 Dice 过拟合。
2. 跑 M1 decoder-only PEFT，先主攻 `label_RC` 或 `tumor_TC`。
3. 在 patient-level test fold 上比较 zero-shot vs trained checkpoint。
4. 同步设计 GRPO-style segmentation trainer，先做离线 reward audit，再做 RL 更新。

---

## 2026-06-30 GRPO clean 16/8 short run

目标：把 GRPO 的 group-relative 训练思想迁移到 SAM-Med3D 权重更新，而不是继续做后处理搜索。

干净实验目录：

```text
workspace/BraTS/grpo/label_RC_m0init_16train_8val_e5_clean_20260630/
```

同一 8-val、同 seed、同 audit-only 口径：

| Checkpoint | Reward | Dice |
|---|---:|---:|
| M0 init | 0.28598 | 0.26830 |
| supervised 16/8 e5 best | 0.29478 | 0.28344 |
| GRPO 16/8 e5 best | 0.31907 | 0.29956 |

结论：GRPO 当前有小幅正信号，并且在统一 audit 口径下超过同规模 supervised 对照。
这仍只是 clean-val 小样本结果；下一步应扩大到 32/16 或多 seed 复核，再决定是否上 test fold。

---

## 2026-06-30 GRPO 32/16 clean-val 复核

设置：`label_RC`，M0 初始化，32 train / 16 val，5 epoch，seed 2026。

同一 16-val audit-only 口径：

| Checkpoint | Reward | Dice |
|---|---:|---:|
| M0 init | 0.30776 | 0.29950 |
| supervised 32/16 e5 best | 0.47535 | 0.42186 |
| GRPO 32/16 e5 best | 0.47818 | 0.42782 |

结论：GRPO 与 supervised 都明显优于 M0；GRPO 略高于 supervised，但优势很小。
下一步应做多 seed 复核，不应直接上 test fold 或写强结论。

---

## 2026-06-30 GRPO 32/16 多 seed 复核

同一 16-val audit-only 口径：

| Seed | M0 Dice | Supervised Dice | GRPO Dice |
|---:|---:|---:|---:|
| 2026 | 0.29950 | 0.42186 | 0.42782 |
| 2027 | 0.29950 | 0.42676 | 0.41756 |

两 seed 均值：supervised Dice 0.42431，GRPO Dice 0.42269。

结论：GRPO 能训练权重并稳定超过 M0，但目前没有稳定超过同规模 supervised。
下一步应做 GRPO 方法学 ablation，例如 group size 4、降低 supervised anchor、重审 reward，而不是直接上 test fold。

---

## 当前实验结论摘要

目前结论：GRPO-style 训练已经成功迁移到 SAM-Med3D/BraTS 权重更新，能稳定超过 M0 起点；
但在 32 train / 16 val 的两 seed 复核中，没有稳定超过同规模 supervised decoder-only。

关键数字：两 seed 均值 supervised Dice 0.42431，GRPO Dice 0.42269。

因此下一步不是上 test fold，而是在 clean-val 上做 GRPO 方法学 ablation：group size、supervised anchor、reward penalty。

---

## 2026-06-30 GRPO g4 anchor0.05 ablation seed 2026

设置：`label_RC`，32 train / 16 val，M0 初始化，`group_size=4`，`supervised_anchor=0.05`，seed 2026。

reload audit-only 结果：reward 0.50245，Dice 0.44541。

对比：supervised seed2026 Dice 0.42186，旧 GRPO g2 Dice 0.42782。

结论：g4 + 低 anchor 在 seed 2026 上给出当前最强 clean-val 信号；需要 seed 2027 复核，不能直接上 test fold。

---

## 2026-06-30 GRPO g4 anchor0.05 多 seed 复核

设置：`label_RC`，32 train / 16 val，M0 初始化，`group_size=4`，`supervised_anchor=0.05`。

同一 16-val audit-only 口径：

| Seed | Supervised Dice | GRPO g2 Dice | GRPO g4 anchor0.05 Dice |
|---:|---:|---:|---:|
| 2026 | 0.42186 | 0.42782 | 0.44541 |
| 2027 | 0.42676 | 0.41756 | 0.43855 |

两 seed 均值：supervised Dice 0.42431，GRPO g2 Dice 0.42269，GRPO g4 anchor0.05 Dice 0.44198。

结论：g4 + 低 anchor 是目前第一个两 seed 都超过 supervised 的 GRPO 配置。下一步应做单变量 ablation，拆分 group size 与 anchor 的贡献。

---

## 2026-06-30 GRPO anchor 单变量 ablation

seed 2026，32 train / 16 val，reload audit-only：

| Method | Reward | Dice |
|---|---:|---:|
| GRPO g2 anchor0.10 | 0.47818 | 0.42782 |
| GRPO g2 anchor0.05 | 0.49447 | 0.43290 |
| GRPO g4 anchor0.05 | 0.50245 | 0.44541 |

结论：低 anchor 有帮助，但不能完全解释 g4 anchor0.05 的提升；需要继续跑 g4 anchor0.10。

---

## 2026-06-30 GRPO group-size 单变量 ablation

seed 2026，32 train / 16 val，reload audit-only：

| Method | Reward | Dice |
|---|---:|---:|
| GRPO g2 anchor0.10 | 0.47818 | 0.42782 |
| GRPO g2 anchor0.05 | 0.49447 | 0.43290 |
| GRPO g4 anchor0.10 | 0.49462 | 0.43922 |
| GRPO g4 anchor0.05 | 0.50245 | 0.44541 |

结论：group size 4 与低 anchor 都有贡献，组合后最强。继续大跑前建议先清理重复 checkpoint。

---

## 2026-06-30 磁盘清理与继续方向

系统盘已清理：`/` 从 114G used / 12G avail / 92% 降到 104G used / 22G avail / 84%。

删除内容：受污染旧 GRPO 目录、重复 latest/reward/loss checkpoint、`__pycache__`。
保留内容：所有有效 run 的 `sam_model_val_dice_best.pth`、M0、supervised baseline、GRPO ablation best 权重、原始数据和 venv。

继续方向：补跑 `group_size=4 + supervised_anchor=0.10` 的 seed 2027，复核 group size 单变量贡献是否稳定；仍只用 clean-val，不上 test fold。

---

## 2026-06-30 GRPO group-size 多 seed 复核

补跑：`group_size=4 + supervised_anchor=0.10`，seed 2027，32 train / 16 val。

reload audit-only：reward 0.48917，Dice 0.43618。对比同 seed supervised Dice 0.42676、旧 g2 anchor0.10 Dice 0.41756。

两 seed 均值：

| Method | Dice mean |
|---|---:|
| Supervised 32/16 | 0.42431 |
| GRPO g2 anchor0.10 | 0.42269 |
| GRPO g4 anchor0.10 | 0.43770 |
| GRPO g4 anchor0.05 | 0.44198 |

结论：group size 4 本身稳定有效；anchor 0.05 仍是当前最优。下一步只需小范围看 anchor 0.03 是否还能提升，否则应转向更大 train cases。

---

## 2026-07-03 GRPO anchor0.03 与当前交接结论

补跑：`group_size=4 + supervised_anchor=0.03`，seed 2026，32 train / 16 val。

reload audit-only：reward 0.49803，Dice 0.44097。

对比 seed2026：

| Method | Dice |
|---|---:|
| GRPO g4 anchor0.10 | 0.43922 |
| GRPO g4 anchor0.05 | 0.44541 |
| GRPO g4 anchor0.03 | 0.44097 |

结论：anchor 0.03 没有超过 0.05，不扩第二个 seed。当前主配置仍是 `group_size=4 + supervised_anchor=0.05`。

当前交接状态：GRPO 训练方向已经跑通并有稳定正结果；不要回到后处理搜索。下一步建议把 g4 anchor0.05 扩到更多 train cases，继续只用 clean-val 和 reload audit 口径，暂不上 test fold。

---

## 2026-07-03 GRPO g4 anchor0.05 扩样本到 64/96

设置：`label_RC`，M0 初始化，`group_size=4`，`supervised_anchor=0.05`，5 epoch，
同一 patient-level split，同一 16-val clean-val，不使用 test fold。

执行方式：训练脚本不支持单进程多卡。本轮在 2 张 A10 上并行跑两个独立 seed：
seed 2026 用 `cuda:0`，seed 2027 用 `cuda:1`。

reload audit-only 汇总：

| Method | Train / Val | Seed 2026 Dice | Seed 2027 Dice | Dice mean |
|---|---:|---:|---:|---:|
| Supervised best | 32 / 16 | 0.42186 | 0.42676 | 0.42431 |
| GRPO g4 anchor0.05 | 32 / 16 | 0.44541 | 0.43855 | 0.44198 |
| GRPO g4 anchor0.05 | 64 / 16 | 0.50979 | 0.51516 | 0.51247 |
| GRPO g4 anchor0.05 | 96 / 16 | 0.53206 | 0.54348 | 0.53777 |

训练内 best：

| Train cases | Seed | Best epoch | Val reward | Val Dice |
|---:|---:|---:|---:|---:|
| 64 | 2026 | 5 | 0.56586 | 0.49641 |
| 64 | 2027 | 4 | 0.60033 | 0.52152 |
| 96 | 2026 | 5 | 0.61287 | 0.53394 |
| 96 | 2027 | 5 | 0.62954 | 0.54855 |

结论：扩样本是强正信号。64 train 已明显超过 32 train，96 train 又超过 64 train；
当前 clean-val 最强为 `GRPO g4 anchor0.05, 96 train / 16 val`，reload audit Dice mean 0.53777。

清理：本轮 64/96 四个训练目录均已删除 `sam_model_latest.pth` 和
`sam_model_val_reward_best.pth`，只保留 `sam_model_val_dice_best.pth`。

下一步：若继续训练，优先同配置跑 full train（当前 split 下 `label_RC` train 共 118 cases）；
若准备上 test，必须先冻结配置，不能在 test 上继续调 epoch、reward、后处理或训练样本数。

---

## 2026-07-03 GRPO g4 anchor0.05 扩到 full train

设置：`label_RC`，M0 初始化，`group_size=4`，`supervised_anchor=0.05`，5 epoch，
118 train / 16 val，同一 patient-level split，同一 clean-val reload audit-only，不使用 test fold。

reload audit-only 汇总：

| Method | Train / Val | Seed 2026 Dice | Seed 2027 Dice | Dice mean |
|---|---:|---:|---:|---:|
| Supervised best | 32 / 16 | 0.42186 | 0.42676 | 0.42431 |
| GRPO g4 anchor0.05 | 32 / 16 | 0.44541 | 0.43855 | 0.44198 |
| GRPO g4 anchor0.05 | 64 / 16 | 0.50979 | 0.51516 | 0.51247 |
| GRPO g4 anchor0.05 | 96 / 16 | 0.53206 | 0.54348 | 0.53777 |
| GRPO g4 anchor0.05 | 118 / 16 | 0.55016 | 0.54380 | 0.54698 |

full-train 训练内 best：

| Train cases | Seed | Best epoch | Val reward | Val Dice |
|---:|---:|---:|---:|---:|
| 118 | 2026 | 5 | 0.64104 | 0.55336 |
| 118 | 2027 | 5 | 0.62052 | 0.54254 |

full-train reload audit-only：

| Seed | Reward | Dice |
|---:|---:|---:|
| 2026 | 0.63555 | 0.55016 |
| 2027 | 0.62623 | 0.54380 |
| mean | 0.63089 | 0.54698 |

结论：扩样本曲线仍然单调变好。当前 clean-val 最强为
`GRPO g4 anchor0.05, 118 train / 16 val`，reload audit Dice mean 0.54698；
相对 96 train 提升 +0.00921，相对 32-case g4 anchor0.05 提升 +0.10500，
相对 supervised 32/16 提升 +0.12267。两个 seed 的 best epoch 都在 epoch 5，
暂未显示出扩样本后的明显过拟合。

清理：两个 full-train 训练目录均已删除 `sam_model_latest.pth` 和
`sam_model_val_reward_best.pth`，只保留 `sam_model_val_dice_best.pth`。

论文启发的下一步（仍属于 work #2，训练/微调权重，不回到 inference 后处理）：

1. 优先做 **组内偏好优化 / DPO-style ablation**：参考 SAMPO、SAM+DPO、MAPO，
   在当前 GRPO group 内拿 reward best-vs-worst rollout 构造 pairwise preference loss，
   保留 anchor0.05 与 reload audit 口径。
2. 其次做 **rollout diversity**：只在训练时引入 decoder dropout 或轻量 mask perturbation，
   让组内候选更像 MAPO 的 stochastic hypotheses；eval 不改。
3. 小范围试 **空间奖励项**：参考 MedGround-R1 和 staged voxel-level RL，把 boundary/surface Dice
   或 center-of-mass reward 作为小系数辅助项；这仍然优化训练权重，不碰 test-time postprocess。
4. 对 supervised anchor 试 **置信度/边界降权**：参考 probabilistic-aware weak supervision，
   降低边界不确定 voxels 的 anchor 权重，避免 RC 边界噪声过度牵制策略更新。

参考方向：SAM-Med3D (https://arxiv.org/abs/2310.15161)，SAMPO
(https://arxiv.org/abs/2508.02464)，SAM+DPO
(https://arxiv.org/abs/2503.04639)，MAPO (https://arxiv.org/abs/2512.15009)，
MedGround-R1 (https://arxiv.org/abs/2507.02994)，staged voxel-level RL
(https://arxiv.org/abs/2601.03875)，interactive 3D medical segmentation with multi-agent RL
(https://arxiv.org/abs/1911.10334)，probabilistic-aware weakly supervised 3D segmentation
(https://arxiv.org/abs/2403.02566)。

---

## 2026-07-03 DPO-style pairwise preference ablation

设置：`label_RC`，M0 初始化，32 train / 16 val，`group_size=4`，
`supervised_anchor=0.05`，同一 clean-val reload audit-only，不使用 test fold。

实现：`scripts/train_sammed3d_grpo.py` 新增
`--objective {grpo,pairwise,grpo_pairwise}`，默认仍为 `grpo`。`pairwise` 用组内
reward best-vs-worst rollout 构造 DPO-style loss；`grpo_pairwise` 保留原 GRPO，
额外加 `preference_coef * pairwise_loss`。本轮使用 `preference_beta=8.0`，
hybrid 使用 `preference_coef=0.05`。

reload audit-only 汇总：

| Method | Train / Val | Seed 2026 Dice | Seed 2027 Dice | Dice mean |
|---|---:|---:|---:|---:|
| Supervised best | 32 / 16 | 0.42186 | 0.42676 | 0.42431 |
| GRPO g4 anchor0.05 | 32 / 16 | 0.44541 | 0.43855 | 0.44198 |
| Pure pairwise beta8 | 32 / 16 | 0.41357 | 0.42373 | 0.41865 |
| GRPO + pairwise coef0.05 beta8 | 32 / 16 | 0.42429 | 0.43048 | 0.42739 |
| GRPO g4 anchor0.05 | 118 / 16 | 0.55016 | 0.54380 | 0.54698 |

训练内 best：

| Method | Seed | Best epoch | Val reward | Val Dice |
|---|---:|---:|---:|---:|
| Pure pairwise beta8 | 2026 | 3 | 0.46923 | 0.41846 |
| Pure pairwise beta8 | 2027 | 2 | 0.47245 | 0.42469 |
| GRPO + pairwise coef0.05 beta8 | 2026 | 4 | 0.48349 | 0.42810 |
| GRPO + pairwise coef0.05 beta8 | 2027 | 3 | 0.49196 | 0.43928 |

结论：纯 pairwise 是负结果；hybrid 略高于 supervised，但低于旧 GRPO 32-case，
且 seed2026 明显拖后腿。当前 pairwise 写法不扩样本。主结论不变：
当前 clean-val 最强仍是 `GRPO g4 anchor0.05, 118 train / 16 val`，Dice mean 0.54698。

清理：pairwise、hybrid、smoke 训练目录均只保留 `sam_model_val_dice_best.pth`；
重复 `latest/reward` checkpoint 已删除。两张 A10 空闲。

下一步建议：不继续扩 pairwise；如果继续缝合论文 idea，优先试训练侧几何 reward
（boundary/surface Dice 或 center-of-mass 小系数 reward），仍然只在 clean-val 内做 ablation。

---

## 2026-07-03 center-of-mass spatial reward ablation

设置：`label_RC`，M0 初始化，32 train / 16 val，`group_size=4`，
`supervised_anchor=0.05`，5 epoch，同一 clean-val reload audit-only，不使用 test fold。

实现：`scripts/train_sammed3d_grpo.py` 新增 `--center-reward-coef`，默认 `0.0`。
本轮使用 `center_reward_coef=0.05`，在 reward 中加入中心点对齐小项。reward 数字包含
新增中心项，因此主要看 Dice。

reload audit-only 汇总：

| Method | Train / Val | Seed 2026 Dice | Seed 2027 Dice | Dice mean |
|---|---:|---:|---:|---:|
| Supervised best | 32 / 16 | 0.42186 | 0.42676 | 0.42431 |
| GRPO g4 anchor0.05 | 32 / 16 | 0.44541 | 0.43855 | 0.44198 |
| GRPO + center reward coef0.05 | 32 / 16 | 0.42642 | 0.43188 | 0.42915 |
| GRPO g4 anchor0.05 | 118 / 16 | 0.55016 | 0.54380 | 0.54698 |

训练内 best：

| Seed | Best epoch | Val reward | Val Dice |
|---:|---:|---:|---:|
| 2026 | 4 | 0.53352 | 0.42983 |
| 2027 | 3 | 0.52725 | 0.42790 |

结论：center reward 略高于 supervised，但低于原始 GRPO 32-case，不扩样本。
center-of-mass 奖励太粗，不能替代 Dice reward 本身对 overlap/边界的直接约束。
当前 clean-val best 仍是 `GRPO g4 anchor0.05, 118 train / 16 val`，Dice mean 0.54698。

清理：center reward 两个训练目录均只保留 `sam_model_val_dice_best.pth`；
重复 `latest/reward` checkpoint 已删除。test fold 仍未使用。

---

## 2026-07-06 原始 GRPO full-train 8 epoch

设置：`label_RC`，M0 初始化，原始 `grpo` objective，118 train / 16 val，
`group_size=4`，`supervised_anchor=0.05`，同一 patient-level split，同一 clean-val
reload audit-only，不使用 test fold。目的只是确认 full-train 5 epoch 是否欠训练。

训练内 best：

| Seed | Best epoch | Val reward | Val Dice |
|---:|---:|---:|---:|
| 2026 | 7 | 0.67535 | 0.58189 |
| 2027 | 8 | 0.65702 | 0.57021 |
| mean | - | 0.66618 | 0.57605 |

reload audit-only 汇总：

| Method | Train / Val | Epochs | Seed 2026 Dice | Seed 2027 Dice | Dice mean |
|---|---:|---:|---:|---:|---:|
| Supervised best | 32 / 16 | 5 | 0.42186 | 0.42676 | 0.42431 |
| GRPO g4 anchor0.05 | 32 / 16 | 5 | 0.44541 | 0.43855 | 0.44198 |
| GRPO g4 anchor0.05 | 96 / 16 | 5 | 0.53206 | 0.54348 | 0.53777 |
| GRPO g4 anchor0.05 | 118 / 16 | 5 | 0.55016 | 0.54380 | 0.54698 |
| GRPO g4 anchor0.05 | 118 / 16 | 8 | 0.57084 | 0.56763 | 0.56923 |

结论：按当时的旧 scan-level 划分，clean-val 最强更新为
`GRPO g4 anchor0.05, 118 train / 16 val, 8 epoch`，reload Dice mean 0.56923。
相对 5 epoch full-train 提升 +0.02225，相对 supervised 32/16 提升 +0.14492。

因此，前面尝试的 pure pairwise、GRPO+pairwise、center-of-mass reward 目前都不如
原始 GRPO 主线；最值得继续推进的是原始 GRPO 的训练时长/吞吐优化，以及更直接的
boundary/surface 类训练侧 reward 小实验。

清理：两个 8 epoch 训练目录均只保留 `sam_model_val_dice_best.pth`；
重复 `latest/reward` checkpoint 已删除。test fold 仍未使用。

---

## 2026-07-06 A100 full-train 10 epoch

设置：单张 A100-SXM4-80GB，`label_RC`，M0 初始化，原始 `grpo` objective，
118 train / 16 val，`group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`，
10 epoch，`num_workers=2`，同一 clean-val reload audit-only，不使用 test fold。

训练内 best：

| Seed | Best epoch | Val reward | Val Dice |
|---:|---:|---:|---:|
| 2026 | 10 | 0.69022 | 0.59192 |
| 2027 | 9 | 0.68534 | 0.58999 |
| mean | - | 0.68778 | 0.59095 |

reload audit-only 汇总：

| Method | Train / Val | Epochs | Seed 2026 Dice | Seed 2027 Dice | Dice mean |
|---|---:|---:|---:|---:|---:|
| Supervised best | 32 / 16 | 5 | 0.42186 | 0.42676 | 0.42431 |
| GRPO g4 anchor0.05 | 32 / 16 | 5 | 0.44541 | 0.43855 | 0.44198 |
| GRPO g4 anchor0.05 | 96 / 16 | 5 | 0.53206 | 0.54348 | 0.53777 |
| GRPO g4 anchor0.05 | 118 / 16 | 5 | 0.55016 | 0.54380 | 0.54698 |
| GRPO g4 anchor0.05 | 118 / 16 | 8 | 0.57084 | 0.56763 | 0.56923 |
| GRPO g4 anchor0.05 | 118 / 16 | 10 | 0.59415 | 0.58450 | 0.58932 |

结论：按当时的旧 scan-level split，clean-val 最强更新为
`GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch`，reload Dice mean 0.58932。
相对 8 epoch 提升 +0.02009，相对 5 epoch full-train 提升 +0.04234，
相对 supervised 32/16 提升 +0.16501。

A100 审计结论：当前脚本是单进程单卡，不支持 DDP；batch>1 仍不安全。
decoder-only GRPO 只占约 3.5-4GB，瓶颈主要是单样本顺序 rollout，不是显存容量。
`num_workers=2` 是本轮保留的吞吐优化；并发两个训练能提高瞬时 GPU util，但会拖慢主实验。

附带负结果：`group_size=6` seed2026 在 epoch3 只有 Dice 0.51206，慢且落后于 g4/e10，
已早停，不扩 seed。

清理：e10 两个训练目录和 g6 早停目录均只保留 `sam_model_val_dice_best.pth`；
重复 `latest/reward` checkpoint 已删除。test fold 仍未使用。

---

## 2026-07-06 insight/survey 汇总

本轮追加两个只读 agent：一个复核本地 clean-val / reload audit-only 结果，一个做外部论文和跨领域方法 survey。
结论是算法在当时的旧 scan-level 划分上明确 work：历史 best
`GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch`
reload Dice mean 0.58932，相对 supervised 32/16 的 0.42431 提升 +0.16501，
相对 32-case GRPO 的 0.44198 提升 +0.14734。

核心 insight：

- 原始 GRPO 有效的关键是 `group_size=4 + supervised_anchor=0.05 + KL + Dice/volume/component dense reward`。
  扩样本 32 -> 64 -> 96 -> 118 和延长 epoch 5 -> 8 -> 10 都带来同向提升。
- 之前尝试的 pure pairwise、GRPO+pairwise、center-of-mass reward 都不如原始 GRPO。
  pairwise 大概率丢掉 dense reward magnitude；center reward 太粗，不能替代 overlap/边界目标。
- A100 显存不是当前瓶颈；decoder-only GRPO 只占约 3.5-4GB，瓶颈是单样本顺序 rollout/reference forward。
  g6 早期慢且差，不继续扩 seed。
- test fold 仍未使用。若进入论文级 test，必须先冻结 e10 配置，只跑一次 held-out test，不在 test 上调参。

外部 survey 对下一步的启发：

| Source | 可迁移点 | 当前使用方式 |
|---|---|---|
| DeepSeekMath GRPO: https://arxiv.org/abs/2402.03300 | group-relative advantage + KL，适合规则 reward。 | 保留 GRPO 主线。 |
| DINO-R1: https://arxiv.org/abs/2505.24025 | 视觉 foundation model 也可做 group-normalized RL-style alignment。 | 说明 group-relative reward 可迁移到视觉，但需 dense reward。 |
| SAMPO: https://arxiv.org/abs/2508.02464 | SAM 分割可做 preference optimization。 | 支持方向，不支持继续 naive pairwise。 |
| Boundary Loss: https://arxiv.org/abs/1812.07032 | boundary/surface loss 补 Dice/CE 边界不足。 | 下一步优先 surface Dice / boundary-F1 reward。 |
| Boundary Uncertainty: https://arxiv.org/abs/2111.00533 | 医学边界标注有不确定性，可边界带 soft/降权。 | 候选 `boundary_weighted` supervised anchor。 |
| Dual uncertainty self-training: https://arxiv.org/abs/2304.04441 | sample/pixel uncertainty 可做 curriculum。 | 候选 hard-case sampler / uncertainty-weighted anchor。 |

下一步优先级：先做同配置 12 epoch seed2026 复核；若不超过 e10 seed2026 reload Dice 0.59415，
不补 seed2027。新 idea 首选 32/16 小成本 boundary/surface reward，而不是继续 pairwise、center、g6 或后处理搜索。

---

## 2026-07-06 e12 seed2026 复核

设置：同当前 best 配置，只把训练从 10 epoch 延到 12 epoch；`label_RC`，M0 初始化，
原始 `grpo`，118 train / 16 val，`group_size=4`，`supervised_anchor=0.05`，
`kl_coef=0.02`，`num_workers=2`，seed2026，仍只用 clean-val，不使用 test fold。

结果：

| Run | Seed | Best epoch | Training best Dice | Reload Dice |
|---|---:|---:|---:|---:|
| e10 g4 anchor0.05 nw2 | 2026 | 10 | 0.59192 | 0.59415 |
| e12 g4 anchor0.05 nw2 | 2026 | 10 | 0.59470 | 0.58548 |

结论：e12 训练内略高，但保存 checkpoint 后的 reload audit 低于 e10；
因此不补 seed2027，不继续 e14/e16。当时旧 scan-level clean-val 主配置仍是
`GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`。
下一步若继续 clean-val 科研，应转向 32/16 小成本 boundary/surface reward 或 boundary-weighted anchor。

清理：e12 训练目录已删除 `sam_model_latest.pth` 和 `sam_model_val_reward_best.pth`，
只保留 `sam_model_val_dice_best.pth` 与日志/CSV/JSON。test fold 仍未使用。

---

## 2026-07-06 surface reward 小实验

实现：在 `scripts/train_sammed3d_grpo.py` 增加默认关闭的
`--surface-reward-coef` / `--surface-tolerance`，用 3D surface Dice 作为额外 reward。
默认系数为 0，不改变原始 GRPO 复现行为。

设置：`label_RC`，M0 初始化，32 train / 16 val，原始 `grpo`，
`group_size=4`，`supervised_anchor=0.05`，`surface_reward_coef=0.05`，
`surface_tolerance=2`，seed2026，clean-val reload audit-only，不使用 test fold。

结果：

| Method | Train / Val | Seed 2026 reload Dice |
|---|---:|---:|
| Supervised best | 32 / 16 | 0.42186 |
| GRPO g4 anchor0.05 | 32 / 16 | 0.44541 |
| GRPO + surface coef0.05 tol2 | 32 / 16 | 0.42378 |

结论：surface reward 训练内能提高 train Dice / surface reward，但 clean-val Dice 不过原始 GRPO32；
且 distance transform 让 32-case epoch 变慢到约 113 秒。因此不补 seed2027，不扩 full train。
当前主配置仍是 `GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`。
下一步若继续新 idea，更建议做 `boundary_weighted` supervised anchor 小实验。

清理：surface 训练目录已删除 `sam_model_latest.pth` 和 `sam_model_val_reward_best.pth`，
只保留 `sam_model_val_dice_best.pth` 与日志/CSV/JSON。test fold 仍未使用。

---

## 2026-07-06 boundary-weighted anchor 小实验

实现：在 `scripts/train_sammed3d_grpo.py` 增加默认关闭的
`--anchor-mode boundary_weighted`，用 GT dilation/erosion 生成边界带，
边界带权重 0.5，anchor loss 改为 weighted BCE + weighted soft Dice。
默认 `anchor-mode=dicece`，不改变原始 GRPO。

设置：`label_RC`，M0 初始化，32 train / 16 val，原始 `grpo`，
`group_size=4`，`supervised_anchor=0.05`，`anchor_mode=boundary_weighted`，
`boundary_ignore_radius=1`，`boundary_weight=0.5`，seed2026，clean-val reload audit-only，
不使用 test fold。

结果：

| Method | Train / Val | Seed 2026 reload Dice |
|---|---:|---:|
| Supervised best | 32 / 16 | 0.42186 |
| GRPO g4 anchor0.05 | 32 / 16 | 0.44541 |
| GRPO + surface coef0.05 tol2 | 32 / 16 | 0.42378 |
| GRPO + boundary-weighted anchor | 32 / 16 | 0.40737 |

结论：boundary-weighted anchor 明显不如原始 DiceCE anchor，也不如 supervised baseline；
训练 Dice 上升但 val Dice 不跟，说明它加剧了小样本过拟合。该方向不补 seed2027，不扩 full train。
boundary 系列暂时停止，不做系数/radius 网格搜索。

清理：boundary-weighted 训练目录已删除 `sam_model_latest.pth` 和
`sam_model_val_reward_best.pth`，只保留 `sam_model_val_dice_best.pth` 与日志/CSV/JSON。
test fold 仍未使用。

---

## 2026-07-06 hard-case curriculum 小实验

实现：在 `scripts/train_sammed3d_grpo.py` 增加默认关闭的
`--curriculum-mode hard_case`。每个 epoch 结束按 case-level mean reward 排序，
最低 30% 病例下一轮采样权重设为 2x，并写出 `curriculum_cases.csv`。默认 `none`，
不改变原始 GRPO。

设置：`label_RC`，M0 初始化，32 train / 16 val，原始 `grpo`，
`group_size=4`，`supervised_anchor=0.05`，`curriculum_mode=hard_case`，
`hard_case_after_epoch=2`，`hard_case_fraction=0.30`，`hard_case_weight=2.0`，
seed2026，clean-val reload audit-only，不使用 test fold。

结果：

| Method | Train / Val | Seed 2026 reload Dice |
|---|---:|---:|
| Supervised best | 32 / 16 | 0.42186 |
| GRPO g4 anchor0.05 | 32 / 16 | 0.44541 |
| GRPO + hard-case curriculum | 32 / 16 | 0.42267 |

结论：hard-case curriculum 没有超过原始 GRPO32，只接近 supervised。训练 Dice 到 epoch5
升到 0.56575，但 reload Dice 只有 0.42267，说明重采样低 reward train case 更像加强小样本拟合，
没有提升 clean-val 泛化。不补 seed2027，不做 fraction/weight 网格。

清理：hard-case curriculum 训练目录已删除 `sam_model_latest.pth` 和
`sam_model_val_reward_best.pth`，只保留 `sam_model_val_dice_best.pth` 与日志/CSV/JSON。
test fold 仍未使用。

---

## 2026-07-06 rollout logit noise 小实验

实现：在 `scripts/train_sammed3d_grpo.py` 增加默认关闭的
`--rollout-logit-noise-std`。训练时只对 rollout 的 reward/action 打分 logits 加高斯噪声，
policy logprob / KL / supervised anchor 仍使用原始 logits；validation 和 reload audit 不加噪声。
默认 `0.0` 不改变原始 GRPO。

设置：`label_RC`，M0 初始化，32 train / 16 val，原始 `grpo`，
`group_size=4`，`supervised_anchor=0.05`，分别测试 `rollout_logit_noise_std=0.25/0.50`，
seed2026，clean-val reload audit-only，不使用 test fold。两个 run 并发跑在单张 A100 上，
GPU util 约 97%，显存约 7GB。

结果：

| Method | Train / Val | Seed 2026 reload Dice |
|---|---:|---:|
| Supervised best | 32 / 16 | 0.42186 |
| GRPO g4 anchor0.05 | 32 / 16 | 0.44541 |
| GRPO + logit noise std0.25 | 32 / 16 | 0.43340 |
| GRPO + logit noise std0.50 | 32 / 16 | 0.44108 |

结论：`std=0.50` 接近原始 GRPO32 两 seed mean，但没有超过同 seed 原始 GRPO32；
`std=0.25` 更弱。logit noise 有一点正则化/多样性迹象，但没有形成独立增益。
不补 seed2027，不扩 full train，也不继续做 std 网格。当前主配置仍是
`GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`。

清理：两个 logit-noise 训练目录已删除 `sam_model_latest.pth` 和
`sam_model_val_reward_best.pth`，只保留 `sam_model_val_dice_best.pth` 与日志/CSV/JSON。
test fold 仍未使用。

---

## 2026-07-06 tumor_ET 跨目标迁移

目的：确认 `GRPO g4 anchor0.05` 是否只在 `label_RC` 上有效。选择另一个弱目标
`tumor_ET` 做迁移验证，仍用 patient-level split、clean-val、reload audit-only，不使用 test fold。

先补了 `tumor_ET` 的 M0 初始化：

| Method | Train / Val | Reload Dice |
|---|---:|---:|
| ET M0 overfit4 init | 4 overfit / 14 val audit | 0.13207 |

然后跑 32 train / 14 val，两个 seed，对比 supervised 和 GRPO：

| Seed | Method | Reload Dice |
|---:|---|---:|
| 2026 | supervised | 0.38604 |
| 2026 | GRPO g4 anchor0.05 | 0.39092 |
| 2027 | supervised | 0.40026 |
| 2027 | GRPO g4 anchor0.05 | 0.38089 |

两 seed mean：

| Method | Dice mean |
|---|---:|
| ET supervised | 0.39315 |
| ET GRPO g4 anchor0.05 | 0.38590 |

结论：ET 上 GRPO 不是稳定正结果。seed2026 小幅超过 supervised，但 seed2027 反转；
两 seed mean 下 GRPO 低于 supervised -0.00725。因此不能把 RC 的强结论直接外推到 ET。
当时不扩 ET 到 64/full train，也不上 test；RC full-train e10 只是旧 scan-level
split 内的历史最强主线，不构成 subject-disjoint 证据。

清理：ET M0、ET supervised 两 seed、ET GRPO 两 seed均已删除重复 checkpoint，
只保留 `sam_model_val_dice_best.pth` 与日志/CSV/JSON。test fold 仍未使用。

---

## 2026-07-06 tumor_TC 跨目标迁移

目的：ET 上 GRPO 迁移不稳后，再用 `tumor_TC` 复核 `GRPO g4 anchor0.05` 是否能跨目标成立。
仍使用 patient-level split、clean-val、reload audit-only，不使用 test fold。

先补了 `tumor_TC` 的 M0 初始化：

| Method | Train / Val | Reload Dice |
|---|---:|---:|
| TC M0 overfit4 init | 4 overfit / 14 val audit | 0.11818 |

然后跑 32 train / 14 val，两个 seed，对比 supervised 和 GRPO：

| Seed | Method | Reload Dice |
|---:|---|---:|
| 2026 | supervised | 0.41400 |
| 2026 | GRPO g4 anchor0.05 | 0.42027 |
| 2027 | supervised | 0.42073 |
| 2027 | GRPO g4 anchor0.05 | 0.40505 |

两 seed mean：

| Method | Dice mean |
|---|---:|
| TC supervised | 0.41736 |
| TC GRPO g4 anchor0.05 | 0.41266 |

结论：TC 上 GRPO 也不是稳定正结果。seed2026 小幅超过 supervised（+0.00627），但 seed2027
反转（-0.01568）；两 seed mean 下低于 supervised -0.00470。结合 ET 结果，当前
`g4+anchor0.05` 的明确强结论仍只在 `label_RC` 上成立，不能直接外推到 TC/ET。

执行与资源：四个 32-case 训练进程并发跑在单张 A100 上，峰值约 10.9GB/80GB，GPU util 约 90%+；
这比单进程更适合当前脚本。GRPO 仍不要强行 batch>1 或 DDP。

下一步科研策略：

- 不扩 TC 到 64/full train，也不上 test。
- 回到 `label_RC 32/16` 做低成本算法改造：优先 `KL/anchor/reference schedule`、
  `variance-reduced advantage`、`mixed soft reward`。
- 如果新算法在 RC32 seed2026 超过原始 0.44541，并且 seed2027 不反转，再考虑 full train 或
  回到 TC/ET 复核。
- 暂停继续投入 pairwise、center reward、surface reward、boundary-weighted anchor、
  hard-case curriculum、rollout logit-noise 网格、`g6`、单纯延长 e12/e14。

清理：TC M0、TC supervised 两 seed、TC GRPO 两 seed均已删除重复 checkpoint，
只保留 `sam_model_val_dice_best.pth` 与日志/CSV/JSON。test fold 仍未使用。

---

## 2026-07-06 RC anchor/KL schedule 小实验

实现：`scripts/train_sammed3d_grpo.py` 新增默认关闭参数
`--supervised-anchor-schedule` 和 `--kl-coef-schedule`。默认空字符串时旧行为不变。
本实验用 schedule：

- anchor：`0.08,0.08,0.05,0.05,0.03`
- KL：`0.04,0.04,0.02,0.02,0.01`

设置：`label_RC`，M0 初始化，32 train / 16 val，原始 `grpo`，
`group_size=4`，5 epoch，clean-val reload audit-only，不使用 test fold。

结果：

| Seed | Original GRPO Dice | Schedule Dice | Delta |
|---:|---:|---:|---:|
| 2026 | 0.44541 | 0.44653 | +0.00112 |
| 2027 | 0.43855 | 0.42734 | -0.01121 |
| mean | 0.44198 | 0.43694 | -0.00504 |

结论：anchor/KL schedule 不是稳定改进。seed2026 的提升太小，seed2027 明显反转；
两 seed mean 下降。因此不扩 full train，不回到 TC/ET 复核，也不上 test。下一步若继续算法侧，
优先试 variance-reduced advantage 或 mixed soft reward，而不是继续 schedule 网格。

清理：两个 schedule 训练目录已删除 `sam_model_latest.pth` 和
`sam_model_val_reward_best.pth`，只保留 `sam_model_val_dice_best.pth` 与日志/CSV/JSON。

---

## 2026-07-06 RC rank advantage 小实验

实现：`scripts/train_sammed3d_grpo.py` 新增默认关闭参数
`--advantage-mode standard|rank`。默认 `standard` 与旧 GRPO 完全一致；
`rank` 模式只用 group 内 reward 排名做 advantage，不增加 forward，也不改变 reward/audit 口径。

设置：`label_RC`，M0 初始化，32 train / 16 val，原始 `grpo`，
`group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`，seed2026，clean-val reload audit-only。

结果：

| Method | Seed | Reload Dice |
|---|---:|---:|
| original GRPO g4 anchor0.05 | 2026 | 0.44541 |
| rank advantage | 2026 | 0.44234 |

结论：rank advantage 没超过原始同 seed，不补 seed2027，不扩 full train，不上 test。
本轮说明“简单降低 advantage 数值方差”不足以改进 RC32；下一步若继续算法侧，应转向
`mixed soft reward` 或 reward-weighted BC/AWR，而不是继续 advantage/schedule 网格。

清理：rank advantage 训练目录已删除 `sam_model_latest.pth` 和
`sam_model_val_reward_best.pth`，只保留 `sam_model_val_dice_best.pth` 与日志/CSV/JSON。

---

## 2026-07-07 RC mixed-soft reward 小实验

实现：`scripts/train_sammed3d_grpo.py` 新增默认关闭参数
`--reward-mode hard|mixed_soft` 和 `--soft-reward-weight`。`mixed_soft` 在原始 hard Dice
reward 中掺入少量 soft Dice / soft IoU：

```text
0.85 * hard_dice + 0.15 * (0.5 * soft_dice + 0.5 * soft_iou)
```

同时，`--audit-only` 已强制回 hard reward 口径，保证 reload audit 仍可与旧结果公平对比。

设置：`label_RC`，M0 初始化，32 train / 16 val，原始 `grpo`，
`group_size=4`，`supervised_anchor=0.05`，`kl_coef=0.02`，seed2026，clean-val reload audit-only。

结果：

| Method | Seed | Reload Dice |
|---|---:|---:|
| original GRPO g4 anchor0.05 | 2026 | 0.44541 |
| mixed-soft reward w0.15 | 2026 | 0.44743 |

结论：mixed-soft 只有 +0.00203 的微小提升，不足以说明方法稳定有效；
不补 seed2027，不扩 full train，不上 test。当时的旧 scan-level clean-val best 仍是
`GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`，
两 seed reload Dice mean 0.58932。

下一步算法候选改为 reward-weighted BC/AWR 风格小筛选：参考 AWR
（https://arxiv.org/abs/1910.00177）、AWAC（https://arxiv.org/abs/2006.09359）
和 CRR（https://arxiv.org/abs/2006.15134），用 group 内高 reward rollout 对 decoder 做
加权模仿，避免只用 rank 或 best-worst pairwise 丢掉 dense reward magnitude。

清理：mixed-soft 训练目录已删除 `sam_model_latest.pth` 和
`sam_model_val_reward_best.pth`，只保留 `sam_model_val_dice_best.pth` 与日志/CSV/JSON。

---

## 2026-07-07 RC GRPO+AWR 小实验

实现：`scripts/train_sammed3d_grpo.py` 新增默认关闭 objective
`--objective awr|grpo_awr`。`grpo_awr` 保留原始 GRPO loss，再加一个
reward-weighted BC / AWR 辅助项：只对 group 内高 reward rollout 的 hard mask 做加权模仿。
默认 objective 仍是 `grpo`，旧实验不变。

设置：`label_RC`，M0 初始化，32 train / 16 val，`group_size=4`，
`supervised_anchor=0.05`，`kl_coef=0.02`，seed2026，clean-val reload audit-only。

结果：

| Method | Reload Dice |
|---|---:|
| original GRPO g4 anchor0.05 | 0.44541 |
| GRPO+AWR coef0.02 tau0.10 wmax5 | 0.43841 |
| GRPO+AWR coef0.05 tau0.10 wmax5 | 0.43593 |
| GRPO+AWR coef0.03 tau1.00 wmax3 | 0.43469 |

结论：AWR 三组都没有超过原始 GRPO32 seed2026，不补 seed2027，不扩 full train，不上 test。
当前判断是：只模仿高 reward rollout 会强化模型自己的局部伪标签，但没有比原始 GRPO
的正负 advantage 更新更好。当时的旧 scan-level clean-val best 仍是
`GRPO g4 anchor0.05, 118 train / 16 val, 10 epoch, num_workers=2`，
两 seed reload Dice mean 0.58932。

清理：三个 AWR 训练目录已删除 `sam_model_latest.pth` 和
`sam_model_val_reward_best.pth`，只保留 `sam_model_val_dice_best.pth` 与日志/CSV/JSON。

---

## 2026-07-07 RC full-train supervised clean 对照

背景：5 个只读 side agents 完成分工审计后，Results Auditor 指出当前最关键缺口是
`label_RC` full-train GRPO 缺少同规模 supervised control。此前 GRPO118 e10 主要和
32-case supervised 比，不能完全排除“只是训练样本更多”的解释。

设置：`label_RC`，M0 初始化，118 train / 16 val，supervised decoder-only，
10 epoch，lr `2e-5`，seed2026/2027。两个 seed 在 A100 上并发跑，峰值 util 约 99%。
训练后统一用 `sam_model_val_dice_best.pth` 做 hard reload audit-only，不使用 test。
旧 `audit_label_RC_16val_sup118_e10_best_g4_seed*` 数值不再作为结论；本节使用
`_sup_clean_seed2026/2027` 完整重跑后的 audit。

结果：

| Method | Seed2026 Dice | Seed2027 Dice | Mean Dice |
|---|---:|---:|---:|
| supervised32 e5 | 0.42186 | 0.42676 | 0.42431 |
| supervised118 e10 clean | 0.56454 | 0.58113 | 0.57283 |
| GRPO118 e10 g4 anchor0.05 | 0.59415 | 0.58450 | 0.58932 |

结论：在旧 scan-level split 内，同规模 supervised 已经很强，说明数据扩展本身贡献很大；
GRPO118 当时仍比
supervised118 clean 高 +0.01649 mean，且两 seed 都有正 margin（+0.02961 / +0.00337）。
因此只能说旧 split 内观察到原始 GRPO 的小幅同规模优势；它不是 subject-disjoint
泛化证据，也不能直接外推到当前全量协议。
但优势比旧估计更小，没有达到 +0.020 mean gate；写论文时应表述为“小幅同规模优势”。
test fold 仍未使用。

清理：两个 supervised118 clean 训练目录均已删除 `sam_model_latest.pth` 和
`sam_model_val_loss_best.pth`，只保留 `sam_model_val_dice_best.pth` 与日志/CSV/JSON。

---

## 2026-07-07 RC corrective GRPO 小实验

背景：AWR/reward-weighted BC 为负后，测试更保守的 corrective GRPO：正 advantage
仍走原始 GRPO，负 advantage 改成 supervised-anchor 风格 correction。

设置：`label_RC`，M0 初始化，32 train / 16 val，`group_size=4`，
`supervised_anchor=0.05`，`kl_coef=0.02`，5 epoch，seed2026，clean-val reload audit-only。

结果：

| Method | Reload Dice |
|---|---:|
| original GRPO32 seed2026 | 0.44541 |
| corrective coef0.05 | 0.42572 |
| corrective coef0.10 | 0.43740 |

结论：corrective 没超过原始 GRPO32，同样不补 seed2027、不扩 full train、不上 test。
当前 insight 是：RC 上原始 GRPO 的正负 advantage 推拉比“负样本只拉回 GT”的保守修正更有效。

清理：两个 corrective 训练目录均已删除重复 checkpoint，只保留
`sam_model_val_dice_best.pth` 与日志/CSV/JSON。

---

## 2026-07-17 B200 + BraTS2024 全量 subject-clean work #2

### 数据与 split 重建

完整数据共有 1350 scans / 613 subjects / 6334 target rows。subject id 是
`BraTS-GLI-xxxxx`，末尾 `-100/-101/...` 是纵向 timepoint。新 split 为：

| Split | Subjects | Scans | Rows |
|---|---:|---:|---:|
| train | 419 | 948 | 4431 |
| clean val | 67 | 133 | 630 |
| sealed test | 127 | 269 | 1273 |

subject overlap 为 0；历史 39 subjects 与 4 个曾做人工/原始体素检查的 subjects
全部强制进入 train。split 前的自动 invariant QA 曾抽样 10 subjects，其中 3 个后来
随机进入 test；它只检查数据结构/标签一致性，没有模型预测、指标或方法选择。test JSONL
只放在 `sealed_test/`，最终协议冻结前没有做 test evaluation；冻结后唯一一次尝试的环境
失败记录见下文。

### B200 与训练实现

- 两张 NVIDIA B200 183359 MiB；torch `2.12.0+cu130`、torchvision `0.27.0`、
  cuDNN 92000。SM100 kernel、Conv3D、BF16 与真实 SAM-Med3D forward/backward 均通过。
- 新增 target token、12-block image-encoder Q/V LoRA r4/a8、adapter-aware strict reload、
  固定 validation RNG、target macro 选模、原子 checkpoint。
- strict audit 固定 5-click clean val，同时报告 row-weighted 与先 timepoint 后 patient
  聚合的 subject-balanced 指标。
- exact hierarchical sampler 每轮 4428 samples、六 target 各 738；target 内 patient
  round-robin，patient 内 timepoint seeded cycle，并写出逐期 schedule SHA。

### Subject-clean pilot 与两 seed gate

| Arm | 方法 | 6-target macro | 4-weak macro | 判定 |
|---|---|---:|---:|---|
| D0 | decoder/prompt | 0.51938 | 0.43340 | control |
| D1 | + target token | 0.51781 | 0.43167 | reject |
| L0 | + Q/V LoRA r4/a8 | 0.53542 | 0.45446 | promote |
| L1 | token + LoRA | 0.53855 | 0.46325 | SNFH drop 超限，reject |

L0-D0 weak macro 在 seed2026/2027 分别为 `+0.02107/+0.01548`。strict reload 的
subject-balanced L0-D0 分别为 `+0.01845/+0.01682`；L0 相对 raw turbo 分别为
`+0.10428/+0.10186`。target token 没有晋级，GRPO 继续后置。

### Full-scale seed2026 结果

共同设置：4431 train rows / 419 subjects、完整 630-row clean val、3 epoch、5 clicks、
BF16、seed2026、val seed42026、按 `val_macro_dice` 选模。D0/L0 三期 sampler
schedule 完全相同，双方 selected epoch 均为 2。

| Arm | 6-target macro | 4-weak macro | Subject-balanced macro |
|---|---:|---:|---:|
| D0 | 0.533289 | 0.452116 | 0.527801 |
| L0 | 0.551479 | 0.468865 | 0.544192 |
| L0-D0 | **+0.018190** | **+0.016749** | **+0.016391** |

row-weighted per-target L0-D0：NETC `+0.04040`、RC `+0.01837`、ET `+0.01316`、
TC `-0.00492`、WT `+0.02760`、SNFH `+0.01454`。primary weak-target gate 为正，
3/4 弱目标不下降，两个约束目标都上升。D0/L0 selected checkpoint 的 strict reload
均与训练 history 精确复现。

正式 pair artifact：
`workspace/BraTS/training_protocols/brats2024_fullscale_seed2026_pair_results_v1.json`，
SHA-256 `1ed89ea686c07421d82651767c9b6a5e5c8774ee9961abb8ed30d60adaba8418`；
`pair_validated=true`、`test_accessed=false`。

### Full-scale seed2027 结果

seed2027 replication 在启动前写入独立协议，protocol SHA-256 为
`806d77817dd52f9d2f5401d17fa1938943a707d39f19871e7187d921be3945f7`。
provenance attestation 位于
`workspace/BraTS/training_protocols/brats2024_fullscale_seed2027_prereg_provenance_attestation_v1.json`，
SHA-256 `5081c903b4ecbfbb5396acd1bfa74ff54c9222cf5612b9607e15c1309006fe2a`；它只澄清
`best_sha256` 的 metadata 语义，不改变任何训练或决策。
2026-07-17 10:20 UTC 在两张 B200 并行启动并已完成；除 train seed 2027 外，其余训练
设置与 seed2026 相同，三期 schedule SHA 全部精确复现。

| Arm | Selected epoch | 6-target macro | 4-weak macro | Subject-balanced macro |
|---|---:|---:|---:|---:|
| D0 | 2 | 0.539155 | 0.465323 | 0.532617 |
| L0 | 3 | 0.560209 | 0.480259 | 0.554441 |
| L0-D0 | - | **+0.021054** | **+0.014936** | **+0.021824** |

row-weighted 六 target delta 全为正：NETC `+0.003296`、RC `+0.017289`、ET
`+0.015712`、TC `+0.023449`、WT `+0.023317`、SNFH `+0.043261`。两条 strict reload
与 history 精确一致。正式 pair artifact：
`workspace/BraTS/training_protocols/brats2024_fullscale_seed2027_pair_results_v1.json`，
SHA-256 `629377f47580c772dc445f4a83a921bc097b4087ceb9d1fbcf02d10b6e65e653`；
`pair_validated=true`、`test_accessed=false`。两 seed full-scale 均支持 L0，L0 已冻结为最终配置。

### 最终配置冻结与 test 解封工程

frozen-test runner 经独立 adversarial review 后，已 fail-closed 绑定 canonical split、两份
pair result、两份训练协议、prereg/crosswalk、四份 strict audit、五个 checkpoint 和实际运行
代码；全套 `181 passed`。runner SHA-256 为
`25da9a3b12c6539d9b12ff14e77a27d7f76c186ec698b7e078a01305c973bb89`，patient-id helper
SHA-256 为 `831d40e83d5f514075aa86df5d6f15fbe483e088c3d07714e0aeef565be08a66`。

冻结协议：
`workspace/BraTS/training_protocols/brats2024_final_frozen_test_l0_v1.json`，SHA-256
`9dfdf06d0ab9afbdc487f18aea677769dfe3f0d0d9abd1f0ce260eb4bb5d33f7`。它在 test 访问、
全局 ledger 和输出目录创建前落盘，预注册 primary 为两 seed L0 相对 raw turbo 的
subject-balanced weak macro，key secondary 为两 seed paired L0-D0；仅非 test preflight
已通过。冻结时 sealed test 仍未读取或评估。

### 唯一 test 尝试与 postmortem

attempt ledger `11:59:46Z` 创建，test access `12:00:03Z` 开始；`12:03:12Z` 在
`checkpoint_audit:raw_turbo` 失败。实际 worker launcher 漏掉 overlay cuDNN 的
`LD_LIBRARY_PATH`，写 summary 时检测到 torch 编译 cuDNN `9.20.0`、运行时 `9.7.1`。
没有 run 目录、cases、summary、COMPLETE 或任何 test 指标，故不能报告 test 效果。

协议固定 access 后 `retries=0`；global ledger 已消费该 manifest，禁止重跑。证据根目录：
`/mnt/bn/liveaim/songjunyi/aflow_brats_work2_v4/final_frozen_test_eval_l0_v1_seed42026`。
`ATTEMPT/TEST_ACCESS_STARTED/FAILED/ledger` SHA-256 依次为：

```text
ef1849aff80d9942aa6a2a02ee038df6640f9560bfccad1d5e1106c59572f628
6c33294c3aee3c88dc497e8c1700089d98907ee086fd7ce29280c126f2ad6579
7e0354d2be15ce64f0091cb5abb323b1e716244d3e3e7e731a76620275e5b720
8f6dd93057a6139f2609e19326c2c8b2929d1d66683ab74b9b5b5f5ca987a96d
```

postmortem runner 新增 ledger 前的真实 CUDA/BF16/cuDNN Conv3D probe，冻结 summary 复用
preflight runtime；全套 `184 passed`。新 runner SHA-256
`9d2582c979488b4fd5787af79f088f8d659e068eb908be2477fff853bad4872d`，测试 SHA-256
`1b88e491e7c38d780cca16bb9044b047825b331a919e13e4c6313c6290ef6e64`。
冻结协议保持原字节与原 provenance，不允许用修补后 runner 重试当前 manifest。

### 后续 work #2 训练候选

只保留三个带 paired control 的高信息量方向：第一优先是四个弱目标的专属 decoder
experts，对照 equal-step shared-decoder continued-SFT；第二是保持 Q/V LoRA r4/a8、只训练
upscaler/hypernetwork output 的 partial-decoder LoRA；第三先采集约 100 个六目标 macro-step
的 gradient cosine，只有负冲突达到预设门槛才启动 PCGrad。统一要求 clean-val weak macro
相对 paired control 至少 `+0.005`、至少 3/4 弱目标不下降，并约束 WT/SNFH；不做 rank
网格、threshold、TTA 或 postprocess 搜索。

本地旧 200-scan 原始数据副本 `/mlx_devbox/users/songjunyi.666/playground/brats2024`
已确认完全包含于 `/mnt` 的 1350-scan 数据后删除，释放约 41G；当前训练、全量预处理、
base checkpoint、venv、历史结果、失败回执和 test 快照均未删除。

---

## 2026-07-20 work #1 收口审计与 gradient-directed continuation

本轮先核查硕士工作 #1，再继续 work #2。边界保持不变：没有运行 inference workflow、
threshold、TTA、postprocess，也没有访问 sealed test。

### Work #1 审计结论

工程主干可判定完成并继续冻结。准确方法名应为 **UCB-guided AFlow workflow tree
search**：实现包含 UCB 风格父节点选择、LLM workflow 生成和执行反馈，但不包含标准 MCTS
的 rollout、`N/W/Q` 统计或 ancestor backprop。论文若要把它提升为强方法学结论，仍需新的
subject-clean holdout 和 equal-budget search-policy ablation；这些缺口不在本轮重开。

审计 artifact：
`workspace/BraTS/research_audits/work1_ucb_aflow_completion_audit_20260720.md`，SHA-256
`d2de9ad28fd6cd9935fc5f26c63affc109805c63b178e58cbd2ff7c37eea6828`。

### Gradient conflict 结果与分支

对 seed2026/2027 selected full-scale L0 checkpoint 使用同一 train-only schedule，各审计
100 个六目标 macro-steps。四个互斥原子梯度组完整覆盖 7,731,760 个 L0 可训练参数；另报告
decoder/all-trainable 聚合统计。两 seed 所有 eligible group/weak-target 的 mean cosine 都为正，
没有同一单元在两 seed 同时满足 `conflict fraction >= 0.20` 且 `mean <= -0.02`。

冻结决策为 `partial_decoder`，共同 PCGrad trigger 为 0。两份 audit SHA-256：

```text
seed2026 7a42eba7b7ad04c496ac369e450c2316852f0fecb98e9be47714d9004d2712c8
seed2027 9351185d8390a35a41732609d499e43999af7627631f7e76d4098ebd6b1b8ade
```

protocol SHA-256：
`878d2cfc05de55ad5e950636c9510ab80ebc2b07a760caa97f9d961d6a96c291`；decision SHA-256：
`a93a98fbde1c945e623d100354b57612844d1b16c33a18b2ce61a3ad8e6d628c`。

### Continuation 工程与当前状态

冻结四臂、两 seed、GPU cross-balanced 的 paired experiment：full-L0 continuation 对
partial decoder。candidate 只训练 Q/V LoRA、`output_upscaling` 和实际使用的
`output_hypernetworks_mlps.0`，共 793,728 参数；control 训练完整 L0 的 7,731,760 参数。
双方使用相同 2-epoch exact sampler schedule、click RNG、clean val 和选模规则。

历史 trainer 保持 SHA-256
`0c5026e98b2e0f848c16b436610e9b7eca391995c062ac128ce1bee26d1bfa4f` 不变；新增独立
continuation runner SHA-256
`993421f36ee5d58b8269783437d65c292e390ecba1e373056d0f3de8619a435b`，全套测试
`205 passed`。训练协议 SHA-256：
`fd956ef4649422a7c2d9f01a35c8440eb66a4fd7c04e0ed44fa8fbd7a3dba3bf`。

第一次提交于北京时间 `12:05:45` 因非交互 SSH 选择了过期 CUDA 12.6 compatibility
`libcuda` 而在 runtime preflight 失败；四臂均未加载 manifest、模型或 optimizer，训练步数
为 0。该基础设施失败已独立冻结，SHA-256
`7786b321a481b793adb32725c12ec61726b4ddcdd3dec74bb53c5551414284ee`。绑定真实 580.105.08
driver 并通过两卡 CUDA/SM100/BF16/cuDNN/Conv3D preflight 后，于 `12:12:25` 做唯一一次
infrastructure-only resume；科学协议和 runner 未改。四臂均跑满 2 epochs，并按冻结规则
选择 epoch 1，最晚 `12:46:23` 完成。

随后四个 selected checkpoint 均独立重载，完成完整 630-row / 67-subject strict clean-val：

| Seed | Full-L0 weak macro | Partial weak macro | Delta weak | Delta six-target macro |
|---:|---:|---:|---:|---:|
| 2026 | 0.466649 | 0.472120 | +0.005471 | +0.006798 |
| 2027 | 0.468040 | 0.473098 | +0.005058 | +0.005664 |
| 两 seed 均值 | - | - | **+0.005265** | **+0.006231** |

六目标两 seed subject-balanced 平均 delta 全部为正；冻结 promotion gate 全部通过，结果为
`promote_clean_val_candidate`。由于 weak macro 仅比门槛高 `0.000265`，且存在单 seed 目标
符号反转，该结果只支持 clean-val 候选晋级，不支持 test/外部有效性声明。冻结结果 artifact
SHA-256 为 `3daefb85a8bb588cbf63470171cd59251f701a84534e12925a04249a6db8abee`；最终完整测试
`211 passed`。四个训练和四个审计进程均已退出，无用 SSH/agent terminal 已关闭；只保留
当前 Codex 与维持 2xB200 配额的 worker launcher。本轮 continuation 未读取 test metric，
也没有新的 test 授权；第 42 节的 sealed-test manifest 已消费且禁止重试。

---

## 2026-07-20 Work #1 更名与外部 SOTA 目标冻结

### Work #1 更名

Work #1 的论文级名字改为 **MedTAWS（Medical Tree-guided Adaptive Workflow Search）**，
中文为“医学树引导自适应工作流搜索”。算法和代码完全不变；首次介绍必须写明它建立在
AFlow 上，使用 UCB-guided workflow-tree policy 和 SAM-Med3D-specific operators。由于没有
rollout、`N/W/Q` 和 ancestor backup，不能包装成标准 MCTS。

命名决策 artifact SHA-256：
`083fa3de5fdeaafc40b33069cad15b7115c835ed8bd71e2b5e8af9f704173b35`。

### 1350-scan 论文对标结果

当前 partial-decoder 两 seed subject-balanced 六区 macro 为 `0.554582`。官方 2024 冠军
hidden-test 六区 descriptive macro 为 `0.82457`；当前 ET/NETC/RC/SNFH/TC/WT 六项 raw
数字全部更低，macro raw gap 为约 `-0.2700`。因此目前没有超过论文。

更重要的是，当前 local protocol 有 5 个 GT-error corrective clicks，只使用 T1c 或 T2f
单模态，只评 foreground-positive rows，使用普通 binary Dice，并且 legacy WT 包含 RC；官方
协议是四模态、全自动、多类、全病例和 lesion-wise Dice/HD95，官方 WT 不含 RC。两组数字
不能形成正式 SOTA 比较。历史 `RC=0.58932` 还有 subject leakage，不得进入横向表。

### 新的主线验收标准

1. 先核实本地 1350 scans 是否对应官方修正 v2，并重建四模态 1 mm、613-subject grouped
   protocol。
2. 接入官方四原子类、TC/WT 和 lesion-wise Dice/HD95，对 absent classes 也计分。
3. 在同一 folds 上训练 nnU-Net/MedNeXt 强基线。
4. 构建无 GT prompt 的四模态自动多类 SAM-Med3D，先超过同协议强基线。
5. 配置冻结后，才允许在已核实且此前未访问的 additional 271 上做一次性外部确认。

现有 5-click SAM 结果保留为 interactive prompted delineation 支线，不再承担“超过 BraTS
自动分割论文”的 headline。完整 benchmark review SHA-256：
`65242dc51894d909096ff06ca2240247faff51973f5a2d7fbf21bce19ba24397`。

---

## 2026-07-20 四模态统一重启周报

### 方向纠正

用户确认主线从 Work #1 起都必须以 `t1n+t1c+t2w+t2f` 四模态联合输入。旧的单模态、
GT crop、GT-error click、positive-only binary Dice 接入整体退出硕士论文主证据链；保留的仅是
MedTAWS 随机 UCB-guided workflow-tree search 算法原型和历史 provenance。它不是标准 MCTS。

### 本轮完成

- 对 Work #1 做逐层审计并冻结四模态、无 GT、自动多类的新契约；新增 MedTAWS v2 搜索内核和
  MedTAWS / score-softmax / greedy / uniform 四臂严格等预算消融控制器。
- 对 1350 scans / 613 subjects 做源数据聚合体检：每例四模态加分割共 6750 文件，形状、方向、
  spacing、标签域和同例配准一致。该体检不可避免地读取了全源文件，但没有读取封存 manifest、
  运行预测或计算 sealed 指标；历史 269 例仍不得重新称为 clean evidence。
- 生成 provisional subject-disjoint 开发协议：948 train scans / 419 subjects，133 val scans /
  67 subjects。余下历史封存边界不变，additional 271 继续未访问。manifest SHA-256：
  `840c27488c1650a7c1b66a841b2e3b0a30d8780afae11112bbd4ded442f0fc52`。
- 完成严格四模态安全 loader、M1 checkpoint inflation、C0 learned scalar gate、无 prompt 四原子
  semantic decoder、固定 CE+atomic Dice+TC/WT loss，以及 BraTS 2024 公共算法版本的六区
  lesion-wise Dice/HD95。公共实现的逐 case 可执行 parity 仍是正式训练前 gate。
- 完成固定官方 MD5 的 fail-closed 源数据 verifier；尚未对真实源运行，因为缺少官方修正 v2 zip，
  且完整 hashing 需要另行批准 sealed-boundary 行为。

### B200 工程验证

两张 NVIDIA B200 上并行完成真实 SAM-Med3D base checkpoint 的 BF16 forward、loss、backward：

| Adapter | Loss | Peak CUDA memory | 结果 artifact SHA-256 |
|---|---:|---:|---|
| C0 | 2.953255 | 6,319,043,584 bytes | `46f036363af0c6a69e12edca50d336e904ac04ca32a5780ee1169c3b800d36cb` |
| M1 | 2.962843 | 6,388,141,056 bytes | `93686e7fcfcb7848eca155e9f1d701cac1d108b7fb0131998ab4d3ad94ba100a` |

两臂 patch embed / decoder 梯度均有限且非零，C0 gate 也有非零梯度。最终完整测试为
`320 passed`，新增模块 `py_compile` 和 scoped `git diff --check` 均通过；训练、评估、SSH 和
manifest builder 后台进程均已退出。

### 当前结论与下一步

本轮没有产生新 Dice，没有启动正式训练，也没有授权 test。下一步固定为：取得/核验官方修正
v2 source parity；完成 lesion-wise metric 的逐 case executable parity；在 provisional 协议上做
2-case trainer smoke；随后冻结 paired C0/M1 与 nnU-Net 强基线 pilot。只有固定 checkpoint 后才
允许 MedTAWS 自动多类搜索；不回 threshold、TTA、morphology 或 postprocess 搜索。

审计 artifact：`workspace/BraTS/research_audits/work1_four_modal_restart_audit_20260720.md`，
SHA-256 `c6a2618a74cd15853618eb7868c661bd4e15048c006ae96008f4d4360aafc91e`。Phase A 协议：
`workspace/BraTS/training_protocols/brats2024_four_modal_restart_phase_a_v1.md`，SHA-256
`b4e49b957391df4e4ebf522e188c43e105795666c808f7c0e4c5268100254f48`；其中
`authorizes_formal_training=false`、`authorizes_test_access=false`。

---

## 2026-07-21 四模态 Phase B 周报

- 冻结 Synapse 公开 lesion-wise scorer commit `43c9052`，13 个合成对抗病例 Dice、HD95 和 lesion
  counts 与本地实现完全一致，最大绝对差 0；报告 SHA-256
  `8c1754ea3ff36df0529e45f64dc6958a7cc4056d489827e732009cff1b31a1e8`。
- 定位官方 corrected-v2 为 `syn60086071` version 2、MD5
  `1d910b17d6cd32e38aa6296b8dfb7c77`。本地 Kaggle 三包只有版本完成标志，无官方 zip/file hashes，
  所以 source parity 继续 blocked，不能据“Kaggle version 2”启动长训练。
- 新增 label-free full-grid sliding-window inference、C0 strict checkpoint round-trip 和真实数据 smoke
  runner。双 B200 上 C0/M1 各完成两个不同 subject 的 2-step train、save、strict reload 和
  182x218x182 restore；不读 val、不计算指标。两步后全背景只作工程现象，不作性能结论。
- 948-case development-train 冻结为 764/335 fit 与 184/84 tune；133/67 原 val 改为一次性
  confirmatory development set，三者 subject overlap 为 0。manifest SHA-256
  `eaed5710dce8f10f6367b531d4161323028a7c6f449f3e702f7b7cb4b0e2aa35`。
- 完整测试 `327 passed`，worker 无 GPU background process。本轮无新 Dice、无正式训练、无
  test/additional-271 访问。

Phase B readiness audit SHA-256：
`5bf1a1d06d9e75c07b0f10a40a9657d7f021fd36f315e31971ed01829bc7ea0c`。下一步只准备 fit-only
nnU-Net v2 ResEnc N0 工程；正式 paired C0/M1/N0 pilot 等官方 archive parity 与 v2 preregistration。

---

## 2026-07-21 Work #1 冻结 base 先测周报

> **撤销说明（15:44 CST）：** 下列 frozen 数值来自错误的 mask-token 语义假设，只能作为失败
> 诊断，不能作为 baseline、搜索增益或论文结果。

- 即时纠正科研顺序：先用完全冻结的四模态 SAM-Med3D source base 测裸 baseline，再运行
  MedTAWS；Work #2 权重训练后置。184-case tune 已授权，133-case confirmatory 继续封存。
- 双 B200 完成 184 scans / 84 subjects 全网格裸评测：C0 六区 lesion-wise Dice `0.082263`，
  M1 `0.083186`。M1 ET/TC 均约 `0.000026`，macro 主要被 NETC absent-case credit 抬高；冻结
  source base 不具备可用的 BraTS 自动分割能力。
- tier1 搜索冻结为 native C0/M1 × 24 个全局 token mapping，五 seeds、每 seed/arm 12 draws，
  预算为 2208 logical backend units，四臂完全相同。MedTAWS final-best mean `0.092935`，相对
  M1 提升 `+0.009749`；但 greedy 为 `0.093116` 且 5/5 seeds 命中 oracle，优于 MedTAWS。
- oracle 增益分解：structure `0`、token mapping `+0.009930`、joint `+0.009930`、interaction
  `0`。因此这轮有 tune 提升，但全部来自全局 token 语义排列，不能写成结构搜索贡献。
- 修复了原四臂 controller 的不可变候选池问题；现在每 policy 有独立动态树，新 child、父节点
  child count、visit proxy、candidate digest 都逐轮更新。完整测试 `400 passed`。
- 7 个真实 2-unit tier2 结构模板已完成工程与 synthetic tests，但尚未跑真数据，不是结果。
  confirmatory/test/additional 271 均未访问，权重训练为 0 step。

baseline 汇总 SHA-256：
`405b1c26e35e44cf1a8a1cffe0e1cf7ab7b783eb4569d60de6636202333d0d2b`；tier1 search 汇总
SHA-256：`0fd00297292d8d983f88bd71175f4f4b50f538908ba51711de5db6e2d461714b`。下一步先为
tier2 构建共享 primitive cache 并做同成本层搜索；只有 workflow 与统计冻结后才允许一次
confirmatory 对照。

### 同日根因审计纠正

- 三个并行审计一致确认：SAM-Med3D token0 是 single-mask，token1-3 是同一目标的 ambiguity
  proposals，四者不是 NETC/SNFH/ET/RC 类别；因此 24 permutations 和 tier2 contract 全部撤销。
- 数据、标签、checkpoint load、完整网格恢复和 scorer 无额外压分 bug；`0.083186` 反而有 80.6%
  来自空区奖励，present-region 诊断约 `0.01616`。
- 官方风格 prompt 二值旧复现 WT/SNFH 分别为 `0.7161/0.7219`，证明 checkpoint 可用；但它依赖
  GT corrective click，不能与自动 BraTS leaderboard 比较。
- 旧入口已 fail-fast；新增 supervised target-prompt semantic adapter，每类仅使用官方 token0，
  定向测试 `35 passed`、完整 suite `403 passed`。B200 两病例 `m2` smoke 已完成：loss
  `1.881118 -> 1.537241`，三组梯度非零、参数更新和 strict reload/full-grid restore 通过，峰值
  `6.65 GB`；两步后全背景且未评分。下一阶段先训练有效自动语义底座，再恢复 MedTAWS 搜索。

---

## 2026-07-21 17:03:06（北京时间）baseline v2 纠错与单 seed 周报

> **历史 preliminary，已由 2026-07-21 17:39:58 五 seed final supersede。** 以下单 seed
> 数值保留作 provenance；“还缺四 seeds”的状态已经失效。

- 撤销 v1 whole-brain baseline。旧 smoke 的主要失效因素是 fused lowres 经 `128^3`
  upsample/downsample roundtrip、whole-brain resize 对 checkpoint 造成尺度/context OOD，以及
  zero-preserving population-std normalization 不等于 upstream TorchIO fixed-window exact 语义。
- 冻结 v2 A1：首 RITM 正点击居中、native-spacing `128^3` 固定窗；窗内四模态分别按 `x>0`
  计算 mean/unbiased sample std 后变换整个 tensor；四个原生单通道分支等权 late fusion；直接递归
  exact fused decoder low-resolution logits。
- seed `20260721` 的严格 184-case / 84-subject tune 已完成。D6@1/3/5 为
  `0.282351/0.386175/0.418646`，D3@1/3/5 为 `0.289645/0.393663/0.431232`。
- Dice@5 分区为 ET `0.361107`、NETC `0.226620`、RC `0.550598`、SNFH `0.440961`、
  TC `0.429663`、WT `0.502927`。full-grid scorer 保留窗外 FN；846 present queries coverage
  mean/min 为 `0.986089/0.501321`，`148/846` queries coverage 不完整。
- merged artifact：
  `workspace/BraTS/prompt_binary_native4_late/a1_torchio_v2_seed20260721_tune184_merged.json`，
  SHA-256 `78998bfd21c1afff0a5aa550271bc6591ddd4421e8be80759180cf4f7e21c405`。artifact 内嵌
  protocol SHA-256 `d4d34b6a1e876b3742616d3c752708d28de39fd399c9c0f44ed1b562969b0673`；v2 协议
  文件 SHA-256 `408bb8dfd185dcc296f113f46b7315f6c6d552fd5112c948ac7e02294b2a1f15`。
- 当前只是 5 seeds 的 `stage 1/5` preliminary；还缺 `20260722..20260725`。未授权也未运行
  MedTAWS、权重训练、confirmatory 或 test，不得把 `0.418646` 写成最终 baseline 或论文结论。

---

## 2026-07-21 17:39:58（北京时间）baseline v2 五 seed final 周报

- A1 oracle-interactive baseline 已完成预注册 seeds `20260721..20260725` 的全部 `5/5`，每个 seed
  都是相同 184-case / 84-subject development tune；旧 single-seed preliminary 已被本节覆盖。
- D6@1/3/5 mean ± population std 为 `0.281557±0.004033 / 0.382280±0.004517 /
  0.416978±0.004925`；D3 为 `0.289178±0.006505 / 0.389753±0.006895 /
  0.430292±0.008256`。
- region Dice@5：ET `0.367905±0.006310`、NETC `0.226707±0.007179`、RC
  `0.534767±0.010500`、SNFH `0.449514±0.006454`、TC `0.436665±0.004577`、WT
  `0.486307±0.020216`。
- per-seed D6@5：`20260721 0.418646`、`20260722 0.415488`、`20260723 0.422110`、
  `20260724 0.408165`、`20260725 0.420479`。
- aggregate：
  `workspace/BraTS/prompt_binary_native4_late/five_seed/a1_five_seed_tune184_aggregate.json`，
  SHA-256 `145067a8e77594d9c0c6e9274d6158774aab95bc42196c4ec8afce579acf1438`；绑定 protocol
  SHA-256 `408bb8dfd185dcc296f113f46b7315f6c6d552fd5112c948ac7e02294b2a1f15`。
- 该 final 仍是 GT corrective-click、region-present、binary-query 的 oracle-interactive tune
  baseline，不是 automatic BraTS 指标，不能与 leaderboard 直接比较。MedTAWS、权重训练、
  confirmatory 和 test 仍未授权、未运行。

---

## 2026-07-21 21:01:37（北京时间）官方提示协议 A0 对照周报

- 完成 SAM-Med3D upstream validation-protocol 对照：development tune 184 cases / 84 subjects，
  seed 2023；四个模态分别作为官方一通道输入，不做 fusion。
- 严格复现 GT label-centered TorchIO `CropOrPad(128^3)`、crop-only Dice、FP32、click 1-2
  RITM / click 3-5 uniform error、current sparse point only、exact lowres recurrence，以及上游 click
  raw-logit `>0.5` / score raw-logit `>0` 的阈值差异。
- official best-of-five D6：T1n `0.375347`、T1c `0.415088`、T2w `0.417798`、T2f
  `0.490891`；对应 exact-click-5 D6 为 `0.351904/0.393888/0.397107/0.470783`。
- T2f official D3 `0.496426`，相对 A1 D3 `0.430292` raw delta `+0.066134`；D6 raw delta
  `+0.073913`。官方提示/裁窗有提升，但没有把六区域指标提升到李云开论文的 `0.60-0.73`。
- T2f 的 SNFH/WT 达 `0.711023/0.745269`，T1c 更适合 ET/TC，T2w 更适合 RC。事后
  per-target modality oracle D6 `0.554994` 只用于提出 target-conditioned fusion 假设，不是结果。
- 完整性：3384 evaluated modality-query records；`1865` 条 exact click-5 低于 earlier best；无
  nonfinite、阈值或参数漂移。定向测试 `17 passed`。
- aggregate SHA-256：
  `ef08b4b957a06ca6bdedd261f8b5b32c08708251604f59ef36f7593075fdfbba`。confirmatory/test
  继续封存；本对照没有启动 MedTAWS 或训练。

---

## 2026-07-22 Paper1913 自动四模态纠错周报

- 三路并行完成论文协议、作者后补代码与本地差距审计。Paper1913 是 from-scratch automatic
  four-modal 3D U-Net，不是 SAM-Med3D 提示分割；论文 D3 U-Net/nnU-Net/Proposed 为
  `0.8388/0.8588/0.8745`。
- 官方页无 code/supplement；作者后补仓库自称 partial implementation，无 LICENSE/权重，且
  `128^3` forward 与 loss smoke 均维度失败。本地采用独立 clean-room reconstruction。
- 新增 P0 3D U-Net、training-only tumor-centered `128^3` crop、D3 voxel Dice/HD95、resume、
  manifest hash binding、label-free full-grid validation 与 fit-crop overfit gate。完整测试
  `434 passed`，定向回归 `31 passed`。
- 两例 500-step gate：CE/Composite crop D3=`0.889378/0.895191`，证明四模态、标签、loss 与模型
  能学习完整 WT/TC/ET，原低分根因是错误任务接口加未训练权重。
- 764-step、8-case full-grid pilot：CE D3 `0.269754`，WT/TC/ET=`0.770658/0.019193/0.019409`；
  Composite D3 `0.513081`，WT/TC/ET=`0.735078/0.400264/0.403900`。Composite 晋级正式长训，
  CE 保留为论文文字对齐弱基线。
- D6 lesion-wise 为 `0.192669/0.214819`，含明显 absent-region credit，不能单看 macro。8-case
  结果只作工程 pilot，不是正式论文对照。
- 133 confirmatory、consumed test、additional labels 均未访问；MedTAWS 等待自动 checkpoint
  正式训练与冻结。双 B200 pilot 已结束，无训练/评估 compute 残留。

---

## 2026-07-22 13:10 CST 正式四模态 baseline 长训周报

- 建立长程 goal：full-184 tune D3 先过 `0.80`，优先超过 Paper1913 U-Net `0.8388`，高分目标
  `0.8588`；未达标继续逐因素训练迭代，不能用短 pilot 或 MedTAWS 搜索替代 baseline。
- 数据边界保持 764 fit / 184 tune subject-clean；133 confirmatory/test 继续封存。
- 长训器加入严格 run-config/RNG resume、逐 epoch atomic latest、D3/D6 双 best、present-only D3、
  D6 FP-lesion 汇总和 SW batch 8；修复多 worker 下 crop epoch 不更新。B200 smoke 峰值
  `7.073 GiB`，resume pass，完整 suite `435 passed`。
- 双 B200 正在并行跑 100 epochs Composite：P0 tumor-centroid 与独立 P0+ label-balanced
  (foreground 0.70)。共同为 batch1/BF16/AdamW `1e-4,1e-5`、完整 184 tune 每 10 epochs、
  label-free full-grid validation、按 D3 选择。
- 13:11 CST 确认 worker `1007515/1007672` 各一训练进程；输出目录与全部冻结项见
  `workspace/BraTS/training_protocols/paper1913_p0_formal_goal_v2.md`。当前只有启动状态，尚无正式
  full-184 Dice；baseline 冻结前不启动 MedTAWS。
- 13:16 CST P0/P0+ epoch 1 均已落盘，loss=`2.209684/2.224704`、crop foreground
  min=`803/33`、global step 均为 764；run_config 除 sampler 外一致。首次 full-184 指标在
  epoch 10，协议 SHA-256 `15cbc6a7bb4d57672aaad43021766dd96266cf112a7c68408f6afb4a41c3a55a`。
- full-184 结果产生前已预注册互斥 contingency：FP 爆炸只改 Tversky，小区域欠学只改 focal CE，
  广泛 underfit 只换约 18.97M 参数 MONAI SegResNet，明确过拟合才只加三轴 flip。当前不授权
  contingency arm，也不提前停止 v2；详见 `paper1913_p0_contingency_audit_20260722.md`。
- P0 epoch-10 首次 full-184：all-case D3 `0.624460`，WT/TC/ET
  `0.851969/0.509812/0.511597`；present-only D3 `0.726572`，TC/ET
  `0.660042/0.667704`；D6 `0.362550`。FP lesions 总数 3975，`F=3.60054`。主要短板为
  TC/ET-absent cases 的假阳性；P0 继续训练，等待 P0+ epoch10 与 20..100 曲线，不提前启动新臂。
- P0+ epoch-10 full-184 D3 `0.610302`、present D3 `0.711927`、D6 `0.361959`、F `3.61866`。
  P0 在 all/present D3 高 `0.01416/0.01464`，D6 近似持平；P0+ 只改善 RC，未降低总 FP，故按
  预注册规则暂保留 P0。两臂继续到 100 epochs，不能提前转 MedTAWS。
- P0 epoch-20 full-184 D3 `0.647468`（e10 +`0.023008`），present D3 `0.748068`，
  WT/TC/ET `0.858391/0.540770/0.543242`，D6 `0.366376`；F 从 `3.60054` 降到 `3.44475`。
  曲线仍改善，继续正式训练；尚未过 0.80，不能解封 MedTAWS/confirmatory/test。

---

## 2026-07-22 14:55 CST DPD-SAM 单模态方向纠正

- 用户撤销四模态五类 P0/P0+ performance 主线，改为 T1ce 单通道 residual-tumor 二值分割；
  旧 D3/D6 与新 binary DSC 不可比较。
- 两条训练已 Ctrl-C；P0/P0+ 最后 atomic epoch 为 29/17，两个 B200 均确认无 compute process，
  output 写入 no-resume termination marker。
- 本地 DPD-SAM PDF 核验：whole-volume `128^3`、automatic continuous prompt、patient-level
  8:2 + train内约20% val、200 epochs/batch2/AdamW4e-4 wd0.1、五独立 seed；BraTS2024
  SAM-Med3D DSC `74.1+/-1.5`，DPD-SAM `86.8+/-0.6`、IoU `77.0+/-0.5`、HD95
  `5.0+/-0.8 mm`。
- 当前只剩一个必须解决的定义问题：PDF 未声明 BraTS 1/2/3/4 如何折叠为 residual-tumor binary
  GT。映射冻结前不启动 GPU；新草案为 `dpdsam_t1ce_binary_restart_v0.md`。

---

## 2026-07-22 16:21 CST T1ce corrected baseline 周报

- residual-tumor target 已冻结为 `{1,2,3}`，RC=4 排除；新 subject-disjoint development split
  为 872 fit / 209 validation，outer 269 继续封存。
- 1081-case 审计发现并修复 legacy nearest 与 center-aligned image 的系统错位。GT roundtrip DSC
  mean `0.788933 -> 0.924792`；旧 T1ce smoke/pilot/checkpoint 统一 invalid。
- 新 evaluator 固定 dual-grid：model-grid 作为论文对标 headline，nearest-exact source-grid 强制附报。
- corrected 256/64 e20：B1 static+LoRA-r4 model/source DSC `0.325470/0.325024`；D1 dynamic
  K4/N1 `0.161892/0.161581`。D1 不扩全量、不接 CDR。
- 16:27 CST 用户要求暂停确认 target。corrected B1 全量停在 epoch 1（无 validation），B1
  no-augmentation pilot 停在 epoch 8（e5 DSC `0.202683`），A2 接入停止；两张 B200 已核验无
  compute。尚未启动 MedTAWS，尚未访问 outer test，未经确认不恢复。
- 工程已补 strict resume、完整 RNG、fit-only clustering firewall、dual-grid atomic validation、D1/A2
  模型；完整 suite `479 passed`。v2 protocol SHA-256
  `5feed4bb21f3c9d8f89e8fcd1d9ac48a74d71657409540d2048f69d386539cd4`。

---

## 2026-07-22 16:34 CST 全量 80/20 随机切分周报

- 1350 scans / 613 subjects 按 subject 随机分组，seed `20260722`；精确得到 1080 train scans /
  494 subjects 与 270 sealed-test scans / 119 subjects。
- case/subject overlap=0，union=1350；全部 6750 个原始 modality/seg 路径存在，test NIfTI 未打开。
- 同 seed 重建逐字节一致；split manifest SHA-256
  `215df5594550d939707b25dc3671082ae68cb000b8c1fa9615d9bcb604c0bc25`。
- 当前只完成外层 split，尚未切 train 内 validation、未恢复 GPU 训练、未启动 MedTAWS。

---

## 2026-07-22 16:52 CST 原始 SAM-Med3D T1ce 零训练 test 周报

- 用户授权访问新 20% test；270 cases 全部完成，仅用 T1ce 和原始、未微调
  `sam_med3d_turbo.pth`，整卷 resize `128^3`，固定阈值 0.5，无 TTA/后处理/参数搜索。
- 严格无提示 model-grid macro DSC：ET `0.002272`，residual `{1,2,3}` `0.006127`；
  image-only T1ce 脑区中心正点分别为 `0.002231/0.005656`。
- ET 仅 209/270 present，无提示 ET present-only DSC `0.002936`；residual 270/270 present。
- 结论：原始 SAM-Med3D 没有目标提示时不能直接充当自动 ET/residual segmenter；该实验不是
  DPD-SAM 论文所报 prompted SAM-Med3D `0.741` 的复现。后续必须把“官方/GT oracle prompt
  sanity check”和“可部署 automatic prompt baseline”分开报告。
- 完整报告 SHA-256：`88995830d20ba732f37fd2f1da04c7156aaacf7f164532431b07f608263f5273`；
  单卡 B200 任务已退出，无残留 compute。

---

## 2026-07-22 17:02 CST 三次累计 GT corrective-click 周报

> 17:18 CST 纠错：本节 `0.011020` 缺少 upstream recurrent mask，不是有效 baseline；以文末
> 官方接口纠错结果为准。

- 同一 270-case test、T1ce、原始未微调 SAM-Med3D；目标为 residual `{1,2,3}`。
- 第 1 点均匀采 GT 前景正点；第 2/3 点均匀采 prediction XOR GT，FN 正/FP 负；累计历史点，
  不传 recurrent mask，固定阈值 0.5，无搜索或后处理。
- Model-grid DSC click-1/2/3=`0.008049/0.010133/0.011020`；source-grid=
  `0.008079/0.010126/0.010966`。Click-3 相对 click-1 平均 `+0.002970`，179/76/15 例
  改善/退化/相同。
- 270 条轨迹全部通过点击合约核验，但结果远低于论文 prompted SAM-Med3D `0.741`，属于有效的
  negative reproduction。下一步应单因素核验 upstream GT-centered ROI 与 recurrent low-res mask，
  不能进行 threshold/TTA/postprocess 搜索。
- 报告 SHA-256：`c495cf2626dbc7edcbeb17f2c9a72a8d16b68e2b2f633a73c4525c55637ad7c8`；
  单张 B200 已退出，无残留 compute。

---

## 2026-07-22 17:18 CST SAM-Med3D 官方接口纠错周报

- 当前 runtime 重跑仓库 Phase-0，3 例历史 `tumor_WT` best-of-5 均值 `0.7194`，证明原始
  checkpoint/decoder 正常。
- 早先 `0.011020` 的主 bug 是没有把上一轮 low-res logits 作为 recurrent dense mask prompt，
  同时还错误沿用了整卷 LAS resize，而非 upstream `ToCanonical + GT-centered CropOrPad128`。
- 新 270 例、T1ce、residual `{1,2,3}`、用户规定累计三点击 + recurrent mask：C1/C2/C3
  DSC=`0.243101/0.291748/0.330096`，best-of-3=`0.339958`。
- 完全 upstream RITM/current-point/recurrent-mask 的 C3=`0.330285`、best-of-3=`0.352253`。
- `0.330096` 是当前有效三点击 oracle-crop结果，但不等于论文 `0.741`：论文未公开 residual
  raw-label mapping 和 automatic prompt；仓库历史约 `0.716` 是 T2-FLAIR、`{1,2,3,4}` WT、
  5-click best-of-5 的不同任务。
- 完整报告 SHA-256：`18b6e245fe0a566ec7498d3f478b1d9e4405d4ae951b4b09e4031223ade37460`。
- 命名核验：官方 post-treatment WT 与当前 residual 都是 raw `{1,2,3}`，所以 C3
  `0.330096` 同时就是 270-case T1ce official-WT 结果；历史 `tumor_WT={1,2,3,4}` 含 RC，
  是旧兼容口径而非本轮目标。

---

## 2026-07-22 18:17 CST MedTAWS-GRPO 论文初稿

- 三个并行只读 agent 完成参考论文结构、MedTAWS 和 GRPO 证据审计；主 agent 完成英文全文。
- 产物位于 `../article/MedTAWS_GRPO/`：4,203 词 Markdown、8 页 PDF、LaTeX/BibTeX、
  方法图、结果图、证据矩阵和 confirmatory 实验清单。
- PDF SHA-256：
  `0d2d98f93b13beb7c377927a20b5b09b9d6a5a92d7c38876cc438e3c50d4f374`。
- 论文统一使用 MedTAWS（built on AFlow、stochastic UCB-guided workflow-tree search）和
  GRPO-style group-relative prompt-trajectory alignment，分别避免伪装标准 MCTS 与标准 PPO。
- 历史 RC 匹配对照为 supervised `0.57283` vs GRPO-style `0.58932`，delta `+0.01649`；
  由于单模态、oracle、GT crop 和 subject leakage，只作为 retrospective development evidence。
- 当前没有 subject-clean 1,350-scan 联合结果，不能声称超过外部论文。最终需要冻结
  base / MedTAWS / GRPO-style / combined 四臂后一次性评估 holdout。
- 本轮未启动 GPU、未访问新 test、未恢复 inference workflow 或后处理搜索。

---

## 2026-07-22 19:55 CST official-WT Dice 核验

- residual tumor 定义冻结为 official `WT={1,2,3}`；现有 270-case 结果本来就使用该 target，
  所以无需重跑。
- 累计三点击 + recurrent mask：scan C1/C2/C3 =
  `0.243101/0.291748/0.330096`，119-subject balanced =
  `0.242981/0.292463/0.329269`。
- C3 subject-bootstrap 95% interval `[0.304634,0.353668]`；best-of-3
  scan/subject = `0.339958/0.338133`。
- crop 外 WT 前景计 FN 后 C3 仅变为 scan/subject `0.329909/0.329066`，低分不是 crop
  truncation 造成。
- 历史 `~0.7` 是 T2-FLAIR、legacy `{1,2,3,4}`、5-click best-of-5，不能混用。
- 本轮只重聚合现有 JSON，无 GPU inference、无新 test 访问或搜索。

---

## 2026-07-23 12:07 CST T1ce→WT SAM-Med3D 微调周报

- 完成 subject-clean 两级切分：外层 1,080 train / 270 test；训练池内 972 train /
  108 validation，内外 split 均无 subject overlap。
- 任务固定为 T1ce 单通道输入、binary official `WT={1,2,3}`，RC `{4}` 排除。
- 完成三次累计点击 + recurrent mask 的专用训练/评估链；只训练 prompt encoder 和 mask
  decoder，10 epochs、BF16、batch 1、LR `1e-4`。
- epoch 8 validation 最佳。独立 reload 后 C1/C2/C3=`0.514882/0.530766/0.535888`；
  untouched base 为 `0.224873/0.273186/0.307312`，同 seed C3 gain=`+0.228576`。
- 配置先冻结、再唯一一次评估 270-case test：scan C1/C2/C3 =
  `0.530552/0.544826/0.549828`；119-subject balanced C3=`0.531978`。
- 最佳 checkpoint SHA-256：
  `4a325a6ffca148941c5d66f4802f331d13b8ddb7862da299981e8fd6be84d7c8`；
  test report SHA-256：
  `91f6ff7e9426d74e6d7fae73fe172806c1427da17d951b18614e65a9389700db`。
- 审计限制：270-case test 在历史上做过 zero-shot baseline，不是全局 pristine；
  当前训练/epoch 选择未使用 test labels。旧/新 test click seed 不同，不作 paired claim。
- 无 MedTAWS、threshold、TTA、postprocess 搜索；完整回归测试 `501 passed`。错误协议
  早期 run 已标记 `INVALID_*`，B200 训练与评估进程均已退出。

---

## 2026-07-23 13:58 CST 低学习率重训周报

- v1 的 train Dice 持续升高、validation 在 epoch 8 后下降，说明不是单纯 epoch 太少，
  而是原恒定 LR `1e-4` 已进入过拟合。
- 只用 972 train / 108 validation 并行比较三个低 LR 日程；固定 validation seed，
  全程未访问 270-case test。
- 独立 reload C3：v2a base→`3e-5` e9=`0.531548`；v2b v1-e8→`1e-5`
  refine-e1=`0.539075`；v2c v1-e8→`3e-6` refine-e6=`0.534862`。
- v2b 为 validation winner，相对 v1 `0.535888` 提升 `+0.003187`；
  subject-balanced `0.535369→0.539514`。
- 配对结果 56/52 cases、27/23 subjects improve/worse；subject bootstrap 95% interval
  `[-0.003220,0.011167]` 包含 0，结论限于小幅正点估计。
- v2b checkpoint SHA-256：
  `2b7d5907c78ce2ea7e1b01ff393a5597289af08670f9b4cf2d6e203d2bed5015`。
- 本轮未跑 test、MedTAWS、threshold、TTA 或 postprocess；两张 B200 无 compute 残留。

---

## 2026-07-23 14:07 CST v2b 冻结 test 周报

- 用户授权后先冻结 v2b 配置，再唯一一次评估 270-case test；输入仍为 T1ce，目标
  `WT={1,2,3}`，三次累计 GT/error click 和 recurrent mask。
- v2b scan C1/C2/C3=`0.532414/0.548724/0.553028`，subject-balanced C3=`0.535621`。
- 相对 v1，C3 scan/subject delta=`+0.003200/+0.003643`；158/112 cases、66/53
  subjects improve/worse。
- 10,000 次 subject bootstrap 95% interval=`[-0.002009,0.009182]`，仍跨 0，
  结论限于小幅正点估计。
- test report SHA-256：
  `827a98496c4554f029471d7d6a9f25fc613a00e80f4743977e0b070af7040045`；
  冻结配置 SHA-256：
  `981dee6cf926f2a52edc15c3e57d9e42919673900dd79bcd7df4b31e692e14c3`。
- 最终冻结/结果绑定记录 SHA-256：
  `884973d5f69ff8012089f762c54cf723ecce23346f19dc5a2f14f3f749c84bb1`。
- test 未参与训练/validation 选型，结果不再驱动学习率调参；但该集合此前已用于 v1，
  因而不是 globally pristine external test。
- 当前只有 worker `1007672` 的一张 NVIDIA B200；270-case 评估约 175 秒，已退出，
  未运行 MedTAWS、threshold、TTA 或 postprocess。

---

## 2026-07-23 14:38 CST 本地存储清理

- 审计本地 54G `workspace/BraTS`，其中 119 个 `.pth` 占 51.084 GiB。
- 删除 23 个有 best 替代的 unreferenced latest、7 个 smoke/INVALID 权重及仓库缓存，
  释放 12.905 GiB。
- 清理后剩余 89 个 checkpoint / 38.180 GiB；本地盘使用率 `79%→69%`，可用空间
  `27G→40G`。
- 正式 best、当前 v2b、GRPO/DPD-SAM 研究证据、结果 JSON、日志、源码、venv 和
  `/mnt` 数据均未删除。
- v1/v2a/v2b/v2c 及 v2b 270-case 报告哈希复核通过；剩余权重没有字节级重复。
- 清理记录：`workspace/BraTS/LOCAL_STORAGE_CLEANUP_20260723.json`，SHA-256
  `1474cdf75044fb9c8e13debf8d34f0e97e480c1c17f271feaf2c7b309a953671`。

---

## 2026-07-23 15:10 CST whole-volume encoder-PEFT 周报

- 三路并行完成 whole-volume/augmentation、image-encoder adaptation 和实验协议审计。
- 新管线为完整体积 `128^3`、单次 source-nonzero z-score、train-only affine/elastic/noise；
  validation 强制无增强。
- 首轮 encoder adaptation 为 12-block Q/V LoRA-r4/a8，不更新 encoder base weights，
  不能表述为 full encoder 微调。
- 定向测试 `33 passed`，实现侧完整集合 `44 passed`；真实 development 数据 smoke 通过。
- A10 2/2/e1 GPU smoke 成功，train/validation C3=`0.215246/0.085975`，仅作工程 gate。
- 正式 972/108、50-epoch arm 已于 15:08 CST 启动，PID `3166`，A10 稳态显存约 6.9 GiB。
- 预注册 SHA-256：
  `3b2e9268d109bf62258bc3ddfcb37719c980614bf75a74d989eb5a4298eaa2e6`。
- 270-case test 保持封存；未运行 MedTAWS、threshold、TTA 或 postprocess。

---

## 2026-07-23 15:22 CST 自动 test watcher 周报

- 用户授权当前 FV-LoRA-Aug 训练结束后自动评测 270 cases。
- 修复旧 evaluator 的 GT-crop 写死和 LoRA checkpoint strict-load 缺口；whole-volume 一例
  clean-validation 端到端 smoke 通过。
- watcher PID `176743` 已挂载，当前只等待训练 PID `3166`，未打开 test manifest。
- 完成后必须先核验 50 epochs、validation-selected best 与协议/代码哈希，再落 freeze JSON，
  最后才运行一次 whole-volume 270-case 评测；任一检查失败则停止。
- evaluator/watcher/launcher SHA-256 已记录在 handoff 第 66 节；定向回归 `24 passed`。

---

## 2026-07-24 10:36 CST FV-LoRA-Aug 完成与冻结 test 周报

- 正式 972 train / 108 clean-validation 训练完成 50 epochs，总耗时 `25,107 s`；按
  validation C3 选择 epoch 33=`0.532817`，epoch 50 为 `0.523547`。
- 固定任务为 T1ce、binary official `WT={1,2,3}`、排除 RC `{4}`；whole-volume
  `128^3`、train-only affine/elastic/Gaussian noise、12-block Q/V LoRA-r4/a8、
  三次累计 click 与 recurrent mask。该 arm 是 encoder PEFT，不是 full unfreeze。
- 冻结后 270-case test：scan C1/C2/C3 =
  `0.508103/0.516545/0.523030`；119-subject balanced =
  `0.494241/0.503163/0.509143`。
- C3 median/std=`0.540526/0.177963`，IoU=`0.373047`；相对 clean-validation
  `-0.009787`。相对旧 GT-crop v2b C3 `0.553028` 为 `-0.029998`，但多项协议同时变化，
  不能写成 LoRA 或 augmentation 的单因素退化。
- watcher 的预期 manifest SHA 有一位抄录错误，评估器在解析 row/NIfTI 前 fail-closed；
  `split_manifest`、`SEAL`、文件实算三重核验后，先写不可覆盖的 SHA 勘误，再用原冻结
  epoch-33 checkpoint 完成唯一一次评估。
- checkpoint/report/final-binding SHA-256 分别为
  `c1d1bd79791aa7beeffc541095472dec968630cc70ea7967128c86552bd7f1ec`、
  `a087c357bbc71c60bba63cf179692266d2ebe19c64f58b44d8821986ae8ab13b`、
  `ae0f636f6246bb9ad4920b28bd3feefd568c56be4aa200812f131b3461c446f4`。
- 270-case 评估耗时 `388.14 s`，A10 观测显存约 `2.49 GiB`，进程已退出。本 arm 未达到
  `0.7`，也没有超过项目内 v2b headline；未运行 MedTAWS、threshold、TTA 或 postprocess。
