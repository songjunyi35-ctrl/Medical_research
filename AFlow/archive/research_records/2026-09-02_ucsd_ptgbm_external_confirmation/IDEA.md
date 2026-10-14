# Experiment idea: 患者完全隔离的 UCSD-PTGBM 外部确认

Status: preregistered; external label contents not yet accessed

## 1. 现在的问题是什么？

当前复用内部 test270 的最高 Scan Dice 点估计是 GCSAM fit864 epoch184 的 `0.752003`，但它的 Subject Dice/Point 只有 `0.737266/252/270`；Full972 random-init voxel-KD epoch168 的 Scan Dice 略低为 `0.751760`，却有更高的 `0.737327/262/270`。两者差异的配对置信区间都跨 0，而且 test270 已多次揭示，不能再承担独立验证。

TCIA 官方 UCSD-PTGBM 主 release 有 136 subjects / 184 studies。官方 `participants.tsv` 表明其中一部分就是现有 BraTS 2024 病例：主 release 的 184 个扫描里，109 个扫描所属患者已经出现在本项目 1,350 例清单；按“患者任一时间点出现过就全部排除”后，尚有冻结的 75 scans / 53 subjects。它们提供了真正未被本项目训练、验证或复用 test 使用的病例级确认队列。

## 2. 一句话直觉

像把两位在旧题库上分数接近的学生带到另一所医院，用他们从未见过、且没有同一患者其他检查的新试卷各答一次；不改答案规则，只看谁的本领更能带出原考场。

## 3. 为什么选择这个 idea？

- 最大证据瓶颈不是再找一个内部点估计，而是 fixed validation108 和 test270 都已反复使用，缺少患者独立的外部确认。
- 三条新稳定性路线都说明单次 inner 上升不能代表跨 seed 泛化：LAWA6 在 fresh Full972 fixed validation 相对 archived champion 为 `-0.007093/-0.004897/-1 Point`；Gamma 增强由 inner `+0.006151/+0.005570/+1` 反转成 final `-0.003848/-0.002312/0`；Lookahead 在 inner 直接为负。
- UCSD 论文说明全部模态已配准到 1 mm 各向同性、256³ MNI 图谱空间；标准 BraTS 四区分割先自动生成，再由 trained radiologists 人工修正，并由 board-certified neuroradiologist 反复审核直至批准。这比自动模型标签更适合作为确认数据。
- 现有负结果否定的是具体优化器、增强或权重平均实现，并没有检验冻结模型跨患者、跨发布队列的泛化；因此它们不构成对外部确认的反证。
- 一手来源：TCIA collection `https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/`；Scientific Data DOI `10.1038/s41597-025-06499-z`；TCIA 官方 GitHub repository `https://github.com/data-nih/tcia/releases`。

## 4. 具体怎么做？

先读取 TCIA 官方 release 资产清单和 `participants.tsv`，把 `sub-XXXX_ses-YY` 映射到 BraTS ID。主 release 必须精确形成 184 个唯一的 `run-post_T1w`（T1 对比增强）与 `BraTS_tumor_seg` 配对，并与本项目冻结的 1,080 train + 270 reused-test 清单做患者级差集，只保留 75 scans / 53 subjects。

下载时逐文件核对官方 SHA-256。每例按现有系统的相同输入链处理：转 canonical RAS → N4 偏置场校正（shrink 2）→ 只根据图像非零区域裁脑 → 在源图像非零体素上做总体标准化 → 三线性缩放图像、最近邻缩放标签到 128³。Whole Tumor（全肿瘤，WT）只取标签 `{1,2,3}`，明确排除切除腔标签 `4`。

同一个缓存依次输入两个冻结模型：Primary 为 Full972 random-init voxel knowledge distillation（逐体素知识蒸馏）epoch168；Secondary 为 GCSAM（梯度中心化锐度感知最小化）fit864 epoch184。两者只做一次前向，以 raw logit `>0` 得到分割，并以最高 logit 位置作为自动点。输出每扫描 Dice、每患者平均 Dice、阳性扫描 Point hit；对 WT 为空的扫描另报空预测率和假阳性体积。

## 5. 唯一变化与固定项

- 唯一变化：评测病例从已复用内部 test270 换成患者级完全不重叠的 UCSD-PTGBM 75 scans / 53 subjects 外部确认队列。
- 固定不变：两个 checkpoint、4.70M plain SegResNet 架构、单通道 T1ce、WT=`{1,2,3}` 且 RC=`4` 排除、canonical/N4/image-only brain crop/128³/源非零总体 z-score、logit 阈值 0、1× forward、无测试时增强（Test-Time Augmentation）、无 Segment Anything Model、无后处理。
- 不训练、不选 epoch、不调 threshold、不按外部结果选择模型、不做 ensemble。

## 6. 成功、失败和停止规则

- Primary 指标：75 个冻结扫描上的 Scan Dice；共同主要报告 Subject Dice、WT 阳性扫描 Point hit，以及 WT 阴性扫描假阳性体积。
- 绝对泛化支持门：Primary Full972 KD 的 Scan Dice `>=0.70`、Subject Dice `>=0.70`、阳性扫描 Point hit `>=0.90`。
- 两模型比较支持门：Full972 KD 相对 GCSAM 的 Scan 和 Subject 点估计均不低，二者配对 bootstrap 95% CI 下界均 `>=-0.003`，且阳性 Point hit 少不超过 1 个。
- 数据工程门：官方主 release 必须正好 136 subjects / 184 image-seg pairs；参与者表 178 subjects / 243 timepoints；患者级差集必须正好 53 subjects / 75 scans；与全部 1,350 例现有清单 case/subject overlap 均为 0；文件 hash、图像/标签 geometry、finite、label subset `{0,1,2,3,4}` 全部通过。
- 所有 75 例在标签访问前冻结。即使 WT 为空、质量较差或分数低也不删除。
- 任一数据工程门失败：在模型前停止；不得用近似病例数、模糊文件名或部分下载替代。
- 结果门失败：照实归档为外部负结果；不得调阈值、N4、crop、标签集合、病例或模型后重测。
- Bootstrap 固定 10,000 次，seed `2026090201`；Scan 以扫描重采样，Subject 以 53 位患者重采样；模型差值使用同一单位配对重采样。

## 7. 成本与风险

- 训练时间：0；不改变任何权重。
- 峰值显存：预计不超过历史单模型约 20.3GiB；可以分两卡各跑一个模型。
- 磁盘：官方选择性下载约 256MB；75×128³ float32+uint8 cache 约 0.73GiB；连报告预计 `<1.5GiB`。
- 推理：两个冻结模型各 75 次，共 150 次 locator forward；正式部署仍是任一单模型 1× forward。
- 最大技术风险：UCSD 是术后、同源 BraTS 标注体系但存在明显域差异；现有 1,350 例与 UCSD 有已证实的重叠，必须依赖官方映射做患者级排除。官方 main release 与另列的 BraTS challenge test package 必须严格分开。

## 8. 最终结果（完成后填写）

- External confirmation：待运行。
- 置信区间/病例分布：待运行。
- 是否支持原直觉：待运行。
- 学到了什么：待运行。
- 下一步：冻结为一次性外部确认；结果不得反向调参。
- 报告/归档/哈希：待填写。

