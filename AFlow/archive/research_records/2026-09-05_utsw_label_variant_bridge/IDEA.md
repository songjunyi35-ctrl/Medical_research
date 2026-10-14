# Experiment idea: UTSW人工标签双版本的固定3例来源桥接

Status: completed，预注册唯一pair门失败；0模型访问。

## 1. 现在的问题是什么？

UTSW-Glioma 官方清单已证明625例中362例是人工细化标签，完整包为6349文件/24,667,030,300 bytes；但每个人工病例同时有 `rtumorseg_manual_correction.nii.gz` 与 `tumorseg_manual_correction.nii.gz`，影像也有 `brain_t1ce.nii.gz` 与 `brain_t1ce_ants.nii.gz`。不知道哪一对共享物理网格，就不能把人工标签接给模型。ReMIND刚因物理网格门失败，说明这一步不能省略。

## 2. 一句话直觉

同一题有两版答题纸和两版底图，先把三份固定样卷叠起来看格子是否严丝合缝；不能根据模型哪版得分高来倒选答案。

## 3. 为什么选择这个idea？

- UTSW来自UT Southwestern，不属于BraTS 2024 post-treatment七家来源机构；官方TSV可逐例区分362份人工细化标签。
- 官方目录没有MD5/SHA，也没有解释两个manual文件名；全量来源可用前只剩传输复现、几何和标签类别的接线问题。
- 既往N4、RHUH与ReMIND负结果没有访问UTSW；它们只说明必须先冻结来源桥接，不能证明UTSW不可用。
- UTSW是术前，而模型训练是治疗后；即使未来可用也只能作为单独的跨阶段外推队列，不混入现有test270榜。

一手来源：TCIA [UTSW-Glioma](https://www.cancerimagingarchive.net/collection/utsw-glioma/)、[数据论文](https://doi.org/10.1038/s41597-026-07274-4)、[数据DOI](https://doi.org/10.7937/DFAE-1B86)。

## 4. 具体怎么做？

输入是预先固定的首3个人工病例BT0001/BT0002/BT0003。每例只取两份T1ce、两份人工标签和一份原FeTS自动标签，共15文件；从同一官方路径下载到两个独立目录。先比较两次文件字节，全部一致后才用Nibabel读取header和数组。对四种 `2 images × 2 manual labels` 配对逐例比较shape和affine，并记录标签整数值、非零支持及与FeTS标签的关系。没有teacher、loss、融合或模型输出。

## 5. 唯一变化与固定项

- 唯一变化：允许访问固定15个UTSW样本文件的两份传输副本，判断标签版本接线。
- 固定不变：病例BT0001/2/3；五个文件名；官方package1265；清单路径+size SHA=`0f7fd05936e9b216ae744961d45d33a602d4ea7ba5e7fcaa5d6a54c8bd2c9a3c`；15文件单份总大小28,880,493 bytes；不访问其余622例；不加载checkpoint；旧cohort、训练、阈值、TTA、SAM、后处理均不变。

## 6. 成功、失败和停止规则

- Primary工程指标：15/15两次SHA-256与大小一致；四个预列配对中在三例均shape exact且affine max绝对差≤`1e-6`的数量。
- Promotion gate：两次传输15/15 exact；所有影像finite/nonempty；两份人工标签均为finite整数、含0且非零；四个配对中恰好一个三例全过几何门。只有这时才能把该唯一配对记为未来输入候选。
- Failure/stop rule：下载不复现、任一数组无效、没有配对或多于一个配对全过，都停止为“标签版本仍未消歧”；不换病例、不用Dice挑标签、不放宽affine门。
- 是否允许进入test：否。本桥接不运行任何模型，也不授权下载全包。

## 7. 成本与风险

- 训练时间：0。
- 峰值显存：0 GPU。
- 磁盘：两份副本预计57,760,986 bytes，加少量报告。
- 推理forwards/延迟：0。
- 最大技术风险：两份标签只是不同方向/重采样表达，可能同时与两个同网格影像对齐，导致固定门按“多解”失败；这不否定UTSW来源，只表示还需要发布者说明或另立不依赖模型分数的语义门。

## 8. 最终结果（完成后填写）

- Validation：不适用。
- Test（如授权）：不适用。
- 置信区间/病例分布：固定3例来源工程探针，不作人群统计；双传输15/15 exact。
- 是否支持原直觉：部分支持。`r`标签是三例一致的完整240×240×155网格，但两种T1ce都与它几何exact，唯一pair门失败。
- 学到了什么：非`r`标签可能是裁剪块；`r`是全网格版本。几何不能在FeTS与ANTs两种skull-strip影像之间选择。
- 下一步：停止当前实现；如继续须在模型前另立官方处理谱系输入门，禁止用Dice挑图。
- 报告/归档/哈希：[RESULT.md](RESULT.md)、[RESULT.json](RESULT.json)、RAW_QC SHA=`0f31398be358b4c38d3c961adcf08b63eae4617002ea5b2814aa773b484ee80f`。
