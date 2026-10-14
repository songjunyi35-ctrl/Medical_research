# GLIS-RT外部来源审计：受控访问且标签不等同BraTS WT

日期：2026-09-05 UTC

## 结论

GLIS-RT规模大、T1ce和人工标注质量高，但不能作为当前T1ce/Whole Tumor同口径外部确认：官方主体数据为NIH Controlled，当前没有相应受控访问资格；更关键的是人工真值为放疗Gross Tumor Volume（GTV，肉眼可见肿瘤靶区）和Clinical Target Volume（CTV，加入潜在扩散与放疗规划边界的临床靶区），不是BraTS Whole Tumor的坏死/瘤周异常/强化区union。即使取得权限，也不能把二者Dice直接放进当前外部榜。

本审计只读取TCIA官方collection网页及公开646,164-byte metadata manifest。没有申请受控访问，没有下载28.26GB DICOM/REG/RTSTRUCT，没有读取任何影像/轮廓或运行模型。

## 初学者七问式判断

1. **现在的问题：** 当前真正的新外部队列RHUH要等待no-N4 Stage A；需要并行寻找不依赖同一路线的合法人工标签来源。
2. **一句话直觉：** 新考卷不只要有人认真批改，题目也必须和原考试相同；放疗医生画的治疗靶区不能直接当成BraTS肿瘤真值。
3. **为什么看它：** TCIA官方称230例均有术后增强3D T1和手工放疗靶区，规模远大于RHUH40人；旧输入负结果未涉及它。
4. **具体怎么做：** 当前只核验官方病例数、模态、标注者、数据格式、许可和轻量manifest，不碰受控主体。
5. **只改变什么：** 只增加来源证据，不改变模型、输入、标签映射或任何评测。
6. **成功/失败：** 必须同时满足合法匿名访问、T1ce、与WT同语义的专家体素标签和患者独立性。受控访问及标签语义任一失败即停止。
7. **代价：** 网络读取约646KB manifest，0 GPU、0模型forward、0受控数据；主体28.26GB未取。

## 官方证据与机器统计

- TCIA Version 1列出230 subjects、468 studies、1,915 series、78,311 images、28.26GB，格式为DICOM，许可栏为`NIH Controlled Data Access Policy`。
- 官方说明所有患者在手术后接受增强3D-T1与2D T2-FLAIR，并由治疗放射肿瘤科医生手工勾画GTV/CTV；198例GBM，其余32例为AAC/AC/AODG/ODG。
- 公开manifest `GC_manifest_GLIS-RT_20260326.csv`为646,164 bytes，SHA256=`f36372539f3790a712675fa01274e09d6df3ce881f31d009d8367617ec2ace41`。本地只放`/tmp/aflow_glis_rt_manifest_20260326.csv`，未复制进仓库。
- manifest精确1,915 rows/230 participants，全部`File Type=DICOM`、`Study Data Type=Image Annotation,Imaging`；模态计数REG690、RTSTRUCT535、MR460、CT230。每人CT×1、MR×2、REG×3；155人RTSTRUCT×2，MICCAI ABCs子集75人RTSTRUCT×3。

从官方GTV/CTV定义推断，GTV偏向可见肿瘤，CTV还包含安全边界/扩散风险区域；两者都不能在没有逐结构语义验证时冒充WT。这是基于官方用途的语义推断，不是读取轮廓后的实测结论。

## 停止边界与下一决定

关闭“GLIS-RT作为当前同口径WT外部榜”路线。不会请求NIH受控资格、尝试绕过许可、只挑GBM病例下载、把GTV或CTV重命名为WT，亦不会跑Point-only后再决定是否保留。该数据未来可作为独立放疗靶区迁移任务，但那是新的目标与排行榜，需要用户明确扩展任务范围。

一手来源：[TCIA GLIS-RT collection](https://www.cancerimagingarchive.net/collection/glis-rt/)、[TCIA DOI 10.7937/TCIA.T905-ZQ20](https://doi.org/10.7937/TCIA.T905-ZQ20)。
