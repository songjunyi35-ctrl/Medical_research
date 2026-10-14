# UTSW-Glioma 官方来源审计计划

冻结时间：2026-09-05，首次正式 Faspex 目录访问之前。

## 1. 现在的问题是什么？

当前模型缺少一份来源、标签和患者角色都清楚的新外部证据。UCSD75 的旧正式结果只适用于 Pittsburgh Fiber Data Hub 的 uint8 镜像；RHUH80 必须等待 no-N4 的 E42/F42 门和 Full972；ReMIND43 因标签与影像物理网格不能按预注册门严谨对应而关闭。

## 2. 一句话直觉是什么？

先找到一叠“来自另一所医院、而且每张答案都由人复核过的试卷”，确认试卷和答案一一对应，再决定以后是否让模型作答。

## 3. 为什么现在选它？

TCIA 官方页面和同行评议数据论文称 UTSW-Glioma 有625名术前患者，其中362例分割经过人工细化及神经放射科复核；官方 TSV 又给出逐病例 `Manually Refined Segmentation` 字段。UT Southwestern 不属于 BraTS 2024 post-treatment 的七个贡献机构。既往 N4、RHUH 与 ReMIND 负结果没有检查或否定这个新来源。术前与治疗后的域差异很大，因此它未来只能是单独的跨阶段外推队列，不能混入现有 test270 排名。

一手来源：

- https://www.cancerimagingarchive.net/collection/utsw-glioma/
- https://doi.org/10.1038/s41597-026-07274-4
- https://doi.org/10.7937/DFAE-1B86

## 4. 具体怎么做？

输入是 TCIA 官方网页、官方96KB TSV 和 Faspex package1265 的目录元数据。审计器先核对网页中的包 ID 和许可，再统计625行及362个 `Yes`，随后只浏览官方目录，检查这些病例是否各有 post-contrast T1 (`T1GD`) 与 segmentation 文件。没有模型、teacher、loss、融合或预测；输出只是一份来源可行性 JSON/Markdown。

## 5. 只改变什么，哪些不变？

唯一新增因素是“一份候选外部数据来源的来源证据”。当前 no-N4/Otsu 训练、E/F 划分、checkpoint、阈值、Flip8、bootstrap、旧 validation/outer/confirmation/test270/UCSD75/RHUH 均不改变也不访问。

## 6. 怎样算成功/失败？

- 成功：官方 package ID=1265，CC BY 4.0；TSV恰有625个唯一 Subject ID、362个明确人工细化 `Yes`；官方目录完整；362例人工子集能分别唯一匹配所需 T1GD 与 segmentation；若官方提供 checksum 清单则只允许下载该小清单并冻结哈希。
- 失败：包身份、许可、逐病例人工标志、目录完整性或一一配对任一不成立，立即停止。
- 即使本审计成功，也只代表“来源可进入后续输入桥接研究”，不授权下载22.9GB像素、不授权模型评测。

## 7. 代价是什么？

预计数分钟 CPU、约百 KB TSV/网页和目录 JSON、0 GPU、0模型 forward、0部署复杂度。本步不下载影像或标签体素。

## 冻结访问标志

- image_voxels_accessed: false
- label_voxels_accessed: false
- model_inference: false
- old_cohort_accessed: false
- package_payload_download_authorized: false

