# UPENN随访来源审计：60例均缺发布的体素分割，停止术后评测路线

状态：2026-09-05 source-only audit完成；未下载NIfTI/分割、未运行模型、未分配新的评测角色。

官方临床/可用性表与论文一致包含671 scans/630 patients：611次术前、60次随访，随访中41人有配对基线、19人只有随访。但关键的`UPENN-GBM_data_availability.csv`明确显示：

| 60次随访的数据项 | available | not available |
|---|---:|---:|
| Structural imaging | 60 | 0 |
| Automatic Tumor Segmentation | 0 | 60 |
| Corrected Tumor Segmentation | 0 | 60 |

因此不能把论文所述60次再次手术前随访当成60例带WT评分真值的外部队列。此结论只涉及当前公开发布可用性，不能推断作者私下没有标注或将来不可能发布。

611次术前有232例明确列为提供Corrected Tumor Segmentation、379例该项not available；这些术前数据不能混入术后主榜。论文对自动分割人工QC的总体描述也不能代替每个时间点的实际文件可用性。当前不把术前或无标签随访改名为术后带标签证据。

成本仅四份官方小CSV共396,335 bytes，无GPU/推理/大缓存。患者ID、时间点与标签可用性已核对；`radiology_mapping.csv`为病理图像与放射影像映射，没有提供本任务需要的BraTS2024患者桥接。尚未进行NIfTI目录遍历，因为标签可用性门已失败。

下一决定：关闭本次UPENN术后分割评测可行性路线，不下载无标签扫描、不运行模型、不建榜。以后只有新的官方专家分割发布，才重新审计身份/标签/文件哈希并预注册；不把data-nih等镜像当作新增真值来源。

证据：[元数据收据](METADATA_RECEIPT.json)、[官方可用性表归档](source_metadata/UPENN-GBM_data_availability.csv)，SHA256 `7a4f0eebfa34c86a3380f2695457007936b6e24e62960b68f10fde18a726dd89`。

来源：[TCIA collection](https://www.cancerimagingarchive.net/collection/upenn-gbm/)、[官方可用性表](https://www.cancerimagingarchive.net/wp-content/uploads/UPENN-GBM_data_availability.csv)、[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9338035/)。
