# UTSW-Glioma 来源审计 Amendment 01

冻结时间：2026-09-05，首次完整 iteration-token 遍历之前。

## 为什么需要修订

初版 `SOURCE_AUDIT.json` 必须保留为实现失败证据。它错误地把 Faspex `/page` 的100项响应当成完整目录，并把影像名预设为 T1GD、标签预设为唯一 segmentation。实际固定探针显示：

- IBM 官方文档和 TCIA 当前前端均指定：下一页 token 位于响应头 `x-aspera-next-iteration-token`，下一请求通过 query `iteration_token` 传回；第二页已从 BT0169 连续开始且仍有 token。
- 官方文件名是 `brain_t1ce.nii.gz`，不是 T1GD。
- TSV 首个 `Yes` 病例 BT0001 有11文件，其中两份人工标签为 `rtumorseg_manual_correction.nii.gz` 与 `tumorseg_manual_correction.nii.gz`；首个 `No` 病例 BT0008 恰有9文件。
- `362×11 + 263×9 = 6349`，精确等于 package1265 自报总文件数。

这属于分页与文件名解析实现错误，不是对来源资格门的放宽，也没有访问任何像素。

## 修订后的单因素和固定门

唯一变化是目录解析实现：按官方 token 翻页，并按已公开文件名做精确集合核对。数据来源、625/362 cohort、许可、0像素/0模型访问均不变。

通过门：

1. 根目录唯一为 `/UTSW-Glioma`；完整翻页后目录集合与官方 TSV 的625个 Subject ID 完全相等，无重复 token/路径。
2. package 自报 `total_files=6349`，逐病例清单总数也必须6349。
3. 263个 `No` 病例文件集合必须精确等于固定9文件；362个 `Yes` 病例必须精确等于这9文件加两份 manual-correction 标签。
4. 每个目标都必须是指向文件的 symbolic link，大小为正；所有响应页完整结束。

失败门：任一数量、ID、文件名、类型、大小、分页连续性不符即失败。

即使通过，仍不选择两份人工标签中的哪一个，也不授权像素下载或模型评测。下一步若有必要，必须另行冻结固定病例与几何/标签语义桥接。

## 访问与成本

- image_voxels_accessed: false
- label_voxels_accessed: false
- model_inference: false
- package_payload_downloaded: false
- 预计约625次小型目录 API 请求，CPU/网络元数据数分钟，0 GPU。

一手分页依据：IBM Aspera Faspex 5.0.8 release notes（iteration_token）及 TCIA 当前 Faspex 前端 `x-aspera-next-iteration-token` 实现。

