# UPENN-GBM逐文件镜像复审：147不等于官方232

状态：❌ metadata-only completeness/source/overlap门；0 release asset/NIfTI/seg/model。

旧审计已证明60份follow-up全部没有corrected segmentation，不能做术后WT。由于术前外部已被WAW/CPTAC证明可有价值，本次只读复查2026年更新的`data-nih/tcia` GitHub release能否合法提供官方232份术前corrected标签。

结果：release API实际4,424 assets、62,874,269,790 bytes，全部带SHA-256；其中630名均有`ce-gd_T1w`，但只有147名有`seg`且均能配对，总计427,673,840 bytes。TCIA官方availability表明确术前611 scans中232份有Corrected Tumor Segmentation，因此镜像少85份；网页又显示4,426 assets，与API实际数差2。不能把“镜像恰好发布的147例”冒充官方完整corrected cohort或在未知缺失机制下称代表性样本。

来源边界同样失败：release正文自称由Pittsburgh Fiber Data Hub整理的TCIA derivatives，不是TCIA官方NIfTI分发。UCSD75直接比较已经证明该provider可把官方float32图变成uint8且数组0/75相同，因此asset SHA只能证明镜像内部完整，不能证明官方parity。UPENN论文还说明173例曾进入旧BraTS challenge；当前没有UPENN↔BraTS2024患者桥。

决定：0资产下载、0模型。只有官方Aspera/NIfTI manifest能冻结完整232并逐member对照镜像、同时患者重叠可在像素前排除，才可重开；不能因147对体积小就选择它们。

证据：[机器摘要](SOURCE_EVIDENCE.json)、[GitHub release API](https://api.github.com/repos/data-nih/tcia/releases/tags/upenn-gbm)、[TCIA UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/)、[官方论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9338035/)、[旧随访审计](../2026-09-05_upenn_source_audit/RESULT.md)。
