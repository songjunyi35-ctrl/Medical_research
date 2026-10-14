# CPTAC-GBM / BraTS2021公开标签来源审计结果

状态：✅ public labeled Training33冻结并完成后续确认；原CPTAC39合并方案因Validation无seg失败并保留。

## 来源结构

TCIA BraTS2021官方映射表1,486行中，CPTAC-GBM有39位唯一PatientID/BraTS ID：Segmentation Training33、Validation6。Package636目录与映射39/39一致。

原计划要求39人都有T1CE+seg；固定首个Validation病例`BraTS2021_00447`实测只有flair/t1/t1ce/t2，0 seg，故原39方案按门失败、无payload。随后另行冻结只使用public Training33，并验证：

- Training33：33/33各5个正大小NIfTI，唯一`t1ce`和`seg`；
- Validation6：6/6各4影像、0 seg，永久排除；
- 选中66文件78,013,574 bytes/副本；
- Hidden challenge test未使用。

选择33完全由公开标签可用性决定，发生在患者payload/模型前，不按结果删病例。证据身份是`BraTS2021 public Training labels used external to this project`，不是原challenge validation/test或hidden blind。

## 完整性与身份

Package根`.sums`为49,850,968 bytes、407,245条MD5；选中66路径全部唯一覆盖且无空文件hash。双份官方下载后，66/66同时命中官方目录size、官方MD5和跨副本SHA。

Mapping把BraTS ID连到CPTAC PatientID；CPTAC/CHOP-UPenn来源不在本项目BraTS2024七个贡献站，但没有CPTAC↔BraTS2024 patient bridge，所以只称source/institution-disjoint no-known overlap，不声称ID绝对证明patient-disjoint。

## 标签/后续

首例和全33工程链确认raw标签union=`{0,1,2,4}`，符合BraTS2021的NCR/ED/ET，术前WT取all positive。后续Africa-informed冻结模型确认已按独立预注册运行一次，见[结果](../../research_records/2026-09-06_cptac33_africa_informed_confirmation/RESULT.md)。

## 证据

- [原计划](PLAN.md)、[39方案失败](EXECUTION_01_FAILURE.json)、[Training33修订](AMENDMENT_01_TRAINING33.md)
- [官方33 inventory](OFFICIAL_TRAINING33_INVENTORY.json)，SHA `228785f0434037295a2e65a0d1a031419b074aff918fae0bc8cbbd10e8395418`
- [冻结33 cohort](FROZEN_TRAINING33_COHORT.json)，SHA `2e1391cfd110721788aa2ddaaa4f34827eaab49b9c354d775c80651afc292208`
- [官方mapping](source_metadata/BraTS2021_MappingToTCIA.xlsx)，SHA `223243c5821c282e90da980ba38d860ab2d2a597d1f4f1800ebe227329a2dc0b`
- [TCIA BraTS2021 analysis](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/)、[CPTAC-GBM](https://www.cancerimagingarchive.net/collection/cptac-gbm/)
