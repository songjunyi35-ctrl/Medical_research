# Post-treatment T1ce/seg外部cohort可得性审计

> 2026-09-02后续更新：本审计最初记录的Aspera整包/磁盘阻塞已由TCIA官方`data-nih/tcia` GitHub逐文件release assets解除。主release中184对T1-post/seg只需选择性下载；官方participants映射揭示109例已暴露，按患者排除后得到75 scans/53 subjects零重叠队列，并已按预注册完成一次性外部确认。最终证据见[`archive/research_records/2026-09-02_ucsd_ptgbm_external_confirmation/RESULT.md`](../../research_records/2026-09-02_ucsd_ptgbm_external_confirmation/RESULT.md)。以下保留发现新入口之前的审计快照与当时判断。

> MU-Glioma-Post也完成元数据复审：没有官方逐文件GitHub release；临床表出现597个非空timepoint值而官方影像口径为596，且无跨库ID bridge；11GB包与自动模型融合标签均不适合当前正式确认。未下载NIfTI/seg、未评模型，见[`archive/reviews/2026-09-02_mu_glioma_post_followup/RESULT.md`](../2026-09-02_mu_glioma_post_followup/RESULT.md)。

> RHUH-GBM是更强的下一候选（40人、后两时点可组成80例、专家人工修正、2.9GB），但TCIA Faspex5.0.15 public JWT scope兼容错误阻止官方inventory；无官方hash时不采用第三方镜像。未下载NIfTI/seg、未评模型，见[`archive/reviews/2026-09-02_rhuh_gbm_external_feasibility/RESULT.md`](../2026-09-02_rhuh_gbm_external_feasibility/RESULT.md)。

> 其他高质量候选已排边界：BraTS 2024一手论文明确列出UCSF约600例来自既有UCSF-LPTDG/ALPTDG，因此它是当前BraTS源cohort而非新external；LUMIERE提供专家RANO临床评价而非同口径四区voxel segmentation。两者均不进入当前T1ce/WT外部榜。

日期：2026-09-02 UTC

## 结论

找到两个官方TCIA治疗后胶质瘤候选，但当前尚不能执行或宣称外部确认。

首选是UCSD-PTGBM主集：官方TCIA页面列出136 subjects/184 studies的NIfTI images+segmentations整包33.1GB；论文说明BraTS式四区分割先自动生成，再由trained radiologists人工修正并由board-certified neuroradiologist反复复核批准。TCIA另行列出42 subjects/59 studies、11.88GB的“BraTS-GLI 2024 Test Data”，这部分必须排除，不能当新外部证据。

次选MU-Glioma-Post：203 subjects、596 post-treatment timepoints、约11GB并含四区segmentation，但论文/官方摘要明确其分割以nnU-Net自动流程为基础。即使有label refinement，它仍可能与BraTS/FL-PoST工具链共享模型偏倚，证据等级低于UCSD人工批准主集。

## 为什么当前不能直接下载/评测

- Workspace当前可用约14GB；UCSD主集整包33.1GB，下载后还需要T1ce/SEG抽取与约数GB cache，无法安全容纳。
- `/mnt/hdfs/sjy`容量充足，但当前会话权限只读；没有授权写入或删除现有material data。
- TCIA官方页面使用IBM Aspera Faspex package1009/1019式外链（UCSD为package id1019），公开页面要求Aspera客户端/OAuth public-link流程。只读检查前端确认API prefix`/aspera/faspex/api/v5`，但未获得可匿名列出并只选T1ce/SEG的子文件清单接口。
- UCSD主集与BraTS test包在TCIA页面被分开列出，支持“主集可能独立”，但当前BraTS病例ID已匿名，仍必须下载manifest后做patient/timepoint/文件hash重叠审计，不能先验宣称零重叠。
- MU与当前BraTS 2024/FL-PoST来源关系未得到足够patient-level证据，不能只凭不同collection name称外部。

## 冻结的外部确认建议

若获得存储/下载授权，优先做UCSD主集而不是BraTS test子包：

1. 只下载官方“Images and Segmentations”136-subject主集；明确排除“BraTS-GLI 2024 Test Data”42-subject包。
2. 保存TCIA版本、DOI、package/manifest、license、文件size和SHA-256；不混版本2/3。
3. 在任何模型运行前，以subject/timepoint ID、NIfTI header/shape和内容hash审计与当前972 train、fixed validation108、reused test270零重叠；无法证明则标记potential overlap而非external。
4. 只抽取post-contrast T1和BraTS-style segmentation；将WT固定定义为labels`{1,2,3}`、排除resection cavity label4，与当前项目一致。
5. 写独立preprocessing entry point，把256³ MNI数据按image-only nonzero brain crop→true-128³处理；不使用GT定位，不用外部结果选threshold/epoch/postprocess。
6. 先冻结当前推荐checkpoint、raw-logit threshold0、1×forward、无TTA/SAM/postprocess与bootstrap规则；外部集只作一次confirmation报告。
7. 在Leaderboard中建立独立External UCSD-PTGBM section，绝不与reused internal test270混排；报告grid/atlas与标签审核差异。

## 资源估算与授权需求

- 原始主集33.1GB；只保留T1ce+SEG后预计明显更小，但在看到包清单前不能可靠承诺大小。
- 128³ cache粗估：184 timepoints约1.8GB（float32 image+uint8 label，不含几何/临时文件）。
- 需要至少约40GB安全临时空间，或HDFS可写目录+流式筛选；不得静默删除现有code/checkpoints/reports/cache。
- 需要用户授权其一：可恢复地清理已hash归档且raw-rebuildable的派生cache，或允许写入HDFS/更大挂载并安装/使用Aspera客户端。

## 一手来源

- [TCIA UCSD-PTGBM collection](https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/)
- [UCSD-PTGBM Scientific Data paper](https://www.nature.com/articles/s41597-025-06499-z)
- [TCIA MU-Glioma-Post collection](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)
- [MU-Glioma-Post Scientific Data paper](https://www.nature.com/articles/s41597-025-06011-7)
- [BraTS 2024 post-treatment challenge paper](https://arxiv.org/abs/2405.18368)

## Access声明

本审计只读取公开网页/论文与本地磁盘状态；没有下载影像、创建cache、访问任何新标签cohort或运行模型。Internal train-only inner/fixed validation/test的既有访问状态未改变。
