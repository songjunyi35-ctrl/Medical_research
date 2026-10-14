# Burdenko-GBM-Progression外部来源可行性审计

状态：❌ 标注目标与BraTS Whole Tumor不一致；网页metadata后停止，0主体/模型。

## 为什么值得先看

TCIA Burdenko-GBM-Progression含180位原发胶质母细胞瘤患者，既有放疗规划MRI，也有每人1–8次随访；第一次随访约在放疗结束3周，第二次约2–3个月，治疗阶段比术前外部队列更接近当前post-treatment任务。

## 为什么不能进入当前榜

- RTSTRUCT标注是放疗计划的Gross Tumour Volume（肉眼可见肿瘤，GTV）、Clinical Target Volume（临床靶区，CTV）和Planning Target Volume（计划靶区，PTV），另有正常器官结构。
- 只有部分患者的RTSTRUCT含随访GTV；不是180人完整人工标签队列。
- GTV只表示肉眼可见放疗靶区，CTV/PTV还加入临床/计划边界；它们都不等于本项目包含原标签1/2/3的BraTS Whole Tumor，也不能把三者union冒充WT。
- MRI保留原始采集空间，RTSTRUCT/RTPLAN/RTDOSE对齐topometric CT；官方不提供MRI→CT registration，只给外部GitHub容器建议。即使配准完成，也只解决几何，不解决标签语义。

因此在当前WT Dice/Point榜上没有合法target。不能因为治疗时间点合适就重命名放疗靶区。

## 访问与成本

只读取TCIA公开页面；没有下载131.23GB manifest所指DICOM、29KB临床CSV、RTSTRUCT、影像或GitHub容器。Patient/series/pixel/label/model/GPU均为0；internal test270、UCSD、Africa95、fixed validation、outer、confirmation没有新访问。

## 结论

关闭当前WT路线。这不否定该集合用于GTV/CTV/PTV自动勾画、放疗配准或进展分类；若未来目标明确改成radiotherapy target segmentation，应建立独立任务/榜单，不能进入现有direct-locator WT排行。

一手来源：[TCIA collection](https://www.cancerimagingarchive.net/collection/burdenko-gbm-progression/)、[发布者配准代码](https://github.com/kurmukovai/burdenko_glioma_progression)。页面SHA-256 `ecc577b1704026308013471795e9b49ca4f0271df1f83c809149d942f4b568ce`；机器摘要见[SOURCE_EVIDENCE.json](SOURCE_EVIDENCE.json)。
