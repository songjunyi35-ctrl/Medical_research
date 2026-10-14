# ReMIND公开外部来源审计：43例同T1ce人工肿瘤候选，NRRD首探针传输门失败

状态：来源metadata可行、当前NRRD probe实现失败；DICOM SEG替代入口尚未执行。不是模型结果，不进入Leaderboard。

## 候选队列

TCIA ReMIND Version 1为CC BY 4.0，114名在Brigham and Women’s Hospital接受影像引导切除的连续患者；该机构不在BraTS2024公开列出的七个贡献站。官方一手论文报告92 gliomas/11 metastases/11 other，manual tumor由neurosurgical fellow勾画、attending neurosurgeon复核，增强T1的10例双标注子集中interobserver Dice `92.2%±2.8%`。

官方NBIA API给出1,346 series/114 patients：MR670、SEG356、US320。Preop有105名患者各唯一一份`T1_postcontrast`，共20,416 DICOM instances、3,803,222,860 bytes。官方NRRD `.sums`列出356路径，其中113人/127份manual preop tumor，67人有直接参考T1 postcontrast的唯一tumor NRRD；再按像素前官方Histopathology精确保留Glioblastoma/Astrocytoma/Oligodendroglioma，形成43人：19/15/9，35增强、8不强化。34/43还有同T1ce automatic cerebrum mask。

患者身份只能达到“不同公开机构与namespace、无ID桥”的证据；正式使用前仍需像素级近重复审计，不能称密码学证明零重叠。

## 官方inventory异常

Aspera package755详情声明357 files；目录浏览只返回315 NRRD及100号以前目录，而官方`.sums`提供356/356路径，额外41条正是100–114号，说明浏览API不完整。`.sums`文件SHA256=`6daa7391…`，但356/356 MD5全部是空文件值`d41d8cd…`，与非零远端size矛盾；因此它可冻结路径，不能作为内容hash。

## 固定首例probe

预先固定43例排序首位ReMIND-002，不按图像/标签/模型结果挑例。NBIA T1ce ZIP下载8,896,341 bytes；含192个DICOM和一个2,786-byte LICENSE，192 DICOM未压缩总量21,660,158与官方FileSize exact。实现误把LICENSE算成第193实例和数据bytes，产生一个可定位的假门失败。

两份NRRD首轮官方Aspera传输成功：manual tumor 4,747 bytes/SHA256 `2fb1bf3d…`，automatic cerebrum 1,947,370 bytes/SHA256 `63ddccbd…`，均匹配浏览size。预注册要求第二次独立传输来弥补发布者无有效hash；第二轮在文件前报`Unable to connect via SSH`，两文件缺失，故双传输门真实失败。没有重试、换病例或继续数组QC。

## 访问与下一决定

- 已下载但未解码：ReMIND-002一条DICOM series、两份首轮NRRD；segmentation voxel access=false。
- 其余42例无像素；0 checkpoint、0 GPU、0模型forward、0Dice/Point；所有旧cohort无新增访问。
- 临时数据保留于`/tmp/aflow-remind-probe-20260905`，可按官方源重建，未删除。
- ReMIND同一TCIA collection为ReMIND-002另有T1ce引用的DICOM SEG（tumor17,148 bytes、cerebrum4,834,884 bytes）。该独立路线随后以临时pydicom3.0.1和新预注册执行：五次NBIA下载内容复现全过，但T1ce header投影gap相对中位最大偏差`0.000860mm`，超过冻结`1e-5mm`稳定门，故在任何pixel array/SEG frame前关闭且不调容差。证据见[DICOM SEG结果](../../research_records/2026-09-05_remind_dicomseg_source_probe/RESULT.md)。

后续又在SEG pixel前检验量化感知物理grid：source191个interval全过，但tumor/cerebrum内部frame gap相对声明厚度的最大层残差分别`5.79e-6/2.42e-5`，仍超由DS文本末位导出的严格区间；nearest映射未执行。当前本轮不再围绕ReMIND-002换容差/病例/decoder。43例metadata价值保留，但输入与标签数组链未过门，不能评模型。

证据：[官方metadata机器报告](REMIND_PUBLIC_SOURCE.json)、[官方sums收据](REMIND_SUMS_DOWNLOAD.json)、[checksum发现](CHECKSUM_FINDING.json)、[43例清单](../../research_records/2026-09-05_remind_t1ce_source_probe/COHORT_CANDIDATES.json)、[NRRD probe结果](../../research_records/2026-09-05_remind_t1ce_source_probe/RESULT.md)、[DICOM SEG probe结果](../../research_records/2026-09-05_remind_dicomseg_source_probe/RESULT.md)、[量化/物理grid结果](../../research_records/2026-09-05_remind_physical_grid_probe/RESULT.md)。

一手来源：[TCIA ReMIND](https://www.cancerimagingarchive.net/collection/remind/)、[Scientific Data论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11093985/)、[BraTS2024来源机构](https://arxiv.org/html/2405.18368v1)。
