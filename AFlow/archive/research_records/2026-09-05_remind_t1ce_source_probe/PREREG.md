# ReMIND-002 T1ce/seg来源探针冻结协议

状态：2026-09-05 UTC，在任何ReMIND DICOM/NRRD内容或模型访问前冻结。

## 冻结来源

- `REMIND_PUBLIC_SOURCE.json` SHA256=`b2f281023dda7ba2fc95767a00f7af67e66a945f158e8f5b1c9bb9ccce2f2b9a`；TCIA series API 1,346 rows/114 patients，全CC BY 4.0。
- 官方`.sums` SHA256=`6daa7391546fca3d4a9bc985241c8f499726f9b3116617efcc89d53f2ece5116`，356/356 NRRD MD5；浏览API漏41条但无browse-only条。
- 临床表SHA256=`5c416ec4a4aa40247feee993af8af017e73ec3455fd05bab8c30ee03627343e8`。
- 候选定义在像素前固定：Histopathology精确属于`Glioblastoma/Astrocytoma/Oligodendroglioma`，存在唯一preop MR series description含`T1_postcontrast`，存在唯一直接参考T1_postcontrast的`preop-SEG-tumor` NRRD。预期43人；不按grade/enhancement/volume筛选。
- Probe固定候选排序首例`ReMIND-002`，不因结果换例。

## 固定资产

- T1ce series、manual tumor和automatic cerebrum的UID/path/MD5必须由生成的`COHORT_CANDIDATES.json`给出，并在下载前写`PROBE_PREACCESS.json`。
- NBIA只请求该单个SeriesInstanceUID；DICOM实例数/PatientID/StudyUID/SeriesUID全部核对。
- Aspera package755只请求两份冻结NRRD；不得请求其他patient/segmentation或整个包。官方`.sums` MD5和transfer bytes均须通过。

## 工程门

- 使用SimpleITK读取DICOM series与NRRD；不做重采样。三者必须finite 3D、正spacing。
- Manual tumor允许值固定为二值`{0,1}`（整数容差1e-5）、非空；cerebrum同样二值非空，且tumor必须是cerebrum子集。
- T1ce与manual tumor的size、spacing、origin、direction逐项差≤1e-5；cerebrum同样比较。记录物理8角点最大差。
- 影像只记录dtype/min/max/negative/zero/positive、六面众数占比、非零bbox和cerebrum外非零比例；不据此选择阈值/形态学/N4/brain mask。
- 全门通过只授权后续另立输入桥接，不授权其余42例像素、模型或评测。失败不换病例、不调容差、不以label做crop。

## 访问与成本

- 本阶段ReMIND图像1 series、manual label1、automatic brain mask1；0 checkpoint、0 GPU、0 model forward、0 Dice/Point。
- 数据置`/tmp/aflow-remind-probe-20260905`；预期<100MB，墙钟≤20分钟。路径已存在则拒绝覆盖。
