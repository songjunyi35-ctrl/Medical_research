# ReMIND-002 DICOM SEG来源探针冻结协议

状态：2026-09-05 UTC。NRRD probe已下载首例文件但未解码并因双传输门关闭；本协议在任何DICOM SEG下载、任何ReMIND数组解码或模型访问前冻结。

## 冻结输入与依赖

- 43例候选SHA256=`8118827aa001dead98cc8cc1f6ee5a2fa9b797a6a8797329ba22f4766d669536`；旧probe结果/下载收据必须引用，不抹去既有像素下载。
- 首例固定ReMIND-002，三个NBIA series：
  - T1ce=`1.3.6.1.4.1.14519.5.2.1.309430968157051488200208000420039817667`，Study UID=`1.3.6.1.4.1.14519.5.2.1.322752919633264035254585915433270238948`，192 instances，21,660,158 bytes。
  - manual tumor SEG=`1.3.6.1.4.1.14519.5.2.1.80716149485382078399551532220853537216`，1 instance，17,148 bytes，description=`tumor seg - MR ref: 3D_AX_T1_postcontrast`。
  - automatic cerebrum SEG=`1.3.6.1.4.1.14519.5.2.1.323423363124946855158015352096264740086`，1 instance，4,834,884 bytes。
- pydicom wheel path固定`/tmp/aflow-remind-dicomseg-deps/pydicom-3.0.1-py3-none-any.whl`，SHA256=`db32f78b2641bd7972096b8289111ddab01fb221610de8d7afa835eb938adb41`，不安装到venv。

## 下载合同

1. 新工作目录`/tmp/aflow-remind-dicomseg-probe-20260905`必须不存在。
2. 三个series均只用TCIA NBIA `getImage?SeriesInstanceUID=`。T1ce下载一次，与旧probe ZIP中192个`.dcm`按文件名+SHA exact；两个SEG各下载两次到隔离目录，真正`.dcm`的bytes/SHA必须重复exact。
3. ZIP内`LICENSE`只作为许可附件单列，不计DICOM实例或series bytes；其他非`.dcm`成员失败。所有`.dcm`未压缩bytes总和须匹配官方FileSize。
4. HTTPS返回、ZIP SHA、每个DICOM SHA和本地路径保存；不访问其他series/患者。

## DICOM/SEG解析合同

1. 所有T1ce实例PatientID/StudyUID/SeriesUID/SOPInstanceUID核对，SOP必须192 unique。按ImagePositionPatient沿ImageOrientationPatient法向投影排序，位置必须严格单调且间隔稳定；不用文件名顺序。
2. 两个SEG必须Modality=SEG、Study/Patient正确、SeriesUID/SeriesDescription正确、SegmentationType=BINARY、BitsAllocated=1、SegmentSequence恰一项。
3. 每个frame只允许ReferencedSegmentNumber=1；DerivationImageSequence→SourceImageSequence必须恰引用一个T1ce SOP。每个SOP最多一frame；未引用slice补0。Rows/Columns必须等于T1ce。
4. pydicom `pixel_array`必须finite、shape=`[frames,rows,cols]`且值域`{0,1}`。按SOP映射构造192×rows×cols mask；tumor/cerebrum都非空，tumor必须为cerebrum子集。
5. 同时从PlanePositionSequence的ImagePositionPatient核对每frame与被引用T1ce位置差≤1e-5mm；PixelMeasures spacing与T1ce行列spacing差≤1e-5。任何缺字段/多重引用/压缩decoder需求/不确定映射均失败，不猜。
6. 影像数组仅在所有下载门后读取；记录finite/min/max/背景/nonzero/cerebrum外非零比例，不做预处理选择。

## 访问与停止

- 当前授权仅ReMIND-002三series工程probe；0模型/0Dice/Point。不打开旧NRRD数组，不使用其内容补DICOM SEG。
- 全门通过也只允许新input-bridge计划；不自动下载43例。失败不换decoder版本/病例/series或容差。
- 时间≤20分钟，新增临时数据<100MB；超限停止。
