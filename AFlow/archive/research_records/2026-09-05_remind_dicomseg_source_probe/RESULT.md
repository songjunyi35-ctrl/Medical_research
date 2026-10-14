# ReMIND-002 DICOM SEG探针：源切片间隔exact门失败

状态：当前DICOM SEG来源探针在SEG pixel decode前关闭。五个NBIA下载的重复内容门全部通过；读取T1ce DICOM header后，源切片投影间隔没有满足冻结的`1e-5mm`稳定门。没有访问T1ce `pixel_array`、DICOM SEG frames/标签数组、模型或指标。

## 下载证据

- 新T1ce ZIP 8,896,341 bytes；192个`.dcm`的文件名/bytes/SHA清单与旧probe保留副本192/192 exact。DICOM bytes总和21,660,158，与官方series metadata exact。
- Manual tumor SEG两次ZIP各5,340 bytes，实际单DICOM内容SHA/bytes重复exact；官方series bytes17,148。
- Automatic cerebrum SEG两次ZIP各792,967 bytes，实际单DICOM内容SHA/bytes重复exact；官方series bytes4,834,884。
- 每ZIP额外`LICENSE`按预注册单列，不计实例或series bytes。未使用失败NRRD内容补洞。

## 失败边界

T1ce header为192个unique ImagePositionPatient、192个unique SOPInstanceUID、InstanceNumber 1–192 exact，所有切片ImageOrientationPatient完全一致；SliceThickness与SpacingBetweenSlices均声明1mm。按固定orientation法向投影并排序后，位置严格单调，但191个gap分布在`0.999347827–1.000722284mm`，中位`0.999861846mm`，最大中位偏差`0.000860437mm`，超过实现冻结`1e-5mm`稳定门。

这很可能与DICOM Decimal String中方向/位置有限小数精度有关，但该解释是在header后得出，不能据此事后把容差改成0.001mm。PREREG明确失败后不换decoder/病例/series/容差，因此没有继续读取任何`pixel_array`或SEG frame mapping。

## 解释

“TCIA DICOM交付可重复”直觉得到支持；“当前exact几何实现全门通过”不成立。失败不说明DICOM SEG与T1ce一定错位，也不否定ReMIND43一般来源；它只说明预注册的过严位置间隔实现无法在不调门的情况下完成源探针。此轮不再围绕同一ReMIND-002调几何容差或换首例。

## 成本和访问

- 4项DICOM SEG合同+前一来源probe5项合同；临时pydicom3.0.1不安装，wheel/290文件manifest均冻结。
- 网络新下载约11.28MB ZIP；0 GPU、0checkpoint、0model forward、0Dice/Point。
- DICOM header access=true；T1ce pixel array=false、SEG pixel array=false、old NRRD array=false、其他42例像素=false。
- 临时数据保留`/tmp/aflow-remind-dicomseg-probe-20260905`，未删除。

证据：[`PROBE_PREACCESS.json`](PROBE_PREACCESS.json)、[`DOWNLOAD_RECEIPT.json`](DOWNLOAD_RECEIPT.json)、[`IMPLEMENTATION.json`](IMPLEMENTATION.json)、[`PREREG.md`](PREREG.md)、[`AMENDMENT_01.md`](AMENDMENT_01.md)。
