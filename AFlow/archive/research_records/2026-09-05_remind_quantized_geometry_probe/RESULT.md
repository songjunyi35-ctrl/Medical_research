# ReMIND量化感知几何：source gap门通过，SOP映射假设失败

状态：当前量化感知SOP-frame实现关闭。191个source gap区间全部通过后首次读取了T1ce pixel arrays；两个DICOM SEG在pixel array访问前因Rows/Columns与源图不同而停止。0模型/0指标，其他42例无像素。

## 支持的机制

把DICOM Decimal String的每个坐标按实际小数位转换为半量化区间后，191个相邻切片真实距离区间全部包含共同声明的1mm且下界>0。说明前一固定`1e-5mm`派生gap失败主要是有限文本精度造成，量化区间直觉成立；没有手填0.001容差或用标签/模型调门。

## 新失败边界

源T1ce是192 slices，每slice Rows×Columns=`256×216`。Manual tumor DICOM SEG header为36 frames、Rows×Columns=`49×32`、PixelSpacing约`0.9765625×0.9765626mm`、SliceThickness1mm；automatic cerebrum为275 frames、`415×335`、PixelSpacing`0.4×0.4mm`、SliceThickness`0.3995816mm`。

两个SEG的PerFrameFunctionalGroups只有FrameContent、PlanePosition和SegmentIdentification，没有冻结实现要求的DerivationImageSequence/SourceImageSequence；顶层也无ReferencedSeriesSequence。它们是带独立物理坐标的裁剪/不同分辨率grid，不能用“每frame引用一个源SOP，Rows/Columns exact”无重采样拼回T1ce。实现先检查Rows/Columns，故两个SEG的`pixel_array`都未访问。

## 解释

“按DS量化验证源slice”得到支持，但“DICOM SEG可按SOP无插值映射”不成立。一般ReMIND来源仍未被否定；标准的下一工程路径应读取SEG的PlanePosition/Orientation/PixelMeasures，构造其物理grid，再以nearest-neighbor显式重采样到T1ce grid。这属于新的唯一因素和rasterization caveat，必须另行预注册，不能修改本实现或用旧NRRD补洞。

## 成本与访问

- 3项量化区间合同；0网络新增、0GPU、0checkpoint、0forward。
- T1ce `pixel_array`访问=true；SEG header=true、SEG `pixel_array`=false；旧NRRD array=false。
- 临时数据与两个失败probe完整保留；不改已冻结代码/门。

证据：[`IDEA.md`](IDEA.md)、[`PREREG.md`](PREREG.md)、[`IMPLEMENTATION.json`](IMPLEMENTATION.json)、前一[DICOM SEG下载收据](../2026-09-05_remind_dicomseg_source_probe/DOWNLOAD_RECEIPT.json)。
