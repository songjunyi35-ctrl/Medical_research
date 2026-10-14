# Experiment idea: ReMIND DICOM量化感知几何探针

Status: preregistered after fixed-1e-5 header failure; SEG pixel arrays still unseen

## 1. 现在的问题是什么？

ReMIND-002的T1ce和两个DICOM SEG重复下载内容全部exact，但上一实现把由DICOM文本坐标计算的切片gap要求稳定到`1e-5mm`。192片位置/SOP/Instance都unique、方向字符串完全一致、声明SliceThickness/SpacingBetweenSlices均1mm；原始ImagePositionPatient分量却只保留到约0.001mm，导致派生gap范围`0.999347827–1.000722284mm`、最大中位误差0.000860mm，工程门在任何pixel array前失败。

## 2. 一句话直觉

如果尺子只刻到千分之一毫米，就把真实位置看成一个小区间，而不是拿被四舍五入的数字要求十万分之一毫米精确。

## 3. 为什么选择这个idea？

- 旧实现否定的是固定绝对`1e-5`比较，不是否定192片身份、顺序或DICOM SEG内容；所有下载重复门已通过，SEG像素仍未看。
- DICOM的DS（Decimal String，十进制字符串）是有限字符表示。直接从每个实际字符串的小数位数推导半量化宽度，比在看到0.000860后挑一个0.001容差更可复现。
- 新门不改变模型或标签，只要求“声明的1mm能否落在原始文本精度允许的真实距离区间”，并继续以Referenced SOP UID作为frame所有权主键。
- 不修改已关闭的NRRD或fixed-1e-5 probe；新记录显式引用其失败和已发生的数据访问。

## 4. 具体怎么做？

复用已冻结下载收据，不新增网络。对每个T1ce ImagePositionPatient原始DS分量，按其小数位数d设半量化宽度`0.5×10^-d`。相邻两点的位移误差上界是三个坐标半宽相加后的L2 norm；真实欧氏gap必在`[nominal−bound, nominal+bound]`。要求191个区间都包含各切片共同声明的1mm，且下界严格>0；orientation原始字符串必须192片exact。

之后才读取两个SEG。每帧Referenced SOP UID必须唯一属于T1ce；SEG与对应源slice的每个ImagePositionPatient DS区间必须相交，PixelSpacing DS区间也必须相交。用SOP UID把1-bit frame放回192-slice数组，未引用slice补0，无插值。最后检查两个mask二值/非空及manual tumor完全落在automatic cerebrum中；影像只记录格式，不选择预处理。

## 5. 唯一变化与固定项

- 唯一变化：几何门从对派生浮点gap固定`1e-5mm`，改为由每个DICOM DS文本自身小数精度决定的区间验证。
- 固定不变：ReMIND-002、三个series、重复SHA、pydicom3.0.1、SOP映射、无重采样、标签/包含门、0模型。没有手填0.001或从标签/分数选择容差。

## 6. 成功、失败和停止规则

- Primary门：191个source gap区间全部包含声明1mm且下界>0；orientation字符串exact；两个SEG身份/二进制合同；每frame唯一SOP、position/spacing区间相交；两个mask非空，tumor⊂cerebrum。
- Promotion gate：全部通过才称首例源/标签原生对齐，并只允许另立input bridge；不自动下载43例。
- Failure/stop rule：任一门失败即停止，不扩大区间、舍弃frame、换病例/series/decoder或用旧NRRD补洞。
- 是否允许test：不允许模型/指标。本阶段只解码固定首例。

## 7. 成本与风险

- 新网络/磁盘：0，复用已下载数据和冻结临时pydicom。
- CPU：秒级；GPU/模型forward：0。
- 最大风险：DICOM SEG可能缺少预注册的DerivationImage/PlanePosition结构，或manual tumor不完全在automatic cerebrum；任一均失败。即使源通过，raw head MRI仍需独立去颅骨input bridge，且ReMIND manual whole tumor与BraTS多模态WT有语义caveat。

## 8. 最终结果（完成后填写）

- Validation/Test/CI：无模型指标。
- 是否支持直觉：待量化区间和SEG frame检查。
- 下一步：只按工程门决定是否保留ReMIND source，不访问模型。
- 报告/哈希：待填写。
