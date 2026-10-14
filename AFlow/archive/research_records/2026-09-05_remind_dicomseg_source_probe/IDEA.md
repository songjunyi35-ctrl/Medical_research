# Experiment idea: ReMIND固定首例DICOM SEG来源探针

Status: preregistered after NRRD transport failure; no DICOM arrays or model inference yet

## 1. 现在的问题是什么？

ReMIND官方metadata已在任何模型前固定43例T1ce-referenced manual tumor弥漫性胶质瘤候选，但首个NRRD probe因第二轮Aspera SSH失败而在数组QC前关闭。固定首例ReMIND-002同一TCIA collection还提供直接引用其T1 postcontrast的DICOM SEG：manual tumor17,148 bytes、automatic cerebrum4,834,884 bytes。需要验证这个独立官方交付格式能否给出原生对齐mask；已发生的ReMIND-002 DICOM/NRRD下载必须显式保留，不能假装新盲。

## 2. 一句话直觉

既然单独寄来的答案包第二次快递失败，就改查医院把“图像和答案装在同一个DICOM盒子”里的官方副本，但题目和首例都不换。

## 3. 为什么选择这个idea？

- NRRD失败发生在第二轮SSH连接、文件前；它没有否定DICOM SEG内容或几何。
- ReMIND一手论文说明DICOM SEG由NRRD通过DCMQI转换，引用对应MR DICOM，并在技术验证中确认影像DICOM往返值保持；这是同发布者的官方替代交付，不是第三方镜像。
- ReMIND-002、43例筛选、manual tumor/cerebrum语义全部不变，避免失败后挑新病例。
- 本机无DCMQI/highdicom/pydicom安装；固定使用不安装venv的纯Python`pydicom==3.0.1` wheel，SHA256 `db32f78b2641bd7972096b8289111ddab01fb221610de8d7afa835eb938adb41`，已与PyPI官方JSON核对。

## 4. 具体怎么做？

只通过TCIA NBIA API按三个冻结SeriesInstanceUID下载。T1ce再下载一次，与旧probe保留的192个`.dcm`逐文件SHA清单exact；tumor和cerebrum两个SEG series各独立下载两次，忽略每个ZIP中明确名为LICENSE的附件，只比较实际`.dcm` bytes/SHA并核对官方FileSize/UID。

用固定pydicom读取SEG header和1-bit frames。每帧必须只有一个SegmentNumber，并通过DerivationImage/SourceImage的ReferencedSOPInstanceUID映射到T1ce的192个源切片；未出现的切片补0，不插值。重建manual tumor与automatic cerebrum原生3D mask，核对rows/columns、方向/位置、二值、非空和tumor是否在cerebrum内。只记录影像头颅外组织比例等格式事实，不选择N4、阈值或brain extraction。

## 5. 唯一变化与固定项

- 唯一变化：标签官方交付从失败的package755 NRRD切换为同collection NBIA DICOM SEG。
- 固定不变：43例metadata队列、首例ReMIND-002、T1ce/manual tumor/cerebrum语义、原生几何门、0模型、无GT crop。旧下载/失败收据保留且在新preaccess中引用。

## 6. 成功、失败和停止规则

- Primary工程门：T1ce两次192个DICOM逐文件内容manifest exact；两个SEG各两次DICOM bytes/SHA exact且符合官方FileSize；Patient/Study/Series/Referenced SOP UID全部正确；1-bit帧能无歧义映射到源slice；两个mask二值/非空、geometry exact，tumor⊂cerebrum。
- Promotion gate：全部通过才称DICOM SEG source probe可用，并仅允许另立输入桥接；不直接授权43例或模型。
- Failure/stop rule：任一重复SHA、UID、frame mapping、geometry/label门失败即停止，不换病例、不装另一个decoder、不用NRRD补洞、不重采样修复。
- 是否允许进入test：不允许。本阶段0 checkpoint/forward/指标；未来仍需完整七问、全量source/input/overlap门与独立排行榜分区。

## 7. 成本与风险

- 训练/GPU/forward：0。
- 网络：重下ReMIND-002 T1ce ZIP约8.9MB；tumor/cerebrum SEG各下载两次，预计压缩总量<15MB。
- 依赖：2.4MB pydicom wheel仅在`/tmp`，不安装；代码自己按SOP UID组装，不依赖隐式frame顺序。
- 最大风险：DICOM SEG多帧/引用结构可能超出简化解析合同；raw T1ce仍含头颅，来源通过也不等于模型输入桥接通过。Manual “whole tumor”按手术规划和单序列可见范围生成，与BraTS多模态WT仍有语义caveat。

## 8. 最终结果（完成后填写）

- Validation/Test/CI：无模型指标。
- 是否支持直觉：待DICOM SEG probe。
- 下一步：只按工程门决定是否设计input bridge。
- 报告/哈希：待填写。
