# UCSD75来源更正：患者与标签确认，图像是8位量化镜像

状态：2026-09-05完成全部75个既有外部病例的官方来源核验；没有模型推理、预测指标重算、病例/参数选择或旧报告修改。

## 结论

历史`data-nih/tcia`仓库自述为Pittsburgh Fiber Data Hub整理镜像，并非TCIA官方GitHub。直接从TCIA公开主包1019获取相同75例的T1post和BraTS分割150个文件，合计426,129,798 bytes，与原镜像逐例比较：

- 75/75官方图像float32，75/75镜像图像uint8；0/75图像数组相等。
- 75/75官方分割与镜像分割的标签值逐体素相等；150/150图像/标签空间几何一致。
- 量化后总共18,156个原图非零体素变为零，每例9–655个，中位232个；没有源零值变镜像非零。
- 非零区域图像相关系数范围0.960006–0.999933，中位0.999695。高相关不等于无损，也不能证明N4和后续模型输出保持不变。
- TCIA最新临床v3表直接复核75/75病例映射一致；所选53个患者与项目既有患者的重叠为0。患者身份来自官方表，不依赖镜像名称自证。

因此既有UCSD75结果仍是患者隔离、专家标签一致、两系统共同输入的一次性外部比较，但**输入限定为镜像8位量化版本**。不能称TCIA原始浮点NIfTI表现，不能把镜像GitHub asset SHA叫官方发布SHA。原始文档/预注册/报告哈希保留，通过本更正与canonical说明补充准确来源。

## 原结果的有效范围

在镜像8位版UCSD75（75 scans/53 subjects）上，GCSAM184的Scan/Subject/Point=0.755647/0.755152/71，Full972 KD168=0.745554/0.739055/70。二者Dice配对95%区间[-0.002465,+0.027164]/[-0.000631,+0.038954]跨0；仍只称该输入版本上的点估计领先，不称显著或官方原始图像最佳。

内部reused test270的8×总体点估计最高仍为Full972 KD168 Flip8，0.757270/0.741254/265；1×Scan最高GCSAM184，1×Subject/Point最高Full972 KD168。此次来源检查没有改变任何分数或排序。

## 访问与成本

这次重新读取的是已经评测过的同75患者的原始文件，只比较身份、几何、dtype与数组；没有运行任何模型、没有修改旧cache/报告、没有重新计算分割预测Dice。挑战测试包1221、新患者、旧test270模型均未访问。官方临床表与既有1350清单只用于身份排除。

正式源文件证明目录`workspace/BraTS/ucsd_official_source_verification_20260905/raw`约407MiB；首例先行证明另约5MiB；均保留，未删除材料。下载客户端是由IBM官方索引取得的临时ascp。该来源核验结果不用于正在训练的no-N4方法选择；no-N4七问/预注册/代码冻结早于本次官方像素核验，动机来自train图的独立数值审计。

## 证据

- [镜像自述归档](source_metadata/DATA_NIH_TCIA_README_20260905.txt)
- [官方临床映射审计](UCSD_OFFICIAL_CLINICAL_MAPPING_AUDIT.json)
- [官方文件访问前收据](UCSD_ALL75_DIRECT_PREACCESS.json)
- [官方下载收据](UCSD_ALL75_DIRECT_DOWNLOAD.json)
- [全部75例来源比较](UCSD_ALL75_DIRECT_SOURCE_COMPARISON.json)，SHA256 `1dc05453fce200e11739dafbc1b73d33a5ef1bd491d3174b3ebb42504d101ac7`
- [当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[方法目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[排行榜](../../../AFLOW_BRATS_LEADERBOARD.md)
- [原始评测归档](../../research_records/2026-09-02_ucsd_ptgbm_external_confirmation/RESULT.md)保持历史哈希，来源称谓以本更正为准。

一手来源：[TCIA UCSD collection](https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/)、[TCIA临床v3表](https://www.cancerimagingarchive.net/wp-content/uploads/UCSD_PTGBM-clinical-information_v3_2026-12-Mar.xlsx)、[镜像仓库自述](https://github.com/data-nih/tcia)。
