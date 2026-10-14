# OpenNeuro ds007045 v1.0.1来源审计

状态：版本化metadata与完整对象树门通过；冻结334例T1-CE/专家复核标签交集。后续已读取全部334对做输入门，2例零裁剪损失失败；0 checkpoint-with-source、0外部模型forward、0性能访问。

## 来源与标签

- 一手数据描述发表于Scientific Data（2026-08-07），337名组织学确诊、IDH-wildtype原发胶质母细胞瘤患者，均为术前MRI，来自八家医院；T1平扫、T1对比增强、T2和FLAIR齐全。
- 最终标签按BraTS约定：0背景、1坏死核心、2非强化/水肿区域、3强化肿瘤。SegResNet只负责初始轮廓；所有结果由有神经肿瘤经验的神经放射科医师逐例检查，87例小修、11例大幅人工重画。它是expert-reviewed/refined，不描述为每例从空白开始的纯手工标签。
- OpenNeuro DOI固定为`10.18112/openneuro.ds007045.v1.0.1`，许可CC0。v1.0.1变化记录为增加预处理数据；本审计通过GraphQL查询`ds007045:1.0.1`快照，所有选中对象URL均带`versionId`，不使用2026-09-04仍变化的未固定draft。
- 一手入口：[Scientific Data文章](https://www.nature.com/articles/s41597-026-07953-2)、[OpenNeuro v1.0.1](https://openneuro.org/datasets/ds007045/versions/1.0.1)、[官方API说明](https://docs.openneuro.org/api.html)。

## 像素前冻结队列

论文/participants.tsv有337个唯一ID。为避免发布者已做N4和z-score后再次重复处理，项目只选`derivatives/processing`中的MNI152配准、skull-stripped、但未做发布者N4/z-score的`t1ce`，再配`derivatives/segmentation`最终mask。

严格交集为334例：

- `sub-BO199`、`sub-NSK07`：有预处理T1-CE但v1.0.1无最终segmentation，像素前排除。
- `sub-BO97`：有segmentation但无对应`derivatives/processing` T1-CE，像素前排除；不换到raw或另一预处理版本救场。
- `sub-BO8`的标签文件由发布者命名为`MNI-152`而非其余病例的`MNI152`；对象树中只有这一个同subject mask，保留，但仍须在首例/全例geometry门验证，不能仅凭名字假定一致。

冻结334例站点计数：BA 3、BO 170、FO 9、JS 11、KA 80、ME 7、NSK 40、TO 14。README写JS 12，participants.tsv与文件交集实际为11；显式记录发布者计数笔误，不补造病例。八家医院均不在BraTS2024训练的Duke/UCSF/Missouri/UCSD/Heidelberg/Michigan/Indiana七站，因此只称institution-disjoint/no-known-overlap；没有跨库patient-ID桥，不能称ID证明的patient-disjoint。

## 对象与磁盘门

- 334个T1-CE共2,168,435,952 bytes；334个标签共8,980,427 bytes；合计2,177,416,379 bytes（约2.03 GiB）。
- 每个对象的git-annex ID同时编码精确size与SHA-256；334对版本化URL及SHA写入机器清单，规范化manifest SHA-256=`8342256890d3b6a8ebf408ff081a84867f76fd816da0e97e76f98b41e1cce6cb`。
- 当前磁盘仅约1.9 GiB可用，不能常驻完整source或334×128³ float32 cache。下一步只允许单例探针；若通过，正式方案必须逐例流式下载、SHA验证、预处理和推理，临时文件逐例覆盖，raw计数与收据常驻。不得静默删现有材料腾双份cache。

## 决定与下一门

来源身份、许可、标签语义、版本化对象树和334例像素前交集通过。首例双重预处理也通过；随后334例全输入门发现BO118有4个、KA86有6个WT体素落在图像正值crop外。按预注册零丢失门，性能面板在模型前关闭，不排例/GT扩框/换variant。详见[负结果](../../research_records/2026-09-06_openneuro_gbm334_frozen_panel/RESULT.md)。

- [机器清单](SOURCE_INVENTORY.json)，SHA-256 `4bba11fb31913b18f44428087d9daf9dcd8f49590577ea11b715ddbcbbbf66eb`
- 审计脚本 `scripts/audit_openneuro_ds007045_source.py`，SHA-256 `55cf513cb5fbb6cdf52ad4c73f4e6e0b30cc0af7a1d8d478460bb06a14e3b4dd`
