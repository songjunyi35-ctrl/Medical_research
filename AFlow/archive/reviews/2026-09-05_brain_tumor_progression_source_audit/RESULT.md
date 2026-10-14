# TCIA Brain-Tumor-Progression外部来源审计

## 结论

这个来源的时间点很适合治疗后泛化，但当前不能作为BraTS Whole Tumor（WT，全肿瘤）外部真值队列：主体受NIH Controlled访问限制；官方只写“用T1w图像生成的二值肿瘤mask”，没有说明人工勾画、专家复核、质量控制或包含FLAIR水肿。因此不能把它重命名为BraTS WT。

## 为什么它看起来有吸引力

- 20名新诊断glioblastoma患者，每人2次MRI：放化疗结束90天内、临床进展时，共40 studies。
- T1增强前后、FLAIR、T2、ADC与灌注均有；所有series配准到T1+C。
- 官方规模383 series/8,798 images/3.16GB，并包含binary tumor masks。

这比术前UTSW更接近当前post-treatment域，但“域接近”不能弥补标签语义/来源不清。

## 为什么当前关闭

1. 官方下载许可是NIH Controlled Data Access Policy；本轮没有受控资格或申请授权。
2. 官方页面没有说mask由人工/专家产生，也没有推荐数据作者论文补充说明。
3. “generated using T1w images”更像T1可见肿瘤，不能证明包含BraTS WT需要的T2/FLAIR异常区。
4. 找到的第三方复现包只含11个目录，并把DICOM合为NIfTI、最近邻resize到256×256、裁到22层再pad到24层；这是部分且改变geometry的衍生物，不能替代官方20人原生证据。

## 决定、成本和边界

不申请受控访问、不下载manifest/DICOM/mask、不跑模型、不建榜。未来只有同时取得受控授权及一手人工/专家标签语义证据，才可重新评估；即使可用，也须先冻结是否为增强肿瘤而非WT的独立任务口径。

本审计仅阅读官方网页与公开衍生说明：0大文件、0体素、0GPU、0forward。

一手来源：[TCIA BRAIN-TUMOR-PROGRESSION](https://www.cancerimagingarchive.net/collection/brain-tumor-progression/)，DOI `10.7937/K9/TCIA.2018.15quzvnb`。

