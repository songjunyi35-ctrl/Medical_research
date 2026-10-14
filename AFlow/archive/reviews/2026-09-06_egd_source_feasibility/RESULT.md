# Erasmus Glioma Database（EGD）外部来源可行性审计

状态：🟡 科学/标签条件合格；官方账号与Data Usage Agreement（数据使用协议）阻塞。没有患者清单、影像、标签或模型访问。

## 为什么查它

UTSW362虽然是高质量人工多类标注，但官方文件含论文和数据字典未定义的正值3/5，不能猜Whole Tumor（全肿瘤，WT）语义。EGD一手论文明确提供二值WT，可能绕开多类数值映射问题，并提供跨医院术前数据。

## 一手证据确认了什么

- EGD来自荷兰Erasmus MC，含774位2008–2018年术前胶质瘤患者，每人有T1、T1增强（`T1GD.nii.gz`）、T2和T2-FLAIR，仿射配准到MNI152 1mm grid并去面部。
- 374位患者有人工`MASK.nii.gz`，400位只有卷积神经网络自动mask。论文明确人工mask中`1=肿瘤、0=背景`，是基于T2或T2-FLAIR勾画的whole tumor；`Segmentation_source.xlsx`区分四位人工观察者`OBS1–OBS4`与`AUTO`。
- 因此未来只能冻结374个manual候选，不能把400个自动标签混入人工外部榜。T1增强可作为模型输入，但真值是T2/FLAIR定义的WT，这是必须保留的跨模态标签caveat。
- 论文说明数据通过Health-RI XNAT项目`egd`和API提供，数据使用协议为补充材料；文章/补充材料按CC BY开放。

## 实际访问门

官方XNAT项目metadata可匿名读取，但项目description明确写明：不能直接在XNAT申请账号；须邮件联系`servicedesk@health-ri.nl`，签署数据使用协议后才创建账号。匿名查询`/projects/egd/subjects?format=json`返回HTTP 403。

签署协议、代表机构作承诺及发送外部邮件都需要用户/机构授权，Codex不能代办。因此本轮在患者清单前停止：

- subject IDs：0；
- patient metadata：0；
- NIfTI images / masks：0 / 0；
- model forwards / GPU：0 / 0；
- internal test270、fixed validation、outer、confirmation、UCSD：均未访问。

## 患者隔离与未来门

目前不能称患者独立：未登录前拿不到774个subject清单，也尚未完成Erasmus与本地BraTS来源/患者排除证明。获得账号后仍须在任何影像/标签前：归档已签协议范围；只读项目清单/质量标志/`Segmentation_source.xlsx`；冻结manual且quality acceptable病例；审计与BraTS 1,350 union的机构和可用ID重叠；无法做患者级排除时只能标机构差异证据，不能冒充patient-disjoint。

只有这些metadata门通过，才能另立七问/预注册、固定T1GD→原输入链、binary WT、checkpoint、阈值、视图、cache、bootstrap和单次访问规则。当前不下载Docker downloader、不创建账号、不发邮件、不跑模型。

## 结论

EGD比当前未解码的UTSW标签更适合同口径WT：374例人工二值mask、标签0/1明确、公开论文完整。但它不是可立即匿名下载的队列；现阻塞是需要用户/机构签署数据使用协议的外部协调，不是代码或GPU问题。保留为最高优先级的授权后候选，不用自动400例替代。

## 一手来源与证据哈希

- [EGD一手数据论文](https://doi.org/10.1016/j.dib.2021.107191)，Europe PMC full-text XML SHA-256 `fb6c9845a653468c69e6b8b2e2f6badc3401cc838547febb93f1be98b796bdef`
- [官方Health-RI XNAT项目](https://xnat.bmia.nl/data/archive/projects/egd)，匿名project JSON SHA-256 `4062e422f30b6d78fad9438e305d0a6e4bcb9e097020eb9348dada025d083c52`
- [官方数据下载器归档](https://doi.org/10.5281/zenodo.4761088)（本轮未下载）
- 机器摘要：[SOURCE_EVIDENCE.json](SOURCE_EVIDENCE.json)
