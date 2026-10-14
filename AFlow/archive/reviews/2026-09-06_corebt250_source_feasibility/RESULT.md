# CoRe-BT250来源可行性审计

状态：标签和任务语义初步合格，但University of Washington预审批与Data Use Agreement阻塞；0表单、0邮件、0DUA、0病例清单、0影像、0标签、0模型、0性能。

## 为什么它有价值

CoRe-BT是2026年发布的华盛顿大学多模态脑肿瘤数据，2023-08至2025-08共310名胶质瘤患者。250人有完整术前T1、T1对比增强、T2、FLAIR，60人不完整；其中95人还有数字病理。310人的分类标签来自最终临床/病理诊断，包含176例GBM/IDH-wildtype HGG、71例IDH-mutant astrocytoma、31例少突胶质瘤等。

一手论文说明，完整四模态病例的三个标准胶质瘤区域（edema、enhancing tumor、necrotic core）由BraTS2023优胜nnU-Net ensemble初始化，再由有丰富胶质瘤分割经验的标注者逐例检查并修正错误区域，组合成ground-truth mask。把三个正区取并集在语义上可形成术前WT。University of Washington不在本项目BraTS2024的Duke/UCSF/Missouri/UCSD/Heidelberg/Michigan/Indiana七站；没有跨库patient-ID桥，只能称institution-disjoint/no-known-overlap。

一手证据：[CoRe-BT论文](https://arxiv.org/pdf/2603.03618)第4–5页、[官方仓库](https://github.com/KurtLabUW/CoreBT)。

## 当前阻塞

官方仓库要求：

1. 填写dataset access form；
2. 联系两位University of Washington负责人，并提供姓名、机构和用途；
3. 等待pre-approval；
4. 收到并由用户/机构签署Data Use Agreement（数据使用协议，DUA）；
5. UW审核签署文件后才授权数据。

Codex不能代表用户/机构提交身份、发送外部联系或签署DUA。本轮没有触发这些动作。仓库代码的CC BY 4.0声明不能替代数据访问DUA。

## 解除后的第一门

用户/机构若完成授权，必须先只读冻结：实际mask病例数是否等于论文完整四模态250；逐例modality/mask清单与label数值字典；expert correction状态；是否混入recurrent/treatment-related病例；CoRe-BT与BraTS2024 patient crosswalk；官方checksum与许可范围。只有metadata门过后才写七问/预注册和访问NIfTI。

当前结论是“高优先级、用户可解除”，不是可立即运行，也不是hidden blind。不得通过ImageCLEF随机placeholder labels、第三方镜像或仅公开embedding绕过DUA。

