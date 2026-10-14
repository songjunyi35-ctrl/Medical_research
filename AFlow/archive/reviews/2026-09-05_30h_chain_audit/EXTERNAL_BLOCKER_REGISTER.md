# 外部证据阻塞登记表（2026-09-06）

本表只管理“怎样获得下一份合法证据”，不授权模型、调参或外部联系。UCSD75、Africa95、CPTAC33、WAW-preop152与WAW-postop-only31均已完成并关闭，不在待办队列。

2026-09-06 Fable补记：UCSD-PTGBM package 1221（BraTS-GLI 2024 test的UCSD份额，2026-01-22公开）经来源审计+患者排重后已作为UCSD-BraTS2024test56完成一次性评测并关闭（[结果](../../research_records/2026-09-06_ucsd_brats2024test_frozen_panel/RESULT.md)）；该来源不再是候选。本轮复查Africa2024（仍V1）、UTSW字典（仍无）、UCSF-PDGM桥（仍无）、UPENN manifest（仍无），均未解除。

## 1. EGD374人工binary WT：用户/机构可解除

- 价值：774位Erasmus术前患者中374位manual `MASK=1 tumor/0 background`，T1GD齐全，标签语义最清楚。
- 阻塞：Health-RI要求邮件service desk、签Data Usage Agreement后建账号；匿名subjects API 403。
- 谁能解除：用户/所属机构；Codex不能代签或代表机构承诺。
- 解除后第一步：只读manual/quality/subject清单和许可范围，冻结重叠审计；未过metadata门前不下载NIfTI。

## 2. CoRe-BT250专家修订mask：用户/机构可解除

- 价值：UW 250名完整四模态术前病例，三分区由BraTS2023优胜模型初始化后经有经验标注者逐例纠错，可组成WT；机构非BraTS2024七站。
- 阻塞：官方要求form、负责人联系、pre-approval和用户/机构签署DUA；当前还没有实际mask清单、label字典或patient bridge。
- 谁能解除：用户/所属机构完成UW申请和DUA；Codex不能代填身份、发邮件或签署。
- 解除后第一步：metadata-only核实际mask数、label、expert状态、recurrent混入、overlap、checksum和许可，不先下载NIfTI。

## 3. UTSW362人工多类：需要发布者数据字典

- 价值：362例专家细化、UTSW非BraTS2024来源站，双源724/724 exact。
- 阻塞：manual NIfTI含未定义3/5；3见116例，5见5例，非r原始label同样存在。论文只定义ET/NCR/ED，FeTS代码只列0/1/2/4，不能证明3/5含义或all-positive安全。
- 谁能解除：UTSW/TCIA发布者提供数值映射或更正版本。
- 解除后第一步：哈希归档发布者说明，重新冻结target语义；不能沿用当前失败cache直接跑模型。

## 4. BraTS-Africa 2024扩展：等待TCIA公开版本

- 价值：2024在2023的95人基础上增加20个新test病例，仍来自非洲低资源域。
- 阻塞：一手2026论文称expanded 2024 dataset将归档，当前TCIA仍只有Version1/package948；本轮已访问并关闭2023的95例，不能再用同95重测。
- 谁能解除：TCIA/BraTS发布新版本及标签。
- 解除后第一步：metadata-only比较病例ID，严格只冻结未进入当前95的新患者；隐藏test未公开标签时不得绕过。

## 5. UCSF-PDGM剩余199人：需要BraTS2024 patient crosswalk

- 价值：专家修订/双专家批准、T1GD齐全。
- 阻塞：495真实患者中296位至少一时点明确映射BraTS21；UCSF又是BraTS2024来源站，余199位无PDGM↔2024桥，不能证明独立。
- 谁能解除：UCSF/TCIA/BraTS组织者提供患者级映射。
- 解除后第一步：像素前排除任何BraTS2024重叠；不能用空BraTS21字段或图像hash证明跨时点患者独立。

## 6. BraTS2024训练源官方parity：用户Synapse自签

- 价值：解决所有继承checkpoint的训练源provenance限制。
- 阻塞：corrected-v2 `syn60086071`匿名下载403；SelfSign requirement9606685要求登录用户接受条款。本地Kaggle三包与官方zip/member parity未证明。
- 谁能解除：用户本人登录Synapse自签；Codex不能代同意。
- 解除后第一步：下载官方zip、核MD5与逐member SHA；不改变现有分数，只更正来源证据。

## 7. ReMIND43人工T1ce tumor：需要发布者几何/可靠原生交付

- 价值：BWH非七站、manual neurosurgical fellow+attending复核、43例。
- 阻塞：NRRD双传输第二份SSH失败；替代DICOM SEG没有source SOP引用且内部frame间距超冻结DS区间。当前轮已明确不加第五种容差。
- 谁能解除：发布者给出原生image-label grid说明/校验和或稳定NRRD交付。
- 解除后第一步：新窗口重做来源级双传输/几何合同；不能放宽旧容差或用GT重采样后冒充原生exact。

## 8. RHUH80专家纵向：缺兼容且过门的训练/输入链

- 价值：40人术后/复发80 scans，专家人工修正。
- 阻塞：发布态z-score float32负背景；canonical N4等价门失败，no-N4输入工程虽过但no-N4模型Stage A失败，Otsu输入复现门失败。
- 谁能解除：未来独立训练方法在新证据上过门，或发布者提供可验证pre-zscore源。
- 解除后第一步：另立预注册；禁止把旧canonical checkpoint接no-N4输入绕过资格。

## 9. MOTUM-HGG29：需要发布者一致脑区或预先独立训练的兼容输入链

- 价值：29名中国三院WHO 3/4级原发胶质瘤，双语义mask经医师校正，双官方source exact。
- 阻塞：共享orientation已无插值通过，但0001有892/21007 WT体素在T1ce正值与非零支持bbox之外；当前canonical image-only crop会丢4.246% GT。21/24层、5.5–6.75mm也强重采样。
- 谁能解除：发布者提供与标签一致的brain mask/修正版processed image；或未来另一个从零训练且在新证据过门的全FOV/发布者brain-support模型。
- 解除后第一步：新窗口先做input-only exact门；不得看GT扩框、事后挑padding、排0001或把旧checkpoint硬接source-tuned offset。

## 10. UPENN术前corrected：需要官方完整manifest/parity与患者桥

- 价值：官方611术前中232份expert-corrected多区标签，UPenn机构不同。
- 阻塞：data-nih逐文件镜像只有147 seg而非官方232，provider parity未证；论文173例进入旧BraTS且无UPENN↔BraTS2024患者桥。
- 谁能解除：TCIA/UPENN提供可选择官方NIfTI manifest/checksum与挑战映射。
- 解除后第一步：metadata-only冻结完整232并排重；不能按镜像恰有seg的147选择病例。

## 11. OpenNeuro-GBM334：需要发布者一致brain support或独立全FOV模型

- 价值：八医院334个版本化T1-CE/专家复核修订WT配对，样本量大、机构与BraTS2024七站不同。
- 阻塞：全输入门中BO118有4/348742、KA86有6/97700个WT体素在发布者skull-stripped图像正值crop外；按冻结零丢失门，现成canonical checkpoint链0模型关闭。
- 谁能解除：发布者提供与最终mask一致的brain support/修正版；或未来从零训练且在新train-only证据预先过门的全FOV模型。
- 解除后第一步：另立input-only协议，不能排两例、裁GT、按本次GT选padding或换variant冒充原冻结链。

## 12. IMAGO249术前/190术后自动mask：需要人工修订标签与用户DUA

- 价值：Amsterdam500人全有术前四模态、381人有早期术后，249/190有分区mask，机构非BraTS2024七站。
- 阻塞：mask由PICTURE自动生成，医师只accept/reject且错误者被剔除，未逐体素修订；术后只含强化残留。EUCAIM/Health-RI还需注册、negotiator和DUA。
- 谁能解除：发布者提供未按自动质量筛选的专家人工修订WT；用户/机构另须完成DUA。
- 解除后第一步：先冻结人工标注级别、全部eligible病例和patient overlap；不能把自动mask或accept/reject称ground truth。

## 13. Fresh内部inner仍不足；复用开发替代路线已获授权

- 价值：未来方法需要不依赖外部test的开发证据。
- 阻塞：immutable role下只剩2 subjects/3 scans未作为评测揭示，不能建立42-scan fold。
- 2026-09-07用户已明确批准复用开发：排除旧confirmation后864 scans/391人，固定313/78患者fit/dev，实际684/180扫描。该范围现在可开展预注册方法开发；历史fresh不足事实仍成立。
- 当前动作：角色已冻结、无教师配对辅助监督已启动，先完成当前两seed训练与dev；其他test/outer/confirmation/外部角色继续保护。不得把reused development称fresh/blind。
- 证据：[授权](../../research_records/2026-09-07_reused_development_roles/AUTHORIZATION.md)、[划分](../../research_records/2026-09-07_reused_development_roles/SPLIT.json)。

## 14. BraTS-PEDs257（2026-09-06新增）：需要skull-stripped一致版本或独立过门的skull-strip前置链

- 价值：257名treatment-naive儿科高级别胶质瘤，专家迭代修订四亚区标签（ET1/NET2/CC3/ED4）、CC BY 4.0、T1c齐全；儿科人群与成人训练队列实际不重叠（Duke机构重叠、无patient桥）。
- 阻塞：官方发布明确not skull-stripped（仅defaced）、无模板/1mm标准化；本项目canonical "image-only brain crop"输入链只对skull-stripped输入定义，含颅骨图像会使crop/z-score系统性偏离训练定义。自加skull-strip是未经门检的新输入组件，无fresh容量为其建立证据。
- 谁能解除：发布者提供skull-stripped/brain-mask版本；或未来新数据角色下独立预注册并过门的skull-strip桥。
- 解除后第一步：冻结all-positive {1,2,3,4}目标（显式标注与儿科官方WT=ET+NET+CC的差异）、双传输SHA门（包无.sums）、非1mm grid caveat。见[来源审计](../2026-09-06_brats_peds_source_audit/RESULT.md)。

机器版：[EXTERNAL_BLOCKER_REGISTER.json](EXTERNAL_BLOCKER_REGISTER.json)。
