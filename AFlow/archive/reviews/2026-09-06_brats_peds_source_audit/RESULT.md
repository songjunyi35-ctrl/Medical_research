# BraTS-PEDs（TCIA package 1217）来源可行性审计：标签合格、输入前提失败，模型前关闭

日期：2026-09-06（Fable 8.5h session）。只读metadata/目录审计，0像素下载、0模型。

## 数据事实（一手来源：TCIA collection页 + BraTS-PEDs 2024论文 arXiv 2404.15009）

- Package 1217 `BraTS-PEDs-v1`：348例目录=Training 257（t1c/t1n/t2w/t2f/seg五文件）+Validation 91（四模态、无seg），1,649文件/35.17GB；无`.sums`。inventory SHA=`d108bf42f96bc8d8d7fa819fb1ae130d5b30de0df30f973f92e8faea2da92282`。348例ID全部唯一subject（无多时点）。选中候选=257 Training的T1c+seg，共约6.8GB。
- 标签：儿科专用自动分割初始化→志愿neuroradiologist逐例ITK-SNAP人工修订→board-certified attending neuroradiologist迭代复核。数值字典明确：ET=1、NET=2、CC=3、ED=4。treatment-naive（术前/治疗前）。CC BY 4.0。
- 机构：CBTN、DMG/DIPG registry、Boston Children's、**Duke**、Yale。Duke与成人BraTS2024七来源站重叠；无patient桥，但儿科人群与成人治疗后GBM人群年龄/疾病不重叠（不可作ID证明）。

## 两个语义/输入门结论

1. **WT语义分歧（可解决、须冻结）：** 儿科官方WT=ET+NET+CC（**不含ED**），与成人WT（含FLAIR异常/水肿）不同。若未来评测，与项目既往术前队列一致的目标应为all-positive `{1,2,3,4}`并显式标注与儿科官方WT的差异；不得结果后换目标。
2. **输入前提失败（当前关闭原因）：** 官方明确"released images are **not skull-stripped**（only defaced）"，且无模板配准/1mm标准化说明。本项目冻结的canonical输入链（image-only positive bbox "brain crop"→源非零z-score→128³）只对skull-stripped、背景为零的输入定义；训练数据与全部七个已评cohort均为skull-stripped。对含颅骨/头皮图像，该链会退化为"头颅crop"、z-score被颅骨脂肪等高信号主导，与训练输入定义系统性偏离——这不是可用大caveat掩盖的评测，而是RHUH（发布态z-score负背景）/MOTUM（brain support不含GT）同类的输入前提失败。自行加skull-strip（SynthStrip/HD-BET等）属于未经门检的新输入链组件，当前无fresh容量为其建立开发证据，禁止直接接frozen checkpoint。

## 结论与解除条件

- 0像素、0 cache、0模型；本轮关闭。
- 解除条件（登记为阻塞#14）：(a) 发布者提供skull-stripped/brain-mask一致版本；或 (b) 未来经独立预注册且在新train-only证据上过门的skull-strip前置桥；解除后仍须先冻结all-positive目标、双传输门（无.sums）、Duke机构重叠表述与非1mm grid caveat。

## 证据

- [SOURCE_AUDIT.json](SOURCE_AUDIT.json)、[PACKAGE1217_INVENTORY.json](PACKAGE1217_INVENTORY.json)
- 一手：TCIA collection页（defacing/not skull-stripped原文）、arXiv 2404.15009 v4（标签数值/WT定义/treatment-naive）。
