# UCSF-PDGM外部队列可行性审计

状态：❌ 独立患者门无法证明，停在公开metadata；0 NIfTI、0标签体素、0模型。

## 为什么它看起来合适

UCSF-PDGM官方v5有501条术前扫描记录、实际495位患者，均含标准化3T多模态MRI。官方方法说明多区肿瘤分割先由BraTS获胜模型ensemble产生，再由受训放射科医生人工修订并经2名专家批准；包含增强肿瘤、非增强/坏死核心和FLAIR异常区。它有T1 postcontrast并公开CC BY 4.0 NIfTI，看起来适合冻结baseline跨阶段验证。

## 泄漏映射的直接结果

TCIA曾专门增加`BraTS21 ID`和cohort列以防同时使用两数据集时发生data leak。当前官方`UCSF-PDGM-metadata_v5.csv`有501行：

- 298/501已有BraTS2021 ID，其中262为Segmentation Training、36为Segmentation Validation；
- 203/501没有BraTS2021 ID；
- 官方说明6条记录实际是另6条记录的随访扫描，所以总患者数495。按这6对患者合并后，至少296位患者已由任一时间点明确映射进BraTS2021，表面剩余199位未映射患者。

“没有BraTS2021 ID”不等于没进本项目BraTS2024。当前本地post-treatment BraTS2024七个来源站本身包含UCSF，且一手论文说明约600例来自既有UCSF-LPTDG/ALPTDG。官方v5只给BraTS2021映射，没有UCSF-PDGM↔BraTS2024 post-treatment患者桥；同一患者的术前与治疗后影像也不会靠像素hash直接相等。

因此不能把199位表面未映射者声明为patient-disjoint。只按ID namespace或只排除298个BraTS2021映射会漏掉同院跨时点患者风险。

## 访问与停止边界

- 只下载公开TCIA页面、58,149-byte v5 metadata和1,964-byte glossary；解析501行、298个BraTS21映射。
- 当前v5 package1065标称142GB；没有请求package token、遍历患者文件、下载NIfTI/segmentation或运行模型。
- 内部test270、fixed validation、outer、confirmation、UCSD/UTSW/RHUH均未访问。
- 即使标签质量合格，患者隔离是先验硬门；没有官方BraTS2024交叉表或患者级排除证明前不做source probe/cache/model。

## 结论和未来条件

UCSF-PDGM不能作为当前“新独立外部队列”。这不是标签质量否定，而是来源重叠证据不足。未来只有发布者或BraTS组织者提供PDGM↔BraTS2024 patient crosswalk，并在像素前排除所有重叠患者，才可重新审计；不能用图像相似度猜测不同治疗时点是否同一患者。

## 一手来源与哈希

- [TCIA UCSF-PDGM v5](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/)，页面SHA-256 `81373178b958d867ee2b629fc4b2837a7e8b33efbdb4696ab09f81bf134c8829`
- [官方数据论文](https://doi.org/10.1148/ryai.220058)
- 官方v5 metadata SHA-256 `afc1c23a0eb0597b56e603741e72f2308b4860281187cd3808e05b7a0116fd78`
- 官方glossary SHA-256 `f1404bcd730572d05bdf7ead2e62e32843944d0e27a05cd0ef2dcca8ce69378d`
- [机器摘要](SOURCE_EVIDENCE.json)
