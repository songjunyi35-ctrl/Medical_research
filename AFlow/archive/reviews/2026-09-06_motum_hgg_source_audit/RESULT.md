# MOTUM-HGG29来源审计

状态：官方source、Participants表、29例T1ce与双语义mask arrays已审；0 cache/checkpoint/model/performance。

## 结论

MOTUM官方G-Node DOI快照`10.12751/g-node.tvzqc5`于2024-04-30发布、CC BY 4.0、匿名zip 984,434,960 bytes。发布者不提供归档checksum，因此进行了两次独立完整传输；两份SHA-256均为`2626862599a3fcfe4ac0cefcea3af5b190625275036cc8eb4c9039cbd54e2d7c`且byte-exact，双zip CRC全过。

- Participants表67人：`WHO CNS Grade∈{3,4}`且`Origin='/'`在任何模型前精确得到sub-0001至0029共29名原发高等级胶质瘤；38名肺/乳腺/卵巢/胃/皮肤来源脑转移瘤排除。
- HGG29中grade3/4=`12/17`；病理Oligodendroglioma/Astrocytoma/Glioblastoma=`9/3/17`。
- 29/29均有processed T1ce、`flair_seg_label1`和`t1ce_seg_label2`，三者native shape一致，affine最大误差`2.38e-7`。两个mask文件语义分别为non-contrast-enhancing FLAIR signal abnormality与contrast-enhancing tumor，WT冻结为两个语义文件的正类并集。
- FLAIR mask均值域0/1；T1ce mask有28例0/2、1例0/1。不能用数值2硬筛或把异常例排除；按发布的语义文件身份取`>0`可保持29例一致。
- WT均非空；WT体素位于T1ce正值支持的比例min/median/max=`0.9575/0.99995/1.0`。
- 严重grid caveat：processed体积只有21或24层，层厚5.5–6.75mm，in-plane shape/spacings也异质；任何128³评测都会强烈重采样，不能与1mm队列按绝对分数混排。
- 三个采集机构为安徽医科大学第二附属医院、上海长征医院和中科大附一院，不在BraTS2024七站；无patient-ID bridge，只称institution-disjoint/no-known-overlap。

## 决定

来源/标签语义/geometry门通过，可在独立七问和预注册下做一次冻结29例面板。必须保留半自动后医师校正、极厚层、public非hidden blind和无患者桥限制；不得加入38例转移瘤扩大样本，亦不得按一例label值异常剔除。

## 证据

- [机器来源审计](SOURCE_AUDIT.json)，SHA-256 `69d59768cd54ed526aa1cc8b0f66e31884ea4aaad171a1a16bc17c4488412932`
- [冻结HGG29](FROZEN_HGG29_COHORT.json)，SHA-256 `f49e8f888b0fce8b7f34da1f9eadb49039d10aaed8e0538008f215df1b953379`
- [官方DOI快照](https://doi.gin.g-node.org/10.12751/g-node.tvzqc5)
- 审计器SHA-256 `aaeb8cb6e915b8967c2658d85593ea522e417ed7b56f9c0e07cf790d4d49aaa5`

机器审计保留`cache=false/checkpoint_loaded=false/model_forwards=0/performance_metrics=false`边界。

双传输/CRC/SHA证据归档后，为WAW postop官方包留空间，删除冗余第二传输`copy_b` 984,434,960 bytes；第一份官方DOI快照、29×3选中源、审计/桥/失败记录均保留。删除不改变双传输exact结论，第二份可从同一官方URL重建。
