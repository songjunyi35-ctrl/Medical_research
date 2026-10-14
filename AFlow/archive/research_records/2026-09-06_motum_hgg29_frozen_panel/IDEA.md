# Experiment idea: MOTUM-HGG29冻结checkpoint面板

Status: planned; official dual-source and HGG29 array audit complete; cache/checkpoint/performance not accessed.

## 1. 现在的问题是什么？

同一GCSAM184相对Full972 KD168 identity的Scan Dice差在internal270/UCSD75/Africa95/CPTAC33/WAW152为`+0.000244/+0.009634/+0.068617/+0.001954/+0.010223`，只有Africa配对区间全正。固定Flip8相对Full identity在internal/Africa/CPTAC/WAW为`+0.005510/+0.003435/+0.003246/+0.010116`，四处点估计均正、三处区间全正。仍需未见来源检验方向，但不能把已见结果用于改模型或阈值。

MOTUM有67人，其中Participants表在像素/模型前确定29名WHO 3/4级原发高等级胶质瘤（HGG），另38名脑转移瘤必须排除。29例均有T1对比增强图和两个经医师校正的语义mask。

## 2. 一句话直觉

把封存的两位“医生”和固定八方向复核带到中国三家医院的厚层MRI小考；只考29人，所以看方向和不确定区间，不拿一次小考重写答案。

## 3. 为什么选择这个idea？

- 官方G-Node DOI快照开放、CC BY 4.0；发布者没有checksum，故两次984,434,960-byte独立下载逐字节SHA exact且双CRC通过。
- 29例由Participants表`WHO CNS Grade∈{3,4}`且`Origin='/'`精确选出；HGG和38例转移瘤在性能前可分，不靠图像或模型选择。
- 两区域半自动勾画后由有经验医师仔细验证和校正；FLAIR非强化异常与T1ce强化肿瘤的正类并集对应术前WT。
- 安徽医大二附院、上海长征医院、中科大附一院不在BraTS2024七站。无患者桥，只称institution-disjoint/no-known-overlap。
- UTSW/EGD/UCSF/ReMIND等负结果是标签、权限、重叠或几何门，没有运行MOTUM；WAW/CPTAC的结果也没有否定厚层中国临床域。
- 一手来源：[G-Node DOI snapshot](https://doi.gin.g-node.org/10.12751/g-node.tvzqc5)与Scientific Data论文《A Multi-Center, Multi-Parametric MRI Dataset of Primary and Secondary Brain Tumors》。

## 4. 具体怎么做？

输入为每例processed T1ce。图像与两个mask先各自closest canonical RAS，并要求三者shape相同、affine最大误差≤`1e-5`；WT是`flair_seg_label1>0 OR t1ce_seg_label2>0`。之后按冻结canonical pipeline运行N4偏置场校正（shrink2、50×4迭代）、只用图像正值支持求脑框、非零z-score、image trilinear/label nearest缩到128³。

Full972 KD168一次循环运行完整8种三轴镜像，保存identity并把逆翻raw logits按固定顺序float32等权平均成Flip8；GCSAM184只运行identity。阈值`logit>0`得到WT，raw-logit argmax为自动Point；无Segment Anything Model（SAM）、模型ensemble、连通域或其他后处理。29 scans=subjects做绝对Dice/Point与固定10,000次病例bootstrap，并报告两项配对差。

## 5. 唯一变化与固定项

- 唯一变化：新增MOTUM HGG29 public external cohort。
- 固定不变：29例名单/顺序；38转移瘤排除；WT双语义mask正类并集；canonical RAS/N4/crop/z-score/128³；Full/GCSAM checkpoint SHA；Full identity/固定Flip8公式；GCSAM identity；阈值0；无SAM/后处理/模型ensemble；指标/bootstrap与解释门。
- 特殊但冻结的来源规则：一例`t1ce_seg_label2`内部值为1而非2，仍按语义文件`>0`；不能删除或硬筛数值2。

## 6. 成功、失败和停止规则

- Primary：Full identity/Flip8/GCSAM identity的Scan=Subject Dice与Point；GCSAM−Full、Flip8−Full配对Dice/CI/Point。
- 三档解释：delta>0且CI lower>0=`supported_on_MOTUM`；delta>0且CI跨0=`positive_but_uncertain`；delta≤0=`not_supported_or_reversed`。没有promotion gate。
- 工程门：双source SHA/CRC；29×3文件SHA；三NIfTI native geometry；允许mask域、WT非空、图像finite/正支持；首例单线程cache复现；全cache/hash；checkpoint/evaluator/smoke/preaccess。
- 任一失败即模型前停止，不改容差、病例、mask union、N4 support、resolution、threshold或view；性能一旦started不可重跑。
- MOTUM是public external-to-project、非hidden blind；不访问其他cohort。

## 7. 成本与风险

- 训练0；teacher/loss/optimizer不适用。
- 源：双官方快照约1.83 GiB、29×3选中源34.7 MiB。
- cache：29×128³ image+label约290 MiB，预计N4数分钟CPU。
- 推理：29×9=`261`次单例forward，预计数秒、峰值约5.1 GiB；Flip8部署8×、identity 1×。
- 最大风险：原生只有21/24 slices，层厚5.5–6.75mm，缩到128层不能恢复细节；processed T1ce含负背景，canonical pipeline固定仅用正值支持；一例label2数值异常；样本29导致CI宽；半自动后医师校正弱于完全独立双人工；无patient-ID桥；checkpoint训练源parity未解。

## 8. 最终结果（完成后填写）

- Validation：不适用。
- External：未运行；首例cache前geometry门失败。
- 置信区间/病例分布：无性能。Native affine误差`2.384e-7`，independent canonical误差`1.4488e-5`超过`1e-5`；全29有2例越门。
- 是否支持直觉：来源本身支持，当前独立canonical实现不支持。
- 学到什么：极小native header差异会被大索引flip放大；29例离散orientation均相同，可能用无插值共享orientation桥解决，但必须另立协议。
- 下一步：本实验关闭；只允许新input-only桥，不放宽容差/丢病例。
- 报告/哈希：[结果](RESULT.md)、[失败记录](CACHE_PROBE_FAILURE.json)。
