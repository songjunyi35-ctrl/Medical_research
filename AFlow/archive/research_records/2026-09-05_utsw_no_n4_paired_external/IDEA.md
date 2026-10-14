# Experiment idea: UTSW362主FeTS端到端canonical/no-N4配对外部确认

Status: planned/conditional，任何其余UTSW病例下载或模型访问前冻结。

## 1. 现在的问题是什么？

当前no-N4与Otsu只在未揭示的E42/F42内部Stage A训练中；现有RHUH条件链只有80 scans/40 subjects且预注册为no-N4绝对分数。UTSW官方625例中可严格固定362例人工细化术前病例，主FeTS T1ce+全网格人工标签约1.813GB；固定3例的no-N4和canonical N4输入都已工程通过，但还没有合法的全量same-seed端到端外部比较。

## 2. 一句话直觉

把同一批362份异校试卷交给两套完整流程，各自按原配方预处理后作答，再逐人比较差值；不能只看新流程的绝对分。

## 3. 为什么选择这个idea？

- UTSW来自UT Southwestern，不在BraTS 2024 post-treatment七家来源机构中；官方TSV逐例标出362份人工细化标签，完整6349文件清单已通过。
- 固定3例已确定主FeTS图与全网格`r`人工标签；no-N4双副本3/3 exact，canonical N4两个独立进程也3/3 exact，且两臂label/geometry相同、image不同。
- 单连接`--src-base`探针证明多病例同名文件能保留病例目录并复现旧SHA，解决全量下载工程风险。
- 旧N4数值失败涉及负背景编码往返；UTSW主图是非负/黑背景0。ReMIND负结果涉及不同DICOM SEG物理grid；UTSW图与`r`标签在固定3例原生affine exact。这些负结果没有否定UTSW。
- UTSW是术前SRI24，而训练是治疗后MNI；域差使其适合作为单独的跨阶段压力测试，也禁止与post-treatment排行榜混排。

一手来源：[TCIA UTSW-Glioma](https://www.cancerimagingarchive.net/collection/utsw-glioma/)、[数据论文](https://doi.org/10.1038/s41597-026-07274-4)。

## 4. 具体怎么做？

只有no-N4在E/F identity与Flip8全部门通过、条件Full972 same-seed canonical/no-N4 fixed168完成，并且RHUH优先链已有正式完成/关闭证据后才开始。先动态冻结两个checkpoint及上游raw哈希，做不含真实UTSW数据的双模型GPU smoke。随后从TCIA package1265把362个`brain_t1ce.nii.gz`和`rtumorseg_manual_correction.nii.gz`用`--src-base`各下载两份，724/724比较官方size和双份SHA。以copy A构建两个128³ cache：canonical运行原正值mask N4，no-N4不运行N4；两者都转RAS、all-positive标签合为WT、同brain bbox/z-score/插值。全量label/geometry须相同，固定3例再与copy B和已归档QC复核。

模型输入各自cache，same-seed/同架构/epoch168 canonical与no-N4各运行identity和固定Flip8、threshold0、无SAM/后处理。每例一位患者，计算Scan/Subject Dice、Point hit；用seed20261381做10,000次患者配对bootstrap，主比较为Flip8 no-N4−canonical。原始per-case、预测mask hash和所有正负结果均归档。

## 5. 唯一变化与固定项

- 唯一端到端因素：训练与推理输入是否做historical canonical N4。
- 固定不变：362个`Manually Refined Segmentation=Yes`病例；主FeTS图；全网格`r`人工标签；all-positive→WT；same-seed Full972 epoch168模型；4,700,897参数；128³；threshold0；identity/Flip8；无SAM/后处理；bootstrap seed/samples；不访问任何旧cohort。

## 6. 成功、失败和停止规则

- Primary指标：Flip8 paired Scan Dice delta=no-N4−canonical。
- 支持门：Flip8 delta>0；患者bootstrap 95% CI下界≥−0.003；no-N4 Point hits不少于canonical−3/362。Identity为预定次指标，不用于改协议。
- 来源/工程门：双下载724/724 size+SHA exact；362/362 finite/nonnegative/geometry/label值门；双cache label/geometry全exact；固定3例重建与已归档QC exact；GPU smoke、checkpoint identity、6516单例forward和报告哈希门全过。
- Failure/stop rule：任一工程门失败即在该边界停止；指标无论正负只访问一次，不删病例、不改图/标签/N4/阈值/TTA/epoch/bootstrap，不重跑或回流调参。
- 是否允许进入test：这是单独的preoperative cross-stage external confirmation，不授权任何旧test270/validation/outer/confirmation/UCSD重访，也不触发后续选择。

## 7. 成本与风险

- 训练时间：0新增；复用条件Full972模型。
- 峰值显存：预计每模型约20–23GiB；先合成smoke。
- 磁盘：双源约3,626,173,198 bytes；每cache3,795,845,120 bytes，双cache7,591,690,240 bytes；合计约11.22GB，全部放`/tmp`，执行前要求额外5GB余量。
- 推理forwards/延迟：362×(identity1+Flip8 8)×2模型=6516单例forward；部署每模型identity仍1×、Flip8为8×。
- 最大风险：362例术前分布与治疗后任务差异很大；N4全量CPU约1小时；TCIA无官方逐文件checksum，必须双独立下载；即使配对CI正向也只说明跨阶段压力测试，不代表post-treatment临床外推。

## 8. 最终结果（完成后填写）

- External：待条件资格。
- 置信区间/病例分布：362 scans/362 subjects；待执行。
- 是否支持原直觉：待执行。
- 学到了什么：待执行。
- 下一步：待执行。
- 报告/归档/哈希：待执行。

