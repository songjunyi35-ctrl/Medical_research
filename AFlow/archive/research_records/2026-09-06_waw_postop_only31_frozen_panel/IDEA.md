# Experiment idea: WAW术后only-patient31冻结面板

Status: planned; metadata/mask cohort frozen, large postop images/cache/checkpoint/performance not accessed.

## 1. 现在的问题是什么？

现有新外部Africa95、CPTAC33和WAW152均为术前，UCSD official-float是同75例来源纠正。WAW final postop中有31名未进入已见preop152的新患者，最早术后0–22天、29名WHO4级，WT `{1,2,3}`且切除腔4排除，直接贴近BraTS2024 post-treatment任务。

## 2. 一句话直觉

只拿从未考过的患者最早一张术后卷，不让同一人换时间点重复加权，测试冻结模型能否在另一家医院的同类题上迁移。

## 3. 为什么选择这个idea？

- Final160 postop exams/92人中先排57名preop已见患者；35名post-only中排4名日期全空，再对31人固定最早numeric day，选择完全不看模型/图像质量。
- 31/31 label1/2/3 WT非空、30例含resection bed4；语义比术前外部更贴近当前任务。
- Medical University of Warsaw不在BraTS2024七站；无patient bridge，只称institution-disjoint/no-known-overlap。
- MOTUM失败在source image support，CFB是GTV，UPENN follow-up无seg；均未运行这31例。
- 旧WAW preop结果没有参与checkpoint/view/阈值选择，面板沿用其性能前已冻结的Full identity+Flip8/GCSAM identity。
- 一手来源：[Zenodo final record](https://zenodo.org/records/22205631)。

## 4. 具体怎么做？

输入为31份coreg/skull-stripped float32 postop T1c；closest canonical RAS，图/label native shape与affine exact；N4 shrink2、image-only positive bbox、positive z-score、128³。WT=`raw label in {1,2,3}`，resection bed4为背景。

Full972 KD168一次循环输出identity与固定8 reflection逆翻float32 raw-logit均值；GCSAM184 identity。阈值0、Point=argmax，无SAM/后处理/model ensemble。31 scans=31 subjects，报告三绝对Dice/Point、两配对差与10k bootstrap。

## 5. 唯一变化与固定项

- 唯一变化：新增WAW post-only31、每人最早numeric postop exam。
- 固定：patient排重、病例/顺序、WT语义、N4/crop/z-score/128³、checkpoint、views/融合、threshold0、Point、无SAM/后处理、bootstrap。
- 不得换空日期/后续扫描补例或合并preop已见患者。

## 6. 成功、失败和停止规则

- GCSAM与Flip8各按delta>0且CI lower>0=`supported_on_WAW_postop31`，delta>0且跨0=`positive_but_uncertain`，delta≤0=`not_supported_or_reversed`；无promotion。
- 大包size/MD5/CRC、31 member、geometry/label/WT、首例与全cache byte exact、checkpoint/smoke/preaccess任一失败即模型前停止。
- 一次started后不重跑；结果不得回流任何已见cohort或新参数。

## 7. 成本与风险

- 训练0；大包3.09GiB、选中31 T1c预计约150MiB、cache约310MiB；279 forwards、预计数秒/5.1GiB。
- Flip8部署8×，identity1×。
- N=31区间宽；day0–22是早期post-resection而非所有长期post-treatment；同WAW机构但preop患者零重叠；无BraTS患者桥；publisher已做预处理后项目再N4；训练源parity未解。

## 8. 最终结果（完成后填写）

- External：Full identity/Flip8/GCSAM Dice=`0.591647/0.598959/0.593305`，Point=`15/15/14`。
- CI/病例：Flip8−identity=`+0.007312` CI `[-0.001925,+0.016626]`、21/10；GCSAM−Full=`+0.001657` CI `[-0.012498,+0.016205]`、17/14；两者均正向不确定。
- 直觉/成本/访问：没有确定赢家；0训练，cache141.9秒，279 forwards5.76秒/5.08GiB。只访问31例。
- 下一步：cohort关闭，不换患者/时间点/参数重测。
