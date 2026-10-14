# BraTS-Africa95冻结baseline结果

状态：✅ 一次性public post-challenge外部评测完成并关闭；不训练、不调参、不重跑。

## 初学者结论

把当前模型带到尼日利亚四个中心的低场、原始厚层MRI后，Dice从内部/UCSD约`0.75`降到`0.52–0.59`，说明跨设备、地区、治疗阶段和采集质量的域差异非常大。GCSAM184在这个队列的Dice明显优于Full972 KD168 identity：平均高`0.0686`，配对95%区间整体大于0，且80/95例改善；但自动点少命中3例。因此它是**Africa95总体和1×范围的Scan/Subject Dice点估计第1**，不是三项全面更好。

Full972 KD168固定Flip8比identity只提高Dice`0.00344`，区间跨0，Point多1。它是Africa95的Point点估计第1（88/95），代价8×；不能称Dice收益稳定跨域。

## 唯一变化与固定项

唯一新增是TCIA package948发布者`95_Glioma`目录：95 scans / 95 subjects、四个尼日利亚中心。51例OtherNeoplasms在任何像素/模型前按发布者类别排除。模型、checkpoint、all-positive术前WT、canonical RAS、N4 shrink2、image-only crop、source nonzero z-score、128³、阈值0、Full固定Flip8公式、GCSAM identity、无SAM/后处理/模型ensemble、指标和10,000次bootstrap均冻结。

原挑战60 train/15 validation/20 test现已全部公开seg；本项目没有用任何一部分训练或选择，合并一次性报告。它是公开post-challenge external evidence、not hidden blind。四中心与BraTS2024七个欧美来源站机构不同，但无patient-ID bridge，只称institution-disjoint/no-known overlap。

## 工程门

- 官方目录：146×5=730文件；固定Glioma95的T1CE+seg为190文件277,630,584 bytes/副本。
- 双官方传输190/190路径/bytes/SHA exact；source receipt SHA=`f5c129ef…`。
- 首例240×240×155、1mm、RAS，image/seg affine差0，label `{0,1,2,3}`。
- 全95例值域并集`{0,1,2,3}`、geometry/finite/integer/WT非空全部通过；cache image/label SHA=`f95959ac…/9e937ca9…`。
- 初版首例probe遗漏SimpleITK单线程，和worker最大差`4.48e-5`，byte门失败且无index/model。原cache保留不改；V2固定单线程后与worker首例image/label byte exact，finalizer重验95例源和memmap后才写index。没有放宽门或重算模型。
- 模型前收据SHA=`59f2ab8c…`；855 forwards后raw逐病例、bootstrap、cache和榜单由独立验证器重算通过。

## 绝对结果

| 方法 | Scan=Subject Dice（95% CI） | Point hit | forward成本 |
|---|---:|---:|---:|
| GCSAM184 identity | **`0.591619219`** `[0.540914,0.637409]` | 84/95 (`0.884211`) | `1×` |
| Full972 KD168 Flip8 | `0.526437660` `[0.475646,0.577047]` | **88/95 (`0.926316`)** | `8×` |
| Full972 KD168 identity | `0.523002550` `[0.475502,0.567739]` | 87/95 (`0.915789`) | `1×` |

每人只有一次扫描，所以Scan Dice=Subject Dice且使用同一95人bootstrap抽样，区间完全相同。Point只报告计数/比例。

## 冻结配对比较

- GCSAM identity − Full identity：Dice=`+0.068616669`，95% CI=`[+0.044471976,+0.092497676]`，Point=`−3`；80改善/13变差/2 tie。Dice方向明确，Point方向相反。
- Full Flip8 − identity：Dice=`+0.003435109`，95% CI=`[-0.010536166,+0.016814528]`，Point=`+1`；52改善/38变差/5 tie。区间跨0，不支持确定Dice收益。

## 为什么会这样

事后描述显示三个输出普遍低估WT体积：真实平均103,710 voxels；Full identity/Flip8/GCSAM平均预测50,536/49,649/62,255，分别91/92/90例小于真实体积。GCSAM圈得更大、更接近真实范围，可能解释Dice提高；但它的最高logit点少命中3例。该分析不授权调阈值。

四中心均值上GCSAM−Full identity Dice都是正：CRV/LASUTH/MEDHUB/NKDC=`+0.0635/+0.0555/+0.0727/+0.0718`，说明总体方向不由单一中心驱动；LASUTH仅5例，不作单中心推断。Flip8在CRV为`−0.0141`、其他三中心为正，与全95宽区间一致。

患者bootstrap没有把中心作为聚类层。事后敏感性中，GCSAM每次删掉一个中心后的总体差仍为`+0.0654–+0.0707`；四中心等权均值`+0.0659`。Flip8删NKDC后变为`−0.00134`，四中心重抽范围跨0。GCSAM方向不由单中心垄断，但只有4个已观察中心，不能外推所有非洲医院。[中心聚类敏感性](POSTHOC_CENTER_CLUSTER_SENSITIVITY.json)。

按真实WT体积排序的事后四分位显示：最小24例（800–51,587 voxels）GCSAM−Full仅`+0.0039`，其余三组为`+0.0820/+0.0969/+0.0926`；相对优势与log10目标体积Spearman ρ=`0.321`（未做多重校正，只作描述）。GCSAM相对Full丢4个Point、救回1个；小病灶仍是明显薄弱区。不得用这些已揭示病例选择体积router或阈值。[误差图与机器审计](POSTHOC_ERROR_AUDIT.json)、[PNG](POSTHOC_ERROR_AUDIT.png)。

同checkpoint在UCSD official-float的预测/真实体积比接近1，而Africa95中Full/GCSAM均值仅`0.543/0.608`；Africa95目标体积均值约是UCSD的2.47倍，预测均值只增加约1.21/1.48倍。治疗阶段、标签、设备和grid同时变化，故只能说明组合域差异，不定位单一因果。[跨cohort描述](CROSS_COHORT_UCSD_DESCRIPTION.json)。

只读checkpoint头检查也不支持“GCSAM只是把logit bias抬高”：GCSAM最后bias更负（`−0.4804` vs `−0.4551`），说明覆盖差异来自整条学习表示，不能靠手改bias复制。[head审计](CHECKPOINT_HEAD_AUDIT.json)。

从保存的intersection/predicted/target计数分解：Full identity平均precision/recall=`0.823/0.416`，GCSAM=`0.819/0.497`。GCSAM相对Full recall=`+0.0809`，事后95%区间`[+0.0548,+0.1070]`；precision=`−0.0038`且区间跨0。Flip8相对identity precision=`+0.0333`而recall仅`+0.0025`且区间跨0，并平均少预测887 voxels。故GCSAM Dice优势主要来自补回真实WT范围，Flip8更保守/精确但未稳定改善覆盖；这些不是预注册promotion检验。[precision/recall审计](POSTHOC_PRECISION_RECALL.json)。

用训练972目标分布而非Africa结果定义边界：Africa WT在训练经验分位的中位为`95.47%`，62/95超过train p90、54/95超过p95，1例超过train最大值。Train p75以上72例的GCSAM−Full=`+0.0893`，以下23例仅`+0.0040`；说明优势与训练罕见的大范围WT相伴。但阶段/标签/采集混杂，训练分位也不得变成已见Africa的router。[训练支持审计](POSTHOC_TRAIN_TARGET_SUPPORT.json)。

把预测体积也放回train经验分布：Africa target分位中位`95.47%`，Full/GCSAM预测仅`76.13%/85.70%`；真实62/95高于train p90，预测只有19/38例。GCSAM更能扩张，仍明显回缩到训练常见范围。这与学习到size prior相容，但不是因果，且术前all-positive与训练RC-excluded WT语义不同。[训练先验回缩审计](POSTHOC_TRAIN_PRIOR_REGRESSION.json)。

旧GCSAM fixed184内部双复制失败否定的是“当前优化配方稳定可复制”，不等于这个既有checkpoint在每个外部域都差。Africa95不能回流复活训练方法或选择新checkpoint；它只显示该冻结checkpoint在这一域的区域覆盖更稳。

还要强调这不是optimizer因果比较：GCSAM checkpoint只用864 fit并由108 inner选epoch184，Full checkpoint用972 fit、固定epoch168，二者seed也不同；optimizer、训练病例数、epoch/选择同时变化。Africa配对只保证同95病例输入/指标，不保证训练单因素。论文的2D分类generalization主张与观察相容，但不能据此说GCSAM造成了`+0.0686`。[因果边界审计](../../reviews/2026-09-06_gcsam_external_interpretation/RESULT.md)。

## 成本、访问与边界

- 训练0；两个checkpoint均未改。
- 双源约533MiB，cache约951MiB，均在`/tmp`。主cache从文件birth到最后worker写入约783秒；finalizer39.1秒。
- 正式855 forwards耗时17.147秒，进程峰值5,457,695,744 bytes（约5.08GiB）。部署GCSAM/Full identity为1×，Full Flip8为8×。
- 原始1.5T临床数据常有3–5mm层厚；发布者重采样1mm不能恢复真实细节，本项目又rasterize到128³。绝对分数只在本协议内解释。
- Africa95访问=true；internal test270、fixed validation108、outer、confirmation、UCSD、UTSW、RHUH均未访问。
- 两checkpoint训练数据来自本地BraTS2024 Kaggle mirror；官方Synapse corrected-v2 zip/member字节parity仍未证明。Africa package948外部源为官方且双传输exact，但不会反向消除checkpoint训练源caveat。
- 本cohort关闭：不得调阈值/N4/crop/view/权重/病例或重跑。Leaderboard按Africa95独立小节更新，不和UCSD/test270混排。

## 证据

- [七问](IDEA.md)、[预注册](PREREGISTRATION.json)
- [来源结果](../../reviews/2026-09-06_brats_africa_source_audit/RESULT.md)、[冻结cohort](../../reviews/2026-09-06_brats_africa_source_audit/FROZEN_GLIOMA95_COHORT.json)
- [双源收据](SOURCE_DOWNLOAD.json)，SHA-256 `f5c129efcf073f24f39f94129e85f3e81c5614f67eb522ead651feac9a30324f`
- [模型前收据](MODEL_PREACCESS.json)，SHA-256 `59f2ab8c746fb57d4480ebe105975bf5e2e53a4111b51370b72d0d299b0e45e8`
- [raw report](RAW_EXTERNAL95.json)，SHA-256 `c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3`
- [started](RAW_EXTERNAL95_STARTED.json)，SHA-256 `7591379287b649b16b29de9a86d899338995e4f0d8d825b66942dae35ebb39d6`
- [cache index](CACHE_INDEX_V2.json)，SHA-256 `1493bfde0e82309c3bfc73e2b2b9e1ab631921a6aeb988288f16ddbd9cba1c2e`
- [终态独立验证V3](VERIFICATION_RECEIPT_V3.json)，SHA-256 `18aa6752c1a38330dcd1ceb1f98f778d2f5bfabcae65a2592f8d3caf1125329b`；完成时重新hash双源380文件、两个checkpoint、evaluator/raw/cache/榜单。V2保留历史。
- [中心来源更正](CENTER_PROVENANCE_CORRECTION.json)、[中心描述](POSTHOC_CENTER_DESCRIPTION.json)、[体积描述](POSTHOC_VOLUME_DESCRIPTION.json)
- [误差审计图](POSTHOC_ERROR_AUDIT.png)、[跨cohort体积描述](CROSS_COHORT_UCSD_DESCRIPTION.json)、[checkpoint头审计](CHECKPOINT_HEAD_AUDIT.json)
- [precision/recall分解](POSTHOC_PRECISION_RECALL.json)
- [中心聚类/leave-one-center-out敏感性](POSTHOC_CENTER_CLUSTER_SENSITIVITY.json)
- [训练目标支持范围审计](POSTHOC_TRAIN_TARGET_SUPPORT.json)
- [预测向训练size prior回缩描述](POSTHOC_TRAIN_PRIOR_REGRESSION.json)
- Full972 checkpoint：[checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth)，SHA `f7df2a16…`
- GCSAM184 checkpoint：[checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth)，SHA `acf2fce1…`
