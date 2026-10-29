# AFlow BraTS T1ce/WT Leaderboard

更新时间：2026-10-29 UTC

<!-- RACE_V2_STAGE_20261029 -->
## 2026-10-29新阶段最终270完成

原270/119、单T1ce、原生1mm，单模型Scan0.788697343／Subject0.775536380／Point251/270；纯SegResNet双模型Scan0.789070115／Subject0.776314624／Point254/270。Scan>0.8000分别为False/False。新增训练80轮／9760更新，新优化器微调，训练选中best为新增0轮（0表示未胜初始化）；实际部署权重以结果的固定候选为准。最终两个配方先108冻结，继承test-tuned历史，非独立确认。没有新outer/confirmation，本轮停止，不自动扩搜/续训/重测。[完整结果](archive/research_records/2026-10-29_race_v2_stage/RESULT.md)。
<!-- END_RACE_V2_STAGE_20261029 -->

这是项目唯一的快速排行榜入口。详细实验解释、因果边界和哈希以各自归档为准。不同cohort、数据角色、定位权限和交互协议不得混排。

2026-09-05来源补记：最近Flip8 A/B与镜像一致训练C/D的“fresh inner”仅指学生fit隔离及新划分；固定history/EMA教师的756例训练集与四个inner分别重叠87/72/88/72 scans（39/33/39/37 subjects），所以不具备教师—学生全链路患者隔离。存在间接信息泄漏风险，偏差大小尚未量化；不改变已保存测试点估计/区间，也不撤销UCSD75与项目患者差集结论。[清单审计](archive/reviews/2026-09-05_30h_chain_audit/TEACHER_INNER_EXPOSURE.json)。

所有继承checkpoint还有共同训练源限制：本地BraTS2024来自Kaggle mirror，官方Synapse corrected-v2 zip/member字节parity尚未完成；UCSD/Africa/CPTAC/WAW外部源核验不会反向消除这个caveat。[来源复查](archive/reviews/2026-09-05_brats2024_source_provenance_recheck/RESULT.md)。

<!-- CHINESE_METHOD_GUIDE -->
## 方法名怎么理解

- **基础模型**：正常训练，直接分割肿瘤。
- **蒸馏**：让学生模型学习老师模型的预测。
- **权重平滑**：把训练过程中各步的参数逐渐平均，尝试减少波动。
- **八镜像预测**：只在推理时翻转出8个视图，预测还原后平均；不为此重训。
- **预训练**：先用公开334例学习，再用原972例微调。
- **原生／不缩放**：保持原始体素尺寸；**分块**则是一块一块处理影像。
- **预测融合**：把多个模型的输出合起来；**参数平均**则是直接平均模型权重，两者不同。

“第几轮”指保存权重时的训练轮次；“重复甲／乙”是同一方案的两次不同随机训练。名字只作中文展示，原实验编号仍保留在机器排行榜和归档中。不同栅格、训练集和内部／外部队列仍须分开比较。
<!-- END_CHINESE_METHOD_GUIDE -->

## 指标定义

- **Scan Dice**：每个scan的Dice等权平均；多时间点subject会贡献多行。
- **Subject Dice**：先在每个subject内平均scan Dice，再对subject等权平均。
- **Point hit**：plain locator全体素最高raw-logit坐标是否落在GT WT内。
- `1×/2×/8×`表示locator forward数量或近似体素成本，不包含NIfTI预处理。

## External UCSD-PTGBM75：同一患者隔离cohort的两种输入来源版本

全75例官方直接来源核验显示：150/150几何一致、75/75源标签数组相等；官方图像全部float32、镜像全部uint8、0/75源图像数组相等，共18,156个官方非零体素被量化成零。[完整来源更正](archive/reviews/2026-09-05_30h_chain_audit/UCSD_SOURCE_RESULT.md)。2026-09-06按提前冻结的“只换影像来源”协议重建官方float32 cache并一次性运行同两模型；最终75/75 crop与128³标签仍exact，128³影像0/75 exact。

历史两行明确保留为Pittsburgh Fiber Data Hub `data-nih/tcia`整理的**mirror-uint8**版本；新增两行是TCIA package1019直接取得的**official-float32**版本。两种版本是相同75 scans / 53 subjects、相同源标签和相同模型，不是两个独立外部队列；official-float32纠正是在旧结果揭示后对同一cohort的重复访问，必须标为reused/not pristine blind，不能把它称作新的外部确认或用于调参。[官方映射审计](archive/reviews/2026-09-05_30h_chain_audit/UCSD_OFFICIAL_CLINICAL_MAPPING_AUDIT.json)、[来源纠正预注册](archive/research_records/2026-09-06_ucsd_official_float_source_correction/IDEA.md)。

这是独立于下面internal test270的外部排名：TCIA官方主release原有136 subjects/184 studies；用官方`participants.tsv`映射后，109 scans/83 subjects已经出现在本项目1,350例BraTS union中，全部按患者剔除。这里仅保留75 scans/53 subjects，case/subject overlap均为0；官方另列的42-subject/59-study BraTS challenge test package未使用。两模型均在标签访问前冻结，外部结果没有用于调checkpoint、阈值、N4、crop、病例、ensemble或后处理。这是公开外部数据的一次性确认，不是服务器隐藏blind challenge，也不能和internal test270按数值混排。

75例WT均非空，因此Point hit分母为75。原图均为256³、1mm各向同性，两版本统一经canonical RAS、N4 shrink2、image-only brain crop缩至128³；这里存在共同的128³标签rasterization caveat。训练用BraTS副本是182×218×182，故external tensor不能视为internal cache的简单复制。

| 排名 | 方法 | 扫描平均Dice（95%置信区间） | 患者平均Dice（95%置信区间） | 定位命中 | 计算量 | 状态 |
|---:|---|---:|---:|---:|---:|---|
| **1** | **改进优化器（864例，184轮）；官方浮点影像** | **`0.755889505`** `[0.717191,0.789575]` | **`0.755311886`** `[0.707990,0.796823]` | 70/75 (`0.933333`) | `1×` | 官方来源版Scan/Subject点估计第1；同cohort重复纠正，非新确认 |
| 2 | 改进优化器（864例，184轮）；镜像版8位影像 | `0.755647338` `[0.717080,0.789172]` | `0.755151670` `[0.707873,0.796526]` | **71/75 (`0.946667`)** | `1×` | 历史8位镜像行保留；四行中Point点估计最高 |
| 3 | 972例蒸馏模型（168轮）；官方浮点影像 | `0.746255043` `[0.706178,0.782003]` | `0.740164863` `[0.688098,0.784670]` | 70/75 (`0.933333`) | `1×` | 官方来源版；同cohort重复纠正，非新确认 |
| 4 | 972例蒸馏模型（168轮）；镜像版8位影像 | `0.745554128` `[0.704767,0.781595]` | `0.739054558` `[0.686028,0.784535]` | 70/75 (`0.933333`) | `1×` | 历史8位镜像行保留 |

official-float32内GCSAM − Full972 KD：Scan `+0.009634462`、Subject `+0.015147023`、Point `0/75`；paired 95% CI分别为`[-0.002163,+0.025053]`/`[-0.000553,+0.035820]`。Dice点估计偏向GCSAM，但两个CI仍跨0，不能宣称显著。

同模型official-float32 − mirror-uint8：GCSAM Scan/Subject/Point=`+0.000242167/+0.000160216/−1`，paired 95% CI=`[-0.001168,+0.001658]`/`[-0.000869,+0.001170]`；Full972 KD=`+0.000700915/+0.001110305/0`，CI=`[-0.000863,+0.002658]`/`[-0.000735,+0.003806]`。两模型的来源Dice区间都跨0，说明本次量化影响点估计很小且不确定；这不等于所有预处理或模型都对量化不敏感。

证据：[official-float raw](archive/research_records/2026-09-06_ucsd_official_float_source_correction/RAW_OFFICIAL_FLOAT.json)、[cache比较](archive/research_records/2026-09-06_ucsd_official_float_source_correction/CACHE_COMPARISON.json)、[模型前收据](archive/research_records/2026-09-06_ucsd_official_float_source_correction/MODEL_PREACCESS.json)、[mirror-uint8 raw](workspace/BraTS/ucsd_ptgbm_external_confirmation_20260902/UCSD_EXTERNAL_CONFIRMATION.json)、[原结果归档](archive/research_records/2026-09-02_ucsd_ptgbm_external_confirmation/RESULT.md)。

## External BraTS-Africa95：尼日利亚四中心公开post-challenge专家胶质瘤

TCIA package948公开146例×T1/T1CE/T2/FLAIR/seg；按发布者目录在任何像素/模型前固定只取`95_Glioma`，排除51例脑膜瘤、转移瘤等`OtherNeoplasms`。TCIA页面的collection级描述称六中心、challenge论文称七中心，但package datainfo对实际95例逐ID映射只有四中心：NKDC48、CRV28、MEDHUB14、LASUTH5；canonical结果以这个子集直接证据为准。四中心与BraTS2024 post-treatment七个欧美来源站机构不同；没有跨库patient-ID bridge，因此只称institution-disjoint/no-known overlap，不称ID证明的patient-disjoint。[中心来源更正](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/CENTER_PROVENANCE_CORRECTION.json)

这些病例原属于2023 challenge的60 train/15 validation/20 test，TCIA现已公开全部95份专家标签。本项目在任何payload前冻结三行、全95例和协议，不用其中任何部分训练/选择，合并报告为public post-challenge external95；不是隐藏blind test。标签由nnU-Net预标后，经有经验的board-certified neuroradiologist人工修订。术前WT取所有正标签；与post-treatment训练时排除resection cavity的语义不同。

| 排名 | 方法 | 扫描与患者平均Dice（95%置信区间） | 定位命中 | 计算量 | 状态 |
|---:|---|---:|---:|---:|---|
| **1** | **改进优化器（864例，184轮）；单次预测** | **`0.591619219`** `[0.540914,0.637409]` | 84/95 (`0.884211`) | `1×` | 1× Dice点估计第1；相对Full identity的paired CI整体为正，但Point少3 |
| 2 | 972例蒸馏模型（168轮）；八镜像预测 | `0.526437660` `[0.475646,0.577047]` | **88/95 (`0.926316`)** | `8×` | Point点估计第1；相对identity Dice小正但CI跨0 |
| 3 | 972例蒸馏模型（168轮）；单次预测 | `0.523002550` `[0.475502,0.567739]` | 87/95 (`0.915789`) | `1×` | 1×基线 |

GCSAM identity − Full identity：Dice=`+0.068616669`，paired 95% CI=`[+0.044472,+0.092498]`，Point=`−3`；80/95例Dice改善、13例变差、2例tie。这里Dice方向很清楚，但Point目标不同，不能称三项全面更好。

不用于选择的中心描述显示GCSAM−Full identity Dice在CRV/LASUTH/MEDHUB/NKDC均为正：`+0.0635/+0.0555/+0.0727/+0.0718`，说明总体方向不是单一中心贡献；小中心LASUTH只有5例，不单独作推断。[中心描述](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/POSTHOC_CENTER_DESCRIPTION.json)

Full Flip8 − identity：Dice=`+0.003435109`，paired 95% CI=`[-0.010536,+0.016815]`，Point=`+1`；52例改善、38例变差、5例tie。固定8镜像的点估计收益没有在这个域形成确定Dice证据。

原始1.5T临床扫描常有3–5mm层厚，发布者重采样到1mm不能恢复丢失细节；本项目又做image-support crop和128³ rasterization。故本节只在Africa95同协议内排序，不与UCSD75或internal test270按数值混排。

证据：[raw report](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/RAW_EXTERNAL95.json)、[七问](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/IDEA.md)、[模型前收据](archive/research_records/2026-09-06_brats_africa95_frozen_baselines/MODEL_PREACCESS.json)、[来源inventory](archive/reviews/2026-09-06_brats_africa_source_audit/OFFICIAL_INVENTORY.json)。

## External CPTAC33：Africa-informed、BraTS2021公开Training标签确认

TCIA官方映射给出39位CPTAC-GBM患者；public Training33每例有T1CE+人工三子区seg，Validation6只有四影像、0 seg，故在payload/模型前按标签公开性排除。Package636官方`.sums`共407,245条MD5，选中66文件全部覆盖；两份独立下载又66/66 size/MD5/SHA exact。

这些标签在原BraTS2021属于Training，但本项目两个模型只使用BraTS2024，故是external-to-this-project的公开标签确认；不是原challenge validation/test或hidden blind。CPTAC/CHOP-UPenn来源不在BraTS2024七站，但无patient-ID bridge，只称source/institution-disjoint no-known overlap。假设在看到CPTAC payload前由Africa结果明确冻结：GCSAM−Full Dice方向能否复现。

| 排名 | 方法 | 扫描与患者平均Dice（95%置信区间） | 定位命中 | 计算量 | 状态 |
|---:|---|---:|---:|---:|---|
| **1** | **972例蒸馏模型（168轮）；八镜像预测** | **`0.844577117`** `[0.813290,0.870795]` | **33/33 (`1.0`)** | `8×` | 相对identity `+0.003246`，paired CI整体为正 |
| 2 | 改进优化器（864例，184轮）；单次预测 | `0.843285080` `[0.812977,0.868437]` | **33/33 (`1.0`)** | `1×` | Africa-informed方向仅弱支持：相对Full小正但CI跨0 |
| 3 | 972例蒸馏模型（168轮）；单次预测 | `0.841331464` `[0.809100,0.868613]` | **33/33 (`1.0`)** | `1×` | 1×基线 |

GCSAM−Full identity=`+0.001953616`，paired CI=`[-0.006646,+0.012384]`，16/17/0 improve/worse/tie，Point差0。按预注册是弱/不确定方向支持，不复现Africa的`+0.0686`幅度或确定性。

Full Flip8−identity=`+0.003245653`，paired CI=`[+0.000828,+0.005855]`，23/10/0 improve/worse/tie，Point差0。它在这份小队列独立支持固定Flip8 Dice方向；仍不授权新TTA权重/视图或再测。

33人样本小、绝对CI宽；术前WT、公开Training身份和128³ rasterization均需保留。三行只在CPTAC33内排序，不与Africa/UCSD/internal数值混排。

证据：[raw](archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/RAW_CPTAC33.json)、[七问](archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/IDEA.md)、[V2模型前收据](archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/MODEL_PREACCESS_V2.json)、[来源cohort](archive/reviews/2026-09-06_cptac_brats2021_source_audit/FROZEN_TRAINING33_COHORT.json)。

## External WAW-GliomAI152：华沙公开术前人工修订标签

Zenodo final record `22205631`于2026-09-01公开、CC BY 4.0，来自Medical University of Warsaw。发布描述写156份术前检查，但final coreg metadata与preop mask严格交集实际为152名患者/152 scans；以最终文件为准，不补找4例。152/152均有publisher命名为delayed post-contrast T1的`t1c`及专家人工复核mask，原生240×240×155、1mm³、图/标签affine exact，label union `{0,1,2,3}`；WHO grade2/3/4=`44/34/74`。

华沙机构不在BraTS2024的Duke/UCSF/Missouri/UCSD/Heidelberg/Michigan/Indiana七站，只称institution-disjoint/no-known-overlap；没有跨库patient-ID桥，不称ID证明独立。数据是public external-to-project，不是hidden blind。发布者已做coreg/skull-strip/bias correction，项目仍按冻结canonical N4、image-support crop和128³ rasterization；且术前all-positive WT、delayed T1c与post-treatment训练的阶段/命名不同，只能在本cohort内比较。

| 排名 | 方法 | 扫描与患者平均Dice（95%置信区间） | 定位命中 | 计算量 | 状态 |
|---:|---|---:|---:|---:|---|
| **1** | **改进优化器（864例，184轮）；单次预测** | **`0.693601158`** `[0.660656,0.723433]` | 140/152 (`0.921053`) | `1×` | Dice点估计第1；仅比Flip8高`0.000107`，相对Full identity CI跨0 |
| 2 | 972例蒸馏模型（168轮）；八镜像预测 | `0.693494112` `[0.660273,0.723615]` | **143/152 (`0.940789`)** | `8×` | Point第1；相对identity Dice区间全正 |
| 3 | 972例蒸馏模型（168轮）；单次预测 | `0.683377992` `[0.649811,0.713659]` | 141/152 (`0.927632`) | `1×` | 1×基线 |

GCSAM identity−Full identity=`+0.010223166`，paired 95% CI=`[-0.003005,+0.023481]`，84/63/5 improve/worse/tie，Point=`−1`。按预注册是positive-but-uncertain：不复现Africa95的`+0.0686`确定优势，也不是反向。

Full Flip8−identity=`+0.010116120`，paired 95% CI=`[+0.004472,+0.016274]`，93/54/5，Point=`+2`。这是internal270、CPTAC33之后第三处Scan配对区间支持；部署仍8×，不能据四个已见cohort搜索视图/权重/阈值。

证据：[raw](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RAW_WAW152.json)、[七问](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/IDEA.md)、[V2模型前收据](archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/MODEL_PREACCESS_V2.json)、[来源审计](archive/reviews/2026-09-06_waw_gliomai_source_audit/RESULT.md)。

## External WAW-postop-only31：与preop已见患者零重叠的早期术后队列

WAW final postop metadata/mask实际160 scans/92 patients（描述161）。先按同release ID排除57名已在preop152的患者；35名postop-only中要求`days_postop`明确，再每人固定最早一张，得到31 patients/scans、day0–22，preop patient overlap=0。4名日期全空患者、17张后续numeric scans均在性能前排除。31人中29名WHO4级；WT严格为labels1/2/3，30例含但排除resection-bed4。

这是同一Warsaw机构的新患者、public early post-resection external-to-project，不是hidden blind。相对BraTS2024七站institution-disjoint但无跨库patient bridge；day0–22不能代表长期复发。Publisher coreg/skull-strip/bias-correct1mm后项目再N4/crop/128³，仍有双重预处理与rasterization caveat。

| 排名 | 方法 | 扫描与患者平均Dice（95%置信区间） | 定位命中 | 计算量 | 状态 |
|---:|---|---:|---:|---:|---|
| **1** | **972例蒸馏模型（168轮）；八镜像预测** | **`0.598959388`** `[0.532228,0.662012]` | **15/31 (`0.483871`)** | `8×` | Dice点估计第1；相对identity正但CI跨0 |
| 2 | 改进优化器（864例，184轮）；单次预测 | `0.593304725` `[0.527500,0.654293]` | 14/31 (`0.451613`) | `1×` | 最高1× Dice点估计；相对Full正但CI跨0 |
| 3 | 972例蒸馏模型（168轮）；单次预测 | `0.591647416` `[0.522240,0.654848]` | **15/31 (`0.483871`)** | `1×` | 1× Point点估计第1（与Flip8同） |

Flip8−identity=`+0.007311972`，paired CI=`[-0.001925,+0.016626]`，21/10/0 improve/worse/tie，Point差0；GCSAM−Full=`+0.001657309`，CI=`[-0.012498,+0.016205]`，17/14/0，Point−1。两者按预注册均positive-but-uncertain。N=31绝对CI宽，不能把点估计排名写成确定优越。

证据：[raw](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RAW_WAW_POSTOP31.json)、[七问](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/IDEA.md)、[模型前收据](archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/MODEL_PREACCESS.json)、[来源](archive/reviews/2026-09-06_waw_postop_only31_source_audit/RESULT.md)。

## External UCSD-BraTS2024test56：公开的BraTS-GLI 2024 test包（UCSD部分）

TCIA UCSD-PTGBM集合Version 2（2026-01-22）公开了原BraTS-GLI 2024隐藏test中UCSD贡献的package 1221（59 studies/42 subjects），Version 3（2026-03-13）修正了临床表患者ID。按官方v3映射的any-timepoint规则，仅`UCSD-PTGBM-0038`（`BraTS-GLI-02289`）与项目1,350例训练union重叠并在像素前排除；冻结**56 scans / 41 subjects**，与union（613 subjects）及此前已评UCSD75（53 subjects）患者零重叠。包内无`.sums`，传输门为双独立官方传输112文件逐文件SHA-256一致。

身份定位：public post-challenge、**非hidden blind**；有官方patient-ID桥、patient-disjoint已证明；但UCSD是BraTS2024七个来源站之一且与UCSD75同机构，**不是institution-disjoint**——这是"近训练域的新患者"证据，不是跨机构泛化证据。治疗后WT=labels`{1,2,3}`、排除切除腔`4`，与训练完全同口径；原生256³/1mm，经canonical N4、image-only crop与128³ rasterization。

| 排名 | 方法 | 扫描平均Dice（95%置信区间） | 患者平均Dice（95%置信区间） | 定位命中 | 计算量 | 状态 |
|---:|---|---:|---:|---:|---:|---|
| **1** | **972例蒸馏模型（168轮）；八镜像预测** | **`0.732973`** `[0.693045,0.771526]` | **`0.733625`** `[0.694818,0.771538]` | **56/56 (`1.0`)** | `8×` | 三项点估计第1；相对identity CI跨0 |
| 2 | 972例蒸馏模型（168轮）；单次预测 | `0.730002` `[0.690626,0.766791]` | `0.730740` `[0.692275,0.767211]` | **56/56 (`1.0`)** | `1×` | 最高1×；Point全中为全部外部cohort最高 |
| 3 | 改进优化器（864例，184轮）；单次预测 | `0.717689` `[0.671931,0.761285]` | `0.715545` `[0.667478,0.760680]` | 52/56 (`0.928571`) | `1×` | 负点估计；原配对CI略负但Scan数值精度敏感，见下文 |

Flip8−identity=`+0.002971109`，scan配对CI `[-0.002740,+0.008750]`、subject配对CI `[-0.002830,+0.008735]`，32/24/0改善/变差/tie，Point差0：第六个Scan点估计为正的cohort，本队列区间跨0，仍不能称显著。

**GCSAM−Full identity=`−0.012312488`（Subject `−0.015194370`），原冻结scan配对CI `[-0.025822,-0.000092]`、subject配对CI `[-0.031507,-0.000268]`均略低于0，21/35/0，Point`−4`。** 点估计方向与Africa不同；但不再称两项“稳健确定劣于”。2026-09-07事后数值精度审计中20组10,000次Scan区间有17组跨0，固定百万次Scan CI=`[-0.025714,+0.000140]`；Subject有4/20跨0，百万次CI=`[-0.031763,-0.000223]`。原区间和排名保持，补充结论为“负点估计、原区间贴零，Scan显著性对Monte Carlo精度敏感”；不选择seed替换主报告，不回流模型/阈值选择。[数值精度审计](archive/reviews/2026-09-07_ucsd56_bootstrap_precision/RESULT.md)。

证据：[结果](archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/RESULT.md)、[raw](archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/RAW_UCSD_BRATS2024TEST56.json)、[七问](archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/IDEA.md)、[预注册](archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/PREREG.md)、[模型前收据](archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/MODEL_PREACCESS.json)、[终态验证](archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/VERIFICATION_RECEIPT.json)、[来源审计](archive/reviews/2026-09-06_ucsd_brats2024test_source_audit/RESULT.md)。

## Reused internal test270：全自动direct locator

同一270 scans /119 subjects，按Scan Dice降序。复用内部测试，非pristine blind、外部确认或公开SOTA；legacy160³栅格限制仍保留。**2026-09-20起新增原始1mm栅格评分行，其分数与128³栅格行不可直接相减，排名仅按原始数值。**

| 排名 | 方法 | 扫描平均Dice | 患者平均Dice | 定位命中 | 计算量 | 状态 |
|---:|---|---:|---:|---:|---:|---|
| 1 | 不缩放影像的基础模型（173轮，原生栅格） | `0.768577897` | `0.754143911` | 262/270 (`0.970370`) | `1×`（体素量2.5×） | native grid；仅与同栅格行比较；同栅格基线193为`0.761134`，配对差+0.007444、CI[-0.0006,+0.0143]；配对区间跨0，不确定 |
| 2 | 公开数据预训练＋不缩放影像（166轮） | `0.766271575` | `0.750958477` | 236/270 (`0.874074`) | `1×` | 用户授权未过验证门探索；对native173差-0.002306，95%CI[-0.009078,+0.005791]；不跨栅格相减 |
| 3 | 预训练模型（179轮，预测还原到原生栅格） | `0.764544537` | `0.748758964` | 250/270 (`0.925926`) | `1×` | 同一已测试模型的不同评分栅格；非第三训练候选，不能把重评分差当方法增益 |
| 4 | 334例预训练后微调（179轮，128³） | `0.761648736` | `0.745858610` | 246/270 (`0.911111`) | `1×` | 当前128³组总体及1× Dice最高点估计；未过验证提升门；对193差+0.002470，95%CI[-0.004343,+0.009627]，非显著胜出 |
| 5 | 预训练＋每位患者同等权重（172轮） | `0.759967225` | `0.745646555` | 250/270 (`0.925926`) | `1×` | 用户授权未过验证门探索；对pretrained179差-0.001682，95%CI[-0.008389,+0.005860]；不跨栅格相减 |
| 6 | 固定基础模型（972例，第193轮） | `0.759178781` | `0.744984623` | 242/270 (`0.896296`) | `1×` | 用户固定比较基线；历史128³ Dice纪录；复用内部；无蒸馏/TTA，Point非最高 |
| 7 | 先分肿瘤细类、再合并（158轮） | `0.757718039` | `0.743434022` | 238/270 (`0.881481`) | `1×` | 未达到预设验证提升门；复用内部；对193差-0.001461，95%CI[-0.006823,+0.003567] |
| 8 | 972例蒸馏训练＋八镜像预测（168轮） | `0.757269662` | `0.741253585` | `0.981481` | `8×` | 历史状态：两个fresh inner全门后唯一一次post-test/reused评测；三项点估计第1；非盲测/外部确认 |
| 9 | 改进优化器（864例，第184轮） | `0.752003176` | `0.737266420` | `0.933333` | `1×` | 历史状态：用户授权failed-gate探索性；1× Scan点估计第1；非Full972 final，CI跨0且Point低 |
| 10 | 972例蒸馏训练（168轮） | `0.751759537` | `0.737327107` | `0.970370` | `1×` | 历史状态：用户授权failed-gate探索性；1× Subject/Point最高 |
| 11 | 684例训练·八镜像预测（168轮，重复甲） | `0.748065184` | `0.736042143` | 262/270 (`0.970370`) | `8×` | 本次固定面板；复用非盲；完整结果/CI见报告 |
| 12 | 684例训练·八镜像预测（168轮，重复乙） | `0.746917944` | `0.732314896` | 258/270 (`0.955556`) | `8×` | 本次固定面板；复用非盲；完整结果/CI见报告 |
| 13 | 双模型预测融合（原模型75%＋平滑模型25%） | `0.745054064` | `0.727344609` | `0.955556` | `2×` | 历史状态：旧总体最高 |
| 14 | 684例训练·精简多视图预测（168轮，重复甲） | `0.744795478` | `0.733333687` | 257/270 (`0.951852`) | `2×` | 本次固定面板；复用非盲；完整结果/CI见报告 |
| 15 | 684例训练·精简多视图预测（168轮，重复乙） | `0.744617935` | `0.729756966` | 258/270 (`0.955556`) | `3×` | 本次固定面板；复用非盲；完整结果/CI见报告 |
| 16 | 逐体素蒸馏单模型（200轮） | `0.742976353` | `0.725178533` | `0.955556` | `1×` | 历史状态：旧最高单模型 |
| 17 | 旧基础模型（128³，第168轮） | `0.741203949` | `0.723179486` | `0.929630` | `1×` | 历史状态：旧单模型记录 |
| 18 | 684例训练·单次预测（168轮，重复乙） | `0.740868264` | `0.725770935` | 252/270 (`0.933333`) | `1×` | 本次固定面板；复用非盲；完整结果/CI见报告 |
| 19 | 训练权重平滑（168轮） | `0.740696468` | `0.724904822` | `0.948148` | `1×` | 历史状态：Dice持平、point更高 |
| 20 | 684例训练·单次预测（168轮，重复甲） | `0.739987540` | `0.728200802` | 252/270 (`0.933333`) | `1×` | 本次固定面板；复用非盲；完整结果/CI见报告 |
| 21 | 更大输入模型（756例，160³，第200轮） | `0.737060297` | `0.719728630` | `0.874074` | `~1.95×` | 历史状态：负结果；160³标签网格 |
| 22 | 早期基础模型（128³，第99轮） | `0.725820068` | `0.709412503` | `0.900000` | `1×` | 历史状态：2026-08-17记录 |

2026-09-20同栅格比较：原始分辨率e173与原193基线都在原始1mm裁剪栅格上评分，Scan 0.768578 对 0.761134，患者配对差+0.007444、95%CI[-0.000608,+0.014305]，Subject差+0.007177、Point+15，177胜93负。配对区间跨0，不确定。基线在128³栅格重算0.759179与记录0.759179一致。[本次完整结果](archive/research_records/2026-09-20_segresnet_native_res_e173_test270/RESULT.md)。

2026-09-17历史时点的同test270/119自动系统总体及单次前向Scan/Subject最高点估计：纯SegResNet原972 epoch193，**0.759179/0.744985**，仅1次前向。Scan95%CI [0.732638,0.781993]；Point242/270非最高。相对旧Flip8 Scan+0.001909的患者配对95%CI [-0.004567,+0.009086]跨0，不称显著胜出。最高Point仍为旧Flip8的265/270，最高1×Point仍为旧Full972 KD的262/270。[本次完整结果](archive/research_records/2026-09-17_segresnet_plain972_e193_test270/RESULT.md)。

2026-09-21当前同128³评分组：公开334例预训练后微调179的Scan0.761649、Subject0.745859，为复用内部270/119自动总体及单次前向Dice最高点估计；对固定193的Scan差+0.002470、患者配对95%CI[-0.004343,+0.009627]跨0，未证明稳定优越，验证提升门未通过。Point246/270并非最高。不同原始栅格的0.768578不可直接相减。[完整结果](archive/research_records/2026-09-21_segresnet_external_pretrain/RESULT.md)。

以下旧关键增益保留历史时点，不作为当前最高声明；新结果不触发重测/调参。

关键增益：

- Flip8 − same-checkpoint identity：scan `+0.005510124`、subject `+0.003926478`、Point `+3/270`；paired 95% CI为`[+0.002751,+0.008022]`/`[-0.000379,+0.007354]`。Scan区间整体为正，Subject区间跨0。相对此前GCSAM Scan榜首，三项点估计为`+0.005266486/+0.003987165/+13 points`，但未做冻结的二者配对检验，不能称相对GCSAM显著。绝对CI按pretest冻结seed20260817从保存的per-case行离线修正；raw helper的seed偏差已保留归档。

- GCSAM − Full972 KD：scan `+0.000243638`、subject `-0.000060687`、point `-0.037037`（少10/270）；scan/subject paired 95% CI为`[-0.005655,+0.005740]`/`[-0.009153,+0.008094]`，均跨0。GCSAM只在Scan点估计上排第1，不能称显著或三指标全面最佳。

- Full972 − ensemble：scan `+0.006705473`、subject `+0.009982499`、point `+0.014815`；scan/subject paired 95% CI下界均大于0。
- Full972 − KD200：scan `+0.008783184`、subject `+0.012148574`、point `+0.014815`；scan/subject paired 95% CI下界均大于0。

- ensemble − historical168：scan `+0.003850115`、subject `+0.004165123`、point `+0.025926`；scan/subject paired CI下界均大于0。
- voxel-KD − historical168：scan `+0.001772404`、subject `+0.001999047`、point `+0.025926`；Dice paired CI跨0。
- voxel-KD相对ensemble少scan/subject `0.002078/0.002166`，但推理从`2×`恢复`1×`。

报告入口：

- Flip8：[报告](workspace/BraTS/sammed3d_t1ce_wt/flip8_tta_stability_20260904/FLIP8_TEST270.json)，[归档](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.md)，[修正记录](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/BOOTSTRAP_CORRECTION.json)

- GCSAM：[报告](workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/GCSAM_INNER_E184_TEST270.json)，[归档](archive/research_records/2026-08-31_gcsam_full972_stability/RESULT.md)

- Full972：[报告](workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/FULLTRAJECTORY_TEST270.json)，[归档](archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md)

- Ensemble：[报告](workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/LOGIT_ENSEMBLE_TEST270.json)，[归档](archive/research_records/2026-08-19_plain_logit_ensemble/RESULT.md)
- Voxel-KD：[报告](workspace/BraTS/sammed3d_t1ce_wt/plain_voxel_kd_20260819/KD_TEST270.json)，[归档](archive/research_records/2026-08-19_plain_voxel_kd/RESULT.md)
- Historical168：[报告](workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/FINAL_TEST270_EPOCH168.json)
- EMA168：[报告](workspace/BraTS/sammed3d_t1ce_wt/plain_ema_20260819/full756_seed20260833_e200/EMA168_TEST270.json)
- True-160³：[报告](workspace/BraTS/sammed3d_t1ce_wt/plain_resolution_20260818/full756_160_seed20260829_epoch200/FINAL_TEST270_TRUE160_EPOCH200.json)
- Plain epoch99：[报告](workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/FINAL_TEST270.json)

## Fixed validation108/50 subjects：true-128³开发榜

同一历史固定validation cache、raw-logit阈值0，严格按Scan Dice降序。这里只用于开发选择，不与test数字混排。

| 排名 | 方法 | 扫描平均Dice | 患者平均Dice | 定位命中 | 是否进test | 结论 |
|---:|---|---:|---:|---:|---|---|
| **1** | **972例蒸馏训练（168轮）** | **`0.740012535`** | **`0.731266217`** | **`0.953704`** | 用户授权门后探索性 | 当前development最高 |
| 2 | 基础宽度网络（16通道，第166轮） | `0.739335220` | **`0.731320350`** | **`0.972222`** | 否 | width24的matched control |
| 3 | 加宽网络（24通道，第166轮） | `0.736463710` | `0.727794977` | `0.962963` | 否 | 容量扩展失败，低于matched width16 |
| 4 | 固定老师重复三次指导（对照，168轮） | `0.736013432` | `0.728973188` | `0.953704` | 否 | 新matched control，仍低于archived seed32 |
| 5 | 逐体素蒸馏（200轮） | `0.734384308` | `0.724434143` | `0.935185` | 是 | 旧单模validation记录 |
| 6 | 双模型预测融合（平滑模型占25%） | `0.734171097` | `0.724489716` | `0.944444` | 是 | 当前总体teacher |
| 7 | 只用真实标签继续训练（200轮） | `0.733605660` | `0.723671900` | `0.935185` | 否 | KD匹配对照 |
| 8 | 区域均衡蒸馏（192轮） | `0.732959670` | `0.723071822` | `0.944444` | 否 | 未超过最佳base KD |
| 9 | 按训练标签从三个老师中择优指导（168轮） | `0.732909616` | `0.725997970` | `0.925926` | 否 | 三项低于matched control，九门全失败 |
| 10 | 双模型预测融合（平滑模型占50%） | `0.732048303` | `0.722623021` | `0.953704` | 否 | point高但scan次优 |
| 11 | 旧基础模型（168轮） | `0.731601219` | `0.722120565` | `0.925926` | 已历史测评 | 当前起点 |
| 12 | 训练权重平滑（168轮） | `0.718824903` | `0.710217848` | `0.944444` | 用户追加单测 | 弱于历史Dice |
| 13 | 训练权重平滑（200轮） | `0.716017457` | `0.706696547` | `0.935185` | 否 | 负结果 |
| 14 | 同次训练的原始权重（200轮） | `0.715409314` | `0.706007477` | `0.935185` | 否 | EMA200对照 |
| 15 | 同次训练的原始权重（168轮） | `0.706554216` | `0.698670892` | `0.935185` | 否 | EMA168对照 |
| 16 | 不同训练的参数直接平均（平滑权重占25%） | `0.431342862` | `0.430487334` | `0.796296` | 否 | 不同权重盆地，灾难性失败 |

## 非同口径test历史：不得与direct locator主榜混排

| 方法 | Scan Dice | Subject Dice | 权限/协议 |
|---|---:|---:|---|
| residual refiner | `0.703798` | `0.686637` | GT-centered oracle crop/提示 |
| historical brain-crop SAM C3 | `0.625359` | `0.610042` | 点击来自GT |
| upstream-literal vanilla best-of-five | `0.594226` | `0.580970` | GT-centered交互点击 |

<!-- SAM_GRAY_E050_TEST270 -->
## 内部test270：SAM-Med3D标签中心裁剪、五点击最佳

同270扫描/119患者，独立于全自动榜。固定128³/1.5mm、阈值0、无增强/后处理，三行复用内部测试，非外部/新盲测；重建测试张量与旧基线SHA完全一致。161轮在看到50轮测试后由用户指定，属事后诊断，未调整参数；不能称独立胜出确认。

| 排名 | 模型 | 扫描平均Dice（95%置信区间） | 患者平均Dice（95%置信区间） | 定位命中 | 前向计算量 |
|---:|---|---:|---:|---|---|
| 1 | 旧基础模型，第21轮 | 0.594226196 [0.570952, 0.616407] | 0.580970118 [0.557979, 0.602854] | 不适用（交互） | 1编码+5解码 |
| 2 | 灰度模型，第50轮 | 0.566424175 [0.540070, 0.591636] | 0.550521517 [0.524077, 0.575588] | 不适用（交互） | 1编码+5解码 |
| 3 | 灰度模型，第161轮（事后诊断） | 0.515107193 [0.486495, 0.542907] | 0.496310811 [0.468522, 0.523234] | 不适用（交互） | 1编码+5解码 |

第161轮在同一复用内部270扫描/119患者上五点击最佳Scan 0.515107、Subject 0.496311；低于旧基线0.594226，差-0.079119（患者配对95%CI [-0.09230627962951696, -0.0668751612899854]）。相对第50轮0.566424的差为-0.051317（95%CI [-0.0627234925298947, -0.04080606193794094]）。 训练参数与旧基线不完全一致，无灰度因果消融。[第161轮完整证据](archive/research_records/2026-09-17_sammed3d_graymean_e161_test270/RESULT.md) · [第50轮证据](archive/research_records/2026-09-17_sammed3d_graymean_e050_test270/RESULT.md)。训练已按用户要求停在161/200，不自动恢复。

复现审计更正：这里的同测试输入与评分协议，不表示训练配方只差灰度项。旧baseline每轮约2次优化器更新，灰度组前2轮每轮972次、此后122次；分割loss系数、小前景重抽样和seed等也不同。旧记录仍保留，不能由这张表估计灰度独立因果收益。[审计证据](archive/reviews/2026-09-17_graymean_reproduction_audit/RESULT.md)。
<!-- END_SAM_GRAY_E050_TEST270 -->

## 当前决策

- 最高UCSD-BraTS2024test56三项点估计：Full972 KD168固定Flip8（8×）`0.732973/0.733625/56/56`；最高1×为同checkpoint identity `0.730002/0.730740/56/56`。GCSAM184相对Full为负点估计（Scan`−0.012312`，原CI `[-0.025822,-0.000092]`）且Point少4；原区间贴零，后续数值精度审计不支持稳健Scan显著下降。该cohort是patient-disjoint但非institution-disjoint、非hidden blind，已关闭。

- 最高UCSD75 official-float32 Scan/Subject点估计：GCSAM fit864 inner-selected epoch184（1×），`0.755890/0.755312/70/75`；相对Full972 KD的Dice CI跨0。四个来源版本行中Point点估计最高是同模型mirror-uint8的71/75。两来源是同一reused cohort，不能称两次独立外部验证。
- 最高BraTS-Africa95 Dice点估计：GCSAM184 identity（1×）`0.591619/0.591619/84/95`；相对Full identity Dice差`+0.068617`且paired CI整体为正，但Point少3。Africa95 Point最高为Full KD168 Flip8的88/95，代价8×；三行是公开post-challenge、机构独立而非隐藏blind。
- 最高CPTAC33 Scan/Subject点估计：Full KD168 Flip8（8×）`0.844577/0.844577/33/33`，相对identity paired CI整体为正；1× GCSAM `0.843285`只比Full `0.841331`小幅高且CI跨0，Africa-informed方向仅弱支持。Public Training labels external-to-project，不是hidden blind。
- 当前内部test270/119的128³自动总体及1× Scan/Subject最高点估计：公开预训练后微调179，0.761649/0.745859；对193配对区间跨0，未过验证提升门，不与native grid混作优越性声明。
- 在128³组中，1× Point最高仍为Full972 KD168的262/270，总体Point最高仍为其Flip8的265/270；预训练179的246/270非最高。
- 160³、cross-seed parameter soup、ReCo-KD停止test或超参搜索。
- GT最佳老师routing与fresh等权共识均关闭；后续不继续围绕seed32/40/41调teacher权重/router。当前没有已预注册的新候选。

- 2026-09-20原始分辨率e173 test270（原始栅格）：Scan 0.768578，同栅格对193配对差+0.007444、CI[-0.0006,+0.0143]，配对区间跨0，不确定；不与128³行相减，不据此调参或重测。

机器可读严格排序版本：[AFLOW_BRATS_LEADERBOARD.json](AFLOW_BRATS_LEADERBOARD.json)。

<!-- FOLLOWUP_TEST270_GRID_LEADERS -->
最新固定候选评测后：128³组Scan最高点估计0.761649（segresnet_finetune_pretrained_20260921）；native grid组为0.768578（native_resolution_pure_segresnet_full972_epoch173_20260920）。均须按同栅格配对区间解释，参考179的native行是同模型重评分，不是新训练增益。
<!-- END_FOLLOWUP_TEST270_GRID_LEADERS -->


<!-- TEST270_TUNED_NATIVE -->
## 270例定向开发：原始栅格，test-tuned，独立分榜

本榜为复用内部探索，非独立泛化。五镜像＋体积解码配方仅在108验证选择后固定，本次不按270测试调参。统一原生1mm裁剪评分、排除填充；原始171轮阈值0.5历史行保留。前向等价量仅折算输入体素，不代表不同宽度/架构的FLOPs或实测耗时。

<!-- AUTO_INFERENCE12H_NATIVE_SCOPE_20261028 -->
本轮auto12h新行由用户授权直接按270反馈选全局规则，属于当前test调参；上方五镜像固定后测试的描述只对应历史方案。相位新行的体素等价量计入实际额外填充画布，CPU后处理另计。
<!-- END_AUTO_INFERENCE12H_NATIVE_SCOPE_20261028 -->

| 排名 | 方法/快照 | Scan Dice | Subject Dice | Point | 128³前向等价量 | 证据 |
|---:|---|---:|---:|---:|---:|---|
| 1 | nnunet_resenc_20261013_mix_a0.75_t0.40 | 0.812965238 | 0.797511860 | 266/270 | 7.719× | [报告](archive/research_records/2026-10-13_nnunet_resenc_reference/reports/nnunet_resenc_20261013_mix_a0.75_t0.40.json) |
| 2 | nnunet_resenc_20261013_mix_a0.75_t0.50 | 0.812438704 | 0.796841793 | 266/270 | 7.719× | [报告](archive/research_records/2026-10-13_nnunet_resenc_reference/reports/nnunet_resenc_20261013_mix_a0.75_t0.50.json) |
| 3 | nnunet_resenc_20261013_single | 0.811282631 | 0.795536487 | 265/270 | 2.387× | [报告](archive/research_records/2026-10-13_nnunet_resenc_reference/reports/nnunet_resenc_20261013_single.json) |
| 4 | nnunet_resenc_20261013_mix_a0.50_t0.40 | 0.805646185 | 0.790990562 | 266/270 | 7.719× | [报告](archive/research_records/2026-10-13_nnunet_resenc_reference/reports/nnunet_resenc_20261013_mix_a0.50_t0.40.json) |
| 5 | nnunet_resenc_20261013_mix_a0.50_t0.50 | 0.805437345 | 0.790659164 | 266/270 | 7.719× | [报告](archive/research_records/2026-10-13_nnunet_resenc_reference/reports/nnunet_resenc_20261013_mix_a0.50_t0.50.json) |
| 6 | nnunet_resenc_20261013_mix_a0.25_t0.40 | 0.790235410 | 0.776102962 | 267/270 | 7.719× | [报告](archive/research_records/2026-10-13_nnunet_resenc_reference/reports/nnunet_resenc_20261013_mix_a0.25_t0.40.json) |
| 7 | nnunet_resenc_20261013_mix_a0.25_t0.50 | 0.789749882 | 0.775520535 | 267/270 | 7.719× | [报告](archive/research_records/2026-10-13_nnunet_resenc_reference/reports/nnunet_resenc_20261013_mix_a0.25_t0.50.json) |
| 8 | race_segresnet_e164_enhanced_test270_20261029 | 0.789263436 | 0.775438295 | 256/270 | 21.328× | [报告](archive/research_records/2026-10-29_race_e164_inference_test270/RAW_TEST270_enhanced.json) |
| 9 | race_v2_stage_20261029_fusion_fixed108 | 0.789070115 | 0.776314624 | 254/270 | 42.656× | [报告](archive/research_records/2026-10-29_race_v2_stage/RAW_TEST270_fusion.json) |
| 10 | race_v2_stage_20261029_single_fixed108 | 0.788697343 | 0.775536380 | 251/270 | 21.328× | [报告](archive/research_records/2026-10-29_race_v2_stage/RAW_TEST270_single.json) |
| 11 | segresnet_w32_e171_auto12h_0036_local_brain_joint_testtuned_20261028 | 0.784382113 | 0.771378811 | 260/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0036_local_brain_joint/RAW_TEST270.json) |
| 12 | segresnet_w32_e171_auto12h_0041_intensity_ensemble_holes_testtuned_20261028 | 0.784354741 | 0.771605680 | 259/270 | 213.281× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0041_intensity_ensemble_holes/RAW_TEST270.json) |
| 13 | segresnet_w32_e171_auto12h_0040_intensity_ensemble_cc_testtuned_20261028 | 0.784313346 | 0.771638941 | 259/270 | 213.281× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0040_intensity_ensemble_cc/RAW_TEST270.json) |
| 14 | segresnet_w32_e171_auto12h_0023_brain_holes_testtuned_20261028 | 0.784311696 | 0.771405304 | 260/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0023_brain_holes/RAW_TEST270.json) |
| 15 | segresnet_w32_e171_auto12h_0021_brain_domain_cc_testtuned_20261028 | 0.784255388 | 0.771433421 | 260/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0021_brain_domain_cc/RAW_TEST270.json) |
| 16 | segresnet_w32_e171_auto12h_0034_brain_cross128_cc_testtuned_20261028 | 0.784232764 | 0.771396652 | 260/270 | 906.828× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0034_brain_cross128_cc/RAW_TEST270.json) |
| 17 | segresnet_w32_e171_auto12h_0035_brain_cross256_cc_testtuned_20261028 | 0.784230811 | 0.771394963 | 260/270 | 1792.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0035_brain_cross256_cc/RAW_TEST270.json) |
| 18 | segresnet_w32_e171_auto12h_0033_brain_case20_cc_testtuned_20261028 | 0.784199267 | 0.771395766 | 259/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0033_brain_case20_cc/RAW_TEST270.json) |
| 19 | segresnet_w32_e171_auto12h_0008_consensus_cc_testtuned_20261028 | 0.784139146 | 0.771156128 | 258/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0008_consensus_cc/RAW_TEST270.json) |
| 20 | segresnet_w32_e171_auto12h_0022_dense_unary_cc_testtuned_20261028 | 0.784138000 | 0.771220087 | 258/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0022_dense_unary_cc/RAW_TEST270.json) |
| 21 | segresnet_w32_e171_auto12h_0018_adaptive_holes_testtuned_20261028 | 0.784072660 | 0.771075667 | 260/270 | 18.662× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0018_adaptive_holes/RAW_TEST270.json) |
| 22 | segresnet_w32_e171_auto12h_0027_cross256_cc_testtuned_20261028 | 0.784060499 | 0.771127418 | 260/270 | 1792.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0027_cross256_cc/RAW_TEST270.json) |
| 23 | segresnet_w32_e171_auto12h_0026_cross128_cc_testtuned_20261028 | 0.784059438 | 0.771125047 | 260/270 | 906.828× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0026_cross128_cc/RAW_TEST270.json) |
| 24 | segresnet_w32_e171_auto12h_0013_adaptive_cc_testtuned_20261028 | 0.784001240 | 0.771026028 | 260/270 | 18.662× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0013_adaptive_cc/RAW_TEST270.json) |
| 25 | segresnet_w32_e171_auto12h_0006_best_holes_testtuned_20261028 | 0.783849666 | 0.770681114 | 260/270 | 18.662× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0006_best_holes/RAW_TEST270.json) |
| 26 | segresnet_w32_e171_auto12h_0014_nonlinear_cc_testtuned_20261028 | 0.783827031 | 0.771028561 | 260/270 | 18.662× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0014_nonlinear_cc/RAW_TEST270.json) |
| 27 | segresnet_w32_e171_auto12h_0039_intensity_ensemble_testtuned_20261028 | 0.783796238 | 0.771192326 | 259/270 | 213.281× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0039_intensity_ensemble/RAW_TEST270.json) |
| 28 | segresnet_w32_e171_auto12h_0003_core_calibration_cc_testtuned_20261028 | 0.783792134 | 0.770640150 | 260/270 | 18.662× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0003_core_calibration_cc/RAW_TEST270.json) |
| 29 | segresnet_w32_e171_auto12h_0005_fine_calibration_cc_testtuned_20261028 | 0.783786881 | 0.770920141 | 260/270 | 18.662× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0005_fine_calibration_cc/RAW_TEST270.json) |
| 30 | segresnet_w32_e171_auto12h_0020_brain_domain_testtuned_20261028 | 0.783639558 | 0.770874052 | 260/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0020_brain_domain/RAW_TEST270.json) |
| 31 | segresnet_w32_e171_auto12h_0030_brain_case28_testtuned_20261028 | 0.783637798 | 0.770945921 | 259/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0030_brain_case28/RAW_TEST270.json) |
| 32 | segresnet_w32_e171_auto12h_0029_brain_case20_testtuned_20261028 | 0.783626696 | 0.770869348 | 259/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0029_brain_case20/RAW_TEST270.json) |
| 33 | segresnet_w32_e171_auto12h_0038_intensity11_cal_testtuned_20261028 | 0.783603293 | 0.770717996 | 261/270 | 39.990× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0038_intensity11_cal/RAW_TEST270.json) |
| 34 | segresnet_w32_e171_auto12h_0028_brain_case16_testtuned_20261028 | 0.783564824 | 0.770775370 | 260/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0028_brain_case16/RAW_TEST270.json) |
| 35 | segresnet_w32_e171_auto12h_0015_case_consistency_testtuned_20261028 | 0.783559327 | 0.770799974 | 260/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0015_case_consistency/RAW_TEST270.json) |
| 36 | segresnet_w32_e171_auto12h_0031_brain_cross128_testtuned_20261028 | 0.783556388 | 0.770762421 | 260/270 | 906.828× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0031_brain_cross128/RAW_TEST270.json) |
| 37 | segresnet_w32_e171_auto12h_0032_brain_cross256_testtuned_20261028 | 0.783554645 | 0.770760766 | 260/270 | 1792.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0032_brain_cross256/RAW_TEST270.json) |
| 38 | segresnet_w32_e171_auto12h_0016_joint_calibration_testtuned_20261028 | 0.783534898 | 0.770782630 | 259/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0016_joint_calibration/RAW_TEST270.json) |
| 39 | segresnet_w32_e171_auto12h_0025_cross_phase256_testtuned_20261028 | 0.783500675 | 0.770624236 | 260/270 | 1792.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0025_cross_phase256/RAW_TEST270.json) |
| 40 | segresnet_w32_e171_auto12h_0024_cross_phase128_testtuned_20261028 | 0.783499961 | 0.770622264 | 260/270 | 906.828× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0024_cross_phase128/RAW_TEST270.json) |
| 41 | segresnet_w32_e171_auto12h_0019_dense_unary_control_testtuned_20261028 | 0.783491753 | 0.770633068 | 258/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0019_dense_unary_control/RAW_TEST270.json) |
| 42 | segresnet_w32_e171_auto12h_0017_zero_phase_control_testtuned_20261028 | 0.783485779 | 0.770627748 | 260/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0017_zero_phase_control/RAW_TEST270.json) |
| 43 | segresnet_w32_e171_auto12h_0007_consensus_calibration_testtuned_20261028 | 0.783445583 | 0.770508861 | 258/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0007_consensus_calibration/RAW_TEST270.json) |
| 44 | segresnet_w32_e171_auto12h_0037_intensity_api_probe_testtuned_20261028 | 0.783416767 | 0.770318319 | 259/270 | 39.990× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0037_intensity_api_probe/RAW_TEST270.json) |
| 45 | race_segresnet_e164_raw_test270_20261029 | 0.783374226 | 0.769423177 | 249/270 | 2.666× | [报告](archive/research_records/2026-10-29_race_e164_inference_test270/RAW_TEST270_raw.json) |
| 46 | segresnet_w32_e171_auto12h_0004_fine_calibration_testtuned_20261028 | 0.783195303 | 0.770271709 | 260/270 | 18.662× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0004_fine_calibration/RAW_TEST270.json) |
| 47 | segresnet_w32_e171_auto12h_0012_nonlinear_calibration_testtuned_20261028 | 0.783126494 | 0.770329978 | 260/270 | 18.662× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0012_nonlinear_calibration/RAW_TEST270.json) |
| 48 | segresnet_w32_e171_auto12h_0010_extended_calibration_testtuned_20261028 | 0.783093345 | 0.769932445 | 260/270 | 21.328× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0010_extended_calibration/RAW_TEST270.json) |
| 49 | segresnet_w32_e171_auto12h_0002_core_calibration_testtuned_20261028 | 0.783010428 | 0.769852844 | 260/270 | 18.662× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0002_core_calibration/RAW_TEST270.json) |
| 50 | segresnet_w32_e171_auto12h_0009_adaptive_calibration_testtuned_20261028 | 0.782931551 | 0.769998416 | 260/270 | 18.662× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0009_adaptive_calibration/RAW_TEST270.json) |
| 51 | segresnet_w32_e171_auto12h_0011_official_rankseg_testtuned_20261028 | 0.782628427 | 0.769737055 | 260/270 | 18.662× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0011_official_rankseg/RAW_TEST270.json) |
| 52 | segresnet_w32_e171_auto12h_component1000_testtuned_20261028 | 0.781651665 | 0.769047724 | 255/270 | 13.330× | [报告](archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0001_cpu_binary/RAW_TEST270.json) |
| 53 | segresnet_w32_e171_mirror5_volumeT2_test270_20261028 | 0.780995198 | 0.768306279 | 255/270 | 13.330× | [报告](archive/research_records/2026-10-28_segresnet_inference_selected_test270/RAW_TEST270.json) |
| 54 | panel_10 | 0.775891493 | 0.761694199 | 261/270 | 5.332× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_10.json) |
| 55 | panel_11 | 0.775823547 | 0.761543765 | 261/270 | 5.332× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_11.json) |
| 56 | panel_12 | 0.775561193 | 0.761252936 | 261/270 | 5.332× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_12.json) |
| 57 | segresnet_width32_native_epoch171_test270_20261028 | 0.773228040 | 0.760026920 | 256/270 | 2.666× | [报告](archive/research_records/2026-10-28_segresnet_w32_e171_test270/RAW_TEST270.json) |
| 58 | A_6c53d508f4172cbf | 0.772829645 | 0.758800132 | 253/270 | 8.000× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/A_6c53d508f4172cbf.json) |
| 59 | panel_02 | 0.772829645 | 0.758800132 | 253/270 | 8.000× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_02.json) |
| 60 | panel_01 | 0.772810871 | 0.758769746 | 253/270 | 8.000× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_01.json) |
| 61 | panel_03 | 0.772760359 | 0.758769868 | 253/270 | 8.000× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_03.json) |
| 62 | segresnet_multiscale_native_epoch166_test270_20260929 | 0.771386135 | 0.757817294 | 248/270 | 2.666× | [报告](archive/research_records/2026-09-29_segresnet_multiscale_e166_test270/RAW_TEST270.json) |
| 63 | A_4b4a4f1c7b8e7175 | 0.770114302 | 0.756940881 | 253/270 | 8.000× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/A_4b4a4f1c7b8e7175.json) |
| 64 | A_1d7ffa9fe448c903 | 0.769373712 | 0.754786090 | 250/270 | 8.000× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/A_1d7ffa9fe448c903.json) |
| 65 | panel_05 | 0.768573106 | 0.754142334 | 261/270 | 2.666× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_05.json) |
| 66 | ref_native173 | 0.768573106 | 0.754142334 | 261/270 | 2.666× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/ref_native173.json) |
| 67 | panel_04 | 0.768565588 | 0.754105597 | 261/270 | 2.666× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_04.json) |
| 68 | panel_06 | 0.768537587 | 0.754132893 | 261/270 | 2.666× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_06.json) |
| 69 | panel_09 | 0.766437954 | 0.751128725 | 237/270 | 2.666× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_09.json) |
| 70 | panel_08 | 0.766282487 | 0.750971793 | 237/270 | 2.666× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_08.json) |
| 71 | ref_combination166 | 0.766282487 | 0.750971793 | 237/270 | 2.666× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/ref_combination166.json) |
| 72 | panel_07 | 0.766018417 | 0.750707659 | 237/270 | 2.666× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_07.json) |
| 73 | B_4d6b52d8b95261de | 0.764542988 | 0.748758376 | 249/270 | 1.000× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/B_4d6b52d8b95261de.json) |
| 74 | ref_pretrained179 | 0.764542988 | 0.748758376 | 249/270 | 1.000× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/ref_pretrained179.json) |
| 75 | ref_patient172 | 0.762327244 | 0.748078780 | 253/270 | 1.000× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/ref_patient172.json) |
| 76 | ref_pure193 | 0.761135070 | 0.746964020 | 246/270 | 1.000× | [报告](archive/research_records/2026-09-21_t1ce_test270_10h/reports/ref_pure193.json) |
<!-- END_TEST270_TUNED_NATIVE -->


<!-- NNUNET_NATIVE_VALIDATION_20261013 -->
## 原生1mm验证108：残差编码器参考

固定108扫描/50患者，仅验证；NN模型训练550轮，以原生Scan选择第525轮。下表同评分栅格，部署滑窗与整脑成本不同，不和128³开发榜或270测试榜混排。

| 方法 | Scan Dice | Subject Dice | Point | 推理方式 |
|---|---:|---:|---:|---|
| 残差编码器L，选525轮 | **0.784671645** | 0.780820797 | 105/108 | 单模型滑窗，无镜像 |
| 原多尺度SegResNet166参考 | 0.753092133 | 0.739668718 | 94/108 | 单次整脑 |

Scan差+0.031580、患者配对95%CI[+0.021781,+0.043297]，通过事前验证收益门。属于复用验证/test-informed开发，不等于新270或独立泛化成绩。[报告](archive/research_records/2026-10-13_nnunet_resenc_reference/RESULT.md)。
<!-- END_NNUNET_NATIVE_VALIDATION_20261013 -->
