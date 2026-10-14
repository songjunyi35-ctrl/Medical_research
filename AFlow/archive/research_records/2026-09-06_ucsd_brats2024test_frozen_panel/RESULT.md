# UCSD BraTS-GLI 2024公开test包冻结面板：完成并关闭

日期：2026-09-06。状态：一次性评测完成，cohort关闭；禁止调参、重跑或回流选择。

## 一句话

把两个冻结checkpoint带到TCIA 2026-01-22新公开的BraTS-GLI 2024隐藏test中UCSD部分（排除1名union重叠患者后41人/56 scans，治疗后GBM、专家批准标签）只考一次。

## 出发问题

治疗后阶段的外部证据此前只有WAW-post31（两组比较全部positive-but-uncertain）；内部fresh容量只剩2人/3 scans。本包是patient-disjoint、标签词典与训练完全同口径（BraTS2024 post-treatment，WT={1,2,3}排除RC4）的最大新队列。

## 做法与门（全部在像素/模型前冻结）

来源审计（[审计](../../reviews/2026-09-06_ucsd_brats2024test_source_audit/RESULT.md)）→七问IDEA（SHA `73709613…`）→PREREG（SHA `acefe705…`）→冻结41/56 cohort（SHA `aed4cfcb…`）→双独立官方传输112文件逐文件SHA exact（包内无`.sums`，双传输门为替代）→raw标签域扫描（56/56整数、值域⊆{0,1,2,3,4}、无空WT）→首例单线程probe→全量canonical N4 cache（零WT crop丢失56/56、首例byte exact）→synthetic smoke→MODEL_PREACCESS→唯一504 forwards→终态重hash与全统计独立重算（VERIFICATION全过）。6项合成合同测试通过。

## 结果（56 scans / 41 subjects；绝对CI为10k bootstrap）

| 方法 | Scan Dice（95% CI） | Subject Dice（95% CI） | Point hit | 成本 |
|---|---:|---:|---:|---:|
| Full972 KD168固定Flip8 | `0.732972851` `[0.693045,0.771526]` | `0.733624950` `[0.694818,0.771538]` | **56/56 (`1.0`)** | 8× |
| Full972 KD168 identity | `0.730001741` `[0.690626,0.766791]` | `0.730739692` `[0.692275,0.767211]` | **56/56 (`1.0`)** | 1× |
| GCSAM184 identity | `0.717689253` `[0.671931,0.761285]` | `0.715545322` `[0.667478,0.760680]` | 52/56 (`0.928571`) | 1× |

预注册比较：

- **Flip8−identity** = `+0.002971109`，scan配对CI `[-0.002740,+0.008750]`、subject配对CI `[-0.002830,+0.008735]`，32/24/0 improve/worse/tie，Point差0 → `positive_but_uncertain`。第六个Scan点估计为正的cohort，但本队列区间跨0。
- **GCSAM−Full identity** = `−0.012312488`（Subject `−0.015194370`），scan配对CI `[-0.025822,-0.000092]`、subject配对CI `[-0.031507,-0.000268]`，21/35/0，Point`−4` → `not_supported_or_reversed`。**这是七个已揭示cohort中GCSAM checkpoint第一次在配对CI整体为负的意义上确定劣于Full972 KD168**，且发生在与训练阶段/标签语义最匹配、患者隔离的队列上。

## 直觉是否支持

- Flip8：方向支持（六cohort Scan点估计全正），强度不确定（本队列CI跨0）；八镜像仍8×成本。
- GCSAM：**不支持并被反转**。此前"GCSAM在外部Dice点估计普遍偏高（仅Africa CI确定）"的叙事必须更新为：其优势不仅queue-specific，在stage-matched patient-disjoint外部可以显著反向。与其五外部coverage倾向（预测体积偏大、precision不占优）在近域可能变成劣势相容，但机制归因仍不能单因化（fit864 vs fit972、epoch、seed、optimizer同时不同）。
- Full972 KD168 Point 56/56为全部外部cohort最高（对照WAW-post31仅15/31），支持"同阶段、同预处理谱系域中定位极稳"。

## 边界与caveat

- Public post-challenge、**非hidden blind**；UCSD是BraTS2024七来源站之一，**非institution-disjoint**（与UCSD75同机构、不同患者），不能当跨机构泛化证据，不与UCSD75/internal/其他外部混排。
- 训练源caveat继承（Kaggle mirror、官方Synapse parity未证明）。
- 128³ rasterization与N4双重预处理caveat与其他面板相同；原生256³/1mm。

## 代价

训练0；双源2×312MB（copy A workspace持久、copy B /tmp）；cache 587MB（/tmp）；N4 cache 325.7秒/14 workers；正式10.15秒/504 forwards/峰值5.07GiB；部署identity 1×、Flip8 8×。

## 下一决定

- Cohort关闭。不据此调整checkpoint、阈值、视图、N4、后处理或复活/否决GCSAM训练配方（配方双复制失败早已独立成立）。
- 跨cohort综合与claim registry需刷新：GCSAM−Full现为"六负一正确定区间"（Africa正、本队列负、其余跨0），必须停止任何"GCSAM外部普遍偏优"措辞。

## 事后误差分解（0新forward，仅描述）

按保存计数与"空预测precision=1"既有口径：Full identity/Flip8/GCSAM precision=`0.740/0.751/0.717`、recall=`0.749/0.745/0.756`。GCSAM−Full precision=`−0.02263`、事后CI `[-0.04383,-0.00437]`全负，recall=`+0.00691`跨0，预测体素差`+773`跨0——在目标尺寸落于训练分布内的近域队列，GCSAM的扩张倾向只损失precision、换不来recall。Flip8−identity precision=`+0.01111`、CI `[+0.00454,+0.01768]`全正，预测体素平均少558且CI全负，recall近零跨0——保守去假阳性机制在有保存计数的五个外部cohort中5/5复现。train支持描述：本队列raw WT中位40,266.5，仅居train972经验分位48.35%（Africa为95.47%），7/56超train p90、0超max；这与"GCSAM优势只在远超训练支持的大目标域出现"的coverage解释一致，但阶段/采集/标签仍共变，禁止size router或threshold调整。GCSAM四个Point miss：`0052_01/0057_01/0070_01/0109_01`。见[事后审计](POSTHOC_ERROR_AUDIT.json)。

## 证据

- [IDEA](IDEA.md)、[PREREG](PREREG.md)、[来源审计](../../reviews/2026-09-06_ucsd_brats2024test_source_audit/RESULT.md)
- [raw报告](RAW_UCSD_BRATS2024TEST56.json)（SHA `74466d1cdc56d864d48e5d8604e00d004ecdb636882366546b4738a1b794201c`）
- [SOURCE_RECEIPT](SOURCE_RECEIPT.json)（SHA `fd135e7a…`）、[RAW_LABEL_SCAN](RAW_LABEL_SCAN.json)、[FIRST_CASE_PROBE](FIRST_CASE_PROBE.json)、[CACHE_RECEIPT](CACHE_RECEIPT.json)（SHA `c5db9c06…`）、[SMOKE_RECEIPT](SMOKE_RECEIPT.json)、[MODEL_PREACCESS](MODEL_PREACCESS.json)（SHA `41708bbb…`）
- [终态验证](VERIFICATION_RECEIPT.json)（重hash 112源文件/双checkpoint/cache三件+五组统计独立重算全部exact）
- [机器结果](RESULT.json)

## 事后Point类别审计（0新forward）

离线重建56例128³多类标签（WT与cache 56/56 exact）：44/56例存在切除腔label4（RC体素中位数见审计）。**Full972 KD168 identity与Flip8的argmax point 56/56全部落在GT WT内、0次落RC**——与WAW-post31（Full 16个miss中13个落RC）形成直接对照，说明早期术后RC吸引argmax不是普遍规律，而主要出现在跨机构/跨预处理谱系域偏移下；近训练域中当前二值监督足以让定位避开RC。GCSAM 4个miss=2 RC+2普通背景，全部是小WT病例（target 1,713–10,985体素、Dice 0.33–0.47）。仅描述，不授权RC避让/routing/重测。见[Point类别审计](POSTHOC_POINT_CATEGORIES.json)。

## 事后时点组成描述（0新forward）

按官方v3映射的BraTS时点后缀：100/101/102=41/11/4 scans。三行模型在后缀100上的平均Dice约`0.69–0.71`、在101/102上约`0.76–0.79`；多时点患者同时出现在多组、患者组成不同，仅描述队列组成，不得作时点router或因果。见[时点描述](POSTHOC_TIMEPOINT_DESCRIPTION.json)；cache index已归档为[CACHE_INDEX_FINAL.json](CACHE_INDEX_FINAL.json)（SHA与CACHE_RECEIPT一致），/tmp cache可由copy A+归档builder重建。
