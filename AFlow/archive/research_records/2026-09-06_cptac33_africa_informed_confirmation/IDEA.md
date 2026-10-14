# Experiment idea: CPTAC33对Africa checkpoint方向的公开确认

Status: completed once and closed; preregistration below was frozen before `.sums`, patient payload, NIfTI arrays, labels, cache or model access

## 1. 现在的问题是什么？

BraTS-Africa95上，GCSAM184 identity相对Full972 KD168 identity的Scan=Subject Dice为`+0.068617`，paired 95% CI `[+0.044472,+0.092498]`，但Point少3。它只来自四个尼日利亚中心，且两checkpoint同时不同fit864/972、epoch184/168和seed，不能证明optimizer因果，也不知道该冻结checkpoint方向能否跨另一个术前来源复现。

## 2. 一句话直觉

第一次异地考试GCSAM更会圈完整范围；现在让同两位已封卷的阅卷员去CPTAC另一套公开试卷复考，检查方向是不是只属于Africa。

## 3. 为什么选择这个idea？

- TCIA官方`BraTS2021_MappingToTCIA.xlsx`把39位CPTAC PatientID映射到BraTS ID：public Training33、Validation6。目录实测Validation6均无seg，故另行冻结只用33个公开人工标签病例；选择由label可用性在像素/模型前决定。
- BraTS2021 analysis说明三子区标签为人工ground truth、值1=NCR/2=ED/4=ET。CPTAC/CHOP-UPenn来源不在本项目BraTS2024 post-treatment七个贡献站；无跨库patient bridge，故只称source/institution-disjoint no-known overlap。
- 两checkpoint训练清单是本项目BraTS2024，不含BraTS2021/CPTAC；但继承本地Kaggle mirror官方parity未证明caveat。
- Africa结果形成显式待确认方向。UTSW/EGD/UCSF/Burdenko阻塞是标签、协议、重叠或target问题，没有检验CPTAC33。
- 一手来源：[TCIA BraTS2021 analysis](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/)、[CPTAC-GBM](https://www.cancerimagingarchive.net/collection/cptac-gbm/)。

## 4. 具体怎么做？

先下载package636根目录官方`.sums`，解析冻结33例的66个T1CE/seg校验条目；再对66路径做两份独立官方传输。官方checksum、路径、大小、双份SHA全一致才解码。

每例T1CE/seg转canonical RAS，检查geometry、finite、整数标签和值域`{0,1,2,4}`。CPTAC是术前数据，Whole Tumor（全肿瘤，WT）取所有正标签。图像沿当前训练输入运行N4 shrink2、正值image-only crop、source非零population z-score、三线性影像/nearest-exact标签到128³。

Full972 KD168固定跑8个三轴镜像，一次循环同时保存identity与逆翻float32等权Flip8；GCSAM184只跑identity。阈值0、point为最高raw logit，无Segment Anything Model（SAM）、后处理或模型ensemble。计算33 scans=33 subjects的Dice、Point与10,000次percentile bootstrap。

## 5. 唯一变化与固定项

- 唯一变化：新CPTAC-GBM/BraTS2021 public Training33 cohort。
- 固定：全33及顺序、T1CE+seg、preoperative WT=all positive、canonical N4/crop/z-score/128³、Full/GCSAM checkpoint与SHA、identity/Flip8公式、阈值0、Point、无SAM/后处理/ensemble、bootstrap样本/seed。
- Validation6无seg永久排除；不访问hidden challenge test，不按PatientID、日期、大小、标签或模型删病例。
- 面板与Africa95相同；CPTAC假设虽由Africa方向启发，但没有根据CPTAC结果调任何因素。

## 6. 成功、失败和停止规则

- Primary确认：GCSAM identity−Full identity Scan=Subject Dice。点估计`>0`且paired CI lower`>0`称较强独立方向支持；点估计`>0`但CI跨0称弱/不确定支持；点估计`<=0`称方向不复现。Point差永远单列，不能用来事后改Dice结论。
- Secondary：Full Flip8−identity同样报告方向/CI/Point，不作promotion。
- 工程门：package/mapping/cohort hashes；`.sums`包含66唯一条目且校验算法明确；双源66/66官方checksum/size/local SHA exact；33/33 geometry/finite/integer/raw domain/nonempty；首例单线程probe与cache byte exact；cache/checkpoint/evaluator/preaccess hashes；synthetic smoke。
- 任一工程门失败在模型前停止。模型一旦开始只允许一次；故障留边界，不改实现取第二套结果。
- 没有部署/训练promotion gate。结果不允许选择checkpoint、optimizer、epoch、threshold、N4、view、病例、router或后处理。

## 7. 成本与风险

- 训练0；两个checkpoint不改。
- Source：`.sums`约49.85MB；66文件每副本78,013,574 bytes，双份约149MiB。
- Cache：33×128³×5约330MiB，放`/tmp`；CPU N4预计数分钟。
- GPU：Full33×8 + GCSAM33=`297` forwards，预计<10秒，峰值<8GiB；部署1×/8×不变。
- 风险：33人CI宽；public Training不是hidden blind；CPTAC与BraTS2024无patient-ID bridge；术前/术后target仍不同。`.sums`如不覆盖选中文件则停止，不用双传输单独冒充官方checksum。

## 8. 最终结果

- Full identity/Flip8/GCSAM identity=`0.841331/0.844577/0.843285`，Point均33/33。
- GCSAM−Full=`+0.001954`，CI `[-0.006646,+0.012384]`，16胜17负：Africa方向仅弱/不确定支持，强幅度不复现。
- Flip8−identity=`+0.003246`，CI `[+0.000828,+0.005855]`，23胜10负：secondary方向支持。
- 训练0、297 forwards；internal/UCSD/Africa无新访问。Cohort关闭、不调参/重跑。
- [RESULT.md](RESULT.md)、[RESULT.json](RESULT.json)，raw SHA `4b9d8c90e1b5ec45b0e664e560301db20affd6a96387f416a62b561f8d57b4fd`。
