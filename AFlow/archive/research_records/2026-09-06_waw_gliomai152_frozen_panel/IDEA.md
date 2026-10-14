# Experiment idea: WAW-GliomAI152冻结checkpoint面板

Status: planned; metadata/label-domain source audit completed before image archive access; model performance not accessed.

## 1. 现在的问题是什么？

当前reused internal test270总体点估计最高是Full972 voxel knowledge distillation（体素知识蒸馏，KD）epoch168固定Flip8测试时增强（Test-Time Augmentation，TTA）：Scan/Subject Dice/Point=`0.757270/0.741254/265`，但它不是盲测。GCSAM184相对Full identity在Africa95的Scan差为`+0.068617`、配对95%区间全正，在Africa-informed CPTAC33却只有`+0.001954`且区间跨0；固定Flip8在internal/Africa/CPTAC的Scan点估计都正，但Africa区间跨0。项目仍缺一个来自新地区、公开人工标签、未用于训练或选择的新cohort裁决这些结果是否队列特异。

WAW-GliomAI最终公开包的描述写192名成人弥漫性胶质瘤患者、156份术前检查；实际最终coreg metadata与mask包严格交集为152名患者/152份术前检查。每份有T1对比增强（发布名`t1c`）和专家人工修订分区mask。

## 2. 一句话直觉

像把两位已经封存答案的“医生”，以及同一位医生的八方向复核，带到一所从未参加原训练的华沙医院统一考试：只考一次，不能看成绩后换答案。

## 3. 为什么选择这个idea？

- 现有Africa/CPTAC差异说明单一外部cohort不能代表所有域；WAW来自Medical University of Warsaw，不在BraTS2024的Duke/UCSF/Missouri/UCSD/Heidelberg/Michigan/Indiana七个来源站。
- Zenodo记录于2026-09-01公开、CC BY 4.0；最终记录提供官方文件MD5、coreg MRI、preop masks和技术验证。标签由FL-PoST/FeTS自动初始后经神经肿瘤影像专家人工复核与修改，15个术前双读者案例还有Dice/HD95复现资料。
- UTSW的未定义标签3/5、EGD的数据使用协议、UCSF的患者交叉表、ReMIND的几何/交付失败都没有运行WAW模型，也没有否定这个来源。
- WAW是术前成人grade2–4弥漫性胶质瘤，而训练是术后/治疗后；这种差异是外推压力，不是可回流的调参信号。
- 一手来源：[Zenodo final record](https://zenodo.org/records/22205631)与其发布的技术验证包；数据集论文发布者标为Scientific Data。

## 4. 具体怎么做？

输入为每位患者唯一的coreg/skull-stripped float32 delayed post-contrast T1 MRI（`*_brain_t1c.nii.gz`）。先按当前canonical流水线转RAS，运行固定N4 bias field correction（N4偏置场校正，shrink2与原迭代数），只用图像非零区求脑框、非零z-score并缩放到128³；mask原生同grid，只按同crop最近邻缩放，术前whole tumor（WT，全肿瘤）定义为所有正标签`{1,2,3}`。

冻结Full972 KD168依次运行三轴全部8种reflection，一次循环保存identity原图输出，并把8个raw logits逆翻回原坐标后float32等权平均成Flip8。冻结GCSAM184只运行identity。logit阈值`>0`得到WT mask，最高raw logit位置作为自动Point；没有Segment Anything Model（SAM）、连通域、形态学、模型ensemble或其他后处理。

计算152 scans=152 subjects的Scan/Subject Dice、Point hit，以及固定10,000次percentile bootstrap。比较Full Flip8−identity和GCSAM identity−Full identity；三个绝对结果和两个配对结果无论输赢都发布。

## 5. 唯一变化与固定项

- 唯一变化：新增WAW-GliomAI final release的152名术前患者cohort。
- 固定不变：病例由metadata T1c与mask精确交集一次冻结；每人唯一exam0；WT=`mask>0`；canonical RAS/N4 shrink2/image-only crop/z-score/128³；Full972 KD168与GCSAM184 checkpoint及SHA；阈值0；Full identity/固定Flip8视图、顺序与融合；GCSAM identity；无SAM/后处理/模型ensemble；指标、empty policy、10,000 bootstrap和seed。
- 不使用：raw非coreg影像、postop影像/label4、临床字段筛选、T1ce/t1c择优、模型结果驱动排例或重复。

## 6. 成功、失败和停止规则

- Primary指标：三个绝对Scan/Subject Dice/Point；GCSAM identity−Full identity与Full Flip8−identity的配对Scan/Subject差及95%区间。
- GCSAM解释门：差值>0且配对Scan CI下界>0才写“WAW支持checkpoint优势”；差值>0但CI跨0写“不确定”；差值≤0写“不支持/反向”。不设性能promotion。
- Flip8解释门：同样按正且CI全正/正但跨0/非正三档解释；不能合并已见cohort成总体显著性。
- 工程门：官方3,164,137,716-byte zip MD5必须为`0d24a3a2c76bc6fa9b5b6468679676b1`；152个T1c均存在，图像/mask geometry严格一致，label union只允许`{0,1,2,3}`，cache独立重验；checkpoint/evaluator/preaccess哈希一致。
- Failure/stop：任一来源、数量、geometry、label、cache、checkpoint或单线程复现门失败即模型前停止；不得改容差、丢病例、换`t1n`/raw图、阈值或预处理救场。
- 是否允许进入test：WAW本身是新public external-to-project cohort，不是hidden blind；只允许这一次冻结访问。结果不得授权重跑WAW或访问任何其他已关闭cohort。

## 7. 成本与风险

- 训练时间：0；teacher/loss/optimizer均不存在。
- 峰值显存：预计约5.1 GiB；单卡。
- 磁盘：官方coreg包2.95 GiB、mask/metadata约9 MiB、只提取T1c后预计低于1 GiB、128³ image+label cache约1.5 GiB；当前开始前剩9.4 GiB，不建双份大包。
- 推理forwards/延迟：Full固定8视图+GCSAM identity，共152×9=`1368`次单例forward；部署Full Flip8约8×identity延迟，checkpoint/参数不增加。
- 最大风险：发布`t1c`是delayed post-contrast而非训练命名`t1ce`；source已做coreg/skull-strip/bias correction后仍按冻结canonical再做N4；机构不同但无跨库patient-ID桥；checkpoint训练源Kaggle-to-official parity仍未证明。这些必须保留为caveat，不能靠好分数消除。

## 8. 最终结果（完成后填写）

- Validation：不适用；WAW未用于选择。
- Test：一次性public external152完成。Full identity/Flip8/GCSAM identity Dice=`0.683378/0.693494/0.693601`，Point=`141/143/140`。
- 置信区间/病例分布：GCSAM−Full=`+0.010223`，CI `[-0.003005,+0.023481]`，84/63/5；Flip8−identity=`+0.010116`，CI `[+0.004472,+0.016274]`，93/54/5。
- 是否支持原直觉：GCSAM仅正向不确定，不支持Africa强优势普适；Flip8的WAW Scan区间支持。
- 学到了什么：WAW最高1× Dice是GCSAM，最高Point是Full Flip8；二者总体Dice只差`0.000107`，不存在Dice/Point/成本统一冠军。
- 下一步：WAW关闭，不调参/重跑；将其加入独立榜和跨cohort描述。
- 报告/归档/哈希：[结果](RESULT.md)、[raw](RAW_WAW152.json)、[机器结果](RESULT.json)、[终态验证](VERIFICATION_RECEIPT.json)。
