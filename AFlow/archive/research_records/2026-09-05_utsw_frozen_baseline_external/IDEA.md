# Experiment idea: 两位冻结baseline的UTSW362跨阶段外部基线

Status: planned，任何UTSW新病例下载或模型GPU加载前冻结。

## 1. 现在的问题是什么？

no-N4与Otsu在全链路隔离E42/F42上都未通过Stage A，不能进Full972/RHUH/UTSW候选链。但UTSW官方有362份人工细化术前分割，完整来源/标签/三种输入桥都已建立，尚无任何模型分数；现有UCSD75外部结果又限定于uint8镜像。需要一份不再调权重、直接衡量当前冻结模型跨医院+跨治疗阶段表现的基线。

## 2. 一句话直觉

不再改学生，把两位早已封卷的代表直接送到另一所医院、另一治疗阶段考试，先知道现有系统离真正外推有多远。

## 3. 为什么选择这个idea？

- Full972 random-init voxel knowledge distillation（逐体素知识蒸馏，KD）epoch168是当前reused internal test270总体Flip8和1×Subject/Point代表；checkpoint SHA=`f7df2a16…`。
- Gradient-Centralized Sharpness-Aware Minimization（梯度中心化锐度感知优化，GCSAM）fit864 epoch184是当前1×Scan和UCSD75点估计代表；checkpoint SHA=`acf2fce1…`。
- 两位选择只依据UTSW访问前已存在的排行榜，覆盖“总体8×主干”和“1×Scan代表”，不是看UTSW结果挑模型。
- 两checkpoint都已有reused test270与mirror-uint8 UCSD75访问；GCSAM复制Stage A后来失败、KD历史validation CI门也失败。必须披露，UTSW是新的独立cohort但不是盲选模型。
- no-N4/Otsu失败否定新输入候选的稳定晋级，不否定这两位既有canonical-N4 baseline的描述性外部测量。

## 4. 具体怎么做？

先冻结两checkpoint文件/state/schema及历史访问标签，使用随机合成128³输入在一张空闲GPU上分别跑identity与固定Flip8 smoke。随后从TCIA package1265把全部362个`brain_t1ce.nii.gz`和`rtumorseg_manual_correction.nii.gz`各下载两份；因官方目录无checksum，要求724/724官方size与copy A/B SHA exact后才打开NIfTI。

以copy A构建唯一canonical输入cache：转RAS，标签所有positive合为UTSW Whole Tumor（全肿瘤，WT），正值mask N4、shrink2/固定迭代、brain-support bbox、source population z-score、图像三线性/标签nearest-exact到128³；固定BT0001/2/3必须复现已归档canonical桥。两模型读同一cache，各运行identity一次与完整Flip8八次、raw logits平均、threshold0、argmax point，无Segment Anything Model（SAM）、无后处理。报告每模型/视图绝对指标、10,000次患者bootstrap CI，以及同视图GCSAM−KD配对差；不据差值选择或推广。

## 5. 唯一变化与固定项

- 唯一新增因素：UTSW362这个preoperative SRI24跨阶段外部cohort。
- 固定：两checkpoint、模型4,700,897参数、canonical N4、主FeTS图、全网格`r`标签、all-positive WT、128³、threshold0、identity/Flip8、无SAM/后处理、bootstrap seed20261391/10,000、全部362病例。
- 两模型的训练数据量、优化器与历史选择不同，所以模型间差值不是单因素因果实验，只是冻结面板描述。

## 6. 成功、失败和停止规则

- Primary绝对指标：Full972 KD的Flip8 Scan Dice；GCSAM的identity Scan Dice。
- 预定跨阶段参考门：各自主要视图Scan/Subject≥0.70且Point hits≥326/362（≥90%）。这是描述性稳健门，不触发旧test或新训练。
- 工程门：checkpoint/state/schema exact；双源724/724 size+SHA exact；362/362图像finite/nonnegative、标签`{0,1,2,4}`、geometry与非空WT；cache/fixed3/smoke/6516 forward/report hash全过。
- Failure/stop rule：任一工程门在对应访问边界停止；模型结果无论正负只运行一次，不删病例、不改模型/N4/阈值/view/label/bootstrap，不选更好模型重测。
- 是否允许进入test：这是公开UTSW preoperative cross-stage external，不授权fixed validation/outer/confirmation/test270/UCSD/RHUH重访，也不产生新训练。

## 7. 成本与风险

- 训练时间：0。
- 峰值显存：预计20–23GiB，先synthetic smoke，24GiB硬门。
- 磁盘：双源3,626,173,198 bytes；单cache3,795,845,120 bytes；约7.42GB plus报告，全部`/tmp`。
- 推理forwards/延迟：362×9×2=6516单例forward；identity部署1×、Flip8部署8×。
- 最大风险：UTSW是术前SRI24而训练是治疗后MNI，性能下降不等于模型在同域失败；训练源Kaggle与官方corrected-v2 parity未证明；两模型已在UCSD/test上报告过，不能称首次盲模型选择。

## 8. 最终结果（完成后填写）

- UTSW362：待执行。
- CI/病例：362 scans/362 subjects；待执行。
- 是否支持原直觉：待执行。
- 学到了什么：待执行。
- 下一步：待执行。
- 报告/归档/哈希：待执行。

