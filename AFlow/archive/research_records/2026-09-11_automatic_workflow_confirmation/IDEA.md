# 从折策略收束到固定工作流：一次性复用内部test270确认

Status: new separately frozen confirmation stage, after completed development gates. This is not an amendment of the closed development protocol.

## 1. 现在的问题是什么？

主开发方法两个模型分别2/3次前向，实测3.961/2.652倍阶段加速，全部预定门通过；但六个折策略不等于两个最终固定workflow，尚未在本模型未训练的固定报告队列确认。

## 2. 一句话直觉

取最接近三折共同意见的方案，而不是挑看起来分数最高的一折；固定后再交卷，不看试卷反馈改答案。

## 3. 为什么现在选择它？

主实验已建立有限剪枝的开发精度—成本证据和真实执行证据，而random/UCT20种子补充未建立稳定树搜索优势。本阶段只收束已通过的穷举主方法，不选新的搜索器或增强。[镜像预测一手实现](https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/inference/predict_from_raw_data.py)支持融合语义；集合中位代表是本项目开发后冻结的确定性工程收束规则，不冒充已在前一预注册中固定或文献证明最优。

## 4. 具体怎么做？

对每个已冻结native control seed，取三个exhaustive折winner候选，计算每个候选与三者的集合对称差大小之和；最小者为中位代表，并列选视图数更少、workflow ID更小者。只按已选视图集合，不读取新的开发成绩、20种子最优值或test指标选择。该规则在开发结果之后提出，明确development-derived finalization。

每模型最终固定原图identity、代表子集pruned、full8三输出，共六行，不能看test只留好的一行。输入T1ce→固定自动模型镜像预测→逆翻转有序float32平均→阈值0掩膜和global argmax点。无训练/教师/SAM/后处理。full8原子每例只生成一次；pruned额外真实在线重跑，要求与相同原子的有序子集均值逐体素exact；结果只用在线pruned。不把统一评估预计算full8的成本叫部署节省。

## 5. 唯一变化与固定项

只改变镜像集合。两native control权重、fit684/313、epoch168、batch8、bf16前向/float32融合、原128缓存和阈值/点规则不变。权重继承主实验DEV_PREACCESS并重新SHA绑定。当前dev180/78仍属开发，不改成训练。test270/119与fit/dev患者不重叠；旧confirmation/fixed validation/outer/外部不新访问。

本阶段按用户要求继续工作流“完整闭环”的范围执行一次固定内部测试报告，不改变test角色；前阶段的无test准入记录原样保留，不能用本阶段许可去重启其搜索。test是项目已反复揭示的内部报告集，非pristine blind、非外部或官方隐藏榜。主模型没有用这些患者训练，但项目历史知晓测试结果；不得称全历史盲。

测试清单、cache index/images/labels完整SHA由既有weak_elastic_test270_helpers常量提供，全部在TEST_PREACCESS绑定并核验。源码、IDEA、代表规则、已通过开发/在线报告、权重、输出路径、六行与完整统计门在像素/模型推理前冻结。旧hash绑定脚本不修改。先fit前8图像/标签smoke，验证新入口评分及在线/原子parity。

测试Dice沿用历史GPU float32计数除法，另存exact_count_dice用于独立核验；不混用开发float64尾差冒称提分。绝对/配对10,000次患者簇百分位95%CI，seed2026091161+model_ordinal，未作多重比较校正。Scan按扫描等权，Subject先患者内均值再患者等权。Point为最高raw-logit体素是否在WT。附旧总体/单次前向点估计比较，只描述，不据此选方案或改主门。

## 6. 成功、失败与停止

每个模型pruned对full8：Scan/Subject点估计差≥−0.003，两个配对95%CI下界≥−0.005，Point≥max(identity,full8)−1，固定前向≤4；两个模型全部通过才称固定内部确认门通过。它是工程容差门，不是正式非劣统计。其已测阶段speedup必须≥1.5；本阶段若在线/原子数值不exact立即实现失败，不放宽阈值。

六行全部更新MD/JSON排行榜，包括控制/负结果。无论结果好坏一次即关闭，不按test调候选、阈值、epoch、checkpoint、TTA（Test-Time Augmentation，测试时增强）/权重、病例或后处理。确认失败只能记录为开发到报告队列不稳定，不挑某个seed重新定主模型。独立读取保存logits/掩膜重算指标属于审计，不是模型重测。

## 7. 成本与风险

0训练/新权重。正式前向=270×[(8+代表11视图数)+(8+代表12视图数)]；代表预期2/3时为5670，最终以冻结清单计算，smoke=8×同括号。存六套float32输出约12.656GiB和packed masks约405MiB于/tmp；启动前要求20GiB可用，当前约5.2TiB，root不落大cache。峰值预计<12GiB，单次正式执行安全上限30分钟。真实部署只执行2/3视图，而非16+5评测前向；端到端预处理加速尚未证明。

风险：重复测试的历史选择偏差、开发后代表收束、两seed非独立队列、原训练来源字节parity限制、小幅精度损失可能不能复现。均保留，不以“持续工作”掩盖。

## 8. 最终结果

待一次固定报告及独立保存输出核验。测试访问状态以TEST_PREACCESS/STARTED及RAW记录为准。
