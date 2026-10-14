# Experiment idea: 答案标签不变的轻度图像模糊增强

Status: conditional planned, not formally trained. Rotation has priority.

## 1. 现在的问题是什么？

患者均衡两个模型开发Scan0.753475/0.755906、Point167/168；相比native170/163，其seed11点不稳，整块分割的改善没有稳定转成最高分点命中。方向翻转seed12又损失8点。细碎图像信息可能被过度依赖，但这是待检验解释，并非因果结论。

## 2. 一句话直觉

学认略微失焦的照片，不只抓一个细亮点；标准答案保持原样。

## 3. 为什么选择这个idea？

原输入增强包含噪声却没有模糊；nnU-Net官方实现有GaussianBlurTransform，sigma0.5—1，外概率0.2、通道概率0.5：https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py 。本项目单通道总概率0.2、三轴共用sigma，因此是单因素适配，不是官方概率精确复现/最佳参数。
Gamma改变全局明暗关系，软占比标签改变监督，弱弹性和旋转共同移动图与标签；本次仅改变训练图像局部清晰度，既往负结果未直接检验它。模糊可能反而损失T1ce微小肿瘤线索，不能保证涨分。

## 4. 具体怎么做？

原基础增强之后，独立CPU流seed+300000抽取20%病例；每例sigma均匀[0.5,1]输入体素，三轴同sigma，固定7点归一高斯核、逐轴卷积、reflect边界。只处理image，标签返回原tensor而不插值；未选图像byte exact。原患者等概率采样的fit684经原4.70M网络、Dice加0.1交叉熵学习；无教师、RC辅助系数0。推理用原图，logit>0及全图argmax，不加模糊或点规则。

## 5. 唯一变化与固定项

唯一变化为额外训练图像模糊。两个seed2026090711/12均为subject_blur，对照是各自固定患者均衡168轮旧模型；同时与原native对照比较。fit684/313与reused dev180/78、原硬cache、初始化、每轮972抽968、batch8、121步、优化器/学习率/168轮/基础增强与随机流、标签字节、网络、推理1次前向不变。不混入旋转，不选择最好seed。

## 6. 成功、失败和停止规则

仅在旋转无合法晋级资格、所有旋转GPU作业结束且剩余足够完成两模型和核验时启动。工程：核归一核/反射边界/独立SciPy滤波近似/标签byte exact、原随机流、fit8六步配对GPU smoke<24GiB及导出exact。双seed每个相对native Subject>=+.003、Scan>0、Point>=-1，两患者簇95%CI下界>=-.003，两seed平均Scan>=+.003；另各自相对患者均衡对照Scan>0、Subject>0、Point>=0。一次统一dev读出、原硬标签，bootstrap10000次患者簇，统计seed2026091061/62，无多重选择校正。
没有全门通过就关闭，不改sigma/p/kernel/seed/epoch/threshold。只过门时才另冻结同test270的一次固定seed11面板（单模型与八镜像均保留），标fit684、不是full864，不挑较好seed。当前不访问test；如果已有本窗口test面板，不再追加第二面板。所有正式计算21:24:55 UTC前完成，15分钟归档。时间不足不跑残缺epoch当正式终点。

## 7. 成本与风险

两模型双卡预计2–2.3h，4–4.6GPUh，约21GiB，各恢复点+终点合计<200MiB，无新像素cache；部署仍4.70M/1次前向无额外模块。风险是模糊削弱微小高对比信号，以及反复复用78人开发的过拟合；不称blind/外部/泛化已确认。

## 8. 最终结果（完成后填写）

尚未训练/开发，无新checkpoint和CI。优先准备CPU工程，不能抢占旋转GPU。旧test270/119 overall Scan.7572696616942132（8×）；1×Scan.7520031755617648。
