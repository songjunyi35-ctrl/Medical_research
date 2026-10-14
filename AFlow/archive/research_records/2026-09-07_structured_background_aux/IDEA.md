# Experiment idea: 训练期区分切除腔与普通背景

Status: preregistered before implementation/training/development metrics; user-authorized reused development.

## 1. 现在的问题是什么？

新授权开发池864 scans/391 subjects，固定fit684/313与dev180/78。当前划分没有新模型基线分数。原训练池814/972 scans有切除腔标签4，但WT二分类把它与普通背景合并；训练监督中的语义信息丢失。历史内部总体Full KD168 Flip8的Scan Dice为0.757270（8次前向、复用test）；本实验不能直接将新开发结果与它比较。

## 2. 一句话直觉

告诉学生“这里不是肿瘤”之外，再让它练习辨认“这是切除后留下的腔隙”，可能使共享图像特征更清楚。

## 3. 为什么选择这个idea？

训练标签自身有未用信息。BackSplit的背景细分监督包含SegResNet等骨干实验，为语义背景学习提供一手先例；其统计理论有正确模型设定等条件，并不保证本任务Dice提高。本文不直接复现其softmax多分类输出，而采用保留WT主输出的训练期辅助头。论文：[BackSplit](https://arxiv.org/html/2511.19394v2)。

此前中心点头学几何位置，多尺度监督学不同分辨率，均未检验原标签4的语义背景监督。已揭示外部结果不参与权重、epoch、阈值或病例选择；本实验也不声称解决某一外部队列错误。

## 4. 具体怎么做？

T1对比增强MRI经现有N4偏置场校正、图像裁剪、标准化和128³缓存输入网络。共享SegResNet最后输出两个通道：WT（whole tumor，全肿瘤）主通道与RC（resection cavity，切除腔）辅助通道。主通道沿用Dice与二元交叉熵；辅助通道只在真实WT之外学习RC相对普通背景，使用masked Dice与二元交叉熵。两通道共同接收同一翻转/平移后的标签。

两臂从相同随机初始化训练，均无知识蒸馏教师、无历史checkpoint暖启动。Control辅助权重0；candidate固定0.1。部署裁掉辅助通道，只读主通道logit>0及其最大点，一次前向；不会将预测RC乘进WT或根据GT避让。

## 5. 唯一变化与固定项

- 唯一变化：训练辅助损失系数0→0.1。
- 固定项：双通道结构/初值、fit684/dev180、现有128³图像/WT、原生体积采样、batch8、972 draws/968有效draw/121步、168轮、学习率计划、AdamW、增强、主损失、阈值0、无测试时增强/Segment Anything Model/后处理；两臂均无教师。
- 新baseline是teacher-free paired control；和历史知识蒸馏（Knowledge Distillation, KD）冠军比较不构成单因素证据。

## 6. 成功、失败和停止规则

固定两seed，四训练全部完成冻结后才访问dev一次。主要指标为Subject Dice，Scan Dice和Point一并报告。两个seed各自Subject与Scan差都>0、Subject差都≥0.003、Point各最多损失1例；每seed Subject和患者聚类Scan配对95%CI下界均≥−0.003；两seed平均Scan差≥0.003。全部满足才称开发支持，否则当前配方失败/不确定。seed复核不是独立患者复现。

工程门：fit/dev/protected患者不交叉；原始RC标签合法且重建WT与既有WT逐例byte exact；两臂初值/采样/增强一致，模型与loss有限、辅助梯度有效、裁除后主输出一致。任何工程失败先停并记录范围，不改正式超参；每臂>6h或>24GiB停止。失败不改0.1、不延长epoch。开发支持不自动授权test/外部。

## 7. 成本与风险

四条168轮轨迹，双A100两批；每臂上限6h/24GiB。fit RC uint8缓存684×128³约1.34GiB，不重建影像cache。最终裁掉辅助头，主干约4.70M参数、1次前向。可能辅助任务与WT竞争；RC类别不均衡；复用开发集有历史适应偏差。只有新数据才能提供独立确认。

## 8. 最终结果（完成后填写）

训练与开发尚未运行。结果另存RESULT.md/JSON，不覆盖本冻结七问。无test/outer/confirmation/外部新访问。
