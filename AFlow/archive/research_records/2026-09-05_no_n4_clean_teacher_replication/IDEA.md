# Experiment idea: 不额外做N4亮度偏置校正，教师与学生全链路隔离验证

Status: preregistered before split generation, cache creation, new teacher/student implementation or training, and E/F inner metrics.

## 链路审视与启发

[全链路审视](../../reviews/2026-09-05_30h_chain_audit/SESSION.md)发现两个尚未解决的问题：输入数值单位通过N4影响最终张量，以及旧fresh-inner虽与学生fit隔离却曾被固定教师训练过。旧A/B/C/D与teacher训练集重叠87/72/88/72 scans，所以本次先重建每fold自己的教师。另把现存全部inner清单和旧selector/confirmation患者保守排除后，972训练池仅剩38 subjects/87 scans可作新内部评测；不能声称还有两套完全新108例。

至少三条数字启发：旧brain crop约+0.054250、完整972轨迹对control验证Scan+0.010943支持先检查输入/数据；160³和width24 fresh final -0.002872/-0.003525限制了堆计算；一致性loss下降35–36%却无准确率收益说明机制和主指标要双检。本次真实输入零点解码支持/包围盒6/6 exact，但N4后最坏差0.057679、平均0.002245，暴露强度单位敏感性。以上来自已归档结果及train图工程检查，不使用新test指标选择设置。

## 1. 现在的问题是什么？

原始MRI经过N4（非参数非均匀强度校正的一种实现，校正缓慢变化的亮度偏置）后才进入当前4.70M自动模型。但没有当前自动主线的匹配有/无N4全程训练比较。同一训练图的比例编码竟令最终128³张量最大差0.057679；RHUH80又全部是含负数的发布格式，不能原样套正值N4。当前reused test270总体点估计0.757270/0.741254/265来自固定八镜像，无法替代新开发证据。

## 2. 一句话直觉

照片已经经过整理后，额外自动补光不一定总有帮助；让两位学生分别看“补光”和“未补光”的同一教材，比较谁在新患者上学得更稳。

## 3. 为什么选择这个idea？

- ITK官方N4说明明确指出负值及小于1的强度可产生差结果；该算法包含对数直方图、反卷积、迭代拟合，不能把理想尺度不变当作实际数值保证。
- BraTS2024官方源数据经过脑提取与空间配准等处理；是否再加N4有益，需要匹配实验，不假定原始发布已经做过N4。
- 百分位裁剪去掉亮暗尾部，本方法不裁剪正强度；Gamma随机改曲线，灰度refiner改预测，均不同于移除一个固定的输入校正步骤。
- 历史upstream-literal无N4是另一交互模型/裁剪协议，不否定当前自动SegResNet的匹配比较。全图160³、width24、多老师加权、困难loss和一致性loss均不重启。
- [ITK官方说明](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1N4BiasFieldCorrectionImageFilter.html)、[N4作者官方算法说明](https://github.com/ANTsX/ANTs/wiki/N4BiasFieldCorrection)、[BraTS2024数据论文](https://arxiv.org/html/2405.18368v1)。这是假设检验，不是论文复现或已证明的改进。

## 4. 具体怎么做？

两组学生均从随机初始化训练168轮。Control输入为原N4→图像正值脑区裁剪→源非零总体z-score→128³；Candidate只去掉N4，保留原正值支持（原来≤0清零规则两组一致），随后相同裁剪、标准化和缩放。图像、标签空间一一对应，WT仍取原标签1/2/3、排除4。

为解决教师泄漏，每fold只用其930 fit病例新训两位独立教师：一位保留168轮raw权重，另一位保留从epoch100末开始的指数移动平均（EMA，Exponential Moving Average，逐步平滑参数）168轮权重。两位教师都只使用Control输入，不见本fold 42例inner。然后两组学生共享冻结的0.75 raw+0.25 EMA输出作为知识蒸馏（KD，Knowledge Distillation）目标，原监督Dice与交叉熵（CE，Cross Entropy）及逐体素概率分布差异（KL，Kullback–Leibler divergence）目标不变。两组教师接收各自学生的增强输入；唯一区别仍是输入N4有无。

全部4教师和4学生训练完成、检查点冻结后，E/F各评一次，同时固定报告identity单次前向和测试时增强（TTA，Test-Time Augmentation）的全部8个三轴镜像平均。所有模型输出阈值0、point为最高raw logit，无其他后处理。

## 5. 唯一变化与固定项

- 每fold学生pair唯一变化：是否做N4输入偏置校正。
- 固定：fit病例、教师权重、初始学生参数、sampling、增强随机数、标签、模型4.70M、batch8、AdamW、968有效draw/121步/epoch、学习率schedule、固定epoch168、损失系数、部署阈值和视图公式。
- 教师重训是两组共享的防泄漏实验基础设施，不能把新Control称为历史Full972原checkpoint的直接复现；阶段结果只比较同fold的新匹配学生。
- E/F各42 scans，患者互斥且排除所有冻结旧评测患者。每fold fit=Full972减本fold inner，930 scans。另一fold inner可进入本fold fit，但不允许训练/选择使用本fold inner。所有结果一起揭示，禁止跨fold先看后改。
- 全链路患者隔离针对本次新教师/学生训练；整个研究已有post-test开发历史，不能宣称从未接触过这些训练池患者或pristine blind。

## 6. 成功、失败和停止规则

- 主指标：每fold identity Scan/Subject Dice均严格提高，Point至多少1/42，两项患者簇配对95% CI下界均≥−0.003。
- 次要稳健门：每fold固定Flip8也须同时提高Scan/Subject，Point至多少1，两个同样CI安全门。
- 跨fold：平均identity Scan差>0.001、平均Flip8 Scan差>0.001，两个Subject平均差>0，所有工程/访问门通过。
- 10,000次patient-cluster bootstrap；每次重抽患者并保留其全部时间点，分别计算扫描等权和患者等权统计，避免把同一患者的多次扫描当独立样本。没有另选bootstrap种子或靠扫描层窄区间放行。
- 样本小且两fold来自同一训练池，并非完全独立外部复制；CI预计较宽，只支持严格定义的开发比较，不宣称显著或普遍适用。
- 任一失败：关闭当前no-N4实现，不改变N4开关/参数、epoch、teacher、seed、视图、阈值或门槛，不新增第三fold，不访问旧fixed validation/outer/confirmation/test270/UCSD75。
- 全过仅允许Full972 fresh教师与配对学生按固定epoch168重训并冻结；不自动允许任何外部/旧test评测，新外部需独立预注册。
- 工程失败须在相应访问前记录并修接线；方法公式或门槛不得根据结果改变。

## 7. 成本与风险

- Stage A：4次教师+4次学生训练，2×A100两组共4波，预计10–14h墙钟，20–28 GPU-hours。每模型硬6h/24GiB门。正式launch必须通过比例GPU smoke。
- 新no-N4缓存972×128³×5 bytes约10.19GB（9.49GiB），放/tmp，先查磁盘；不删除旧材料。8个checkpoint、日志/报告约0.5GB。
- 通过才Stage B：2次新Full972教师+2次配对学生，另约5–7h墙钟，最多新增约0.25GB；若剩余30小时窗口不足以完成，不开始必然跨窗的正式阶段，归档等待新窗口。
- 部署仍4.70M、identity1×或固定Flip8 8×，不额外部署教师；Candidate省掉N4 CPU步骤。是否更准尚未知。
- 风险：N4可能确实提供重要的偏置校正；小样本大CI与新教师质量限制结论。不能用旧高分老师或旧验证集弥补结果门失败。

## 8. 最终结果

尚未运行；结果将另存RESULT.md/JSON，保持本预访问文档哈希不变。
