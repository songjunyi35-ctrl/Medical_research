# 实验idea：镜像前后答案一致训练

状态：完成并关闭；一致性机制成立但fresh C/D准确率门失败，未访问任何旧榜单cohort

日期：2026-09-04 UTC

## 1. 现在的问题是什么？

固定Flip8测试时增强（Test-Time Augmentation，测试时把同一图像做可逆变换并融合预测）在test访问前的两个fresh inner复制中，Scan/Subject Dice平均提高`+0.006828/+0.007132`，A/B四个配对置信区间下界都为正。这说明4.70M参数direct locator对空间朝向仍有可复现误差。但当前解决办法每例需要8次locator forward；虽然reused test270点估计达到`0.757270/0.741254/265`，部署成本约8×。下一问题不是在已揭示test上搜索更漂亮的融合，而是只用新的train-role证据，尝试把“镜像后应得到同一答案”写进训练，让identity 1×和固定Flip8都更稳。

## 2. 一句话直觉

让学生把同一道题正着看、镜像看，再把镜像答案翻回来；两份答案必须一致，久而久之学生正着只看一次也不容易被摆放方向骗到。

## 3. 为什么选择这个idea？

- 方法选择由test访问前的Stage A双fresh-inner结果驱动：两个独立模型的Flip8都提高Scan和Subject，证明方向敏感不是单seed偶然。
- 64个train-role病例的冻结尺度审计显示7个非identity翻转的sigmoid soft-Dice一致性损失均值`0.08029`、范围`0.03758–0.19191`；权重`0.1`约贡献`0.008`，只占末期约`0.22`主损失的3–4%，不会让一致性压倒监督。
- Li等人的一手工作在医学图像分割中要求变换前后的像素预测经坐标对齐后保持一致，并覆盖2D/3D任务：[IEEE/PubMed](https://pubmed.ncbi.nlm.nih.gov/32479407/)、[arXiv](https://arxiv.org/abs/1903.00348)。Bortsova等也用双分支和逆变换后一致性损失学习分割等变性：[arXiv](https://arxiv.org/abs/1911.01218)。这些来源支持机制，不保证本项目一定提高。
- 旧KAS负结果把最深encoder feature投影到final logit，并在epoch200后同时加入MSE、Dice consistency和contrastive loss；本方法从随机初始化全程约束同一个final segmentation head在可逆翻转前后等变，张量、时机和唯一因素都不同，因此旧负结果没有否定它。
- 研究发生在test270再次揭示之后，必须标记post-test development；但本方法、loss形式、权重和门槛只由pre-test Stage A、一手文献和train-only尺度审计决定，不使用test270选择任何超参数。

## 4. 具体怎么做？

每个训练batch先走现有的随机翻转、平移、缩放、偏移和噪声增强。随后同一个student看两份输入：第一份保持当前增强后的方向；第二份按固定顺序循环7种非identity三轴镜像之一。第二份输出沿相同轴翻回第一份坐标。

历史监督和知识蒸馏（Knowledge Distillation，KD）完全不变：原方向输出与真实WT标签计算DiceCE，并模仿固定history/EMA老师的逐体素概率。额外一致性只比较两份student最终分割概率：

`L_cons = mean_batch[1 - (2*sum(sigmoid(z1)*sigmoid(invflip(z2)))+1)/(sum(sigmoid(z1))+sum(sigmoid(invflip(z2)))+1)]`。

Candidate总损失=`L_existing + 0.1*L_cons`；Control=`L_existing + 0.0*L_cons`。Control也执行第二次student forward并构造同一loss图，因此dropout随机数、模型状态更新次数和计算路径配平；唯一差别是该项是否产生梯度。

训练完成并冻结四个checkpoint后，独立evaluator在每个新inner同时报告identity 1×与固定Flip8 8×。输出仍是raw logit阈值`>0`的mask和最高logit Point；不接Segment Anything Model（SAM）、连通域或其他后处理。

## 5. 唯一变化与固定项

- **唯一变化：** 最终mask的翻转等变soft-Dice consistency权重，Control=`0.0`，Candidate=`0.1`。
- **固定不变：** T1ce、WT=`{1,2,3}`、image-only brain crop true128³ cache、4.70M SegResNet、init_filters16、dropout0.1、history/EMA teacher、DiceCE+voxel-KL、native-volume sampling、每轮972请求/968消耗/121步、batch8、历史增强、AdamW、late-cosine、固定epoch168、threshold0、Point规则、无SAM/后处理。
- 两臂同初值、同sampler、同增强batch、同7-view cycle、同两次student forward；每个复制只改变一致性项的梯度权重。

## 6. 成功、失败和停止规则

- 新Replicate C/D的inner都为108 scans；两者subject互斥，并排除已访问Flip8 A/B inner的全部subjects。每个复制自身864 fit/108 inner、subject overlap0。
- 四条训练（C/D×Control/Candidate）全部完成固定168轮并冻结后，才允许构造任一inner loader。
- **Primary：** 每个复制Candidate Flip8−Control Flip8的Scan Dice；共同指标为Subject Dice与Point hit。
- **每复制主门：** Candidate Flip8的Scan>Control、Subject>Control、Point delta≥−1；Scan/Subject配对10,000次bootstrap 95% CI下界≥`-0.003`。
- **1×安全/目标门：** Candidate identity的Scan>Control、Subject>Control、Point delta≥−1。
- **机制门：** Candidate最后5轮train-only一致性损失均值≤Control的`0.90×`；两臂初值、采样、增强和reflection schedule工程配对全部通过。
- **跨复制门：** 两个Candidate−Control Flip8 Scan delta平均>`0.001`，Subject平均>0；identity Scan/Subject平均均>0。
- 任一门失败立即停止，不调weight、loss、reflection schedule、epoch、threshold或seed，不加第三复制。
- 本实验不允许访问fixed validation108、outer、confirmation、reused test270、UCSD75或其他已揭示cohort。即使通过，也只能进入一个另行冻结的未见外部cohort或新的患者隔离确认，不能回到test270重测。

## 7. 成本与风险

- 四个模型、两张A100分两波训练；每步在原有teacher计算外多一次student forward。预计双卡墙钟约10–14h、总20–28 GPU-hours。
- 双student计算图可能把峰值从约23GB提高到35–50GB；GPU smoke硬门为≤60GiB，超过即视为实现/资源失败，正式训练不启动。
- 新split、四checkpoint、日志和报告预计<150MB；复用现有7.93GB train cache，不创建新cache，不需要删除材料。
- 部署模型参数不变；identity仍1×，固定Flip8仍8×。若只提高8×而1×失败，整体实验按预注册失败，不把它包装成“已把收益学进模型”。
- 最大风险是共享权重的两分支可能学到平凡的一致而不是更准；小权重、原监督/KD不变、1×/8×双门和fresh C/D会阻止这种假成功。

## 8. 最终结果

- Fresh C identity Candidate−Control=`+0.000316/+0.001386/+1`，Flip8=`-0.001773/+0.000643/+2`；Fresh D identity=`-0.000546/+0.000085/-1`，Flip8=`-0.002063/-0.003967/-1`。
- C Flip8 Scan/Subject 95% CI=`[-0.006636,+0.003320]/[-0.005893,+0.007807]`；D=`[-0.005778,+0.001629]/[-0.009182,+0.000867]`。两复制平均Flip8 Scan/Subject=`-0.001918/-0.001662`，identity Scan/Subject=`-0.000115/+0.000735`。
- 机制门成立：最后5轮Candidate/Control一致性loss比例C=`0.636`、D=`0.648`；工程配对全部exact。但更一致没有更准，直觉的准确率部分不受支持。
- 四模型fixed168并冻结后才访问C/D inner；没有访问fixed validation、outer、confirmation、test270、UCSD75或其他外部cohort。
- 原公式在相等预测时loss非0，训练前透明AMENDMENT_01改为平方分母；没有用inner修公式。
- 关闭当前weight0.1实现，不调参、不加seed、不进榜。完整证据见[RESULT.md](RESULT.md)和[RESULT.json](RESULT.json)。
