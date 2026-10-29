# 无训练提升Dice：网上原始资料与本轮落实

固定用户选择的单T1ce、SegResNet32通道171轮。这里的“无训练”指网络权重/归一化状态不更新；全球温度、权重和形态规则允许按复用270反馈选择，所以本轮全部test-tuned。不是新隐藏榜、独立泛化或首创方法证明。实现/预测函数不接受GT或病例ID查表。

| 方向 | 原始资料 | 本轮具体落实与边界 |
|---|---|---|
| 医学TTA、输入噪声与不确定性 | [Wang等，Neurocomputing2019](https://arxiv.org/abs/1807.07356) | 八镜像、轴置换、相位、轻微尺度、noise64；变换后还原同一native栅格。论文的brain任务支持候选方向，不承诺当前模型涨点。 |
| 冻结模型多视图是否会饱和 | [PIE预印本2026](https://arxiv.org/abs/2609.21412) | 该心脏/眼底工作报告视图饱和、一些后处理反而变差；只是预印本和其他任务。我们因此同时记录1/64/128/256/512成本与负结果，不能把“更多”当保证。 |
| 下采样对整数位移敏感 | [Zhang，ICML2019](https://arxiv.org/abs/1904.11486) | 不重训BlurPool，改为原图每轴保留全部8相位、加零填充后精确还原；flip0/7分别512枚举。填充改变GroupNorm空间统计，另存单phase控制避免混淆。 |
| MC Dropout | [Gal/Ghahramani，ICML2016](https://arxiv.org/abs/1506.02142) | 仅启用已有Dropout层、归一化eval，p0.05/0.1/0.2、16样本；原训练p0.1之外只算敏感性候选，不给其他p套贝叶斯解释。 |
| 概率置信度与校准 | [Mehrtash等，2020](https://arxiv.org/abs/1911.13273) | 同时比较视图前温度、融合后温度/偏置、概率与Logit融合；不能假定高置信必可靠。大网格裸最高反而用负置信指数，不能宣称该符号普遍最好。 |
| F1/Dice与固定0.5阈值 | [Lipton等，2014](https://arxiv.org/abs/1402.1892) | 固定阈值和按自身概率决定体积都测试；理论的校准条件不能直接套未校准CNN。没有GT逐例最优阈值或真实体积输入。 |
| 排名型Dice解码 | [RankSEG，JMLR2023](https://www.jmlr.org/papers/v24/22-0712.html)、[官方实现](https://github.com/rankseg/rankseg) | 官方0.0.8 BA/RMA隔离/tmp跑，160配置裸最高0.782628。自写期望体积比值/改beta不冒充精确RankDice；更灵活校准与官方有限格不是公平理论否定。 |
| 标签共识与估计来源质量 | [STAPLE2004](https://pubmed.ncbi.nlm.nih.gov/15250643/)、[SIMPLE2010](https://pubmed.ncbi.nlm.nih.gov/20667809/) | hard-EM、fractional-vote EM、预测一致性加权/挑选。当前赢家是预测Dice一致性加权Logit，不是STAPLE。八个同模型视图相关，不能假定独立专家。 |
| 概率标签的融合 | [PSTAPLE2013](https://pubmed.ncbi.nlm.nih.gov/23744673/) | 只借鉴无GT估计共识思路。fractional EM不等同PSTAPLE原式；不训练模板分类器、不加载其他分割网络。 |
| 局部与全连接CRF | [DenseCRF2011](https://arxiv.org/abs/1210.5644)、[DeepMedic2017](https://pubmed.ncbi.nlm.nih.gov/27865153/) | 局部6/26邻域MRF完整搜索裸最高0.781622；另单独实现真正native3D全连接XYZ/I图，27候选带无边/仅高斯控制，正在CPU原地跑。不是拿local近似当完整复现。 |
| 形状距离而非概率平均 | [Shape-based averaging2007](https://pubmed.ncbi.nlm.nih.gov/17283774/) | 八预测阈值图→signed EDT，mean/trim/median/quantiles/min/max与全局距离level，630配置。数据留原worker计算，无GT进场函数。 |
| 连通域、孔、滑窗 | [nnU-Net官方后处理](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)、[MONAI滑窗](https://monai.readthedocs.io/en/stable/inferers.html) | 全局6/18/26连接、绝对/相对体积、最大域保护、封闭孔/高斯/形态组合。当前6连接512过滤有增益；规则只对当前复用队列选优。旧108阶段128滑窗已有记录，本轮主要检验整脑/ROI/相位。 |

当前结果不是只挑正面：普通171推理0.773228；最初固定5视图0.780995；7视图加形态0.783850/Point260；7视图自适应校准加6连通750达到0.784001/Point260；8视图预测一致性加权加6连通512目前0.784139/Point258。各新候选都有完整270独立掩膜/分数核对。独立部署API再次从MRI输入跑270，0像素差、point全同，证明不靠病例阈值查表。仍不构成独立泛化确认。

两个易混淆方向本轮排除：[Tent](https://arxiv.org/abs/2006.10726)会更新归一化仿射参数，不满足当前固定权重；[TTGA/MedIA2026](https://arxiv.org/abs/2406.17608)使用领域微调生成器，不满足只用这一个SegResNet权重的边界。没有为了追热门而偷偷增加模型。

没有给论文收益直接加到本任务分数上，也没有把某个有限格负结果当整个方向无效。所有具体实现、门、患者访问/成本和选择时点见ROUND001–020及对应分析目录；完整快照计数为4072835个配置元组（同掩膜的不同参数可能重复，不等于独立试验次数）。最终仍须到08:41结束后归档与推送。

## 中期新增实测（03:20）

形状signed EDT已完成630配置，最高0.781269641/Point258，没刷新0.784139；保留完整负矩阵。case×pixel联合中最高退化到pixel指数0控制，裸0.783534898，尚不能说pixel权重在case一致性之上有额外贡献。跨方向64相位组合的最高同样是alpha0纯基础控制，裸0.783485779；不能把零相位控制的数值/校准收益算给512相位。全连接CRF和图像支撑域解码仍在运行，未到结题。

## 原生全连接CRF已完成（03:34）

27候选×39＋后校准完整270结束；裸阈值和体积校准最高均是无边项控制（校准0.783491753/Point258）。当前证据没有显示该Gaussian/双边图有限参数格带来额外收益；这不是所有CRF方法无效的结论。若此控制叠加形态后有小增益，也只能归给概率表示/校准/形态，不归给不存在的pairwise消息传递。

2026-10-29 04:10补充：再次检索原始SegTTA https://arxiv.org/abs/2604.17451 核对gamma/contrast/Gaussian blur/noise。现补齐ROUND026本单权重强度TTA，非其多MedSAM2模型复现；28组仅已登记待phase依赖，尚无此分支患者分数。OptTTA https://proceedings.mlr.press/v172/tomar22a.html 和JCSAugment https://pubmed.ncbi.nlm.nih.gov/37276115/ 有学习/优化策略，不直接视为当前固定无训练复现；本轮只用手工有限网格。
