# Experiment idea: 3D Voxel-wise Modulated Dice（VM Dice）

Status: completed fixed-validation negative; stopped without test access

## 1. 现在的问题是什么？

当前最佳单模型voxel-KD200在fixed validation108的Scan/Subject/Point为`0.734384308/0.724434143/0.935185`。它相对matched control平均只提高Scan `0.00077865`，而最差10个case占全部负收益约63%。

只读错误分层显示，KD200 Dice与GT体积Spearman `rho=+0.484`，与暴露边界/体积代理`rho=-0.675`，与26邻接连通域数`rho=-0.378`。边界复杂度最高四分位27例的KD-control平均delta为`-0.001202`，仅9例改善、18例变差；其余三个边界四分位均为正。问题集中在高边界、低填充、碎片化的困难尾部，而不是所有小病灶都差。

## 2. 一句话直觉

像老师批卷时不再反复奖励已经会的题，而让Dice把主要注意力放到当前预测错误最大的体素，尤其是模糊边界和碎小病灶。

## 3. 为什么现在选它？

- 2025/2026论文《Pixel-wise Modulated Dice Loss for Medical Image Segmentation》提出按当前预测误差给每个pixel/voxel动态加权；作者把3D推广称为Voxel-wise Modulated Dice（VM Dice）。
- 当前唯一稳定的形态盲点是高边界/低填充困难尾部，和论文的“difficulty imbalance、边界体素”动机直接一致。
- hACE只校准概率分布，虽然Point和ACE改善却损害Dice；VM Dice保留空间位置并直接优化Dice几何，因此hACE负结果没有否定它。
- 论文ACDC中PM Dice mDice `90.61`，CE+Dice `89.94`；MSSEG脑MRI 2D切片中PM Dice `69.07`，CE+Dice `67.23`。但没有BraTS或真正3D网络证据，只报告3-run均值、无置信区间，属于高风险外推。
- 没有官方代码；只能按论文公式独立实现并用手算reference锁定。

Primary source: <https://arxiv.org/pdf/2506.15744>

## 4. 具体怎么做？

两张A100各训练一臂，都从同一个KD200 checkpoint开始，不运行teacher。

student输入T1ce，输出原始foreground logit `z`。以float32计算foreground概率`p_fg=sigmoid(z)`，显式构造两类概率`p=[1-p_fg,p_fg]`和两类GT `y=[1-y_fg,y_fg]`。

每个volume、每个GT中存在的class计算：

`m = |y - stopgrad(p)|^gamma`

`score = (2*sum(m*y*p)+1e-6) / (sum(m*(y^2+p^2))+1e-6)`

先对该volume存在的foreground/background类别平均，再对batch平均，`L_VM=1-mean(score)`。`m`必须detach；Dice分子/分母里的正常`p`不得detach。

- Arm A：`gamma=0`，此时`m`严格为1，是论文自身的squared-denominator Dice control。
- Arm B：`gamma=1`，论文默认VM Dice。
- 两臂总loss均为`L_VM + 0.1*BCEWithLogits(z,y_fg)`。

保留CE0.1是为了当前项目单因素一致性；论文通常发现PM Dice单独使用更好，但项目不得同时删CE。推理仍只用原始final logit阈值0。

## 5. 唯一改变什么，哪些保持不变？

唯一因果变化是`gamma:0→1`，即是否启用difficulty modulation。新跑gamma0 control是必须的，因为历史MONAI Dice使用非平方分母且只含foreground，不能隔离论文modulation。

固定：

- 起点KD200 SHA-256 `56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7`；其历史validation已选择且随后访问过test，当前方法不引用test数值调参。
- 756 scans/346 subjects train cache和manifest不变；true-128³；seed20260870；batch8；weighted replacement；shift8。
- fresh AdamW、LR`1e-5`、weight decay`1e-5`、betas`0.9/0.999`、eps`1e-8`、clip5。
- global201–232，共32轮、每轮94 steps；保存208/216/224/232。
- 两臂初始state、case顺序、首batch raw/augmented image/label、每epoch CUDA RNG before/after必须完全一致。
- fixed validation108、raw threshold0、无teacher、SAM、TTA或postprocessing；部署4,700,897参数、1×forward。

Source使用Adam、weight decay`1e-4`、poly LR（`1e-4`或`5e-4`），但没有报告总epoch，无法忠实比例映射。项目32轮`1e-5`是强KD basin上的明确适配，不是原论文训练复现。

## 6. 什么算成功，什么立即停止？

Checkpoint selection只按gamma1 arm的validation Scan Dice最大、Subject exact tie-break；不得按hard-stratum、Point或任何test信息选点。

Promotion全部必须通过：

1. selected gamma1的Scan/Subject严格超过same-epoch gamma0 control；
2. selected gamma1的Scan/Subject严格超过historical KD200；
3. Point至少保持KD200的101/108；
4. 相对gamma0与KD200的paired Scan/Subject bootstrap 95% CI下界均`>=-0.003`；
5. 冻结的boundary最高四分位27例、KD200 Dice bottom27例，gamma1对same-epoch gamma0的mean Dice delta都必须`>0`；
6. 全部implementation、pairing、reload、deploy、access门通过。

Technical stop：gamma0不等于手算squared Dice；modulation未detach或正常概率路径无梯度；高错误体素权重不高于低错误体素；任一loss/gradient NaN/Inf；gamma0/gamma1 pairing不同；任一arm batch8 peak allocated超过24GiB；一步外推超过45分钟；checkpoint/deploy不等价。

失败后不得搜索gamma、epsilon、background规则、CE权重、LR、epoch、threshold或postprocessing；不得访问test270。

## 7. 时间、显存、磁盘和推理成本

- 两张A100并行，每臂1×student forward；预计20–25min墙钟。
- 每卡预计20.5–21.5GiB allocated，24GiB硬门。
- 不建新cache；8个plain checkpoint和报告预计约150MB。
- 训练只新增float32 sigmoid、detached modulation和少量逐体素乘加。
- 推理无额外loss、参数、teacher或forward，仍4.70M参数、1×forward。

## 8. 最终结果（完成后填写）

- Validation：gamma1按Scan选择epoch224，Scan/Subject/Point=`0.732177684/0.721937736/0.953704`。同轮gamma0为`0.732818245/0.722533978/0.953704`，KD200为`0.734384308/0.724434143/0.935185`。
- Test：未授权、未访问。
- Hard-stratum mechanism：boundary-top27 mean delta `+0.000902898`，但KD200-bottom27 delta `-0.001494994`；只支持高边界子组，不支持更广困难尾部。
- 置信区间：相对gamma0 Scan/Subject delta=`-0.000640561/-0.000596242`，95% CI=`[-0.001979059,+0.000667640]/[-0.002381061,+0.001346782]`。相对KD200 delta=`-0.002206624/-0.002496407`，CI下界`-0.004481/-0.005445`，安全门失败。
- 是否支持直觉：部分机制支持、整体不支持。VM loss从`0.885725`降到`0.297783`，boundary-top27改善，但overall和bottom27变差。
- 下一步：当前gamma1适配停止，不搜索gamma、epsilon、CE、epoch、LR、threshold或postprocessing。若继续困难尾部方向，优先case-level FairDRO而非另一体素loss；必须重新七问和预注册。
