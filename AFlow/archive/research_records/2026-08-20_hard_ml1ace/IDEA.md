# Experiment idea: foreground hard-binned mL1-ACE校准增量

Status: completed fixed-validation negative; stopped without new test access

## 1. 现在的问题是什么？

当前voxel-KD200 fixed validation Scan/Subject/Point为`0.734384308/0.724434143/0.935185`。KAS实验中的teacher-free control epoch224达到`0.734413371/0.724669610/0.925926`，Dice点估计略高但Point少1例；KAS selected epoch232进一步把Point降到`0.916667`。

多次实验显示整体Dice只小幅变化，但少数错误的高logit峰会直接影响automatic point。现有loss没有直接要求“模型说90%时，实际也应约90%正确”。下一步只约束最终前景概率的校准，不再增加teacher、feature或decoder结构。

## 2. 一句话直觉

把置信度分成20个成绩档：模型在某档平均说“90%把握”，但这一档真实只有70%体素属于肿瘤，就惩罚20%的自信差距。

## 3. 为什么选择这个idea？

- 一手论文《Average Calibration Losses for Reliable Uncertainty in Medical Image Segmentation》直接使用MONAI SegResNet、Dice+CE和BraTS。
- BraTS 2021上hard mL1-ACE将三层级类别平均macro-ACE从`0.146`降到`0.132`，DSC保持`0.905→0.905`；soft版本校准更强但DSC降到`0.888`。官方raw CSV中WT单项Dice其实从`0.9212`降到`0.9190`，WT macro-ACE从`0.1577`降到`0.1421`。因此一手证据支持“校准改善且总体Dice大致守住”，不支持WT Dice提升，必须保留项目Dice/Point promotion gate。
- LoMix/DCD/KAS都增加feature或语义约束；hard mL1-ACE只调整最终foreground probability的可靠性，机制不同，已有负结果没有否定它。
- 论文是多通道softmax、含background、Dice:CE:ACE=`1:1:1`、从头长训练。本项目是单通道sigmoid WT、现有Dice:CE=`1:0.1`、KD200后32轮；这是foreground-only二元适配，不是完整复现。
- 起点KD200及matched control均已有历史validation/test证据；本方法不使用test数字选择bins、weight、epoch或checkpoint。

Primary sources:

- Paper v4: <https://arxiv.org/html/2506.03942v4>
- Official repository: <https://github.com/cai4cai/Average-Calibration-Losses>
- Official loss: <https://github.com/cai4cai/Average-Calibration-Losses/blob/main/src/losses/hardl1ace.py>

## 4. 具体怎么做？

只训练一个hACE arm，复用KAS实验已经完成的teacher-free control作为matched comparator。hACE arm从同一KD200开始，使用完全相同的seed、sampler、增强、fresh AdamW、LR和32轮。

student输出raw logit `z∈R[B,1,128,128,128]`。以float32计算`p=sigmoid(z)`。对每个volume的单一foreground channel：

1. 用`torch.linspace(0,1+float32_eps,21)`生成20个等宽hard-bin边界；
2. `torch.bucketize(p.detach(), boundaries[1:], right=false)`生成整数bin assignment；assignment不求梯度；恰好位于边界的概率归较低bin，必须直接使用bucketize，不能用`floor(p*20)`替代；
3. 对每个occupied bin以scatter统计count、`sum(p)`和`sum(label)`；bin内`mean_p=sum(p)/count`保持对p可微，`mean_gt=sum(label)/count`固定；
4. 该volume ACE为occupied bins上`abs(mean_p-mean_gt)`的算术平均；空bin忽略；若整幅GT WT为空，则该volume ACE固定为0；
5. batch loss为8个volume ACE等权平均。

禁止构造`B×20×128³`membership矩阵；实现必须是逐volume、单channel的流式bucketize/scatter。

总目标：

`L_control = existing DiceCE(lambda_dice=1, lambda_ce=0.1)`

`L_hACE = L_control + 1.0 * foreground_hard_mL1_ACE`。

论文/官方使用Dice:CE:ACE=`1:1:1`（配置中各`0.33`）。项目保留历史Dice:CE=`1:0.1`并只加ACE1.0，所以实际`1:0.1:1`未被论文验证，可能过强；这是单因素所需的预先冻结适配，smoke只记录尺度，不调权。

## 5. 唯一变化与固定项

- 唯一变化：增加weight1.0、20-bin、right=false、foreground-only hard mL1-ACE。
- 起点：voxel-KD200 SHA-256 `56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7`。
- matched control：KAS实验teacher-free control，runner SHA `289168d96d3494449d9b822513fd759c69aff1b4d1c502fc59b114c1cde62ea8`，TRAIN_DONE SHA `ce987642413fbf86ccbb45ec263851b8a12cfbec6f21d2325521529b583c6392`；checkpoints208/216/224/232 SHA为`3c62166b...b845/9140f1ef...3b4/5c2746dc...f60c/2d13e2bd...245e`。
- paired train：756 scans/346 subjects；manifest SHA `a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16`；cache index SHA `d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d`。
- seed20260860、batch8、WeightedRandomSampler、756 samples/epoch、shift8、fresh AdamW、LR`1e-5`、weight decay`1e-5`、betas`0.9/0.999`、eps`1e-8`、clip5、bfloat16 model forward、global201–232、checkpoints208/216/224/232固定。
- hACE计算全float32。bin assignment对detached probability生成；bin内mean probability不得detach。
- hACE arm必须匹配control的initial student state、first-batch case/raw/augmented image/label hashes，以及201–232每epochcase-order和CUDA RNG before/after hashes。
- fixed validation108 manifest SHA `f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb`，cache index SHA `9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322`。
- 推理固定：plain 4,700,897参数、raw threshold0、无SAM/TTA/postprocessing、1×forward。

## 6. 成功、失败和停止规则

- Primary checkpoint selection：只按hACE validation Scan Dice最大，Subject exact tie-break；不得按ACE选checkpoint。
- Promotion全部必须通过：
  1. selected hACE Scan/Subject严格高于same-epoch matched control；
  2. selected hACE Scan/Subject严格高于KD200；
  3. Point至少100/108；
  4. 相对KD200 paired Scan/Subject bootstrap95% CI下界均`>=-0.003`；
  5. selected hACE macro-ACE相对same-epoch control至少降低`0.005`，且paired macro-ACE delta的95% CI上界`<0`；
  6. selected hACE official-micro ACE严格低于same-epoch control；
  7. pairing/reload/deploy/implementation gates全部通过。
- Calibration口径：macro-ACE=每scan occupied-bin ACE后scan等权；official-micro遵论文仓库：每bin`mean_p`跨scan等权nanmean，`mean_gt`按bin count加权；另外报告真正pooled-voxel ACE但不用于promotion。所有口径20 hard bins、sigmoid raw foreground probability、无threshold。
- Bootstrap：10,000 paired percentile draws；segmentation按108 scans与50 pre-averaged subjects；macro-ACE按108 paired scan ACE；seeds在pre-implementation amendment冻结。
- Technical stop：边界/bin count不匹配官方reference；空bin产生未处理NaN；ACE对logit无finite nonzero gradient；probability被整体detach；任一loss/gradient非有限；weighted ACE或total gradient非有限/异常clip；pairing evidence不同；batch8 peak allocated超过24GiB；checkpoint/deploy不等价。
- Failure：任一promotion gate未过即停止，不改bins、right、ACE weight、起点、LR、epoch、checkpoint或threshold，不访问test。

## 7. 成本与风险

- 只训练一个hACE arm，复用现有control。每step仍1×student forward；额外是float32 sigmoid、bucketize和流式scatter。
- control formal约18.3min/20.29GiB；hACE预计18–30min、20–22GiB，硬门24GiB。使用一张A100-40GB，第二张空闲。
- 不建cache；四个checkpoint与报告预计80–100MB；根盘约余20G。
- 推理无额外参数、计算或后处理，仍1×forward。
- 最大风险：项目`1:0.1:1`相对官方`1:1:1`让ACE偏强；hard bin只在bin内提供梯度且剥离空间信息；foreground-only边界处不与显式background严格对称。全部在执行前冻结，不据validation修正。

## 8. 最终结果（完成后填写）

- Validation：hACE按Scan选择epoch224，Scan/Subject/Point=`0.723869764/0.713277712/0.953704`（103/108）。same-epoch control为`0.734413371/0.724669610/0.925926`，historical KD200为`0.734384308/0.724434143/0.935185`。
- Test（如授权）：未授权、未访问。
- Calibration metrics：epoch224 macro/official-micro/pooled ACE=`0.125926688/0.071769588/0.071816333`；same-epoch control为`0.230865180/0.197533637/0.197531864`。macro ACE绝对下降`0.104938487`，paired 95% CI `[-0.112169538,-0.097648101]`。
- 置信区间/病例分布：相对control Scan delta `-0.010543607`，95% CI `[-0.014594454,-0.006692960]`；Subject delta `-0.011391898`，95% CI `[-0.017179631,-0.006042880]`。30/108 scans、12/50 subjects改善；78 scans、38 subjects变差。
- 是否支持原直觉：部分支持。它强力改善概率校准并提高Point，但没有守住Dice，因此整体promotion直觉不支持。
- 学到了什么：weight1的foreground hard-bin目标确实被优化，train ACE从`0.201153`降到`0.070369`；不是实现失效。但项目`Dice:CE:ACE=1:0.1:1`对T1ce-only WT continuation过强，校准与重叠质量存在明显冲突，不能事后降权补救本实验。
- 下一步：当前适配停止，不搜索bin数、right、weight、起点、LR、epoch、threshold或后处理。下一候选转向单因素PM Dice难体素调制，并须先重新核验一手论文/实现与预注册。
- 报告/归档/哈希：raw validation `b40b688c5372825894280e45f2392212e5c8f011afc5591b51fbd3d77ebdd477`；selected checkpoint `e773e50f6f5941cdfcf37a60e9915c76b0a86acaec29f3afa4dff9e1789b71ab`；详见`RESULT.md`与`RESULT.json`。
