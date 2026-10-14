# Experiment idea: voxel-KD后的teacher-free KAS语义同化

Status: preregistered before implementation, smoke, training, or new validation access

## 1. 现在的问题是什么？

当前fixed validation108上，voxel-KD200相对matched supervised control只提高Scan/Subject `+0.000778648/+0.000762243`。随后DCD继续强迫student模仿KD200 encoder detail，selected epoch200反而相对KD200下降`-0.000780964/-0.000635983`。LoMix decoder supervision同样为负。

这些结果共同提示：瓶颈可能不是teacher知识还不够多，而是持续外部约束覆盖了student自身已经有用的表示。需要测试teacher离场后，student能否把已经注入的知识重新整理成自身一致的encoder→decoder语义。

## 2. 一句话直觉

前一阶段老师把知识灌进学生；后一阶段老师离场，让学生用自己的语言重新整理，而不是继续逐字抄答案。

## 3. 为什么选择这个idea？

- 一手依据：CVPR 2026 IAD《From Infusion to Assimilation Distillation for Medical Image Segmentation》提出Knowledge Infusion Stage（KIS，知识注入阶段）与Knowledge Assimilation Stage（KAS，知识同化阶段）。论文Table 9明确把KAS作为plug-in接到OFD、CrossKD、CILD、DIST+、VL2Lite、SIKD等既有KD方法后。
- 项目对应：现有voxel-KD200可作为已经完成infusion的冻结起点；本实验只测试其后的KAS，不重新实现IAD专有KIS。因此方法名必须是“KAS-on-voxel-KD”，不是full IAD。
- DCD负结果没有否定它：DCD继续看teacher并匹配teacher detail；KAS完全不运行teacher，只让student自身最深encoder语义与最终logit一致，并把互补negative representation推远。
- KD200选择边界：起点只因为既有fixed validation选择epoch200；该checkpoint随后已有历史test访问，但本实验不使用test数值选择起点、loss、LR、epoch或checkpoint。
- 论文外推边界：论文是2D SAM-L/LoRA teacher→EfficientNet-B1等student，Synapse/ACDC/Polyp；项目是3D单通道T1ce、二值WT、同一个4.70M SegResNet的teacher-free第二阶段。

Primary sources:

- Accepted paper PDF: <https://openaccess.thecvf.com/content/CVPR2026/papers/Hong_From_Infusion_to_Assimilation_Distillation_for_Medical_Image_Segmentation_CVPR_2026_paper.pdf>
- Official repository: <https://github.com/hjklearn/IAD>
- Binary/Polyp training source: <https://github.com/hjklearn/IAD/blob/main/Polyp_Tool/train_isic.py>

## 4. 具体怎么做？

两个matched arms都从同一voxel-KD200 checkpoint开始，使用fresh AdamW、相同数据/RNG，以固定`1e-5`继续32轮；整个阶段没有teacher forward。

### Matched control

同一plain SegResNet只使用当前DiceCE真实标签监督。这隔离“额外32轮低学习率训练”的收益。

### KAS arm

同一student一次forward得到final raw logit `z∈R[B,1,128,128,128]`和最深encoder feature `F∈R[B,128,16,16,16]`。预先注册、进入optimizer的`Conv3d(128,1,kernel=1,bias=true)`投影F，再用trilinear、`align_corners=false`显式放大到z的空间尺寸，得到raw projected logit `x`。

Reverse feature constraint：

- `L_mse = mean((x-z)^2)`，x与z均不detach；
- `L_dice = mean_b[1-(2 sum sigmoid(x_b)sigmoid(z_b)+1e-6)/(sum sigmoid(x_b)+sum sigmoid(z_b)+1e-6)]`。

Contrastive semantic self-optimization：

- positive pair：raw `z`与raw `x`；
- negative representation：`n=(1-sigmoid(x))+(1-sigmoid(z))`，遵二分类官方源码，不增加论文multi-class路径中的negative Conv；
- 每个volume分别flatten，计算`cos_pos=cos(z,x)`与`cos_neg=cos(z,n)`；温度`t=1`；
- `L_css=mean_b[-log(exp(cos_pos)/(exp(cos_pos)+exp(cos_neg)))]`，实现用logsumexp保证稳定。

总目标采用accepted paper二分类权重`gamma=0.01, delta=1`：

`L_control=L_DiceCE`

`L_KAS=L_DiceCE+0.01*(L_mse+L_dice+L_css)`。

训练后删除projector，只保存和部署原plain student final logit；threshold仍为0，point仍取argmax。

## 5. 唯一变化与固定项

- 唯一变化：KAS arm增加一个训练期registered projector和`0.01*(MSE+DiceConsistency+Contrastive)`；control无此项。
- 固定起点：voxel-KD200 checkpoint，SHA-256 `56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7`；既有validation选择，随后历史test已访问。
- 固定train：756 scans/346 subjects；manifest SHA-256 `a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16`；true-128 cache index SHA-256 `d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d`。
- 固定两臂：seed`20260860`、global batch8、WeightedRandomSampler replacement、756 samples/epoch、shift8、drop_last、fresh AdamW、weight decay`1e-5`、betas`0.9/0.999`、eps`1e-8`、global clip5、bfloat16 autocast。
- 固定第二阶段schedule：32 epochs，project-global epochs201–232；LR恒为history schedule floor`1e-5`；保存global208/216/224/232。
- 固定双卡执行：control独立进程绑定GPU0，KAS独立进程绑定GPU1；不用DDP。两进程所有模型/aux/optimizer/loader构造后重置Python/NumPy/CPU/CUDA RNG到20260860，sampler用独立Generator同seed。记录首batch case IDs、原图/增强图/label tensor SHA，要求两臂完全匹配。
- 固定模型训练范围：两臂都全量更新SegResNet；KAS projector同optimizer/同LR/同weight decay/同clip。论文官方student冻结部分encoder，而本3D适配全网可训练，matched control隔离该差异。
- 固定validation：108 scans/50 subjects，manifest SHA-256 `f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb`，cache index SHA-256 `9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322`。
- 固定推理：4,700,897参数plain locator、raw-logit threshold0、无SAM/TTA/postprocessing，1×forward。

## 6. 成功、失败和停止规则

- Primary：fixed validation108 Scan Dice；Subject Dice为共同promotion指标。
- Selection：global epochs208/216/224/232中最大KAS Scan Dice，Subject Dice只用于Scan完全相同tie-break。
- Promotion全部必须通过：
  1. selected KAS Scan和Subject严格高于同global epoch matched control；
  2. selected KAS Scan和Subject严格高于原voxel-KD200 `0.734384308/0.724434143`；
  3. Point hit不低于KD200 `0.935185185-1/108`；
  4. 相对KD200 paired Scan/Subject bootstrap 95% CI下界均`>=-0.003`；
  5. 两臂first-batch case/raw/augmented/label hashes完全相同；四个保存点reload与stripped deploy max abs difference均0；
  6. implementation、synthetic、paired双卡GPU smoke、hash/role/access gates全部通过。
- Test：仅`all_gates_passed=true`后冻结唯一KAS checkpoint、新test evaluator SHA及pre-access receipt，再一次测reused internal test270。
- Scientific failure：任一promotion gate未过即停止，不改gamma、内部loss权重、projection、detach、contrast聚合、LR、epoch、checkpoint、threshold或postprocessing。
- Technical stop：projector未注册/未进optimizer；任一KAS分量NaN/Inf；KAS-only backward时projector、deepest encoder、final-logit branch任一无finite nonzero gradient；KAS权重0的一步student更新不与control完全一致；两臂batch/增强hash不同；任一arm batch8 peak allocated超过32GiB；checkpoint/deploy不等价。

## 7. 成本与风险

- 硬件现场：2×NVIDIA A100-SXM4-40GB，每张约40.3GB空闲，NV12连接；不是10GB设备。若用户指的是额外10GB软件配额，真实batch8 smoke的32GiB门会暴露不兼容并停止。
- 训练：每臂只有1次student forward；control历史同类约20.07GiB，KAS预计22–26GiB。两臂并行，预计15–30min墙钟；单卡硬门32GiB。
- 磁盘：不建新cache；8个checkpoint和报告预计`<0.2G`，根盘约余20G。
- 推理：projector删除，仍4.70M参数、1×forward。
- 最大风险：官方release关键KIS/KAS代码被注释、loss未定义、projector batch内临时创建；binary注释内部权重与paper冲突。项目预先选择accepted paper等权KAS公式、binary sigmoid negative形式和per-volume cosine，不能声称官方代码精确复现。

## 8. 最终结果（完成后填写）

- Validation：selected KAS232 Scan `0.732915355`、Subject `0.723199040`、Point `0.916667`；相对same-epoch control `-0.001219/-0.001419/0`，相对KD200 `-0.001469/-0.001235/-0.018519`，promotion失败。
- Test（如授权）：未授权、未访问本方法的新test。
- 置信区间/病例分布：相对KD200 Scan CI `[-0.002870,-0.000151]`，Subject `[-0.003062,+0.000420]`；50/108 scans、21/50 subjects改善，58 scans、29 subjects变差。
- 是否支持原直觉：不支持accepted-paper等权gamma0.01的3D KAS-on-voxel-KD适配。
- 学到了什么：KAS raw loss从48.06降到1.54，模块确实收敛；但初期loss尺度过强且最终Dice/Point低于control/KD200。teacher-free control224点估计略高于KD200 Dice但Point少1例，未预注册为test候选。
- 下一步：停止KAS、不调权；下一候选为hard-binned mL1-ACE/校准难例loss，需新预注册。
- 报告/归档/哈希：[RESULT.md](RESULT.md)、[RESULT.json](RESULT.json)；raw validation SHA `c727472954103b1f21b7e2e1bac02ea67a0c85837bde3b359fc13db70e6a6c0c`；selected checkpoint SHA `845a6448a5d472c643146ecd11b43e03f440a29eae34ec029344f47d7acda409`。
