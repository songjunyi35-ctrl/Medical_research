# Experiment idea: DCD方向细节蒸馏增量

Status: preregistered before implementation, smoke, training, or new validation access

## 1. 现在的问题是什么？

当前 fixed validation108 上，matched supervised continuation epoch200 的 Scan/Subject Dice 是 `0.733605660/0.723671900`，voxel-logit KD epoch200 是 `0.734384308/0.724434143`。知识蒸馏相对同训练量 control 只多 `0.000778648/0.000762243`。刚完成的 LoMix-3D selected epoch200 反而只有 `0.732479468/0.721607103`。

这说明 decoder 多尺度监督没有突破当前平台。下一步要测试一个不同位置的机制：普通输出蒸馏可能保住“肿瘤大概在哪”，但没有专门保护 encoder 反复下采样时最容易丢失的方向边缘和细小结构。

## 2. 一句话直觉

普通KD像临摹整幅画，大片背景容易盖住细线；DCD先用“小波筛子”挑出轮廓和方向纹理，再只要求学生临摹这些最容易消失的细线。

## 3. 为什么选择这个idea？

- 当前现象：voxel-KD 相对matched control只有约`+0.00078/+0.00076`；LoMix的decoder supervision为负，说明需要从encoder detail而不是更多decoder输出入手。
- 一手证据：MICCAI 2026论文《Detail Consistent Stage-Wise Distillation for Efficient 3D MRI Segmentation》直接研究三维MRI。其BraTS 2024压缩student mDice从`63.60%`升到`68.51%`，细节敏感NETC从`41.26%`升到`54.36%`；官方代码公开了3D DWT/IDWT实现。
- 既有负结果没有否定它：160³改变整图输入网格；ReCo-KD处理输出区域；LoMix处理decoder supervision；DCD处理encoder方向detail subspace。Cross-seed parameter soup直接平均不同权重，而本实验的detail teacher KD200由history168同一路径继续训练，hidden-channel basis更可能对齐。
- teacher选择边界：KD200只因既有fixed validation选择了epoch200且与history168同初始化路径而冻结为detail teacher；不用其历史test结果选择teacher。该checkpoint后来已有历史test访问，必须在provenance中披露，但权重没有因test改变。无需数据的权重几何审计显示KD200/history参数cosine `0.999208`、relative-L2 `0.03987`，四级合成feature cosine为`0.99933/0.99739/0.99289/0.98941`；cross-seed EMA/history参数cosine仅`0.369009`且深层feature接近0或负相关，因此不直接做EMA hidden-feature MSE。
- 外推边界：论文是四模态、多类别、约101.9M teacher→6.38M r=4压缩student、nnU-Net、1000轮；本项目是单通道T1ce、二分类WT、同容量4.70M SegResNet、history168后32轮。因此这是“DCD对当前voxel-KD的增量适配”，不是论文复现。

Primary sources:

- Paper: <https://arxiv.org/html/2605.26382v1>
- Official repository: <https://github.com/ClinicaAlpha/DCD-3D-MedSeg>
- Official DCD implementation: <https://github.com/ClinicaAlpha/DCD-3D-MedSeg/blob/9d3684ffd58333874109bb8efebb8c933fa5ac66/dcd_medseg/training/distillation/methods/dcd.py>
- Released BraTS config: <https://github.com/ClinicaAlpha/DCD-3D-MedSeg/blob/d3c27af9e4d1efdd2080bd1a882350566c427c32/configs/dcd_brats2024_gli.yaml>

## 4. 具体怎么做？

同一增强后的128³ T1ce每个训练step走四条路径：

1. student从historical epoch168初始化，产生最终WT logit和四级encoder feature：`16@128³, 32@64³, 64@32³, 128@16³`。
2. 原有voxel-logit teacher保持不变：冻结history168与EMA168，输出logit按`0.75/0.25`混合；现有温度`T=2`、权重`0.3`逐体素KL完全保留。
3. 新的detail teacher固定为voxel-KD200，只提供同四级encoder feature；teacher全程eval、无梯度。
4. 每级student/KD200 feature逐通道做3D离散小波变换：`db4`、3 levels、filter length8、padding7、stride2。8个band顺序为`LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH`，固定只保留index1–6，排除低频LLL和噪声风险最高的HHH；每level用ConvTranspose3d逆变换并crop回输入shape，得到detail-only空间feature。每stage的3个level MSE取mean，四个stage loss直接sum。student/teacher通道相同，因此遵官方代码不加projection。

官方源码方向审计边界：官方用PyWavelets `dec` bank进入PyTorch `conv3d`，再用`rec` bank进入`conv_transpose3d`；但PyTorch前者是cross-correlation，转置卷积已是其伴随。固定数值核验中，官方组合的全8-band重建MSE约`1.4–1.7`，而synthesis继续使用同一`dec` bank时MSE约`2.6e-13`、max error约`4.5e-6`。本实验在任何执行前固定论文语义正确版：analysis/synthesis都使用同一冻结db4 decomposition bank，并要求全8-band重建max error `<1e-5`。这是一项公开披露的方向修正，不会在validation后与官方代码版二选一。

总目标是在现有voxel-KD上只加DCD：

`L_total = L_DiceCE + 0.3 L_voxelKD + mu(k) * sum_stage(mean_level(MSE(detail_student, detail_teacher)))`。

论文正文写`mu=0.05`，但官方首发至今的BraTS YAML使用最大`0.01`和`cosine_healing(start150,ramp800,total1000)`。本实验优先遵循可执行BraTS配置，并把进度比例一次性映射到32轮continuation：local zero-based index `k=global_epoch-169`，`start=5`、`ramp=26`：

- `k<5`: `mu=0`；
- `k>=5`: `mu=0.01*0.5*(1-cos(pi*min((k-5)/26,1)))`。

global176/184/192/200对应`k=7/15/23/31`和`mu=0.000145291/0.003226976/0.007840324/0.010000000`。32轮平均mu为`0.00421875`。

训练后只保存和部署student final model；history、EMA、KD200 teacher以及DWT/IDWT模块全部删除。最终阈值仍为raw logit `>0`，point仍取argmax。

## 5. 唯一变化与固定项

- 唯一变化：在现有voxel-KD arm上增加冻结KD200 teacher的四stage DCD loss及其预先冻结的mu schedule。
- 固定student：historical true-128³ epoch168，SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`。
- 固定voxel-logit teachers：history168与EMA168；EMA168 SHA-256 `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`；logit formula、T=2、weight0.3不变。
- 固定detail teacher：voxel-KD200，SHA-256 `56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7`；teacher由既有validation选择，后来已有历史test访问，但本实验不使用test数字选择或调参。
- 固定train：756 scans/346 subjects，manifest SHA-256 `a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16`；true-128 cache index SHA-256 `d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d`。
- 固定训练协议：seed20260846、batch8、weighted replacement sampler、756 samples/epoch、shift8、AdamW、weight decay`1e-5`、global gradient clip5、bfloat16 autocast、global epoch169–200 late-cosine、checkpoint176/184/192/200。
- 固定模型/随机流：student、history teacher、EMA teacher、KD200 detail teacher和固定wavelet modules全部构造/加载后，重置Python/NumPy/CPU/CUDA RNG到20260846，使student dropout与base voxel-KD可比。student EMA不加入。
- 固定validation：108 scans/50 subjects，manifest SHA-256 `f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb`，cache index SHA-256 `9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322`。
- 固定推理：plain 4,700,897参数SegResNet，raw-logit threshold0，无SAM、无TTA、无postprocessing，1× locator forward。
- base comparator：既有voxel-KD checkpoints global176/184/192/200和`KD_VALIDATION.json` SHA-256 `db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57`。

## 6. 成功、失败和停止规则

- Primary：fixed validation108 Scan Dice；Subject Dice为共同promotion指标。
- Selection：global epochs176/184/192/200中最大DCD Scan Dice，Subject Dice只用于Scan完全相同时tie-break。
- Promotion全部必须通过：
  1. selected DCD Scan和Subject严格高于同global epoch base voxel-KD；
  2. selected DCD Scan和Subject严格高于当前voxel-KD200 `0.734384308/0.724434143`；
  3. Point hit不低于KD200 `0.935185185 - 1/108`；
  4. 相对KD200 paired Scan和Subject bootstrap 95% CI下界都`>=-0.003`；
  5. final checkpoint reload和stripped deploy max abs difference均为0；
  6. 全部implementation、synthetic、GPU smoke和access/hash gates通过。
- Test：只有`all_gates_passed=true`才冻结唯一selected checkpoint、新test evaluator SHA和pre-access receipt，然后对reused internal test270测评一次。
- Scientific failure：任一promotion gate未过即停止，不改teacher、wavelet、levels、bands、stage、mu、schedule、epoch、threshold或postprocessing，不访问test。
- Technical stop：任一DWT/IDWT shape不精确；冻结db4 coefficient不匹配canonical值；全8-band重建max error不低于`1e-5`；identical feature DCD loss不为0；任何stage raw/weighted loss出现NaN/Inf；student四级detail gradient任一级缺失或非有限；任一teacher出现gradient；DCD weight0的一步student更新不与base voxel-KD完全一致；student final logit与普通locator不等价；checkpoint reload不精确；batch8 smoke OOM、peak allocated超过72GiB或one-step机械外推formal超过2小时。

## 7. 成本与风险

- 训练时间：base voxel-KD32轮实测`1180.85s`。DCD从student+2 teachers共3次forward增至student+3 teachers共4次forward，并增加四stage三level DWT/IDWT；smoke前估计30–60min。
- 显存：base voxel-KD peak allocated约`20.29GiB`；DCD最高分辨率stage为`16@128³`，估计25–55GiB，硬停止线72GiB。
- 磁盘：不建新cache；四个deploy/training checkpoint与报告预计`<0.15G`。最近根盘约余21G。
- 推理：仍为4.70M student、1× locator forward；无teacher、DWT、projection、SAM、TTA或postprocessing。
- 最大风险：论文的mu和官方BraTS YAML冲突；官方1000轮schedule只能按比例适配到32轮；同容量teacher/student不同于压缩场景；KD200虽same-basin但hidden-channel对齐仍是推断；raw feature MSE尺度可能过小或压制seg。smoke只记录这些量，不据validation改参。

执行前filter方向修正与完整实现门见`AMENDMENT_001_PRE_IMPLEMENTATION.json`；该文件在模型/训练代码、合成测试、GPU smoke和新validation access之前冻结。

## 8. 最终结果（完成后填写）

- Validation：selected epoch200 Scan `0.733603344`、Subject `0.723798160`、Point `0.935185`；相对base KD200为`-0.000780964/-0.000635983`，promotion失败。
- Test（如授权）：未授权、未访问本方法的新test。
- 置信区间/病例分布：相对KD200 Scan 95% CI `[-0.001888,+0.000058]`，Subject `[-0.001581,+0.000123]`；49/108 scans和20/50 subjects改善，59 scans和30 subjects变差。
- 是否支持原直觉：不支持当前同容量、T1ce/WT、32轮DCD-on-voxel-KD适配。
- 学到了什么：raw DCD下降约86%，四stage都有有效梯度，说明detail matching被学会；但最终Dice几乎退回supervised control，没有保住base voxel-KD的小增益。这是科研负结果，不是实现失败。
- 下一步：停止本方法，不调teacher/wavelet/band/stage/mu/schedule；下一候选为CVPR 2026 IAD，需新预注册。
- 报告/归档/哈希：[RESULT.md](RESULT.md)、[RESULT.json](RESULT.json)；raw validation SHA `0bb0da55ec9f753378feada2f4b6cda6c81eb128ed8efe5f948cff2e404d6357`；selected checkpoint SHA `9354a0f6ef9e6747729ca5be61e3faa357b76f558898940e52e8f7b274f543dd`。
