# Experiment idea: 覆盖优先的小病灶回放蒸馏

Status: planned and frozen before implementation, GPU smoke, formal training, and validation access.

## 1. 现在的问题是什么？

当前reused internal test270总体最高是两模型logit融合，Scan/Subject/Point为`0.745054/0.727345/0.955556`，但推理要`2×` locator；最高单模型voxel-KD200为`0.742976/0.725179/0.955556`，只需`1×`。

在独立开发用fixed validation108上，972例原生体积感知voxel-KD candidate200达到Scan/Subject/Point `0.734060/0.724703/101/108`：相对same-epoch native756 control为`+0.000689/+0.000666`，Subject也比历史KD200高`0.000269`，但Scan仍低`0.000324`。两次972病例池实验均超过各自matched control，说明真实病例多样性有用；问题是每轮带放回只抽756次、实际消费752次，同一病例会重复，972例中的很多病例在一轮内完全看不到。

## 2. 一句话直觉

像老师复习时先保证大多数学生轮流被点到，再留约一成机会复问小病灶病例：既不漏掉班里的多样性，也不把全部时间只花在少数难题上。

## 3. 为什么选择这个idea？

- 项目内最强直接证据是两次972病例池相对matched 756 control均提高Dice；当前应修复“病例池扩大但每轮覆盖不足”，而不是继续更换loss。
- MICCAI 2025 CSAL-3D把diversity sampling（多样性采样）与uncertainty sampling（不确定性采样）分层结合，在Brain Tumor、Heart和Spleen三维任务验证了二者互补；官方代码公开。这里不复现其主动标注系统，只借用“先覆盖分布、再把少量预算给信息病例”的采样原则。
- MICCAI 2025 Active Source-Free Cross-Domain/Modality Adaptation进一步用图像敏感性和器官异质性联合查询三维病例，支持病例级覆盖不应只看全图背景。
- 本项目没有可靠的未泄漏模型不确定性校准；而原生WT体积权重已把Point从99/108恢复到101/108。因此回放项使用预先冻结、train-only的原生小病灶权重，不把未经验证的entropy当真值。
- Batch-CVaR50把一半batch预算集中给高loss病例后Dice和Point均下降；这不否定当前idea，因为本方法不改loss、不丢普通病例，并把回放限制为实际draw的`72/752=9.57%`。
- LoMix、DCD、KAS、hACE、VM Dice、DMRFB和旧972负/部分正结果都没有测试跨轮覆盖队列加低比例小病灶回放，因此没有提前否定它。

Primary sources:

- CSAL-3D paper: https://papers.miccai.org/miccai-2025/paper/2315_paper.pdf
- CSAL-3D official code: https://github.com/HiLab-git/CSAL-3D
- Image Sensitivity and Organ Heterogeneity Sampling paper: https://papers.miccai.org/miccai-2025/paper/3160_paper.pdf

## 4. 具体怎么做？

输入仍是972个development-train病例的单通道T1ce，经冻结的N4校正、image-only brain crop和`128^3`缩放。Student从historical epoch168初始化；Teacher仍让history168和EMA168分别看同一个增强输入，再把raw logits按`0.75/0.25`融合。Student仍学习Dice加`0.1`交叉熵监督损失，以及权重`0.3`、温度`T=2`的逐体素KL（Kullback–Leibler divergence，衡量学生和教师概率差异）蒸馏损失。输出仍是plain locator raw logit，阈值`>0`，最高logit作为自动point。

正式训练有两臂，均使用同一972病例池、原生WT体积权重、每轮请求756 draws并由batch8/drop-last实际消费752 draws、94个optimizer steps、32轮：

- `native_replacement_control`：与上次972 candidate相同，按原生体积权重带放回抽756次。
- `coverage_replay_candidate`：采样器先从跨轮滚动的随机覆盖队列取680例；队列在耗尽前不重复，耗尽后用同一冻结seed生成下一轮972例排列。再按原生体积权重带放回抽76例，其中72例参与训练，最后4例固定放在末尾被drop-last丢弃。每轮有效构成为680 coverage + 72 small-lesion replay。

覆盖队列和回放各用从主seed确定性派生的独立generator；候选的前752个index再做一次确定性shuffle，避免训练先看到全部coverage、后看到全部replay。四个丢弃index始终只来自额外replay，不会破坏coverage门。

## 5. 唯一变化与固定项

- 两臂唯一变化：病例采样策略，即“原生体积加权、全带放回”对比“跨轮覆盖队列680 + 小病灶加权回放72”。
- 固定不变：972 scans / 444 subjects manifest、同一9.5GB cache、原生体积工件和权重公式`sqrt(median/volume)` clip`[0.5,4]`、history168 student初始化、history/EMA teacher与`0.75/0.25`融合、DiceCE、voxel-KD weight`0.3`、temperature`2`、seed`20260884`、batch8、752有效draw、94步、32轮、shift8增强、AdamW、late-cosine LR、weight decay、clip5、epoch `176/184/192/200`、threshold0、无TTA（test-time augmentation，测试时增强）、无SAM（Segment Anything Model）、无后处理、4.70M参数与`1×`部署。
- 相对历史native972实验，新的两臂共同使用新seed和同一972病例池；历史结果只作独立第二基准，不能冒充本轮matched control。

## 6. 成功、失败和停止规则

- Primary：fixed validation108 Scan Dice；Subject Dice只作exact tie-break，再以更早epoch作最终tie-break。
- 固定候选网格：global epoch `176/184/192/200`；不得根据validation新增epoch。
- Promotion：selected candidate必须同时严格超过same-epoch `native_replacement_control`与archived KD200的Scan/Subject Dice；Point至少`101/108`；相对两个comparator的Scan/Subject 10,000次paired percentile bootstrap 95% CI下界均不低于`-0.003`。
- Sampler门：相同seed重建的前两轮index必须exact；每轮前752项必须是680 coverage + 72 replay，末4项全部是replay；任意连续两轮的coverage union必须包含全部972例；coverage队列在单次排列耗尽前不得重复。
- Engineering stop：cache/index/native artifact/hash不符、任一两臂draw/step/模型初始化不匹配、loss或梯度NaN/Inf、teacher出现梯度、batch8 peak allocated超过24GiB、单臂formal超过45min或checkpoint reload非exact立即停止。
- 只有全部validation门通过，才冻结唯一selected checkpoint、写pre-access receipt并访问一次reused internal test270。失败后不调整680/72、weight、seed、epoch、LR、KD、threshold或postprocess。

## 7. 成本与风险

- 训练：两张A100-SXM4-40GB各跑一臂，预计每臂约32–40min，并行墙钟相同；硬门45min。
- 峰值显存：预计每卡约20.3GiB，硬门24GiB。
- 磁盘：复用9.5GB cache和433KB原生体积工件；8个checkpoint、smoke和报告预计小于250MB。当前根盘可用约18GB。
- 推理：仍为4,700,897参数单模型、`1×` locator forward；无额外延迟、SAM、TTA或后处理。
- 最大技术风险：增加覆盖会减少小病灶的重复频率；9.57%回放可能不足以维持Point。论文研究的是主动选择而不是全监督重复采样，因此这里只能作为项目特定的新假设，不能把论文结果当作本方法已成立。

## 8. 最终结果（完成后填写）

- Validation：待运行。
- Test：仅全部promotion门通过后允许。
- 置信区间/病例分布：待运行。
- 是否支持原直觉：待运行。
- 学到了什么：待运行。
- 下一步：待运行。
- 报告/归档/哈希：待运行。
