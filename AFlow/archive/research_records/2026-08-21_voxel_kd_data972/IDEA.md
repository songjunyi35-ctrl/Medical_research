# Experiment idea: 固定计算量的972例voxel-KD数据扩容

Status: preregistered before implementation/formal training/validation access; causal-control details superseded by `AMENDMENT_001_MATCHED_CONTROL.md`

## 1. 现在的问题是什么？

当前test270最佳单模型voxel-KD200为Scan/Subject/Point `0.742976/0.725179/0.955556`，总体最佳双模型ensemble为`0.745054/0.727345/0.955556`。最近多种teacher、loss、病例加权和瓶颈adapter在同一fixed validation108上只产生约`-0.003`到`+0.0005`的变化，说明继续在同一756例训练池上改目标函数已接近平台。

当前成功的voxel-KD只使用756 scans / 346 subjects。开发数据另有108个旧selector scans和108个旧confirmation scans；三组病例和subject均互斥，合并后恰为972 scans / 444 subjects。当前fixed validation108 / 50 subjects与这972例完全不重叠。

## 2. 一句话直觉

与其反复改变同一本练习册的批改方式，不如在相同做题次数下让模型见到216份新的真实病例。

## 3. 为什么选择这个idea？

- 现有负结果大多证明辅助目标能被学会，却没有稳定提高整体Dice；瓶颈可能是病例覆盖而非loss表达力。
- 训练池从756扩到972，增加28.6% scans和28.3% subjects，但仍保留独立fixed validation108。
- 972缓存已经存在，不需重建多GB数据。对重叠756例的全量逐体素审计显示：image最大绝对差为`0`，756/756 image cases和756/756 label cases完全一致。因此候选不会同时改变预处理。
- 这不是新论文模块，而是标准的数据规模/多样性消融；最强依据是项目自身的角色审计和缓存等价证据。此前loss、patch与adapter负结果没有测试过额外216个真实训练病例。
- 旧selector/confirmation已经完成历史角色并被多次揭示；这里明确把它们重新标记为development-train。它们不是test。以后不能再把这216例当独立确认集。

## 4. 具体怎么做？

输入仍是N4校正、image-only brain crop后resize到`128^3`的单通道T1ce，目标仍是WT二值mask。Student仍从historical epoch168初始化。Teacher仍由historical168和EMA168对同一增强输入各做一次forward，raw logits按`0.75/0.25`混合。

Student仍计算Dice加0.1交叉熵监督损失，并加`0.3 * T^2 * voxel KL`，温度`T=2`。唯一变化是WeightedRandomSampler的候选池从756例变为972例。为了不把更多数据和更多计算混在一起，每轮仍只抽756次、drop-last后94个batch/752个实际draw，共32轮、3008个optimizer steps。

输出仍是一个plain AutomaticPromptLocator的128³ raw logit；阈值仍为`>0`，Point仍取最高raw-logit体素。无模型集成、测试时增强、SAM或后处理。

## 5. 唯一变化与固定项

- 唯一变化：训练候选池由756 scans / 346 subjects扩为972 scans / 444 subjects。
- 固定不变：historical168 student初始化、history/EMA teacher及0.75/0.25公式、KD weight `0.3`、temperature `2`、DiceCE、seed `20260846`、batch8、每轮756 sampler draws、94 optimizer steps、32轮、weighted replacement、shift8增强、AdamW、late-cosine LR、weight decay、clip5、保存epoch `176/184/192/200`、128³输入、raw-logit阈值、模型结构和1×部署。
- 两个新arm从同一旧972 cache运行：`data756_control`只暴露旧756例，`data972_candidate`暴露全972例。两臂都用缓存后128³标签体积计算权重并每轮抽756次；历史KD200是第二基准。

## 6. 成功、失败和停止规则

- Primary：fixed validation108 Scan Dice；Subject Dice为精确tie-break，再以更早epoch作最终tie-break。
- 固定候选：只比较global epoch `176/184/192/200`。
- Promotion gate：selected 972-KD必须同时严格超过新same-epoch `data756_control`和历史KD200的Scan/Subject；Point至少`101/108`；相对两者Scan/Subject的10,000次paired percentile bootstrap 95% CI下界均不低于`-0.003`；所有implementation/cache/reload/memory/access门通过。
- Stop：cache重叠不exact、sampler不是972权重池/756 draws、loss与原voxel-KD不等价、NaN/Inf、teacher有梯度、batch8 peak allocated超过24GiB、机械外推超过45分钟或checkpoint reload不exact立即停止。
- 只有全部validation门通过，才冻结唯一candidate并访问一次reused internal test270。失败后不改变draw数、sampling权重、epoch、LR、KD weight、阈值或后处理。

## 7. 成本与风险

- 训练时间：每arm预计约20分钟，两卡并行墙钟约20分钟；每arm硬门45分钟。
- 峰值显存：预计约20.5GiB；硬门24GiB。
- 磁盘：复用现成9.5GB cache，不新增cache；8个checkpoint和报告预计小于200MB。当前根盘剩余约17GB。
- 推理：4,700,897参数、1×forward，与KD200相同。
- 最大风险：在固定756 draws下，每个旧病例平均重复次数会下降；如果额外病例与现有分布相似，更多多样性可能只增加梯度噪声。旧selector/confirmation被回收后也不再具有独立确认身份。

## 8. 最终结果（完成后填写）

- Validation：待运行。
- Test：仅validation全部过门后允许。
- 是否支持原直觉：待运行。
- 下一步：待运行。
