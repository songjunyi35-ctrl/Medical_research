# Experiment idea: GCSAM epoch184双新seed固定复制

Status: planned; frozen before split generation, implementation, training, or new inner access

## 链路审视与启发

当前链路是`BraTS train角色972例 → N4与image-only brain crop 128³ cache → 864 fit/108 train-only inner → history/EMA teacher逐体素知识蒸馏 → epoch184冻结checkpoint → 一次inner readout → 仅复制全过才允许Full972 → fixed validation → 不再访问已关闭test/external`。

1. GCSAM原inner相对AdamW只有Scan/Subject/Point `+0.000807/+0.002558/0`，paired 95% CI下界`-0.005464/-0.005327`，不足以证明稳定提升。
2. UCSD External75上GCSAM点估计比Full972 KD高`+0.010093/+0.016097/+1`，但只赢36/75 scans、25/53 subjects，paired CI跨0且单例`+0.489233`影响很大。它说明值得复制，不允许据此调`rho`、epoch或输入。
3. LAWA6、Gamma增强、width24都出现inner正向、fresh final反转，说明当前最重要的不是再造一个模块，而是用新seed复核同一机制。
4. 因此本实验不提出新GCSAM变体，只复制已经冻结的paper-form `rho=0.05` GCSAM，并把epoch固定为184，不再通过新inner选epoch。

## 1. 现在的问题是什么？

GCSAM fit864 epoch184目前是UCSD External75的Scan/Subject/Point三项点估计第一（`0.755647/0.755152/71/75`），也是reused internal test270的Scan点估计第一（`0.752003`）。但原train-only inner的优势很小且两个CI门失败；它不是Full972 final，不能因为榜单好看就直接补训或调参。

## 2. 一句话直觉

一次抛硬币朝上不等于硬币偏向正面；保持硬币和抛法不变，在两套新的训练划分与随机种子上各完整重做一次，才知道结果是不是偶然。

## 3. 为什么选择这个idea？

- 它直接针对项目反复出现的跨seed翻转，不新增模型容量、loss、teacher或推理技巧。
- 2025 GCSAM（Gradient-Centralized Sharpness-Aware Minimization，梯度中心化锐度感知最小化）论文的直觉是：先沿中心化梯度轻推参数，再要求轻推后仍答得好，使模型落在更宽的谷底。Primary sources：`https://arxiv.org/abs/2501.11584`、`https://doi.org/10.1109/ACCESS.2025.3624332`、`https://github.com/mhassann22/GCSAM`。
- 原实验只否定“一个inner seed足以promotion”，没有否定固定GCSAM在其他新seed上可能稳定；因此复制不是推翻旧负结果，而是补旧结果明确缺少的证据。
- UCSD75和test270已经访问，只能解释为什么需要复制，不能决定任何超参数、epoch、split结果或promotion门。本实验所有设置沿用访问它们之前已冻结的GCSAM配置。

## 4. 具体怎么做？

从同一个972例开发train角色生成两个新的subject-disjoint划分。每个划分精确留108 scans作train-only inner，其余864 scans训练；两个复制使用不同split seed和training seed。

训练时，增强后的单通道T1对比增强MRI先交给两个冻结teacher：history epoch168和Exponential Moving Average（指数移动平均，EMA）epoch168；两者raw logit固定按`0.75/0.25`融合。Student仍计算Dice加Cross-Entropy（交叉熵，CE）监督loss，再加weight `0.3`、temperature `2`的逐体素Kullback–Leibler（KL）知识蒸馏。

- AdamW control：一次student前传/反传，clip5后更新。
- GCSAM candidate：第一次反传后，将每个多维参数梯度在非输出维减均值，以全局L2范数归一化到固定扰动`rho=0.05`；在同一batch、同一teacher logits、同一dropout随机状态下第二次反传，clip5，恢复原参数，再由同一AdamW更新。

四条轨迹全部训练到固定epoch184，训练过程不得构造inner DataLoader、计算inner metric或保存inner-selected best。四个checkpoint和配对工程证据全部冻结后，独立evaluator才对两个inner108各访问一次。输出仍是raw logit `>0` mask，最高logit为自动point。

## 5. 唯一变化与固定项

- 每个复制pair唯一arm变化：student optimizer update从AdamW变成固定paper-form GCSAM。
- Replicate A：split seed `20261101`，paired training seed `20261102`。
- Replicate B：split seed `20261103`，paired training seed `20261104`。
- 固定epoch184，不以任何新inner选择epoch。
- 固定Full972来源、864 fit/108 inner、native WT volume weighted replacement、每轮972 requested/968 consumed/121 steps、batch8、4.70M width16 plain SegResNet、dropout0.1、两个teacher与`0.75/0.25`、DiceCE、voxel-KL、LR与200轮denominator的late cosine、shift8、clip5、阈值0、无Test-Time Augmentation（测试时增强，TTA）、无Segment Anything Model（SAM）、无postprocess、单模型1×推理。
- 两个复制的split不同是预先规定的复制条件，不是结果后搜索。

## 6. 成功、失败和停止规则

- Primary：每个新inner108的Scan Dice；共同安全指标为Subject Dice和Point hit。
- 所有四条训练必须在任何inner指标访问前完成，且每个pair初始state、sampling weights、184轮sampler index和首增强batch exact。
- 每个复制都必须满足：Candidate Scan与Subject严格高于same-seed Control；Point最多少1/108；Scan case-paired与Subject subject-paired 10,000次bootstrap 95% CI下界都`>=-0.003`。
- 两个复制的Scan delta平均与Subject delta平均均须`>0`；全部工程/finite/reload/time/memory/access门通过。
- 任一复制或工程门失败：停止，不运行Full972 Stage B，不访问fixed validation、outer、confirmation、test270或UCSD75；不调`rho`、epoch、seed、centralization、clip、loss、threshold或postprocess。
- 只有两个复制全部过门，才授权Stage B：AdamW/GCSAM以fresh seed `20261107`在Full972从同一初始state训练到epoch184，然后一次fixed validation108。Stage B另行在访问前冻结收据；相对matched control和archived Full972 KD seed32的Scan/Subject点估计必须严格更高、Point≥103/108、四个CI下界≥`-0.003`。
- 即使Stage B全过，也不授权reused test270或UCSD75；未来只能等待新的独立cohort。

## 7. 成本与风险

- Stage A：四条184轮轨迹。两张GPU先并行两个AdamW control，再并行两个GCSAM candidate；预计约21–22 GPU-hours、10–11小时墙钟。
- 若Stage A全过，Stage B预计再约10–11 GPU-hours、6–7小时墙钟。
- 峰值显存：Control历史约20.29GiB、GCSAM约20.36GiB；硬门24GiB/arm。
- 磁盘：复用现有972例cache，不新建多GiB cache；四条epoch184 checkpoint、日志和报告预计<300MB。
- 推理：仍4,700,897参数、1×forward；无teacher、扰动器或额外模块部署成本。
- 最大风险：GCSAM每步两次student反传，训练慢约1.55×；原inner CI已失败，两次复制很可能证实它不稳定而停止。这仍是有价值的排除结论。

## 8. 最终结果（完成后填写）

- Train-only inner A/B：待运行。
- Full972/fixed validation：仅Stage A全门通过后允许。
- Test/external：不允许。
- 置信区间/病例分布：待运行。
- 是否支持原直觉：待运行。
- 下一步：待运行。
- 报告/归档/哈希：待运行。

