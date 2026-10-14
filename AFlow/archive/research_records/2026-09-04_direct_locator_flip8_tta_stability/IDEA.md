# 实验idea：全自动Direct Locator固定8翻转测试时增强

状态：完成；两个fresh inner复制全门通过，唯一一次reused internal test270已完成并刷新全面点估计

日期：2026-09-04 UTC

## 链路审视与启发

当前链路是：

`BraTS raw NIfTI → subject-disjoint manifest → N4/image-only brain crop/128³ cache → 4.70M SegResNet + history/EMA teacher + DiceCE/voxel-KL → train-only inner → Full972 fixed checkpoint → reused internal test270或独立外部cohort`。

1. **可靠收益主要来自表示与数据时机。** Image-only brain crop旧匹配协议约`+0.054` Scan；Full972从第一轮纳入新增216例，相对matched Full756 fixed validation提高`+0.010943/+0.007982`。160³近`1.95×`体素计算和width24约`2.25×`参数却在fresh final失败，说明“盲目加算力”不是答案。
2. **训练方法跨seed稳定性差。** Gamma从fresh inner `+0.006151/+0.005570`翻成Full972 final `-0.003848/-0.002312`；GCSAM固定184双复制平均Scan为`-0.000519`；1–99 percentile clipping双复制平均Scan/Subject为`-0.007865/-0.009912`。下一方法应尽量复用成熟训练，不再增加一个不稳定优化轴。
3. **强模型与榜首差距极小。** Reused test270上Full972 KD Scan=`0.751759537`，GCSAM榜首=`0.752003176`，差仅`0.000243638`；Full972 KD的Subject/Point反而更高`0.737327107/262`。一个小而稳定的同模型推理改进就可能越过Scan点估计榜首。
4. **训练与推理存在等变性缺口。** 主训练增强对三个空间轴独立以0.5概率翻转，等价于模型见过全部8种reflection；部署却只看identity。若模型不是严格翻转等变，把8个预测逆翻回原方向后平均可消除方向偶然性。旧oracle refiner三seed ensemble中，raw 8-flip TTA相对identity raw ensemble由`0.697902`升至`0.698235`，约`+0.000332`；这是不同协议的弱机制证据，不能直接当主线收益。
5. **Point与Dice必须共同设门。** Percentile复制B Point增加5/108但Scan/Subject下降`0.007520/0.009875`；GCSAM test Scan领先却比KD少10 Point。TTA不能只追Scan，必须冻结Subject与Point安全门。
6. **证据边界。** Fixed validation、outer、confirmation、test270与UCSD75均已揭示。Stage A只允许两个新train-only inner；只有双复制全过，用户本轮明确的“超越Leaderboard”目标才授权一次reused test270探索性报告。即使刷新榜单也必须标记post-test/reused、不是pristine blind或外部确认。

由此得到三条直接用于本实验的启发：

- 用同一checkpoint的视图平均，隔离推理等变性，避开已关闭的优化器、loss、容量、强度和跨seed权重轴。
- 双fresh seed检验“增益是否稳定”，而不是因`0.000244`目标很小就直接在test尝试多种视图。
- 以固定全部8种三轴reflection形成完整有限群，不搜索单轴、四视图、权重或阈值；若失败，整条direct-locator flip-TTA轴关闭。

## 1. 现在的问题是什么？

当前reused internal test270最高Scan点估计是fit864 GCSAM epoch184的`0.752003176`；Full972 KD epoch168为`0.751759537`，只低`0.000243638`，而Subject/Point更高。GCSAM的双新seed复制没有稳定成立，因此下一步不再改优化器，而要检查成熟Full972-style KD模型是否因单一推理朝向留下可平均掉的小误差。

## 2. 一句话直觉

让同一位医生把MRI按八种镜像方向都看一遍，再把所有判断翻回原方向后平均，减少“图像摆放方向”造成的偶然误差。

## 3. 为什么选择这个idea？

- 当前训练对三个空间轴独立随机翻转，模型已被要求学习这些视图；identity-only部署没有利用这种已训练不变性。
- MONAI官方`TestTimeAugmentation`明确采用可逆空间变换、逆变换预测并聚合；官方实现：`https://github.com/Project-MONAI/MONAI/blob/dev/monai/data/test_time_augmentation.py`。
- nnU-Net官方实现默认使用训练时允许的mirroring；官方仓库：`https://github.com/MIC-DKFZ/nnUNet`。这支持工程做法，不保证本项目一定提高。
- 3D MRI分割的一手TTA研究报告TTA可优于单次预测：`https://arxiv.org/abs/1807.07356`；BraTS专门工作也使用3D翻转/旋转等TTA：`https://arxiv.org/abs/1810.07884`。
- 旧TTA只在依赖GT-centered crop/coarse输入的residual-refiner支线验证，不能否定全自动direct locator；跨seed ensemble混合不同权重盆地，本实验始终使用同一checkpoint，也不是同一机制。

## 4. 具体怎么做？

Stage A先用两个新的subject-disjoint split各训练一个标准Full972-style KD base：864 fit、108 inner；训练期间不构造inner loader，只保存固定epoch168 checkpoint。

冻结两个checkpoint后，独立evaluator对每个inner病例：

1. identity输入前向一次，得到Control logit；
2. 枚举空间轴`(-3,-2,-1)`是否翻转的全部8种组合；
3. 每个翻转图像独立前向，把输出logit沿相同轴翻回；
4. 用float32等权平均8份raw logits；
5. identity和平均logit都以`>0`得到mask，以各自全体素argmax得到自动Point。

8视图包含identity，因此一次评估共8×forward，不额外做第9次。没有阈值搜索、连通域、形态学、Segment Anything Model或其他后处理。

## 5. 唯一变化与固定项

- **唯一变化：** identity 1× raw logit变为同一checkpoint全部8种三轴reflection、逆reflection后的float32等权raw-logit平均。
- **固定：** T1ce/WT=`{1,2,3}`、brain-crop true128 cache、每个fresh split/seed、4.70M width16、history/EMA teacher、DiceCE+voxel-KL、native-volume sampling、972 requested/968 consumed/121 steps、batch8、AdamW、late-cosine、固定epoch168、阈值0、无SAM/connected-component/postprocess。
- Stage A同一checkpoint产生Control与Candidate，因此模型参数、训练随机性和数据完全相同。

## 6. 成功、失败和停止规则

- **Stage A数据：** Replicate A split/train/bootstrap seeds=`20261201/20261202/20261205`；B=`20261203/20261204/20261206`。每个都是864 fit/108 inner、subject-disjoint；两个模型全部训练完成并冻结后才打开任一inner。
- **Primary：** 每个inner108的Scan Dice；共同安全指标为Subject Dice与Point hit。
- **每复制门：** Flip8−Identity Scan>0、Subject>0、Point hits delta≥−1；Scan case-paired与Subject subject-paired 10,000次bootstrap 95% CI下界≥`-0.003`；全部训练/cache/checkpoint/访问门通过。
- **跨复制门：** 两个Scan delta平均>当前榜首差距`0.000243638`，两个Subject delta平均>0。
- **失败：** 任一门失败立即关闭，不搜索单轴/四视图/权重/阈值，不训练第三seed，不访问test270或UCSD75。
- **通过：** 冻结单独pre-test receipt，把完全相同8视图公式应用到archived Full972 KD epoch168 checkpoint，对reused internal test270运行一次。该访问由用户本轮Leaderboard目标明确授权，只能标记`post-test development; reused internal test; not pristine blind; not external`。
- **Test榜单成功：** Flip8 Scan严格超过当前GCSAM `0.752003176`且超过same-checkpoint identity `0.751759537`；Subject不低于identity、Point最多下降1/270。无论成败都更新Leaderboard Markdown/JSON，不得按test结果改视图、权重、阈值或checkpoint重测。

## 7. 成本与风险

- Stage A训练：两张A100各跑一个fit864 fixed168，预计约3.4h墙钟、6.8 GPU-hours。
- Stage A评估：2×108×8=`1,728` locator forwards；若通过，test270再`2,160` forwards。
- 峰值显存仍以单视图顺序推理，训练≤24GiB；推理不把8视图堆成batch。
- 磁盘复用现有cache，只新增两份checkpoint、日志和报告，预计<200MB。
- 部署模型仍4.70M参数，但推理计算/延迟约8×；这是用算力换精度，不是免费提升。
- 最大风险：卷积网络已通过训练翻转接近等变，8视图收益可能只有噪声；旧refiner机制证据约`+0.000332`很小。双复制门会在test前否决不稳定收益。

## 8. 最终结果

Stage A两个fresh inner复制均支持直觉。A的Flip8−Identity Scan/Subject/Point=`+0.006668/+0.006930/-1`，配对95% CI下界均为正；B=`+0.006988/+0.007333/+3`，下界也均为正。平均Scan/Subject提升=`+0.006828/+0.007132`，全部冻结门通过。

唯一一次reused internal test270上，同一Full972 KD168 checkpoint的identity=`0.751759537/0.737327107/262`，与旧报告完整summary及逐病例exact；固定Flip8=`0.757269662/0.741253585/265`，即`+0.005510124/+0.003926478/+3`。Scan paired 95% CI=`[+0.002751,+0.008022]`整体为正；Subject CI=`[-0.000379,+0.007354]`跨0。四个test成功门全过，成为270 scans/119 subjects reused internal test的Scan/Subject/Point全面点估计第1；证据仍是post-test、非盲测、非外部确认，禁止调参或重测。

raw report的描述性绝对CI helper误用seed20260826而非pretest冻结的20260817；点估计、配对seed20261207和所有成功门不受影响。保留raw，并仅从已保存per-case行离线重算正确绝对CI，无模型/缓存访问。完整解释与证据见[RESULT.md](RESULT.md)和[RESULT.json](RESULT.json)。
