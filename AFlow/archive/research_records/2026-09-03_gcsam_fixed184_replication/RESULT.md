# GCSAM固定epoch184双新seed复制结果

## 结论

固定`rho=0.05`的paper-form GCSAM（Gradient-Centralized Sharpness-Aware Minimization，梯度中心化锐度感知最小化）没有在两个新seed上稳定超过AdamW。复制A的Scan Dice下降`0.001214`；复制B的Scan只升`0.000176`，但Subject Dice下降`0.002230`且Point少2个。两个复制的平均Scan差为`-0.000519`，两组Scan/Subject配对置信区间下界也都低于冻结的`-0.003`安全门。

因此`stage_b_authorized=false`。按预注册立即停止：不做Full972 Stage B，不访问fixed validation108、outer、confirmation、reused test270或UCSD External75，也不根据结果调整`rho`、epoch、seed、阈值或后处理。

通俗地说：原来epoch184那次高分更像“一位学生在一次考试里发挥得好”，而不是这种优化方法能稳定让不同学生都进步。两次新比赛一负一近乎打平，不能拿旧榜单点估计当作可复制提升。

## 问题、直觉与为什么做复制

原GCSAM在一个train-only inner上相对AdamW只提高Scan/Subject `+0.000807/+0.002558`，两个95%置信区间下界为`-0.005464/-0.005327`，未过稳定性门。它随后在已经复用的internal test270上得到Scan点估计`0.752003`，在UCSD External75上三项点估计领先Full972 KD，但两处配对置信区间都跨0。

一句话直觉是：如果GCSAM真的把模型带进更宽、更稳的“山谷”，同一配方换两套全新的训练划分和随机种子后，也应同时提高Scan和Subject，而不是只在一个旧checkpoint或少数病例上好看。

既有inner失败只否定“单seed证据足以晋级”，尚未直接检验固定epoch184能否跨seed重现；所以本实验只复制已冻结配方，不提出新变体，也不让已访问的test/external反向选参数。

## 做法与唯一变化

每个复制都从BraTS train角色构建subject-disjoint的864 fit/108 train-only inner。输入是T1ce，经固定N4与image-only brain crop变为128³；4.70M参数SegResNet使用同一history/EMA teacher、Dice加交叉熵以及逐体素Kullback–Leibler divergence知识蒸馏训练。

- Control：普通AdamW。
- Candidate：第一次反传后中心化多维参数梯度，按全局L2范数把权重轻推固定`rho=0.05`，在同一batch、同一teacher logits和回放的随机状态下做第二次反传，再恢复参数并由同一AdamW更新。

唯一变化是student optimizer update。数据、每个pair的初始state、sampling weights、184轮sampler顺序、首增强batch、first-pass loss、模型、teacher、loss、batch8、late-cosine、固定epoch184、阈值0、无测试时增强、无Segment Anything Model、无后处理和1×部署全部固定。四条训练完成前不构造inner loader、不计算inner metric；四个checkpoint冻结后，独立评估器才对两个inner各读取一次。

## Stage A结果

| 复制 / train-only inner | 方法 | Scan Dice | Subject Dice | Point |
|---|---|---:|---:|---:|
| A：108 scans / 47 subjects | AdamW Control | **`0.739166642`** | `0.722541102` | `103/108` |
| A | GCSAM Candidate | `0.737952565` | **`0.726094759`** | **`104/108`** |
| A Candidate − Control |  | `-0.001214077` | `+0.003553656` | `+1` |
| B：108 scans / 45 subjects | AdamW Control | `0.767249025` | **`0.748281742`** | **`104/108`** |
| B | GCSAM Candidate | **`0.767425437`** | `0.746051589` | `102/108` |
| B Candidate − Control |  | `+0.000176412` | `-0.002230153` | `-2` |

- A Scan paired 95% CI：`[-0.008946,+0.006706]`；52/108改善、55变差、1持平。
- A Subject paired 95% CI：`[-0.009746,+0.017919]`；24/47 subjects改善、22变差、1持平。
- B Scan paired 95% CI：`[-0.005494,+0.006665]`；48/108改善、59变差、1持平。
- B Subject paired 95% CI：`[-0.010571,+0.007987]`；18/45 subjects改善、26变差、1持平。
- 两复制平均Candidate−Control：Scan `-0.000518833`，Subject `+0.000661752`。
- Bootstrap 10,000次；A/B seed分别为`20261105/20261106`。

## 冻结门与工程证据

每个复制预注册五个科学门：Scan和Subject点估计严格为正、Point最多下降1、两个paired 95% CI下界均不低于`-0.003`；此外要求全部工程配对exact，并要求两个复制的平均Scan/Subject差都为正。

- A只通过Subject点估计与Point门，Scan点估计及两个CI门失败。
- B只通过Scan点估计门；Subject、Point及两个CI门失败。
- 跨复制平均Subject为正，但平均Scan为负。
- 两个pair的初始state、sampling weights、全部184轮sampler indices、首增强batch和first-pass loss均exact；四条训练的finite、steps、reload、显存和未访问inner门全部通过。因此这是泛化结果失败，不是训练接线失败。

## 直觉是否支持、什么被排除

完整直觉不支持。GCSAM在A更偏向Subject/Point，在B仅取得`0.000176`的Scan微升却损失Subject和Point；两个复制没有一致方向，CI也容许有意义的负效应。旧epoch184在test270与UCSD75的漂亮点估计不能被解释为稳定、可复制的optimizer优势。

这排除的是当前T1ce/WT、4.70M SegResNet、history/EMA teacher、固定`rho=0.05` paper-form GCSAM、固定epoch184这一完整主张；不证明所有锐度感知优化永远无效。但按冻结协议不得在这两个已揭示inner上搜索`rho`、adaptive版本、epoch、seed或阈值，也不加第三个seed追正结果。

## 成本、访问与证据

- 四条formal训练共`19.14 GPU-hours`；双卡先controls后candidates的阶段墙钟约`9.59h`。
- A/B GCSAM相对各自AdamW耗时为`1.578×/1.581×`；峰值显存`21,859,250,688` bytes，约20.36GiB。
- Workspace约109M；复用现有cache，没有新建多GiB缓存或删除材料。
- 推理仍为4.70M参数、1×forward，没有部署复杂度增加。
- 两个fresh train-only inner已各访问一次；fixed validation108、outer、confirmation、test270和UCSD External75均未访问。
- IDEA SHA`54896ebda0f4c117d4d8b13d887ffb77e4fdb4dc77342fcfb3cb6acc2a1ce640`；PREREG SHA`3afbeb97293e1813df2faef39e2bb136d66534638092f02681a7929b667be04f`。
- Inner pre-access SHA`af239bdffbcb7a864de583c44723f0e9b6cff96048e6e592e63e44c1d2ba75ab`；raw inner SHA`0cb5004a522b36059eb1b2f02690003a450b5ea4967b80d0a66d9b165d7c5a6d`。
- GCSAM checkpoint SHA：A `5d1334d2b5ebe09e9bee9a3d511cbedd38c4da6562f5993adf6670ac724b3bcc`；B `d4162fa0b37fb952e8e910a75cea4d2a96922f12330d3d9f9675ca9b307660d0`。

## 下一决定

关闭GCSAM fixed184复制分支，不运行Stage B、不访问新的validation/test/external，不调`rho`、epoch、seed、loss、threshold或后处理。后续提高指标不能继续围绕这个已揭示结果做优化器搜索；应回到完整链路审视，从尚未解决的数据表示或训练信息瓶颈提出新单因素方法，并重新建立fresh证据。

Evidence links: [七问](IDEA.md)、[预注册](PREREG.md)、[预访问收据](INNER_PREACCESS.json)、[raw inner](RAW_INNER.json)、[机器结果](RESULT.json)、[A GCSAM checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/gcsam_fixed184_replication_20260903/formal_a_gcsam_candidate/fixed_epoch184.pth)、[B GCSAM checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/gcsam_fixed184_replication_20260903/formal_b_gcsam_candidate/fixed_epoch184.pth)。
