# T1ce脑区固定1–99百分位裁剪稳定性结果

## 结论

固定1st/99th percentile强度裁剪在两个全新train-only复制中都明确降低Dice，因此没有超过Leaderboard，也没有资格进入Full972 Stage B。

- 复制A Candidate−Control Scan/Subject/Point=`-0.008210/-0.009950/-4`；两个Dice 95% CI整体低于0。
- 复制B Candidate−Control=`-0.007520/-0.009875/+5`；两个Dice 95% CI同样整体低于0。
- 两复制平均Scan/Subject差=`-0.007865/-0.009912`。

`stage_b_authorized=false`。按预注册停止：不训练Full972、不调百分位或顺序、不加第三个seed，也不访问fixed validation108、outer、confirmation、reused test270或UCSD External75。因为没有运行任何named test/external cohort，本实验不更新Leaderboard。

通俗地说：把每张MRI最亮和最暗的各1%“剪平”，确实让亮度尺子更规整，但也剪掉了对肿瘤边界有用的强度差异。两套新考卷都少了约0.008 Scan Dice，所以不是偶然没赢，而是当前固定做法稳定有害。

## 问题、直觉与为什么尝试

开始时，Full972 KD在fixed validation108的Scan/Subject/Point为`0.740013/0.731266/103/108`。GCSAM、Gamma和Lookahead等训练配方跨seed不稳，而image-only brain crop曾提供约`+0.054`的可靠收益，说明输入表示仍可能是有效杠杆。

本实验的直觉是：少量异常亮暗体素可能牵动每例z-score，就像几个极端分数拉歪全班平均值；在N4后固定剪掉脑内两端各1%，再标准化，可能让不同扫描的组织亮度更一致。

Gamma负结果测试的是随机非线性亮度扰动，160³与width24测试的是分辨率/容量，GCSAM测试的是优化器，它们都没有直接检验确定性去极值。因此这个idea在实施前尚未被既有负结果否定。

## 方法与唯一变化

Control路径保持：raw T1ce → canonical RAS → N4 bias-field correction（偏置场校正）→ image-only brain crop → source非零总体z-score → trilinear resize到128³。

Candidate只在N4与z-score之间增加：用NumPy linear方法计算每例非零脑区1st/99th percentile，把两端强度夹到阈值，再按裁剪后的非零总体均值/标准差做z-score。标签仍以相同nearest-exact方式缩放。

之后两臂都使用同一4.70M SegResNet、`0.75 history168 + 0.25 EMA168`冻结teacher、Dice加0.1交叉熵与权重0.3/温度2的逐体素Kullback–Leibler divergence知识蒸馏、原生WT体积采样、AdamW、batch8、固定epoch168、阈值0、无测试时增强/Segment Anything Model/后处理和1×部署。

唯一变化是source强度是否执行固定1%/99%裁剪。每个pair的初始state、sampling weights、168轮sampler、增强随机流和标签全部exact；输入image按预期不同。

## Stage A：两个fresh train-only inner

| 复制 / cohort | 方法 | Scan Dice | Subject Dice | Point |
|---|---|---:|---:|---:|
| A：108 scans / 50 subjects | Control | **`0.730148456`** | **`0.714339197`** | **106/108** |
| A | 1–99 clip | `0.721938530` | `0.704389381` | `102/108` |
| A Candidate−Control |  | `-0.008209926` | `-0.009949816` | `-4` |
| B：108 scans / 53 subjects | Control | **`0.738567643`** | **`0.734514387`** | `98/108` |
| B | 1–99 clip | `0.731047454` | `0.724639360` | **103/108** |
| B Candidate−Control |  | `-0.007520189` | `-0.009875027` | `+5` |

- A Scan paired 95% CI：`[-0.016222,-0.001269]`；49/108改善、58变差、1持平。
- A Subject paired 95% CI：`[-0.022694,-0.000456]`；25/50 subjects改善、24变差、1持平。
- B Scan paired 95% CI：`[-0.012519,-0.002623]`；45/108改善、61变差、2持平。
- B Subject paired 95% CI：`[-0.017121,-0.003004]`；19/53 subjects改善、34变差。
- Bootstrap 10,000次；A/B seed分别为`20261115/20261116`。

## 成功门与工程证据

每个复制预注册五个科学门：Candidate Scan与Subject严格提高、Point最多下降1、两个paired 95% CI下界均≥`-0.003`；另要求两复制平均Dice差为正及所有工程门通过。

- A五个科学门全部失败。
- B只通过Point门；其余四门失败。
- 两个跨复制平均Dice门均失败。
- 972例candidate cache labels与Control整文件SHA同为`bf470d48...`；5/5独立单线程重建逐字节exact。
- A/B每pair的初始state、sampling weights、全部168轮sampler indices、全部增强随机流与首增强label exact；四条训练的step/draw/finite/cache/reload/time/memory/access门全部通过。

所以失败是方法导致的泛化差异，不是标签、采样、增强、随机种子接线或checkpoint错误。

## 直觉是否支持、什么被排除

直觉不支持。两个独立split的Scan和Subject均下降约`0.008–0.010`，且四个paired CI上界都低于0，说明当前固定裁剪不仅没有稳定收益，还稳定损害Dice。

复制B尤其有解释力：Candidate最终fit train Dice `0.792885`略高于Control `0.791634`，Point也多5个，但inner Scan/Subject仍低`0.007520/0.009875`。这说明裁剪可以让训练拟合或最高logit位置看起来更好，却不能保留完整mask的泛化边界信息，再次证明Point与Dice不是同一目标。

本实验排除的是当前T1ce/WT、N4后source linear 1st/99th clipping、clip后z-score、128³、Full972-style teacher/student训练这一完整配方；不证明所有MRI强度归一化永远无效。但已揭示inner不得用于搜索0.5/99.5、2/98、裁剪顺序、阈值、epoch或teacher输入方式。

## 实现边界与透明故障

V1 cache主构建的24个worker从始至终固定SimpleITK单线程。972例构建完成后，父进程复算函数漏设单线程，第一例独立复算不exact并在模型前停止。只读诊断用与worker相同的单线程条件重算该例，image最大差0、label exact；随后只修复验证环境，原cache不重建，预注册5/5独立复算全部exact。这个故障没有触及模型或inner，完整记录在`AMENDMENT_001_PRE_MODEL_SINGLE_THREAD_REBUILD_VERIFY.md`。

四条训练完成后原长期runner会话在写`INNER_PREACCESS.json`之前失效；磁盘上四份`TRAIN_DONE`和checkpoint完整。恢复时先由冻结evaluator重新验证四报告、四checkpoint和A/B配对，再手工运行原冻结freezer与唯一一次evaluator。没有重复训练或重复inner访问。

## 成本、访问与证据

- 四条formal训练共`13.55 GPU-hours`；双卡两阶段墙钟约`6.78h`。
- Candidate cache构建约`0.715h`，临时cache约9.5G，位于`/tmp`；持久workspace约110M。
- 四臂峰值allocated均`21,788,258,304` bytes，约20.29GiB。
- Stage A评估共432个1×forward；部署仍是4.70M参数、1×forward，只增加确定性预处理。
- 两个fresh train-only inner已各访问一次；fixed validation108、outer、confirmation、test270与UCSD75均未访问。
- 训练绑定的pre-result IDEA SHA`7504085184ea0e17d97268e9233ff4eb857f9f80c1b49e301aa1534416a24b0a`；补写最终结果后的IDEA SHA`66dda6f80819417d7b47e94c82af3cf4fb68f6f3a450ee9e26f309cb289fa701`；PREREG SHA`cf4706a98e6053799f698d83dcd59efbfd87d7ff3880b6bea09624b4ed99e031`。
- Preaccess SHA`27a6c4aaa0d1b635181f20f5184a11a414e48e8301f6dfe09a979611710d7f3e`；raw inner SHA`1d6ce57f2702e6d574937252aa1ecd59ac40d805aaeb85b03baa263497279a64`。
- Candidate checkpoint SHA：A `29e7a52de65f0d7c072e1d7bb320fe9145310c0671a23c7c1defb5b47928d13a`；B `4c304fc730e9a0e15084858a47aee3514cf7f0564c5d4a8a4dfbcdd2207c78f6`。

## 下一决定

关闭固定1–99百分位裁剪分支，不训练Full972、不加第三个seed、不调百分位/顺序/epoch/threshold/teacher，不访问任何Leaderboard或外部cohort。结果进一步说明：当前T1ce中极端强度并非纯噪声，强度“规整”可能牺牲肿瘤判别信息；后续不能继续在这两个inner上搜索强度归一化设置。

Evidence links: [七问与链路复盘](IDEA.md)、[预注册](PREREG.md)、[cache收据](CACHE_RECEIPT.json)、[修正说明](AMENDMENT_001_PRE_MODEL_SINGLE_THREAD_REBUILD_VERIFY.md)、[预访问收据](INNER_PREACCESS.json)、[raw inner](RAW_STAGE_A_INNER.json)、[机器结果](RESULT.json)、[A Candidate checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/percentile_clip_stability_20260903/formal_a_percentile_clip/fixed_epoch168.pth)、[B Candidate checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/percentile_clip_stability_20260903/formal_b_percentile_clip/fixed_epoch168.pth)。
