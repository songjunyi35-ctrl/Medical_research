# 预注册：UCSD BraTS-GLI 2024公开test包冻结面板

冻结时间：2026-09-06（任何package-1221像素访问之前）。来源审计见
[../../reviews/2026-09-06_ucsd_brats2024test_source_audit/RESULT.md](../../reviews/2026-09-06_ucsd_brats2024test_source_audit/RESULT.md)。

## 1. 数据与cohort（像素前冻结）

- 来源：TCIA UCSD-PTGBM package 1221（`UCSD-PTGBM-BraTS-2024-test-set`，集合V2 2026-01-22发布、V3 2026-03-13临床表修正，CC BY 4.0）。
- Cohort：包内59 studies中排除`UCSD-PTGBM-0038`全部3个study（官方v3映射显示其BraTS时间点`BraTS-GLI-02289`在项目1,350例union内；该患者也在先前访问的main release 136人中）。冻结**56 scans / 41 subjects**，每scan固定`T1post.nii.gz`+`BraTS_tumor_seg.nii.gz`共112文件、312,277,333 bytes（清单=SOURCE_AUDIT_V2.json `selected_files`，SHA-256=`9fbc2467e84bbf536676f056ffd581c4807b259453a47cb9e49692c9b06fbbf4`）。
- 身份：public post-challenge（原BraTS-GLI 2024隐藏test的UCSD部分，现已公开）；**非hidden blind**；patient-disjoint-by-official-mapping（相对1,350例union与已访问UCSD main release），**非institution-disjoint**（UCSD属BraTS2024七来源站）；独立分榜，不与UCSD75/internal/其他外部混排。
- 确定性排除规则（性能前）：若某scan raw `BraTS_tumor_seg`中labels{1,2,3}全部缺失（空WT），该scan在性能前排除并在报告中逐例列出；不做其他任何基于影像/标签内容的选择。

## 2. 传输与工程门

1. 双次独立官方Aspera传输全部112文件（每次≤60路径/批），逐文件两副本SHA-256必须exact；包无`.sums`，此门代替publisher checksum（Africa95/MOTUM先例）。失败即停止。
2. 逐例native门：image/label shape exact、affine最大绝对误差≤1e-5、raw label经rint后值域⊆{0,1,2,3,4}、image全finite且有正值。
3. 预处理（与训练/既有外部面板完全一致）：canonical RAS→N4 shrink2、4×50、SimpleITK线程1→image-only正值bbox crop→源非零z-score→128³（影像三线性、标签最近邻）→脑mask清零。复用`scripts/build_no_n4_clean_cache.arrays(use_n4=True)`。
4. 零WT crop丢失门：每例raw WT体素必须100%落在image-positive bbox内；任何丢失即整体停止（不排例、不扩框、不padding）。
5. 首例probe（单线程）与全量worker输出image/label SHA byte exact；cache全finite、标签binary、（排除空WT后）全非空。
6. cache建在/tmp；建cache前`df`检查（需求+2GiB reserve）。

## 3. 模型面板（性能前冻结）

- Full972 random-init voxel-KD epoch168：checkpoint SHA-256=`f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`；identity（1×）与固定Flip8（固定顺序8种三轴reflection、逆翻后float32等权raw-logit平均，8×）。
- GCSAM fit864 inner-selected epoch184：checkpoint SHA-256=`acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d`；仅identity（1×）。
- 阈值0；自动point=全体素最高raw logit；无SAM/连通域/后处理/模型ensemble。
- synthetic smoke（9 forwards，随机张量）通过后冻结MODEL_PREACCESS，再启动唯一一次正式评测；started receipt先于checkpoint加载。正式forwards必须等于`可评scan数×9`。

## 4. 指标与统计（性能前冻结）

- Scan Dice：可评scan等权平均。Subject Dice：先每subject内平均scan Dice，再subject等权平均。Point hit：阳性scan分母。
- Bootstrap：percentile、10,000次。绝对CI：scan级重采样与subject级（先聚合患者均值再重采样）；配对CI：scan配对差与subject配对差各自重采样。seed：绝对Full identity=20261601、Full Flip8=20261611、GCSAM=20261621；配对Flip8−identity=20261701、GCSAM−Full=20261801。
- 报告：三行绝对值+两组配对差、improve/worse/tie（scan与subject两级）、预测/目标体素、每例per-case行完整保存。
- 解释门（唯一结论口径）：delta>0且scan配对CI下界>0→`supported_on_ucsd_brats2024test56`；delta>0且CI跨0→`positive_but_uncertain`；否则→`not_supported_or_reversed`。
- 无performance promotion门；无论结果如何一次归档并更新独立分榜；禁止据此调参、重跑或访问任何旧cohort。

## 5. 榜单与归档义务

- 更新`AFLOW_BRATS_LEADERBOARD.md`与`.json`：新增独立External UCSD-BraTS2024test分节（3行，按Scan Dice降序），必要时扩展validator对新分节的检查；运行leaderboard validator。
- 归档：RESULT.md、RESULT.json、raw报告、SOURCE_RECEIPT、CACHE_RECEIPT、FIRST_CASE_PROBE、SMOKE_RECEIPT、MODEL_PREACCESS、VERIFICATION_RECEIPT（终态重hash）及SHA-256清单；更新HANDOFF与METHOD_CATALOG。
- 访问声明：本实验不访问internal test270、fixed validation108、outer、confirmation、UCSD75、Africa95、CPTAC33、WAW各cohort的任何模型评测；训练源caveat（Kaggle mirror、官方Synapse parity未证明）继承。
