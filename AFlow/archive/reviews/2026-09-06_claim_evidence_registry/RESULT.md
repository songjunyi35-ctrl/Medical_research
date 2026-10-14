# 可说/不可说：当前结论证据登记

2026-09-07优先补记：涉及UCSD-BraTS2024test56的“确定为负/两指标显著反向”须收窄。原冻结区间保持，但数值精度审计20组10k Scan CI有17组跨0、百万次Scan CI跨0；Subject4/20跨0、百万次仍略负。可以说负点估计及原区间贴零，不可称两项稳健显著反向。本补记覆盖旧CLAIMS_V2对显著性强度的解读，不替换原模型分数/区间或选择seed。[审计](../2026-09-07_ucsd56_bootstrap_precision/RESULT.md)。

这份表不是新实验；它把已冻结结果翻译成论文/汇报可用的严格措辞，防止点估计、置信区间、数据身份和因果边界被混写。

| 主题 | 当前可说 | 不可说 |
|---|---|---|
| 固定Flip8 | 五个已运行cohort Scan点估计都正；internal/CPTAC/WAW-pre Dice CI全正，Africa/WAW-post跨0；四外部precision点估计全正且Africa/两WAW的precision CI全正，recall近零或混合；Point差均不负 | 普遍显著、免费、1×单模型改进，或稳定补回漏画 |
| GCSAM配方/checkpoint | Africa Dice CI正；五外部预测体积/recall点估计全正，但precision混合、Point从不更高；配方双复制失败 | optimizer因果、配方稳定复现，或coverage等于全面更准 |
| Africa结果 | 只在Africa95内，GCSAM Dice/recall相对Full更高，Point少3 | GCSAM三指标全面最好或所有非洲医院都成立 |
| WAW结果 | GCSAM/Flip8 Dice `0.693601/0.693494`几乎tie；GCSAM相对Full不确定，Flip8 CI全正/Point+2 | hidden blind、GCSAM确定优越或通用TTA定律 |
| UCSD来源 | 官方float与uint8镜像75/75图像不同；指标来源差很小且CI跨0 | 镜像就是TCIA官方，或来源纠正是第二个独立外部 |
| N4家族 | no-N4只在E强、F不稳；Otsu E小正/F负，两个实现均未晋级 | N4永远必要，或所有no-N4/Otsu思想都不可能 |
| “最好” | 必须同时写cohort、Scan/Subject/Point和1×/8×作用域 | 不带作用域直接称best/SOTA |
| 外部身份 | Africa/CPTAC/WAW-pre与patient-disjoint WAW-post均public非hidden；UCSD是同cohort纠正 | pristine blind、服务器hidden、或无桥时称ID证明patient-disjoint |
| MOTUM | 来源与共享坐标桥通过，图像支持门失败，0性能 | 报MOTUM分数，或事后丢0001/按GT加padding |
| T1ce-only可观测性 | WAW三输出共同5个零Dice全grade2，4例纯FLAIR label2；only-label2占WAW17.11%而train3.09%，支持可观测性+覆盖限制 | 按真值label/grade排例或router，或宣称所有非强化瘤都不可能 |
| 早期术后Point | WAW-post miss约81–82%落GT RC；Point仅诊断、不会回灌当前direct mask；二值WT折叠RC支持表示缺口 | 宣称Point导致Dice、RC head已证明、用GT避让/加后处理或外推所有阶段 |
| OpenNeuro-GBM334 | 334个版本化配对中332过输入门，BO118/KA86各有4/6个WT体素在图像正值crop外；0外部模型/性能 | 报332例分数、排两例、按GT选padding，或说checkpoint已在该cohort失败 |

最重要的一句：**没有单一全域、Dice/Point/成本统一冠军。** 当前最稳定的方向证据是固定Flip8，但它需要8×forward；最醒目的单队列差是GCSAM在Africa95，但不是可复制优化器因果证据。

机器版：[CLAIMS.json](CLAIMS.json)。总体数字入口：[六cohort证据矩阵](../2026-09-06_five_cohort_evidence_matrix/RESULT.md)、[Flip8误差机制](../2026-09-06_flip8_overlap_synthesis/RESULT.md)、[Leaderboard](../../../AFLOW_BRATS_LEADERBOARD.md)、[方法目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)。

## V2补记（2026-09-06 Fable session）

新增第七cohort UCSD-BraTS2024test56后发布[CLAIMS_V2.json](CLAIMS_V2.json)（V1保留）：C1 Flip8方向扩为六cohort点估计全正、三处CI全正、Point差均不负；C3改为`supported_within_africa95_only_and_reversed_on_ucsd_brats2024test56`——GCSAM−Full在该新队列scan/subject配对CI首次整体为负（`−0.012312`，Point−4），"GCSAM外部普遍偏高"措辞永久作废；新增C13定义新cohort可说/不可说边界（public post-challenge、patient-disjoint by official mapping、非institution-disjoint、非hidden blind、不与其他榜混排、不重跑）。
