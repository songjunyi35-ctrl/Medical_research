# 1→8→24 路由控制分析预注册

冻结时间：2026-08-14 UTC，在读取新控制分析结果前冻结。

输入只能是已经冻结的
`workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/FINAL_ADAPTIVE_ROUTER_RESULT.json`
（SHA-256 `dbba96e593c4bcf7742edb7f1e8afc614b058c6625ec8548e418849aef74eddf`）和其
`stage_features_and_dice.npz`。不重新读取/修改 workflow，不访问 270-case test。

## 分析 1：固定阶段的配对区间

以 subject 为重采样单位、seed `20260814`、10,000 bootstrap replicates，报告：

- fixed stage1 减 locked stage24；
- fixed stage8 减 locked stage24；
- fixed stage8 减 fixed stage1；
- adaptive strict-margin (`0.000`) 减 fixed stage1；
- adaptive strict-margin (`0.000`) 减 fixed stage8。

同时报告改善/下降/不变 subject 数。CI 跨 0 时不得宣称显著 Dice 改善或非劣。

## 分析 2：fold/count-matched 随机路由

对四个 accuracy margin 分别固定每个 outer fold 实际选择得到的 1/8/24 stop 数量；每次
replicate 只在该 outer fold 内随机打乱相同的 stage multiset，然后按对应病例的 fixed
stage Dice 计算 108-scan OOF subject-macro Dice。共运行 10,000 replicates，seed 为
`20260814 + margin_index`。

报告随机路由 subject Dice 的均值、标准差、2.5%/97.5% 分位数、学习路由相对随机均值
的差，以及 `P(random >= learned)`。这是预算匹配的随机化诊断，不是独立显著性检验。

若 learned strict-margin 没有超过随机路由均值，或优势小于 `0.0005` 且随机化分布大量
覆盖 learned，则 adaptive router 记为没有证明路由智能；效率收益归因给固定成员/视图
冗余。无论结果如何，fixed stage1/stage8 的性能仍可作为独立的开发集效率结果报告。

正式输出：
`workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/FINAL_ADAPTIVE_ROUTER_CONTROL.json`。
