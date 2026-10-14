# BraTS T1ce/WT MCTS 推理 Workflow 搜索结果（2026-08-13）

## 结论

真正的 UCT MCTS 推理 workflow 搜索已完成实现和三折 nested validation 运行。它在
cross-fitted OOF 上没有提高 Dice：subject-macro 相对 locked `0.700419` workflow 的配对
差为 `-0.0003363`，50-subject bootstrap 95% CI 为
`[-0.0019888, +0.0013652]`。因此不能宣称 MCTS 带来精度创新。

搜索过程得到的三个折内 winner 平均使用 7 个 member-view forwards，而 locked workflow
使用 24 个，即 OOF 搜索过程估计减少约 70.8% forwards；对应 subject Dice 只下降
`0.0003363`。这是一个有潜力的精度—计算量 Pareto 结果，但三个 fold winner 不同，不能把
它写成单一已冻结部署 workflow 的成绩。后续需要预注册可接受的 non-inferiority margin，
在新的 external confirmation set 验证一个固定 workflow。

整轮仍是 GT-centered、N4、固定 click-3 coarse-logit oracle geometry；不是自动定位，
不是 test。最终 workflow、训练和搜索都不含 gray loss。已经揭示的 270-case test 没有进入
MCTS reward、selection、threshold、TTA、component filter 或 outer evaluation。

## Exact atom cache 与 parity

三个固定 robust checkpoint × 8 reflections × 108 validation scans 被一次性物化为 float32
logit atoms，避免每个 MCTS 节点重跑 3D 网络。正式 cache：

`workspace/BraTS/cache/t1ce_wt_robust_mcts_atoms_f32/`

第一次单一 `2e-6` gate 被正确拦截，原失败收据保留。独立 BF16 重放的差异为：locked
scan-macro `0.7004216820` vs 历史 `0.7004189626`，subject-macro 差 `1.25e-6`，最大逐病例
Dice 差 `2.71e-4`。在查看任何 MCTS score 前新增 parity-review prereg，冻结 aggregate
`1e-5`、per-case `5e-4` 容差；promotion runner 无 tolerance CLI，重新核对三 seed identity、
identity ensemble、3×8 raw、3×8 locked 的逐病例结果后发布 cache。

关键 SHA-256：

| Artifact | SHA-256 |
|---|---|
| atom `index.json` | `4da4b3db787a8281fb37d134ed7a2e4dcd6b7365fcedf1463a5550aabf1f02de` |
| `PARITY_REVIEW.json` | `0670a404e2e33e6660c054451798c2aad4b4438d91078f9b93fa2251173b05b7` |
| original `PARITY_FAILED.json` | `304e6c641851c087adc897245b76ef3b3546e4cfa92f88ad15a88132cfcbce13` |

## 搜索协议

- 50 subjects / 108 scans 按 subject 确定性拆为 3 个 outer folds，每折 36 scans；
- 每折剩余 subjects 再拆为 36-case `mcts_search` 和 36-case `inner_selector`；
- 每折 128 个 unique workflows，最多 512 simulations；locked workflow 是 mandatory anchor，
  计入 unique budget但不伪造 tree visits；
- `mcts_search` 产生候选，top-8 只在 `inner_selector` 比较；winner 冻结后 outer 只读一次；
- reward 主项为 subject-macro Dice，并惩罚 paired subject SE、实际 forwards 和 workflow
  complexity；scan-macro 同时报告；
- 每折保存完整 `N/W/Q`、UCT 分量、rollout actions、RNG、代码/数据/action-space/role SHA；
  completed outer fold 只能在证据链 hash 全部匹配时只读恢复。

第一次 `v1` pilot 在 fold 0 的 60 unique workflows 后、任何 selector/outer 访问前停止，因为
尚未加入 mandatory locked anchor。该 state 原样保留，不作为结果。`v2` 新目录从零开始，
加入 anchor 后完成正式三折。

## 三折 winner

| Fold | Workflow 摘要 | Outer subject Dice | Locked | 配对差 | Forwards |
|---:|---|---:|---:|---:|---:|
| 0 | seed 20260815，8 views，logit mean，tau -0.25，beta -0.5，CC 250 | `0.7187281` | `0.7184996` | `+0.0002285` | 8 |
| 1 | 3 seeds，4 views，logit median，tau -0.25，beta -0.5，CC 500 | `0.6961677` | `0.6966935` | `-0.0005258` | 12 |
| 2 | seed 20260815，identity，logit mean，tau -0.75，beta 0，CC 500 | `0.6716086` | `0.6722869` | `-0.0006783` | 1 |

尽管 action space 包含 prediction-only squared-soft-Dice disagreement 的 3→12→24 adaptive
workflow，三个 inner-selector winner 都是 fixed workflow。因此当前结果不能宣称自适应
gate 已得到验证；它只是实现完成、进入了可审计搜索空间。

## Cross-fitted OOF

| 指标 | 三折 winner OOF | Locked reference OOF | 差值 |
|---|---:|---:|---:|
| scan-macro Dice | `0.6998208` | `0.7004217` | `-0.0006008` |
| subject-macro Dice | `0.6950369` | `0.6953732` | `-0.0003363` |
| mean member-view forwards | `7.0` | `24.0` | `-17.0`（`-70.8%`） |

subject-paired bootstrap 使用 seed `20260813`、10,000 replicates，差值 95% CI 为
`[-0.0019888, +0.0013652]`。区间跨 0，不能证明改善，也不能在没有预注册等效界值的情况下
声称 non-inferiority。

正式结果：

`workspace/BraTS/sammed3d_t1ce_wt/mcts_workflow_20260813_v2/final_report.json`

SHA-256：`b3cabb91642e8f0b711b1a9acfd2567fa305ba57850d49d3ec5247024ed22009`。

`deployment_candidate` 是三个已观察 fold winner 的 canonical medoid，只是待研究候选；runner
有意没有在完整 108 validation 上再次打分，避免用同一数据继续选择。下一步应先冻结一个
精度—成本目标和单一 workflow，再去新的 external confirmation set。自动定位仍是另一个
必须单独解决的研究问题。
