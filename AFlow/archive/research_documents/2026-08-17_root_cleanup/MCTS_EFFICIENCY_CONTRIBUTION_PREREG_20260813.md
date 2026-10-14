# BraTS T1ce/WT 推理效率贡献：开发阶段预注册

## 研究定位

本轮不再把“提高 Dice”设为 MCTS 必须达到的目标。研究问题冻结为：在不重新训练三个
residual-refiner、且不读取 270-case test 的条件下，能否自动搜索出更低成本的三维推理
workflow，并透明报告精度—成本 Pareto 前沿。

论文中建议使用的名称是：

> 精度约束的可审计 MCTS 三维分割推理 workflow 搜索

旧的 `mcts_workflow_20260813_v2` 结果已经揭示，不能作为本预注册之后的全新确认集。本轮是
方法完善和开发阶段对照；最终泛化仍需要一个新的 external confirmation set。

## 冻结边界

- 数据仍为 108 例 / 50 subjects 的 validation，所有角色按 subject 隔离；
- 输入仍为 GT-centered、N4、128³、固定 click-3 coarse logit；
- 三个 robust refiner checkpoint 和 exact float32 24-atom cache 不变；
- gray loss 不进入任何候选；
- 270-case test 在搜索、选择、阈值、benchmark 和报告入口均不可达；
- 历史 MCTS 源码、报告和 SHA 不改写，新实现使用独立文件。

## 必须修正的协议点

locked 24-forward workflow 不仅要进入 search evaluation cache，还必须占用一个 inner-selector
名额。其余 `top_k - 1` 个名额才按 search 排名补齐。这样 selector 永远可以选择“不改变
locked reference”。

## 不再隐藏 trade-off

除旧 scalar reward 外，正式报告必须给出 Dice—member-forward nondominated frontier，并同时
报告相对 locked reference 的以下开发容差档位：

- 0.000 Dice；
- 0.001 Dice；
- 0.002 Dice；
- 0.005 Dice。

每个档位都选择满足精度 floor 的最低 member-forward workflow；并列时优先 selector Dice，
再按 workflow hash。该规则只是开发选择，不等同于统计学 non-inferiority。outer OOF 和未来
external confirmation 必须另报 paired interval。

## 方法对照

在完全相同的 workflow grammar、subject roles、unique-workflow budget、selector budget 和
mandatory anchor 下比较：

1. UCT MCTS；
2. uniform random search；
3. greedy coordinate search。

报告每种方法找到的 Pareto frontier、各容差档位的 outer OOF Dice/forwards，以及 search
budget 曲线。若 MCTS 不优于简单对照，论文贡献应降级为“可审计推理 workflow 优化框架”，
不能声称 MCTS 搜索策略本身更优。

## 部署候选与真实性能

现有三个 outer winner 的 canonical medoid（workflow
`a48d7c60047321da9933f5f29a924ff826b1dbcf533c4fa53e4aafbda8f63400`）冻结为第一版
benchmark candidate：seed 20260815、8 reflections、logit mean、threshold -0.25、
uncertainty beta -0.5、26-connectivity CC 250。冻结只用于无标签工程 benchmark，不把完整
108 validation 再次用于调选。

必须在同型号 A100 上分别用独立进程按需加载/执行 candidate 和 locked reference，至少报告：

- warm-up 后每病例 mean / median / p95 latency；
- cases/s；
- 模型加载后 allocated memory 和推理 peak allocated memory；
- 实际执行的 member-view forwards；
- refiner-stage 与端到端的口径边界。

若 runner 仍先生成全部 24 atoms 再筛选，不能计为部署加速。

## 可支持的硕士论文结论

只有完成 anchor 修正、等预算搜索对照、固定候选和真实 benchmark 后，才建议把这一部分列为
独立贡献。允许的结论是“以极小精度变化换取显著 refiner-stage 推理节省”；除非外部确认
支持，否则不能写成“提高分割精度”或“证明统计学非劣”。
