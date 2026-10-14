# T1ce/WT 推理剪枝与自适应路由结果（2026-08-14）

## 一句话结论

按病例学习的 1→8→24 adaptive router 没有证明比等成本随机路由更聪明，因此记为负
结果；真正稳定的发现是 robust refiner 的 3-model × 8-view 推理高度冗余。冻结的单模型
单视图 workflow 在当前 oracle validation 上相对 24-forward locked reference 的 subject
Dice 只变化 `-0.000358`，同时将 refiner-stage 实测延迟降低 `92.68%`，取得
`13.66×` A100 加速。固定 4/8-forward workflow 的 Dice 点估计还略高于 locked，但配对
CI 跨 0，不能声称显著提升。

## 1. 证据边界

- 数据：已经反复用于开发的 BraTS 2024 T1ce/WT 108-case、50-subject validation；
- geometry：GT-centered、N4、128³ oracle crop，不是自动定位；
- 输入：三个 robust refiner checkpoint × 八个 reflection views 的 exact float32 atom cache；
- 270-case test：所有新入口不可达、全程未使用；
- 口径：member-view/refiner stage，不是完整 SAM-Med3D 端到端耗时；
- 因为 stage workflow 继承自此前 validation 开发，本结果不是外部确认，也不能证明统计
  非劣或 test 泛化。

## 2. 1→8→24 adaptive router

### 2.1 方法

三个固定阶段是：seed15 identity 的 1-forward、同一 seed 八视图的 8-forward、以及完整
三 seed × 八视图的 24-forward locked reference。路由器只使用预测体积、logit margin、
边界接触、视图 disagreement、mask disagreement 和 probability variance 等 label-free
特征。每个 outer fold 只用 subject-disjoint search subjects 拟合 subject-balanced ridge，
只用 inner selector 选择 escalation threshold，然后一次性读取 outer holdout。

### 2.2 OOF 结果

| selector margin | scan Dice | subject Dice | 相对 locked | 95% CI | mean forwards | 1/8/24 stops |
|---:|---:|---:|---:|---:|---:|---:|
| `0.000` | `0.700208` | `0.695260` | `-0.000113` | `[-0.002271,+0.002112]` | `2.556` | `84/24/0` |
| `0.001` | `0.699997` | `0.694902` | `-0.000471` | `[-0.002649,+0.001748]` | `1.944` | `98/8/2` |
| `0.002` | `0.700106` | `0.695016` | `-0.000358` | `[-0.002564,+0.001845]` | `1.000` | `108/0/0` |
| `0.005` | `0.700106` | `0.695016` | `-0.000358` | `[-0.002567,+0.001950]` | `1.000` | `108/0/0` |

路由预测对真实 escalation gain 的 OOF Pearson correlation 只有：stage1→8 `0.0806`、
stage8→24 `-0.2298`。严格策略相对 fixed1 只提高 `+0.000244`，CI
`[-0.000254,+0.000794]`。

### 2.3 等成本随机控制

保持每个 outer fold 完全相同的 1/8/24 stop 数，随机打乱病例 10,000 次。严格策略的
subject Dice 为 `0.695260`，随机均值 `0.695004`，差 `+0.000256`；随机分布 95% 区间
`[0.694581,0.695450]`，`13.74%` 的随机路由达到或超过学习路由。宽松 `0.001` 策略还比
随机均值低 `-0.000090`。

因此必须按预注册判为：**没有证据表明当前 adaptive router 学会了有效的按病例预算
分配**。它可以保留为负结果和后续 difficulty proxy 的基线，但不能作为新的正贡献。

## 3. 固定预算剪枝 Pareto 曲线

六个 workflow 在完全相同的 108-case exact atom/label 上评测：

| workflow | forwards | scan Dice | subject Dice | 相对 locked subject | 95% CI |
|---|---:|---:|---:|---:|---:|
| seed15 identity | `1` | `0.700106` | `0.695016` | `-0.000358` | `[-0.002498,+0.001879]` |
| 3 models identity | `3` | `0.699829` | `0.695029` | `-0.000344` | `[-0.001798,+0.001099]` |
| seed15 single flips | `4` | `0.701150` | `0.696179` | `+0.000806` | `[-0.000702,+0.002307]` |
| seed15 all 8 views | `8` | **`0.701571`** | **`0.696271`** | `+0.000897` | `[-0.000432,+0.002257]` |
| 3 models single flips | `12` | `0.700626` | `0.696146` | `+0.000773` | `[-0.000350,+0.001962]` |
| 3 models all 8 views | `24` | `0.700422` | `0.695373` | `0` | — |

按 subject Dice 与 forwards，1/3/4/8 在当前有限候选集中属于 Pareto 点；12-forward 被
4-forward 支配，24-forward 被 4/8-forward 支配。最值得实际报告的三个工作点是：

- 极致效率：1 forward，Dice 点差很小；
- 平衡点：4 forwards，几乎取得 8-forward 的全部 Dice；
- 当前最高开发点：8 forwards，scan `0.701571`、subject `0.696271`。

4/8-forward 的 Dice 点估计高于 locked，但 CI 跨 0，所以表述必须是“当前 validation 上
同时降低计算且点估计不降/略升”，不能写成显著 Dice 提升。

逐病例从 1/8/24 中使用标签选最优阶段的 oracle ceiling 为 scan `0.704775`、subject
`0.699711`、mean forwards `10.306`。这说明病例级路由存在理论空间，但当前 prediction-only
proxy 没有把它可靠实现。

## 4. Fixed1 双 A100 真实在线 benchmark

使用两张 A100-SXM4-80GB 做交叉换卡：每个 receipt warmup 3，24 cases × 2 repeats；
candidate/locked 各 96 个正式 latency measurements。

| refiner-stage 指标 | fixed1 | locked24 | 收益 |
|---|---:|---:|---:|
| member-view forwards | `1` | `24` | `-95.83%` |
| mean latency | `0.078786 s/case` | `1.075861 s/case` | **`13.6555×`** |
| throughput | `12.6926 case/s` | `0.9295 case/s` | `13.6555×` |
| model allocated after load | `18,818,560 B` | `56,455,680 B` | `-66.67%` |
| peak allocated | `696,172,032 B` | `952,568,320 B` | `-26.92%` |

两个换卡方向的 speedup 分别为 `13.6362×` 和 `13.6749×`。fixed1 和 locked 各 8 例的
online/exact atom parity 全部逐体素一致：minimum mask Dice `1.0`、maximum different
voxels `0`。这证明节省来自真正少执行 23 次模型—视图 forward，不是 cache 计数技巧。

## 5. 对硕士论文贡献的建议

这部分现在最稳妥的贡献名称是：

> 面向三维医学分割的可审计推理 workflow 搜索与冗余剪枝：在严格 subject-clean
> validation 和真实双 GPU benchmark 下，识别并验证多模型多视图推理的精度—成本
> Pareto 前沿。

可以报告的正收益：

1. 强类型 workflow、subject-clean nested protocol、test isolation、exact atom parity 和
   完整 SHA 证据链；
2. 发现 24-way ensemble/TTA 高度冗余，并给出 1/3/4/8/12/24 的固定预算曲线；
3. fixed1 在 Dice 点变化很小的情况下取得 `13.66×` refiner-stage 在线加速；
4. fixed4/fixed8 在当前 validation 上同时降低计算并取得略高 Dice 点估计；
5. MCTS 未胜 random/greedy、adaptive 未胜 budget-matched random，均被诚实记录为算法
   消融/负结果，而不是隐藏。

不能声称：MCTS 优于简单 controller、adaptive early exit 有效、Dice 显著提升、统计非劣、
test 泛化、自动定位或端到端 `13.66×` 加速。

## 6. 权威证据

| 记录 | SHA-256 |
|---|---|
| `ADAPTIVE_ROUTER_PREREG_20260814.md` | `f1aa616b29a750f0fc9487b8376c1a257e31c95f1cbcb55e7009bb8c5b489627` |
| `workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/FINAL_ADAPTIVE_ROUTER_RESULT.json` | `dbba96e593c4bcf7742edb7f1e8afc614b058c6625ec8548e418849aef74eddf` |
| `ADAPTIVE_ROUTER_CONTROL_PREREG_20260814.md` | `9d49d1f74c697bda4bf7fc7d9a4a8963f56cd2dce3e75c9372e550ea07213e84` |
| `workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/FINAL_ADAPTIVE_ROUTER_CONTROL.json` | `662ca760e061d9c207abb9733b1286be16ff533bae7ebd560f8dc0445f4f1f14` |
| `FIXED_BUDGET_FRONTIER_PREREG_20260814.md` | `d45129a4082effcaf006e92cde47b105d6d5e5fa9b0bbc56263cc6b1193a1b5e` |
| `workspace/BraTS/sammed3d_t1ce_wt/fixed_budget_frontier_20260814/FINAL_FIXED_BUDGET_FRONTIER.json` | `9cc01c56659f1ca38e67ef8511ecba5376abf0e1ae1d7d1bee2567fec5480de0` |
| `FIXED1_DEPLOYMENT_CANDIDATE_20260814.json` | `e162f1bc337fb83c19163a6723a6d62f136c2518caaeb7def2d04eda6651a6d4` |
| `FIXED1_BENCHMARK_PREREG_20260814.md` | `7c760475b10043c1934c147edaa32936211d9b561f5b734274074707efd9cd69` |
| `workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/fixed1_benchmark/FINAL_FIXED1_DEPLOYMENT_BENCHMARK.json` | `672551a39bab4bb260a2dbfbd07301584301597cef255076b83d5679f83455df` |

新增后全仓测试：`228 passed, 26 warnings`。测试 warning 为既有 CUDA-unavailable 和
SciPy deprecated namespace 提示，不影响通过状态。

## 7. 下一步

1. 不再继续调当前 108 validation 上的 routing feature；否则只会增加重复使用开发集的
   选择偏差。
2. 在新外部确认集一次性比较 frozen fixed1/fixed4/fixed8/locked24，主要终点为 subject
   Dice、配对区间和真实 refiner-stage latency。
3. 若仍研究 adaptive，必须在新的 training/confirmation subjects 上学习 difficulty
   proxy，并保留 count-matched random control；当前 ridge router 作为负基线。
4. 自动定位替换 GT-centered crop 仍是比继续微调推理阈值更重要的系统性问题。
