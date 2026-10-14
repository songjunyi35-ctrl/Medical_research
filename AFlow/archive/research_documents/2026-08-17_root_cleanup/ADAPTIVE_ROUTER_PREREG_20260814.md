# T1ce/WT 1→8→24 自适应推理路由预注册

冻结时间：2026-08-14 UTC

状态：在执行本方法的任何新标签驱动分析前冻结。本文不改变或覆盖 2026-08-13 的
MCTS、controller comparison、deployment candidate 或 benchmark 证据。

## 1. 研究问题

旧 inference grammar 的 adaptive acquisition 从三个模型的 identity prediction 起步，
只能执行 3→12→24 member-view forwards；三个 outer-fold winner 最终全部是 fixed
workflow。新实验检验：一个 prediction-only、按病例路由的 1→8→24 cascade，是否能在
当前 108 例 GT-centered oracle validation 上，比固定 8-forward candidate 获得更好的
Dice—compute 折中，同时保留 24-forward locked reference 作为 mandatory anchor。

这是一项已经反复使用过的 validation 上的 development experiment，不是独立确认，
不访问 270-case test，也不能建立 test 泛化或统计非劣结论。

## 2. 固定的三个推理阶段

三个阶段均来自已经冻结的历史 workflow，不在本实验中重新选择模型、阈值、融合或
后处理：

1. `stage1`：seed 20260815、identity view、logit mean、threshold `-0.75`、beta `0`、
   26-connectivity CC `500`；成本 1 forward。它来自历史 fold-2 fixed winner。
2. `stage8`：seed 20260815、all 8 reflection views、logit mean、threshold `-0.25`、
   beta `-0.5`、CC `250`；成本 8 forwards。它是冻结的 deployment candidate
   `a48d7c60047321da9933f5f29a924ff826b1dbcf533c4fa53e4aafbda8f63400`。
3. `stage24`：三个 seed × all 8 views、logit mean、threshold `-1.0`、beta `0`、
   CC `500`；成本 24 forwards。它是 locked reference。

三个 atom 集合满足 stage1 ⊂ stage8 ⊂ stage24，因此在线执行只需逐级补充新 forward，
不会重复计算已经取得的 atom。

## 3. Label-free 路由特征

`stage1` 路由器只能读取 stage1 logits/mask，使用以下固定特征：

- `log_mask_voxels`：后处理 mask 体积的 `log1p`；
- `roi_fraction`：logit 大于 stage threshold 减 2 的体素比例；
- `near_threshold_fraction`：上述 ROI 内绝对 margin 不超过 0.5 的比例；
- `mean_abs_margin` 和 `p10_abs_margin`：ROI 内阈值绝对 margin 的均值和 10% 分位数；
- `face_contact_fraction`：mask 位于 crop 六个边界面的比例。

`stage8` 路由器使用全部 stage1 特征，并增加：

- 八视图 squared-soft-Dice disagreement；
- stage1 与 stage8 postprocessed mask 的 `1-Dice`；
- 两阶段 mask 的 log-volume ratio；
- stage8 ROI 内 voxelwise probability standard deviation 的均值和 90% 分位数；
- stage8 ROI 内 near-threshold fraction。

所有特征在部署时只依赖模型输出，不读取图像标签、病例编号或 GT 体积。

## 4. 路由器训练和数据隔离

沿用 seed `20260813` 的三组 subject-disjoint nested roles。每个 outer fold：

1. 只用 `mcts_search` subjects 拟合两个确定性的 subject-balanced ridge regression：
   - stage1 特征预测 `Dice(stage8)-Dice(stage1)`；
   - stage8 特征预测 `Dice(stage24)-Dice(stage8)`。
2. 每个 subject 的总训练权重相同；特征只用 search role 的加权均值/标准差标准化；
   ridge alpha 固定为 `1.0`。
3. 只在 `inner_selector` 上选择两个 escalation threshold。
4. outer holdout 的标签只用于一次最终 readout，不能参与拟合、阈值或 margin 选择。

决策规则固定为：预测 stage8 相对 stage1 的增益不高于 `t1` 时停在 stage1；否则执行
stage8。已经执行 stage8 的病例，预测 stage24 相对 stage8 的增益不高于 `t8` 时停在
stage8，否则执行 stage24。

threshold 候选仅取 selector prediction 的 0%、10%、…、100% 分位点，再加 `-inf/+inf`。
这包括 always-stage1、always-stage8 和 always-stage24 极端策略。

## 5. 选择规则与主要终点

对 accuracy margins `0.000/0.001/0.002/0.005` 分别选择策略。selector 上的 accuracy
floor 等于同 role locked stage24 subject-macro Dice 减 margin；在满足 floor 的策略中，
依次优先：

1. 更低 mean member-view forwards；
2. 更高 subject-macro Dice；
3. 更低 maximum forwards；
4. threshold 的确定性字典序。

主要报告为 cross-fitted 108-scan/50-subject OOF：scan Dice、subject Dice、相对 locked
配对 subject 差及 10,000 次 bootstrap 95% CI、mean/max forwards、1/8/24 停止数量。

必须同时报告：固定 stage1/stage8/stage24 的全 validation 开发指标，以及逐病例使用标签
选择三个阶段中最佳者的 oracle routing ceiling。oracle ceiling 只诊断可路由空间是否有
潜力，绝不是可部署结果。

## 6. 成功、失败与表述边界

- 若 OOF subject Dice 不低于 locked 的点估计且 mean forwards < 8，可称为本开发集上的
  精度—成本联合改善，但仍需外部确认。
- 若 OOF subject Dice 的点差落在 `[-0.001,0]` 且 mean forwards < 8，只能称精度变化
  很小的进一步效率收益，不能称 Dice 提升或统计非劣。
- 若不能优于 fixed stage8 的 Dice—compute 点，adaptive routing 记为负结果。
- 无论结果如何，禁止使用 270-case test 调整特征、ridge alpha、threshold grid、margin、
  stage workflow 或报告口径。
- 本实验仍为 GT-centered N4 128³ oracle geometry、refiner stage，不是自动定位或完整
  SAM-Med3D 端到端推理。

## 7. 产物约定

- 实现：`sammed3d/t1ce_wt_adaptive_router.py`
- 正式 runner：`scripts/evaluate_t1ce_wt_adaptive_router.py`
- 单元测试：`tests/test_t1ce_wt_adaptive_router.py`
- 输出目录：
  `workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/`
- 正式汇总：`FINAL_ADAPTIVE_ROUTER_RESULT.json`

正式 JSON 必须记录本预注册、atom index、validation index、三个历史 workflow、源码和
测试的 SHA-256，并明确 `test_access: false`。
