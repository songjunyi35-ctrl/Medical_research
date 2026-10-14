# Robust refiner 固定预算 Pareto 消融预注册

冻结时间：2026-08-14 UTC，在执行本消融的标签评分前冻结。

## 目的与边界

adaptive 1→8→24 路由未通过 count-matched random control，但固定 1-forward 和
8-forward workflow 显示出明显的 ensemble/TTA 冗余。本实验在完全相同的 exact float32
108-case validation atom cache 上比较六个预先存在或按自然层级定义的 fixed workflow，
建立 member-view forwards 与 Dice 的开发集 Pareto 曲线。

不访问 270-case test，不重新训练，不根据本实验结果修改任何 workflow。所有结果仍是
GT-centered N4 128³ oracle validation 和 refiner-stage 计算，不是独立外部确认。

## 冻结 workflow

1. `fixed1_seed15_identity`：seed15 identity，logit mean，threshold -0.75，beta 0，CC500。
2. `fixed3_models_identity`：三个 seed identity，logit mean，threshold -1，beta 0，CC500。
3. `fixed4_seed15_single_flips`：seed15 identity + 三个单轴 flip，logit mean，threshold
   -0.25，beta -0.5，CC250。
4. `fixed8_seed15_all_views`：冻结 deployment candidate，seed15 八视图，logit mean，
   threshold -0.25，beta -0.5，CC250。
5. `fixed12_models_single_flips`：三个 seed × identity/三个单轴 flip，logit median，
   threshold -0.25，beta -0.5，CC500；来自历史 fold-1 winner。
6. `fixed24_locked`：三个 seed × 八视图，logit mean，threshold -1，beta 0，CC500。

## 终点

每个 workflow 报告 scan-macro Dice、subject-macro Dice、相对 fixed24 的 subject 配对差、
seed `20260814 + workflow_index` 的 10,000 次 subject bootstrap 95% CI、改善/下降 subject
数和 exact forwards。

Pareto 定义为：不存在另一个 workflow 同时具有不低的 subject Dice 和不高的 forwards，
且至少一项严格更好。由于这组 workflow/后处理来自先前 validation 开发，Pareto 只属于
当前开发集；CI 跨 0 不得写成显著 Dice 改善或统计非劣。

正式输出：
`workspace/BraTS/sammed3d_t1ce_wt/fixed_budget_frontier_20260814/FINAL_FIXED_BUDGET_FRONTIER.json`。
