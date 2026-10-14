# Exact MCTS Atom Cache Parity Review Preregistration

冻结时间：2026-08-13 UTC。对象仅为 108-case validation；`test_access=false`，禁止读取或
使用已经揭示的 270-case test。

第一次 exact float32 atom materialization 在相同 A100/BF16 推理协议下完成 108/108，
但预先设置的单一 `2e-6` parity tolerance 失败。失败收据必须原样保留：

`workspace/BraTS/cache/t1ce_wt_robust_mcts_atoms_f32/PARITY_FAILED.json`

失败值在查看任何 MCTS workflow score 前已经确定：locked scan-macro `0.7004216820470339`
vs 历史 `0.700418962602465`，差 `2.7194e-6`；subject-macro 差 `1.2496e-6`；最大逐病例
Dice 差 `2.7146e-4`。三个 identity seed 的当前重放值与训练期权威 raw 值一致，而历史
ensemble evaluator 自身相对训练期已有约 `5e-6` 重放差异。这表明误差来自独立 BF16
重放的边界 voxel 数值非确定性，不是 atom order、checkpoint、cache 或 workflow 语义变化。

在发布 cache 前冻结以下 promotion 规则：

- checkpoint、validation index、coarse/image/label blob、24-atom order、atom file、旧失败
  收据和所有相关源码均重新核 SHA；
- 重新从 float32 atoms 计算三 seed identity raw、三 seed identity ensemble raw、3×8 raw、
  3×8 locked；所有项目都逐病例对照原报告；
- scan-macro 和 subject-macro absolute tolerance 固定为 `1e-5`；
- 任一逐病例 Dice absolute tolerance 固定为 `5e-4`；
- 不修改/删除 `PARITY_FAILED.json` 或 `BUILD_STATE.json`；promotion 只新增
  `PARITY_REVIEW.json` 和最终 `index.json`；
- 规则不允许 CLI 放宽。任一项失败则不发布 cache、不启动 MCTS。

这些容差只验证同一固定 workflow 的数值重放，不参与 MCTS reward、threshold、TTA、
component filter 或任何 workflow 选择。
