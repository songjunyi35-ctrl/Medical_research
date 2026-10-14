# SAGE-3D frozen-base safe adapter 结果（2026-08-14）

## 结论

SAGE-3D 的共同 frozen-base uncertainty adapter 达到预注册的开发集成功门槛；implicit
gate 没有达到额外贡献门槛。

| arm | scan Dice | 相对 frozen base | subject Dice | subject 配对差 | subject 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|
| frozen robust base | 0.699043355 | — | 0.694567241 | — | — |
| plain safe adapter | 0.701991124 | +0.002947769 | 0.697102151 | +0.002534910 | [+0.001085,+0.004047] |
| gated safe adapter | **0.702090475** | **+0.003047120** | **0.697139906** | **+0.002572664** | **[+0.000877,+0.004349]** |

两臂都超过预注册的 adapter 成功线 `0.701043355`，subject Dice 也没有下降。gated 相对
plain 只有 scan `+0.000099351`、subject `+0.000037755`；subject 配对 CI
`[-0.000248,+0.000379]`，远未达到预注册 `+0.001`。因此：

- 可以把正收益归因于 frozen-base、zero-initialized、uncertainty-conditioned adapter；
- 不能把收益归因于 implicit gate；
- 推荐部署/后续复核优先使用更简单的 plain arm，而不是按最高点估计选择 gated。

50-subject CI 使用固定 seed `20260814`、100,000 次 subject bootstrap；先在每个 subject
内平均纵向 scans，再对 subjects 有放回重采样。

## 架构与可归因性

两臂使用相同的 robust seed-20260815 frozen base、同一 seed、同一数据顺序和完全相同
容量：总参数 `5,878,379`，其中仅 adapter 的 `1,177,050` 参数可训练。

adapter 输入：

1. T1ce；
2. frozen SAM coarse logit；
3. frozen robust-base logit；
4. base uncertainty `4p(1-p)`；
5. `abs(base_probability-coarse_probability)` disagreement。

末层 residual/gate 全零初始化，因此 epoch 0 的 scan/subject Dice 均逐指标精确复现
`0.699043355 / 0.694567241`。plain 输出 `base+residual`；gated 输出
`base+2*sigmoid(gate)*residual`。两臂只用最终 segmentation DiceCE，不再使用上一轮导致
过度抑制的稀疏 error-map auxiliary loss。

这说明最可靠的改进机制是“保留已验证强基座，只学习小幅追加纠错”，而非全局重写已
学会的 residual。它是 SAM-Med3D coarse-to-fine 下游架构创新，不是对原始 100M ViT
encoder 的直接修改。

## 证据边界与运行偏差

- 数据：972 train / 108 validation，subject-clean；T1ce，WT=`{1,2,3}`；
- 输入几何：GT-centered N4 `128^3` oracle-localization refiner cache；
- 270-case test、sealed manifest 和 test evaluator 全程不可达；
- 108 validation 已参与此前开发，所以这是正向 development result，不是独立 test；
- 训练按预注册运行到 epoch 10；两臂 best 都在 epoch 3，之后连续 7 epochs 无新高；
- 外部 SSH/agent 会话中断终止了进程，距离 patience=8 的正常自动停止还差 1 epoch，
  因而没有生成 `TRAIN_DONE.json`。该偏差必须保留，不能把本次写成完整正常收尾；
- 两个 best checkpoint 已在 2×A100 上独立 reload 完整 108 例，scan/subject 指标与
  checkpoint 内嵌值差均为严格 `0.0`。

该中断不会改变已经保存的 epoch-3 best 数字，但正式论文结论仍应补至少第二 seed，并
在未暴露确认集复核。当前不触发 270 test。

## 证据文件

| artifact | SHA-256 |
|---|---|
| plain best checkpoint | `185fc6e316e3fb8595172dea844e33fe6527f64609bd3ca9b7288e729f1f8ec0` |
| gated best checkpoint | `1070be97d48e12100bebac57d9128b8047a907930874eef1da7fd21aa64c65e8` |
| plain progress through epoch 10 | `9032b0731a94e9b6e67aca1e6aeabeebd5bc1cebb65850232f0cea0810cfb360` |
| gated progress through epoch 10 | `c25980894839500452da9929046fbc6eed8a16c492aa3be4497b453311d8a19a` |
| plain independent reload | `e3d7bb5a54637d8bdb2c393ac658427d819f8e3dd9833dab153b3cd0f64131e0` |
| gated independent reload | `779dd60bbf42b57a4e4f123e42f75d9e7fd5da5d50c773c9b6b4844457aa662e` |

目录：`workspace/BraTS/sammed3d_t1ce_wt/safe_adapter_pair_20260814/`。

核心代码：

- `sammed3d/t1ce_wt_safe_adapter.py`
- `scripts/train_t1ce_wt_safe_adapter.py`
- `scripts/reload_validate_t1ce_wt_safe_adapter.py`
- `SAFE_ADAPTER_PAIR_PREREG_20260814.md`

加入本轮架构、训练和 reload evaluator 后，全仓回归为 `236 passed, 24 warnings`。

## 下一步

1. 冻结 plain adapter 为更简洁的候选，不根据 gated 的 `+0.000099` 选择复杂模型；
2. 补两个独立训练 seed，primary 改为 seed mean 与方向一致性；
3. 在新的未暴露 confirmation set 上一次性确认 base vs plain adapter；
4. 只有监督架构稳定后，再把 GRPO 用于候选提示点/正负类型/STOP 的离散策略，不用
   GRPO 替代 dense mask DiceCE。
