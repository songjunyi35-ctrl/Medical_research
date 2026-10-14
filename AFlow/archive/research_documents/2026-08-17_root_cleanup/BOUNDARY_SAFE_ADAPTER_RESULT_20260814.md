# SAGE-3D seed17 复现与 boundary auxiliary 结果（2026-08-14）

## 结论

plain frozen-base safe adapter 的正收益在第二个训练 seed 上成功复现；Dice-only 三维
boundary auxiliary 没有达到预注册的额外贡献门槛。

| seed/arm | scan Dice | subject Dice | 相对 frozen base scan | subject 配对 95% CI |
|---|---:|---:|---:|---:|
| frozen robust base | 0.699043355 | 0.694567241 | — | — |
| seed16 plain | 0.701991124 | 0.697102151 | +0.002947769 | [+0.001085,+0.004047] |
| seed17 control/plain | 0.702191495 | 0.697217550 | +0.003148140 | [+0.001015,+0.004384] |
| two-seed plain mean | **0.702091309** | **0.697159851** | **+0.003047954** | — |
| seed17 boundary | 0.702631580 | 0.697579311 | +0.003588225 | [+0.001416,+0.004628] |

seed17 control 超过复现成功线 `0.701043355`，且 subject Dice 同方向改善。因此上一轮
safe adapter 不是单一 seed16 的偶然尖峰。

boundary 相对同 seed control：

- scan：`+0.000440085`；
- subject：`+0.000361761`；
- subject 配对 95% CI：`[-0.000626,+0.001339]`；
- best boundary-head Dice：`0.343184`。

boundary head 明确学到了非零边界信号，但 scan 差没有达到预注册 `+0.001`，CI 也跨 0。
因此不能声称 boundary supervision 有额外 Dice 收益。后续方法候选仍应冻结更简单的
plain safe adapter，不能按 `0.702632` 的单点最高值选择 boundary arm。

## 方法与配对控制

两臂使用完全相同的 frozen robust seed15 base、SAGE plain adapter、seed20260817、
sampler、augmentation、optimizer 和 972/108 full-data protocol。总参数 `5,878,379`，
可训练 adapter 参数 `1,177,050`。两臂 epoch 0 都精确复现 base
`0.699043355/0.694567241`。

唯一差异：boundary arm 加入
`0.1 * DiceLoss(boundary_head, radius-1 symmetric 3D GT boundary band)`。boundary head
不进入最终 segmentation 输出，也不在推理时使用。它不使用 BCE/CE，目的是避免此前
ELPGR error-map auxiliary 的背景主导问题。

结果显示 boundary head 从 epoch1 Dice `0.297` 学到 best epoch8 的 `0.343`，训练后期
仍约 `0.35–0.38`；但主 segmentation 在 epoch8 后持续回落。这说明辅助任务可学习，
但没有足够证据证明其改善主任务。

## 运行与证据边界

- control 在 epoch9 由 patience 正常停止，best epoch1，耗时约 729 秒；
- boundary 跑满 16 epochs，best epoch8，耗时约 1290 秒；
- 两张 A100-SXM4-80GB 各约 4.1 GiB，GPU utilization 约 99%；
- 进程通过 `nohup` 持久运行，两个 `TRAIN_DONE.json` 正常生成；
- 两个 best checkpoint 已独立 reload 完整 108 例，scan/subject/boundary 指标差严格为 0；
- 全仓测试：`240 passed, 24 warnings`；
- 972 train / 108 validation，subject-clean；GT-centered N4 oracle refiner；
- 270 test、sealed manifest 和 test evaluator 全程不可达；
- 108 validation 已有开发暴露，本结果不是 official15 自动系统或独立 test。

## 完整性证据

| artifact | SHA-256 |
|---|---|
| control `TRAIN_DONE.json` | `c7fbb7465b5b9bce4197eb85f3fef40c357901ac85be7016e5d5ad5754544d1f` |
| boundary `TRAIN_DONE.json` | `296dac0cda17db22df3b611f35200c942f89e02d6772a7134818418f56f30dc2` |
| control best checkpoint | `d2e62fb72cca85a750c409cd8a5a053da94235390fe4f68ecf4be3faa751b89f` |
| boundary best checkpoint | `835da22365579e6af450e10e19e13dbe6ca7b8cda8143949112570e50be11de5` |
| control independent reload | `775a919f695ced64673c4ac399de4c993381518a836f990efbfd177729d14eec` |
| boundary independent reload | `90607fae753e9d62b70ab226451977f862d11ed39b6203aeaf5cd2814e50ac0a` |

正式目录：`workspace/BraTS/sammed3d_t1ce_wt/boundary_safe_adapter_pair_20260814/`。

## 下一步

1. 将论文架构候选冻结为 plain SAGE-3D safe adapter，并报告 seed16/17 两 seed mean；
2. boundary auxiliary 作为负/不充分消融保留，不继续在当前 108 validation 搜权重；
3. 下一项真正独立科研应转向未暴露 confirmation set，或自动提示/自动定位；
4. 若开展 GRPO，只优化候选 prompt/STOP 的离散策略，mask decoder 继续用 DiceCE。
