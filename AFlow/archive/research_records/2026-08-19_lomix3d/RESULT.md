# 三尺度 LoMix-3D fixed-validation 负结果

## Outcome

NeurIPS 2025 LoMix 的三维、三 decoder-stage 适配完成了预注册、独立新入口、12项合成测试、真实 batch8 H100 smoke、global epoch169–200 formal continuation，以及一次 fixed validation108 决策。所有工程门、重载门和 Softplus 防塌缩门都通过，但所有关键 Dice promotion gate 均失败，因此 **不访问 reused internal test270**。

这次结果不支持“在历史 epoch168 后加入原论文式、未归一化 Softplus 加权的三尺度 raw-logit mixing，能改善最终 128³ head”这一具体直觉。它排除的是当前实现、初始化和32轮 continuation 协议，不证明所有 LoMix、规范化权重版本或从头训练都不可能有效；这些若再做必须是新方法和新预注册，不能由本次 validation 反向调参。

## Idea and only changed factor

输入仍是同一 128³ T1ce。原 SegResNet 只做一次 encoder/decoder forward；32³、64³两个 coarse stage 新增训练期辅助 head，128³继续使用原 final head。三个 raw logits 对所有 `01/02/12/012` subset 做 add、multiply、registered concat `1×1×1 Conv3d`、subset-specific attention fusion，共19张训练监督图。每张使用现有 DiceCE，未归一化总损失为：

`sum_u softplus(alpha_u) * DiceCE(z_u, y)`。

没有 teacher。唯一变化是训练期 LoMix supervision。history168初始化、756例train cache/manifest、seed20260846、weighted sampler、shift8、batch8、AdamW/LR/weight decay、global epoch169–200、checkpoint grid、validation108、threshold0、无SAM/TTA/postprocessing和1×部署全部固定。matched control 是已归档的完全相同 supervised continuation。

## Fixed validation108 / 50 subjects

按预注册只比较 global epochs176/184/192/200，最大 Scan Dice、Subject exact tie-break 选择 epoch200。

| LoMix checkpoint | Scan Dice | Subject Dice | Point hit |
|---|---:|---:|---:|
| epoch176 | `0.731369884` | `0.720065536` | `0.925926` |
| epoch184 | `0.727070494` | `0.715689368` | `0.935185` |
| epoch192 | `0.729881564` | `0.718783701` | `0.944444` |
| **epoch200 selected** | **`0.732479468`** | **`0.721607103`** | **`0.935185`** |

### Selected epoch200 versus controls

| Comparison | Scan delta | Scan paired 95% CI | Subject delta | Subject paired 95% CI | Point delta |
|---|---:|---:|---:|---:|---:|
| LoMix − same-epoch supervised control | `-0.001126193` | `[-0.002963550,+0.000624292]` | `-0.002064797` | `[-0.004123937,-0.000092064]` | `0` |
| LoMix − voxel-KD epoch200 | `-0.001904840` | `[-0.004241536,+0.000246735]` | `-0.002827040` | `[-0.005072642,-0.000650705]` | `0` |

相对 matched control，48/108 scans 改善、60变差；23/50 subjects 改善、27变差。相对 voxel-KD，43/108 scans 改善、65变差；20/50 subjects 改善、30变差。Point hit 与两者持平，但不能补偿整体 overlap 的下降。

## Promotion decision

通过：合成/GPU smoke、formal completion、checkpoint reload exact、stripped deploy exact、Point within one case、所有保存epoch平均Softplus未塌缩。

失败：

- selected LoMix 未超过同 epoch control 的 Scan 或 Subject；
- selected LoMix 未超过 voxel-KD epoch200 的 Scan 或 Subject；
- 相对 voxel-KD 的 Scan/Subject paired CI 下界均低于 `-0.003` 安全线。

结论：`all_gates_passed=false`，`lomix3d_candidate_authorized_for_test=false`。没有test evaluator运行、没有test receipt、没有Leaderboard变更。

## Mechanism interpretation

平均 Softplus 权重从初始 `0.693147` 缓慢降到 epoch176/184/192/200 的 `0.676059/0.664580/0.658119/0.654070`，远高于预注册 collapse line `0.173287`。所有19个权重都下降；epoch200范围仅 `0.652029–0.659909`，差异很小。这符合公式“loss梯度直接推动alpha下降”的审计风险，不支持它在本次短 continuation 中完成了有意义的分支搜索。

工程上模块确实把梯度传到所有2个aux、4个concat、4个attention和19个alpha，并保持最终 deploy logit 完全等价；因此负结果发生在 **有效运行后的科研效果层**，不是未注册卷积、缺梯度、OOM或checkpoint损坏。

## Cost and deployment

- Formal training：`1045.64s`（17.43min），约为历史 matched control `661.0s` 的 `1.58×`。
- Peak allocated/reserved GPU：`25.61/31.48 GB`（约 `23.85/29.32 GiB`）；H100 80GB。
- 四个 checkpoint：每个 `37,691,475` bytes；完整 LoMix workspace 约144M；没有新图像cache。
- 部署：剥离aux/fusion/alpha后仍是 `4,700,897` 参数 plain locator、1× forward；threshold0，无SAM/TTA/postprocessing。
- Formal wrapper reload max abs difference：`0`；stripped deploy max abs difference：`0`。

## Evidence and hashes

- Raw validation report: `workspace/BraTS/sammed3d_t1ce_wt/plain_lomix3d_20260819/LOMIX3D_VALIDATION.json`, SHA-256 `ff12f2c74a9c27be5cc9d6fbfe096d22c115fab443b003b70bab1196cb633e65`.
- Validation started receipt: SHA-256 `dde21958f7ec6d35ad2e772d45d833052c8997ec06a58525317b6275d8e7edad`.
- Validation pre-access receipt: SHA-256 `a8e51c831d1922d159baaa2be0323177cbad5001d2529695ed9fdfdec9554fd7`.
- Formal training report: SHA-256 `6c9ebaab275637ea8669d150340552e9f8cc9dfb946b734f368f8b476d303440`.
- Implementation receipt: SHA-256 `05b540aeae35cce48e1835d28626167cf57f7447fd1d28dfb79bb3d1c3fa6f9a`.
- Selected epoch200 checkpoint: SHA-256 `f9bfcca36a904341f4eeebf1c71f218b997ab3e753be4815daa2789b2c7245fd`.
- Model/trainer/validation evaluator SHA-256: `1e6c1f...caa7d` / `205d4e...f4832` / `1c08c5...b61c4`.

## Evidence scope and next decision

Validation108 已用于本次冻结选择，不是外部确认。outer selector、confirmation和test270均未访问。本方法停止，不搜索operator、alpha归一化、loss weight、epoch、threshold或postprocessing。下一前沿候选回到handoff顺序中的 2026 DCD detail distillation；它需要新的七问与预注册。

Primary source: [NeurIPS 2025 LoMix paper](https://papers.nips.cc/paper_files/paper/2025/file/5f0e54e34ba4c30fbc6f229986dbffb4-Paper-Conference.pdf), [official repository](https://github.com/SLDGroup/LoMix).
